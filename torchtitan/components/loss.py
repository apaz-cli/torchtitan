# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def fused_linear_cross_entropy_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Fused linear + cross entropy using Liger kernel.

    This function fuses the final linear projection (lm_head) with the cross-entropy
    loss computation, avoiding materialization of the full logits tensor and using
    a more efficient kernel.

    Args:
        hidden_states: Hidden states from final layer norm, shape (B, T, H)
        labels: Target token indices, shape (B, T)
        weight: Output projection weight, shape (V, H)
        bias: Optional bias, shape (V) - Qwen3 doesn't use this
        ignore_index: Index to ignore in loss computation (default: -100)
        lse_square_scale: Z-loss coefficient for logit regularization (default: 0.0)
        label_smoothing: Label smoothing factor (default: 0.0)

    Returns:
        Scalar loss tensor (main loss + z_loss if lse_square_scale > 0)
    """
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    # Flatten batch and sequence dimensions
    # hidden_states: (B, T, H) -> (B*T, H)
    # labels: (B, T) -> (B*T,)
    if hidden_states.dim() == 3:
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)
    if labels.dim() == 2:
        labels = labels.view(-1)

    # Determine if we need z_loss
    return_z_loss = lse_square_scale > 0.0

    # Call Liger fused kernel
    # Function signature (excluding ctx):
    # _input, weight, target, bias, ce_weight, ignore_index, lse_square_scale,
    # label_smoothing, reduction, softcap, return_z_loss, accum_dtype, use_token_scaling
    # Returns: (loss, z_loss) where z_loss can be None if return_z_loss=False
    loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
        hidden_states,
        weight,
        labels,
        bias,
        None,  # ce_weight (class weights)
        ignore_index,
        lse_square_scale,
        label_smoothing,
        "mean",  # reduction
        None,  # softcap
        return_z_loss,
        None,  # accum_dtype (use default dtype)
        False,  # use_token_scaling
    )

    # If z_loss is requested, it's already added to the main loss by Liger
    # The z_loss output is just for logging/monitoring
    return loss


def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        compile_kwargs = {"backend": job_config.compile.backend}
        if job_config.compile.mode is not None:
            compile_kwargs["mode"] = job_config.compile.mode
        loss_fn = torch.compile(loss_fn, **compile_kwargs)
    return loss_fn


def build_liger_fused_linear_cross_entropy_loss(job_config: JobConfig, **kwargs):
    """
    Build Liger fused linear cross entropy loss function.

    Note: This loss function cannot be compiled with torch.compile since it uses
    custom CUDA kernels from Liger. The loss function will be used as-is.
    """
    del kwargs  # delete any unused arguments

    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.warning(
            "Liger fused linear cross entropy loss cannot be compiled. "
            "Ignoring compile settings for loss."
        )

    logger.info("Using Liger fused linear cross entropy loss")
    return fused_linear_cross_entropy_loss


class RescaleAccumulatedLoss:
    def __init__(self, unwrapped_loss_fn, accumulation_steps):
        self.unwrapped_loss_fn = unwrapped_loss_fn
        self.accumulation_steps = accumulation_steps
        self.skip_rescale = False

        # Copy over attributes from the original function, but don't
        # copy the dict, which interferes with nested wrapping.
        functools.update_wrapper(self, unwrapped_loss_fn, updated=tuple())

    def __call__(self, *args, **kwargs):
        loss = self.unwrapped_loss_fn(*args, **kwargs)
        if self.skip_rescale:
            return loss
        return loss / self.accumulation_steps

    @contextlib.contextmanager
    def no_rescale(self):
        """Context manager for disabling rescaling"""
        previous = self.skip_rescale
        self.skip_rescale = True
        try:
            yield
        finally:
            self.skip_rescale = previous


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """
    return RescaleAccumulatedLoss(unwrapped_loss_fn, accumulation_steps)
