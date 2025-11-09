#!/bin/bash

set -ex
CONFIG_FILE="$1"
shift # Passthrough the rest

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
  --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  -m ${TRAIN_FILE} \
  --job.config_file ${CONFIG_FILE} \
  "$@"
