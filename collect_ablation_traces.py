#!/usr/bin/env python3

"""
Script to collect profiling traces for ablation studies.

This script:
1. Uses a base config file (TOML)
2. Generates variations by adding CLI overrides
3. Runs training for each config variation to collect profiling traces
4. Organizes traces in separate directories for easy comparison

Usage:
    python collect_ablation_traces.py --base_config path/to/config.toml
"""

import argparse
import os
import subprocess
import sys
import time
import itertools

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def should_skip_config(current_combo, failed_combos, options):
    """
    Determine if a configuration should be skipped based on previous failures.

    Uses monotonicity rules: if an option is marked as 'monotonic', and a configuration
    with a smaller value failed (with all other settings the same), skip larger values.

    Args:
        current_combo: Dict mapping option keys to values for the current config
        failed_combos: List of dicts for configurations that have failed
        options: The options dict defining all configuration dimensions

    Returns:
        True if this config should be skipped, False otherwise
    """
    for failed_combo in failed_combos:
        # Check each monotonic dimension
        for key, opt in options.items():
            if not opt.get('monotonic', False):
                continue

            # Check if all other dimensions match
            other_dims_match = all(
                failed_combo.get(other_key) == current_combo.get(other_key)
                for other_key in options.keys()
                if other_key != key
            )

            if not other_dims_match:
                continue

            # Check if failed config has smaller or equal value on this monotonic dimension
            values_list = opt['values']
            try:
                failed_idx = values_list.index(failed_combo.get(key))
                current_idx = values_list.index(current_combo.get(key))

                # If failed value comes before current value in the list, skip
                if failed_idx <= current_idx:
                    return True
            except (ValueError, TypeError):
                # Value not in list, skip this check
                continue

    return False


def generate_config_variations():
    """
    Generate config variations for ablation study.

    Edit this function to create the specific configurations you want to test.
    Each variation should be a dict with:
    - 'name': A descriptive name for this configuration
    - 'overrides': List of config override strings (e.g., '--model.use_flex_attn')

    Returns:
        List of dicts, each containing 'name' and 'overrides'
    """
    # Edit this dict to specify what you want to test
    options = {
        'use_liger_loss': {
            'values': [True, False],
            'flag': '--model.use_liger_loss',
            'name': 'liger',
        },
        'use_flex_attn': {
            'values': [True, False],
            'flag': '--model.use_flex_attn',
            'name': 'flex',
        },
        'batch_size': {
            'values': [1, 2, 3, 4, 5, 6, 7, 8],
            'flag': '--training.local_batch_size',
            'name': 'bs',
            'monotonic': True,  # Skip larger values if smaller ones fail
        },
        'ac_mode': {
            'values': ['full', 'selective', 'none'],  # Ordered: most resource-intensive to least
            'flag': '--activation_checkpoint.mode',
            'name': 'ac',
            'monotonic': True,  # If full fails, skip selective and none
        },
        'selective_ac_option': {
            'values': ['int', 'op'],
            'flag': '--activation_checkpoint.selective_ac_option',
            'name': 'sac',
        },
    }

    # Generate all combinations
    variations = []
    option_keys = list(options.keys())
    option_values = [options[key]['values'] for key in option_keys]

    for combination in itertools.product(*option_values):
        name_parts = []
        overrides = []
        combo_dict = dict(zip(option_keys, combination))

        for key, value in zip(option_keys, combination):
            opt = options[key]

            # Add to overrides if value is True (for bool) or always (for other types)
            if isinstance(value, bool):
                if value:
                    overrides.append(opt['flag'])
                    name_parts.append(opt['name'])
            else:
                overrides.extend([opt['flag'], str(value)])
                name_parts.append(f"{opt['name']}{value}")

        # Build the name with "Ablation:" prefix
        base_name = '_'.join(name_parts) if name_parts else 'baseline'
        name = f'Ablation_{base_name}'

        variations.append({
            'name': name,
            'overrides': overrides,
            'combo': combo_dict,  # Store for skip logic
        })

    return variations, options


def run_training_for_profile(base_config_path, overrides, output_dir, run_name, pbar=None):
    """
    Run training with profiling enabled to collect a trace.

    Args:
        base_config_path: Path to base config file
        overrides: List of config override arguments
        output_dir: Base directory for all outputs
        run_name: Name for this run (used in output path)
        pbar: Optional tqdm progress bar to update
    """
    # Construct the command using run_train.sh
    cmd = [
        './run_train.sh',
        '--job.dump_folder', os.path.join(output_dir, run_name),
        '--job.run_name', run_name,
        '--profiling.enable_profiling',
        '--profiling.kill_after_profile',
        '--profiling.profile_freq', '4',
        '--profiling.profiler_warmup', '3',
        '--profiling.profiler_active', '1',
    ]

    # Add user-specified overrides
    cmd.extend(overrides)

    # Set environment variables for run_train.sh
    env = os.environ.copy()
    env['CONFIG_FILE'] = base_config_path
    # Default to single GPU for profiling
    if 'NGPU' not in env:
        env['NGPU'] = '1'

    # Create log directory
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, 'training.log')

    if pbar:
        pbar.set_description(f"Collecting {run_name}")

    # Run the command with output redirected to log file
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        elapsed = time.time() - start_time
        if pbar:
            pbar.write(f"✓ {run_name} completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        if pbar:
            pbar.write(f"✗ {run_name} failed after {elapsed:.1f}s (see {log_file})")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Collect profiling traces for ablation studies'
    )
    parser.add_argument(
        '--base_config',
        type=str,
        required=True,
        help='Path to base configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ablation_traces',
        help='Directory to save all ablation traces (default: ./ablation_traces)'
    )

    args = parser.parse_args()

    # Validate base config exists
    if not os.path.exists(args.base_config):
        print(f"Error: Base config file not found: {args.base_config}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base config: {args.base_config}")
    print(f"Output directory: {output_dir}")

    # Generate config variations
    variations, options = generate_config_variations()

    print(f"\nGenerated {len(variations)} config variations:")
    for i, var in enumerate(variations, 1):
        ov = f": {' '.join(var['overrides'])}" if var['overrides'] else ""
        print(f"  {i}. {var['name']}{ov}")

    # Run each variation
    results = {}
    timings = {}
    failed_combos = []  # Track failed configs for skip logic

    print(f"\n{'='*80}")
    print("Starting ablation runs...")
    print(f"{'='*80}\n")

    # Use tqdm if available, otherwise basic iteration
    if HAS_TQDM:
        iterator = tqdm(variations, desc="Progress", unit="run")
    else:
        iterator = variations
        print(f"Running {len(variations)} configurations...")

    for i, variation in enumerate(iterator, 1):
        run_name = variation['name']
        overrides = variation['overrides']
        combo = variation['combo']

        # Check if we should skip this config based on previous failures
        if should_skip_config(combo, failed_combos, options):
            if HAS_TQDM and iterator:
                iterator.write(f"⊘ {run_name} skipped (monotonicity rule)")
            elif not HAS_TQDM:
                print(f"[{i}/{len(variations)}] ⊘ Skipping {run_name} (monotonicity rule)")
            results[run_name] = 'skipped'
            timings[run_name] = 0.0
            continue

        if not HAS_TQDM:
            print(f"[{i}/{len(variations)}] Running {run_name}...")

        success, elapsed = run_training_for_profile(
            args.base_config,
            overrides,
            output_dir,
            run_name,
            pbar=iterator if HAS_TQDM else None
        )

        results[run_name] = success
        timings[run_name] = elapsed

        # Track failed configs for skip logic
        if not success:
            failed_combos.append(combo)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [name for name, result in results.items() if result is True]
    failed = [name for name, result in results.items() if result is False]
    skipped = [name for name, result in results.items() if result == 'skipped']

    total_time = sum(timings.values())
    total_runs = len(successful) + len(failed)
    print(f"Completed: {len(successful)}/{total_runs} runs in {total_time:.1f}s")
    if skipped:
        print(f"Skipped: {len(skipped)} configs (monotonicity rules)")

    if successful:
        print(f"\n✓ Successful ({len(successful)}):")
        for name in successful:
            print(f"  • {name} ({timings[name]:.1f}s)")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for name in failed:
            print(f"  • {name} ({timings[name]:.1f}s)")

    if skipped:
        print(f"\n⊘ Skipped ({len(skipped)}):")
        for name in skipped:
            print(f"  • {name}")

    print(f"\nOutput directory: {output_dir}")
    print("  • Traces: {run_name}/profile_traces/")
    print("  • Logs: {run_name}/training.log")
    print("\nTo view traces, open Chrome and navigate to https://www.ui.perfetto.dev/")
    print("Then load the JSON files from the respective profile_traces subdirectories")

    # Exit with error code if any runs failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
