#!/usr/bin/env python3
"""
CLI Command Line Interface for NILM AI Dataset Synthesis.
Combines preprocessed individual appliance CSVs into a multi-appliance aggregate dataset
for NILM AI model training & disaggregation tasks.
"""

import argparse
import glob
import os
import sys
import yaml

from nilm_preprocessing.synthesizer import NILMSynthesizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize multi-appliance NILM dataset for AI training."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Input directory containing processed individual appliance CSV files (default: ./output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join(".", "output", "synthetic_nilm_dataset.csv"),
        help="Output CSV path for the synthetic aggregate dataset (default: ./output/synthetic_nilm_dataset.csv)",
    )
    parser.add_argument(
        "-n",
        "--total-cycles",
        type=int,
        default=60000,
        help="Total duration of synthesized dataset in 60Hz cycles (default: 60000 ~16.6 min)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthesis (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print(" NILM AI Dataset Synthesizer")
    print("==================================================")
    print(f"Input Directory : {args.input_dir}")
    print(f"Output File     : {args.output}")
    print(f"Total Duration  : {args.total_cycles} cycles (~{round(args.total_cycles*0.016667/60, 2)} min)")
    print("--------------------------------------------------")

    # Find preprocessed appliance CSV files
    processed_files = sorted(glob.glob(os.path.join(args.input_dir, "processed_*.csv")))
    # Exclude previously generated synthetic aggregate CSVs
    processed_files = [f for f in processed_files if "synthetic_" not in os.path.basename(f) and "all_nilm" not in os.path.basename(f)]

    if not processed_files:
        print(f"[ERROR] No processed_*.csv files found in {args.input_dir}", file=sys.stderr)
        print("Please run preprocessing first using: python run_preprocessing.py", file=sys.stderr)
        sys.exit(1)

    appliance_files = {}
    noise_file = None

    import re
    for file_path in processed_files:
        base_name = os.path.basename(file_path).replace("processed_", "").replace(".csv", "")
        if "noise_noselfpower" in base_name:
            noise_file = file_path
        elif "noise" in base_name and noise_file is None:
            noise_file = file_path
        elif "noise" not in base_name:
            # Group multi-run measurement files for the same appliance (e.g. kettle_run1, kettle_run2 -> kettle)
            app_key = re.sub(r"(_run\d+|_session\d+|_trial\d+|_\d+)$", "", base_name)
            if app_key not in appliance_files:
                appliance_files[app_key] = []
            appliance_files[app_key].append(file_path)

    print(f"Found {len(appliance_files)} appliance category/categories:")
    for app_name, app_paths in appliance_files.items():
        if len(app_paths) == 1:
            print(f"  - Appliance Category: {app_name:<20} (1 file: {os.path.basename(app_paths[0])})")
        else:
            files_str = ", ".join([os.path.basename(p) for p in app_paths])
            print(f"  - Appliance Category: {app_name:<20} ({len(app_paths)} trial runs pooled: {files_str})")



    # Load config.yaml if exists
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    synthesizer = NILMSynthesizer(config=config)


    try:
        synth_df, summary = synthesizer.synthesize(
            appliance_files=appliance_files,
            background_noise_file=noise_file,
            total_cycles=args.total_cycles,
            output_file=args.output,
            seed=args.seed,
        )

        print("\n[Synthesis Complete]")
        print(f"  - Total Cycles      : {summary['total_synthetic_cycles']}")
        print(f"  - Duration (seconds): {summary['duration_seconds']}s ({round(summary['duration_seconds']/60, 2)} min)")
        print(f"  - Max Agg Power     : {summary['max_aggregate_power_w']} W")
        print(f"  - Mean Agg Power    : {summary['mean_aggregate_power_w']} W")

        print("\nAppliance Activation Schedule Statistics:")
        for app_name, stats in summary["appliance_schedules"].items():
            print(f"  - {app_name:<20}: {stats['on_events_count']} ON events, Duty Cycle = {stats['on_duty_cycle_pct']}%")

        print(f"\nSaved synthetic dataset to: {os.path.abspath(args.output)}")
        print("==================================================")

    except Exception as e:
        print(f"\n[ERROR] Synthesis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
