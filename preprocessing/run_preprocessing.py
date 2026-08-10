#!/usr/bin/env python3
"""
CLI Command Line Interface for NILM AI Raw Data Preprocessing.
Usage:
    python run_preprocessing.py --input "../data/*.csv" --output "./output/processed_data.csv"
"""

import argparse
import json
import os
import sys
import yaml

from nilm_preprocessing.pipeline import PreprocessingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw NILM sensor data for AI model training."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=os.path.join("..", "data", "*.csv"),
        help="Input CSV file path, directory, or glob pattern (default: ../data/*.csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Output directory for processed CSV files (default: ./output)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all input CSVs into a single output CSV file instead of separate files",
    )
    parser.add_argument(
        "--on-power",
        type=float,
        default=None,
        help="Override ON active power threshold (W)",
    )
    parser.add_argument(
        "--off-power",
        type=float,
        default=None,
        help="Override OFF active power threshold (W)",
    )
    parser.add_argument(
        "--transient-window",
        type=int,
        default=None,
        help="Override transient window size in cycles",
    )
    parser.add_argument(
        "--target-cycles",
        type=int,
        default=None,
        help="Target total cycle count to reach via data augmentation (e.g. 50000)",
    )
    parser.add_argument(
        "--augment-factor",
        type=float,
        default=None,
        help="Data volume multiplier factor for data augmentation (e.g. 2.0 to double size)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # CLI Overrides
    if "labeling" not in config:
        config["labeling"] = {}
    if args.on_power is not None:
        config["labeling"]["on_power_threshold"] = args.on_power
    if args.off_power is not None:
        config["labeling"]["off_power_threshold"] = args.off_power
    if args.transient_window is not None:
        config["labeling"]["transient_window_cycles"] = args.transient_window

    print("==================================================")
    print(" NILM AI Raw Data Preprocessing & Augmentation")
    print("==================================================")
    print(f"Input Pattern : {args.input}")
    print(f"Output Dir    : {args.output_dir}")
    print(f"Config File   : {args.config}")
    print(f"Mode          : {'Combined Single File' if args.combine else 'Separate Per-File Processing'}")
    if args.target_cycles:
        print(f"Augmentation  : Target Cycles = {args.target_cycles}")
    elif args.augment_factor:
        print(f"Augmentation  : Augment Factor = {args.augment_factor}x")
    print("--------------------------------------------------")

    pipeline = PreprocessingPipeline(config)
    
    try:
        if args.combine:
            output_file = os.path.join(args.output_dir, "processed_all_nilm.csv")
            processed_df, summary = pipeline.run(args.input, output_file)
            print("\n[Processing Complete]")
            print(f"Saved combined dataset to: {os.path.abspath(output_file)}")
        else:
            batch_summaries = pipeline.run_batch(
                args.input,
                args.output_dir,
                target_cycles=args.target_cycles,
                augmentation_factor=args.augment_factor,
            )
            print("\n[Processing Complete - Individual Files]")
            for file_name, sum_info in batch_summaries.items():
                align = sum_info["alignment"]
                dist = sum_info["state_distribution"]
                out_p = sum_info["output_file"]
                aug = sum_info.get("augmentation", {})
                aug_str = f", Augment: +{aug.get('added_cycles', 0)} cycles" if aug.get("added_cycles", 0) > 0 else ""
                print(f"\n[File] {file_name}")
                print(f"   - Saved To: {os.path.abspath(out_p)}")
                print(f"   - Total Cycles: {sum_info['total_processed_cycles']} (Gaps: {align['total_gaps']}{aug_str})")
                print(f"   - States: OFF={dist.get('STEADY_OFF', 0)}, ON_TRANS={dist.get('ON_TRANSIENT', 0)}, STEADY_ON={dist.get('STEADY_ON', 0)}, OFF_TRANS={dist.get('OFF_TRANSIENT', 0)}")
                
        print("\n==================================================")

        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

