"""
Main Execution Script for NILM Data Preprocessing & Multi-Tier State Labeling
Processes all raw CSV files in data/, cleans and reconstructs 60Hz timelines,
extracts physical/spectral features, applies multi-state labeling, and exports:
  1. Annotated CSV datasets (processed_data/clean_devices/*.csv)
  2. High-performance NumPy Binary Archives (processed_data/npz/*.npz with 2-channel Real/Imaginary & complex64)
  3. Events and Summary JSON metadata (processed_data/labels/*.json)
  4. Diagnostic Visual Profile PNGs (processed_data/plots/*.png)
"""
from pathlib import Path
import json
import os
import sys
import time
import pandas as pd

# Safe utf-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.numpy_exporter import NumpyDatasetExporter
from src.labeling.annotator import DataAnnotator
from src.visualization.plot_labeled_data import plot_appliance_states


def run_full_pipeline(
    raw_data_dir: str = "data",
    output_dir: str = "processed_data",
    generate_plots: bool = True,
    export_npz: bool = True,
):
    start_time = time.time()
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)

    clean_dev_dir = out_path / "clean_devices"
    npz_dir = out_path / "npz"
    labels_dir = out_path / "labels"
    plots_dir = out_path / "plots"

    for d in [clean_dev_dir, npz_dir, labels_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(sampling_hz=60.0, noise_floor_w=1.4)
    annotator = DataAnnotator(sampling_hz=60.0)
    npz_exporter = NumpyDatasetExporter(harmonics_count=15)

    raw_files = sorted(raw_path.glob("*.csv"))
    print(f"\n" + "=" * 80)
    print(f"[NILM AI] Starting Preprocessing, Multi-Tier Labeling & NumPy Binary Export")
    print(f"[NILM AI] Input: {len(raw_files)} CSV files | Target: CSV + NPZ (2-Channel Real/Imag) + JSON + Plots")
    print("=" * 80 + "\n")

    global_report = {
        "processed_files_count": len(raw_files),
        "sampling_hz": 60.0,
        "devices": {},
        "total_cleaned_samples": 0,
        "total_duration_hours": 0.0,
        "total_events_detected": 0,
    }

    for idx, f in enumerate(raw_files, 1):
        stem = f.stem
        print(f"[{idx:2d}/{len(raw_files):2d}] Processing: {f.name:22s} ... ", end="", flush=True)
        
        # 1. Clean and extract features
        df_clean, clean_stats = pipeline.process_file(f)
        
        # 2. Annotate states and detect events
        appliance_type = clean_stats["appliance_type"]
        df_annotated, events, label_summary = annotator.annotate_dataframe(
            df_clean, appliance_type=appliance_type
        )
        
        # 3. Save clean annotated dataset (CSV)
        annotated_csv_path = clean_dev_dir / f"{stem}_annotated.csv"
        df_annotated.to_csv(annotated_csv_path, index=False)

        # 4. Save NumPy Binary (.npz) with 2-channel Real/Imaginary & complex64
        npz_path_str = None
        if export_npz:
            npz_file = npz_dir / f"{stem}.npz"
            npz_metadata = {
                "source_file": f.name,
                "appliance_type": appliance_type,
                "korean_name": label_summary["korean_name"],
                "sampling_hz": 60.0,
                "duration_s": label_summary["duration_s"],
                "state_distribution": label_summary["state_distribution"],
            }
            npz_path_str = npz_exporter.export_to_npz(
                df_annotated,
                output_path=npz_file,
                metadata=npz_metadata,
                compress=True,
            )
        
        # 5. Save events and summary JSON
        events_json_path = labels_dir / f"{stem}_events.json"
        summary_json_path = labels_dir / f"{stem}_summary.json"
        
        with open(events_json_path, "w", encoding="utf-8") as fp:
            json.dump([e.__dict__ for e in events], fp, indent=2, ensure_ascii=False)
            
        file_summary = {
            **clean_stats,
            **label_summary,
            "annotated_csv": str(annotated_csv_path),
            "npz_file": npz_path_str,
            "events_json": str(events_json_path),
        }
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(file_summary, fp, indent=2, ensure_ascii=False)

        # 6. Generate diagnostic plot
        plot_path_str = None
        if generate_plots:
            plot_file = plots_dir / f"{stem}_profile.png"
            plot_path_str = plot_appliance_states(
                df_annotated,
                title=f"{stem} ({label_summary['korean_name']}) - NILM Multi-Tier Profile",
                output_path=plot_file,
            )

        # Update global stats
        global_report["total_cleaned_samples"] += len(df_annotated)
        global_report["total_events_detected"] += len(events)
        global_report["devices"][stem] = {
            "appliance_type": appliance_type,
            "korean_name": label_summary["korean_name"],
            "rows": len(df_annotated),
            "duration_min": round(len(df_annotated) / 60.0 / 60.0, 2),
            "p_mean": clean_stats["p_mean"],
            "p_max": clean_stats["p_max"],
            "on_percentage": label_summary["on_percentage"],
            "events_count": len(events),
            "npz_file": npz_path_str,
            "state_distribution": label_summary["state_distribution"],
        }

        print(f"DONE | Rows: {len(df_annotated):6d} ({len(df_annotated)/3600:4.1f}h) | "
              f"ON: {label_summary['on_percentage']:5.1f}% | NPZ: saved")

    global_report["total_duration_hours"] = round(global_report["total_cleaned_samples"] / 60.0 / 3600.0, 2)
    elapsed = round(time.time() - start_time, 2)

    # Save Global Report
    global_report_path = out_path / "global_dataset_report.json"
    with open(global_report_path, "w", encoding="utf-8") as fp:
        json.dump(global_report, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[NILM AI] Preprocessing & NPZ Export Completed in {elapsed}s")
    print(f"[NILM AI] Total Cleaned Samples: {global_report['total_cleaned_samples']:,} cycles ({global_report['total_duration_hours']} hours)")
    print(f"[NILM AI] Total Events Detected: {global_report['total_events_detected']:,}")
    print(f"[NILM AI] NPZ Binary Directory: {(out_path / 'npz').resolve()}")
    print("=" * 80 + "\n")

    return global_report


if __name__ == "__main__":
    run_full_pipeline()
