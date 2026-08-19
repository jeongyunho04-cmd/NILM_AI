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
from src.preprocessing.file_registry import FileRole, classify_file, require_known
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
    # 복합 부하 실측(test*, nilm_*)은 별도 디렉터리로 보낸다.
    # SegmentPool 이 npz/ 만 훑기 때문에, 물리적으로 분리해 두면
    # 검증용 파일이 합성 세그먼트 풀에 섞일 경로 자체가 없어진다.
    eval_npz_dir = out_path / "composite_eval"

    for d in [clean_dev_dir, npz_dir, labels_dir, plots_dir, eval_npz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(sampling_hz=60.0, noise_floor_w=1.4)
    annotator = DataAnnotator(sampling_hz=60.0)
    npz_exporter = NumpyDatasetExporter(harmonics_count=15)

    all_csv = sorted(raw_path.glob("*.csv"))

    # 처리 전에 모든 파일의 역할을 먼저 확정한다. 미등록 파일이 하나라도 있으면
    # 여기서 즉시 실패하므로, 정체 모를 파일이 가전으로 둔갑한 채
    # 파이프라인 끝까지 흘러가는 일이 생기지 않는다.
    classifications = [require_known(classify_file(f)) for f in all_csv]
    by_role = {role: [] for role in FileRole}
    for f, c in zip(all_csv, classifications):
        by_role[c.role].append((f, c))

    raw_files = by_role[FileRole.DEVICE] + by_role[FileRole.NOISE]
    eval_files = by_role[FileRole.COMPOSITE_EVAL]

    print(f"\n" + "=" * 80)
    print(f"[NILM AI] Starting Preprocessing, Multi-Tier Labeling & NumPy Binary Export")
    print(f"[NILM AI] Input: {len(all_csv)} CSV files")
    print(f"[NILM AI]   - Single-appliance (segment pool): {len(by_role[FileRole.DEVICE])}")
    print(f"[NILM AI]   - Baseline noise                 : {len(by_role[FileRole.NOISE])}")
    print(f"[NILM AI]   - Composite eval (POOL-EXCLUDED) : {len(eval_files)}"
          f"  -> {', '.join(c.stem for _, c in eval_files) if eval_files else '(none)'}")
    print("=" * 80 + "\n")

    global_report = {
        "processed_files_count": len(raw_files),
        "sampling_hz": 60.0,
        "devices": {},
        "composite_eval_files": {},
        "total_cleaned_samples": 0,
        "total_duration_hours": 0.0,
        "total_events_detected": 0,
    }

    for idx, (f, classification) in enumerate(raw_files, 1):
        stem = classification.stem
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
                # 합성 단계에서 필요한 물리 메타데이터
                "file_role": clean_stats["file_role"],
                "load_class": clean_stats["load_class"],
                "v_ref_v": clean_stats["v_ref_v"],
                "noise_floor_w": clean_stats["noise_floor_applied_w"],
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
            "file_role": clean_stats["file_role"],
            "load_class": clean_stats["load_class"],
            "rows": len(df_annotated),
            "duration_min": round(len(df_annotated) / 60.0 / 60.0, 2),
            "p_mean": clean_stats["p_mean"],
            "p_max": clean_stats["p_max"],
            "v_ref_v": clean_stats["v_ref_v"],
            "invalid_samples_flagged": clean_stats["invalid_samples_flagged"],
            "invalid_samples_dropped": clean_stats["invalid_samples_dropped"],
            "sessions_detected": clean_stats["sessions_detected"],
            "on_percentage": label_summary["on_percentage"],
            "events_count": len(events),
            "npz_file": npz_path_str,
            "state_distribution": label_summary["state_distribution"],
        }

        drop_note = (
            f" | GATED: {clean_stats['invalid_samples_dropped']}"
            if clean_stats["invalid_samples_dropped"] else ""
        )
        print(f"DONE | Rows: {len(df_annotated):6d} ({len(df_annotated)/3600:4.1f}h) | "
              f"ON: {label_summary['on_percentage']:5.1f}% | V_ref: {clean_stats['v_ref_v']:6.1f}V{drop_note}")

    # ── 복합 부하 검증 파일: 정제만 하고 세그먼트 풀과 분리된 곳에 저장 ──────
    # 라벨링은 하지 않는다. 여러 가전이 겹쳐 있어 단일 가전 상태 정의를 적용할 수 없고,
    # 억지로 붙인 라벨이 정답처럼 쓰이면 검증 자체가 무의미해지기 때문이다.
    for f, classification in eval_files:
        stem = classification.stem
        print(f"[eval] Composite load: {f.name:22s} ... ", end="", flush=True)
        df_eval, eval_stats = pipeline.process_file(f)
        eval_npz_file = eval_npz_dir / f"{stem}.npz"
        npz_exporter.export_to_npz(
            df_eval,
            output_path=eval_npz_file,
            metadata={
                "source_file": f.name,
                "file_role": eval_stats["file_role"],
                "sampling_hz": 60.0,
                "duration_s": eval_stats["duration_s"],
                "v_ref_v": eval_stats["v_ref_v"],
                "note": "복합 부하 실측. 단일 가전이 아니므로 상태 라벨과 세그먼트 풀 사용 금지.",
            },
            compress=True,
        )
        global_report["composite_eval_files"][stem] = {
            "rows": eval_stats["cleaned_rows"],
            "duration_s": eval_stats["duration_s"],
            "p_mean": eval_stats["p_mean"],
            "p_max": eval_stats["p_max"],
            "v_ref_v": eval_stats["v_ref_v"],
            "npz_file": str(eval_npz_file),
            "excluded_from_segment_pool": True,
        }
        print(f"DONE | Rows: {eval_stats['cleaned_rows']:6d} | "
              f"V_ref: {eval_stats['v_ref_v']:6.1f}V | (세그먼트 풀 제외)")

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
    if global_report["composite_eval_files"]:
        print(f"[NILM AI] Composite Eval (pool-excluded): {eval_npz_dir.resolve()}")
    print("=" * 80 + "\n")

    return global_report


if __name__ == "__main__":
    run_full_pipeline()
