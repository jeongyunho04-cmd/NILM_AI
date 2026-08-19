"""
Main Execution Script for NILM Load Signal Synthesis & Data Augmentation
Generates rich multi-appliance household scenarios, exports .npz benchmarks,
produces diagnostic visualization plots with voltage drop curves, and tests batch generation speed.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import sys
import time
import numpy as np

# Safe utf-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import LoadSynthesizer, SyntheticLoadSample
from src.synthesis.scenario_generator import ScenarioGenerator
from src.synthesis.dataset import NILMBatchGenerator


def plot_synthetic_scenario(sample: SyntheticLoadSample, title: str, output_path: Union[str, Path]):
    """Generates a comprehensive 3-panel plot of the synthetic scenario."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Malgun Gothic", "Gulim", "DejaVu Sans", "Arial"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[Warning] Matplotlib not available for synthetic plotting.")
        return None

    # Downsample for smooth plotting
    downsample = 10
    t = sample.t_rel_s[::downsample]
    p_tot = sample.power_features[::downsample, 0]
    q_tot = sample.power_features[::downsample, 1]
    v_bus = sample.v_bus[::downsample]
    thd_i = sample.power_features[::downsample, 5]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Composite Power vs Disaggregated Appliance Ground Truths
    ax1 = axes[0]
    ax1.plot(t, p_tot, label="Composite P_total (W)", color="black", linewidth=1.5, alpha=0.9)
    ax1.plot(t, q_tot, label="Composite Q_total (VAR)", color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

    # Stack/plot individual ground truth powers
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for i, app in enumerate(sample.active_appliances):
        app_p = sample.gt_target_power_w[app][::downsample]
        if np.max(app_p) > 1.0:
            c = colors[i % len(colors)]
            ax1.plot(t, app_p, label=f"GT: {app}", color=c, linewidth=1.2, alpha=0.85)

    ax1.set_ylabel("Power (W / VAR)", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_title("Composite Aggregate Load & Disaggregated Appliance Ground Truths", fontsize=11, loc="left")

    # Panel 2: Terminal Bus Voltage (Voltage Drop Sag Simulation)
    ax2 = axes[1]
    ax2.plot(t, v_bus, color="#d62728", linewidth=1.3, label="Terminal Bus Voltage V_bus (V)")
    ax2.axhline(220.0, color="gray", linestyle="--", alpha=0.5, label="Nominal 220V")
    ax2.set_ylabel("Voltage (V)", fontsize=11)
    ax2.set_ylim(min(210.0, np.min(v_bus) - 2.0), max(225.0, np.max(v_bus) + 2.0))
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_title("Grid Impedance Voltage Sag (Z_grid Feedback)", fontsize=11, loc="left")

    # Panel 3: Current Harmonic Distortion (THD_i)
    ax3 = axes[2]
    ax3.plot(t, thd_i * 100.0, color="#9467bd", linewidth=1.2, label="Composite Current THD_i (%)")
    ax3.set_ylabel("THD_i (%)", fontsize=11)
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_title("Composite Current Distortion & Non-linear Fingerprint", fontsize=11, loc="left")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def run_full_synthesis(
    npz_input_dir: str = "processed_data/npz",
    output_dir: str = "synthetic_data",
):
    start_time = time.time()
    out_path = Path(output_dir)
    plots_dir = out_path / "plots"
    scenarios_dir = out_path / "scenarios"

    for d in [out_path, plots_dir, scenarios_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[NILM AI] Initializing Synthesis Engine & Appliance Segment Pool")
    print("=" * 80)

    pool = SegmentPool(npz_dir=npz_input_dir)
    print(f"Loaded {len(pool.get_appliance_types())} appliance categories: {', '.join(pool.get_appliance_types())}")
    for app, acts in pool.appliance_activations.items():
        total_dur = sum(a.duration_s for a in acts)
        print(f"  - {app:20s}: {len(acts):2d} segmented activations (Total: {total_dur/60:.1f} min)")

    grid_sim = GridSimulator(r_grid=0.25, x_grid=0.05)
    augmentor = DataAugmentor(duration_scale_range=(0.6, 2.2), power_scale_std=0.05)
    synthesizer = LoadSynthesizer(segment_pool=pool, grid_simulator=grid_sim, augmentor=augmentor)
    scenario_gen = ScenarioGenerator(synthesizer=synthesizer)

    print("\n" + "=" * 80)
    print("[NILM AI] Generating Benchmark Household Scenarios")
    print("=" * 80 + "\n")

    scenarios = [
        ("morning_routine", "Morning Routine (Kettle, Dryer, Charger, Fan)", lambda: scenario_gen.create_morning_routine(15.0)),
        ("evening_cooking", "Evening Cooking Routine (AC, Oven, Hotplate, Projector)", lambda: scenario_gen.create_evening_cooking_routine(20.0)),
        ("work_office", "Work Office Routine (MiniPC, Charger, Fan, Kettle Breaks)", lambda: scenario_gen.create_work_office_routine(25.0)),
        ("random_scenario_1", "Random Multi-Load Scenario #1", lambda: scenario_gen.create_random_scenario(15.0, num_activations=8)),
        ("random_scenario_2", "Random Multi-Load Scenario #2", lambda: scenario_gen.create_random_scenario(20.0, num_activations=10)),
    ]

    report = {"generated_scenarios": {}, "batch_benchmark": {}}

    for name, desc, func in scenarios:
        print(f"Generating: {name:20s} ({desc}) ... ", end="", flush=True)
        sample: SyntheticLoadSample = func()
        
        # Save NPZ
        npz_file = scenarios_dir / f"{name}.npz"
        ScenarioGenerator.export_synthetic_sample_to_npz(sample, npz_file)
        
        # Save Plot
        plot_file = plots_dir / f"{name}_profile.png"
        plot_synthetic_scenario(sample, title=f"NILM Synthetic Scenario: {desc}", output_path=plot_file)

        report["generated_scenarios"][name] = {
            "description": desc,
            "duration_s": sample.duration_s,
            "duration_cycles": sample.duration_cycles,
            "active_appliances": sample.active_appliances,
            "mean_p_w": sample.metadata["mean_p_w"],
            "max_p_w": sample.metadata["max_p_w"],
            "min_v_bus": sample.metadata["min_v_bus"],
            "max_v_drop": sample.metadata["max_v_drop"],
            "npz_file": str(npz_file),
            "plot_file": str(plot_file),
        }
        print(f"DONE | Duration: {sample.duration_s/60:.1f}m | Max P: {sample.metadata['max_p_w']}W | Max V-Drop: {sample.metadata['max_v_drop']}V")

    # Benchmark on-the-fly Batch Generator for PyTorch Training
    print("\n" + "=" * 80)
    print("[NILM AI] Benchmarking On-The-Fly Real-Time Batch Generator (W=600 cycles = 10s)")
    print("=" * 80)

    batch_gen = NILMBatchGenerator(
        segment_pool=pool,
        window_size_cycles=600,
        max_concurrent_appliances=3,
        target_mode="seq2point",
    )

    t0 = time.time()
    num_batches = 10
    batch_size = 32
    total_samples = num_batches * batch_size

    for _ in range(num_batches):
        X, y_pow, y_state, y_on = batch_gen.generate_batch(batch_size=batch_size)

    elapsed_gen = time.time() - t0
    samples_per_sec = total_samples / elapsed_gen
    print(f"Generated {total_samples} training windows in {elapsed_gen:.2f}s ({samples_per_sec:.1f} windows/sec)")
    print(f"Tensor Shapes -> X: {X.shape}, Y_power: {y_pow.shape}, Y_state: {y_state.shape}, Y_on: {y_on.shape}")

    report["batch_benchmark"] = {
        "batch_size": batch_size,
        "window_size_cycles": 600,
        "samples_per_second": round(samples_per_sec, 1),
        "x_shape": list(X.shape),
        "y_power_shape": list(y_pow.shape),
    }

    # Save synthesis report
    report_file = out_path / "synthesis_report.json"
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    total_elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"[NILM AI] Synthesis & Augmentation Pipeline Completed in {total_elapsed}s")
    print(f"[NILM AI] Synthetic Dataset Saved To: {out_path.resolve()}")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    run_full_synthesis()
