"""
Visualization Tools for NILM Cleaned & Labeled Appliance Signals
Plots electrical power (P, Q, S), state transitions, and harmonic profiles.
"""
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def plot_appliance_states(
    df: pd.DataFrame,
    title: str = "Appliance NILM States",
    output_path: Optional[Union[str, Path]] = None,
    downsample_factor: int = 10,
) -> Optional[str]:
    """Generates a diagnostic 3-panel figure showing P/Q, states, and harmonics.

    Configures Korean font support for Windows environments.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        # Support Korean font on Windows
        matplotlib.rcParams["font.sans-serif"] = ["Malgun Gothic", "Gulim", "DejaVu Sans", "Arial"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[Warning] Matplotlib is not installed. Skipping graphical plot generation.")
        return None

    # Downsample for faster plotting if large dataset
    df_plot = df.iloc[::downsample_factor].copy().reset_index(drop=True)
    t = df_plot["t_rel_s"].values if "t_rel_s" in df_plot.columns else np.arange(len(df_plot)) / (60.0 / downsample_factor)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.5]})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Power (P, Q, Target P)
    ax1 = axes[0]
    ax1.plot(t, df_plot["p_w"], label="Active Power P (W)", color="#1f77b4", linewidth=1.2, alpha=0.85)
    if "p_target_w" in df_plot.columns:
        ax1.plot(t, df_plot["p_target_w"], label="Target Power (Noise Subtracted)", color="#2ca02c", linestyle="--", linewidth=1.0)
    if "q_var" in df_plot.columns:
        ax1.plot(t, df_plot["q_var"], label="Reactive Power Q (VAR)", color="#ff7f0e", linewidth=1.0, alpha=0.75)
    ax1.set_ylabel("Power (W / VAR)", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_title("Active & Reactive Power Profile", fontsize=11, loc="left")

    # Panel 2: Operational Multi-State
    ax2 = axes[1]
    if "state_id" in df_plot.columns:
        ax2.step(t, df_plot["state_id"], where="post", color="#d62728", linewidth=1.5, label="State ID")
        if "is_on" in df_plot.columns:
            ax2.fill_between(t, 0, df_plot["is_on"], step="post", alpha=0.15, color="#d62728", label="Is ON (Binary)")
        
        # Unique states for y-ticks
        unique_states = sorted(df["state_id"].unique())
        state_names = []
        for s in unique_states:
            names = df[df["state_id"] == s]["state_name"].values
            state_names.append(f"{s}: {names[0]}" if len(names) > 0 else f"State {s}")
        ax2.set_yticks(unique_states)
        ax2.set_yticklabels(state_names, fontsize=9)
    ax2.set_ylabel("State Class", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_title("Multi-Tier State Classification", fontsize=11, loc="left")

    # Panel 3: Harmonics & THD
    ax3 = axes[2]
    if "thd_i" in df_plot.columns:
        ax3.plot(t, df_plot["thd_i"] * 100.0, label="Current THD (%)", color="#9467bd", linewidth=1.2)
    for h in [3, 5, 7]:
        col = f"ih_ratio_{h}"
        if col in df_plot.columns:
            ax3.plot(t, df_plot[col] * 100.0, label=f"ih{h}/ih1 (%)", linewidth=1.0, linestyle="--", alpha=0.8)
    ax3.set_ylabel("Harmonic Ratio (%)", fontsize=11)
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_title("Harmonic Fingerprint & Distortion", fontsize=11, loc="left")

    plt.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_p)

    plt.close(fig)
    return None
