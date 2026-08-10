"""
NILM AI Data Synthesizer Module.
Synthesizes multi-appliance aggregate load datasets by randomly scheduling, overlaying,
and superimposing individual preprocessed appliance streams (power, current, harmonics, states)
onto background noise streams for NILM AI model training & evaluation.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class NILMSynthesizer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        synth_cfg = self.config.get("synthesis", {})

        self.default_total_cycles = synth_cfg.get("total_cycles", 50000)
        self.min_on_cycles = synth_cfg.get("min_on_cycles", 300)      # ~5 seconds minimum ON
        self.max_on_cycles = synth_cfg.get("max_on_cycles", 3600)     # ~60 seconds maximum ON
        self.min_off_cycles = synth_cfg.get("min_off_cycles", 200)     # ~3.3 seconds minimum OFF
        self.max_off_cycles = synth_cfg.get("max_off_cycles", 2400)    # ~40 seconds maximum OFF
        self.keep_standby_power = synth_cfg.get("keep_standby_power", True)
        self.seed = synth_cfg.get("seed", 42)


    def synthesize(
        self,
        appliance_files: Dict[str, str],
        background_noise_file: Optional[str] = None,
        total_cycles: Optional[int] = None,
        output_file: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Synthesizes aggregate load dataset from individual appliance preprocessed CSVs.

        Args:
            appliance_files: Dict mapping appliance_name -> CSV file path
                             e.g. {'laptop': 'processed_laptop.csv', 'kettle': 'processed_kettle.csv'}
            background_noise_file: Path to background noise CSV (optional)
            total_cycles: Desired duration in 60Hz cycles (default: 50,000)
            output_file: File path to save synthetic dataset CSV
            seed: Random seed for reproducibility

        Returns:
            Tuple of (synthetic_df, summary_report)
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        N = total_cycles or self.default_total_cycles

        # 1. Load appliance dataframes (supports single file or list of trial files per appliance)
        appliance_dfs = {}
        for name, file_path_or_list in appliance_files.items():
            if isinstance(file_path_or_list, str):
                paths = [file_path_or_list]
            else:
                paths = file_path_or_list

            dfs = []
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Appliance CSV file not found: {path}")
                dfs.append(pd.read_csv(path))

            appliance_dfs[name] = dfs


        # 2. Base Noise Stream & Base Harmonics
        harmonics_agg = {f"ih{h}": np.zeros(N) for h in range(1, 16)}

        if background_noise_file and os.path.exists(background_noise_file):
            noise_df = pd.read_csv(background_noise_file)
            p_noise = self._tile_or_truncate(noise_df["p_w"].values, N)
            i_noise = self._tile_or_truncate(noise_df["irms"].values, N)
            for h in range(1, 16):
                h_col = f"ih{h}"
                if h_col in noise_df.columns:
                    harmonics_agg[h_col] = self._tile_or_truncate(noise_df[h_col].values, N)
        else:
            p_noise = np.clip(np.random.normal(1.40, 0.05, size=N), 0.0, None)
            i_noise = np.clip(np.random.normal(0.0068, 0.0005, size=N), 0.0, None)
            harmonics_agg["ih1"] = i_noise.copy()

        # 3. Create Timeline & Aggregate Accumulators
        timeline_df = pd.DataFrame()
        timeline_df["global_cycle"] = np.arange(N)
        timeline_df["t_s"] = timeline_df["global_cycle"] * 0.016667
        
        p_agg = p_noise.copy()
        i_sq_agg = i_noise ** 2

        appliance_stats = {}

        # 4. Schedule and Superimpose Each Appliance
        for app_name, app_df in appliance_dfs.items():
            app_p, app_i, app_state, app_state_label, app_h_dict, stats = self._generate_appliance_timeline(
                app_df, N
            )

            # Superimpose Power & Current RMS
            p_agg += app_p
            i_sq_agg += app_i ** 2
            
            # Superimpose Harmonics
            for h in range(1, 16):
                h_col = f"ih{h}"
                if h_col in app_h_dict:
                    harmonics_agg[h_col] += app_h_dict[h_col]

            # Add ground truth columns for this appliance to synthetic dataset
            timeline_df[f"p_w_{app_name}"] = app_p
            timeline_df[f"irms_{app_name}"] = app_i
            timeline_df[f"state_{app_name}"] = app_state
            timeline_df[f"state_label_{app_name}"] = app_state_label

            appliance_stats[app_name] = stats

        # 5. Calculate Final Aggregate Features
        timeline_df["p_w_agg"] = p_agg
        timeline_df["irms_agg"] = np.sqrt(i_sq_agg)
        timeline_df["d_pw_agg"] = timeline_df["p_w_agg"].diff().fillna(0.0)
        timeline_df["d_irms_agg"] = timeline_df["irms_agg"].diff().fillna(0.0)
        
        # Smooth aggregate features
        timeline_df["p_w_agg_smooth"] = (
            timeline_df["p_w_agg"]
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )
        timeline_df["irms_agg_smooth"] = (
            timeline_df["irms_agg"]
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )

        # Aggregate Harmonics Ratios & Decay Weighting
        fund_agg = harmonics_agg["ih1"]
        fund_safe = np.where(fund_agg > 1e-6, fund_agg, np.nan)
        
        for h in range(1, 16):
            h_col = f"ih{h}"
            timeline_df[f"{h_col}_agg"] = harmonics_agg[h_col]
            
            ratio_series = pd.Series(harmonics_agg[h_col] / fund_safe).fillna(0.0)
            timeline_df[f"{h_col}_ratio_agg"] = ratio_series
            timeline_df[f"{h_col}_ratio_w_agg"] = ratio_series * (1.0 / h)

        # 6. Save if output_file specified
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            timeline_df.to_csv(output_file, index=False)


        summary = {
            "total_synthetic_cycles": N,
            "duration_seconds": round(N * 0.016667, 2),
            "max_aggregate_power_w": round(float(p_agg.max()), 2),
            "mean_aggregate_power_w": round(float(p_agg.mean()), 2),
            "appliance_schedules": appliance_stats,
            "output_file": output_file,
        }

        return timeline_df, summary

    def _generate_appliance_timeline(
        self, app_dfs: List[pd.DataFrame], N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray], Dict]:
        """
        Generates realistic ON/OFF event activation timeline for a single appliance,
        randomly sampling across multiple measurement trial DataFrames if available,
        and preserving appliance standby power during OFF/standby periods.
        """
        # Calculate mean standby power across trial DataFrames (cycles where state <= 0)
        standby_p_list = []
        standby_i_list = []
        for df_item in app_dfs:
            if "state" in df_item.columns and "p_w" in df_item.columns:
                off_mask = df_item["state"] <= 0
                if off_mask.any():
                    standby_p_list.append(df_item.loc[off_mask, "p_w"].mean())
                    if "irms" in df_item.columns:
                        standby_i_list.append(df_item.loc[off_mask, "irms"].mean())

        mean_p_standby = float(np.mean(standby_p_list)) if standby_p_list else 0.0
        mean_i_standby = float(np.mean(standby_i_list)) if standby_i_list else 0.0

        if self.keep_standby_power and mean_p_standby > 0:
            p_stream = np.full(N, mean_p_standby)
            i_stream = np.full(N, mean_i_standby)
        else:
            p_stream = np.zeros(N)
            i_stream = np.zeros(N)

        state_stream = np.zeros(N, dtype=int)
        state_label_stream = ["STEADY_OFF"] * N

        h_dict = {f"ih{h}": np.zeros(N) for h in range(1, 16)}
        events_count = 0
        current_idx = np.random.randint(50, 300)


        while current_idx < N - 100:
            # Random OFF duration
            off_dur = np.random.randint(self.min_off_cycles, self.max_off_cycles)
            current_idx += off_dur
            if current_idx >= N:
                break

            # Pick a trial DataFrame at random from the appliance trial pool
            app_df = app_dfs[np.random.randint(0, len(app_dfs))]

            app_p = app_df["p_w"].values if "p_w" in app_df.columns else np.zeros(len(app_df))
            app_i = app_df["irms"].values if "irms" in app_df.columns else np.zeros(len(app_df))
            app_state = app_df["state"].values if "state" in app_df.columns else np.zeros(len(app_df), dtype=int)
            app_state_label = app_df["state_label"].values if "state_label" in app_df.columns else ["STEADY_OFF"] * len(app_df)

            app_length = len(app_df)
            on_indices = np.where(app_state > 0)[0]
            if len(on_indices) == 0:
                on_indices = np.arange(app_length)

            # Random ON segment extraction
            on_dur = np.random.randint(self.min_on_cycles, self.max_on_cycles)
            seg_end = min(N, current_idx + on_dur)
            actual_len = seg_end - current_idx

            # Pick random slice from appliance trial data containing ON/Transient behavior
            slice_start = np.random.choice(on_indices)
            slice_end = min(app_length, slice_start + actual_len)
            slice_len = slice_end - slice_start

            # Overlay slice
            p_stream[current_idx : current_idx + slice_len] = app_p[slice_start:slice_end]
            i_stream[current_idx : current_idx + slice_len] = app_i[slice_start:slice_end]
            state_stream[current_idx : current_idx + slice_len] = app_state[slice_start:slice_end]

            # Copy labels
            for offset in range(slice_len):
                state_label_stream[current_idx + offset] = app_state_label[slice_start + offset]

            # Copy Harmonics
            for h in range(1, 16):
                h_col = f"ih{h}"
                if h_col in app_df.columns:
                    h_dict[h_col][current_idx : current_idx + slice_len] = app_df[h_col].values[slice_start:slice_end]

            events_count += 1
            current_idx += slice_len

        stats = {
            "on_events_count": events_count,
            "on_duty_cycle_pct": round((state_stream > 0).sum() / N * 100, 2),
            "trial_runs_pooled": len(app_dfs),
        }

        return p_stream, i_stream, state_stream, state_label_stream, h_dict, stats


    def _tile_or_truncate(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        """Repeats or truncates array to match exact target_length."""
        if len(arr) == 0:
            return np.zeros(target_length)
        if len(arr) >= target_length:
            return arr[:target_length]
        reps = int(np.ceil(target_length / len(arr)))
        return np.tile(arr, reps)[:target_length]
