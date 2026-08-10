"""
Feature engineering module for NILM AI preprocessing.
Calculates smoothed signals, delta features (dP, dI), and harmonic ratios.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        features_cfg = self.config.get("features", {})
        cols_cfg = self.config.get("columns", {})

        self.power_col = cols_cfg.get("active_power", "p_w")
        self.irms_col = cols_cfg.get("irms", "irms")
        self.harmonics_cols = cols_cfg.get(
            "harmonics_current",
            [f"ih{i}" for i in range(1, 16)],
        )

        self.smooth_window = features_cfg.get("smooth_window", 5)
        self.calculate_deltas = features_cfg.get("calculate_deltas", True)
        self.normalize_harmonics = features_cfg.get("normalize_harmonics", True)
        self.drop_voltage_harmonics = features_cfg.get("drop_voltage_harmonics", True)
        self.apply_harmonic_decay_weight = features_cfg.get("apply_harmonic_decay_weight", True)
        self.harmonic_decay_exponent = features_cfg.get("harmonic_decay_exponent", 1.0)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds engineered features to the DataFrame."""
        df = df.copy()
        if len(df) == 0:
            return df

        # 1. Drop Voltage Harmonics (vh1 ~ vh15) if enabled
        if self.drop_voltage_harmonics:
            vh_cols = [f"vh{i}" for i in range(1, 16)]
            df = df.drop(columns=[c for c in vh_cols if c in df.columns], errors="ignore")

        # 2. Reactive Power (Q), Apparent Power (S), Power Factor (PF), and Phase Shift Angle
        if self.power_col in df.columns and self.irms_col in df.columns and "vrms" in df.columns:
            p_val = np.maximum(df[self.power_col].values, 0.0)
            v_val = df["vrms"].values
            i_val = df[self.irms_col].values
            
            s_val = v_val * i_val  # Apparent Power (VA)
            q_sq = np.maximum(s_val**2 - p_val**2, 0.0)
            q_val = np.sqrt(q_sq)  # Reactive Power (var)
            
            pf_val = np.divide(p_val, s_val, out=np.ones_like(p_val), where=(s_val > 1e-4))
            pf_val = np.clip(pf_val, 0.0, 1.0)  # Power Factor [0, 1]
            
            phase_rad = np.arccos(pf_val)  # Phase angle shift in radians

            
            df["s_va"] = s_val
            df["q_var"] = q_val
            df["power_factor"] = pf_val
            df["phase_rad"] = phase_rad

        # 3. Delta features
        if self.calculate_deltas:
            if self.power_col in df.columns:
                df["d_pw"] = df[self.power_col].diff().fillna(0.0)
            if self.irms_col in df.columns:
                df["d_irms"] = df[self.irms_col].diff().fillna(0.0)
            if "q_var" in df.columns:
                df["d_qvar"] = df["q_var"].diff().fillna(0.0)


        # 3. Moving average smoothing
        if self.smooth_window > 1:
            if self.power_col in df.columns:
                df["p_w_smooth"] = (
                    df[self.power_col]
                    .rolling(window=self.smooth_window, min_periods=1, center=True)
                    .mean()
                )
            if self.irms_col in df.columns:
                df["irms_smooth"] = (
                    df[self.irms_col]
                    .rolling(window=self.smooth_window, min_periods=1, center=True)
                    .mean()
                )

        # 4. Harmonic Normalization & Order Decay Weighting (1 / h^exponent)
        existing_h_cols = [c for c in self.harmonics_cols if c in df.columns]
        if "ih1" in df.columns and len(existing_h_cols) > 0:
            fund = df["ih1"].replace(0.0, np.nan)
            
            for col in existing_h_cols:
                # Extract harmonic order number (e.g. 'ih5' -> order 5)
                order_str = col.replace("ih", "")
                order = int(order_str) if order_str.isdigit() else 1
                decay_w = 1.0 / (order ** self.harmonic_decay_exponent)

                if self.normalize_harmonics:
                    ratio_col = f"{col}_ratio"
                    df[ratio_col] = (df[col] / fund).fillna(0.0)

                if self.apply_harmonic_decay_weight:
                    if self.normalize_harmonics:
                        weighted_col = f"{col}_ratio_w"
                        df[weighted_col] = df[f"{col}_ratio"] * decay_w
                    else:
                        weighted_col = f"{col}_w"
                        df[weighted_col] = df[col] * decay_w

        return df

