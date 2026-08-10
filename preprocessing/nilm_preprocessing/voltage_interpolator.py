"""
Voltage Interpolator module for NILM AI raw data preprocessing.
Performs 2Hz to 60Hz upsampling & interpolation (linear/cubic) on packet-level slow
voltage parameters (vrms, thd_v, freq_hz) to remove 0.5s step discontinuities.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class VoltageInterpolator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        v_cfg = self.config.get("voltage_interpolation", {})
        cols_cfg = self.config.get("columns", {})

        self.enabled = v_cfg.get("enabled", True)
        self.method = v_cfg.get("method", "linear")  # 'linear', 'cubic', 'pchip'
        self.target_cols = v_cfg.get("cols_to_interpolate", ["vrms", "thd_v", "freq_hz"])
        self.replace_original = v_cfg.get("replace_original", True)
        self.vrms_col = cols_cfg.get("vrms", "vrms")

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolates 2Hz step-wise voltage/frequency signals into smooth 60Hz continuous signals.
        """
        df = df.copy()
        if len(df) <= 1 or not self.enabled:
            return df

        cols_present = [c for c in self.target_cols if c in df.columns]
        if not cols_present:
            return df

        # Identify packet boundary indices (where cycle == 0 or seq changes)
        if "cycle" in df.columns:
            packet_starts = df["cycle"] == 0
        else:
            # Fallback to every 30 rows
            packet_starts = pd.Series(df.index % 30 == 0, index=df.index)

        # Ensure the first and last rows are treated as sample points
        sample_mask = packet_starts.copy()
        sample_mask.iloc[0] = True
        sample_mask.iloc[-1] = True

        for col in cols_present:
            # Extract packet-level sample points
            series_samples = df[col].copy()
            # Set non-packet boundary values to NaN for interpolation
            series_samples[~sample_mask] = np.nan

            # Interpolate
            if self.method in ["cubic", "pchip", "quadratic"]:
                try:
                    interp_series = series_samples.interpolate(
                        method=self.method, limit_direction="both"
                    )
                except Exception:
                    # Fallback to linear if cubic interpolation fails (e.g. not enough points)
                    interp_series = series_samples.interpolate(
                        method="linear", limit_direction="both"
                    )
            else:
                interp_series = series_samples.interpolate(
                    method="linear", limit_direction="both"
                )

            # Fill remaining NaNs if any at edges
            interp_series = interp_series.bfill().ffill()

            if self.replace_original:
                df[col] = interp_series
            else:
                df[f"{col}_interp"] = interp_series

        return df
