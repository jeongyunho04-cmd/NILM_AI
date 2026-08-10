"""
Data Augmenter module for NILM AI raw data.
Performs realistic time-series augmentation (Gaussian noise injection, grid voltage drift scaling,
and harmonic perturbation) to match target data volume and enrich steady/transient states.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataAugmenter:
    def __init__(self, config: dict = None):
        self.config = config or {}
        aug_cfg = self.config.get("augmentation", {})
        cols_cfg = self.config.get("columns", {})

        self.enabled = aug_cfg.get("enabled", False)
        self.target_cycles = aug_cfg.get("target_cycles", None)
        self.augmentation_factor = aug_cfg.get("augmentation_factor", 1.0)
        self.noise_std = aug_cfg.get("noise_std", 0.015)
        self.grid_voltage_drift = aug_cfg.get("grid_voltage_drift", 0.03)
        self.augment_states = aug_cfg.get("augment_states", [1, 2, 3])

        self.power_col = cols_cfg.get("active_power", "p_w")
        self.irms_col = cols_cfg.get("irms", "irms")
        self.harmonics_cols = cols_cfg.get(
            "harmonics_current",
            [f"ih{i}" for i in range(1, 16)],
        )

    def augment(
        self,
        df: pd.DataFrame,
        target_cycles: Optional[int] = None,
        augmentation_factor: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Augments DataFrame to meet desired target_cycles or augmentation_factor.
        Returns augmented DataFrame and augmentation report.
        """
        df = df.copy()
        initial_len = len(df)
        if initial_len == 0:
            return df, {"initial_len": 0, "augmented_len": 0, "added_cycles": 0}

        np.random.seed(seed)

        target = target_cycles if target_cycles is not None else self.target_cycles
        factor = augmentation_factor if augmentation_factor is not None else self.augmentation_factor

        # Determine total needed rows
        if target is not None and target > initial_len:
            needed = target - initial_len
        elif factor > 1.0:
            needed = int(initial_len * (factor - 1.0))
        else:
            needed = 0

        if needed <= 0:
            return df, {
                "initial_len": initial_len,
                "augmented_len": initial_len,
                "added_cycles": 0,
                "factor_achieved": 1.0,
            }

        # Filter candidate rows for augmentation (e.g. STEADY_ON, TRANSIENT states, or all if none matched)
        if "state" in df.columns and self.augment_states:
            candidate_mask = df["state"].isin(self.augment_states)
            candidate_indices = df.index[candidate_mask].values
            if len(candidate_indices) == 0:
                candidate_indices = df.index.values
        else:
            candidate_indices = df.index.values

        # Sample indices to replicate with replacement
        sampled_indices = np.random.choice(candidate_indices, size=needed, replace=True)
        aug_df = df.loc[sampled_indices].copy().reset_index(drop=True)

        # Apply augmentation transforms
        aug_df = self._apply_transforms(aug_df)

        # Update global_cycle for continuity
        if "global_cycle" in df.columns:
            max_gc = df["global_cycle"].max()
            aug_df["global_cycle"] = np.arange(max_gc + 1, max_gc + 1 + len(aug_df))
            aug_df["is_augmented"] = True
            df["is_augmented"] = False
        else:
            aug_df["is_augmented"] = True
            df["is_augmented"] = False

        combined_df = pd.concat([df, aug_df], ignore_index=True)

        report = {
            "initial_len": initial_len,
            "augmented_len": len(combined_df),
            "added_cycles": len(aug_df),
            "factor_achieved": round(len(combined_df) / initial_len, 2),
        }

        return combined_df, report

    def _apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Gaussian noise, voltage drift scaling, and harmonic perturbation."""
        n = len(df)
        
        # 1. Grid Voltage Drift Scaling (random scaling per segment between 1-drift and 1+drift)
        drift_factor = 1.0 + np.random.uniform(-self.grid_voltage_drift, self.grid_voltage_drift, size=n)

        # 2. Gaussian relative noise
        power_noise = 1.0 + np.random.normal(0, self.noise_std, size=n)
        irms_noise = 1.0 + np.random.normal(0, self.noise_std, size=n)

        # Apply to Active Power & RMS Current
        if self.power_col in df.columns:
            df[self.power_col] = (df[self.power_col] * drift_factor * power_noise).clip(lower=0.0)
            if "p_w_smooth" in df.columns:
                df["p_w_smooth"] = (df["p_w_smooth"] * drift_factor * power_noise).clip(lower=0.0)

        if self.irms_col in df.columns:
            df[self.irms_col] = (df[self.irms_col] * drift_factor * irms_noise).clip(lower=0.0)
            if "irms_smooth" in df.columns:
                df["irms_smooth"] = (df["irms_smooth"] * drift_factor * irms_noise).clip(lower=0.0)

        # 3. Harmonics Perturbation
        for col in self.harmonics_cols:
            if col in df.columns:
                h_noise = 1.0 + np.random.normal(0, self.noise_std * 1.5, size=n)
                df[col] = (df[col] * drift_factor * h_noise).clip(lower=0.0)

        # Recalculate derivative features if present
        if "d_pw" in df.columns and self.power_col in df.columns:
            df["d_pw"] = df[self.power_col].diff().fillna(0.0)
        if "d_irms" in df.columns and self.irms_col in df.columns:
            df["d_irms"] = df[self.irms_col].diff().fillna(0.0)

        return df
