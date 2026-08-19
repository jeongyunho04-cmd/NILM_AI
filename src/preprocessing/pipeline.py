"""
Preprocessing Pipeline for NILM AI
Executes full data cleaning and feature engineering across all raw appliance CSV files.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import pandas as pd

from .cleaner import DataCleaner
from .feature_extractor import FeatureExtractor


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for NILM raw files."""

    DEVICE_MAP = {
        "air_conditioner": "air_conditioner",
        "beam_projector": "beam_projector",
        "electiric_kettle": "electiric_kettle",
        "fan_1": "fan",
        "fan_2": "fan",
        "fan_3": "fan",
        "hair_dryer_1": "hair_dryer",
        "hair_dryer_2": "hair_dryer",
        "hotplate_1": "hotplate",
        "hotplate_2": "hotplate",
        "laptop_charger_1": "laptop_charger",
        "laptop_charger_2": "laptop_charger",
        "minipc_1": "minipc",
        "minipc_2": "minipc",
        "noise_noselfpower": "noise_noselfpower",
        "noise_selfpower": "noise_selfpower",
        "oven": "oven",
    }

    def __init__(
        self,
        sampling_hz: float = 60.0,
        noise_floor_w: float = 1.4,
        harmonics_count: int = 15,
    ):
        self.cleaner = DataCleaner(sampling_hz=sampling_hz, noise_floor_w=noise_floor_w)
        self.extractor = FeatureExtractor(harmonics_count=harmonics_count)

    def process_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Loads, cleans, extracts features, and optionally saves the preprocessed dataset."""
        path = Path(file_path)
        stem = path.stem
        appliance_type = self.DEVICE_MAP.get(stem, stem)

        # Load raw CSV
        df_raw = pd.read_csv(path)

        # Set device-specific noise floor if self-powered noise
        custom_noise = 2.37 if "noise_selfpower" in stem else None

        # Clean data
        df_clean, clean_stats = self.cleaner.clean_dataframe(
            df_raw, custom_noise_floor=custom_noise
        )

        # Extract physical and harmonic features
        df_features = self.extractor.extract_features(df_clean)

        # Add metadata columns
        df_features["source_file"] = path.name
        df_features["appliance_type"] = appliance_type

        # Save to output directory if specified
        output_file = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"{stem}_clean.csv"
            df_features.to_csv(output_file, index=False)

        stats = {
            **clean_stats,
            "source_file": path.name,
            "appliance_type": appliance_type,
            "output_file": str(output_file) if output_file else None,
            "feature_columns_count": len(df_features.columns),
            "p_mean": round(float(df_features["p_w"].mean()), 2),
            "p_max": round(float(df_features["p_w"].max()), 2),
            "irms_mean": round(float(df_features["irms"].mean()), 4),
        }

        return df_features, stats

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.csv",
    ) -> Dict[str, Dict]:
        """Processes all CSV files matching pattern in input_dir."""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = sorted(in_path.glob(pattern))
        all_stats = {}

        for f in files:
            _, stats = self.process_file(f, output_dir=out_path)
            all_stats[f.stem] = stats

        return all_stats
