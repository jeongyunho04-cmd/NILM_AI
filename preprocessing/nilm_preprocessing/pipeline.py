"""
End-to-end preprocessing pipeline for NILM AI raw data.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import yaml
import pandas as pd

from .data_loader import DataLoader
from .sequence_aligner import SequenceAligner
from .labeler import ApplianceStateLabeler
from .feature_engineering import FeatureEngineer
from .voltage_interpolator import VoltageInterpolator
from .augmenter import DataAugmenter


class PreprocessingPipeline:
    def __init__(self, config_path_or_dict: Union[str, dict] = None):
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            self.config = {}

        self.loader = DataLoader(self.config)
        self.aligner = SequenceAligner(self.config)
        self.v_interpolator = VoltageInterpolator(self.config)
        self.labeler = ApplianceStateLabeler(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.augmenter = DataAugmenter(self.config)

    def process_dataframe(
        self,
        raw_df: pd.DataFrame,
        target_cycles: Optional[int] = None,
        augmentation_factor: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Processes a single raw DataFrame:
        1. Aligns sequences and cleans duplicates/gaps
        2. Interpolates 2Hz step-wise voltage/frequency to continuous 60Hz signals
        3. Labels ON/OFF/Transient appliance states
        4. Computes engineered features
        5. (Optional) Augments dataset to meet target_cycles or augmentation_factor
        """
        aligned_df, align_report = self.aligner.align(raw_df)
        v_interp_df = self.v_interpolator.interpolate(aligned_df)
        labeled_df = self.labeler.label_states(v_interp_df)
        processed_df = self.engineer.process(labeled_df)


        aug_report = {}
        # Apply data augmentation if target_cycles/augmentation_factor requested or config enabled
        if target_cycles or augmentation_factor or self.augmenter.enabled:
            processed_df, aug_report = self.augmenter.augment(
                processed_df,
                target_cycles=target_cycles,
                augmentation_factor=augmentation_factor,
            )

        label_summary = self.labeler.get_summary(processed_df)

        summary = {
            "alignment": align_report,
            "state_distribution": label_summary,
            "augmentation": aug_report,
            "total_processed_cycles": len(processed_df),
        }
        return processed_df, summary


    def run_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        target_cycles: Optional[int] = None,
        augmentation_factor: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Executes preprocessing on a single CSV file and saves to output_file if provided.
        """
        raw_df = self.loader.load_csv(input_file)
        raw_df["source_file"] = os.path.basename(input_file)
        
        processed_df, summary = self.process_dataframe(
            raw_df,
            target_cycles=target_cycles,
            augmentation_factor=augmentation_factor,
        )
        summary["source_file"] = os.path.basename(input_file)
        summary["output_file"] = output_file

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            processed_df.to_csv(output_file, index=False)

        return processed_df, summary

    def run_batch(
        self,
        input_paths_or_pattern: Union[List[str], str],
        output_dir: str,
        prefix: str = "processed_",
        target_cycles: Optional[int] = None,
        augmentation_factor: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """
        Executes per-file preprocessing on each CSV file matching pattern/list
        and saves each output file separately into output_dir.
        """
        import glob
        if isinstance(input_paths_or_pattern, str):
            files = sorted(glob.glob(input_paths_or_pattern))
            if not files and os.path.exists(input_paths_or_pattern):
                files = [input_paths_or_pattern]
        else:
            files = input_paths_or_pattern

        if not files:
            raise FileNotFoundError(f"No CSV files found matching: {input_paths_or_pattern}")

        os.makedirs(output_dir, exist_ok=True)
        batch_summaries = {}

        for file_path in files:
            base_name = os.path.basename(file_path)
            out_name = f"{prefix}{base_name}"
            out_path = os.path.join(output_dir, out_name)

            _, summary = self.run_file(
                file_path,
                output_file=out_path,
                target_cycles=target_cycles,
                augmentation_factor=augmentation_factor,
            )
            batch_summaries[base_name] = summary

        return batch_summaries


    def run(
        self,
        input_paths_or_pattern: Union[List[str], str],
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Executes preprocessing pipeline on combined CSV files (legacy single-df interface).
        """
        raw_df = self.loader.load_multiple_csv(input_paths_or_pattern)
        processed_df, summary = self.process_dataframe(raw_df)
        summary["output_file"] = output_path

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            processed_df.to_csv(output_path, index=False)

        return processed_df, summary

