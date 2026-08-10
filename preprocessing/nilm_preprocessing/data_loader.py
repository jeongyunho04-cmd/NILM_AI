"""
Data loader module for NILM AI preprocessing.
Handles loading, concatenating, and initial validation of raw CSV data files.
"""

import glob
import os
from typing import List, Union
import pandas as pd


class DataLoader:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.columns_cfg = self.config.get("columns", {})
        self.timestamp_col = self.columns_cfg.get("timestamp", "host_time")

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV file and parse timestamps."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        df = self._clean_and_cast(df)
        return df

    def load_multiple_csv(self, file_paths_or_pattern: Union[List[str], str]) -> pd.DataFrame:
        """Load and concatenate multiple CSV files matching pattern or list."""
        if isinstance(file_paths_or_pattern, str):
            files = sorted(glob.glob(file_paths_or_pattern))
            if not files and os.path.exists(file_paths_or_pattern):
                files = [file_paths_or_pattern]
        else:
            files = file_paths_or_pattern

        if not files:
            raise FileNotFoundError(f"No CSV files found matching: {file_paths_or_pattern}")

        dfs = []
        for file in files:
            df = self.load_csv(file)
            df["source_file"] = os.path.basename(file)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _clean_and_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from column names and cast essential data types."""
        df.columns = df.columns.str.strip()

        # Parse timestamp if available
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")

        # Essential numeric columns
        numeric_cols = ["seq", "cycle", "irms", "p_w", "vrms", "t_s", "phase_deg", "freq_hz"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
