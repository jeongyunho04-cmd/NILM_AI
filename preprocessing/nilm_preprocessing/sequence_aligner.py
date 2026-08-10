"""
Sequence aligner module for NILM AI preprocessing.
Sorts sequence data by packet sequence number and cycle index, handles duplicates,
calculates continuous global cycle index, and identifies sequence gaps.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


class SequenceAligner:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sorting_cfg = self.config.get("sorting", {})
        self.cols_cfg = self.config.get("columns", {})
        
        self.seq_col = self.cols_cfg.get("seq", "seq")
        self.cycle_col = self.cols_cfg.get("cycle", "cycle")
        self.timestamp_col = self.cols_cfg.get("timestamp", "host_time")
        self.rel_time_col = self.cols_cfg.get("rel_time", "t_s")
        self.cycles_per_packet = self.sorting_cfg.get("cycles_per_packet", 30)
        self.deduplicate = self.sorting_cfg.get("deduplicate", True)

    def align(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Aligns data frame by sequence and cycle, removes duplicates,
        computes continuous global_cycle, and identifies gap statistics.
        """
        df = df.copy()
        initial_len = len(df)

        # 1. Remove NaN in sequence or cycle
        if self.seq_col in df.columns and self.cycle_col in df.columns:
            df = df.dropna(subset=[self.seq_col, self.cycle_col]).copy()
            df[self.seq_col] = df[self.seq_col].astype(int)
            df[self.cycle_col] = df[self.cycle_col].astype(int)

        # 2. Sort by sequence and cycle or timestamp
        if self.seq_col in df.columns and self.cycle_col in df.columns:
            sort_cols = [self.seq_col, self.cycle_col]
            if self.timestamp_col in df.columns:
                sort_cols = [self.timestamp_col] + sort_cols
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
        elif self.timestamp_col in df.columns:
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        # 3. Deduplicate
        duplicates_removed = 0
        if self.deduplicate:
            dedup_subset = []
            if self.seq_col in df.columns and self.cycle_col in df.columns:
                dedup_subset = [self.seq_col, self.cycle_col]
                if "source_file" in df.columns:
                    dedup_subset.append("source_file")
            
            if dedup_subset:
                before_dedup = len(df)
                df = df.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
                duplicates_removed = before_dedup - len(df)

        # 4. Compute global_cycle index
        if self.seq_col in df.columns and self.cycle_col in df.columns:
            min_seq = df[self.seq_col].min() if len(df) > 0 else 0
            df["global_cycle"] = (df[self.seq_col] - min_seq) * self.cycles_per_packet + df[self.cycle_col]
            
            # Check cycle gap / missing frames
            if len(df) > 1:
                df["cycle_diff"] = df["global_cycle"].diff()
                df["is_gap"] = df["cycle_diff"] > 1
                total_gaps = int(df["is_gap"].sum())
                total_missing_cycles = int((df["cycle_diff"].clip(lower=1) - 1).sum())
            else:
                df["cycle_diff"] = 1
                df["is_gap"] = False
                total_gaps = 0
                total_missing_cycles = 0
        else:
            df["global_cycle"] = np.arange(len(df))
            df["is_gap"] = False
            total_gaps = 0
            total_missing_cycles = 0

        report = {
            "initial_rows": initial_len,
            "final_rows": len(df),
            "duplicates_removed": duplicates_removed,
            "total_gaps": total_gaps,
            "total_missing_cycles": total_missing_cycles,
            "start_seq": int(df[self.seq_col].min()) if self.seq_col in df.columns and len(df) > 0 else None,
            "end_seq": int(df[self.seq_col].max()) if self.seq_col in df.columns and len(df) > 0 else None,
        }

        return df, report
