import pytest
import pandas as pd
import numpy as np
from nilm_preprocessing.sequence_aligner import SequenceAligner


def test_sequence_aligner_sorting_and_dedup():
    raw_data = {
        "seq": [2, 1, 1, 0, 0],
        "cycle": [0, 1, 1, 1, 0],
        "p_w": [10, 5, 5, 1, 0],
        "irms": [0.05, 0.02, 0.02, 0.005, 0.0],
    }
    df = pd.DataFrame(raw_data)

    aligner = SequenceAligner({"sorting": {"deduplicate": True, "cycles_per_packet": 30}})
    aligned_df, report = aligner.align(df)

    # Verify length after deduplication
    assert len(aligned_df) == 4
    assert report["duplicates_removed"] == 1

    # Verify order
    assert list(aligned_df["seq"]) == [0, 0, 1, 2]
    assert list(aligned_df["cycle"]) == [0, 1, 1, 0]

    # Verify global_cycle computation: seq*30 + cycle
    assert list(aligned_df["global_cycle"]) == [0, 1, 31, 60]

    # Verify gap detection
    assert report["total_gaps"] > 0
