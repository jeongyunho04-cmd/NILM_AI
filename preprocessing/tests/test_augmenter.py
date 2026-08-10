import pytest
import pandas as pd
import numpy as np
from nilm_preprocessing.augmenter import DataAugmenter


def test_data_augmenter_target_cycles():
    df = pd.DataFrame(
        {
            "global_cycle": list(range(100)),
            "p_w": [25.0] * 100,
            "irms": [0.12] * 100,
            "ih1": [0.10] * 100,
            "state": [2] * 100,
        }
    )

    augmenter = DataAugmenter(
        {
            "augmentation": {
                "noise_std": 0.02,
                "grid_voltage_drift": 0.03,
            }
        }
    )

    aug_df, report = augmenter.augment(df, target_cycles=250)

    assert len(aug_df) == 250
    assert report["added_cycles"] == 150
    assert "is_augmented" in aug_df.columns
    # Check that augmented samples have slight variation (noise)
    aug_samples = aug_df[aug_df["is_augmented"]]
    assert not (aug_samples["p_w"] == 25.0).all()
