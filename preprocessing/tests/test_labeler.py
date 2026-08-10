import pytest
import pandas as pd
import numpy as np
from nilm_preprocessing.labeler import ApplianceStateLabeler


def test_appliance_state_labeler():
    # Synthetic load profile: 5 cycles DISCONNECTED (0.1W), 5 cycles STEADY_OFF (1.9W), ON step change with transient, STEADY ON, then OFF
    p_w = [0.1] * 5 + [1.9] * 5 + [40.0, 35.0, 30.0, 25.0] + [25.0] * 15 + [1.9] * 10
    irms = [0.001] * 5 + [0.015] * 5 + [0.2, 0.18, 0.15, 0.12] + [0.12] * 15 + [0.015] * 10


    df = pd.DataFrame({"p_w": p_w, "irms": irms})

    cfg = {
        "labeling": {
            "on_power_threshold": 5.0,
            "off_power_threshold": 2.0,
            "disconnected_power_threshold": 1.8,
            "transient_window_cycles": 5,
            "use_dynamic_transient": False,
        }
    }

    labeler = ApplianceStateLabeler(cfg)
    labeled_df = labeler.label_states(df)

    assert "state" in labeled_df.columns
    assert "state_label" in labeled_df.columns

    # First 5 cycles (0.1W < 1.8W) should be DISCONNECTED (-1)
    assert (labeled_df["state"].iloc[0:5] == -1).all()

    # Next 5 cycles (2.37W) should be STEADY_OFF (0)
    assert (labeled_df["state"].iloc[5:10] == 0).all()

    # Cycles starting at index 10 should be ON_TRANSIENT (1)
    assert (labeled_df["state"].iloc[10:15] == 1).all()

    # Steady ON should be (2)
    assert (labeled_df["state"].iloc[15:25] == 2).all()

    summary = labeler.get_summary(labeled_df)
    assert summary["DISCONNECTED"] > 0
    assert summary["STEADY_OFF"] > 0
    assert summary["ON_TRANSIENT"] > 0
    assert summary["STEADY_ON"] > 0
    assert summary["OFF_TRANSIENT"] > 0

