import os
import pytest
import pandas as pd
import numpy as np
from nilm_preprocessing.synthesizer import NILMSynthesizer


def test_nilm_synthesizer(tmp_path):
    app1_file = os.path.join(tmp_path, "processed_app1.csv")
    app2_file = os.path.join(tmp_path, "processed_app2.csv")
    output_file = os.path.join(tmp_path, "synthetic_nilm_dataset.csv")

    # Mock appliance 1
    df1 = pd.DataFrame(
        {
            "p_w": [0.1] * 20 + [50.0] * 50 + [0.1] * 30,
            "irms": [0.001] * 20 + [0.25] * 50 + [0.001] * 30,
            "state": [0] * 20 + [2] * 50 + [0] * 30,
            "state_label": ["STEADY_OFF"] * 20 + ["STEADY_ON"] * 50 + ["STEADY_OFF"] * 30,
            "ih1": [0.001] * 20 + [0.2] * 50 + [0.001] * 30,
        }
    )
    df1.to_csv(app1_file, index=False)

    # Mock appliance 2
    df2 = pd.DataFrame(
        {
            "p_w": [0.1] * 10 + [120.0] * 40 + [0.1] * 50,
            "irms": [0.001] * 10 + [0.6] * 40 + [0.001] * 50,
            "state": [0] * 10 + [2] * 40 + [0] * 50,
            "state_label": ["STEADY_OFF"] * 10 + ["STEADY_ON"] * 40 + ["STEADY_OFF"] * 50,
            "ih1": [0.001] * 10 + [0.5] * 40 + [0.001] * 50,
        }
    )
    df2.to_csv(app2_file, index=False)

    synthesizer = NILMSynthesizer(
        {
            "synthesis": {
                "min_on_cycles": 20,
                "max_on_cycles": 40,
                "min_off_cycles": 10,
                "max_off_cycles": 20,
            }
        }
    )

    synth_df, summary = synthesizer.synthesize(
        appliance_files={"app1": app1_file, "app2": app2_file},
        total_cycles=500,
        output_file=output_file,
        seed=123,
    )

    assert os.path.exists(output_file)
    assert len(synth_df) == 500
    assert "p_w_agg" in synth_df.columns
    assert "irms_agg" in synth_df.columns
    assert "p_w_app1" in synth_df.columns
    assert "state_app1" in synth_df.columns
    assert "p_w_app2" in synth_df.columns
    assert "state_app2" in synth_df.columns

    # Verify linear superposition: p_w_agg >= p_w_app1 + p_w_app2
    assert (synth_df["p_w_agg"] >= synth_df["p_w_app1"] + synth_df["p_w_app2"] - 1.0).all()
