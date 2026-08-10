import os
import pytest
import pandas as pd
from nilm_preprocessing.pipeline import PreprocessingPipeline


def test_pipeline_execution(tmp_path):
    # Create temporary CSV file
    input_csv = os.path.join(tmp_path, "sample_raw.csv")
    output_csv = os.path.join(tmp_path, "sample_processed.csv")

    df = pd.DataFrame(
        {
            "host_time": ["2026-08-08 19:34:00.000"] * 20,
            "t_s": [i * 0.016667 for i in range(20)],
            "seq": [0] * 20,
            "cycle": list(range(20)),
            "irms": [0.005] * 10 + [0.1] * 10,
            "p_w": [0.01] * 10 + [20.0] * 10,
            "ih1": [0.003] * 20,
            "ih2": [0.001] * 20,
        }
    )
    df.to_csv(input_csv, index=False)

    cfg = {
        "columns": {
            "timestamp": "host_time",
            "rel_time": "t_s",
            "seq": "seq",
            "cycle": "cycle",
            "active_power": "p_w",
            "irms": "irms",
            "harmonics_current": ["ih1", "ih2"],
        },
        "labeling": {
            "on_power_threshold": 5.0,
            "off_power_threshold": 2.0,
            "transient_window_cycles": 3,
        },
    }

    pipeline = PreprocessingPipeline(cfg)
    processed_df, summary = pipeline.run(input_csv, output_path=output_csv)

    assert os.path.exists(output_csv)
    assert len(processed_df) == 20
    assert "state" in processed_df.columns
    assert "d_pw" in processed_df.columns
    assert "ih2_ratio" in processed_df.columns
    assert summary["total_processed_cycles"] == 20
