import pytest
import pandas as pd
import numpy as np
from nilm_preprocessing.voltage_interpolator import VoltageInterpolator


def test_voltage_interpolator_linear():
    # 60 cycles total (2 packets of 30 cycles each)
    # Packet 0 (cycle 0..29): vrms = 220.0
    # Packet 1 (cycle 0..29): vrms = 230.0
    vrms_raw = [220.0] * 30 + [230.0] * 30
    cycles = list(range(30)) + list(range(30))
    df = pd.DataFrame({"cycle": cycles, "vrms": vrms_raw})

    interpolator = VoltageInterpolator(
        {
            "voltage_interpolation": {
                "enabled": True,
                "method": "linear",
                "cols_to_interpolate": ["vrms"],
                "replace_original": True,
            }
        }
    )

    interp_df = interpolator.interpolate(df)

    # Check that mid-packet cycle 15 has smooth intermediate voltage between 220.0 and 230.0
    vrms_mid = interp_df["vrms"].iloc[15]
    assert 220.0 < vrms_mid < 230.0
    # Check start and end
    assert interp_df["vrms"].iloc[0] == 220.0
    assert interp_df["vrms"].iloc[30] == 230.0
