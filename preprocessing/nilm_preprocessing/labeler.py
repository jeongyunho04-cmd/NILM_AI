"""
Appliance state labeler module for NILM AI preprocessing.
Labels ON/OFF states and transient state transitions (ON_TRANSIENT, OFF_TRANSIENT).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class ApplianceStateLabeler:
    """
    State Labels:
     -1: DISCONNECTED (Sensor disconnected / power off)
      0: STEADY_OFF (Measuring board online, standby power ~2.37W)
      1: ON_TRANSIENT (Appliance turning ON surge)
      2: STEADY_ON (Appliance running)
      3: OFF_TRANSIENT (Appliance turning OFF decay)
    """

    STATE_CODES = {
        "DISCONNECTED": -1,
        "STEADY_OFF": 0,
        "ON_TRANSIENT": 1,
        "STEADY_ON": 2,
        "OFF_TRANSIENT": 3,
    }

    STATE_LABELS = {v: k for k, v in STATE_CODES.items()}

    def __init__(self, config: dict = None):
        self.config = config or {}
        labeling_cfg = self.config.get("labeling", {})
        cols_cfg = self.config.get("columns", {})

        self.power_col = cols_cfg.get("active_power", "p_w")
        self.irms_col = cols_cfg.get("irms", "irms")
        
        self.on_power_threshold = labeling_cfg.get("on_power_threshold", 5.0)
        self.off_power_threshold = labeling_cfg.get("off_power_threshold", 2.0)
        self.disconnected_power_threshold = labeling_cfg.get("disconnected_power_threshold", 1.8)
        self.irms_threshold = labeling_cfg.get("irms_threshold", 0.02)
        self.transient_window_cycles = labeling_cfg.get("transient_window_cycles", 30)
        self.use_dynamic_transient = labeling_cfg.get("use_dynamic_transient", True)
        self.min_delta_p_transient = labeling_cfg.get("min_delta_p_transient", 1.5)

    def label_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates appliance state (-1, 0, 1, 2, 3) for each cycle in the DataFrame.
        """
        df = df.copy()
        n = len(df)
        if n == 0:
            df["state"] = []
            df["state_label"] = []
            return df

        power = df[self.power_col].values if self.power_col in df.columns else np.zeros(n)
        irms = df[self.irms_col].values if self.irms_col in df.columns else np.zeros(n)

        # 1. Determine raw binary state using Hysteresis
        raw_on = np.zeros(n, dtype=bool)
        current_state = False

        for i in range(n):
            p = power[i]
            i_rms = irms[i]
            
            # Condition for turning ON
            if not current_state:
                if p >= self.on_power_threshold or i_rms >= self.irms_threshold:
                    current_state = True
            # Condition for turning OFF
            else:
                if p <= self.off_power_threshold and i_rms < self.irms_threshold:
                    current_state = False

            raw_on[i] = current_state

        # 2. Identify transitions
        state = np.zeros(n, dtype=int)  # Default 0: STEADY_OFF (Board online, standby)
        state[raw_on] = self.STATE_CODES["STEADY_ON"]  # 2: STEADY_ON

        # Mark DISCONNECTED (-1) for cycles below disconnected power threshold
        disconnected_mask = (power < self.disconnected_power_threshold) & (~raw_on)
        state[disconnected_mask] = self.STATE_CODES["DISCONNECTED"]  # -1

        # Detect OFF -> ON transitions
        on_transitions = np.where((~raw_on[:-1]) & (raw_on[1:]))[0] + 1
        # Detect ON -> OFF transitions
        off_transitions = np.where((raw_on[:-1]) & (~raw_on[1:]))[0] + 1


        # Calculate derivative dP for dynamic transient bound
        dP = np.abs(np.diff(power, prepend=power[0]))

        # 3. Label ON_TRANSIENT (1)
        for t_idx in on_transitions:
            window_end = min(n, t_idx + self.transient_window_cycles)
            if self.use_dynamic_transient:
                # Find where power derivative settles below threshold or raw_on flips back
                sub_dP = dP[t_idx:window_end]
                stable_idx = np.where(sub_dP < self.min_delta_p_transient)[0]
                if len(stable_idx) > 2:  # require at least 2 consecutive stable cycles
                    # Find first stable point after peak
                    peak_offset = np.argmax(sub_dP)
                    after_peak_stable = stable_idx[stable_idx > peak_offset]
                    if len(after_peak_stable) > 0:
                        window_end = min(window_end, t_idx + after_peak_stable[0])

            # Assign ON_TRANSIENT for the window duration where raw_on is True
            valid_mask = raw_on[t_idx:window_end]
            state[t_idx:window_end][valid_mask] = self.STATE_CODES["ON_TRANSIENT"]

        # 4. Label OFF_TRANSIENT (3)
        for t_idx in off_transitions:
            # OFF transient covers cycles leading up to or immediately following the shutdown
            window_start = max(0, t_idx - min(10, self.transient_window_cycles // 3))
            window_end = min(n, t_idx + min(20, self.transient_window_cycles * 2 // 3))
            
            # Apply OFF_TRANSIENT to the transition boundary
            state[window_start:window_end] = self.STATE_CODES["OFF_TRANSIENT"]

        df["state"] = state
        df["state_label"] = df["state"].map(self.STATE_LABELS)

        return df

    def get_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Returns count of each state label in the dataset."""
        if "state_label" not in df.columns:
            return {}
        counts = df["state_label"].value_counts().to_dict()
        return {code: counts.get(code, 0) for code in self.STATE_CODES.keys()}
