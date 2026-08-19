"""
Appliance State Segment Pool for NILM Load Synthesis
Extracts isolated active activations (turn-on inrush, steady state, mode transitions, turn-off),
clean net currents, and exact standby/idle electrical profiles (phasors & power) for plugged-in simulation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.preprocessing.numpy_exporter import load_nilm_npz


@dataclass
class ApplianceActivation:
    """Represents a single continuous operation cycle of an appliance."""
    appliance_type: str
    source_file: str
    korean_name: str
    duration_cycles: int
    duration_s: float
    
    # Net current harmonics (idle baseline subtracted): (L, 15, 2) float32
    net_harmonics_ri: np.ndarray
    # Net complex harmonics: (L, 15) complex64
    net_harmonics_complex: np.ndarray
    # Net power features: [p_net, q_net, s_net, pf, vrms, thd_i] (L, 6) float32
    net_power_features: np.ndarray
    
    # Ground Truth labels
    is_on: np.ndarray          # (L,) int8 (all 1)
    state_id: np.ndarray       # (L,) int16
    target_power_w: np.ndarray # (L,) float32
    
    # Transient inrush split index (where inrush ends and steady-state begins)
    inrush_cycles: int


@dataclass
class StandbyProfile:
    """Represents the electrical standby profile of an appliance when plugged in but turned OFF."""
    appliance_type: str
    harmonics_ri: np.ndarray       # (15, 2) float32
    harmonics_complex: np.ndarray  # (15,) complex64
    power_w: float                 # Standby active power (W)
    reactive_var: float            # Standby reactive power (VAR)


class SegmentPool:
    """Manages segmented appliance activations, standby profiles, and background noise pools."""

    def __init__(self, npz_dir: Union[str, Path] = "processed_data/npz"):
        self.npz_dir = Path(npz_dir)
        self.appliance_activations: Dict[str, List[ApplianceActivation]] = {}
        self.noise_pool: Optional[np.ndarray] = None  # (Total_Noise_Samples, 15, 2)
        self.noise_complex: Optional[np.ndarray] = None
        self.noise_power: Optional[np.ndarray] = None
        self.standby_profiles: Dict[str, StandbyProfile] = {}

        self.load_all_npz_files()

    def load_all_npz_files(self):
        """Loads all .npz files and extracts activations, standby profiles, and noise reference."""
        npz_files = sorted(self.npz_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.npz_dir}. Run preprocessing first!")

        # Standby accumulation per appliance category
        raw_standby_ri: Dict[str, List[np.ndarray]] = {}
        raw_standby_c: Dict[str, List[np.ndarray]] = {}
        raw_standby_p: Dict[str, List[float]] = {}
        raw_standby_q: Dict[str, List[float]] = {}

        for f in npz_files:
            stem = f.stem
            data = load_nilm_npz(f)
            meta = data.get("metadata", {})
            appliance_type = meta.get("appliance_type", stem)
            korean_name = meta.get("korean_name", stem)

            # 1. Background Noise Reference
            if "noise" in stem:
                if self.noise_pool is None:
                    self.noise_pool = data["harmonics_ri"]
                    self.noise_complex = data["harmonics_complex"]
                    self.noise_power = data["power_features"]
                else:
                    self.noise_pool = np.concatenate([self.noise_pool, data["harmonics_ri"]], axis=0)
                    self.noise_complex = np.concatenate([self.noise_complex, data["harmonics_complex"]], axis=0)
                    self.noise_power = np.concatenate([self.noise_power, data["power_features"]], axis=0)
                continue

            # 2. Extract Standby Profile from idle samples (is_on == 0)
            is_on = data["is_on"]
            idle_mask = (is_on == 0)
            if np.any(idle_mask):
                mean_idle_ri = np.mean(data["harmonics_ri"][idle_mask], axis=0)  # (15, 2)
                mean_idle_c = np.mean(data["harmonics_complex"][idle_mask], axis=0)  # (15,)
                mean_idle_p = float(np.mean(data["power_features"][idle_mask, 0]))  # P_idle
                mean_idle_q = float(np.mean(data["power_features"][idle_mask, 1]))  # Q_idle
            else:
                mean_idle_ri = np.zeros((15, 2), dtype=np.float32)
                mean_idle_c = np.zeros(15, dtype=np.complex64)
                mean_idle_p = 0.0
                mean_idle_q = 0.0

            if appliance_type not in raw_standby_ri:
                raw_standby_ri[appliance_type] = []
                raw_standby_c[appliance_type] = []
                raw_standby_p[appliance_type] = []
                raw_standby_q[appliance_type] = []

            raw_standby_ri[appliance_type].append(mean_idle_ri)
            raw_standby_c[appliance_type].append(mean_idle_c)
            raw_standby_p[appliance_type].append(mean_idle_p)
            raw_standby_q[appliance_type].append(mean_idle_q)

            # 3. Segment Contiguous Active Activations (is_on == 1)
            on_indices = np.where(is_on == 1)[0]
            if len(on_indices) == 0:
                continue

            splits = np.where(np.diff(on_indices) > 1)[0] + 1
            blocks = np.split(on_indices, splits)

            if appliance_type not in self.appliance_activations:
                self.appliance_activations[appliance_type] = []

            for block in blocks:
                if len(block) < 30:
                    continue
                start_i = block[0]
                end_i = block[-1] + 1

                # Subtract idle baseline to isolate pure net appliance load
                raw_ri = data["harmonics_ri"][start_i:end_i]
                raw_c = data["harmonics_complex"][start_i:end_i]
                raw_pow = data["power_features"][start_i:end_i]

                net_ri = np.maximum(0.0, raw_ri - mean_idle_ri)
                net_c = raw_c - mean_idle_c
                net_pow = raw_pow.copy()
                net_pow[:, 0] = np.maximum(0.0, net_pow[:, 0] - mean_idle_p)

                inrush_len = min(len(block) // 3, 60)

                act = ApplianceActivation(
                    appliance_type=appliance_type,
                    source_file=stem,
                    korean_name=korean_name,
                    duration_cycles=len(block),
                    duration_s=round(len(block) / 60.0, 2),
                    net_harmonics_ri=net_ri.astype(np.float32),
                    net_harmonics_complex=net_c.astype(np.complex64),
                    net_power_features=net_pow.astype(np.float32),
                    is_on=data["is_on"][start_i:end_i],
                    state_id=data["state_id"][start_i:end_i],
                    target_power_w=data["target_power_w"][start_i:end_i],
                    inrush_cycles=inrush_len,
                )
                self.appliance_activations[appliance_type].append(act)

        # Build combined standby profiles per appliance type
        for app in raw_standby_ri:
            self.standby_profiles[app] = StandbyProfile(
                appliance_type=app,
                harmonics_ri=np.mean(raw_standby_ri[app], axis=0).astype(np.float32),
                harmonics_complex=np.mean(raw_standby_c[app], axis=0).astype(np.complex64),
                power_w=float(np.mean(raw_standby_p[app])),
                reactive_var=float(np.mean(raw_standby_q[app])),
            )

    def get_appliance_types(self) -> List[str]:
        """Returns list of available appliance categories."""
        return sorted(self.appliance_activations.keys())

    def sample_activation(self, appliance_type: str) -> ApplianceActivation:
        """Samples a random activation of the specified appliance type."""
        acts = self.appliance_activations.get(appliance_type, [])
        if not acts:
            raise ValueError(f"No activations available for appliance type '{appliance_type}'")
        idx = np.random.randint(0, len(acts))
        return acts[idx]

    def get_standby_profile(self, appliance_type: str) -> StandbyProfile:
        """Returns the standby electrical profile for a given appliance."""
        if appliance_type in self.standby_profiles:
            return self.standby_profiles[appliance_type]
        # Fallback zero standby
        return StandbyProfile(
            appliance_type=appliance_type,
            harmonics_ri=np.zeros((15, 2), dtype=np.float32),
            harmonics_complex=np.zeros(15, dtype=np.complex64),
            power_w=0.0,
            reactive_var=0.0,
        )

    def sample_noise_slice(self, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples a contiguous slice of background noise."""
        if self.noise_pool is None or len(self.noise_pool) == 0:
            return (
                np.zeros((length, 15, 2), dtype=np.float32),
                np.zeros((length, 15), dtype=np.complex64),
                np.zeros((length, 6), dtype=np.float32),
            )
        total_n = len(self.noise_pool)
        if length >= total_n:
            reps = int(np.ceil(length / total_n))
            full_ri = np.tile(self.noise_pool, (reps, 1, 1))[:length]
            full_c = np.tile(self.noise_complex, (reps, 1))[:length]
            full_p = np.tile(self.noise_power, (reps, 1))[:length]
            return full_ri, full_c, full_p

        start_idx = np.random.randint(0, total_n - length)
        end_idx = start_idx + length
        return (
            self.noise_pool[start_idx:end_idx],
            self.noise_complex[start_idx:end_idx],
            self.noise_power[start_idx:end_idx],
        )
