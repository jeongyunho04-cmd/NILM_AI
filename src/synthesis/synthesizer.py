"""
Multi-Appliance Load Signal Synthesizer for NILM AI
Combines augmented appliance activations via phasor vector addition,
supports stochastic plugged-in standby power vs unplugged zero-load states,
injects single baseline background noise, and simulates dynamic grid voltage sag.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .augmentor import DataAugmentor
from .grid_simulator import GridSimulator
from .segment_pool import ApplianceActivation, SegmentPool, StandbyProfile


@dataclass
class ApplianceSchedule:
    """Defines the scheduling of a single appliance in a synthetic timeline."""
    appliance_type: str
    start_cycle: int
    duration_cycles: Optional[int] = None
    power_scale: Optional[float] = None
    phase_jitter_deg: Optional[float] = None


@dataclass
class SyntheticLoadSample:
    """Complete synthesized aggregate electrical signal and ground truth targets."""
    duration_cycles: int
    duration_s: float
    appliance_types: List[str]
    active_appliances: List[str]
    plugged_in_appliances: List[str]
    
    # Aggregate Input Features (for AI Model Input)
    harmonics_ri: np.ndarray          # (N, 15, 2) float32 [Real, Imag]
    harmonics_complex: np.ndarray     # (N, 15) complex64
    power_features: np.ndarray        # (N, 6) float32 [P, Q, S, PF, V_bus, THD_i]
    v_bus: np.ndarray                 # (N,) float32 terminal voltage
    t_rel_s: np.ndarray               # (N,) float32
    
    # Ground Truth Targets per Appliance (for AI Model Training)
    gt_is_on: Dict[str, np.ndarray]          # appliance -> (N,) int8
    gt_state_id: Dict[str, np.ndarray]       # appliance -> (N,) int16
    gt_target_power_w: Dict[str, np.ndarray] # appliance -> (N,) float32
    gt_harmonics_ri: Dict[str, np.ndarray]   # appliance -> (N, 15, 2) float32
    
    # Summary metadata
    metadata: Dict


class LoadSynthesizer:
    """Synthesizes realistic composite household electrical loads."""

    def __init__(
        self,
        segment_pool: SegmentPool,
        grid_simulator: Optional[GridSimulator] = None,
        augmentor: Optional[DataAugmentor] = None,
    ):
        self.pool = segment_pool
        self.grid_sim = grid_simulator or GridSimulator()
        self.augmentor = augmentor or DataAugmentor()
        self.known_appliances = self.pool.get_appliance_types()

    def synthesize_scenario(
        self,
        total_duration_cycles: int,
        schedules: List[ApplianceSchedule],
        plugged_in_appliances: Optional[Dict[str, bool]] = None,
        default_plugged_prob: float = 0.7,
        include_noise: bool = True,
        simulate_voltage_drop: bool = True,
    ) -> SyntheticLoadSample:
        """Synthesizes a composite aggregate load timeline according to schedules.

        Args:
            total_duration_cycles: Total length in cycles (e.g. 600 = 10s, 3600 = 60s)
            schedules: List of ApplianceSchedule instances to place
            plugged_in_appliances: Dict mapping appliance -> bool (whether plugged into wall socket)
            default_plugged_prob: Probability that an appliance is plugged in if not specified
            include_noise: Whether to inject real background noise
            simulate_voltage_drop: Whether to simulate grid impedance and voltage sag

        Returns:
            SyntheticLoadSample containing aggregate inputs and individual GTs.
        """
        N = total_duration_cycles
        num_harmonics = 15

        # 1. Determine Plugged-In Status per Appliance
        # If an appliance is scheduled to turn on, it must be plugged in at least during activation
        is_plugged: Dict[str, bool] = {}
        for app in self.known_appliances:
            if plugged_in_appliances is not None and app in plugged_in_appliances:
                is_plugged[app] = plugged_in_appliances[app]
            else:
                # Appliances with continuous standby electronics (AC, TV/Projector, Oven, MiniPC)
                # have higher plug-in probability than portable tools (dryer, kettle)
                if app in ["air_conditioner", "oven", "beam_projector", "minipc"]:
                    prob = 0.85
                elif app in ["electiric_kettle", "hair_dryer"]:
                    prob = 0.50
                else:
                    prob = default_plugged_prob
                is_plugged[app] = bool(np.random.rand() < prob)

        # 2. Initialize per-appliance ground truth arrays and complex layers
        gt_is_on = {app: np.zeros(N, dtype=np.int8) for app in self.known_appliances}
        gt_state_id = {app: np.zeros(N, dtype=np.int16) for app in self.known_appliances}
        gt_target_p = {app: np.zeros(N, dtype=np.float32) for app in self.known_appliances}
        gt_harm_ri = {app: np.zeros((N, num_harmonics, 2), dtype=np.float32) for app in self.known_appliances}
        app_complex_layers = {app: np.zeros((N, num_harmonics), dtype=np.complex64) for app in self.known_appliances}
        standby_p_layers = {app: np.zeros(N, dtype=np.float32) for app in self.known_appliances}

        # Initialize Standby Current Phasors for plugged-in appliances
        for app in self.known_appliances:
            if is_plugged[app]:
                st_profile: StandbyProfile = self.pool.get_standby_profile(app)
                # Fill timeline with standby current
                app_complex_layers[app][:] = st_profile.harmonics_complex
                standby_p_layers[app][:] = st_profile.power_w

        active_set = set()

        # 3. Place and augment scheduled active activations
        for sched in schedules:
            app_type = sched.appliance_type
            if app_type not in self.known_appliances:
                continue

            t_start = sched.start_cycle
            if t_start >= N:
                continue

            raw_act = self.pool.sample_activation(app_type)
            aug_act = self.augmentor.augment_activation(
                raw_act,
                target_duration_cycles=sched.duration_cycles,
                power_scale=sched.power_scale,
                phase_jitter_deg=sched.phase_jitter_deg,
            )

            act_len = aug_act.duration_cycles
            t_end = min(N, t_start + act_len)
            place_len = t_end - t_start

            if place_len <= 0:
                continue

            active_set.add(app_type)

            act_c_slice = aug_act.net_harmonics_complex[:place_len]
            act_ri_slice = aug_act.net_harmonics_ri[:place_len]
            act_p_slice = aug_act.target_power_w[:place_len]
            act_state_slice = aug_act.state_id[:place_len]

            # Net active current is layered on top of standby (or 0 if unplugged)
            app_complex_layers[app_type][t_start:t_end] += act_c_slice
            gt_harm_ri[app_type][t_start:t_end] += act_ri_slice
            gt_target_p[app_type][t_start:t_end] += act_p_slice
            gt_is_on[app_type][t_start:t_end] = 1
            gt_state_id[app_type][t_start:t_end] = np.maximum(gt_state_id[app_type][t_start:t_end], act_state_slice)

        # 4. Sum all appliance layers (Phasor superposition)
        total_complex = np.zeros((N, num_harmonics), dtype=np.complex64)
        for app in self.known_appliances:
            total_complex += app_complex_layers[app]

        # 5. Inject Single Background Noise Slice
        noise_p_base = 0.0
        if include_noise:
            noise_ri, noise_c, noise_pow = self.pool.sample_noise_slice(N)
            total_complex += noise_c
            noise_p_base = np.mean(noise_pow[:, 0])

        # 6. Grid Impedance & Voltage Drop Simulation (Z_grid Feedback)
        if simulate_voltage_drop:
            v_bus, kappa = self.grid_sim.compute_voltage_drop(total_complex)
            
            coupled_total_complex = np.zeros((N, num_harmonics), dtype=np.complex64)
            if include_noise:
                coupled_total_complex += noise_c

            for app in self.known_appliances:
                if is_plugged[app] or (app in active_set):
                    coupled_c = self.grid_sim.apply_cross_appliance_coupling(
                        app, app_complex_layers[app], kappa
                    )
                    coupled_total_complex += coupled_c
                    # Update ground truth active portion
                    if app in active_set:
                        gt_harm_ri[app][:, :, 0] = np.real(coupled_c)
                        gt_harm_ri[app][:, :, 1] = np.imag(coupled_c)
                        if app in ["electiric_kettle", "hotplate", "hair_dryer", "oven"]:
                            gt_target_p[app] *= (kappa**2)
                        elif app in ["minipc", "laptop_charger", "beam_projector"]:
                            pass
                        else:
                            gt_target_p[app] *= (kappa**0.7)

            total_complex = coupled_total_complex
        else:
            v_bus = np.full(N, 220.0, dtype=np.float32)

        # 7. Compute Aggregate Electrical Output Features
        harmonics_ri = np.zeros((N, num_harmonics, 2), dtype=np.float32)
        harmonics_ri[:, :, 0] = np.real(total_complex)
        harmonics_ri[:, :, 1] = np.imag(total_complex)

        mag_sq = np.real(total_complex)**2 + np.imag(total_complex)**2
        irms_total = np.sqrt(np.sum(mag_sq, axis=1)).astype(np.float32)
        i1_mag = np.sqrt(mag_sq[:, 0]) + 1e-6

        # Active Power P = sum(P_ground_truth) + sum(P_standby) + P_noise
        p_total = np.zeros(N, dtype=np.float32)
        for app in self.known_appliances:
            p_total += gt_target_p[app]
            if is_plugged[app]:
                p_total += standby_p_layers[app]
        p_total += float(noise_p_base)

        s_total = (v_bus * irms_total).astype(np.float32)
        q_sq = np.maximum(0.0, s_total**2 - p_total**2)
        q_sign = np.where(np.imag(total_complex[:, 0]) < 0, -1.0, 1.0)
        q_total = (np.sqrt(q_sq) * q_sign).astype(np.float32)
        pf_total = np.clip(p_total / (s_total + 1e-6), 0.0, 1.0).astype(np.float32)

        higher_h_sq = np.sum(mag_sq[:, 1:], axis=1)
        thd_i_total = (np.sqrt(higher_h_sq) / i1_mag).astype(np.float32)

        power_features = np.stack(
            [p_total, q_total, s_total, pf_total, v_bus, thd_i_total], axis=1
        ).astype(np.float32)

        t_rel_s = (np.arange(N) / 60.0).astype(np.float32)
        plugged_list = [app for app, p in is_plugged.items() if p]

        meta = {
            "duration_cycles": N,
            "duration_s": round(N / 60.0, 2),
            "num_schedules": len(schedules),
            "active_appliances": sorted(list(active_set)),
            "plugged_in_appliances": sorted(plugged_list),
            "mean_p_w": round(float(np.mean(p_total)), 2),
            "max_p_w": round(float(np.max(p_total)), 2),
            "min_v_bus": round(float(np.min(v_bus)), 1),
            "max_v_drop": round(float(220.0 - np.min(v_bus)), 2),
        }

        return SyntheticLoadSample(
            duration_cycles=N,
            duration_s=round(N / 60.0, 2),
            appliance_types=self.known_appliances,
            active_appliances=sorted(list(active_set)),
            plugged_in_appliances=sorted(plugged_list),
            harmonics_ri=harmonics_ri,
            harmonics_complex=total_complex,
            power_features=power_features,
            v_bus=v_bus,
            t_rel_s=t_rel_s,
            gt_is_on=gt_is_on,
            gt_state_id=gt_state_id,
            gt_target_power_w=gt_target_p,
            gt_harmonics_ri=gt_harm_ri,
            metadata=meta,
        )

    def synthesize_random_window(
        self,
        window_size_cycles: int = 600,
        max_concurrent_appliances: int = 3,
        plugged_prob: float = 0.6,
    ) -> SyntheticLoadSample:
        """Synthesizes a fast random window with stochastic plug-in states and 0~max active appliances."""
        n_avail = len(self.known_appliances)
        k_active = np.random.randint(0, min(max_concurrent_appliances + 1, n_avail + 1))
        
        # Randomize plug-in states
        plugged_dict = {
            app: bool(np.random.rand() < plugged_prob) for app in self.known_appliances
        }

        schedules = []
        if k_active > 0:
            selected_apps = np.random.choice(self.known_appliances, size=k_active, replace=False)
            for app in selected_apps:
                plugged_dict[app] = True  # Must be plugged in if active
                start_c = int(np.random.randint(-window_size_cycles // 2, window_size_cycles // 2))
                dur_c = int(np.random.randint(window_size_cycles // 2, window_size_cycles * 3))
                schedules.append(
                    ApplianceSchedule(
                        appliance_type=app,
                        start_cycle=max(0, start_c),
                        duration_cycles=dur_c,
                    )
                )

        return self.synthesize_scenario(
            total_duration_cycles=window_size_cycles,
            schedules=schedules,
            plugged_in_appliances=plugged_dict,
            include_noise=True,
            simulate_voltage_drop=True,
        )
