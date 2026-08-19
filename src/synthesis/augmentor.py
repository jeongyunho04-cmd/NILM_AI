"""
Data Augmentor for NILM Appliances
Applies physical time warping (duration stretching), power scaling,
switching phase jitter, and harmonic phase rotation on segmented appliance activations.
"""
from typing import Optional, Tuple
import numpy as np

from .segment_pool import ApplianceActivation


class DataAugmentor:
    """Applies domain-specific physical data augmentations on appliance activations."""

    def __init__(
        self,
        duration_scale_range: Tuple[float, float] = (0.5, 2.5),
        power_scale_std: float = 0.05,
        phase_jitter_max_deg: float = 4.0,
        switching_inrush_jitter: bool = True,
    ):
        self.duration_scale_range = duration_scale_range
        self.power_scale_std = power_scale_std
        self.phase_jitter_max_deg = phase_jitter_max_deg
        self.switching_inrush_jitter = switching_inrush_jitter

    def augment_activation(
        self,
        act: ApplianceActivation,
        target_duration_cycles: Optional[int] = None,
        duration_scale: Optional[float] = None,
        power_scale: Optional[float] = None,
        phase_jitter_deg: Optional[float] = None,
    ) -> ApplianceActivation:
        """Applies comprehensive physical augmentations to a single appliance activation.

        Returns:
            A new augmented ApplianceActivation instance.
        """
        orig_len = act.duration_cycles
        inrush_len = act.inrush_cycles

        # 1. Determine Duration & Time Warping
        if target_duration_cycles is not None:
            new_len = max(30, target_duration_cycles)
        elif duration_scale is not None:
            new_len = max(30, int(orig_len * duration_scale))
        else:
            scale = np.random.uniform(self.duration_scale_range[0], self.duration_scale_range[1])
            new_len = max(30, int(orig_len * scale))

        # Time warping: Keep inrush fixed, resample steady-state
        if new_len == orig_len:
            aug_c = act.net_harmonics_complex.copy()
            aug_pow = act.net_power_features.copy()
            aug_state = act.state_id.copy()
            aug_target_p = act.target_power_w.copy()
        else:
            aug_c, aug_pow, aug_state, aug_target_p = self._warp_time_series(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                inrush_len=inrush_len,
                target_len=new_len,
            )

        # 2. Power / Amplitude Scaling
        if power_scale is None:
            p_scale = float(np.clip(1.0 + np.random.normal(0, self.power_scale_std), 0.85, 1.15))
        else:
            p_scale = float(power_scale)

        aug_c *= p_scale
        aug_pow[:, 0] *= p_scale  # P
        aug_pow[:, 1] *= p_scale  # Q
        aug_pow[:, 2] *= p_scale  # S
        aug_target_p *= p_scale

        # 3. Switching Phase Jitter & Harmonic Phase Rotation
        if phase_jitter_deg is None:
            jitter_deg = float(np.random.uniform(-self.phase_jitter_max_deg, self.phase_jitter_max_deg))
        else:
            jitter_deg = float(phase_jitter_deg)

        if abs(jitter_deg) > 1e-4:
            jitter_rad = np.radians(jitter_deg)
            # Harmonic k rotates by k * jitter_rad
            h_count = aug_c.shape[1]
            harmonics_indices = np.arange(1, h_count + 1)
            rotation_vector = np.exp(1j * harmonics_indices * jitter_rad).astype(np.complex64)
            aug_c = aug_c * rotation_vector[np.newaxis, :]

        # 4. Intra-Cycle Inrush Contact Phase Modulation (first 2 cycles)
        if self.switching_inrush_jitter and len(aug_c) >= 3:
            contact_angle_rad = np.radians(np.random.uniform(-15.0, 15.0))
            aug_c[0] *= np.exp(1j * contact_angle_rad)
            aug_c[1] *= np.exp(1j * contact_angle_rad * 0.5)

        # 5. Reconstruct Real / Imaginary 2-Channel array
        aug_ri = np.zeros((new_len, aug_c.shape[1], 2), dtype=np.float32)
        aug_ri[:, :, 0] = np.real(aug_c)
        aug_ri[:, :, 1] = np.imag(aug_c)

        return ApplianceActivation(
            appliance_type=act.appliance_type,
            source_file=act.source_file,
            korean_name=act.korean_name,
            duration_cycles=new_len,
            duration_s=round(new_len / 60.0, 2),
            net_harmonics_ri=aug_ri,
            net_harmonics_complex=aug_c.astype(np.complex64),
            net_power_features=aug_pow.astype(np.float32),
            is_on=np.ones(new_len, dtype=np.int8),
            state_id=aug_state.astype(np.int16),
            target_power_w=aug_target_p.astype(np.float32),
            inrush_cycles=min(inrush_len, new_len // 3),
        )

    def _warp_time_series(
        self,
        c_series: np.ndarray,       # (L_orig, 15) complex64
        pow_series: np.ndarray,     # (L_orig, 6) float32
        state_series: np.ndarray,   # (L_orig,) int16
        target_p: np.ndarray,       # (L_orig,) float32
        inrush_len: int,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preserves inrush portion and resamples steady-state portion to match target_len."""
        orig_len = len(c_series)
        inrush_len = min(inrush_len, orig_len // 2, target_len // 2)

        if target_len <= inrush_len:
            # Ultra short window: just take prefix
            return (
                c_series[:target_len].copy(),
                pow_series[:target_len].copy(),
                state_series[:target_len].copy(),
                target_p[:target_len].copy(),
            )

        # Inrush prefix (preserved as-is)
        c_inrush = c_series[:inrush_len]
        pow_inrush = pow_series[:inrush_len]
        state_inrush = state_series[:inrush_len]
        target_p_inrush = target_p[:inrush_len]

        # Steady state portion to resample
        c_steady = c_series[inrush_len:]
        pow_steady = pow_series[inrush_len:]
        state_steady = state_series[inrush_len:]
        target_p_steady = target_p[inrush_len:]

        orig_steady_len = len(c_steady)
        target_steady_len = target_len - inrush_len

        x_orig = np.linspace(0, 1, orig_steady_len)
        x_target = np.linspace(0, 1, target_steady_len)

        # Resample complex harmonics: real & imag separately
        c_steady_re = np.zeros((target_steady_len, c_series.shape[1]), dtype=np.float32)
        c_steady_im = np.zeros((target_steady_len, c_series.shape[1]), dtype=np.float32)
        for k in range(c_series.shape[1]):
            c_steady_re[:, k] = np.interp(x_target, x_orig, np.real(c_steady[:, k]))
            c_steady_im[:, k] = np.interp(x_target, x_orig, np.imag(c_steady[:, k]))
        c_steady_warped = (c_steady_re + 1j * c_steady_im).astype(np.complex64)

        # Resample power features
        pow_steady_warped = np.zeros((target_steady_len, pow_series.shape[1]), dtype=np.float32)
        for i in range(pow_series.shape[1]):
            pow_steady_warped[:, i] = np.interp(x_target, x_orig, pow_steady[:, i])

        # Resample target power & state (nearest neighbor for discrete states)
        target_p_warped = np.interp(x_target, x_orig, target_p_steady).astype(np.float32)
        nearest_idx = np.clip(np.round(x_target * (orig_steady_len - 1)).astype(int), 0, orig_steady_len - 1)
        state_warped = state_steady[nearest_idx]

        # Concatenate inrush + warped steady
        out_c = np.concatenate([c_inrush, c_steady_warped], axis=0)
        out_pow = np.concatenate([pow_inrush, pow_steady_warped], axis=0)
        out_state = np.concatenate([state_inrush, state_warped], axis=0)
        out_target_p = np.concatenate([target_p_inrush, target_p_warped], axis=0)

        return out_c, out_pow, out_state, out_target_p
