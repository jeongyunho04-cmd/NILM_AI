"""
가전 활성화 물리 증강기 (Data Augmentor)
=========================================
세그먼트로 잘라 둔 가전 동작 구간에 시간 신축, 전력 스케일링, 투입 위상 지터,
고조파 위상 회전을 적용해 학습 데이터의 다양성을 넓힌다.

[부하 종류에 따라 시간 신축 방식이 다르다]
1. 일반 부하 (리샘플링):
   에어컨 냉방, 선풍기 회전처럼 사람이 켜 둔 시간만큼 지속되는 동작은
   정상상태를 늘이거나 줄여도 물리적으로 자연스럽다.

2. 주기 부하 (잘라내기 / 이어붙이기):
   오븐과 핫플레이트는 서모스탯/릴레이가 정해진 주기로 통전을 끊는다.
   핫플레이트는 약 1초 ON / 1초 OFF, 오븐은 10~25초 히터 펄스를 반복한다.
   이 파형을 2.2배로 리샘플링하면 10초 펄스가 22초가 되어 실제로는 존재할 수 없는
   기기가 만들어진다. 주기 부하는 늘일 때 원본을 순환 이어붙이고 줄일 때 잘라내어
   주기 자체를 보존한다.

   부수 효과로 오븐처럼 활성화 구간이 2개뿐인 기기도, 32.9분짜리 원본에서
   매번 다른 위상을 잘라 쓰게 되어 실질적인 다양성이 크게 늘어난다.
"""
from typing import Optional, Tuple
import numpy as np

from .segment_pool import ApplianceActivation

MIN_AUGMENTED_CYCLES = 30


class DataAugmentor:
    """가전 활성화 구간에 도메인 특화 물리 증강을 적용한다."""

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
        """활성화 구간 1개에 종합적인 물리 증강을 적용한 새 인스턴스를 반환한다."""
        orig_len = act.duration_cycles
        inrush_len = act.inrush_cycles

        # 1. 목표 길이 결정
        if target_duration_cycles is not None:
            new_len = max(MIN_AUGMENTED_CYCLES, int(target_duration_cycles))
        elif duration_scale is not None:
            new_len = max(MIN_AUGMENTED_CYCLES, int(orig_len * duration_scale))
        else:
            scale = np.random.uniform(*self.duration_scale_range)
            new_len = max(MIN_AUGMENTED_CYCLES, int(orig_len * scale))

        # 2. 시간 신축 - 부하 종류에 따라 방식이 갈린다
        if new_len == orig_len:
            aug_c = act.net_harmonics_complex.copy()
            aug_pow = act.net_power_features.copy()
            aug_state = act.state_id.copy()
            aug_target_p = act.target_power_w.copy()
        elif act.periodic_duty:
            # 서모스탯 주기를 보존해야 한다 - 늘이면 순환 이어붙이기, 줄이면 잘라내기
            aug_c, aug_pow, aug_state, aug_target_p = self._tile_or_crop(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                target_len=new_len,
            )
        else:
            aug_c, aug_pow, aug_state, aug_target_p = self._warp_time_series(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                inrush_len=inrush_len,
                target_len=new_len,
            )

        # 3. 전력 / 진폭 스케일링
        if power_scale is None:
            p_scale = float(np.clip(1.0 + np.random.normal(0, self.power_scale_std), 0.85, 1.15))
        else:
            p_scale = float(power_scale)

        aug_c = aug_c * p_scale
        aug_pow = aug_pow.copy()
        aug_pow[:, 0] *= p_scale  # P
        aug_pow[:, 1] *= p_scale  # Q
        aug_pow[:, 2] *= p_scale  # S
        aug_target_p = aug_target_p * p_scale

        # 4. 투입 위상 지터 및 고조파 위상 회전
        #    전체 파형을 시간축에서 조금 밀면 k차 고조파는 k*theta 만큼 회전한다.
        if phase_jitter_deg is None:
            jitter_deg = float(np.random.uniform(-self.phase_jitter_max_deg, self.phase_jitter_max_deg))
        else:
            jitter_deg = float(phase_jitter_deg)

        if abs(jitter_deg) > 1e-4:
            jitter_rad = np.radians(jitter_deg)
            harmonics_indices = np.arange(1, aug_c.shape[1] + 1)
            rotation_vector = np.exp(1j * harmonics_indices * jitter_rad).astype(np.complex64)
            aug_c = aug_c * rotation_vector[np.newaxis, :]

        # 5. 스위치 접점이 닫히는 순간의 위상 (돌입 전류 첫 2주기)
        #    주기 부하는 잘라낸 위치가 펄스 중간일 수 있어 이 처리를 하지 않는다.
        if self.switching_inrush_jitter and not act.periodic_duty and len(aug_c) >= 3:
            contact_angle_rad = np.radians(np.random.uniform(-15.0, 15.0))
            aug_c[0] *= np.exp(1j * contact_angle_rad)
            aug_c[1] *= np.exp(1j * contact_angle_rad * 0.5)

        # 6. Real / Imag 2채널 배열 재구성
        actual_len = len(aug_c)
        aug_ri = np.zeros((actual_len, aug_c.shape[1], 2), dtype=np.float32)
        aug_ri[:, :, 0] = np.real(aug_c)
        aug_ri[:, :, 1] = np.imag(aug_c)

        return ApplianceActivation(
            appliance_type=act.appliance_type,
            source_file=act.source_file,
            korean_name=act.korean_name,
            duration_cycles=actual_len,
            duration_s=round(actual_len / 60.0, 2),
            net_harmonics_ri=aug_ri,
            net_harmonics_complex=aug_c.astype(np.complex64),
            net_power_features=aug_pow.astype(np.float32),
            is_on=np.ones(actual_len, dtype=np.int8),
            state_id=aug_state.astype(np.int16),
            target_power_w=aug_target_p.astype(np.float32),
            inrush_cycles=min(inrush_len, max(1, actual_len // 3)),
            v_ref_v=act.v_ref_v,
            periodic_duty=act.periodic_duty,
        )

    # ── 주기 부하: 잘라내기 / 순환 이어붙이기 ───────────────────────────────
    def _tile_or_crop(
        self,
        c_series: np.ndarray,
        pow_series: np.ndarray,
        state_series: np.ndarray,
        target_p: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """서모스탯 주기를 그대로 둔 채 길이만 맞춘다.

        시작 위치를 매번 무작위로 잡아 순환 인덱싱하므로, 같은 원본에서도
        서모스탯 주기의 다른 위상이 잘려 나온다. 활성화 구간이 2개뿐인 오븐 같은
        기기에서 실질적인 다양성을 크게 늘려 주는 효과가 있다.
        """
        orig_len = len(c_series)
        start = int(np.random.randint(0, orig_len))
        idx = (np.arange(target_len) + start) % orig_len
        return (
            c_series[idx].copy(),
            pow_series[idx].copy(),
            state_series[idx].copy(),
            target_p[idx].copy(),
        )

    # ── 일반 부하: 돌입 구간 보존 리샘플링 ──────────────────────────────────
    def _warp_time_series(
        self,
        c_series: np.ndarray,       # (L_orig, 15) complex64
        pow_series: np.ndarray,     # (L_orig, 6) float32
        state_series: np.ndarray,   # (L_orig,) int16
        target_p: np.ndarray,       # (L_orig,) float32
        inrush_len: int,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """돌입 전류 구간은 그대로 두고 정상상태만 리샘플링해 길이를 맞춘다."""
        orig_len = len(c_series)
        inrush_len = max(0, min(inrush_len, orig_len // 2, target_len // 2))

        if target_len <= inrush_len or orig_len - inrush_len < 2:
            # 아주 짧은 윈도우이거나 정상상태가 없다 - 앞부분을 그대로 쓴다
            take = min(target_len, orig_len)
            return (
                c_series[:take].copy(),
                pow_series[:take].copy(),
                state_series[:take].copy(),
                target_p[:take].copy(),
            )

        c_inrush = c_series[:inrush_len]
        pow_inrush = pow_series[:inrush_len]
        state_inrush = state_series[:inrush_len]
        target_p_inrush = target_p[:inrush_len]

        c_steady = c_series[inrush_len:]
        pow_steady = pow_series[inrush_len:]
        state_steady = state_series[inrush_len:]
        target_p_steady = target_p[inrush_len:]

        orig_steady_len = len(c_steady)
        target_steady_len = target_len - inrush_len

        x_orig = np.linspace(0, 1, orig_steady_len)
        x_target = np.linspace(0, 1, target_steady_len)

        # 복소 고조파는 실수부/허수부를 따로 보간한다
        c_steady_re = np.empty((target_steady_len, c_series.shape[1]), dtype=np.float32)
        c_steady_im = np.empty((target_steady_len, c_series.shape[1]), dtype=np.float32)
        for k in range(c_series.shape[1]):
            c_steady_re[:, k] = np.interp(x_target, x_orig, np.real(c_steady[:, k]))
            c_steady_im[:, k] = np.interp(x_target, x_orig, np.imag(c_steady[:, k]))
        c_steady_warped = (c_steady_re + 1j * c_steady_im).astype(np.complex64)

        pow_steady_warped = np.empty((target_steady_len, pow_series.shape[1]), dtype=np.float32)
        for i in range(pow_series.shape[1]):
            pow_steady_warped[:, i] = np.interp(x_target, x_orig, pow_steady[:, i])

        target_p_warped = np.interp(x_target, x_orig, target_p_steady).astype(np.float32)
        # 상태 ID 는 이산값이므로 최근접 이웃으로 옮긴다
        nearest_idx = np.clip(np.round(x_target * (orig_steady_len - 1)).astype(int), 0, orig_steady_len - 1)
        state_warped = state_steady[nearest_idx]

        return (
            np.concatenate([c_inrush, c_steady_warped], axis=0),
            np.concatenate([pow_inrush, pow_steady_warped], axis=0),
            np.concatenate([state_inrush, state_warped], axis=0),
            np.concatenate([target_p_inrush, target_p_warped], axis=0),
        )
