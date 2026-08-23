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
        max_stretch: float = 3.0,
        duty_on_scale_range: Tuple[float, float] = (0.5, 2.0),
        duty_off_scale_range: Tuple[float, float] = (0.5, 2.0),
        randomize_duty: bool = True,
        load_stratified: bool = True,
        load_strata_candidates: int = 16,
    ):
        self.duration_scale_range = duration_scale_range
        self.power_scale_std = power_scale_std
        self.phase_jitter_max_deg = phase_jitter_max_deg
        self.switching_inrush_jitter = switching_inrush_jitter
        # 주기 부하의 통전/휴지 **길이** 를 흔든다 (`_retime_duty` 주석).
        self.duty_on_scale_range = duty_on_scale_range
        self.duty_off_scale_range = duty_off_scale_range
        self.randomize_duty = bool(randomize_duty)
        # 긴 활성화에서 창을 자를 때 **자르는 지점을 전력으로 계층화**한다
        # (`_crop_window` 주석). 12.34.6 의 미니PC 문제 대응이다.
        self.load_stratified = bool(load_stratified)
        self.load_strata_candidates = max(2, int(load_strata_candidates))
        # 원본보다 이 배율 이상으로는 늘이지 않는다. 0.5초짜리 동작을 30초로 늘이면
        # 60배 느려진 파형이 되어 실제로는 존재할 수 없는 기기가 만들어진다.
        # 한도를 넘으면 늘이는 대신 그 길이에서 동작이 끝난 것으로 처리한다.
        self.max_stretch = max(1.0, float(max_stretch))

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

        # 지나친 확대는 물리적으로 말이 안 되므로 그 지점에서 동작이 끝난 것으로 본다.
        if new_len > orig_len * self.max_stretch:
            new_len = max(MIN_AUGMENTED_CYCLES, int(orig_len * self.max_stretch))

        # 2. 시간 신축 - 부하 종류와 방향에 따라 방식이 갈린다
        includes_onset = True
        if new_len == orig_len:
            aug_c = act.net_harmonics_complex.copy()
            aug_pow = act.net_power_features.copy()
            aug_state = act.state_id.copy()
            aug_target_p = act.target_power_w.copy()
            aug_on = act.is_on.copy()
        elif act.periodic_duty:
            # 파형은 그대로 두고 통전/휴지 **길이** 만 흔든 뒤, 길이를 맞춘다.
            src = self._retime_duty(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                act.is_on,
            )
            aug_c, aug_pow, aug_state, aug_target_p, aug_on = self._tile_or_crop(
                *src, target_len=new_len,
            )
            includes_onset = False
        elif new_len < orig_len:
            # 원본이 목표보다 길다 - 압축하지 않고 잘라 쓴다.
            #
            # 미니PC 는 한 번에 2500초를 연속으로 돌았다. 이것을 10초 윈도우에
            # 맞추려고 250배로 압축하면, 몇 분에 걸쳐 일어나는 IDLE->ACTIVE 전이가
            # 수 밀리초 만에 끝나는 파형이 된다. 실제로 그 기기는 계속 돌고 있었으므로
            # 10초짜리 창으로 보면 '진짜 10초'가 보여야 한다.
            aug_c, aug_pow, aug_state, aug_target_p, aug_on, includes_onset = self._crop_window(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                act.is_on,
                target_len=new_len,
            )
        else:
            aug_c, aug_pow, aug_state, aug_target_p, aug_on = self._warp_time_series(
                act.net_harmonics_complex,
                act.net_power_features,
                act.state_id,
                act.target_power_w,
                act.is_on,
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
        #    파형 중간을 잘라 온 경우에는 첫 샘플이 투입 순간이 아니므로 적용하지 않는다.
        if self.switching_inrush_jitter and includes_onset and len(aug_c) >= 3:
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
            # 통전이 끊긴 구간(서모스탯/릴레이 OFF)을 1 로 덮어쓰면 안 된다.
            # 그렇게 하면 주기 부하가 연속 발열로 둔갑한다.
            is_on=np.asarray(aug_on, dtype=np.int8),
            state_id=aug_state.astype(np.int16),
            target_power_w=aug_target_p.astype(np.float32),
            inrush_cycles=min(inrush_len, max(1, actual_len // 3)),
            v_ref_v=act.v_ref_v,
            periodic_duty=act.periodic_duty,
        )

    # ── 주기 부하: 잘라내기 / 순환 이어붙이기 ───────────────────────────────
    def _retime_duty(
        self,
        c_series: np.ndarray,
        pow_series: np.ndarray,
        state_series: np.ndarray,
        target_p: np.ndarray,
        on_series: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """통전/휴지 **구간 길이**를 흔든다. 파형은 리샘플링하지 않는다.

        [왜 이것이 필요한가 - 설계 문서 12.16절]
        학습 풀의 핫플 활성화는 3개, 포트는 13개(녹화 1개)뿐이고 합성기는 그것을
        그대로 재생한다. 그래서 합성 창에는 풀 파형의 글자 그대로의 복사본이 들어가고,
        모델은 *"이 파형이 있는가"* 만 맞추면 된다 - **듀티를 볼 이유가 없다.**
        실제로 60초 내내 연속 통전한 창에서도 검출이 1.000 이었다 (12.16.2절 ①).
        그런데 실측에서는 그 복사본이 없으므로 듀티라도 써야 하는데, 학습에서 쓸 일이
        없었으니 안 배웠다.

        [왜 리샘플링이 아니라 구간 길이인가]
        이 모듈 서두가 주기 부하를 리샘플링하지 말라고 못박은 이유는 *파형* 왜곡이다 -
        10초 히터 펄스를 2.2배로 늘이면 22초짜리, 실재하지 않는 기기가 된다.
        **여기서는 파형을 늘이지 않는다.** 통전 구간 안에서 사이클을 순환 반복해
        릴레이가 더 오래/짧게 닫혀 있게만 한다. 서모스탯 부하에서 통전 길이와 주기는
        **기기의 성질이 아니라 설정·주위 온도·부하의 함수**다. 실측이 그것을 보여 준다:

            핫플 통전율   51.3% / 57.8% / 35.8%   (활성화 3개, 12.13.1절)
            오븐 듀티     22 / 28 / 39 / 46 / 81%  (활성화 5개, 12.11절)

        같은 기기가 실제로 이만큼 흔들린다. 기본 배율 (0.5, 2.0) 은 그 범위를 덮는다.

        [한계]
        통전 구간 안에서 사이클을 반복하므로 승온에 따른 저항 드리프트가 평평해진다.
        저항 부하의 정상상태에서는 작고, `_tile_or_crop` 이 구간 사이에서 이미 하고
        있는 가정과 같은 종류다.
        """
        state = np.asarray(state_series)
        tp = np.asarray(target_p, dtype=np.float64)
        if not self.randomize_duty or state.size < 2:
            return c_series, pow_series, state_series, target_p, on_series

        # **구간은 `state_id` 로 나눈다.** `is_on` 으로 나누면 오븐이 빠진다 -
        # 오븐의 `is_on` 은 활성화 단위라 내내 1 이고, 히터 펄스는 state 에만 있다
        # (state 2=히터 / 1=팬·조명, 듀티 0.206~0.808 로 12.11절의 22~81% 와 맞는다).
        # 핫플은 state 0/1 이 릴레이와 같이 간다.
        edges = np.flatnonzero(np.diff(state)) + 1
        if edges.size == 0:
            return c_series, pow_series, state_series, target_p, on_series
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [state.size]))

        # 어느 구간이 "통전" 인지는 상태 번호가 아니라 **전력** 으로 정한다.
        # 기기마다 상태 번호의 의미가 다르기 때문이다.
        hot = tp > 0.5 * np.percentile(tp, 99)
        on_scale = float(np.random.uniform(*self.duty_on_scale_range))
        off_scale = float(np.random.uniform(*self.duty_off_scale_range))

        idx = []
        for a, b in zip(starts, stops):
            run = b - a
            scale = on_scale if hot[a:b].mean() >= 0.5 else off_scale
            new_run = max(1, int(round(run * scale)))
            # 구간 **안에서** 순환한다. 구간을 넘어가지 않으므로 통전/휴지가 안 섞인다.
            idx.append(a + (np.arange(new_run) % run))
        idx = np.concatenate(idx)
        return (c_series[idx].copy(), pow_series[idx].copy(), state_series[idx].copy(),
                target_p[idx].copy(), np.asarray(on_series)[idx].copy())

    def _tile_or_crop(
        self,
        c_series: np.ndarray,
        pow_series: np.ndarray,
        state_series: np.ndarray,
        target_p: np.ndarray,
        on_series: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """서모스탯 주기를 그대로 둔 채 길이만 맞춘다.

        시작 위치를 매번 무작위로 잡아 순환 인덱싱하므로, 같은 원본에서도
        서모스탯 주기의 다른 위상이 잘려 나온다. 활성화 구간이 2개뿐인 오븐 같은
        기기에서 실질적인 다양성을 크게 늘려 주는 효과가 있다.

        전제: 원본 활성화가 통전 구간과 휴지 구간을 **모두** 담고 있어야 한다.
        통전 펄스만 잘라 온 배열을 여기에 넣으면 순환 이어붙이기가 휴지 구간을
        만들어 내지 못해 100% 통전 파형이 나온다. 세션 병합은
        SegmentPool._merge_duty_blocks 가 담당한다.
        """
        orig_len = len(c_series)
        start = int(np.random.randint(0, orig_len))
        idx = (np.arange(target_len) + start) % orig_len
        return (
            c_series[idx].copy(),
            pow_series[idx].copy(),
            state_series[idx].copy(),
            target_p[idx].copy(),
            on_series[idx].copy(),
        )

    # ── 원본이 목표보다 길 때: 잘라내기 ─────────────────────────────────────
    def _crop_window(
        self,
        c_series: np.ndarray,
        pow_series: np.ndarray,
        state_series: np.ndarray,
        target_p: np.ndarray,
        on_series: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """긴 동작 구간에서 목표 길이만큼 연속 구간을 잘라낸다 (시간 압축 없음).

        어디를 자르느냐에 따라 보이는 과도현상이 달라지므로 세 위치를 섞는다.
        한쪽만 쓰면 그 기기의 특정 전이만 학습하게 된다.

        [정상 운전 구간은 전력으로 계층화해 뽑는다] (12.34.6)
        시간 균등으로 뽑으면 **그 기기가 그 상태로 오래 있었던 만큼** 뽑힌다.
        미니PC 가 그 예다 - 33.9분짜리 활성화 안에서 CPU 부하가 걸린 구간이
        6.7% 뿐이라, 풀 전체에서 20W 이상인 사이클이 13.9% 밖에 안 된다.
        실측 미니PC 단독은 30.3W 인데 합성 60초 창의 최대가 27.9W 였다.

        노출 현실성과 동작 범위 커버리지는 다른 요구다. 학습에는 후자가 필요하다.
        후보를 여러 개 뽑아 각자의 통전 전력을 재고, **전력 범위에서 균등하게**
        목표를 정해 가장 가까운 후보를 고른다. 전력이 평평한 활성화(프로젝터)에서는
        저절로 무작위 추출과 같아진다.

        듀티 부하(오븐·핫플)는 여기 오지 않는다 - `augment_activation` 이 앞에서
        `_tile_or_crop` 으로 보낸다. 그쪽은 통전/휴지 비율이 물리라 건드리면 안 된다.

        Returns:
            잘라낸 배열 4개와, 그 구간이 '켜지는 순간'을 포함하는지 여부
        """
        orig_len = len(c_series)
        span = orig_len - target_len
        r = np.random.rand()
        if r < 0.25:
            start = 0                                   # 켜지는 순간(돌입 전류) 포함
        elif r < 0.50:
            start = span                                # 꺼지는 순간 포함
        else:
            start = self._stratified_start(target_p, on_series, target_len, span)

        sl = slice(start, start + target_len)
        return (
            c_series[sl].copy(),
            pow_series[sl].copy(),
            state_series[sl].copy(),
            target_p[sl].copy(),
            on_series[sl].copy(),
            start == 0,
        )

    def _stratified_start(
        self, target_p: np.ndarray, on_series: np.ndarray, target_len: int, span: int
    ) -> int:
        """정상 운전 구간의 자를 지점을 전력 계층으로 고른다 (`_crop_window` 주석).

        후보마다 창 전체의 중앙값을 내면 300,000창 x 후보수 만큼 들어 비싸다.
        창 안에서 균등 간격으로 24점만 찍어 통전 평균을 낸다 - 순위를 매기는 데는
        충분하고 비용은 후보당 24회 조회다.
        """
        n = self.load_strata_candidates
        if not self.load_stratified or span < target_len // 4 or n < 2:
            return int(np.random.randint(0, span + 1))
        starts = np.random.randint(0, span + 1, size=n)
        off = np.linspace(0, target_len - 1, 24).astype(np.int64)
        idx = starts[:, None] + off[None, :]
        m = on_series[idx].astype(bool)
        lvl = np.where(m.any(1),
                       (target_p[idx] * m).sum(1) / np.maximum(m.sum(1), 1),
                       0.0)
        lo, hi = float(lvl.min()), float(lvl.max())
        if hi - lo < 1e-6:                 # 전력이 평평하다 - 계층이 의미 없다
            return int(starts[np.random.randint(0, n)])
        want = lo + np.random.rand() * (hi - lo)
        return int(starts[int(np.argmin(np.abs(lvl - want)))])

    # ── 일반 부하: 돌입 구간 보존 리샘플링 ──────────────────────────────────
    def _warp_time_series(
        self,
        c_series: np.ndarray,       # (L_orig, 15) complex64
        pow_series: np.ndarray,     # (L_orig, 6) float32
        state_series: np.ndarray,   # (L_orig,) int16
        target_p: np.ndarray,       # (L_orig,) float32
        on_series: np.ndarray,      # (L_orig,) int8
        inrush_len: int,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                on_series[:take].copy(),
            )

        c_inrush = c_series[:inrush_len]
        pow_inrush = pow_series[:inrush_len]
        state_inrush = state_series[:inrush_len]
        target_p_inrush = target_p[:inrush_len]
        on_inrush = on_series[:inrush_len]

        c_steady = c_series[inrush_len:]
        pow_steady = pow_series[inrush_len:]
        state_steady = state_series[inrush_len:]
        target_p_steady = target_p[inrush_len:]
        on_steady = on_series[inrush_len:]

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
        on_warped = on_steady[nearest_idx]  # 이진 라벨이므로 최근접 이웃

        return (
            np.concatenate([c_inrush, c_steady_warped], axis=0),
            np.concatenate([pow_inrush, pow_steady_warped], axis=0),
            np.concatenate([state_inrush, state_warped], axis=0),
            np.concatenate([target_p_inrush, target_p_warped], axis=0),
            np.concatenate([on_inrush, on_warped], axis=0),
        )
