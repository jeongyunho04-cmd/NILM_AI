"""
복합 가전 부하 신호 합성기 (Multi-Appliance Load Synthesizer)
==============================================================
여러 가전의 전류 페이저를 중첩해 하나의 계량점에서 볼 법한 합성 신호를 만들고,
가전별 정답(Ground Truth)을 함께 생성한다.

[이 버전에서 바로잡은 정합성 문제]
1. 대기전력과 활성전력의 명시적 분리
   이전에는 대기 전류 위에 활성 전류를 '더했다'. 그러나 활성화 파형은 기기 전체를
   측정한 것이라 그 안에 이미 기기 자신의 대기 회로 소비가 들어 있다. 더하면 이중 계상이다.
   이제 시점마다 활성이면 활성 파형으로, 꺼진 채 꽂혀 있으면 대기 지문으로 '교체'한다.

2. 정답 라벨 상호 모순 제거
   이전에는 gt_is_on = 0, gt_target_power_w = 0.0 인 시점인데 gt_harmonics_ri 는
   0 이 아닌 값(대기 전류)을 갖고 있었다. 멀티태스크 학습에서 "꺼졌고 0W 인데
   고조파는 흐른다"는 모순된 지도신호가 된다. 이제 gt_harmonics_ri 는 활성 성분만 담고,
   대기 성분은 gt_is_plugged / gt_standby_power_w 라는 별도 채널로 명시한다.

3. 전력 분해 검산 가능
   P_aggregate = Σ(활성 전력) + Σ(대기 전력) + 계측계 자체 소비
   가 정확히 성립한다. 계측 보드 소비는 기기 것이 아니므로 딱 한 번만 더해진다.

4. 전압 환산 기준을 v_ref 로
   각 파형이 실제로 녹화된 전압을 기준으로 환산한다. 하드코딩된 220V 가 아니다.

5. 전압-전류 되먹임 2회 반복
   전압 강하가 전류를 바꾸고, 바뀐 전류가 다시 전압을 바꾼다. 1회만 계산하던 것을
   2회 반복해 수렴시킨다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.preprocessing.file_registry import (
    get_resistive_appliances,
    get_smps_appliances,
    get_usage_probability,
    is_low_load,
)

from .augmentor import DataAugmentor
from .grid_simulator import GridSimulator, VoltageEnvironment
from .segment_pool import ApplianceActivation, SegmentPool, StandbyProfile

NUM_HARMONICS = 15

# 멀티탭/차단기 용량 상한. 이 값을 넘는 '지속' 부하 조합은 만들지 않는다.
# 돌입 전류 같은 순간 스파이크는 제한하지 않는다 - 물리적으로도 정상이고
# 모델이 배워야 할 신호이기 때문이다.
# 국내 멀티탭은 보통 15A(3.3kW) ~ 16A(3.5kW) 정격이라 4kW 는 그 위의 안전 한도다.
DEFAULT_SUSTAINED_POWER_LIMIT_W = 4000.0

# `gt_is_plugged` 가 "콘센트에 꽂혀 있음" 이 아니라 **"동작 세션 중"** 을
# 뜻하는 기기 (2026-09-03, 12.164).
#
# 오븐은 '안 켜진' 상태가 **둘**이다:
#     미사용        OFF_STANDBY   0.40 W /  6.44 mA
#     세션 중 히터off FAN_LIGHT   15.02 W / 67.4  mA   <- 팬과 조명이 돈다
# 손실의 `idle = σ(plugged)·(1−σ(on))` 은 구조적으로 뒤쪽 자리인데,
# `gt_plugged` 가 꽂혀 있기만 하면 항상 1 이라 **두 상태를 가를 신호가 없다**.
# 그래서 모델이 평균값(실측 5.37W)으로 수렴하고, 남는 ~10W 가 다른 기기로
# 전가되어 SMPS 배분까지 흔든다. 세션 중에만 1 로 두면 `idle` 이 정확히
# FAN_LIGHT 자리가 되고 오븐 미사용 창에서는 0 이 된다.
#
# `gt_is_on` 과 `gt_standby_p` 는 **건드리지 않는다** - 그 둘은 이미 맞다
# (휴지 구간의 15.02W 는 `net_power_features[:,0]` 에서 온다). 오븐을 3상태로
# 만들면 `L_res`/`L_swap` 이 FAN_LIGHT 창에 히터 전력 1,143W 를 강요한다.
#
# 대기 전류 레이어는 그대로 둔다 - 미사용 오븐도 실제로 6.44mA 를 먹는다.
# 라벨만 바꾸고 관측은 실측 그대로 유지한다.
SESSION_PLUGGED_APPS: Tuple[str, ...] = ("oven",)

# 상시 배경 부하 (2026-09-03, 12.166). 기기가 아니라 **집 자체**가 먹는 것이다.
# 실측 "모든 기기 OFF" 창에서 재면 2.6~5.3W, `k = |I1|·V/P ≈ 3.1` 의 강한 용량성
# 부하가 늘 흐른다. 우리 `noise_signature`(계측계 1.41W, k 1.37)와는 서명 각도가
# 60~79° 로 다른 물건이고, 전력은 3.5배 크다. 즉 **모델링이 빠져 있었다.**
#
# 크기가 문제다 — 배경 5W 의 |I1| 0.074A 가 **미니PC 9.5W 의 0.050A 보다 크다.**
# 합성이 이것을 안 넣으면 모델은 실측에서 만나는 이 상시 전류를 설명할 곳이 없어
# 가장 싼 SMPS 로 흘린다. 12.159 의 장소 B 미니PC −77% 과소평가가 그 모양이다.
BACKGROUND_W_RANGE: Tuple[float, float] = (2.6, 8.3)

# 무작위 윈도우에서 가전을 고르는 방식
SELECTION_REALISTIC = "realistic"  # 기기별 사용률에 따라 각자 독립적으로 켜짐/꺼짐
SELECTION_UNIFORM = "uniform"      # 9종 균등 추첨 (희귀 기기 학습 표본 확보용)

# 지속 부하를 판정하는 창 길이 (초). 이보다 짧게 스쳐가는 피크는 스파이크로 본다.
SUSTAINED_WINDOW_S = 2.0

# seq2point 타깃 시점을 창 '끝'에서 몇 사이클 안쪽에 둘지.
#
# 창 중앙을 타깃으로 잡으면 창 절반만큼의 미래가 필요하다. 10초 창이면 5초,
# 120초 광역 창이면 60초 지연이라 실시간 추론이 성립하지 않는다.
# 그렇다고 0(완전 인과)으로 두면 기기가 켜지는 순간의 증거가 마지막 1샘플뿐이라
# 돌입 전류를 쓸 수 없다 - 드라이기 냉간저항 시정수가 0.15초(9사이클)이고
# 스위칭 과도는 그보다 뒤에 온다.
# 60사이클(1초)을 남기면 돌입·정착 과도를 전부 포함하면서 지연은 1초에 머문다.
# 2026-08-22: 60(1초) -> 360(6초). 12.9.12절 참조 — 오븐+핫플 동시 발열을
# 전기포트로 오인하는 실패를 고치려면 타깃 **이후**의 오븐 전이를 봐야 한다.
# 앞 1초로는 그 전이를 5% 밖에 못 잡고, 6초면 약 50% 를 잡는다.
# **`src/model/inputs.py` 의 TARGET_LOOKAHEAD 가 정본이다 — 여기서 가져온다.**
# 2026-08-24: 두 곳에 따로 적어 두고 회귀 테스트로만 묶어 뒀더니, 12.45 에서
# 입력 쪽만 540 으로 바꾸고 이쪽을 잊었다. 라벨은 3239, 입력은 3059 를 가리키는
# 캐시가 16분에 걸쳐 만들어졌다 (학습 직전 가드가 잡았다). 테스트는 돌려야 잡고
# import 는 안 돌려도 잡는다. **한 곳에서만 정한다.**
# `model.inputs` 는 numpy 만 import 하므로 순환하지 않는다.
from src.model.inputs import TARGET_LOOKAHEAD as DEFAULT_TARGET_LOOKAHEAD_CYCLES


def window_target_index(
    window_size_cycles: int,
    lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
) -> int:
    """창 안에서 seq2point 타깃이 놓이는 인덱스.

    이 값 하나가 배치 생성기·캐시·활성화 배치 세 곳에서 같아야 한다.
    어긋나면 캐시의 균형 가중치가 실제 학습 라벨과 다른 시점을 기준으로 계산되어
    조용히 틀린다.
    """
    n = int(window_size_cycles)
    if n <= 0:
        return 0
    return int(np.clip(n - 1 - int(lookahead_cycles), 0, n - 1))


def _max_sustained_power(p_series: np.ndarray, sampling_hz: float = 60.0) -> float:
    """가장 오래 유지된 부하 크기. 순간 스파이크는 이동평균에 희석되어 잡히지 않는다."""
    n = len(p_series)
    if n == 0:
        return 0.0
    w = max(1, int(SUSTAINED_WINDOW_S * sampling_hz))
    if n <= w:
        return float(np.mean(p_series))
    # 누적합으로 O(N) 이동평균
    c = np.concatenate([[0.0], np.cumsum(p_series, dtype=np.float64)])
    return float(np.max((c[w:] - c[:-w]) / w))

# 기기별 플러그 연결 확률 기본값.
# 상시 대기 회로가 있는 기기는 늘 꽂혀 있고, 휴대용 발열 기구는 쓸 때만 꽂는다.
DEFAULT_PLUG_PROBABILITY: Dict[str, float] = {
    "air_conditioner": 0.85,
    "oven": 0.85,
    "beam_projector": 0.85,
    "minipc": 0.85,
    "electiric_kettle": 0.50,
    "hair_dryer": 0.50,
}


@dataclass
class ApplianceSchedule:
    """합성 타임라인에 배치할 가전 1회 동작 계획.

    start_cycle 은 음수를 허용한다. 음수는 "윈도우가 시작되기 전에 이미 켜져 있었다"는
    뜻이며, 활성화 파형의 앞부분을 잘라내어 배치한다.
    (이전에는 max(0, start) 로 잘라 버려서 활성화 시작점의 52.8% 가 정확히 0번 인덱스에
     몰렸고, 모델이 "돌입 전류는 항상 윈도우 맨 앞에 있다"를 학습하는 편향이 있었다)
    """
    appliance_type: str
    start_cycle: int
    duration_cycles: Optional[int] = None
    power_scale: Optional[float] = None
    phase_jitter_deg: Optional[float] = None


@dataclass
class SyntheticLoadSample:
    """합성된 복합 부하 신호와 가전별 정답."""
    duration_cycles: int
    duration_s: float
    appliance_types: List[str]
    active_appliances: List[str]
    plugged_in_appliances: List[str]

    # ── 모델 입력 (계량점에서 관측되는 값) ──
    harmonics_ri: np.ndarray          # (N, 15, 2) float32 [Real, Imag]
    harmonics_complex: np.ndarray     # (N, 15) complex64
    power_features: np.ndarray        # (N, 6) float32 [P, Q, S, PF, V_measured, THD_i]
    v_bus: np.ndarray                 # (N,) float32 계측 해상도로 계단화된 전압 (모델이 보는 값)
    v_bus_true: np.ndarray            # (N,) float32 연속 실제 전압 (진단용)
    t_rel_s: np.ndarray               # (N,) float32

    # ── 가전별 정답 ──
    gt_is_on: Dict[str, np.ndarray]           # (N,) int8  활성 동작 여부
    gt_is_plugged: Dict[str, np.ndarray]      # (N,) int8  콘센트 연결 여부 (대기전력 유무)
    gt_state_id: Dict[str, np.ndarray]        # (N,) int16
    gt_target_power_w: Dict[str, np.ndarray]  # (N,) float32 활성 전력 (꺼졌으면 0)
    gt_standby_power_w: Dict[str, np.ndarray] # (N,) float32 대기 전력 (활성 중이면 0)
    gt_harmonics_ri: Dict[str, np.ndarray]    # (N, 15, 2) float32 활성 고조파 (꺼졌으면 0)
                                              # compute_gt_harmonics=False 이면 빈 딕셔너리

    # ── 기기 것이 아닌 성분 ──
    p_noise_w: np.ndarray             # (N,) float32 계측계 자체 소비

    metadata: Dict = field(default_factory=dict)
    # 가전별 고조파 정답이 채워져 있는지. 전력 회귀만 학습한다면 필요 없고,
    # 나머지 정답 5종을 합친 것의 10배 용량을 차지하므로 기본적으로 끄는 편이 낫다.
    gt_harmonics_included: bool = True

    def verify_power_decomposition(self, tolerance_w: float = 0.5) -> Tuple[bool, float]:
        """P_total = Σ활성 + Σ대기 + 계측계 소비 가 성립하는지 검산한다."""
        total = self.power_features[:, 0]
        recon = self.p_noise_w.copy()
        for app in self.appliance_types:
            recon = recon + self.gt_target_power_w[app] + self.gt_standby_power_w[app]
        max_err = float(np.max(np.abs(total - recon))) if len(total) else 0.0
        return max_err <= tolerance_w, max_err


class LoadSynthesizer:
    """현실적인 가정 복합 부하를 합성한다."""

    def __init__(
        self,
        segment_pool: SegmentPool,
        grid_simulator: Optional[GridSimulator] = None,
        augmentor: Optional[DataAugmentor] = None,
        voltage_feedback_iterations: int = 2,
        quantize_voltage_measurement: bool = True,
        compute_gt_harmonics: bool = True,
        sustained_power_limit_w: Optional[float] = DEFAULT_SUSTAINED_POWER_LIMIT_W,
        background: bool = False,
        background_w_range: Tuple[float, float] = BACKGROUND_W_RANGE,
    ):
        self.pool = segment_pool
        # 상시 배경 부하 (12.166). 기본은 꺼 둔다 — 켜면 합성 분포가 바뀌므로
        # 캐시를 새로 만들어야 하고, 기존 체크포인트와 비교가 끊긴다.
        self.background_w_range = tuple(background_w_range)
        self._bg = None
        if background:
            from .sp_curves import load_curves, BACKGROUND
            self._bg = load_curves().get(BACKGROUND)
            if self._bg is None:
                raise FileNotFoundError(
                    "배경 곡선을 못 찾았다 — processed_data/sp_curves.npz 가 필요하다")
        self.grid_sim = grid_simulator or GridSimulator()
        self.augmentor = augmentor or DataAugmentor()
        self.known_appliances = self.pool.get_appliance_types()
        # 지속 부하 상한. None 이면 제한하지 않는다.
        self.sustained_power_limit_w = sustained_power_limit_w
        self.voltage_feedback_iterations = max(1, voltage_feedback_iterations)
        self.quantize_voltage_measurement = quantize_voltage_measurement
        # 가전별 고조파 정답을 만들지 여부의 기본값. 호출 시점에 덮어쓸 수 있다.
        # 전력·상태만 학습한다면 쓰이지 않는데, 윈도우당 0.65MB 로
        # 나머지 정답 5종을 합친 것(0.065MB)의 10배를 차지한다.
        self.compute_gt_harmonics = compute_gt_harmonics

    # ── 지속 부하 예산 ──────────────────────────────────────────────────────
    def estimate_steady_power_w(self, appliance_type: str, bus_voltage_v: float) -> float:
        """주어진 계통 전압에서 이 가전이 지속적으로 끌어갈 전력을 추정한다.

        전압 환산이 반드시 필요하다. 전기포트는 212.5V 에서 녹화되어 1271W 였지만,
        240V 환경에서는 P ∝ V² 이므로 1621W 까지 오른다(+27%). 녹화 당시 값으로
        용량을 계산하면 실제로는 한도를 넘는 조합이 통과해 버린다.
        """
        base = self.pool.get_steady_power_w(appliance_type)
        if base <= 0.0:
            return 0.0
        v_ref = self.pool.get_reference_voltage(appliance_type)
        kappa = float(np.clip(bus_voltage_v / max(v_ref, 1.0), 0.80, 1.20))
        exponent = self.grid_sim.power_voltage_exponent(appliance_type)
        return float(base * (kappa ** exponent))

    def _fit_within_power_budget(
        self, candidates: List[str], bus_voltage_v: float, limit_w: Optional[float]
    ) -> Tuple[List[str], List[str]]:
        """지속 부하 합이 한도를 넘지 않도록 가전 목록을 추려낸다.

        Returns:
            (채택된 가전, 예산 초과로 빠진 가전)
        """
        if limit_w is None or not candidates:
            return list(candidates), []

        # 순서를 섞어 특정 가전만 반복적으로 탈락하는 편향을 없앤다.
        order = list(candidates)
        np.random.shuffle(order)

        accepted: List[str] = []
        dropped: List[str] = []
        budget = float(limit_w)
        for app in order:
            need = self.estimate_steady_power_w(app, bus_voltage_v)
            if need <= budget:
                accepted.append(app)
                budget -= need
            else:
                dropped.append(app)
        return accepted, dropped

    # ── 플러그 연결 상태 결정 ───────────────────────────────────────────────
    def _resolve_plugged(
        self,
        plugged_in_appliances: Optional[Dict[str, bool]],
        default_plugged_prob: float,
    ) -> Dict[str, bool]:
        is_plugged: Dict[str, bool] = {}
        for app in self.known_appliances:
            if plugged_in_appliances is not None and app in plugged_in_appliances:
                is_plugged[app] = bool(plugged_in_appliances[app])
            else:
                prob = DEFAULT_PLUG_PROBABILITY.get(app, default_plugged_prob)
                is_plugged[app] = bool(np.random.rand() < prob)
        return is_plugged

    # ── 본체 ────────────────────────────────────────────────────────────────
    def synthesize_scenario(
        self,
        total_duration_cycles: int,
        schedules: List[ApplianceSchedule],
        plugged_in_appliances: Optional[Dict[str, bool]] = None,
        default_plugged_prob: float = 0.7,
        include_noise: bool = True,
        simulate_voltage_drop: bool = True,
        voltage_environment: Optional[VoltageEnvironment] = None,
        compute_gt_harmonics: Optional[bool] = None,
    ) -> SyntheticLoadSample:
        """스케줄에 따라 복합 부하 타임라인을 합성한다.

        Args:
            compute_gt_harmonics: 가전별 고조파 정답을 만들지 여부.
                None 이면 생성자에서 정한 기본값을 따른다.
                전력·상태 회귀만 학습한다면 False 로 두는 편이 낫다.
        """
        N = int(total_duration_cycles)
        want_gt_harmonics = (
            self.compute_gt_harmonics if compute_gt_harmonics is None else bool(compute_gt_harmonics)
        )

        # 1. 배전 환경 결정 (전압 무리, 배선 임피던스, 요동 특성)
        env = voltage_environment or self.grid_sim.sample_environment()

        # 2. 플러그 연결 상태
        is_plugged = self._resolve_plugged(plugged_in_appliances, default_plugged_prob)

        # 3. 레이어 초기화
        gt_is_on = {a: np.zeros(N, dtype=np.int8) for a in self.known_appliances}
        gt_plugged = {a: np.zeros(N, dtype=np.int8) for a in self.known_appliances}
        gt_state_id = {a: np.zeros(N, dtype=np.int16) for a in self.known_appliances}
        gt_active_p = {a: np.zeros(N, dtype=np.float32) for a in self.known_appliances}
        gt_standby_p = {a: np.zeros(N, dtype=np.float32) for a in self.known_appliances}
        # 고조파 정답은 요청했을 때만 할당한다. 9종 x (N,15,2) float32 라
        # 나머지 정답을 전부 합친 것보다 10배 크고, 쓰지 않으면 순수한 낭비다.
        gt_harm_ri = (
            {a: np.zeros((N, NUM_HARMONICS, 2), dtype=np.float32) for a in self.known_appliances}
            if want_gt_harmonics else {}
        )

        # 각 가전의 전류 페이저 레이어와, 그 파형이 녹화된 기준 전압
        layer_c = {a: np.zeros((N, NUM_HARMONICS), dtype=np.complex64) for a in self.known_appliances}
        v_ref_series = {
            a: np.full(N, self.grid_sim.default_ref_voltage, dtype=np.float32)
            for a in self.known_appliances
        }

        # 4. 대기 레이어: 꽂혀 있지만 꺼진 상태
        for app in self.known_appliances:
            if not is_plugged[app]:
                continue
            # 오븐류는 '세션 중'에만 1 이다 (SESSION_PLUGGED_APPS 주석 참조).
            # 5절의 활성화 배치가 자기 구간에서 1 로 올린다.
            if app not in SESSION_PLUGGED_APPS:
                gt_plugged[app][:] = 1
            profile: StandbyProfile = self.pool.get_standby_profile(app)
            if profile.power_w <= 0.0 and not np.any(profile.harmonics_complex):
                continue  # 기계식 스위치 기기 - 꺼지면 회로가 끊겨 대기전력이 없다
            st_c, st_p = self.pool.sample_standby_series(app, N)
            layer_c[app][:] = st_c
            gt_standby_p[app][:] = st_p
            v_ref_series[app][:] = profile.v_ref_v

        active_set = set()

        # 5. 활성화 배치. 활성 구간은 대기 지문을 '덮어쓴다'(더하지 않는다).
        for sched in schedules:
            app = sched.appliance_type
            if app not in self.known_appliances:
                continue

            raw_act = self.pool.sample_activation(app)
            aug_act = self.augmentor.augment_activation(
                raw_act,
                target_duration_cycles=sched.duration_cycles,
                power_scale=sched.power_scale,
                phase_jitter_deg=sched.phase_jitter_deg,
            )

            t_start = int(sched.start_cycle)
            act_len = aug_act.duration_cycles

            # 음수 시작 = 윈도우 이전에 이미 켜져 있었다. 파형 앞부분을 잘라낸다.
            src_offset = 0
            if t_start < 0:
                src_offset = -t_start
                t_start = 0
                if src_offset >= act_len:
                    continue  # 윈도우가 시작되기 전에 이미 종료된 동작
            if t_start >= N:
                continue

            place_len = min(N - t_start, act_len - src_offset)
            if place_len <= 0:
                continue

            t_end = t_start + place_len
            s0, s1 = src_offset, src_offset + place_len

            # 활성화 안에서도 통전이 끊기는 구간이 있다. 서모스탯/릴레이 부하가
            # 그렇다 - 핫플레이트는 약 0.9초 통전 / 1.1초 휴지를 반복한다.
            # 그 구간을 일괄 ON 으로 덮으면 실측 42% 통전이 100% 로 둔갑한다.
            on_slice = aug_act.is_on[s0:s1].astype(bool)
            if on_slice.any():
                active_set.add(app)

            # 동작 구간에서는 기기가 꽂힌 채 돌고 있으므로 대기 지문 대신 실측 파형이
            # 흐른다. 휴지 구간의 전류도 그 파형 안에 이미 들어 있다.
            layer_c[app][t_start:t_end] = aug_act.net_harmonics_complex[s0:s1]
            gt_active_p[app][t_start:t_end] = np.where(
                on_slice, aug_act.target_power_w[s0:s1], 0.0
            )
            # 휴지 구간은 '꺼진 것'이 아니라 '꽂힌 채 통전만 끊긴 것'이다.
            # 그때 실제로 흐르는 전력(계측 바닥 제거본)을 대기 전력으로 잡아야
            # P = Σ활성 + Σ대기 + 계측계 분해가 계속 성립한다.
            gt_standby_p[app][t_start:t_end] = np.where(
                on_slice, 0.0, np.maximum(0.0, aug_act.net_power_features[s0:s1, 0])
            )
            gt_is_on[app][t_start:t_end] = on_slice.astype(np.int8)
            gt_plugged[app][t_start:t_end] = 1      # 돌고 있으면 당연히 꽂혀 있다
            gt_state_id[app][t_start:t_end] = aug_act.state_id[s0:s1]

            # 전압 환산의 기준은 '녹화 당시 그 순간의 전압'이어야 한다.
            # 녹화 파일 전체의 중앙값을 쓰면, 자기 부하로 전압이 내려간 채 측정된
            # 구간을 높은 전압에서 측정한 것으로 착각한다.
            # 오븐이 그 예로, 전체 중앙값은 223.3V 지만 히터가 통전하는 동안에는
            # 자기 강하로 216.8V 였다. 저항 부하는 P∝V^2 이라 6.0% 전력 오차가 되고,
            # PF 가 1 에 가까운 구간에서는 그것이 Q 로 3배 증폭되어 나타났다
            # (실측 -82 VAR vs 합성 -262 VAR).
            vref_slice = aug_act.net_power_features[s0:s1, 4]
            v_ref_series[app][t_start:t_end] = np.where(
                (vref_slice > 150.0) & (vref_slice < 280.0), vref_slice, aug_act.v_ref_v
            )

        # 6. 배경 노이즈 (계측계 자체 소비). 전체에서 딱 한 번만 더한다.
        noise_c = np.zeros((N, NUM_HARMONICS), dtype=np.complex64)
        p_noise = np.zeros(N, dtype=np.float32)
        if include_noise and N > 0:
            _, noise_c, noise_pow = self.pool.sample_noise_slice(N)
            noise_c = noise_c.astype(np.complex64)
            p_noise = noise_pow[:, 0].astype(np.float32)

        # 6b. 상시 배경 부하 (12.166). 계측계와 **다른 성분**이라 따로 더한다.
        # 창 안에서는 상수로 둔다 — 실측에서 배경은 분 단위로 천천히 움직이고
        # 창은 60초다. 창마다 다시 뽑으므로 모델이 "안 변하는 성분 = 배경" 이라는
        # 합성 전용 단서를 배우지는 않는다.
        if self._bg is not None and N > 0:
            # 배경은 **집의 성질**이다 (12.166.4). 그 콘센트의 실측 범위가 있으면
            # 그것을 쓴다 — 없으면 기본 범위. 장소 A 4.0~6.0W / 장소 B 2.0~3.5W 로
            # 2배 다르므로, 하나의 범위로 뽑으면 한쪽이 반드시 어긋난다.
            rng = getattr(env, "background_w_range", None) or self.background_w_range
            bg_w = float(np.random.uniform(*rng))
            bg_c = self._bg.current(bg_w, env.base_voltage_v)[:NUM_HARMONICS]
            noise_c = noise_c + bg_c.astype(np.complex64)[None, :]
            p_noise = p_noise + np.float32(bg_w)

        # 7. 전압 강하와 부하 응답의 되먹임을 반복 수렴시킨다.
        if simulate_voltage_drop and N > 0:
            v_open = self.grid_sim.open_circuit_voltage(N, env)
            coupled_c = {a: layer_c[a] for a in self.known_appliances}
            v_true = None

            for _ in range(self.voltage_feedback_iterations):
                total_c = noise_c.copy()
                for a in self.known_appliances:
                    total_c += coupled_c[a]
                v_true = self.grid_sim.apply_load_drop(v_open, total_c, env)

                # 갱신된 전압으로 각 기기의 전류를 다시 변형한다.
                coupled_c = {}
                for a in self.known_appliances:
                    if not np.any(layer_c[a]):
                        coupled_c[a] = layer_c[a]
                        continue
                    kappa = (v_true / v_ref_series[a]).astype(np.float32)
                    coupled_c[a] = self.grid_sim.apply_cross_appliance_coupling(
                        a, layer_c[a], kappa
                    )

            total_complex = noise_c.copy()
            for a in self.known_appliances:
                total_complex += coupled_c[a]

            # 전력도 최종 전압 기준으로 응답시킨다.
            #   저항성 P∝V^2 / SMPS P=일정 / 모터 P∝V^0.7
            for a in self.known_appliances:
                kappa = (v_true / v_ref_series[a]).astype(np.float32)
                gt_active_p[a] = self.grid_sim.apply_power_voltage_response(a, gt_active_p[a], kappa)
                gt_standby_p[a] = self.grid_sim.apply_power_voltage_response(a, gt_standby_p[a], kappa)
                # 정답 고조파는 '활성 구간만' 담는다. 꺼진 구간은 0 이어야
                # gt_is_on / gt_target_power_w 와 모순이 생기지 않는다.
                if want_gt_harmonics:
                    on_mask = gt_is_on[a].astype(bool)
                    if on_mask.any():
                        gt_harm_ri[a][on_mask, :, 0] = np.real(coupled_c[a][on_mask])
                        gt_harm_ri[a][on_mask, :, 1] = np.imag(coupled_c[a][on_mask])
        else:
            v_true = np.full(N, env.base_voltage_v, dtype=np.float32)
            v_open = v_true
            total_complex = noise_c.copy()
            for a in self.known_appliances:
                total_complex += layer_c[a]
                if want_gt_harmonics:
                    on_mask = gt_is_on[a].astype(bool)
                    if on_mask.any():
                        gt_harm_ri[a][on_mask, :, 0] = np.real(layer_c[a][on_mask])
                        gt_harm_ri[a][on_mask, :, 1] = np.imag(layer_c[a][on_mask])

        # 8. 계측 해상도 반영. 실측 센서는 0.5초에 한 번만 전압을 갱신한다.
        v_measured = (
            self.grid_sim.quantize_measurement(v_true)
            if self.quantize_voltage_measurement else v_true
        )

        # 9. 집계 전기량 계산
        harmonics_ri = np.zeros((N, NUM_HARMONICS, 2), dtype=np.float32)
        harmonics_ri[:, :, 0] = np.real(total_complex)
        harmonics_ri[:, :, 1] = np.imag(total_complex)

        mag_sq = np.real(total_complex) ** 2 + np.imag(total_complex) ** 2
        irms_total = np.sqrt(np.sum(mag_sq, axis=1)).astype(np.float32)
        i1_mag = np.sqrt(mag_sq[:, 0]) + 1e-6 if N else np.zeros(0, dtype=np.float32)

        # 유효전력은 정답의 합으로 정의한다. 이렇게 해야 분해 검산이 정확히 성립한다.
        p_total = p_noise.copy()
        for a in self.known_appliances:
            p_total = p_total + gt_active_p[a] + gt_standby_p[a]

        s_total = (v_measured * irms_total).astype(np.float32)
        q_sq = np.maximum(0.0, s_total ** 2 - p_total ** 2)
        # Q 의 부호는 imag(I1) 의 '반대'다.
        #
        # 펌웨어가 내보내는 두 값의 관계가 phase_deg = -ihdeg1 이고(실측 전 구간에서
        # |phase_deg + ihdeg1| 중앙값이 정확히 0), FeatureExtractor 는 phase_deg 의
        # 부호로 Q 를 정한다. 따라서 imag(I1) = ih1*sin(ihdeg1) 과는 부호가 뒤집힌다.
        # 실측 3개 파일 전 구간에서 Q 와 imag(I1) 의 부호가 같은 비율은 0.0% 였다.
        # 이전 구현은 같은 부호를 주어, 합성 데이터의 Q 채널이 통째로 반대였다
        # (실측 저부하 -89 VAR vs 합성 +77 VAR).
        q_sign = np.where(np.imag(total_complex[:, 0]) < 0, 1.0, -1.0) if N else np.zeros(0)
        q_total = (np.sqrt(q_sq) * q_sign).astype(np.float32)
        pf_total = np.clip(p_total / (s_total + 1e-6), 0.0, 1.0).astype(np.float32)

        higher_h_sq = np.sum(mag_sq[:, 1:], axis=1) if N else np.zeros(0)
        thd_i_total = (np.sqrt(higher_h_sq) / i1_mag).astype(np.float32)

        power_features = np.stack(
            [p_total, q_total, s_total, pf_total, v_measured, thd_i_total], axis=1
        ).astype(np.float32) if N else np.zeros((0, 6), dtype=np.float32)

        t_rel_s = (np.arange(N) / 60.0).astype(np.float32)
        plugged_list = sorted(a for a, p in is_plugged.items() if p)

        standby_total = float(np.mean(sum(gt_standby_p[a] for a in self.known_appliances))) if N else 0.0
        meta = {
            "duration_cycles": N,
            "duration_s": round(N / 60.0, 2),
            "num_schedules": len(schedules),
            "active_appliances": sorted(active_set),
            "plugged_in_appliances": plugged_list,
            "mean_p_w": round(float(np.mean(p_total)), 2) if N else 0.0,
            "max_p_w": round(float(np.max(p_total)), 2) if N else 0.0,
            "mean_standby_p_w": round(standby_total, 3),
            "mean_noise_p_w": round(float(np.mean(p_noise)), 3) if N else 0.0,
            # 전압 강하는 '그 순간의 개방 전압 대비 자기 부하가 끌어내린 양'으로 정의한다.
            # 하드코딩된 220V 와 비교하면 기저 전압이 그보다 높은 환경에서 음수가 나오고
            # (이전 리포트의 -8.56V), 환경 기저값과 비교해도 느린 요동이 위로 올라간
            # 짧은 윈도우에서는 여전히 음수가 된다. 개방 전압 기준이 물리적으로 옳다.
            "voltage_environment": env.source,
            "base_voltage_v": round(env.base_voltage_v, 2),
            "r_grid_ohm": round(env.r_grid_ohm, 4),
            "x_grid_ohm": round(env.x_grid_ohm, 4),
            "mean_v_bus": round(float(np.mean(v_measured)), 2) if N else 0.0,
            "min_v_bus": round(float(np.min(v_measured)), 2) if N else 0.0,
            "max_v_bus": round(float(np.max(v_measured)), 2) if N else 0.0,
            "max_v_sag_v": round(float(np.max(v_open - v_true)), 2) if N else 0.0,
            "gt_harmonics_included": want_gt_harmonics,
            # 2초 이동평균 최댓값. 순간 스파이크가 아니라 '지속' 부하를 본다.
            # 직접 짠 시나리오는 요청대로 만들어 주되, 한도를 넘었으면 알 수 있게 표시한다.
            "max_sustained_p_w": round(_max_sustained_power(p_total), 1) if N else 0.0,
            "sustained_power_limit_w": self.sustained_power_limit_w,
        }
        if self.sustained_power_limit_w is not None and N:
            meta["exceeds_sustained_limit"] = bool(
                meta["max_sustained_p_w"] > self.sustained_power_limit_w
            )

        return SyntheticLoadSample(
            duration_cycles=N,
            duration_s=round(N / 60.0, 2),
            appliance_types=self.known_appliances,
            active_appliances=sorted(active_set),
            plugged_in_appliances=plugged_list,
            harmonics_ri=harmonics_ri,
            harmonics_complex=total_complex,
            power_features=power_features,
            v_bus=v_measured,
            v_bus_true=np.asarray(v_true, dtype=np.float32),
            t_rel_s=t_rel_s,
            gt_is_on=gt_is_on,
            gt_is_plugged=gt_plugged,
            gt_state_id=gt_state_id,
            gt_target_power_w=gt_active_p,
            gt_standby_power_w=gt_standby_p,
            gt_harmonics_ri=gt_harm_ri,
            p_noise_w=p_noise,
            metadata=meta,
            gt_harmonics_included=want_gt_harmonics,
        )

    # ── 무작위 윈도우 ───────────────────────────────────────────────────────
    def synthesize_random_window(
        self,
        window_size_cycles: int = 600,
        max_concurrent_appliances: int = 3,
        plugged_prob: float = 0.6,
        n_active: Optional[int] = None,
        candidate_appliances: Optional[Sequence[str]] = None,
        force_plugged_all: bool = False,
        compute_gt_harmonics: Optional[bool] = None,
        selection_mode: str = SELECTION_REALISTIC,
        sustained_power_limit_w: Optional[float] = -1.0,
        target_biased_placement: bool = False,
        force_active: Optional[Sequence[str]] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    ) -> SyntheticLoadSample:
        """무작위 복합 윈도우를 빠르게 합성한다.

        Args:
            n_active: 활성 가전 수를 직접 지정 (None 이면 selection_mode 를 따른다)
            candidate_appliances: 활성 가전을 고를 후보 목록 (저부하만 뽑는 등)
            force_plugged_all: 모든 가전을 콘센트에 연결된 상태로 둔다 (대기전력 최대)
            compute_gt_harmonics: 가전별 고조파 정답 생성 여부 (None 이면 생성자 기본값)
            selection_mode:
                "realistic" - 기기별 사용률에 따라 각자 독립적으로 켜짐/꺼짐.
                    미니PC 42% / 드라이기 0.4% 처럼 실제 빈도 차이가 반영되고,
                    동시 가동 수도 0~9대로 자연스럽게 분포한다.
                "uniform"   - 9종 균등 추첨. 현실적이지는 않지만 희귀 기기의
                    학습 표본을 확보하는 데 필요하다.
            sustained_power_limit_w: 지속 부하 상한(W). -1 이면 생성자 기본값,
                None 이면 제한 없음. 돌입 스파이크는 제한하지 않는다.
            target_biased_placement: 활성 구간이 seq2point 타깃 시점을 덮도록 배치를
                치우친다. 특정 기기의 표본을 늘리려는 레시피에서, 켜 놓고도 타깃
                시점에 걸리지 않아 라벨이 0 이 되는 낭비를 줄인다.
                타깃은 창 중앙이 아니라 끝쪽이다 (window_target_index 참조).
            target_lookahead_cycles: 타깃을 창 끝에서 몇 사이클 안쪽에 둘지.
                배치 생성기·캐시와 반드시 같은 값을 써야 한다.
            force_active: 켤 가전을 직접 지정한다. 서로 다른 성격의 기기를 조합해야
                하는 레시피(고부하 + 저부하 동시)에서 쓴다. 지정하면
                n_active / selection_mode / candidate_appliances 는 무시된다.
                지속 부하 예산은 그대로 적용되므로 한도를 넘는 조합은 줄어든다.
        """
        candidates = list(candidate_appliances or self.known_appliances)
        candidates = [a for a in candidates if a in self.known_appliances]
        if not candidates:
            candidates = list(self.known_appliances)

        limit = (
            self.sustained_power_limit_w
            if (isinstance(sustained_power_limit_w, float) and sustained_power_limit_w == -1.0)
            else sustained_power_limit_w
        )

        # 전압을 먼저 정해야 지속 부하 예산을 제대로 계산할 수 있다.
        # (같은 전기포트도 212V 에서 1271W, 240V 에서 1621W 를 먹는다)
        env = self.grid_sim.sample_environment()

        # 1. 어떤 가전을 켤지 고른다
        if force_active is not None:
            chosen = [a for a in force_active if a in self.known_appliances]
        elif n_active is not None:
            k = int(np.clip(n_active, 0, len(candidates)))
            chosen = list(np.random.choice(candidates, size=k, replace=False)) if k else []
        elif selection_mode == SELECTION_UNIFORM:
            k = int(np.random.randint(0, min(max_concurrent_appliances, len(candidates)) + 1))
            chosen = list(np.random.choice(candidates, size=k, replace=False)) if k else []
        else:
            # 각 가전이 자기 사용률대로 독립적으로 켜진다.
            chosen = [a for a in candidates if np.random.rand() < get_usage_probability(a)]

        # 2. 멀티탭/차단기 용량을 넘는 조합은 걸러낸다
        chosen, over_budget = self._fit_within_power_budget(chosen, env.base_voltage_v, limit)

        plugged = {
            a: (True if force_plugged_all else bool(np.random.rand() < plugged_prob))
            for a in self.known_appliances
        }

        schedules: List[ApplianceSchedule] = []
        target = window_target_index(window_size_cycles, target_lookahead_cycles)
        for app in chosen:
            plugged[app] = True  # 켜져 있으면 반드시 꽂혀 있다
            dur_c = int(np.random.randint(window_size_cycles // 2, window_size_cycles * 3))
            # 음수 시작을 허용하고 클램프하지 않는다. 그래야 돌입 전류가
            # 윈도우 안 임의 위치에 오거나, 이미 진행 중인 동작으로 나타난다.
            if target_biased_placement:
                # 요청한 dur_c 를 그대로 믿으면 안 된다. 증강기는 원본보다
                # max_stretch(3)배 넘게 늘이지 않으므로, 짧은 활성화에 긴 길이를
                # 요청하면 실제로는 짧은 파형이 돌아온다. 그것을 모르고 멀리
                # 배치하면 활성화가 윈도우 밖으로 나가 통째로 버려진다.
                guaranteed = max(1, min(dur_c, 3 * self.pool.get_min_activation_cycles(app)))
                if np.random.rand() < 0.8:
                    # 타깃 시점을 덮는 범위: start <= target < start + 실제길이
                    start_c = int(np.random.randint(target - guaranteed + 1, target + 1))
                else:
                    # 나머지 20% 는 온셋 위치를 다양화하되, 창과 겹치는 것은 보장한다.
                    # 특정 기기의 표본을 늘리려는 레시피에서 그 기기가 아예 안 나오면
                    # 레시피 자체가 무의미해진다.
                    start_c = int(np.random.randint(-guaranteed + 1, window_size_cycles))
            else:
                start_c = int(np.random.randint(-window_size_cycles, window_size_cycles))
            schedules.append(ApplianceSchedule(app, start_cycle=start_c, duration_cycles=dur_c))

        sample = self.synthesize_scenario(
            total_duration_cycles=window_size_cycles,
            schedules=schedules,
            plugged_in_appliances=plugged,
            include_noise=True,
            simulate_voltage_drop=True,
            voltage_environment=env,
            compute_gt_harmonics=compute_gt_harmonics,
        )
        if force_active is not None:
            sample.metadata["selection_mode"] = "forced"
        else:
            sample.metadata["selection_mode"] = selection_mode if n_active is None else "explicit"
        sample.metadata["sustained_power_limit_w"] = limit
        sample.metadata["dropped_over_budget"] = sorted(over_budget)
        return sample

    def synthesize_standby_only_window(
        self,
        window_size_cycles: int = 600,
        plugged_prob: float = 0.9,
        compute_gt_harmonics: Optional[bool] = None,
    ) -> SyntheticLoadSample:
        """활성 가전이 하나도 없고 대기전력만 존재하는 윈도우.

        모델이 "대기전력 합"을 "저부하 기기 1대"로 오인하지 않게 만드는 핵심 학습 사례다.
        정답은 전 기기 OFF / 0W 이며, 그럼에도 관측 전력은 0 이 아니다.
        """
        plugged = {a: bool(np.random.rand() < plugged_prob) for a in self.known_appliances}
        return self.synthesize_scenario(
            total_duration_cycles=window_size_cycles,
            schedules=[],
            plugged_in_appliances=plugged,
            include_noise=True,
            simulate_voltage_drop=True,
            compute_gt_harmonics=compute_gt_harmonics,
        )

    def synthesize_low_load_among_standby_window(
        self,
        window_size_cycles: int = 600,
        compute_gt_harmonics: Optional[bool] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    ) -> SyntheticLoadSample:
        """대기전력이 잔뜩 깔린 상태에서 저전력 기기 딱 1대만 켜진 윈도우.

        대기전력 오탐이 실제로 일어나는 바로 그 상황이다.
        미니PC 아이들(9.8W)이나 선풍기 1단(23.5W)을, 여러 대기전력의 합과 구분해야 한다.
        """
        low_load = [a for a in self.known_appliances if is_low_load(a)]
        return self.synthesize_random_window(
            window_size_cycles=window_size_cycles,
            n_active=1,
            candidate_appliances=low_load or self.known_appliances,
            force_plugged_all=True,
            compute_gt_harmonics=compute_gt_harmonics,
            # 이 레시피의 존재 이유가 "대기전력 속에 저부하 1대"이므로 그 1대가
            # 반드시 창 안에 있어야 한다. 배치를 자유롭게 두면 활성화가 창 밖으로
            # 나가 버려 대기 전용 윈도우와 구분되지 않는 표본이 섞인다.
            target_biased_placement=True,
            target_lookahead_cycles=target_lookahead_cycles,
        )

    def synthesize_high_low_mixed_window(
        self,
        window_size_cycles: int = 600,
        compute_gt_harmonics: Optional[bool] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    ) -> SyntheticLoadSample:
        """고전력 저항 부하 1~2대와 저전력 기기 2~3대가 동시에 켜진 윈도우.

        이 프로젝트에서 가장 어려운 상황이다. 두 무리의 전력 크기가 31배 차이라
        (고부하 1139W vs 저부하 37W), 고부하 예측이 3% 만 틀려도 그 오차가
        저부하 전체의 93% 를 왜곡할 수 있다.

        그런데 다른 레시피로는 이 조합이 거의 만들어지지 않았다.
        high_power_resistive 는 저항 부하만, low_load_among_standby 는 저부하만 켠다.
        무작위 레시피에 맡기면 동시 가동 창이 3.3% 에 그친다.

        모델이 고부하 오차를 저부하로 흘리지 않도록 배우려면 이 상황을 충분히 봐야 한다.
        """
        resistive = [a for a in self.known_appliances if a in set(get_resistive_appliances())]
        low_load = [a for a in self.known_appliances if is_low_load(a)]
        if not resistive or not low_load:
            return self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                compute_gt_harmonics=compute_gt_harmonics,
            )

        n_hi = 1 if np.random.rand() < 0.7 else 2
        n_lo = int(np.random.randint(2, 4))   # 2 또는 3
        chosen = list(np.random.choice(resistive, min(n_hi, len(resistive)), replace=False))
        chosen += list(np.random.choice(low_load, min(n_lo, len(low_load)), replace=False))

        return self.synthesize_random_window(
            window_size_cycles=window_size_cycles,
            force_active=chosen,
            force_plugged_all=True,
            compute_gt_harmonics=compute_gt_harmonics,
            target_biased_placement=True,
            target_lookahead_cycles=target_lookahead_cycles,
        )

    def synthesize_resistive_overlap_window(
        self,
        window_size_cycles: int = 600,
        compute_gt_harmonics: Optional[bool] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
        min_heat_w: float = 300.0,
        max_tries: int = 20,
        pair: Optional[Sequence[str]] = None,
        exclude_active: Optional[Sequence[str]] = None,
    ) -> SyntheticLoadSample:
        """저항 발열 부하 **2대가 타깃 시점에 동시 통전**하는 윈도우.

        0.2절 마지막 문단: *"저항성끼리 겹칠 때가 진짜 시험대이며, 그것은 학습에서
        확인해야 한다."* 그런데 **그 시험을 칠 데이터가 없었다.** 홀드아웃 8,000창에서
        오븐+핫플 동시 발열이 6창(0.07%), 저항 2종 이상 동시 발열이 76창(0.95%) 뿐이다.

        그 공백이 실측에서 그대로 드러난다. `test_4`(오븐+핫플 동시 가동)에서 모델이
        1,600W 저항 덩어리를 **전기포트(정격 1533W)로 읽는 환각**이 창의 4.5% 에서 나고
        최대 1,550W 까지 간다. 합성 홀드아웃의 오븐→포트 오인은 0/166 으로 완벽한데도 그렇다.

        **2대를 켜는 것만으로는 부족하다.** 오븐은 히터 통전율이 25%, 핫플레이트는 45% 라
        둘 다 켜 두어도 타깃 시점에 동시 통전할 확률이 11% 밖에 안 된다. 그래서
        `high_power_resistive` 가 40% 확률로 2대를 켜는데도 실제 동시 발열은 1% 미만이다.
        여기서는 **타깃 시점의 발열 여부를 확인하고 아니면 다시 뽑는다.**
        """
        resistive = [a for a in self.known_appliances if a in set(get_resistive_appliances())]
        if len(resistive) < 2:
            return self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                compute_gt_harmonics=compute_gt_harmonics,
            )
        ti = window_target_index(window_size_cycles, target_lookahead_cycles)
        # **쌍은 루프 밖에서 한 번만 뽑는다.** 재시도마다 다시 뽑으면 동시 통전이
        # 쉬운 쌍이 채택을 독식한다 - 포트·드라이기는 켜지면 연속 통전이라 바로
        # 통과하고, 정작 겨냥한 오븐(통전율 25%)+핫플(45%)은 계속 기각돼 밀려난다.
        # 첫 판(2026-08-22, cnn_v13)이 그 버그로 실패했다: 포트 양성률이 10.2 ->
        # 14.5% 로 뛰고 오븐은 10.2 -> 11.7% 에 그쳐, 실측 포트 환각이 8.0 -> 13.6%
        # 로 **늘었다**. 사전확률을 올려 놓고 "환각이 줄었나" 를 물은 셈이었다.
        # `pair` 를 주면 그것을 쓴다. 실측이 던지는 구성(오븐+핫플)을 겨냥할 때
        # 필요하다 - 무작위로 뽑으면 6쌍 중 1/6 만 그 조합이라 레시피 5% 중
        # 0.83% 밖에 안 된다 (12.38 측정: 학습 전체의 1.001%).
        if pair is None:
            pair = list(np.random.choice(resistive, 2, replace=False))
        else:
            pair = [a for a in pair if a in self.known_appliances]
            if len(pair) < 2:
                pair = list(np.random.choice(resistive, 2, replace=False))
        # `exclude_active` 는 그 창에서 **활성 후보에서 뺀다.** 강제로 켜는 쌍이
        # 아닌 기기가 곁다리로 켜지는 것을 막는다. 실측 6파일 전부 전기포트가
        # 없으므로, 오븐+핫플 창에서 포트를 빼야 "1600W 저항 덩어리는 포트가
        # 아니다" 를 배운다. 포트 사전확률도 함께 내려간다.
        cand = None
        if exclude_active:
            drop = set(exclude_active) - set(pair)
            cand = [a for a in self.known_appliances if a not in drop]
        sample = None
        for _ in range(max_tries):
            sample = self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                force_active=pair,
                candidate_appliances=cand,
                force_plugged_all=True,
                compute_gt_harmonics=compute_gt_harmonics,
                target_biased_placement=True,
                target_lookahead_cycles=target_lookahead_cycles,
            )
            if all(sample.gt_target_power_w[a][ti] >= min_heat_w for a in pair):
                return sample
        # 다 실패하면 마지막 것을 그대로 쓴다. 그래도 2대가 켜져는 있다.
        return sample

    def synthesize_smps_overlap_window(
        self,
        window_size_cycles: int = 600,
        compute_gt_harmonics: Optional[bool] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
        p_trio: float = 0.5,
        p_resistive: Sequence[float] = (0.45, 0.45, 0.10),
        max_tries: int = 20,
    ) -> SyntheticLoadSample:
        """SMPS **2~3대가 타깃 시점에 동시에 켜져 있는** 윈도우 (12.88.4 의 1번).

        `resistive_overlap` 의 SMPS 판이다. 그쪽이 "저항끼리 겹칠 때가 진짜
        시험대인데 그 시험을 칠 데이터가 없다" 였다면, 여기는 12.81 이 좁힌
        조건이다 - 미니PC 는 저항 부하에 묻히지 않고 **경쟁 SMPS 에 묻힌다**:

            미니PC 재현율      저부하 <200W    고부하 >600W
            경쟁 SMPS 없음      98.2%          99.2%
            경쟁 SMPS 있음      66.6%          30.2%

        그런데 학습 분포에 그 상황이 거의 없다 (2026-08-25 측정):

            타깃 시점 SMPS≥2   합성 12.2%  vs  실측 79%
            타깃 시점 SMPS=3   합성  2.0%  vs  실측 37%

        **전체 동시성을 올리는 것으로는 안 된다.** 12.68 이 그 축을 올렸다가
        실패했고, 이번에 재 보니 `full` 프리셋도 SMPS≥2 를 12.2 -> 22.0% 로만
        올린다 (=3 은 2.0 -> 3.0%). 저항 부하가 늘 뿐이다.

        **저항 배경을 함께 켠다.** 위 표에서 가장 어려운 칸이 "고부하 + 경쟁 SMPS"
        (30.2%) 이고, 실측 test_5/6/7 도 SMPS 3종 위에 핫플·오븐이 얹힌다.
        SMPS 만 켠 조용한 창만 주면 그 칸을 안 배운다.

        Args:
            p_trio: 2대가 아니라 3대를 켤 확률. 실측에서 ≥2 중 3대의 비중이
                37/79 = 0.47 이라 그것을 따랐다.
            p_resistive: 함께 켤 저항 부하 대수(0/1/2)의 확률.
            max_tries: 타깃 시점에 다 켜져 있지 않으면 다시 뽑는 횟수.
                `resistive_overlap` 과 같은 이유다 - 켜 두는 것과 **타깃 시점에
                켜져 있는 것**은 다르다. 다만 여기서는 통전율이 아니라 배치가
                문제라 재시도가 훨씬 덜 필요하다 (SMPS 는 연속 부하다).
        """
        smps = [a for a in self.known_appliances if a in set(get_smps_appliances())]
        if len(smps) < 2:
            return self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                compute_gt_harmonics=compute_gt_harmonics,
            )
        resistive = [a for a in self.known_appliances if a in set(get_resistive_appliances())]
        ti = window_target_index(window_size_cycles, target_lookahead_cycles)

        # **조합은 루프 밖에서 한 번만 뽑는다** - 12.38 이 겪은 버그다. 재시도마다
        # 다시 뽑으면 타깃 시점에 걸리기 쉬운 조합이 채택을 독식해, 정작 겨냥한
        # 조합(3종 동시)의 사전확률이 도리어 낮아진다.
        n_smps = 3 if (len(smps) >= 3 and np.random.rand() < p_trio) else 2
        chosen = list(np.random.choice(smps, min(n_smps, len(smps)), replace=False))
        n_res = int(np.random.choice(len(p_resistive), p=np.asarray(p_resistive, float)
                                     / float(np.sum(p_resistive))))
        if n_res and resistive:
            chosen += list(np.random.choice(resistive, min(n_res, len(resistive)), replace=False))

        sample = None
        for _ in range(max_tries):
            sample = self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                force_active=chosen,
                force_plugged_all=True,
                compute_gt_harmonics=compute_gt_harmonics,
                target_biased_placement=True,
                target_lookahead_cycles=target_lookahead_cycles,
            )
            # 판정은 SMPS 쪽만 본다. 저항 배경은 켜져 있으면 되고 타깃 시점의
            # 히터 통전 여부까지 강제하면 (resistive_overlap 처럼) 기각률이 치솟는다.
            if all(int(sample.gt_is_on[a][ti]) == 1 for a in chosen[:n_smps]):
                return sample
        return sample

    def synthesize_high_power_window(
        self,
        window_size_cycles: int = 600,
        compute_gt_harmonics: Optional[bool] = None,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    ) -> SyntheticLoadSample:
        """고전력 저항 부하를 반드시 1~2대 켜는 윈도우.

        전기포트·오븐·드라이기·핫플레이트는 모두 니크롬선 부하라 고조파 지문이
        사실상 같다(포트 vs 오븐 거리 0.596%p). 서로를 가르는 단서는 시간 패턴뿐인데,
        실제 사용 빈도가 낮아 무작위 추출에만 맡기면 학습 표본이 2016 윈도우당
        11~37개까지 떨어진다. 대기전력 하드네거티브가 45%를 차지하면서
        이 기기들의 자리를 밀어낸 영향도 있다.

        대기전력 학습을 건드리지 않고 이 구간만 따로 보강하기 위한 레시피다.
        1대만 켜는 경우와 2대를 겹치는 경우를 섞어, 단독 신호와 합쳐진 신호를
        모두 보게 한다. 겹쳐서 멀티탭 용량을 넘으면 예산 로직이 알아서 줄인다.
        """
        resistive = [a for a in self.known_appliances if a in set(get_resistive_appliances())]
        if not resistive:
            return self.synthesize_random_window(
                window_size_cycles=window_size_cycles,
                compute_gt_harmonics=compute_gt_harmonics,
            )

        k = 1 if np.random.rand() < 0.6 else 2
        return self.synthesize_random_window(
            window_size_cycles=window_size_cycles,
            n_active=min(k, len(resistive)),
            candidate_appliances=resistive,
            plugged_prob=0.7,
            compute_gt_harmonics=compute_gt_harmonics,
            target_biased_placement=True,
            target_lookahead_cycles=target_lookahead_cycles,
        )
