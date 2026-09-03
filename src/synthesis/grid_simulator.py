"""
계통 전압 환경 시뮬레이터 (Grid Voltage Environment Simulator)
==============================================================
실제 측정 환경마다 인가 전압이 크게 다르다는 사실을 합성 데이터에 반영한다.

[왜 필요한가 - 실측에서 관찰된 사실]
data/ 의 원본 파일별 평균 전압을 재어 보면 두 무리로 갈린다.
    약 221V 무리 : minipc, laptop_charger, oven, hotplate, beam_projector, noise ...
    약 234V 무리 : air_conditioner, fan, hair_dryer ...
서로 다른 콘센트/배전 경로에서 측정했기 때문이다. 파일 안에서도 전압은 계속 흔들리고
(파일 내 표준편차 0.27V ~ 4.57V), 오븐처럼 1.2kW 를 끊었다 켜는 부하는 자기 자신 때문에
7.7V ~ 9.9V 의 순간 전압 강하를 만든다.

[이전 구현의 결정적 오류]
전압 비율을 항상 kappa = V_bus / 220.0 으로 계산했다. 그런데 전기포트는 214.7V 에서,
선풍기는 235.9V 에서 녹화된 파형이다. 214.7V 에서 잰 파형을 "220V 기준"이라고 가정하고
230V 로 환산하면 실제로는 214.7V -> 230V(+7.1%) 인데 220V -> 230V(+4.5%) 로 계산되어
전류와 전력이 체계적으로 어긋난다. 이제 각 활성화 구간이 실제로 녹화된 전압(v_ref)을
기준으로 환산한다.

[모델링하는 전압 변동 4가지]
1. 환경 기저 전압 : 실측 이봉분포 + 미측정 영역 탐색 성분
2. 느린 자연 요동 : 평균 회귀(Ornstein-Uhlenbeck) 과정
3. 자기 부하 강하 : Z_grid 를 통한 순간 전압 강하 (delta_V = R*I_p + X*I_q)
4. 외부 부하 사그 : 이웃 세대/냉장고 기동 등 우리가 측정하지 않은 부하로 인한 순간 강하
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.preprocessing.file_registry import LoadClass, get_load_class


# ── 실측에서 관찰된 콘센트별 전압 무리 ───────────────────────────────────────
@dataclass(frozen=True)
class VoltageCluster:
    """실측으로 확인된 하나의 배전 환경(콘센트)."""
    name: str
    mean_v: float
    std_v: float
    weight: float
    r_grid_ohm: float = 0.8   # 이 콘센트까지의 배선 저항


# data/*.csv 의 파일별 평균 전압과, 복합 부하 실측(test*.csv)에서 회귀로 구한
# 배선 저항이다. 임피던스가 큰 회선일수록 부하 시 전압이 더 내려가므로
# 평균 전압도 낮게 관측된다 - 두 값이 함께 움직이는 것이 물리적으로 맞다.
#
# 측정 방법 3가지가 모두 일치했다 (221V 콘센트 기준):
#   V = V0 - R*I_re 회귀            R = 1.501 Ohm (R^2 = 0.915)
#   X 를 포함한 회귀                 R = 1.499 Ohm
#   오븐 펄스 8개의 dV/dI 직접 측정   R = 1.588 Ohm
# 이전 모델의 0.15~0.35 Ohm 은 실측의 1/4~1/6 수준이었다.
#
# ── 장소 B 를 더한다 (2026-09-03, 12.167) ─────────────────────────────────
# `run_line_impedance` 로 복합 실측 15파일의 Z 를 계단(|ΔP|≥300W)에서 직접 쟀다.
# `Z1 = -(V1_after - V1_before) / (|I1|_after - |I1|_before)` — 전압의 **차분**을
# 쓰므로 스펙트럼 누설과 계통 표류가 상쇄된다.
#
#   장소 A  7파일 52이벤트   Z 중앙 **1.470 Ω**  (1.34~1.67)  무부하 V **223.3**
#   장소 B  4파일 52이벤트   Z 중앙 **0.907 Ω**  (0.88~0.96)  무부하 V **227.5**
#   test_14 (이사 당일)      Z 1.639 Ω          -> 아직 옛 장소 쪽이다
#
# 그룹 내 산포(±0.04Ω)가 그룹 간 차이(0.56Ω)보다 훨씬 작아 두 장소가 깨끗하게
# 갈린다. **장소 A 는 `outlet_low_221v`(1.55Ω)와 잘 맞는데, 장소 B 는 어느
# 무리와도 안 맞았다** — `outlet_high_234v` 는 234.7V/0.45Ω 로 전압이 7V 높고
# 임피던스가 절반이다. 우리가 채점하는 장소가 합성에 없었던 것이다.
#
# `outlet_high_234v` 는 **지우지 않는다.** 기기 녹화 27개 중 7개가 230V 이상이라
# (에어컨 233.6, 선풍기 233.7/235.6/236.0, 드라이기 234.2/235.4, 충전기 237.0)
# 실재하는 콘센트다. 장소 B 를 **더한다.**
OBSERVED_VOLTAGE_CLUSTERS: Tuple[VoltageCluster, ...] = (
    VoltageCluster("outlet_low_221v", mean_v=221.3, std_v=2.4, weight=0.35, r_grid_ohm=1.55),
    VoltageCluster("outlet_siteB_227v", mean_v=227.5, std_v=1.5, weight=0.25, r_grid_ohm=0.91),
    VoltageCluster("outlet_high_234v", mean_v=234.7, std_v=1.0, weight=0.20, r_grid_ohm=0.45),
)

#: 계단으로 직접 잰 장소별 선로 임피던스 (12.167). 출처 기록용.
MEASURED_SITE_Z_OHM = {"A": 1.470, "B": 0.907}

# 두 콘센트만 학습하면 모델이 그 두 전압대에만 맞춰진다. 한국 표준 공급 전압
# 220V +-10%(198~242V) 안에서 측정하지 못한 구간도 일부 섞어 일반화 여력을 남긴다.
EXPLORATION_VOLTAGE_RANGE: Tuple[float, float] = (205.0, 245.0)
EXPLORATION_WEIGHT: float = 0.20


@dataclass
class VoltageEnvironment:
    """합성 1회분이 놓이는 배전 환경. 한 시나리오 동안 고정된다."""
    base_voltage_v: float       # 무부하 상태의 계통 전압
    r_grid_ohm: float           # 옥내 배선 저항
    x_grid_ohm: float           # 옥내 배선 리액턴스
    drift_std_v: float          # 느린 자연 요동의 정상상태 표준편차
    drift_tau_s: float          # 요동의 상관 시간 (초)
    sag_rate_per_min: float     # 외부 부하 사그 발생 빈도
    source: str                 # 이 전압이 어디서 나왔는지 (클러스터명 / exploration)


class GridSimulator:
    """배전 전압 환경 샘플링 및 계통 임피던스 전압 강하(Sag) 시뮬레이터."""

    # 부하 유형별 전압 지수. I ∝ V^(i_exp), P ∝ V^(p_exp)
    #   저항성 히터 : 옴의 법칙 그대로. 전압이 오르면 전류도 전력도 오른다.
    #   SMPS       : 2차측 정전압 제어. 전력은 일정하고 전류는 반대로 움직인다.
    #   모터/인버터 : 슬립-토크 특성상 중간 거동.
    _LOAD_EXPONENTS: Dict[LoadClass, Tuple[float, float]] = {
        LoadClass.RESISTIVE: (1.0, 2.0),
        LoadClass.SMPS: (-1.0, 0.0),
        LoadClass.MOTOR: (0.7, 0.7),
        LoadClass.PASSIVE: (1.0, 1.0),
    }

    def __init__(
        self,
        voltage_clusters: Tuple[VoltageCluster, ...] = OBSERVED_VOLTAGE_CLUSTERS,
        exploration_range: Tuple[float, float] = EXPLORATION_VOLTAGE_RANGE,
        exploration_weight: float = EXPLORATION_WEIGHT,
        default_ref_voltage: float = 220.0,   # v_ref 를 모를 때만 쓰는 최후 기본값
        nominal_voltage: Optional[float] = None,  # 지정 시 전압을 이 값으로 고정
        nominal_voltage_range: Optional[Tuple[float, float]] = None,  # 구버전 호환
        # 실측 두 콘센트가 0.45 / 1.55 Ohm 이었다. 그 사이와 바깥을 조금씩 덮는다.
        # 탐색 성분(무리에 안 속한 20%)의 임피던스 폭. 실측 범위는 0.88~1.67Ω
        # 이므로 위아래로 여유를 둔 0.7~2.0 이다 (12.167). 이전 0.35~1.80 은
        # 아래쪽이 실측의 절반이라 물리적으로 없는 회선을 만들고 있었다.
        r_grid_range: Tuple[float, float] = (0.70, 2.00),
        # X 는 식별이 어렵다. 저항 부하가 지배하는 구간에서는 I_im 변동폭이
        # 0.03A 뿐이라 회귀로 분리되지 않는다(X 를 빼도 R^2 가 같다).
        # 유도성 부하가 있는 test.csv 에서만 X≈0.12 로 잡혀 그 근방을 쓴다.
        x_grid_range: Tuple[float, float] = (0.02, 0.15),
        r_grid: Optional[float] = None,
        x_grid: Optional[float] = None,
        voltage_variation_std: float = 1.0,   # 느린 요동의 정상상태 표준편차 (V)
        drift_tau_s: float = 30.0,            # 요동 상관 시간 (초)
        sag_rate_per_min: float = 0.8,        # 외부 부하 사그 발생 빈도 (회/분)
        max_external_sag_v: float = 10.0,     # 외부 사그 총 강하량 상한 (겹침 누적 방지)
        measurement_frame_cycles: int = 30,   # 실측 센서의 전압 갱신 주기 (0.5초 = 30사이클)
        sampling_hz: float = 60.0,
    ):
        self.default_ref_voltage = default_ref_voltage
        self.r_grid_range = (r_grid, r_grid) if r_grid is not None else r_grid_range
        self.x_grid_range = (x_grid, x_grid) if x_grid is not None else x_grid_range
        self.voltage_variation_std = voltage_variation_std
        self.drift_tau_s = drift_tau_s
        self.sag_rate_per_min = sag_rate_per_min
        self.max_external_sag_v = max_external_sag_v
        self.measurement_frame_cycles = measurement_frame_cycles
        self.sampling_hz = sampling_hz

        # 전압 고정 모드: 단일 값 또는 구버전 범위 지정이 오면 탐색 성분을 끈다.
        if nominal_voltage is not None:
            self.voltage_clusters = (VoltageCluster("fixed", nominal_voltage, 0.0, 1.0),)
            self.exploration_weight = 0.0
            self.exploration_range = (nominal_voltage, nominal_voltage)
        elif nominal_voltage_range is not None:
            self.voltage_clusters = ()
            self.exploration_weight = 1.0
            self.exploration_range = nominal_voltage_range
        else:
            self.voltage_clusters = voltage_clusters
            self.exploration_weight = exploration_weight
            self.exploration_range = exploration_range

    # ── 환경 샘플링 ─────────────────────────────────────────────────────────
    def sample_environment(self) -> VoltageEnvironment:
        """이번 합성이 놓일 배전 환경 하나를 뽑는다."""
        base_v, source, cluster_r = self._sample_base_voltage()
        # 실측 콘센트에서 뽑았다면 그 회선의 배선 저항을 함께 쓴다.
        # 전압과 임피던스는 같은 회선의 성질이므로 따로 뽑으면 짝이 어긋난다.
        if cluster_r is not None and self.r_grid_range[0] != self.r_grid_range[1]:
            r = float(np.clip(np.random.normal(cluster_r, cluster_r * 0.15), 0.1, 3.0))
        else:
            r = float(np.random.uniform(*self.r_grid_range))
        return VoltageEnvironment(
            base_voltage_v=base_v,
            r_grid_ohm=r,
            x_grid_ohm=float(np.random.uniform(*self.x_grid_range)),
            drift_std_v=self.voltage_variation_std,
            drift_tau_s=self.drift_tau_s,
            sag_rate_per_min=self.sag_rate_per_min,
            source=source,
        )

    def _sample_base_voltage(self) -> Tuple[float, str, Optional[float]]:
        """실측 이봉분포 + 미측정 영역 탐색 성분에서 기저 전압을 뽑는다.

        Returns:
            (기저 전압, 출처 이름, 그 회선의 배선 저항 or None)
        """
        cluster_weight = sum(c.weight for c in self.voltage_clusters)
        total = cluster_weight + self.exploration_weight
        if total <= 0:
            return self.default_ref_voltage, "default", None

        r = np.random.rand() * total
        acc = 0.0
        for c in self.voltage_clusters:
            acc += c.weight
            if r < acc:
                v = float(np.random.normal(c.mean_v, c.std_v)) if c.std_v > 0 else c.mean_v
                return float(np.clip(v, *EXPLORATION_VOLTAGE_RANGE)), c.name, c.r_grid_ohm

        lo, hi = self.exploration_range
        return float(np.random.uniform(lo, hi)), "exploration", None

    # ── 전압 시계열 생성 ────────────────────────────────────────────────────
    def _generate_drift(self, n_samples: int, env: VoltageEnvironment) -> np.ndarray:
        """평균 회귀(Ornstein-Uhlenbeck) 과정으로 느린 자연 요동을 만든다.

        이전 구현은 누적합 랜덤워크를 clip 으로 잘랐다. 그 방식은 216,000 샘플(1시간)에서
        표준편차가 9V 까지 벌어져 clip 경계에 달라붙은 채 움직이지 않는 계단이 되어 버렸다.
        전압은 실제로 평균으로 되돌아오는 성질이 있으므로 OU 과정이 물리적으로도 맞다.
        """
        if env.drift_std_v <= 0 or n_samples == 0:
            return np.zeros(n_samples, dtype=np.float32)

        dt = 1.0 / self.sampling_hz
        theta = float(np.clip(dt / max(env.drift_tau_s, dt), 1e-6, 1.0))
        # 정상상태 분산이 drift_std_v^2 가 되도록 잡음 세기를 정한다.
        sigma = env.drift_std_v * np.sqrt(2.0 * theta - theta * theta)

        noise = np.random.normal(0.0, sigma, size=n_samples)
        # 첫 값은 정상상태 분포에서 뽑아 초기 과도구간이 생기지 않게 한다.
        x0 = np.random.normal(0.0, env.drift_std_v)
        decay = 1.0 - theta

        # x[i] = decay*x[i-1] + noise[i] 는 1차 IIR 필터다.
        # 파이썬 for 문으로 돌면 60초 창(3,600 사이클)에서 창당 1.6ms 를 쓴다 -
        # 합성 전체의 7% 였다. lfilter 는 같은 점화식을 C 로 돈다.
        noise = noise.copy()
        noise[0] = x0
        try:
            from scipy.signal import lfilter
            drift = lfilter([1.0], [1.0, -decay], noise)
        except ImportError:
            drift = np.empty(n_samples, dtype=np.float64)
            drift[0] = x0
            for i in range(1, n_samples):
                drift[i] = drift[i - 1] * decay + noise[i]
        return drift.astype(np.float32)

    def _generate_external_sags(self, n_samples: int, env: VoltageEnvironment) -> np.ndarray:
        """우리가 측정하지 않은 외부 부하(이웃 세대, 냉장고 기동)로 인한 순간 전압 강하.

        급격히 떨어지고 지수적으로 회복하는 형태로, 실측 전압 파형에서 흔히 보이는 모습이다.
        반환값은 빼야 할 강하량(양수)이다.
        """
        if n_samples == 0 or env.sag_rate_per_min <= 0:
            return np.zeros(n_samples, dtype=np.float32)

        duration_min = n_samples / self.sampling_hz / 60.0
        n_events = int(np.random.poisson(env.sag_rate_per_min * duration_min))
        if n_events == 0:
            return np.zeros(n_samples, dtype=np.float32)

        sag = np.zeros(n_samples, dtype=np.float32)
        for _ in range(n_events):
            start = int(np.random.randint(0, n_samples))
            depth = float(np.random.uniform(0.5, 8.0))          # 강하 깊이 (V)
            hold = int(np.random.uniform(0.2, 20.0) * self.sampling_hz)   # 유지 시간
            recover = max(1, int(np.random.uniform(0.1, 2.0) * self.sampling_hz))  # 회복 시정수

            end_hold = min(n_samples, start + hold)
            sag[start:end_hold] += depth

            # 지수 회복 꼬리
            tail_end = min(n_samples, end_hold + 5 * recover)
            if tail_end > end_hold:
                t = np.arange(tail_end - end_hold, dtype=np.float32)
                sag[end_hold:tail_end] += depth * np.exp(-t / recover)

        # 겹친 이벤트가 무한정 쌓이지 않게 총 강하량을 제한한다.
        # 긴 구간을 만들수록 이벤트가 많아져 겹칠 확률이 커지는데, 그대로 두면
        # 2시간짜리에서 200V 까지 내려가 한국 표준 공급 전압 하한(198V)에 닿았다.
        # 실제 계통에서 그만한 사그가 반복되면 기기가 먼저 멈춘다.
        return np.minimum(sag, self.max_external_sag_v)

    def open_circuit_voltage(
        self,
        n_samples: int,
        env: VoltageEnvironment,
        include_external_sags: bool = True,
    ) -> np.ndarray:
        """부하가 없을 때의 계통 전압 시계열. 기저 전압 + 느린 요동 + 외부 사그.

        자기 부하로 인한 강하와 분리해 둔 이유는, 전압 강하가 전류를 바꾸고 바뀐 전류가
        다시 전압을 바꾸는 되먹임을 여러 번 계산할 때 같은 요동/사그 실현값을
        그대로 재사용해야 하기 때문이다.
        """
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)
        v_open = env.base_voltage_v + self._generate_drift(n_samples, env)
        if include_external_sags:
            v_open = v_open - self._generate_external_sags(n_samples, env)
        return v_open.astype(np.float32)

    def apply_load_drop(
        self,
        v_open: np.ndarray,
        total_current_complex: np.ndarray,
        env: VoltageEnvironment,
    ) -> np.ndarray:
        """자기 부하 전류가 배선 임피던스에 만드는 순간 전압 강하를 적용한다.

        delta_V ≈ R_grid * I_active + X_grid * I_reactive
        유도성 부하는 전류가 전압보다 뒤지므로 이 좌표계에서 기본파 허수부가 음수다.
        리액턴스 강하는 전압을 '더' 끌어내려야 하므로 부호를 뒤집어 더한다.
        (이전 구현은 +x*i_im 이어서 유도성 부하일수록 강하가 줄어드는 방향으로 어긋나 있었다)
        """
        i1_c = total_current_complex[:, 0]
        delta_v = env.r_grid_ohm * np.real(i1_c) - env.x_grid_ohm * np.imag(i1_c)
        return np.clip(v_open - delta_v, 180.0, 260.0).astype(np.float32)

    def compute_voltage_drop(
        self,
        total_current_complex: np.ndarray,  # (N, 15) complex64
        env: Optional[VoltageEnvironment] = None,
        base_voltage: Optional[float] = None,
        r_grid: Optional[float] = None,
        x_grid: Optional[float] = None,
        include_external_sags: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """순간 단자 버스 전압(V_bus)과 기본 기준 전압 대비 스케일 비율을 계산합니다.

        Returns:
            v_bus: (N,) float32 단자 실효 전압 (Vrms)
            kappa_default: (N,) float32 = V_bus / default_ref_voltage
                기기별 정확한 환산은 voltage_ratio(v_bus, v_ref) 를 쓸 것.
                이 반환값은 v_ref 를 모르는 호출자를 위한 기본값이다.
        """
        n_samples = len(total_current_complex)
        if env is None:
            env = self.sample_environment()
        if base_voltage is not None:
            env.base_voltage_v = float(base_voltage)
        if r_grid is not None:
            env.r_grid_ohm = float(r_grid)
        if x_grid is not None:
            env.x_grid_ohm = float(x_grid)

        if n_samples == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        v_open = self.open_circuit_voltage(n_samples, env, include_external_sags)
        v_bus = self.apply_load_drop(v_open, total_current_complex, env)
        kappa_default = (v_bus / self.default_ref_voltage).astype(np.float32)
        return v_bus, kappa_default

    def quantize_measurement(self, v_bus: np.ndarray) -> np.ndarray:
        """실측 센서와 동일한 시간 해상도로 전압을 계단화한다.

        STM32 펌웨어는 전압/주파수/전압고조파를 0.5초 창(30사이클)마다 한 번만 계산하고
        그 값을 30개 주기 행에 똑같이 복제해 보낸다(원본 CSV 에서 seq 별 vrms 고유값 = 1).
        합성 전압만 60Hz 로 매끄럽게 변하면 실측과 구조가 달라져, 전압을 입력 채널로 쓰는
        모델이 합성에서만 통하는 단서를 학습하게 된다. 같은 계단으로 맞춰 준다.
        """
        n = len(v_bus)
        k = max(1, int(self.measurement_frame_cycles))
        if n == 0 or k == 1:
            return np.asarray(v_bus, dtype=np.float32)

        n_frames = int(np.ceil(n / k))
        padded = np.full(n_frames * k, np.nan, dtype=np.float64)
        padded[:n] = v_bus
        frames = padded.reshape(n_frames, k)
        # 프레임 대표값(평균)을 그 프레임 전체에 복제한다.
        frame_mean = np.nanmean(frames, axis=1, keepdims=True)
        held = np.repeat(frame_mean, k, axis=0).ravel()[:n]
        return held.astype(np.float32)

    # ── 부하별 전압 응답 ────────────────────────────────────────────────────
    def voltage_ratio(self, v_bus: np.ndarray, v_ref: float) -> np.ndarray:
        """이 파형이 실제로 녹화된 전압(v_ref) 기준의 전압 비율을 계산한다.

        하드코딩된 220V 가 아니라 v_ref 를 쓰는 것이 핵심이다.
        전기포트는 214.7V, 선풍기는 235.9V 에서 녹화되었으므로 기준이 서로 다르다.
        """
        ref = float(v_ref) if v_ref and v_ref > 1.0 else self.default_ref_voltage
        return (np.asarray(v_bus, dtype=np.float32) / ref).astype(np.float32)

    def current_voltage_exponent(self, appliance_type: str) -> float:
        """전압 변화에 대한 전류 지수. I ∝ V^exp"""
        return self._LOAD_EXPONENTS[get_load_class(appliance_type)][0]

    def power_voltage_exponent(self, appliance_type: str) -> float:
        """전압 변화에 대한 유효전력 지수. P ∝ V^exp"""
        return self._LOAD_EXPONENTS[get_load_class(appliance_type)][1]

    def apply_cross_appliance_coupling(
        self,
        appliance_type: str,
        harmonics_complex: np.ndarray,  # (N, 15) complex64
        kappa_v: np.ndarray,            # (N,) float32 전압 비율 V_bus(t) / v_ref
    ) -> np.ndarray:
        """전압 변화(kappa_v)에 따른 가전별 비선형 물리 전류 및 고조파 변형을 적용합니다."""
        if harmonics_complex.size == 0:
            return harmonics_complex.astype(np.complex64)

        kappa_col = np.asarray(kappa_v, dtype=np.float32)[:, np.newaxis]  # (N, 1)
        # 수치 폭주 방지. 실제 계통에서 이 범위를 벗어나는 일은 없다.
        kappa_col = np.clip(kappa_col, 0.80, 1.20)

        load_class = get_load_class(appliance_type)
        i_exp = self._LOAD_EXPONENTS[load_class][0]

        # np.power 는 임의 실수 지수를 다루느라 느리다. 실제로 쓰이는 지수는
        # 1.0(저항) / -1.0(SMPS) / 0.7(모터) 셋뿐이라 앞 둘은 특수화한다.
        if i_exp == 1.0:
            scale = kappa_col
        elif i_exp == -1.0:
            scale = 1.0 / kappa_col
        else:
            scale = np.power(kappa_col, i_exp)
        mod_c = harmonics_complex * scale

        # SMPS 는 저전압에서 정류 다이오드 도통각이 좁아져 3차 고조파 왜율이 상승한다.
        if load_class == LoadClass.SMPS and mod_c.shape[1] >= 3:
            # scale > 1.0 이면 저전압 상황(I ∝ 1/V)이다.
            distortion_factor = 1.0 + 0.4 * (scale[:, 0] - 1.0)
            mod_c[:, 2] *= distortion_factor

        return mod_c.astype(np.complex64)

    def apply_power_voltage_response(
        self,
        appliance_type: str,
        power_w: np.ndarray,
        kappa_v: np.ndarray,
    ) -> np.ndarray:
        """전압 변화에 따른 유효전력 변화를 적용한다. P ∝ V^exp"""
        exp = self.power_voltage_exponent(appliance_type)
        p = np.asarray(power_w, dtype=np.float32)
        if exp == 0.0:
            return p                                       # SMPS 정전력
        if not p.any():
            return p                                       # 꺼진 기기 - 계산할 것이 없다
        k = np.clip(np.asarray(kappa_v, dtype=np.float32), 0.80, 1.20)
        scale = k * k if exp == 2.0 else (k if exp == 1.0 else np.power(k, exp))
        return (p * scale).astype(np.float32)
