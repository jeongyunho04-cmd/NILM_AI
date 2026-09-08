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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
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
    #: 자리 (`file_registry.SITE_OF_STEM` 의 글자). 전압 텍스처를 **그 자리에서만** 뽑게 한다 (13.19).
    #: 빈 문자열이면 자리를 모르는 것(탐색 성분)이라 전 자리에서 고른다.
    site: str = ""
    r_grid_ohm: float = 0.8   # 이 콘센트까지의 배선 저항
    #: 그 집의 상시 배경 부하 범위 W (12.166.4). **배경은 집의 성질이다** —
    #: 실측 '모든 기기 OFF' 창에서 장소 A 4.75~5.33W, 장소 B 2.56W 로 **2배**
    #: 다르다. 12.166 은 이것을 무리와 무관하게 (2.6, 8.3) 에서 뽑았고, 그래서
    #: 장소 B 에서 2.1배 과대였다. 전압·임피던스와 같은 이유로 무리에 묶는다.
    #: `None` 이면 `BACKGROUND_W_RANGE` 기본값 (미측정 콘센트).
    background_w_range: Optional[Tuple[float, float]] = None
    #: 이 콘센트의 **h3 전압 왜곡** — 저항 부하가 보는 값 (①a, 12.185.21).
    #: 순저항은 자기 서명이 없다: `I_h = V_h/R` 이라 정규화 서명이 곧 그 콘센트의 전압이다.
    #: 저항 녹화 11개(장소 A 8 · B 2 · C 1)를 `|I_h| = a_h + d_h·|I_1|` 로 갈라 잰 `d_3`:
    #:     장소 A 0.000   장소 B 0.003   장소 C 0.031      <- 10배
    #: h5 이상은 세 장소가 겹쳐서(0.3~2.2%) 장소축이 서지 않는다 — 계측 경로의 인공물이다.
    #: 그래서 **h3 하나만** 싣는다. 0 이면 아무것도 안 한다.
    #: ⚠ 2026-09-06: 위 수치는 옛 계측기 것이다. 새 무리는 전부 0 (①a 끔) — `OBSERVED_VOLTAGE_CLUSTERS` 주석.
    v_distortion_h3: float = 0.0


# ⚠ 아래 두 문단(221V 회귀, 장소 B 계단)은 **옛 계측기(~2026-09-05) 시절의 측정 기록**이다.
#   그 자료는 폐기됐고 장소 A·B 는 갈 수 없다. 이력으로만 남긴다 — 새 무리는 그 아래.
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
# ── ⚠ 2026-09-06 계측기 교체 — 아래 무리는 **새 계측기 자료 11개**로 다시 잰 것이다 (13.1) ────────
# 옛 무리(221V 장소 A / 227.5V 장소 B / 234.7V 장소 C, h3 왜곡 0/0.003/0.031, 계측 바닥 33.5mA∠164°)는
# 차동 ADC 계측기 자료였다. 그 계측기는 전압 채널에 h3 2.6%·짝수차 2.6% 의 인공물을 얹고 있었고
# (12.185.25) 그 인공물이 "장소 지문" 과 "계측 바닥" 으로 들어가 있었다. 자료는 폐기됐다 — READ_ME_FIRST.md.
#
# 새 계측기 11개 파일 (전부 사용자 자리 = 옛 표기 '장소 C', 2026-09-05 21:26 ~ 09-06 01:20):
#   D  vrms 210.7~219.4   vh3/V1 0.59~0.85% ∠−82~−105°   vh7 0.89~1.28%   vh9 0.50~0.64%
#   E  vrms 227.5~231.0   vh3/V1 2.80~3.25% ∠−113~−117°  vh7 0.08~0.19%   vh9 0.92~1.13%
# 같은 콘센트가 시간대에 따라 이렇게 갈린다 — 옛 문서의 "장소" 축은 이제 **세션(시간대)** 축이다.
# 두 무리로 싣고 미측정 구간은 탐색 성분이 덮는다. 가중치 0.40+0.40+탐색 0.20.
# · r_grid_ohm: 옛 계측기의 계단 측정(장소 C 0.45~0.52Ω, 12.167/12.184)을 그대로 둔다 — ΔV/ΔI 비라
#   절대 전압 오차에 덜 민감하지만, **새 계측기 복합 녹화(test_1 계단)로 다시 재야 한다**.
# · background_w_range: 미측정 (None). 옛 값(장소 A 4~6W / B 2~3.5W)은 옛 복합 녹화의 것이다.
# · v_distortion_h3: 전부 0 — **①a 를 끈다.** 새 계측기 저항 녹화는 그 세션의 진짜 vh3 을 서명 안에
#   이미 담고 있어(포트 |I3|/|I1| 3.23% = vh3/V1 3.25%), 옛 방식대로 d3 를 더하면 이중 계상이다.
#   ①a 를 다시 켜려면 녹화 자체의 vh3 을 빼고 환경의 vh3 을 더하는 **차분**으로 다시 설계해야 한다
#   (npz 에 이제 vh/vhdeg 가 있다). 이 결정의 대가: 합성 저항 서명의 h3 이 녹화 세션 값에 박제된다
#   (D 0.6% 무리 vs E 3.2% 무리가 파일에 따라 섞여 들어간다).
# 선로 저항 (새 계측기, 2026-09-06, 13.4): **두 자리의 Z 가 2.7배 다르다.** 계단 앞뒤 3초 창(가드 0.75초)에서
# 두 창이 모두 조용할 때만 (P 산포 <40W) `Z1 = −ΔV1/Δ|I1|` 을 쟀다.
#   E1 (229V)   포트 16계단 0.422 / 드라이기 15계단 0.439 / test_2 13계단 0.424  ->  **0.42Ω**
#   D1 (216V)   오븐(단독) 50계단 1.138±0.071 / test_1 5계단 1.214±0.017        ->  **1.15Ω**
# 13.3 이 "test_1 은 사건이 겹쳐 못 믿는다(1.23±0.22)" 고 보류하고 0.45 로 뒀던 자리다. 창을 좁히니 test_1 의 산포가
# 0.017 로 떨어지고, **D 의 오븐 단독 녹화**가 독립적으로 1.14 를 준다 — D 의 Z 는 정말 E 의 2.7배다.
# (D 는 전압도 13V 낮다. 임피던스가 큰 회선일수록 부하 시 전압이 더 내려가므로 두 값이 함께 움직이는 것이 물리적으로 맞다.)
OBSERVED_VOLTAGE_CLUSTERS: Tuple[VoltageCluster, ...] = (
    VoltageCluster("siteD_216v", mean_v=216.5, std_v=1.5, weight=0.40, site="D",
                   r_grid_ohm=1.15, background_w_range=None, v_distortion_h3=0.0),
    VoltageCluster("siteE_229v", mean_v=229.5, std_v=1.5, weight=0.40, site="E",
                   r_grid_ohm=0.42, background_w_range=None, v_distortion_h3=0.0),
)

#: 옛 계측기가 저항 부하에 얹던 **덧셈** h3 (33.5mA∠164°, 장소 A 8개 회귀). **새 계측기에는 없다** —
#: 포트·핫플·오븐의 |I_h/I_1| / |V_h/V_1| 이 h3·h5 에서 0.8~1.0 이다 (2026-09-06). 0 으로 둔다. 값의 이력은 git.
METER_H3_FLOOR_A: complex = 0j
#: h3 왜곡의 위상(V1 기준). 옛 장소 C 값은 −129.5°(옛 계측기). 새 계측기에서 잰 vh3 위상은 D −82~−105° /
#: E −113~−117°. ①a 재설계 때 쓸 자리 — 지금은 d3=0 이라 안 쓰인다.
SITE_H3_PHASE_RAD: float = np.radians(-110.0)
#: 탐색 성분의 h3 왜곡 상한. ①a 를 껐으므로 0.
EXPLORATION_H3_MAX: float = 0.0

#: 계단으로 직접 잰 세션별 선로 임피던스. A·B 는 옛 계측기(12.167, 장소 폐기), C_* 는 새 계측기 (13.3·13.4).
#: D 가 E 의 2.7배다 — 무부하 전압(216 vs 229V)과 같은 방향으로 움직인다.
MEASURED_SITE_Z_OHM = {"A": 1.470, "B": 0.907,      # 옛 계측기 시대 (자료 삭제됨)
                       "E1": 0.424,      # 포트 0.422 / 드라이기 0.439 / test_2 0.424
                       "D1": 1.15}       # 오븐 1.138 (50계단) / test_1 1.214 (5계단)
#: ⚠ 장소는 **글자로만** 부른다 (사용자 지시 2026-09-06). 키의 D1/E1 은 (자리, 세션) 이다 —
#: 자리는 안 바뀌지만 Z 와 전압은 세션마다 움직인다. 정본은 `file_registry.SITE_SESSIONS`.

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
    #: 이 콘센트의 상시 배경 부하 범위 W (12.166.4). None 이면 생성기 기본값.
    background_w_range: Optional[Tuple[float, float]] = None
    #: 이 콘센트의 h3 전압 왜곡 (저항 부하가 보는 값). `VoltageCluster.v_distortion_h3` 참조.
    v_distortion_h3: float = 0.0
    #: 이 세션의 **전압 텍스처** — 원시 전압 파형의 stem (①b, 12.185.22).
    #: SMPS 전류를 여기에 반응시킨다. `""` 면 아무것도 안 한다.
    texture_stem: str = ""
    #: 13.2 — 이 세션의 전압 텍스처 (`vtexture.Texture`, 2Hz 녹화의 vh·vhdeg 에서). None 이면 텍스처 델타 없음.
    texture: Optional[object] = None
    texture_id: int = -1
    #: SMPS 별 NTC 상태 R [Ω] — pkl 의 실측 범위에서 창마다 뽑는다 (README_v12 "R 은 상태다").
    r_state: Dict[str, float] = field(default_factory=dict)


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
        # 탐색 성분(무리에 안 속한 20%)의 임피던스 폭.
        # ⚠ 2026-09-06(13.20) 갱신: 옛 값 (0.70, 2.00) 은 **옛 계측기 실측(0.88~1.67Ω)** 에
        #   맞춘 것이라 새 실측의 아래쪽을 못 덮었다 — E 는 0.42Ω 인데 탐색은 0.70 아래를 안 뽑아,
        #   단단한 계통은 무리(80% 중 40%)에서만 나오고 탐색 성분에는 아예 없었다.
        #   새 실측 범위는 D 1.04~1.35 / E 0.42Ω 이므로 위아래로 여유를 둔 (0.30, 2.00) 이다.
        r_grid_range: Tuple[float, float] = (0.30, 2.00),
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
        # 13.2 회로 모델 배선. `texture_library`: None = "auto"(processed_data/npz 에서 첫 사용 때 읽는다),
        # `vtexture.VoltageTextureLibrary`, 또는 False(텍스처·결합 둘 다 끔). 옛 인자 texture_stems/texture_model 은
        # 받되 무시한다 (옛 계측기 원시 stem 텍스처는 폐기됐다).
        texture_library=None,
        use_texture: bool = True,
        use_coupling: bool = True,
        couple_ext: bool = False,
        randomize_r: bool = True,
        texture_stems: Optional[Sequence[str]] = None,
        texture_model=None,
    ):
        self._texture_library = texture_library
        self.use_texture = bool(use_texture) and texture_library is not False
        self.use_coupling = bool(use_coupling) and texture_library is not False
        #: 결합 델타의 Σ 에 비SMPS 전류를 넣는가 (13.45). 지금까지 SMPS 3종만 더해서
        #: 오븐 5.2A·에어컨 h3 1.44A 가 빠져 있었고, D 에서 그 강하가 V_h3 자체보다 크다.
        self.couple_ext = bool(couple_ext)
        self.randomize_r = bool(randomize_r)
        self.texture_stems = ()                 # 옛 API 호환
        self._circuit = None
        # ⚠ 텍스처는 **전용 RNG** 로 뽑는다. 창마다 전역 `np.random` 에서 한 번 더 뽑으면
        # 그 흐름이 밀려 **텍스처를 켜고 끄는 것만으로 기기 선택·시각·증강이 전부 달라진다** —
        # 텍스처의 효과만 보려는 A/B 비교가 오염된다. 씨앗은 전역에서 **생성자에서 한 번만**
        # 받아 `np.random.seed()` 의 재현성을 지키고, `texture_stems` 가 비어도 똑같이 받아서
        # 켜고 끔이 전역 흐름을 한 칸도 옮기지 않게 한다.
        self._tex_rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
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
        base_v, source, cluster_r, cluster_bg, d3, site = self._sample_base_voltage()
        # 실측 콘센트에서 뽑았다면 그 회선의 배선 저항을 함께 쓴다.
        # 전압과 임피던스는 같은 회선의 성질이므로 따로 뽑으면 짝이 어긋난다.
        if cluster_r is not None and self.r_grid_range[0] != self.r_grid_range[1]:
            r = float(np.clip(np.random.normal(cluster_r, cluster_r * 0.15), 0.1, 3.0))
        else:
            r = float(np.random.uniform(*self.r_grid_range))
        x = float(np.random.uniform(*self.x_grid_range))
        # 13.2: 텍스처·R 상태의 난수는 이 창의 (기저 전압, R, X) 에서 **파생**한다. 전역 흐름을 한 칸도 안 쓰고
        # (켜고 끄기가 기기 선택·시각을 안 옮긴다), 워커 수·프로세스와 무관하게 같은 창이면 같은 텍스처다
        # (`test_worker_count_does_not_change_the_generated_training_set`). 생성자에서 뽑는 _tex_rng 는 안 쓴다.
        _key = np.array([base_v, r, x], dtype=np.float64).view(np.uint64)
        _rng = np.random.default_rng(int(np.bitwise_xor.reduce(_key) & np.uint64(0x7FFFFFFF)))
        tex = self._sample_texture(base_v, _rng, site)
        r_state: Dict[str, float] = {}
        if tex is not None and self.randomize_r:
            from src.synthesis.coupling import SMPS_DEVICES
            for d_ in SMPS_DEVICES:
                r_ = self.circuit.sample_r(d_, _rng)
                if r_ is not None:
                    r_state[d_] = float(r_)
        return VoltageEnvironment(
            base_voltage_v=base_v,
            r_grid_ohm=r,
            x_grid_ohm=x,
            drift_std_v=self.voltage_variation_std,
            drift_tau_s=self.drift_tau_s,
            sag_rate_per_min=self.sag_rate_per_min,
            source=source,
            background_w_range=cluster_bg,
            v_distortion_h3=d3,
            texture_stem=(tex.stem if tex is not None else ""),
            texture=tex,
            texture_id=(tex.id if tex is not None else -1),
            r_state=r_state,
        )

    def _sample_texture(self, base_v: Optional[float] = None, rng: Optional[np.random.Generator] = None,
                        site: str = ""):
        """이번 합성이 놓일 **전압 텍스처** 하나 (13.2). 기저 전압(D 216V / E 229V)에 가까운 세션에서 뽑는다.

        13.19: 측정된 무리에서 온 창이면 `site` 를 넘겨 **그 자리의 텍스처만** 쓴다. 옛 코드는 전압만
        보고 골라서 236V 위 창(탐색 성분의 22%)이 ±4V 안에 후보가 없어 **라이브러리 전체 균등**으로
        떨어졌고, 그 절반이 216V 자리의 텍스처였다.

        텍스처는 **세션**에 묶인다 — D 는 vh3 0.7%, E 는 3.0% 이고 같은 자리도 세션마다 조금씩 움직인다
        (READ_ME_FIRST §2). 2Hz 녹화의 vh·vhdeg 가 전부 텍스처 라이브러리다 (`vtexture`). 전용 RNG 로 뽑는다.
        """
        if not self.use_texture:
            return None
        lib = self.texture_library
        if lib is None or len(lib) == 0:
            return None
        return lib.sample(rng if rng is not None else self._tex_rng, vrms_target=base_v,
                          site=site or None)

    def _sample_base_voltage(
        self,
    ) -> Tuple[float, str, Optional[float], Optional[Tuple[float, float]], float, str]:
        """실측 이봉분포 + 미측정 영역 탐색 성분에서 기저 전압을 뽑는다.

        Returns:
            (기저 전압, 출처 이름, 배선 저항 or None, 배경 부하 범위 or None, h3 왜곡, 자리)
            자리는 측정된 무리에서 왔을 때만 글자다 — 탐색 성분은 "" (미측정 콘센트).
        """
        cluster_weight = sum(c.weight for c in self.voltage_clusters)
        total = cluster_weight + self.exploration_weight
        if total <= 0:
            return self.default_ref_voltage, "default", None, None, 0.0, ""

        r = np.random.rand() * total
        acc = 0.0
        for c in self.voltage_clusters:
            acc += c.weight
            if r < acc:
                v = float(np.random.normal(c.mean_v, c.std_v)) if c.std_v > 0 else c.mean_v
                return (float(np.clip(v, *EXPLORATION_VOLTAGE_RANGE)), c.name,
                        c.r_grid_ohm, c.background_w_range, c.v_distortion_h3, c.site)

        # 탐색 성분(미측정 콘센트)은 h3 왜곡도 미측정이다. 실측 세 장소가 0.000~0.031 이므로
        # 그 범위에서 균등하게 뽑는다 — 분포의 모양을 모르니 폭만 맞춘다.
        lo, hi = self.exploration_range
        return (float(np.random.uniform(lo, hi)), "exploration", None, None,
                float(np.random.uniform(0.0, EXPLORATION_H3_MAX)), "")

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

    @property
    def texture_library(self):
        """`vtexture.VoltageTextureLibrary` — None 이면 processed_data/npz 에서 첫 호출 때 읽는다. False 면 None."""
        if self._texture_library is False:
            return None
        if self._texture_library is None:
            from src.synthesis.vtexture import default_library
            self._texture_library = default_library()
        return self._texture_library

    @property
    def circuit(self):
        """`coupling.SmpsCircuit` — v12g 모델 + 델타 캐시. 첫 호출에서 만든다."""
        if self._circuit is None:
            from src.synthesis.coupling import SmpsCircuit
            self._circuit = SmpsCircuit()
        return self._circuit

    def texture_file_id(self, stem: str) -> int:
        """녹화 stem -> 텍스처 라이브러리의 파일 id (델타의 기준). 모르면 −1."""
        lib = self.texture_library if (self.use_texture or self.use_coupling) else None
        return -1 if lib is None else int(lib.file_id(stem))

    def apply_voltage_texture(
        self,
        appliance_type: str,
        harmonics_complex: np.ndarray,   # (N, 15) complex64
        power_w: np.ndarray,             # (N,) float — 이 기기의 교류 입력 전력
        env: VoltageEnvironment,
        rec_ids: Optional[np.ndarray] = None,   # (N,) int — 각 사이클이 어느 녹화(파일 id)에서 왔는가
    ) -> np.ndarray:
        """SMPS 전류를 **이 세션의 전압 텍스처**에 반응시킨다 (13.2, 옛 ①b 의 새 계측기 판).

            I_i(생성) = I_i(녹화 재생) + [ I_sim(p_i, 합성 텍스처·v1) − I_sim(p_i, 녹화 텍스처·v1) ]

        차분이라 모델(v12g)의 공통 편향은 상쇄되고 전압 파형의 차이에 대한 응답만 남는다. 두 항 모두 같은
        v1(기저 전압)에서 계산한다 — V1 크기의 효과는 `apply_cross_appliance_coupling` 의 kappa 몫이다.
        기준 텍스처는 **그 활성화가 녹화된 파일** 의 것이다 (`rec_ids`). 전력 5W 구간 × 녹화 파일마다 한 번만 부른다.

        ⚠ 13.22: 더하는 쪽(합성 텍스처)은 **개방 전압** `tex.source_rel()` 이고 빼는 쪽(녹화 텍스처)은
        **단자** 전압 `file_rel` 이다. 부기가 그렇게 맞는다 —
            f(V_rec_단자) + [f(V_env_개방) − f(V_rec_단자)] + [f(V_env_개방 − Z·I) − f(V_env_개방)] = f(V_win_단자)
        녹화 쪽 자기 강하는 뺄셈에서 상쇄되고, 창의 자기 강하는 결합 델타가 넣는다.
        """
        if harmonics_complex.size == 0 or not self.use_texture or rec_ids is None:
            return harmonics_complex
        tex = getattr(env, "texture", None)
        if tex is None or get_load_class(appliance_type) is not LoadClass.SMPS:
            return harmonics_complex
        circ = self.circuit
        if not circ.has(appliance_type):
            return harmonics_complex          # 회로 파라미터가 없는 SMPS 는 건드리지 않는다
        from src.synthesis.coupling import P_BIN_W
        lib = self.texture_library
        p = np.asarray(power_w, dtype=np.float64)
        rid = np.asarray(rec_ids, dtype=np.int64)
        on = (p > 0.5) & (np.abs(harmonics_complex[:, 0]) > 1e-6) & (rid >= 0)
        if not on.any():
            return harmonics_complex
        out = np.asarray(harmonics_complex, dtype=np.complex64).copy()
        v1 = float(env.base_voltage_v)
        R = (getattr(env, "r_state", None) or {}).get(appliance_type)
        pb = np.round(p / P_BIN_W).astype(np.int64)
        keys = pb * 100000 + rid
        for k in np.unique(keys[on]):
            m = on & (keys == k)
            b_, r_id = int(k // 100000), int(k % 100000)
            rel_rec = lib.file_rel_by_id(r_id)
            d = circ.texture_delta(appliance_type, float(b_ * P_BIN_W), tex.source_rel(), tex.id,
                                   rel_rec, r_id, v1, R)
            if d is not None:
                out[m] += d.astype(np.complex64)
        return out

    def apply_smps_coupling(
        self,
        layers: Dict[str, np.ndarray],   # {기기: (N, 15) complex64}
        powers: Dict[str, np.ndarray],   # {기기: (N,) float — 교류 입력 전력}
        env: VoltageEnvironment,
    ) -> Dict[str, np.ndarray]:
        """SMPS 가 켜진 사이클에 **선로 임피던스 결합** 델타를 더한다 (13.2, FCM/가이드 §6.1).

            V_term = V_src − Z(h)·Σ_i I_i     (고정점 3회, Z = r_grid + j·2π·60·h·L, L = x_grid/(2π·60))
            I_i(생성) += I_i(V_term) − I_i(V_src)

        기기별 전력 5W 구간의 조합을 키로 캐시한다.

        ⚠ 13.22: **SMPS 가 하나여도 돈다.** Σ 에 자기 자신이 들어 있으므로 자기 강하가 여기서 들어간다
        (충전기 65W 의 |I9| 를 D 1.15Ω 에서 −22%, E 0.42Ω 에서 −7% 움직인다 — 캐시 창의 14.4%가
        단독 SMPS 창이다). 13.21 에서 한 번 접었던 것을 되살린 것인데, 그때 막았던 이중 계상은
        `V_src` 를 **개방 전압**(`tex.source_rel()`)으로 바꿔 없앴다.
        """
        if not self.use_coupling:
            return layers
        tex = getattr(env, "texture", None)
        if tex is None:
            return layers
        from src.synthesis.coupling import SMPS_DEVICES, P_BIN_W
        circ = self.circuit
        devs = [d for d in SMPS_DEVICES if d in layers and d in powers and circ.has(d)]
        if not devs:
            return layers
        P = np.stack([np.asarray(powers[d], dtype=np.float64) for d in devs], 1)      # (N, k)
        on = P > 0.5
        rows = np.flatnonzero(on.sum(1) >= 1)
        if not len(rows):
            return layers
        pb = np.where(on, np.round(P / P_BIN_W).astype(np.int64), -1)
        out = {d: np.asarray(layers[d], dtype=np.complex64).copy() for d in devs}
        v1 = float(env.base_voltage_v)
        l_line = float(env.x_grid_ohm) / (2.0 * np.pi * 60.0)
        R = getattr(env, "r_state", None) or {}
        # 13.45: 비SMPS 총전류. 격리 녹화 실측으로 오븐 h3 26mA·포트 207mA·드라이기 116mA·
        # 에어컨 1436mA 가 지금까지 Σ 에서 통째로 빠져 있었다. 묶음 안에서 평균내 넘긴다.
        ext_c = None
        if getattr(self, "couple_ext", False):
            ext_apps = [a for a in layers if a not in devs]
            if ext_apps:
                ext_c = np.zeros(np.asarray(layers[devs[0]]).shape, dtype=np.complex128)
                for a in ext_apps:
                    ext_c += np.asarray(layers[a], dtype=np.complex128)
        keys, inv = np.unique(pb[rows], axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        for ki, key in enumerate(keys):
            m_rows = rows[inv == ki]
            pw = {d: float(key[j] * P_BIN_W) for j, d in enumerate(devs) if key[j] >= 0}
            if not pw:
                continue
            i_ext = None if ext_c is None else ext_c[m_rows].mean(0)
            deltas = circ.coupling_delta(pw, tex.source_rel(), tex.id, v1,
                                         float(env.r_grid_ohm), l_line, R, i_ext=i_ext)
            for d, delta in deltas.items():
                out[d][m_rows] += delta.astype(np.complex64)
        res = dict(layers)
        res.update(out)
        return res

    def apply_site_distortion(
        self,
        appliance_type: str,
        harmonics_complex: np.ndarray,   # (N, 15) complex64
        env: VoltageEnvironment,
    ) -> np.ndarray:
        """저항 부하에 **그 콘센트의 h3 전압 왜곡**을 싣는다 (①a, 12.185.21).

        ⚠ 2026-09-06: 새 계측기 무리는 전부 `v_distortion_h3=0` 이라 **지금은 no-op** 이다. 아래 실측
        ("계측기의 덧셈 바닥" 등)은 옛 계측기 이야기다 — `OBSERVED_VOLTAGE_CLUSTERS` 주석과 13.1.

        순저항은 자기 서명이 없다 — `I_h = V_h/R` 이므로 정규화 서명이 곧 그 콘센트의
        전압이다. 그런데 지금까지 저항 부하는 **장소 A 녹화의 서명을 그대로 재생**했고
        전압 변화는 스칼라 배율 하나로만 들어갔다 (`apply_cross_appliance_coupling`) —
        배율은 모든 차수에 공통이라 `|I3|/|I1|` 이 녹화 당시 값에 박제된다.
        그 대가가 12.184.15(a) 다: 운영점이 장소 C 에서 1590W 포트를 포트로 못 보고
        포트/드라이기/핫플/오븐이 20초마다 나눠 갖는다.

        실측(저항 녹화 11개)이 말하는 것:
          · h3 만 장소축이 선다.  d_3 = 장소 A 0.000 / B 0.003 / **C 0.031** (10배)
          · 장소 A·B 의 h3 은 그 장소의 전압이 아니라 **계측기의 덧셈 바닥**이다
            (33.5mA ∠+164°, |I_1| 이 2.1~6.4A 로 3배 달라도 절대값이 일정하고
             위상이 세 기기에서 1.7° 안에 잠겨 있다)
          · h5 이상은 세 장소가 겹친다 (0.3~2.2%) — 장소축이 안 선다. **싣지 않는다.**

        그래서 h3 하나에 `d_3 · I_1 · e^{jθ}` 를 더한다. 이미 녹화 서명 안에 들어 있는
        계측 바닥은 건드리지 않는다 (그건 어느 장소에서나 같다).
        """
        if harmonics_complex.size == 0 or harmonics_complex.shape[1] < 3:
            return harmonics_complex
        if get_load_class(appliance_type) is not LoadClass.RESISTIVE:
            return harmonics_complex
        d3 = float(getattr(env, "v_distortion_h3", 0.0))
        if d3 <= 0.0:
            return harmonics_complex
        out = np.asarray(harmonics_complex, dtype=np.complex64).copy()
        out[:, 2] += (d3 * np.exp(1j * SITE_H3_PHASE_RAD) * out[:, 0]).astype(np.complex64)
        return out

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
