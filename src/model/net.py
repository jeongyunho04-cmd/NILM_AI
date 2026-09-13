"""
NILM 2갈래 CNN (설계 문서 2절)
================================
    세밀 (36, 600) --> dilated CNN --+
                                      |--> concat --> trunk --> 기기별 헤드 9개
    광역 (12, 120) --> 작은 CNN   ----+

[2절과 다른 점 — 다중 해상도 타깃 슬라이스]
2.4절은 전역 풀링(Avg+Max)만으로 요약한다. 그런데 이 프로젝트에서 **같은 실패를
세 번 겪었다**:

    12.7절  핫플레이트(GBM): 특징 범위가 릴레이 주기와 겹쳐 순시 통전 여부가 사라짐
            -> F1 0.521. `near`(±10사이클) 범위를 넣자 F1 0.976
    12.8절  미니PC(GBM): 창을 120초로 늘리자 전역 요약이 무관한 과거에 지배됨
            -> F1 0.903 -> 0.769
    12.9절  핫플레이트(CNN): 마지막 conv 의 타깃 슬라이스를 넣었는데도 F1 0.645

세 번째가 특히 교훈적이다. 대책을 넣었는데 **그 대책의 수용영역이 너무 넓었다**:

    conv d=1     7 사이클 (0.12초)
    conv d=2    19 사이클
    conv d=4    43 사이클
    conv d=8    91 사이클
    conv d=16  187 사이클 (3.12초)   <- 마지막 층
    핫플레이트 릴레이 주기 120 사이클 (2.00초)

마지막 층의 타깃 슬라이스는 이미 릴레이 주기의 1.6배를 적분한다. GBM 이 이긴 이유가
정확히 이것으로, 그쪽 1위 특징은 `p_target`(타깃 샘플 그 자체)과 `p_near`(±10사이클)
였다.

그래서 **여러 깊이의 타깃 슬라이스**를 함께 보낸다 — 원본 입력(수용영역 1) +
얕은 층(7·19 사이클) + 깊은 층(187 사이클). GBM 의 target/near/recent/full 다중
범위에 대응하는 구조다.

[상태 헤드 마스킹]
로짓은 5개로 통일하지만 실제 상태 수는 기기마다 다르다 (포트 2 / 에어컨 5).
마스킹하지 않으면 정의되지 않은 클래스로 확률이 새고 CrossEntropy 가 벌하지 않는다.

[잔차 헤드 없음]
3.3절 참조. 닫힌 세계(9종 한정)라 `R̂` 이 표현할 대상이 없다.
"설명 못 한 전력" 은 추론 시 산술로 낸다.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.inputs import (
    FINE_CHANNELS, FINE_CYCLES, LEGACY_FINE_CHANNELS, POWER_SCALE,
    V_CENTER, V_SPAN, WIDE_CHANNELS, fine_target_index, wide_target_index)

MAX_STATES = 5
V_CH_FINE = 25      #: 세밀 갈래의 `(v − V_CENTER)/V_SPAN` 채널 (`inputs.py:378`)

# ⚠⚠ **`--vexp` 를 본선에 켜지 마라 — 979865 에서 실패했다 (14.16).**
#
# 구조는 정확히 작동한다 (관문: 더해진 양 +2.02, 기대 +2.00). 문제는 **헤드가 안 바뀐다**는
# 것이다. 재학습해도 헤드가 대조군과 똑같은 k=0.66 을 배워 합계가 **2.68** 이 됐다 (물리 2.0).
# 실측에서 test_5 가 과예측(−95W)에서 **미예측(+126W)** 으로 넘어가고 다섯 파일이 전부 나빠졌다.
#
# 원인은 α≈1 도 기울기 부족도 아니다 (생성기는 `P ∝ V²` 를 정확히 넣고 상태 안 전력 산포가
# 7.1% 다). **전압 효과가 전부 "상태 안"에 있는데 헤드는 "상태 사이" 100배를 맞히는 데
# 손실이 지배당해 7% 를 잡음으로 취급한다** (14.6: 상태 사이 기울기 1.00 · 상태 안 0.33~0.83).
# 지수를 얹어도 그 습성이 안 사라지고 그냥 위에 더해진다 = 이중 계산.
#
# 코드는 남긴다 — 헤드 쪽(입력 정규화 또는 짝 손실, 14.16 ⑤)이 고쳐지면 그대로 쓴다.


#: 기기별 **전압 지수** `P ∝ V^e` (14.7, 2026-09-13).
#:
#: 왜 필요한가: 모델은 **상태 명목값**을 잘 낸다 — 상태가 바뀔 때(모양이 바뀔 때) 예측이
#: 참값을 기울기 **1.00** 으로 따라간다(오븐 13W~1326W). 그런데 **같은 상태 안**에서는
#: 0.33~0.83 밖에 안 따라간다. 그리고 고정 저항에서 상태 안 전력 변화는 **오직 전압**이다:
#:     합성 캐시에서 잰 `log P ~ e·log V` — 포트 2.01 · 드라이기 2.00/1.99 · 핫플 2.02 ·
#:     오븐 2.00/1.96 (r 0.46~0.91) · 충전기 0.24/−0.42 · 미니PC −0.02/0.05 (r≈0)
#: 그 결과 모델의 **순 전압 지수가 0.85** 다(물리는 2). 저전압에서 과예측한다 — 실측 test_5
#: (210.6V, 드라이기 녹화 227.2V)가 v25 이후 **모든 모델**에서 +2.3~7.9% 과예측이다 (14.6).
#:
#: ⚠ **상태별이 아니라 기기별로 둔다.** 상태마다 재도 같은 기기 안에서는 일치한다
#: (오븐 2.00/1.96 · 드라이기 2.00/1.99 · 선풍기 0.59/0.65/0.67). 상수를 늘릴 이유가 없다.
#: ⚠ 적합값을 그대로 쓰지 않고 **물리로 정리**했다 — 충전기 −0.42 는 정전압 제어에서 나올 수
#: 없는 값이라 적합 잡음이다. 저항 2.0 · SMPS 0.0 · 유도기 0.6 으로 묶는다.
V_EXP: Dict[str, float] = {
    "electiric_kettle": 2.0, "hair_dryer": 2.0, "hotplate": 2.0, "oven": 2.0,   # 순저항
    "beam_projector": 0.0, "laptop_charger": 0.0, "minipc": 0.0,                # SMPS 정전압
    "fan": 0.6, "air_conditioner": 0.6,                                          # 유도기
}
#: 외삽을 막는 상대전압 상하한. 학습 범위가 192~245V 이므로 ±12% 면 충분히 덮는다.
V_REL_CLAMP = (0.88, 1.12)
P_CH_FINE = 23      # 세밀 갈래의 asinh(P/100) 채널 (배치 v2, 13.12. v1 에서는 30 이었다)
P_CH_WIDE = 0       # 광역 갈래의 asinh(P/100) 채널 (1.3절)
WINDOW_STATS = 4    # 헤드에 직접 잇는 원시 창 통계 (아래 forward 참조)

# 주기성 특징 (12.19.4 후보 2). 자기상관을 볼 지연.
#   세밀 60Hz 10초  -> 0.5~5초. 핫플 릴레이 주기 2.0초가 여기 든다
#   광역 2Hz 60초   -> 1~20초.  오븐 히터 펄스 주기가 여기 든다
PERIOD_LAGS_FINE = (30, 45, 60, 90, 120, 180, 240, 300)
PERIOD_LAGS_WIDE = (2, 4, 6, 8, 12, 20, 30, 40)
N_PERIOD = len(PERIOD_LAGS_FINE) + len(PERIOD_LAGS_WIDE) + 2   # + 평균 교차율 2개


def _autocorr(x: torch.Tensor, lags: Sequence[int]) -> torch.Tensor:
    """(B,T) -> (B,len(lags)). 평균 제거 후 정규화 자기상관.

    **왜 필요한가 (12.19절).** 전력도 고조파도 동점인 저항 부하를 가르는 축은
    시간 구조뿐인데(12.15.3), 지금 그 정보가 헤드에 닿는 경로가 없다. 광역 갈래는
    `mean(-1)` 로 뭉개지므로 듀티 50%/주기 2초와 듀티 50%/주기 20초가 구분되지 않는다.

    12.9.13 이 리플 *채널* 을 넣었다가 실패한 것과 다른 점: (i) 여기서는 conv 도
    GroupNorm 도 안 거치고 헤드 직전에 붙는다 — 12.9.8 의 `원시 창통계` 와 같은 경로,
    (ii) 모델이 스스로 주기성을 추출할 필요가 없다. 이미 계산된 값이다.
    """
    x = x - x.mean(-1, keepdim=True)
    denom = (x * x).sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.stack([(x[:, l:] * x[:, :-l]).sum(-1) for l in lags], dim=1) / denom


def _crossing_rate(x: torch.Tensor) -> torch.Tensor:
    """(B,T) -> (B,). 창 평균선을 오르내린 횟수의 비율.

    핫플(주기 2초)은 높고 포트(연속)는 0 이다. 자기상관이 못 잡는 비주기적
    on/off 도 여기서 잡힌다.
    """
    c = (x - x.mean(-1, keepdim=True)) > 0
    return (c[:, 1:] ^ c[:, :-1]).float().mean(-1)

# 기기가 켜져 있는 창에서 **그 기기 자신의 창 최대 전력** 5백분위 (W).
# 12.9.8절 — on 게이트의 물리 프라이어에 쓴다. 1,200창 측정 (2026-08-22).
#
# **p10 이 아니라 p05 를 쓴다.** 오븐은 히터가 꺼져도 팬/조명(16W)이 `is_on=1` 이라
# p10 이 987.5W 로 튄다. 그 값으로 막으면 팬/조명 구간이 통째로 미탐이 된다.
# p05 는 15.5W 라 오븐에서 프라이어가 사실상 꺼진다 - 그것이 옳은 동작이다.
# 에어컨도 송풍(14.5W)이 있어 16.5W 로 낮다. 프라이어가 실제로 무는 기기는
# 핫플레이트(428.8) / 드라이기(443.0) / 전기포트(1156.5) 셋뿐이다.
MIN_ON_W: Dict[str, float] = {
    "electiric_kettle": 1156.5, "hair_dryer": 443.0, "hotplate": 428.8,
    "laptop_charger": 61.3, "beam_projector": 44.9, "fan": 20.3,
    "air_conditioner": 16.5, "oven": 15.5, "minipc": 11.4,
}


def _blk(cin: int, cout: int, k: int, d: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, dilation=d, padding=d * (k - 1) // 2),
        nn.GroupNorm(min(groups, cout), cout),
        nn.GELU(),
    )


class NILMNet(nn.Module):
    """9종 동시 분해. 기기별 헤드는 파라미터를 공유하지 않는다."""

    def __init__(
        self,
        appliances: Sequence[str],
        n_states: Sequence[int],
        width: int = 1,
        dropout: float = 0.1,
        on_bias_init: float = 2.0,
        prior_kappa: float = 0.0,
        prior_beta: float = 0.5,
        wide_summary: bool = False,
        wide_target: bool = False,
        periodicity: bool = False,
        fine_dropout: float = 0.0,
        min_on_w: Optional[Sequence[float]] = None,
        fine_channels: Optional[int] = None,
        aux_z: bool = False,
        #: 상태별 전력 슬롯을 `S_STATE` 의 잰 값에서 출발시킨다 (13.84.68). 학습된
        #: 체크포인트는 `load_state_dict` 가 덮으므로 **영향 없다** — 새 판에만 듣는다.
        #: `False` 로 두면 옛 초기화(전부 0)로 돌아간다.
        state_power_init: bool = True,
        # ── 합 정합성 사영 (계획 A, 14.3) ────────────────────────────────────
        #: 사영 강도 [0,1]. **0 이면 정확히 지금과 같다** (`run_gate_proj.py` [1] 이 확인).
        proj: float = 0.0,
        #: `|r|` 을 관측 전력의 이 비율로 자른다 — 이상치 창이 배분을 독식하는 것을 막는다.
        proj_cap: float = 0.5,
        #: 책임 분모의 하한 (W). **이것이 로버스트 슬랙이다** — 아래 forward 주석 참조.
        proj_floor: float = 5.0,
        vexp: bool = False,
        #: 책임 가중.
        #:   `power`  매개변수를 안 늘린다 (Wisdom et al. 의 에너지 비례). 다만
        #:            **창별 곱셈 재조정과 같다** — 크기만 고치고 배분은 못 옮긴다
        #:            (13.87 [2] 측정. `_project` 독스트링 참조)
        #:   `head`   `z` 에서 책임을 배운다. **배분을 옮길 수 있는 유일한 모드**다.
        #:            ⚠ 새 키를 만든다 — 옛 체크포인트로 지으면 `load_state_dict` 가
        #:            거부한다 (`aux_z` 와 같은 규약).
        proj_resp: str = "power",
        #: 계측계 바닥 잡음 (W). `file_registry` 의 1.4~2.4 중앙. 라벨이 아니라 상수라
        #: **추론 때도 쓸 수 있다** — `L_cons` 가 쓰는 `tgt["p_noise"]` 와 다른 점이다.
        proj_noise_w: float = 1.9,
        # ── 기기 축 어텐션 (13.93) ───────────────────────────────────────────
        #: 토큰 차원. **0 이면 완전히 꺼진다** (키 자체가 안 생긴다 — `aux_z` 규약).
        #: 기기 9개를 토큰으로 놓고 자기어텐션을 건다. 시간 축이 아니라 **기기 축**이다:
        #: 시간 문맥은 12.8(120초가 더 나쁨)·12.44 가 세 번 반증했고, 빠진 것은
        #: FHMM 의 **기기 간 결합**이다 (13.84.74 [7] · 13.86).
        appl_attn: int = 0,
        #: 어텐션 머리 수. `appl_attn` 이 이것으로 나눠떨어져야 한다.
        appl_attn_heads: int = 4,
    ):
        super().__init__()
        # 세밀 갈래가 실제로 쓸 채널 수. 입력은 항상 FINE_CHANNELS 개로 오지만
        # **앞에서부터** 이만큼만 쓴다. 12.34 에서 고조파 위상 6채널을 뒤에 붙였고
        # (38 -> 44), 그 전에 학습한 체크포인트를 계속 채점하려면 38 로 잘라야 한다.
        # 새 채널을 뒤에 붙이는 규약 덕분에 슬라이스 한 줄로 끝난다.
        self.fine_channels = int(fine_channels or FINE_CHANNELS)
        if self.fine_channels > FINE_CHANNELS:
            raise ValueError(
                f"fine_channels={self.fine_channels} 인데 build_fine 은 "
                f"{FINE_CHANNELS} 채널만 만든다")
        self.appliances = list(appliances)
        k = len(self.appliances)
        # 물리 프라이어 (12.9.8절). kappa=0 이면 완전히 꺼진다.
        self.prior_kappa = float(prior_kappa)
        self.prior_beta = float(prior_beta)
        # 12.19.4 의 후보 1 / 2. 서로 독립이라 따로 켜서 귀속한다.
        self.wide_summary = bool(wide_summary)     # 광역에도 amax + 창끝 슬라이스
        # 광역에 **타깃 블록 슬라이스**를 준다 (13.44). seq2point 인데 광역에는
        # 타깃 포인터가 없었다 — `wide_target_index` 주석에 근거가 있다.
        self.wide_target = bool(wide_target)
        self.periodicity = bool(periodicity)       # 자기상관 + 교차율
        # **갈래 드롭아웃** (12.21절). 학습 중 이 확률로 세밀 갈래 특징을 통째로
        # 가려, 광역만으로도 답할 수 있게 강제한다.
        #
        # 근거: 두 갈래 다 합성에서 A/B 를 선형으로 완벽히 가른다 (AUC 1.0000 /
        # 0.9963). 그런데 **합성에서 학습한 선형 probe 를 실측에 옮기면 세밀은
        # 0.3197 로 뒤집히고 광역은 0.6874 로 옳은 방향을 유지한다.** 세밀이 합성에서
        # 더 강하니 학습이 그쪽으로 몰리고(로짓 기여 7:1), 실측에서 그것이 뒤집히면
        # 백업이 없다. 정보는 광역에 있는데 쓰는 법을 안 배운 것이다.
        self.fine_dropout = float(fine_dropout)
        w_on = ([MIN_ON_W.get(a, 0.0) for a in self.appliances]
                if min_on_w is None else list(min_on_w))
        self.register_buffer(
            "on_threshold_asinh",
            torch.asinh(torch.tensor(w_on, dtype=torch.float32) * self.prior_beta / POWER_SCALE),
        )
        self.register_buffer(
            "state_mask",
            torch.tensor([[1.0 if s < n else 0.0 for s in range(MAX_STATES)]
                          for n in n_states], dtype=torch.float32),
        )
        self.target_pos = fine_target_index()

        c1, c2 = int(64 * width), int(128 * width)
        # Sequential 이 아니라 ModuleList 다. 중간 층의 타깃 슬라이스를 뽑아야 한다.
        self.fine = nn.ModuleList([
            _blk(self.fine_channels, c1, 7, 1), _blk(c1, c1, 7, 2), _blk(c1, c2, 7, 4),
            _blk(c2, c2, 7, 8), _blk(c2, c2, 7, 16),
        ])
        # 타깃 슬라이스를 뽑을 층 (0-based). 0 -> 7사이클, 1 -> 19사이클
        self.tap_layers = (0, 1)
        w1, w2 = int(32 * width), int(64 * width)
        self.wide = nn.Sequential(_blk(WIDE_CHANNELS, w1, 5, 1), _blk(w1, w1, 5, 2), _blk(w1, w2, 5, 4))

        # 전역 평균 + 전역 최대 + 깊은 층 타깃 + 얕은 층 타깃 2개 + 원본 타깃
        # + 광역 평균 + **원시 창 전력 통계 4개**
        trunk_in = c2 * 2 + c2 + (c1 + c1) + self.fine_channels + w2 + WINDOW_STATS
        if self.wide_target:
            trunk_in += w2              # 광역 타깃 블록 (13.44)
        if self.wide_summary:
            trunk_in += w2 * 2          # 광역 amax + 창 끝 슬라이스 (후보 1)
        if self.periodicity:
            trunk_in += N_PERIOD        # 후보 2
        h = int(256 * width)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(),
        )
        # 기기별 헤드 (12.9.10절): power 를 **상태 수만큼** 낸다.
        #   power(5) + state(5) + on(1) + plugged(1) + standby(1) = 13
        #
        # 전력 출력이 1개이던 시절, 한 헤드가 2자릿수 떨어진 두 상태를 동시에
        # 맡아야 했다 (오븐 팬/조명 15W ↔ 히터 1150W). 상태별 손실 척도를 켜자
        # 유효 가중이 208배 벌어져 히터가 68W 로 무너졌다 (12.9.9절 v10).
        # 상태마다 출력을 따로 두면 두 기울기가 **서로 다른 파라미터로** 간다.
        self.n_pow = MAX_STATES
        self.heads = nn.ModuleList([nn.Linear(h, MAX_STATES + MAX_STATES + 3)
                                    for _ in range(k)])
        self.i_state = MAX_STATES              # state 로짓 시작
        self.i_on = 2 * MAX_STATES             # on / plugged / standby
        for hd in self.heads:
            nn.init.zeros_(hd.bias)
            # on 로짓 바이어스를 양수로. 초기에 sigmoid(on)~0.5 면 전력이 절반으로
            # 눌려 수렴이 느려진다 (2.4절).
            with torch.no_grad():
                hd.bias[self.i_on] = on_bias_init
        # ── 상태별 전력 슬롯을 **잰 값 근처에서** 출발시킨다 (13.84.68) ────────
        # 전력은 곱이다: `p_raw = Σ_s mix[s]·p_states[s]`, `p_states = softplus(·)`.
        # 바이어스 0 에서 출발하면 모든 슬롯이 `softplus(0) = 0.69W` 인데 참값은
        # 487~1456W 다. 혼합이 안 고르는 슬롯은 기울기가 `mix[s]·∂L/∂p_raw ≈ 0` 이라
        # 못 오르고, 가중감쇠가 더 밀어내려 **음의 포화**로 간다. 거기서는
        # `∂softplus/∂x ≈ e^x ≈ 0` 이라 나중에 혼합이 그 슬롯을 골라도 **못 살아난다.**
        #
        # 실제로 그렇게 죽었다: 드라이기 약풍 슬롯이 `p_states[1] = 0` 이고, 시퀀스
        # 학습이 `L_state` 로 분류를 정확하게 만들어 `mix[1] = 0.997` 이 되자 전력이
        # **466W -> 3W** 로 무너졌다 (13.84.68). 분류가 좋아져서 전력이 나빠진 것이다.
        #
        # `S_STATE` 는 격리 녹화에서 잰 상태별 정상 전력이다 (`losses.S_STATE`, 규칙 14).
        # `softplus(b) = W` 가 되게 b 를 둔다 — 큰 값에서는 `log(e^W−1) ≈ W` 다.
        # ⚠ **이미 학습된 체크포인트에는 영향이 없다** — `load_state_dict` 가 바이어스를
        #   덮어쓴다. 관문 `src/run_gate_states.py` [1] 이 그것을 확인한다.
        self.state_power_init = bool(state_power_init)
        if self.state_power_init:
            from src.model.losses import S_STATE
            with torch.no_grad():
                for j, a in enumerate(self.appliances):
                    for sid, w in S_STATE.get(a, {}).items():
                        if 0 <= sid < self.n_pow and w > 0 and sid < n_states[j]:
                            self.heads[j].bias[sid] = (
                                float(w) if w > 20.0 else float(np.log(np.expm1(w))))
        # 세밀 유래 차원 표식. **연결 순서를 바꾸지 않고** 마스킹만 한다.
        # 순서를 바꾸면 이전 체크포인트가 뒤섞인 입력을 받는다 (실제로 한 번 겪었다).
        fine_flags: List[int] = (
            [1] * self.fine_channels + [1] * c1 + [1] * c1 + [1] * c2 + [1] * c2 + [1] * c2
            + [0] * w2                                        # 광역 평균
            + ([0] * w2 if self.wide_target else [])           # 광역 타깃 블록 (13.44)
            + ([0] * (w2 * 2) if self.wide_summary else [])   # 광역 amax + 창끝
            + [1, 1, 0, 0]                                    # fp_max, fp_min, wp_max, wp_mean
        )
        if self.periodicity:
            fine_flags += ([1] * len(PERIOD_LAGS_FINE) + [0] * len(PERIOD_LAGS_WIDE)
                           + [1, 0])                          # 교차율 세밀/광역
        assert len(fine_flags) == trunk_in, (len(fine_flags), trunk_in)
        # persistent=False — 옛 체크포인트에 없는 키라 state_dict 호환을 깨면 안 된다.
        self.register_buffer("fine_dim_mask",
                             torch.tensor(fine_flags, dtype=torch.float32),
                             persistent=False)

        # 몸통 -> log(r_grid) 보조 헤드 (13.55). **입력 배치도 trunk_in 도 안 바꾼다** —
        # 파라미터 257개가 z 뒤에 붙을 뿐이라 옛 체크포인트와 모양이 어긋나지 않는다
        # (`aux_z=False` 로 지으면 키 자체가 없다).
        #
        # 왜 필요한가 (13.54): 입력 57채널에서 log Z 를 선형으로 R² 0.935 로 뽑을 수
        # 있는데 몸통 z 256차원에서는 0.661 로 흐려진다. "모른다"도 "알면서 안 쓴다"도
        # 아니고 **뽑다가 잃는다** — 아무도 보존하라고 요구하지 않기 때문이다.
        # 그 사이 같은 창을 Z 만 바꿔 만들면 참 전력은 불변인데 예측이 12.9~51.1W 로
        # 흔들리고 Z=3.0Ω 에서 프로젝터 관문이 0.137 로 무너진다.
        self.aux_z = bool(aux_z)
        if self.aux_z:
            self.z_head = nn.Linear(h, 1)
            nn.init.zeros_(self.z_head.bias)

        # ── 합 정합성 사영 (계획 A, 14.3) ────────────────────────────────────
        # 13.86 이 근거를 크게 키웠다: CO-P 가 **잔차 1.2W** 를 내므로 이산 상태공간
        # 안에 총전력을 맞추는 해가 **실재한다**. 우리 40.7~69.5W 는 물리 한계가 아니다.
        #
        # 12.12.2 가 `L_cons` 를 **벌점**으로 0.05 걸었다가 붕괴했다 (포트 편향 −879W,
        # F1 0.937 → 0.643). 그때는 벌점이 개별 감독과 **경쟁**했다 — 제일 큰 부하에
        # 몰아주면 벌점이 이긴다. 사영은 경쟁 항이 아니라 **재매개화**다:
        # `L_power` 가 채점하는 대상이 사영된 출력이므로 r 을 엉뚱한 기기에 실으면
        # 곧바로 벌받는다. 그것이 이번에 다른 점이다.
        self.proj = float(proj)
        self.proj_cap = float(proj_cap)
        self.proj_floor = float(proj_floor)
        self.proj_resp = str(proj_resp)
        self.proj_noise_w = float(proj_noise_w)
        if self.proj_resp not in ("power", "head"):
            raise ValueError("proj_resp 는 power 또는 head 다: %r" % proj_resp)
        if self.proj_resp == "head":
            self.proj_head = nn.Linear(h, k)
            nn.init.zeros_(self.proj_head.weight)
            nn.init.zeros_(self.proj_head.bias)

        # ── 기기 축 어텐션 (13.93) ───────────────────────────────────────────
        # ⚠ `attn_out` 을 0 으로 초기화한다 — 그래야 출발점이 **정확히 지금 모델**이고
        #   두 팔의 차이가 "배운 것" 이지 출발점 차이가 아니다 (13.87 [5b] 에서
        #   `proj_resp=head` 를 그렇게 맞추지 않았다가 29.6W 어긋난 적이 있다).
        self.appl_attn = int(appl_attn)
        if self.appl_attn:
            d = self.appl_attn
            if d % int(appl_attn_heads):
                raise ValueError("appl_attn %d 이 머리 %d 로 안 나눠떨어진다"
                                 % (d, appl_attn_heads))
            self.attn_in = nn.Linear(h, d)
            self.attn_tok = nn.Parameter(torch.randn(k, d) * 0.02)   # 기기 정체 토큰
            self.attn = nn.MultiheadAttention(d, int(appl_attn_heads), batch_first=True)
            self.attn_out = nn.Linear(d, h)
            nn.init.zeros_(self.attn_out.weight)
            nn.init.zeros_(self.attn_out.bias)

        # 전력 혼합에서 state 0(OFF_STANDBY)은 뺀다 — 켜진 상태들만 섞어야 한다.
        on_states = self.state_mask.clone()
        on_states[:, 0] = 0.0
        self.register_buffer("power_mix_mask", on_states)
        # ── 전압 지수 (14.7) ────────────────────────────────────────────────
        # `p_raw` 는 **V_CENTER 에서의** 상태 명목값이 되고 전압 의존은 구조가 낸다.
        # ⚠ 꺼지면 지수가 전부 0 이라 `vrel**0 = 1` — 옛 동작과 **비트 동일**이다.
        self.vexp = bool(vexp)
        # ⚠ `persistent=False` — **유도 상수**지 배우는 값이 아니다. state_dict 에 넣으면
        #   옛 체크포인트가 "Missing key" 로 안 실린다.
        self.register_buffer("v_exp", torch.tensor(
            [V_EXP.get(a, 0.0) if vexp else 0.0 for a in self.appliances], dtype=torch.float32),
            persistent=False)

    def forward(self, fine: torch.Tensor, wide: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.target_pos
        # 이 체크포인트가 학습된 채널 수만 쓴다 (`self.fine_channels` 주석 참조).
        if fine.shape[1] < self.fine_channels:
            raise ValueError(
                f"세밀 입력 채널이 모자랍니다: {fine.shape[1]} < {self.fine_channels}")
        if fine.shape[1] > self.fine_channels:
            fine = fine[:, :self.fine_channels]
        # 원본 입력의 타깃 샘플. 수용영역 1 - 어떤 conv 로도 뭉갤 수 없는 순시 값이다.
        feats = [fine[:, :, t]]
        h = fine
        for i, blk in enumerate(self.fine):
            h = blk(h)
            if i in self.tap_layers:                # 얕은 층의 타깃 슬라이스
                feats.append(h[:, :, t])
        feats += [h.mean(-1), h.amax(-1), h[:, :, t]]   # 전역 요약 + 깊은 층 타깃
        hw = self.wide(wide)
        feats.append(hw.mean(-1))
        if self.wide_target:
            # **평균에 더한다. 대체하지 않는다** (13.44) — 미니PC 는 60초 문맥이
            # 순시값보다 낫다(0.795 대 0.680). 둘 다여야 0.937 로 최고다.
            feats.append(hw[:, :, wide_target_index(hw.shape[-1])])
        if self.wide_summary:
            # 세밀은 전역평균·전역최대·타깃슬라이스 세 갈래로 오는데 광역은 평균
            # 하나뿐이었다 (12.19.1절). 비대칭을 없앤다.
            feats += [hw.amax(-1), hw[:, :, -1]]

        # 원시 창 전력 통계. **conv 도 GroupNorm 도 거치지 않는다.**
        # 지금까지 헤드가 받는 원시 값은 타깃 시점(fine[:,:,t]) 하나뿐이었고,
        # 창 전체의 최대/최소는 *학습된 특징* 의 amax 로만 있었다(h.amax(-1)).
        # 12.9.8절 측정: 총전력을 1/10 로 줄여도 핫플 on 로짓이 0.09 밖에 안 움직였다.
        fp, wp = fine[:, P_CH_FINE], wide[:, P_CH_WIDE]        # asinh(P/100)
        fp_max, wp_max = fp.amax(-1), wp.amax(-1)
        feats.append(torch.stack([fp_max, fp.amin(-1), wp_max, wp.mean(-1)], dim=1))

        if self.periodicity:
            # 시간 구조를 **직접** 준다 (`_autocorr` 주석). conv 를 안 거친다.
            feats.append(torch.cat([
                _autocorr(fp, PERIOD_LAGS_FINE), _autocorr(wp, PERIOD_LAGS_WIDE),
                _crossing_rate(fp)[:, None], _crossing_rate(wp)[:, None],
            ], dim=1))
        x = torch.cat(feats, dim=1)
        if self.training and self.fine_dropout > 0:
            # 창 단위로 세밀 갈래를 통째로 가린다. 부분 드롭아웃이 아니라 **갈래
            # 전체**여야 광역만으로 답하는 법을 배운다.
            keep = (torch.rand(x.shape[0], 1, device=x.device)
                    >= self.fine_dropout).to(x.dtype)
            x = x * (1.0 - self.fine_dim_mask[None] * (1.0 - keep))
        z = self.trunk(x)

        # ── 기기 축 어텐션 (13.93) ───────────────────────────────────────────
        # 지금까지 기기별 머리 9개가 **같은 z 에서 서로 못 보고** 갈라졌다
        # (`chain.crf_nll`: *"기기 축은 서로 독립이다"*). FHMM 이 구조로 갖고 있는
        # **기기 간 결합**이 우리 구조 어디에도 없다는 것이 13.84.74 [7] 의 결론이고,
        # 13.86 이 그 자리를 다시 가리켰다 — FHMM 이 CO 위에 얹는 것이 그 결합이다.
        #
        # 여기서는 시간 축이 아니라 **기기 축**에 어텐션을 건다. 토큰이 9개뿐이라
        # 자료 요구가 거의 없다 (실측 5.5시간이라 시간 축 트랜스포머는 10절이 막는다).
        # `attn_out` 을 **0 으로 초기화**하므로 출발점이 정확히 지금 모델이다 —
        # `chain.ChainHeads.emit` 과 `proj_head` 와 같은 규약이다.
        zk = z[:, None, :].expand(-1, len(self.heads), -1)      # (B,K,H)
        if self.appl_attn:
            tok = self.attn_in(z)[:, None, :] + self.attn_tok[None]   # (B,K,d)
            att, _ = self.attn(tok, tok, tok, need_weights=False)
            zk = zk + self.attn_out(att)                        # 0 초기화면 zk = z
        o = torch.stack([hd(zk[:, j]) for j, hd in enumerate(self.heads)], dim=1)
        on_logit = o[..., self.i_on]

        # ── 물리 프라이어 (12.9.8절) ──────────────────────────────────────
        # 기기는 **창 최대 총전력이 자기 최소 ON 전력보다 작으면** 켜져 있을 수 없다.
        # 21.8W 창에 428W 핫플레이트는 물리적으로 불가능하다.
        #
        # 타깃 시점이 아니라 **창 최대**를 쓴다. 핫플레이트는 2초 주기로 끊기고
        # 휴지 구간도 is_on=1 이라(11.1절), 순시 전력으로 막으면 휴지마다 미탐이 난다.
        # 세밀(60Hz 뒤 10초)과 광역(2Hz 전체 60초)의 최대를 함께 본다.
        if self.prior_kappa > 0:
            p_max = torch.maximum(fp_max, wp_max)              # (B,) asinh(P/100)
            gap = p_max[:, None] - self.on_threshold_asinh[None]
            on_logit = on_logit + F.logsigmoid(self.prior_kappa * gap)
        state = o[..., self.i_state:self.i_state + MAX_STATES].masked_fill(
            self.state_mask[None] == 0, -1e4)

        # 상태별 전력을 상태 확률로 섞는다. 켜진 상태들만 대상이라 OFF 는 빠진다.
        p_states = F.softplus(o[..., 0:MAX_STATES])                     # (B,K,S)
        mix = state.masked_fill(self.power_mix_mask[None] == 0, -1e4).softmax(-1)
        p_raw = (mix * p_states).sum(-1)                                 # (B,K)
        if self.vexp:
            # 창의 전압을 세밀 채널에서 되살린다 (`inputs.py` 가 (v−V_CENTER)/V_SPAN 로 넣는다).
            v = fine[:, V_CH_FINE].mean(-1) * V_SPAN + V_CENTER           # (B,)
            vrel = (v / V_CENTER).clamp(*V_REL_CLAMP)[:, None]            # (B,1)
            p_raw = p_raw * vrel.pow(self.v_exp[None])                    # 기기별 지수
        out = {
            # 전력은 on/off 로 게이팅한다. 게이팅이 없으면 꺼진 기기에도 전력이 샌다.
            "power": torch.sigmoid(on_logit) * p_raw,
            "power_raw": p_raw,
            "power_states": p_states,
            # 상태 혼합 (B,K,S). 상태별 지문(13.11)이 이것을 쓴다 — 손실 쪽에서
            # `power_mix_mask` 를 다시 만들지 않게 여기서 그대로 내보낸다.
            "power_mix": mix,
            "state": state,
            "on_logit": on_logit,
            "plugged_logit": o[..., self.i_on + 1],
            "standby": F.softplus(o[..., self.i_on + 2]),
        }
        if self.aux_z:
            out["log_z"] = self.z_head(z).squeeze(-1)      # (B,)
        if self.proj > 0:
            self._project(out, fine, z)
        # 몸통 표현. 사슬 구조(13.84.24)가 방출·전이 머리를 여기에 얹는다.
        # 옛 경로는 이 키를 안 보므로 동작은 그대로다.
        out["z"] = z
        return out

    def _project(self, out: Dict[str, torch.Tensor], fine: torch.Tensor,
                 z: torch.Tensor) -> None:
        """합 정합성 사영 (계획 A, 14.3). `out["power"]` 을 **제자리에서** 갈아 끼운다.

            r      = P_관측 − (Σ P̂ + Σ Ŝ + 잡음바닥)
            w_k    = gate_k·p_k / max(Σ gate·p, proj_floor)      (Σw ≤ 1)
            P̂_k   <- relu(P̂_k + proj · w_k · clip(r))

        [왜 이것이 추론 때도 되나] `P_관측` 을 **입력 채널에서** 되꺼낸다 —
        `fine[:, 23, t]` 가 `asinh(P/100)` 이다 (0.0003W 오차로 복원됨, 13.87 [1]).
        라벨이 아니므로 실측·배포에서도 같은 식이 돈다. `L_cons` 가 쓰는
        `tgt["p_noise"]` 는 캐시 라벨이라 2단계에서 못 쓴다 — 그것이 `cons=0.0` 으로
        박혀 있던 이유 중 하나다.

        ⚠⚠ **`proj_resp="power"` 는 결국 창별 곱셈 재조정이다** (13.87 [2] 에서 측정).
        `w_k ∝ gate_k·p_k` 이고 `P̂_k = gate_k·p_k` 이므로

            P̂_k  <-  P̂_k · (1 + proj·r / Σ_j P̂_j)      — **모든 기기에 같은 배율**

        이다 (창 안 기기 간 배율 표준편차 4e-8 = float32 잡음). 따라서:
          · 순위를 보존한다 -> **신원을 못 바꾼다. 좋게도, 나쁘게도.**
          · `on_logit` 을 안 건드린다 -> **검출을 못 바꾼다.**
          · 고칠 수 있는 것은 **크기뿐**이다.
        학습된 판에 그냥 켰을 때 on/off 0.8520 과 신원 0.9640/0.9630 이 **한 자리도**
        안 움직인 것이 그래서다 — 안전하다는 증거가 아니라 **구조적으로 그럴 수밖에**다.
        배분을 옮기려면 `proj_resp="head"` 여야 한다.

        [로버스트 슬랙은 분모 하한이 한다 — 따로 매개변수를 두지 않는다]
        `Σ w_k = min(1, Σgate·p / proj_floor)` 다. 500W 가 관측되는데 모델이 0W 를
        예측하는 창은 *배분* 문제가 아니라 *검출* 문제이고, 거기에 500W 를 쏟으면
        유령이 된다. `L_on`·`L_power` 가 검출을 따로 벌한다. AFAMAP 의 로버스트 성분이
        이 자리다.
        ⚠ 분모가 `Σ gate·p` 라 **게이트가 낮아도 p 가 크면 많이 받는다** — 관문 [4] 가
          `gate<0.5` 인 기기에 w=0.94 가 실린 것을 찍는다. 곱셈 재조정이므로 그 기기의
          몫이 원래 컸다는 뜻이고 유령을 *새로* 만들지는 않지만, "꺼진 기기는 안 받는다"
          는 **아니다**. `head` 모드만 로그게이트로 그것을 민다.

        [자르기] `|r| ≤ proj_cap·P_관측`. 순방향 모형 오차가 큰 창(13.84.23 이 잰
        SMPS 잔차 17~27%)이 배분을 독식하지 않게 한다.

        ⚠ `off_detach_praw` 와 같이 쓰면 손실 쪽(`losses.py:640`)이 `out["power"]` 을
          다시 만들어 **사영이 지워진다.** `build_loss` 가 그 조합을 거부한다.
        """
        t = self.target_pos
        p_obs = torch.sinh(fine[:, P_CH_FINE, t]) * POWER_SCALE          # (B,)
        gate = torch.sigmoid(out["on_logit"])                            # (B,K)
        recon = out["power"].sum(1) + out["standby"].sum(1) + self.proj_noise_w
        r = p_obs - recon
        cap = self.proj_cap * p_obs.abs().clamp(min=1.0)
        r = torch.maximum(torch.minimum(r, cap), -cap)
        e = gate * out["power_raw"]                                      # 에너지 비례
        tot = e.sum(1, keepdim=True)
        # 슬랙 — 두 모드가 **같은** 것을 쓴다. 안 그러면 팔 사이 차이가 슬랙 차이와 섞인다.
        slack = (tot / tot.clamp(min=self.proj_floor)).clamp(max=1.0)
        if self.proj_resp == "head":
            # ⚠ 기준을 `log(gate·p)` 로 잡는다 — **0 초기화에서 `power` 모드와 정확히
            #   같아지게** 하기 위해서다 (`softmax(log(gate·p)) = gate·p / Σ`).
            #   `log(gate)` 로 잡았더니 출발점이 29.6W 어긋났다. `chain.ChainHeads` 가
            #   `emit` 을 0 으로 두어 "초기점이 정확히 v37" 을 만든 것과 같은 규약이다.
            lg = self.proj_head(z) + torch.log(e.clamp(min=1e-9))
            w = lg.softmax(-1) * slack
        else:
            w = e / tot.clamp(min=self.proj_floor)
        out["proj_r"] = r
        out["proj_w"] = w
        out["power"] = (out["power"] + self.proj * w * r[:, None]).clamp(min=0.0)


def appliance_state_counts(appliances: Sequence[str]) -> List[int]:
    from src.labeling.state_definitions import get_appliance_config
    return [min(len(get_appliance_config(a).states), MAX_STATES) for a in appliances]


def harmonic_signatures(pool, appliances: Sequence[str], n_harm: int = 15) -> np.ndarray:
    """기기별 **와트당 고조파 페이저** (K, n_harm, 2) [Re, Im].

    3.4절의 `sig_i` 다. 세그먼트 풀에서 한 번 계산해 상수로 둔다.
    통전 구간(전력이 그 기기 p90 의 절반 이상)만 써서, 팬/조명 같은 저전력
    부수 상태가 지문을 오염시키지 않게 한다 (0.2절의 오븐 사례).
    """
    sig = np.zeros((len(appliances), n_harm, 2), dtype=np.float32)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        cs, ps = [], []
        for a in acts:
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                cs.append(a.net_harmonics_complex[m])
                ps.append(a.target_power_w[m])
        if not cs:
            continue
        c = np.concatenate(cs); p = np.concatenate(ps)[:, None]
        per_w = c / np.maximum(p, 1e-6)
        sig[j, :, 0] = np.median(np.real(per_w), axis=0)
        sig[j, :, 1] = np.median(np.imag(per_w), axis=0)
    return sig


def harmonic_signatures_by_state(pool, appliances: Sequence[str], n_harm: int = 15,
                                 max_states: int = MAX_STATES, min_cycles: int = 200
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """기기 x **상태**별 와트당 고조파 페이저 (K, S, n_harm, 2) 와 쓸 수 있는지 (K, S) bool.

    `harmonic_signatures` 는 기기당 하나라 **한 기기의 상태들이 고조파 모양이 다르면 못 담는다.**
    드라이기가 그 극단이다 — 약풍은 반파(|I2|/|I1| 0.431), 강풍은 순저항(0.0004)인데
    중앙값이 약풍에 앉아 강풍 창에서 없는 h2 를 1.8A 예측하게 만들었다 (13.11).

    상태별 사이클이 `min_cycles` 미만이면 기기 전체 지문으로 되돌린다 — 표본이 얇은 상태를
    억지로 따로 맞추면 그 상태가 잡음을 배운다.
    """
    base = harmonic_signatures(pool, appliances, n_harm)          # (K,H,2)
    sig = np.repeat(base[:, None], max_states, axis=1)            # (K,S,H,2)
    used = np.zeros((len(appliances), max_states), dtype=bool)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        by: dict = {}
        for a in acts:
            st = getattr(a, "state_id", None)
            if st is None:
                continue
            m0 = a.target_power_w > 1.0
            for s in np.unique(np.asarray(st)[m0]).astype(int):
                if not 0 < s < max_states:
                    continue
                m = m0 & (np.asarray(st) == s)
                if m.any():
                    by.setdefault(s, ([], []))
                    by[s][0].append(a.net_harmonics_complex[m])
                    by[s][1].append(a.target_power_w[m])
        for s, (cs, ps) in by.items():
            c = np.concatenate(cs); p = np.concatenate(ps)[:, None]
            if len(c) < min_cycles:
                continue
            per_w = c / np.maximum(p, 1e-6)
            sig[j, s, :, 0] = np.median(np.real(per_w), axis=0)
            sig[j, s, :, 1] = np.median(np.imag(per_w), axis=0)
            used[j, s] = True
    return sig.astype(np.float32), used


def harmonic_signatures_by_power(pool, appliances: Sequence[str], n_harm: int = 15,
                                 n_bands: int = 3, min_cycles: int = 300
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """기기 x **전력대**별 와트당 페이저의 **보정비** (K,B,n_harm,2) 와 경계 (K,B−1), 쓸 수 있는지 (K,B).

    13.84.32/35 — `L_harm` 은 `Σ_k sig_k·power_k` 라 **전력에 선형**인데, 와트당 고차 함량이
    동작점에 따라 47~109% 변한다 (충전기 h11 이 15~28W 에서 2.31, 63~65W 에서 1.10 mA/W).
    13.84.35 에서 전력 구간별 사전으로 바꾸면 순방향 잔차가 **13~34%** 준다.

    ⚠ **지문을 갈아 끼우지 않고 `비`를 낸다.** `g[k,b,h] = sig_band[k,b,h] / sig[k,h]` (복소).
    그래야 상태별 지문(`sig_state`, 13.11)과 **곱해서 같이** 쓸 수 있고, 표본이 얇은 칸은
    `g = 1` 로 두면 지금 동작과 **정확히 같다** ([[copy-the-inclusion-rule-when-adding-an-axis]]).

    경계는 그 기기 통전 전력의 `n_bands` 분위수다. `harmonic_signatures` 와 **같은 포함 규칙**
    (전력이 p90 의 절반 이상)을 쓴다 — 다른 문턱을 쓰면 축이 조용히 빈다.
    """
    base = harmonic_signatures(pool, appliances, n_harm)               # (K,H,2)
    bc = base[..., 0] + 1j * base[..., 1]                             # (K,H)
    gain = np.zeros((len(appliances), n_bands, n_harm, 2), np.float32)
    gain[..., 0] = 1.0                                                # g = 1 (항등)
    edges = np.zeros((len(appliances), max(n_bands - 1, 1)), np.float32)
    used = np.zeros((len(appliances), n_bands), bool)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        cs, ps = [], []
        for a in acts:
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                cs.append(a.net_harmonics_complex[m]); ps.append(a.target_power_w[m])
        if not cs:
            continue
        c = np.concatenate(cs); p = np.concatenate(ps)
        q = np.quantile(p, np.linspace(0.0, 1.0, n_bands + 1))[1:-1]
        edges[j, :len(q)] = q
        per_w = c / np.maximum(p, 1e-6)[:, None]
        b_idx = np.digitize(p, q)
        for b in range(n_bands):
            m = b_idx == b
            if m.sum() < min_cycles:
                continue
            v = np.median(np.real(per_w[m]), 0) + 1j * np.median(np.imag(per_w[m]), 0)
            g = v / np.where(np.abs(bc[j]) > 1e-12, bc[j], 1.0)
            g = np.where(np.abs(bc[j]) > 1e-12, g, 1.0)
            gain[j, b, :, 0], gain[j, b, :, 1] = np.real(g), np.imag(g)
            used[j, b] = True
    return gain, edges, used


def standby_signatures(pool, appliances: Sequence[str], n_harm: int = 15) -> np.ndarray:
    """기기별 **대기 상태 고조파 페이저** (K, n_harm, 2). 3.4절의 누락 항 ①."""
    sig = np.zeros((len(appliances), n_harm, 2), dtype=np.float32)
    for j, app in enumerate(appliances):
        c = pool.get_standby_profile(app).harmonics_complex
        sig[j, :, 0], sig[j, :, 1] = np.real(c), np.imag(c)
    return sig


def standby_powers(pool, appliances: Sequence[str]) -> np.ndarray:
    """기기별 **측정된 대기 전력** (K,) W. `standby_signatures` 와 같은 출처다.

    2단계의 `L_sb`(13.60)가 자유 대기 헤드를 `idle x 이 값` 으로 묶는 데 쓴다.
    `y_standby_power` 의 규약이 "활성 중이면 0" 이므로 짝이 맞는다.
    """
    out = np.zeros(len(appliances), dtype=np.float32)
    for j, app in enumerate(appliances):
        try:
            out[j] = float(np.mean(pool.get_standby_profile(app).power_w))
        except Exception:
            out[j] = 0.0
    return out


def noise_signature(pool, n_harm: int = 15) -> np.ndarray:
    """계측계 자체 고조파 페이저 (n_harm, 2). 3.4절의 누락 항 ②."""
    refs = list(pool.noise_references.values())
    c = np.mean([r.median_phasor for r in refs], axis=0)
    out = np.zeros((n_harm, 2), dtype=np.float32)
    out[:, 0], out[:, 1] = np.real(c), np.imag(c)
    return out


#: `Q/P` 상수를 만들 수 없는 기기 — 격리 통전 폭/|중앙| 이 이 값을 넘으면 뺀다.
#: `power_ref.REFERENCE_W` 의 0.10 과 같은 성격의 문턱이다. **둘 다 만족해야
#: 채택한다** — 상대 폭만 보면 |Q|≈0 인 기기(포트 중앙 0.002)가 폭주해서 떨어지고,
#: 절대 폭만 보면 인버터(에어컨)가 통과한다.
REACTIVE_SPREAD_MAX = 0.60
REACTIVE_ABS_SPREAD_MAX = 0.30


def reactive_signatures(pool, appliances: Sequence[str]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """기기별 **와트당 무효전력** `Q/P` (K,) 와 쓸 수 있는지 표시 (K,) bool.

    12.133 이 찾은 **두 번째 판별자**다. 저항은 등가저항 `R = V²/P` 가 기기
    고유값이라 `resistive_match` 가 조합을 역산할 수 있는데, SMPS 에는 그런 것이
    없어서 배분이 고조파 하나에만 걸려 있었다. `Q/P` 가 그 자리를 채운다:

        기기               Q/P 폭/|중앙|    (참고) 전력 폭/중앙
        beam_projector       0.114           0.077
        laptop_charger       0.156           0.722   <- 전력으로는 못 쓰는데
        minipc               0.289           1.162   <- Q/P 로는 쓴다

        판별력 d′ (창간 산포로 나눔)      Q/P    고조파(와트당)
        프로젝터 vs 충전기                2.31       0.91
        프로젝터 vs 미니PC                4.64       1.85
        충전기  vs 미니PC                3.11       1.41

    ⚠ **이 `Q` 는 기본파 무효분이 아니다.** `sign(phase)·sqrt(S²−P²)` 로 왜곡분을
      포함한 비유효전력이다(`feature_extractor.py:9`). 원리적으로는 가산이 아닌데
      실측 66창 중 62창(94%)에서 관측 `Q/P` 가 정답 support 의 범위 안이었다
      (12.133). **경험적 근거이지 물리적 근거가 아니다.**

    ⚠ 상수가 없는 기기는 `usable=False` 로 뺀다 — 에어컨(인버터, 폭/중앙 4.30)과
      포트·드라이기(|Q|≈0 이라 비가 폭주)다. 손실에서 그 열은 기여를 0 으로 둔다.
    """
    K, W = len(appliances), 3600
    qp = np.zeros(K, np.float32)
    ok = np.zeros(K, bool)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        vals, by_rec = [], {}
        for a in acts:
            p = np.asarray(a.target_power_w, np.float64)
            q = np.asarray(a.net_power_features, np.float64)[:, 1]
            i = np.flatnonzero(p > max(thr, 1.0))
            # **창 단위로 잰다.** 손실은 창 예측 `P̂` 에 걸리므로 사이클 단위
            # 비(比)가 아니라 창 평균의 비가 맞는 통계다. `REFERENCE_W` 를 만드는
            # `recompute_reference` 와 같은 방식이다 (60초, 겹침 1/4).
            for k in range(0, len(i) - W, W // 4):
                s_ = i[k:k + W]
                if s_[-1] - s_[0] > W * 1.5:      # 구간을 넘어 이어붙인 창은 버린다
                    continue
                pm = float(p[s_].mean())
                if pm > 1.0:
                    r = float(q[s_].mean()) / pm
                    vals.append(r)
                    by_rec.setdefault(a.source_file, []).append(r)
        if len(vals) < 3:
            continue
        v = np.asarray(vals)
        lo, mid, hi = np.percentile(v, [5, 50, 95])
        qp[j] = mid
        tight = ((hi - lo) <= REACTIVE_ABS_SPREAD_MAX
                 and (hi - lo) / max(abs(mid), 1e-9) <= REACTIVE_SPREAD_MAX)
        # ⚠ **녹화 사이에서도 맞아야 한다** (규칙 1). 한 분할 안에서만 좁은 것은
        #   상수가 아니라 그 녹화의 성질이다. 에어컨이 정확히 그렇다 — train 활성화
        #   안에서는 0.674~0.692 (폭 0.018) 인데 전 녹화로는 −2.24~0.69 다.
        #   녹화가 하나뿐이면 교차 검증이 불가능하므로 **채택하지 않는다.**
        meds = [float(np.median(x)) for x in by_rec.values() if len(x) >= 2]
        cross = (len(meds) >= 2
                 and (max(meds) - min(meds)) <= REACTIVE_ABS_SPREAD_MAX
                 and (max(meds) - min(meds)) / max(abs(mid), 1e-9) <= REACTIVE_SPREAD_MAX)
        ok[j] = bool(tight and cross)
    return qp, ok


def noise_reactive(pool) -> float:
    """계측계 자체의 무효전력 (VAR). `noise_signature` 의 Q 판.

    `power_features` 열 1 이 Q 다 (`feature_extractor` 의 [p,q,s,pf,vrms,thd_i]).
    """
    v = [float(np.median(np.asarray(r.power_features, np.float64)[:, 1]))
         for r in pool.noise_references.values()]
    return float(np.mean(v)) if v else 0.0


def harmonic_scales(pool, appliances: Sequence[str], n_harm: int = 15) -> np.ndarray:
    """차수별 정규화 스케일 (n_harm,).

    정규화 없이 |pred - obs| 를 평균하면 **I1 이 전부 지배한다** — 포트 I1 이 5.9A 인데
    I15 는 0.001A 다. 그러면 고조파 제약이 사실상 전력 제약과 같아져,
    3.4절이 노린 '배분을 결정하는 30차원' 이 1차원으로 무너진다.
    판별 정보는 높은 차수에 있으므로(0.2절) 차수마다 같은 무게를 준다.
    """
    mags = []
    for app in appliances:
        for a in pool.appliance_activations.get(app, []):
            mags.append(np.abs(a.net_harmonics_complex))
    if not mags:
        return np.ones(n_harm, dtype=np.float32)
    m = np.median(np.concatenate(mags), axis=0)
    return np.maximum(m, 1e-4).astype(np.float32)
