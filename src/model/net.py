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


def _blk(cin: int, cout: int, k: int, d: int, groups: int = 8,
         pad_mode: str = "zeros") -> nn.Sequential:
    """⚠ `pad_mode="replicate"` 는 **경계값을 늘린다** (14.131).

    기본 `zeros` 는 창 밖을 **0** 으로 채우는데, `asinh` 눈금에서 0 은 *"고조파가 0"*
    이라 실측 창에 없는 값이다. 창을 조각내 태울수록 그 비중이 커진다 —
    block 6(d=64)은 600칸에서 탭의 18.3%, 240칸 조각에서 **45.7%** 가 패딩이다.

    그리고 14.116 의 **진단 개입**이 한 것이 정확히 replicate 다 (미래를 타깃값으로
    덮었다). 학습 처치(`--fine-time-split`, 0 패딩)와 **같은 처치가 아니었다** —
    타깃 위치에서 두 특징이 block 0(RF 7)에서 이미 cos **0.825** 로 갈린다.
    개입은 답을 고쳤고(포트 0.003 -> 0.938) 학습 처치는 ★2 를 악화시켰다(+7.10).
    """
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, dilation=d, padding=d * (k - 1) // 2,
                  padding_mode=pad_mode),
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
        #: 세밀 몸통 conv 의 패딩 방식 (14.131). **`"zeros"` 가 기본이라 비트 동일.**
        #: `"replicate"` 는 창 밖을 **경계값으로** 채운다 — 14.116 의 진단 개입이 한 것이
        #: 정확히 이것이다. 광역에는 안 건다 (14.94 가 그 축을 닫았다).
        fine_pad: str = "zeros",
        #: 머리 배치 (14.130). `"v1"` 이 지금까지의 것이고 **기본이라 비트 동일**이다.
        #:
        #: `"v2"` — 사용자: *"지금 너무 많은 요소가 있어서 너도 나도 모델 구조를 완벽히
        #: 파악을 못하고 있잖아. 기존 구조 백업해 두고 한번 깔끔하게 처음부터 해 보자."*
        #:
        #: **규칙 하나로 머리를 설명한다 — "모든 요약은 타깃 기준 2.0초 격자로 낸다.
        #: 전역 요약은 없다."** 세밀 600 = **정확히 5 x 120** 이고 타깃 239 가 두 번째
        #: 토막의 끝이라 격자 경계가 타깃과 맞는다 (K=4·6 은 안 맞는다 — K=5 가 유일하다).
        #: ```
        #:   v1 (765칸 · 아홉 가지)            v2 (다섯 가지)
        #:   원시타깃57 · 탭0 64 · 탭1 64      [1] 타깃 순간  원시 57 · 얕은탭 64 · 깊은탭 128
        #:   탭4 128 · h.mean 128(불변)        [2] 세밀 5토막 얕은탭(RF 19) mean+amax
        #:   h.amax 128(불변) · 깊은탭 128     [3] 깊은 요약  과거 mean+amax · 미래 mean+amax
        #:   hw.mean 64(불변) · 창통계 4(불변) [4] 광역 4토막 타깃에서 갈라 과거 3 · 미래 1
        #:                                     [5] 원시 5토막 P 채널 max/min
        #: ```
        #: 얕은 탭은 `tap_layers` 의 **두 번째**를 쓴다 (기본 배치에서 층 1, RF **19**).
        #: 14.127 의 선형 탐침에서 파일 간 일반화가 가장 나았다 — 최악 파일 AUC
        #: 원시 0.346 · 깊은 hf 0.496 · **탭1 0.613**.
        #: ⚠ `head_drop` 과 같이 못 쓴다 (덩이 경계가 다르다).
        head_layout: str = "v1",
        #: 머리 입력에서 **빼는 덩이** (14.128). 쉼표로 여러 개. **빈 값이면 비트 동일.**
        #:
        #: 왜 — 사용자: *"너무 많은 정보가 헤드에 덕지덕지 붙어 있는 모양새인데."*
        #: 재 봤더니 맞다. 실측 2753창에서 머리 입력 765칸의
        #: ```
        #:   유효 차원  분산 90%까지 79칸 · 99%까지 339칸 · 참여비 **20.2**
        #:   덩이별 중복 — 그 덩이를 **나머지 전부**로 맞힌 R^2
        #:     과거평균 0.982 · 원시창통계 0.982 · 탭0 0.980 · 탭1 0.976
        #:     원시타깃 0.968 · 과거최대 0.947 · 광역평균 0.929 · 탭4 0.925 · 깊은탭 0.905
        #: ```
        #: **어느 덩이를 빼도 나머지로 90~98% 복원된다.**
        #: ⚠ 다만 **불안정의 원인이라는 증거는 없다** — 같은 팔 3시드의 오븐 on_logit
        #:   상관이 0.945~0.960, 판정 일치 0.94~0.95 로 거의 같은 함수를 배운다.
        #:   그래서 이건 "고침" 이 아니라 **"빼도 되나" 를 묻는 실험**이다.
        #: ⚠⚠ 추론에서 0 으로 죽여 보는 것으로는 못 묻는다 — `asinh(P/100)=0` 같은
        #:   실측에 없는 입력이 되어 **분포 이탈**을 재게 된다 (14.123 에서 당했다).
        #:   빼고 **다시 학습**해야 한다.
        #:
        #: 이름: `rawtgt`(원시 타깃 57) · `tap0/tap1/tap4`(얕은 탭) ·
        #:       `pastmean`/`pastmax`(세밀 구간 풀링) · `wide`(광역 평균) ·
        #:       `rawstat`(원시 창통계 4)
        head_drop: str = "",
        #: 미래 조각을 **몇 토막으로 나눠** 요약할지 (14.122). `--fine-time-split` 전용.
        #: **1 이면 지금과 비트 동일** (토막 하나 = 미래 전체).
        #:
        #: 왜 필요한가 — 14.116 이 머리에 준 미래는 `hf.mean(-1)` 과 `hf.amax(-1)` 둘뿐인데
        #: **둘 다 순서에 불변**이다. "앞으로 6초 안에 큰 게 있다" 는 말하지만 **"언제"**
        #: 는 못 말한다. +0.2초 뒤의 계단과 +5.9초 뒤의 계단이 **글자 그대로 같은 값**이다.
        #: 그래서 "이건 미래다" 는 알려 줬는데 **"미래의 언제냐" 는 여전히 안 알려 줬다.**
        #:
        #: 잰 것 (cnn_tsp 3시드, 실측 5파일):
        #: ```
        #:   머리 첫 층의 차원당 기여 — `hf.amax` 가 **단일 덩이 1위 19.2%**
        #:     (깊은 타깃 탭 14.9% · 과거 전역최대 15.1% 보다 크다)
        #:   미래 덩이 둘을 죽이면 오븐 헛게이트 10.8% -> **23.0%** (두 배)
        #: ```
        #: ⇒ 미래는 **순이득**이다. 없애면 안 되고 **시간 해상도**를 줘야 한다.
        #:
        #: ⚠⚠ **깊은 `hf` 를 토막내는 것은 소용없다.** 처음에 그렇게 짰다가 관문 [4] 가
        #: 잡았다 — 미래 몸통의 수용영역이 **763** 인데 미래는 360칸뿐이라 `hf` 의 모든
        #: 열이 이미 미래 전체를 본다. 앞쪽만 흔들어도 토막 셋이 똑같이 움직였다
        #: (0.397 / 0.404 / 0.360). `--seg-pool` 이 실패한 것과 **같은 이유**다
        #: ([[a-diagnosis-expires-when-the-architecture-changes]]).
        #:
        #: 그래서 **원시 미래 입력**을 토막낸다 — 수용영역이 정의상 **1** 이라 토막이
        #: 반드시 구별된다. 머리는 이미 `fine[:, :, t]`(원시 타깃 순시값)를 받고 있고
        #: 그 덩이가 기여 3위(9.0%)다. 이건 그것의 **미래판**이다.
        #: 깊은 `hf.mean/amax`(미래 전체)는 **그대로 둔다** — 그게 순이득이었으므로.
        #: K=3 이면 2.01초 해상도 · K=6 이면 1.00초. 더하는 차원은 `2·K·57`.
        fine_future_segs: int = 1,
        #: 상태 전력 슬롯의 **상한 배수** (14.121). `p_states[j,s] <= R·S_STATE[a][s]`.
        #: **0 이면 상한이 없어 지금과 비트 동일**이다.
        #:
        #: 왜 필요한가 — 13.84.68 이 막은 것은 **아래로** 죽는 슬롯이었다. 위로 가는
        #: 쪽은 안 막혀 있었고, 실제로 갔다. 9개 체크포인트에서 잰 `max(p_states)/초기값`:
        #: ```
        #:   beam_projector s1  초기 10.0W  ->  **83.6배**   (자리채움 슬롯, p90 4.5W)
        #:   laptop_charger s1  초기 36.4W  ->    4.31배
        #:   minipc         s1  초기 10.7W  ->    2.17배
        #:   그 밖 17개 슬롯                ->    1.52배 이하 (대부분 0.8~0.9배)
        #: ```
        #: 기전은 13.84.68 주석의 거울상이다 — 혼합이 **거의** 안 고르는 슬롯은 기울기가
        #: `mix[s]·∂L/∂p_raw` 로 작지만 **부호가 일정**하고 `softplus` 에 상한이 없어서
        #: 300에포크 동안 쌓인다. 그러고 나면 상태 머리가 6%만 그쪽을 골라도
        #: `p_raw` 가 33W 튄다 (test_4 310~371초에서 빔 `p_raw` 가 **99.2W** 였다).
        #: R 은 위 측정에서 온다 — 3.0 이면 병적인 둘만 걸리고 나머지는 안 건드린다.
        p_state_cap: float = 0.0,
        # ── 합 정합성 사영 (계획 A, 14.3) ────────────────────────────────────
        #: 사영 강도 [0,1]. **0 이면 정확히 지금과 같다** (`run_gate_proj.py` [1] 이 확인).
        proj: float = 0.0,
        #: `|r|` 을 관측 전력의 이 비율로 자른다 — 이상치 창이 배분을 독식하는 것을 막는다.
        proj_cap: float = 0.5,
        #: 책임 분모의 하한 (W). **이것이 로버스트 슬랙이다** — 아래 forward 주석 참조.
        proj_floor: float = 5.0,
        vexp: bool = False,
        #: 14.148 — **게이트 경화.** `power = (σ(on) > τ)·p_raw` 로 낸다. 0 이면 끈다
        #: (= 비트 동일). `--on-power-praw`(14.147B)가 손실에서 게이트를 뺐는데 추론은
        #: 그대로라 **학습·추론 불일치**가 생겼다 — 학습은 `p_raw -> y` 로 배우는데
        #: 출력은 `σ(on)·p_raw` 라 게이트가 0.9 면 10%를 깎는다. 실측: 에어컨 `p_raw/y`
        #: 가 pcap 1.146 -> mdec **1.003** 으로 정직해졌는데 출력은 0.993 -> **0.926**.
        #: **재학습 없이** 체크포인트에 이 값만 적어 채점할 수 있다.
        hard_gate: float = 0.0,
        #: 14.150 — ★ **게이트와 전력을 완전히 분해한다.** 지금은 연결점이 여섯이다:
        #:   (1) 값 `power = σ(on)·p_raw`  (2) `∂power/∂p_raw = σ(on)` **흡수 상태**(13.80)
        #:   (3) `∂power/∂on = σ'·p_raw`   (4) 프라이어가 `on_logit` 을 밀어 전력을 민다
        #:   (5) `on_logit`·`state`·`p_states` 가 **같은 머리**에서 나온다 (구조적)
        #:   (6) `L_harm` 도 `out["power"]` 를 쓴다
        #: 14.147 의 A·B 는 (2)(3)을 **참ON 창에서만** 끊었고 (1)(4)는 그대로였다.
        #: ⚠ 곱을 그냥 떼면 안 된다 — `power_mix_mask` 가 state 0 을 빼서 `p_raw` 는
        #:   **구조적으로 "켜졌다면 얼마"** 이고 0 이 될 수 없다. 0W 를 낼 수 있는
        #:   유일한 장치가 게이트다. 그래서 **state 0 을 혼합에 넣고 그 전력을 0 으로**
        #:   둔다 — OFF 일을 **상태 머리**가 맡고 게이트는 순수 검출기가 된다.
        #:   그러면 (1)(2)(3)(4)가 한꺼번에 사라진다. (5)만 남는다 (머리를 쪼개야 없앤다).
        gate_free_power: bool = False,
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
        # ── 구간별 풀링 (14.46) ──────────────────────────────────────────────
        #: 전역 `mean`/`amax` 를 **타깃을 경계로 한 n구간**으로 쪼갠다.
        #: **0 또는 1 이면 창 전체 한 구간 = 지금과 비트 동일**이다 (`pool_segments` 참조).
        #: 세밀·광역·원시 전력 통계 셋 다에 같은 n 을 건다.
        seg_pool: int = 0,
        wide_seg_pool: int = 0,          # 14.88 — 광역에만. 0 이면 seg_pool 을 따른다
        wide_extra_dilations: Optional[Sequence[int]] = None,   # 14.91 — 광역 수용영역
        # -- 세밀 갈래 dilation (14.78) ------------------------------------
        #: 세밀 conv 스택의 dilation. `None` 이면 `(1, 2, 4, 8, 16)` = **지금과 비트 동일**.
        #:
        #: 12.37 이 이미 *"진짜 병목은 수용영역이다"* 라고 적었는데, 그때 처방은 직전 3초
        #: 요약을 **입력 채널로** 미리 계산하는 것이었다 (ch41·42). 그건 **과거**만 다룬다.
        #:
        #: 창은 `타깃 앞 3.98초 | 타깃 | 뒤 6.00초` 인데 깊은 탭의 수용영역은
        #: `1 + 6*(1+2+4+8+16) = 187` 사이클 = **+-1.56초** 뿐이다. 그래서 창의 44%
        #: (타깃 뒤 1.56~6.00초)는 **위치를 모르는** `h.mean`/`h.amax` 로만 머리에 닿는다.
        #: `mean`/`amax` 는 순서 불변이라 "계단이 미래에 있다" 를 표현할 수 없다.
        #:
        #: 14.47 의 반사실이 이것을 겨냥한다 — 미래 6초를 다 지우면 mix 0.137 -> 0.939 인데
        #: **수용영역 밖만** 지워도 **0.941** 이다. 해로운 것은 '미래' 가 아니라 '수용영역 밖'이다.
        #:
        #: `(1, 3, 9, 27, 81)` 이면 `1 + 6*121 = 727` 사이클 = **+-6.06초**로 창을 다 덮는다.
        #: 블록 수가 같아 **파라미터가 안 늘고**, 비율 3 <= 커널-1(6) 이라 틈 없이 덮인다.
        fine_dilations: Optional[Sequence[int]] = None,
        #: 세밀 갈래 **전역 풀링**을 무엇으로 할지 (14.79). `"both"` 가 기본 = **비트 동일**.
        #:
        #: 셋이 서로 다른 이유로 거기 있다:
        #:   `h.mean`  창 평균 문맥. **수용영역이 창을 덮으면 필요 없어진다** — 위치를 아는
        #:             conv 가 어떤 가중평균이든 만들 수 있고, 남기면 **위치를 모르는
        #:             지름길**만 제공한다. 14.47 이 잰 해가 정확히 그 통로에서 나온다.
        #:   `h.amax`  **max 는 conv 가 표현 못 하는 비선형**이다. 핫플은 2초 주기로 끊기고
        #:             휴지 구간도 `is_on=1` 이라(11.1) 순시값만 보면 휴지마다 미탐이 난다.
        #:   `fp.amax/amin` (원시, 아래 따로) conv·GroupNorm 을 안 거친 **날 크기**.
        #:             12.9.8 — 총전력을 1/10 로 줄여도 핫플 on 로짓이 0.09 밖에 안 움직였다.
        #: ⇒ `"amax"` 는 **`mean` 만 뺀다**. `fp.amax/amin` 과 물리 프라이어는 그대로다.
        fine_pool: str = "both",
        #: 세밀 스택 **뒤에 덧붙일** dilation (14.80). `None` 이면 안 붙는다 = **비트 동일**.
        #:
        #: ⚠⚠ 14.78 이 실패한 까닭을 고친 것이다. 거기서는 dilation 을 **교체**해
        #:   (1,2,4,8,16 -> 1,3,9,27,81) 마지막 탭의 RF 를 187 -> 727 로 늘렸는데,
        #:   그 탭이 **유일한 국소 탭**이었다. 실측 (`run_gate_finerf`, 무작위 가중치):
        #:       오프셋   +0      +90
        #:       기본   **2.393**  0.448    <- 타깃에서 5.3배 뾰족
        #:       교체     1.494  **1.959**  <- **어깨가 꼭대기보다 높다**
        #:   국소성을 잃자 저항 신원이 한 시드에서 0.9815 -> **0.9321**, SMPS 신원 0.9518 ->
        #:   0.9219, 판정 줄 0.9133 -> 0.8871 로 넓게 무너졌다.
        #: ⇒ **바꾸지 말고 더한다.** 앞 다섯 블록을 그대로 두고 뒤에 (32, 64) 를 붙인 뒤
        #:   `tap_layers` 에 **4** 를 넣어 옛 마지막 블록(RF 187)의 타깃 슬라이스도 같이 뽑는다.
        #:   그러면 국소 탭(+-1.56초)과 넓은 탭(+-6.36초)이 **둘 다** 특징에 들어간다.
        fine_extra_dilations: Optional[Sequence[int]] = None,
        #: 14.116 — 세밀 몸통을 **타깃에서 둘로** 쪼갠다 (같은 가중치, 파라미터 불변).
        #:   0 이면 옛 경로와 **비트 동일**. `forward` 의 주석이 까닭을 적는다.
        fine_time_split: bool = False,
        #: 타깃 슬라이스를 뽑을 블록 번호 (0-based). `None` 이면 `(0, 1)` = **비트 동일**.
        #: `fine_extra_dilations` 를 쓸 때 **4 를 꼭 넣어라** — 안 넣으면 14.78 과 같은 실수다.
        tap_layers: Optional[Sequence[int]] = None,
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
        dil = tuple(int(x) for x in (fine_dilations or (1, 2, 4, 8, 16)))
        if len(dil) != 5:
            raise ValueError("fine_dilations 는 5개여야 한다 (블록 수가 같아야 "
                             "파라미터가 안 는다): %r" % (dil,))
        self.fine_dilations = dil
        #: 14.116 — 세밀 몸통을 타깃에서 둘로 쪼갠다. **0 이면 옛 경로와 비트 동일**.
        self.fine_time_split = bool(fine_time_split)
        _extra = tuple(int(x) for x in (fine_extra_dilations or ()))
        self.fine_extra_dilations = _extra
        self.fine_pad = str(fine_pad)
        if self.fine_pad not in ("zeros", "replicate"):
            raise ValueError("fine_pad 는 zeros/replicate: %r" % (fine_pad,))
        _pm = self.fine_pad
        _blocks = [_blk(self.fine_channels, c1, 7, dil[0], pad_mode=_pm),
                   _blk(c1, c1, 7, dil[1], pad_mode=_pm),
                   _blk(c1, c2, 7, dil[2], pad_mode=_pm),
                   _blk(c2, c2, 7, dil[3], pad_mode=_pm),
                   _blk(c2, c2, 7, dil[4], pad_mode=_pm)]
        _blocks += [_blk(c2, c2, 7, d, pad_mode=_pm) for d in _extra]   # 14.80 — **더한다**
        self.fine = nn.ModuleList(_blocks)
        #: 블록별 출력 채널 수 (탭 차원을 세는 데 쓴다)
        _chans = [c1, c1, c2, c2, c2] + [c2] * len(_extra)
        # 타깃 슬라이스를 뽑을 층 (0-based). 기본 dilation 에서 0 -> 7사이클, 1 -> 19사이클
        self.tap_layers = tuple(int(x) for x in (tap_layers if tap_layers is not None
                                                 else (0, 1)))
        if any(not (0 <= i < len(_blocks)) for i in self.tap_layers):
            raise ValueError("tap_layers 가 블록 범위를 벗어난다: %r (블록 %d개)"
                             % (self.tap_layers, len(_blocks)))
        self.head_layout = str(head_layout)
        if self.head_layout not in ("v1", "v2"):
            raise ValueError("head_layout 은 v1/v2: %r" % (head_layout,))
        # 14.128 — 머리에서 뺄 덩이. 이름이 틀리면 **조용히 무시하지 않고 터뜨린다.**
        self.head_drop = tuple(sorted(x.strip() for x in str(head_drop or "").split(",")
                                      if x.strip()))
        _ok = {"rawtgt", "pastmean", "pastmax", "wide", "rawstat"} |               {"tap%d" % i for i in self.tap_layers}
        for _d in self.head_drop:
            if _d not in _ok:
                raise ValueError("head_drop 이름이 틀렸다: %r (가능: %s)"
                                 % (_d, ",".join(sorted(_ok))))
        _keep_tap = [i for i in self.tap_layers if ("tap%d" % i) not in self.head_drop]
        _tap_dim = sum(_chans[i] for i in _keep_tap)
        self._keep_tap = tuple(_keep_tap)
        w1, w2 = int(32 * width), int(64 * width)
        # 14.91 — 광역 몸통의 **수용영역**. 기본 (1,2,4) 는 전폭 1+4x7 = 29블록,
        #   즉 타깃에서 **±7초**뿐인데 창은 ±30초다 (**24%**). 머리는 그 ±7초 짜리 유닛을
        #   60초에 걸쳐 평균 내므로 *"지난 20초가 평평했다"* 를 만들 길이 원천적으로 없다.
        #   14.90 이 세밀에서 쟀듯 병을 고치는 증거는 **창 전체의 평평함**이다.
        #   ⇒ 세밀에 통한 처방(`fine_extra_dilations`)을 광역에 **그대로** 옮긴다.
        #   `(8,16)` 이면 전폭 1+4x31 = **125블록 > 창 120** 이라 창 전체를 덮는다.
        #   비어 있으면 블록이 안 생겨 **비트 동일**이다.
        _wextra = tuple(int(x) for x in (wide_extra_dilations or ()))
        self.wide_extra_dilations = _wextra
        self.wide = nn.Sequential(
            *([_blk(WIDE_CHANNELS, w1, 5, 1), _blk(w1, w1, 5, 2), _blk(w1, w2, 5, 4)]
              + [_blk(w2, w2, 5, d) for d in _wextra]))

        # 전역 평균 + 전역 최대 + 깊은 층 타깃 + 얕은 층 타깃 2개 + 원본 타깃
        # + 광역 평균 + **원시 창 전력 통계 4개**
        # 14.46 — 구간 수. 0/1 이면 전체 한 구간이라 아래 식이 옛 값과 **정확히 같다**.
        self.seg_pool = int(seg_pool or 0)
        ns = max(self.seg_pool, 1)
        # 14.88 — **광역에만** 거는 구간 수. 0 이면 `seg_pool` 을 따라가므로 **비트 동일**이다.
        #   왜 따로 두나: 14.87 이 잰 것 — 위치 정보를 담는 몫이 `mean` 22% ·
        #   seg4 67% · **seg8 86%** 로 N 에 단조 증가하는데, 그 정보가 필요한 곳은
        #   **광역뿐**이다 (세밀 창은 과거 3.98초 안에 계단이 있으면 이미 맞힌다).
        #   `seg_pool` 은 세밀·광역·원시전력을 한꺼번에 바꾸고 세밀 쪽은 14.74 에서
        #   2 로 해보고 못 읽었다. 여기만 키우면 파라미터가 `w2 x N` 만 는다.
        self.wide_seg_pool = int(wide_seg_pool or 0)
        wns = max(self.wide_seg_pool or self.seg_pool, 1)
        self.fine_pool = str(fine_pool)
        if self.fine_pool not in ("both", "amax", "mean"):
            raise ValueError("fine_pool 은 both/amax/mean: %r" % (fine_pool,))
        _npool = 2 if self.fine_pool == "both" else 1
        if self.head_layout == "v2":
            if self.head_drop:
                raise ValueError("head_layout=v2 는 head_drop 과 같이 못 쓴다 "
                                 "— 덩이 경계가 다르다")
            #: 세밀 5토막(각 120사이클 = 2.0초). 경계가 타깃 239 직후와 맞는다.
            #: 세밀 5토막. 광역은 **토막내지 않는다** — 14.94 가 광역 축을 닫았다
            #: (손잡이 셋 전부 실패 · 광역을 통째로 지워도 저항 넷 게이트가 0.0~0.2%만
            #: 바뀐다 · *"신호가 없는 갈래에 용량을 주면 해가 여러 개 생긴다"*).
            self.h2_fseg, self.h2_wseg = 5, 1
            #: 얕은 탭 — `tap_layers` 의 두 번째 (기본 배치에서 층 1, RF 19)
            self.h2_tap = int(self.tap_layers[1] if len(self.tap_layers) > 1
                              else self.tap_layers[0])
            _csh = _chans[self.h2_tap]
            self.h2_csh = int(_csh)
        _npool_keep = sum(1 for _p, _n in (("mean", "pastmean"), ("amax", "pastmax"))
                          if self.fine_pool in ("both", _p) and _n not in self.head_drop)
        self._pool_keep = tuple(_p for _p, _n in (("mean", "pastmean"), ("amax", "pastmax"))
                                if self.fine_pool in ("both", _p) and _n not in self.head_drop)
        if self.head_layout == "v2":
            trunk_in = (self.fine_channels + self.h2_csh + c2      # [1] 타깃 순간
                        + 2 * self.h2_fseg * self.h2_csh           # [2] 세밀 5토막
                        + 4 * c2                                   # [3] 깊은 과거/미래
                        + w2                                       # [4] 광역 전역 평균
                        + 2 * self.h2_fseg)                        # [5] 원시 5토막 max/min
        else:
            trunk_in = (c2 * _npool_keep * ns + c2 + _tap_dim
                        + (0 if "rawtgt" in self.head_drop else self.fine_channels)
                        # 14.88 — 원시 창 통계는 **세밀 2개 + 광역 2개**라 각자의 구간
                        #   수를 따른다. `wns == ns` 면 옛 경로와 비트 동일이다.
                        + (0 if "wide" in self.head_drop else w2 * wns)
                        + (0 if "rawstat" in self.head_drop
                           else (WINDOW_STATS // 2) * (ns + wns)))
        # 14.122 — 미래 토막 수. 1 이면 14.116 과 같은 2 덩이라 비트 동일이다.
        self.fine_future_segs = max(1, int(fine_future_segs))
        if self.fine_future_segs > 1 and self.head_layout == "v2":
            raise ValueError("head_layout=v2 는 fine_future_segs 와 같이 못 쓴다 "
                             "— v2 가 이미 미래를 격자로 준다")
        if self.fine_future_segs > 1 and not self.fine_time_split:
            raise ValueError("`fine_future_segs > 1` 은 `fine_time_split` 이 켜져야 한다 "
                             "— 미래 조각 자체가 시간분할에서만 생긴다")
        # ⚠ v2 는 미래 요약을 **자기 배치 안에** 이미 담고 있다 ([3]). 여기서 또 더하면
        #   차원이 어긋난다 (1667 대 1923 으로 처음에 터졌다).
        if self.fine_time_split and self.head_layout != "v2":
            trunk_in += 2 * c2                           # 14.116 — 미래 전체 mean/amax
            if self.fine_future_segs > 1:
                # 14.122 — 원시 미래를 토막낸 mean/amax (수용영역 1 이라 토막이 구별된다)
                trunk_in += 2 * self.fine_future_segs * self.fine_channels
        if self.wide_target:
            trunk_in += w2              # 광역 타깃 블록 (13.44)
        if self.wide_summary:
            trunk_in += w2 * wns + w2   # 광역 amax(구간별) + 창 끝 슬라이스 (후보 1)
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
        if self.head_layout == "v2":
            fine_flags = ([1] * (self.fine_channels + self.h2_csh + c2)
                          + [1] * (2 * self.h2_fseg * self.h2_csh)
                          + [1] * (4 * c2)
                          + [0] * w2                           # 광역 유래
                          + [1] * (2 * self.h2_fseg))
            assert len(fine_flags) == trunk_in, (len(fine_flags), trunk_in)
            self.register_buffer("fine_dim_mask",
                                 torch.tensor(fine_flags, dtype=torch.float32),
                                 persistent=False)
        fine_flags: List[int] = (
            ([] if "rawtgt" in self.head_drop else [1] * self.fine_channels) + [1] * _tap_dim
            # 14.79 — `fine_pool` 이 `both` 면 두 무리, 아니면 한 무리다 (순서는 그대로).
            + [1] * (c2 * _npool_keep * ns) + [1] * c2       # 구간별 풀링 + 깊은 타깃
            # 14.116 — 미래 조각의 mean/amax. **세밀 유래**라 1 이다. `forward` 가
            #   깊은 타깃 바로 뒤에 붙이므로 여기도 같은 자리여야 한다.
            + ([1] * (2 * c2) if self.fine_time_split else [])
            # 14.122 — 원시 미래 토막. **세밀 유래**라 1 이고, `forward` 가 깊은 미래
            #   요약 **바로 뒤**에 붙이므로 여기도 같은 자리여야 한다.
            + ([1] * (2 * self.fine_future_segs * self.fine_channels)
               if (self.fine_time_split and self.fine_future_segs > 1) else [])
            + ([] if "wide" in self.head_drop else [0] * (w2 * wns))   # 광역 평균 (구간별)
            + ([0] * w2 if self.wide_target else [])           # 광역 타깃 블록 (13.44)
            + ([0] * (w2 * wns) + [0] * w2 if self.wide_summary else [])  # 광역 amax + 창끝
            + ([] if "rawstat" in self.head_drop
               else [1, 1] * ns + [0, 0] * wns)  # 구간별 fp(max,min) 그리고 wp(max,mean)
        )
        if self.periodicity:
            fine_flags += ([1] * len(PERIOD_LAGS_FINE) + [0] * len(PERIOD_LAGS_WIDE)
                           + [1, 0])                          # 교차율 세밀/광역
        if self.head_layout != "v2":
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
        # ── 상태 전력 상한 (14.121) ─────────────────────────────────────────
        # ⚠ `persistent=False` — `S_STATE` 와 R 에서 **유도되는 상수**다. state_dict 에
        #   넣으면 옛 체크포인트가 "unexpected key" 로 죽는다.
        self.p_state_cap = float(p_state_cap)
        if self.p_state_cap > 0:
            from src.model.losses import S_STATE
            cap = torch.full((len(self.appliances), self.n_pow), float("inf"))
            for j, a in enumerate(self.appliances):
                for sid, w in S_STATE.get(a, {}).items():
                    if 0 <= sid < self.n_pow and w > 0:
                        cap[j, sid] = self.p_state_cap * float(w)
            self.register_buffer("p_state_cap_w", cap, persistent=False)
        # ── 전압 지수 (14.7) ────────────────────────────────────────────────
        # `p_raw` 는 **V_CENTER 에서의** 상태 명목값이 되고 전압 의존은 구조가 낸다.
        # ⚠ 꺼지면 지수가 전부 0 이라 `vrel**0 = 1` — 옛 동작과 **비트 동일**이다.
        self.vexp = bool(vexp)
        self.hard_gate = float(hard_gate)
        self.gate_free_power = bool(gate_free_power)
        # ⚠ `persistent=False` — **유도 상수**지 배우는 값이 아니다. state_dict 에 넣으면
        #   옛 체크포인트가 "Missing key" 로 안 실린다.
        self.register_buffer("v_exp", torch.tensor(
            [V_EXP.get(a, 0.0) if vexp else 0.0 for a in self.appliances], dtype=torch.float32),
            persistent=False)

    def _feats_v2(self, fine, wide, t):
        """머리 배치 v2 (14.130) — **시간 부호를 주려면 그 조각을 따로 태운다.**

        14.130 이 잰 것: `_block` 은 `Conv1d -> GroupNorm(C, T 전체) -> GELU` 라
        **수용영역과 무관하게** 통계가 창 전체에서 나온다. 그래서 풀링만 토막내면
        토막이 안 갈린다 — 한 토막을 흔들면 다섯 토막이 다 움직였다
        (대각 1.435 대 비대각 0.830, **1.73배**뿐). `--seg-pool`(14.72) ·
        `--fine-future-segs`(14.122) · v2 의 첫 판이 전부 같은 이유로 떨어졌다.
        반대로 `--fine-time-split` 이 통한 까닭도 여기 있다 — 풀링을 쪼갠 게 아니라
        **몸통을 따로 태워 GroupNorm 을 따로 돌린 것**이다 (과거↔미래 Δ가 **0.000**).

        ⇒ 그 처방을 **2조각에서 5조각으로 일반화**한다.
        ```
          세밀 600 = 5 x 120 (타깃 239 가 두 번째 토막의 끝 — K=5 만 경계가 맞는다)
          얕은 스택 (층 0..h2_tap, RF 19 < 120)  토막 **5개를 각각 따로** 태운다
          깊은 스택 (전 층, RF 763)             **과거/미래 둘로만** 태운다
                                                (120칸에 dilation 64 는 거의 패딩이다)
        ```
        ⚠ 광역은 **토막내지 않는다.** 14.94 가 그 축을 닫았다 — 손잡이 셋이 전부
          실패했고 광역을 통째로 지워도 저항 넷 게이트가 0.0~0.2%만 바뀐다.
          규칙의 **명시적 예외**다.
        """
        F, TAP = self.h2_fseg, self.h2_tap
        n = fine.shape[-1]
        segs = [(n * k // F, n if k == F - 1 else n * (k + 1) // F) for k in range(F)]

        # ── 깊은 경로 — 타깃에서 둘로 (GroupNorm 도 따로 돈다) ──────────────
        hp, hf = fine[:, :, :t + 1], fine[:, :, t + 1:]
        sh_t = None
        for i, blk in enumerate(self.fine):
            hp, hf = blk(hp), blk(hf)
            if i == TAP:
                sh_t = hp[:, :, -1]                      # 얕은 탭의 **타깃** 값
        dp_t = hp[:, :, -1]
        self._tap_now = dp_t

        # ── 얕은 경로 — 토막 **5개를 각각 따로** 태운다 (가중치 공유) ───────
        seg_feats = []
        for a, b in segs:
            z = fine[:, :, a:b]
            for i, blk in enumerate(self.fine):
                if i > TAP:
                    break
                z = blk(z)
            seg_feats += [z.mean(-1), z.amax(-1)]

        feats = [fine[:, :, t], sh_t, dp_t]              # [1] 타깃 순간
        feats += seg_feats                               # [2] 세밀 5토막
        feats += [hp.mean(-1), hp.amax(-1), hf.mean(-1), hf.amax(-1)]   # [3] 깊은 과거/미래
        feats.append(self.wide(wide).mean(-1))           # [4] 광역 (14.94 로 닫힌 축)
        fp = fine[:, P_CH_FINE]                          # [5] 원시 5토막
        feats.append(torch.stack([v for a, b in segs
                                  for v in (fp[:, a:b].amax(-1), fp[:, a:b].amin(-1))], 1))
        return feats

    def forward(self, fine: torch.Tensor, wide: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.target_pos
        # 이 체크포인트가 학습된 채널 수만 쓴다 (`self.fine_channels` 주석 참조).
        if fine.shape[1] < self.fine_channels:
            raise ValueError(
                f"세밀 입력 채널이 모자랍니다: {fine.shape[1]} < {self.fine_channels}")
        if fine.shape[1] > self.fine_channels:
            fine = fine[:, :self.fine_channels]
        # ⚠ 14.138 — **분기보다 위**에 둔다. 물리 프라이어(12.9.8)가 이 둘을 쓰는데
        #   v1 분기 안에만 있어서 `--head-layout v2` 가 `UnboundLocalError` 로 죽었다
        #   (984924, 48초). 관문 넷이 못 잡은 까닭은 `NILMNet` 의 `prior_kappa` 기본이
        #   **0.0** 이라(트레이너는 **8.0**) 관문이 지은 물건에서 그 블록이 아예 안
        #   돌았기 때문이다 — "관문은 진짜 객체를 지어야 한다".
        # ⚠⚠ 이 둘은 **창 전체(600 · 120)의 최대**다. 미래 6.0초가 그대로 들어간다.
        #   `--fine-time-split` 은 몸통만 쪼개고 이 줄은 **안 끊는다** (14.138 참조).
        fp, wp = fine[:, P_CH_FINE], wide[:, P_CH_WIDE]            # asinh(P/100)
        fp_max, wp_max = fp.amax(-1), wp.amax(-1)
        # 원본 입력의 타깃 샘플. 수용영역 1 - 어떤 conv 로도 뭉갤 수 없는 순시 값이다.
        if self.head_layout == "v2":
            feats = self._feats_v2(fine, wide, t)
        else:
            feats = [] if "rawtgt" in self.head_drop else [fine[:, :, t]]
            if self.fine_time_split:
                # ── 14.116 — 몸통을 **타깃에서 둘로** (같은 가중치) ─────────────
                #   까닭: `--fine-extra-dilations 32,64` 를 켜면 깊은 탭의 수용영역이
                #   763사이클(타깃 좌우 ±6.36초)이라 **창 전체**를 덮는다. 그래서
                #   `h[:, :, t]` 라는 "지금의 특징" 안에 **미래 6.02초가 이미 섞여**
                #   들어간다. 모델은 그것을 미래라고 부를 길이 없으므로 무시할 수도 없다.
                #   실측 개입으로 확인: 미래를 타깃값으로 덮으면 핫플 헛게이트가
                #   0.878 -> 0.004 로 사라진다 (test_5 260.0초).
                #   고침은 정보를 **버리는 것이 아니라 시간 부호를 붙이는 것**이다 —
                #   과거 조각과 미래 조각을 따로 통과시켜 머리에 **따로** 준다.
                #   ⚠ `--seg-pool` 과 다르다. 저쪽은 **전역 풀링**을 쪼개는데, 지금
                #     새는 길은 탭 **안**이라 풀링을 쪼개도 못 막는다 (14.42 의 처방은
                #     wtap 이 없던 RF 187 짜리 몸통에 맞춘 것이었다).
                hp, hf = fine[:, :, :t + 1], fine[:, :, t + 1:]
                for i, blk in enumerate(self.fine):
                    hp, hf = blk(hp), blk(hf)
                    if i in self._keep_tap:
                        feats.append(hp[:, :, -1])      # 얕은 탭도 **과거만** 본다
                h = hp                                  # 아래 타깃 탭·풀링은 과거 쪽
                self._h_fut = hf
            else:
                h = fine
                for i, blk in enumerate(self.fine):
                    h = blk(h)
                    if i in self._keep_tap:             # 얕은 층의 타깃 슬라이스
                        feats.append(h[:, :, t])
                self._h_fut = None
            # 14.46 — 구간별 요약. `seg_pool` 이 0/1 이면 구간이 창 전체 하나라
            #   `h[:, :, 0:n].mean(-1)` == `h.mean(-1)` 이고 **옛 경로와 비트 동일**이다.
            fsegs = pool_segments(h.shape[-1], min(t, h.shape[-1] - 1),
                                  max(self.seg_pool, 1))
            for _a, _b in fsegs:
                # 14.79 — `both` 면 순서까지 옛 경로 그대로라 **비트 동일**이다.
                if "mean" in self._pool_keep:
                    feats.append(h[:, :, _a:_b].mean(-1))
                if "amax" in self._pool_keep:
                    feats.append(h[:, :, _a:_b].amax(-1))
            _tap = h[:, :, min(t, h.shape[-1] - 1)]         # 깊은 층 타깃
            #: 진단용 — 관문이 "이 탭이 미래를 보나" 를 **실제 forward 경로에서** 잰다.
            #: 학습에는 안 쓴다 (`feats` 에 들어가는 것은 `_tap` 자신이다).
            self._tap_now = _tap
            feats.append(_tap)
            if self.fine_time_split:
                # 미래 조각의 요약 — 정보는 그대로 주되 **과거와 섞지 않는다**
                # 14.122 — 그리고 **언제냐**도 준다. 토막마다 mean/amax 를 따로 낸다.
                #   `K=1` 이면 토막이 미래 전체 하나라 `hf[:, :, 0:n]` == `hf` 이고
                #   순서도 mean, amax 그대로다 -> **비트 동일**이다.
                hf = self._h_fut
                feats.append(hf.mean(-1))
                feats.append(hf.amax(-1))
                # 14.122 — **언제냐**. 깊은 `hf` 는 수용영역 763 이라 토막내도 전부 같이
                #   움직인다 (관문 [4] 가 잡았다). **원시 미래 입력**을 토막낸다 —
                #   수용영역이 정의상 1 이라 +0.2초 계단과 +5.9초 계단이 다른 자리에 간다.
                K = self.fine_future_segs
                if K > 1:
                    raw_f = fine[:, :, t + 1:]
                    nfu = raw_f.shape[-1]
                    for _k in range(K):
                        _a = (nfu * _k) // K
                        _b = nfu if _k == K - 1 else (nfu * (_k + 1)) // K
                        _b = max(_b, _a + 1)
                        feats.append(raw_f[:, :, _a:_b].mean(-1))
                        feats.append(raw_f[:, :, _a:_b].amax(-1))
            hw = self.wide(wide)
            wsegs = pool_segments(hw.shape[-1], wide_target_index(hw.shape[-1]),
                                  max(self.wide_seg_pool or self.seg_pool, 1))
            if "wide" not in self.head_drop:
                for _a, _b in wsegs:
                    feats.append(hw[:, :, _a:_b].mean(-1))
            if self.wide_target:
                # **평균에 더한다. 대체하지 않는다** (13.44) — 미니PC 는 60초 문맥이
                # 순시값보다 낫다(0.795 대 0.680). 둘 다여야 0.937 로 최고다.
                feats.append(hw[:, :, wide_target_index(hw.shape[-1])])
            if self.wide_summary:
                # 세밀은 전역평균·전역최대·타깃슬라이스 세 갈래로 오는데 광역은 평균
                # 하나뿐이었다 (12.19.1절). 비대칭을 없앤다.
                for _a, _b in wsegs:
                    feats.append(hw[:, :, _a:_b].amax(-1))
                feats.append(hw[:, :, -1])

            # 원시 창 전력 통계. **conv 도 GroupNorm 도 거치지 않는다.**
            # 지금까지 헤드가 받는 원시 값은 타깃 시점(fine[:,:,t]) 하나뿐이었고,
            # 창 전체의 최대/최소는 *학습된 특징* 의 amax 로만 있었다(h.amax(-1)).
            # 12.9.8절 측정: 총전력을 1/10 로 줄여도 핫플 on 로짓이 0.09 밖에 안 움직였다.
            # ⚠ `fp`/`wp`/`fp_max`/`wp_max` 는 **분기 위로 올라갔다** (14.138) — v2 에도
            #   물리 프라이어가 필요한데 v1 분기 안에만 있어서 984924 가 죽었다.
            #   머리에 주는 통계만 구간별로 쪼갠다 (14.46).
            _st = []
            for _a, _b in fsegs:
                _st += [fp[:, _a:_b].amax(-1), fp[:, _a:_b].amin(-1)]
            for _a, _b in wsegs:
                _st += [wp[:, _a:_b].amax(-1), wp[:, _a:_b].mean(-1)]
            if "rawstat" not in self.head_drop:
                feats.append(torch.stack(_st, dim=1))

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
        if self.p_state_cap > 0:
            # 상한 밑에서는 **손도 안 댄다** (`minimum` 은 그 아래에서 항등이다).
            # 위에서는 기울기가 0 이라 표류가 거기서 멈춘다 — 되돌리지는 못하지만
            # `p_raw` 가 물리적으로 묶인다. 이것이 사려던 전부다.
            p_states = torch.minimum(p_states, self.p_state_cap_w[None])
        if self.gate_free_power:
            # state 0 을 **혼합에 넣고** 그 전력을 0 으로 둔다 -> `p_raw` 가 0 이 될 수 있다.
            # `state` 는 위에서 이미 무효 상태가 -1e4 로 막혀 있다.
            p_states = p_states * self.power_mix_mask[None]
            mix = state.softmax(-1)
        else:
            mix = state.masked_fill(self.power_mix_mask[None] == 0, -1e4).softmax(-1)
        p_raw = (mix * p_states).sum(-1)                                 # (B,K)
        if self.vexp:
            # 창의 전압을 세밀 채널에서 되살린다 (`inputs.py` 가 (v−V_CENTER)/V_SPAN 로 넣는다).
            v = fine[:, V_CH_FINE].mean(-1) * V_SPAN + V_CENTER           # (B,)
            vrel = (v / V_CENTER).clamp(*V_REL_CLAMP)[:, None]            # (B,1)
            p_raw = p_raw * vrel.pow(self.v_exp[None])                    # 기기별 지수
        # 14.148 — `hard_gate` 가 0 이면 `_g` 가 `sigmoid` 그대로라 **비트 동일**이다.
        _g = torch.sigmoid(on_logit)
        if self.hard_gate > 0:
            _g = (_g > self.hard_gate).to(_g.dtype)
        if self.gate_free_power:
            # ★ 곱을 뗀다. 게이트는 `out["on_logit"]` 로만 남아 검출·채점에 쓰인다.
            _g = torch.ones_like(_g)
        out = {
            # 전력은 on/off 로 게이팅한다. 게이팅이 없으면 꺼진 기기에도 전력이 샌다.
            "power": _g * p_raw,
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


def pool_segments(n_total: int, target: int, n_seg: int):
    """타깃 **직후를 반드시 경계로** 두고 과거/미래를 각각 등분한 구간 목록 (14.46).

    `n_seg <= 1` 이면 창 전체 하나 — 그 경우 `h[:, :, 0:n].mean(-1)` 가
    `h.mean(-1)` 과 **같은 계산**이므로 옛 경로와 **비트 동일**이다.

    왜 필요한가 (14.41~14.42): 머리는 `h.mean(-1)`·`h.amax(-1)` 같은 **전역 요약**을
    받는데 그것은 **과거와 미래를 구별하지 못한다.** 세밀 창 600 중 360(6초)이 타깃보다
    뒤이고, 깊은 탭의 수용영역은 ±93 뿐이라 그 바깥 증거가 머리에 닿는 길이 전역 요약
    하나다. 실제로 **수용영역 밖만** 지워도 오븐 혼합이 0.137 -> 0.941 로 살아났다.
    ⇒ 없애지 말고 **쪼갠다** — `amax` 는 conv 로 표현되지 않는 연산자이고(12.9.8 전례),
      `fp_max` 는 물리 프라이어가 쓴다. 정보를 하나도 안 버리고 **시간 부호만** 붙인다.

        n_seg=2, 600, t=239 -> [(0,240), (240,600)]              과거 / 미래
        n_seg=4             -> [(0,120), (120,240), (240,420), (420,600)]
    """
    n_total = int(n_total); target = int(target); n_seg = int(n_seg)
    if n_seg <= 1:
        return [(0, n_total)]
    cut = min(max(target + 1, 1), n_total - 1)          # 타깃 직후 = 반드시 경계
    n_past = min(max(int(round(n_seg * cut / n_total)), 1), n_seg - 1)
    n_fut = n_seg - n_past
    edges = [int(round(cut * i / n_past)) for i in range(n_past)]
    edges += [cut + int(round((n_total - cut) * i / n_fut)) for i in range(n_fut)]
    edges.append(n_total)
    return [(edges[i], edges[i + 1]) for i in range(n_seg)]


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


def harmonic_signature_vref(pool, appliances: Sequence[str],
                            max_states: int = MAX_STATES, min_cycles: int = 200
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """`sig` 를 **적합한 전압** (K,) 과 (K,S). 14.49 의 고침이 쓴다.

    ⚠⚠ **`harm_sig_vnorm` 이 222V 에 고정돼 있었다.** `sig = median(I_h/P)` 는 각 기기의
    **격리 녹화 전압**에서 잰 값인데, 보정이 `sig × (V/V_CENTER)^e` 로 **모두 222V 에서
    적합했다고 가정**했다. 실제 적합 전압은 기기마다 다르다 (2026-09-14 실측):
    ```
    오븐 210.4V · 핫플 214.4V · 미니PC 218.7V · 드라이 227.3V · 포트 227.7V
    에어컨 227.7V · 선풍 228.8V · 프로젝 228.7V · 충전기 229.0V
    ```
    어긋남은 `(V_CENTER/V_적합)^e` 라는 **기기별 상수**이고, `e=−1`(저항)이면 오븐에서
    `222/210.4 = 1.055` — 지문이 5.5% 크게 잡혀 `L_harm` 이 전력을 그만큼 **덜 요구**한다.
    합성 홀드아웃에서 실제로 그 순서대로 나온다 (참값 기준 `p_states/참`):
    ```
    기기   적합 V   예상 편향   관측 편향
    오븐   210.4V   −5.5%     **−3.0%**
    핫플   214.4V   −3.5%     **−1.5%**
    드라이  227.3V   +2.3%     **+3.0%**
    포트   227.7V   +2.5%     **+1.4%**
    ```
    **부호 4/4 · 순서 4/4.** (`L_power` 가 반대로 당기므로 크기는 절반쯤에서 평형이다.)

    ⚠ **선택 규칙을 `harmonic_signatures`·`harmonic_signatures_by_state` 와 한 글자도
      다르게 쓰지 마라** — 다르면 두 입구가 또 갈린다
      ([[pin-the-two-entry-points-against-each-other]]). 아래는 그 둘을 복사한 것이다.
    """
    vref = np.zeros(len(appliances), dtype=np.float32)
    vref_state = np.zeros((len(appliances), max_states), dtype=np.float32)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)          # `harmonic_signatures` 와 같다
        vs, by = [], {}
        for a in acts:
            pf = np.asarray(a.net_power_features, dtype=np.float64)
            if pf.ndim != 2 or pf.shape[1] <= 4:
                continue
            v = pf[:, 4]
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                vs.append(v[m])
            st = getattr(a, "state_id", None)             # `..._by_state` 와 같다
            if st is None:
                continue
            m0 = a.target_power_w > 1.0
            for s_ in np.unique(np.asarray(st)[m0]).astype(int):
                if not 0 < s_ < max_states:
                    continue
                ms = m0 & (np.asarray(st) == s_)
                if ms.any():
                    by.setdefault(s_, []).append(v[ms])
        if vs:
            vref[j] = float(np.median(np.concatenate(vs)))
        for s_, vv in by.items():
            c = np.concatenate(vv)
            if len(c) >= min_cycles:                      # `..._by_state` 의 문턱과 같다
                vref_state[j, s_] = float(np.median(c))
    # 상태별로 못 채운 칸은 기기 전체 값으로 되돌린다 (`..._by_state` 와 같은 규약)
    for j in range(len(appliances)):
        vref_state[j][vref_state[j] <= 0] = vref[j]
    return vref, vref_state


def harmonic_signature_vhrel(pool, appliances: Sequence[str], n_harm: int = 15,
                             source: str = "conducting") -> np.ndarray:
    """기기별 **녹화의 상대 전압 파형** `v_h_rel = V_h/V_1` (K, n_harm, 2) [Re, Im] (14.56).

    왜: 순저항은 `I_h = V_h/R`, `P = V_1²/R` 이므로
    ```
        sig_h = median(I_h/P) = v_h_rel,h / V_1
    ```
    14.49 의 앵커는 **`1/V_1` 쪽만** 고쳤다. `v_h_rel` 은 **그 녹화 세션 값 그대로 박제**돼
    있고, 세션마다 크게 다르다 — 자리 D 는 vh3/V1 0.6~0.85%, 자리 E 는 2.8~3.25% 로 4배다.
    실측했을 때 (14.55, `run_diag_sigvh`) 실측 고전력 창 대 녹화의 비:
    ```
        오븐   h7 0.81 · h11 0.61      (D 녹화 · D 창인데 **세션**이 다르다)
        포트   h3 **0.22**             (E 녹화 · D 창 — 4.6배 틀렸다)
        드라이 h3 **0.24**              (〃)
    ```
    13.69 가 **생성기**에서 없앤 바로 그 가짜 판별자인데 (`apply_site_distortion`),
    **손실의 `sig` 에는 그 고침이 없었다.**

    ⚠⚠ **생성기와 같은 것을 쓴다** — `vtexture.file_rel(stem)`, 그 녹화의 **단자** 상대
      텍스처다. `apply_site_distortion` 이 `rel_rec` 으로 쓰는 바로 그 객체다. 모델은
      합성으로 학습하므로 손실은 **생성기 규약에 못 박혀야** 한다
      ([[pin-the-two-entry-points-against-each-other]]).
      ⚠ 대안은 `sig` 와 같은 **통전 사이클**의 중앙값인데, 활성화가 전압 고조파를 안 싣고
        있어 같은 사이클 선택을 재현할 수 없다. 둘은 1~15% 다르다 (오븐 h3 0.89 · h11 1.71).
        활성화에 전압 고조파를 실으면 그때 바꿔라 — 그 전까지는 **생성기 쪽이 정답**이다.

    기기당 여러 녹화가 있으면 `sig` 와 같이 **중앙값**으로 묶는다. h1 은 정의상 1+0j 다.
    라이브러리가 비면 전부 `1+0j` 로 두고, 그러면 보정이 **항등**이라 비트 동일이다.
    """
    out = np.zeros((len(appliances), n_harm, 2), dtype=np.float32)
    out[:, 0, 0] = 1.0
    # ── 14.61: **`sig` 와 같은 사이클**에서 낸다 (기본) ────────────────────
    #   14.56 은 `vtexture.file_rel`(파일 전체 중앙값)을 썼는데 `sig` 는 **통전
    #   사이클**에서 적합된다. 둘은 1~15% 다르다 (오븐 h3 0.889 · h11 1.707) —
    #   보정하려는 바로 그 양에서 기준이 11% 어긋난 채 걸었고 14.60 에서 기각됐다.
    #   ⚠ **선택 규칙을 `harmonic_signatures` 와 한 글자도 다르게 쓰지 마라.**
    #     아래는 그것을 복사한 것이다 (`run_gate_vhrel` 이 사이클 수를 대조한다).
    if source == "conducting":
        for j, app in enumerate(appliances):
            acts = pool.appliance_activations.get(app, [])
            if not acts:
                continue
            thr = 0.5 * pool.get_steady_power_w(app)          # `harmonic_signatures` 와 같다
            vs = []
            for a in acts:
                vh = getattr(a, "net_voltage_harmonics_complex", None)
                if vh is None or len(vh) != len(a.target_power_w):
                    continue
                m = a.target_power_w > max(thr, 1.0)           # 〃
                if not m.any():
                    continue
                v = np.asarray(vh[m][:, :n_harm], dtype=np.complex128)
                v1 = np.abs(v[:, 0])
                ok = v1 > 1.0
                if ok.any():
                    vs.append(v[ok] / v1[ok][:, None])
            if not vs:
                continue
            c = np.concatenate(vs)
            r = np.median(c.real, 0) + 1j * np.median(c.imag, 0)
            r[0] = 1.0 + 0j
            out[j, :len(r), 0] = r.real.astype(np.float32)
            out[j, :len(r), 1] = r.imag.astype(np.float32)
        return out
    # ── 옛 경로: 그 녹화의 **파일 전체** 단자 텍스처 (14.56) ───────────────
    try:
        from src.synthesis.vtexture import default_library
        lib = default_library()
    except Exception:
        return out
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        stems = sorted({getattr(a, "source_file", "") for a in acts}) if acts else []
        rels = [lib.file_rel(st) for st in stems if st]
        rels = [r for r in rels if r is not None]
        if not rels:
            continue
        m = np.stack([np.asarray(r, dtype=np.complex128)[:n_harm] for r in rels])
        v = np.median(m.real, 0) + 1j * np.median(m.imag, 0)
        v[0] = 1.0 + 0j
        out[j, :len(v), 0] = v.real.astype(np.float32)
        out[j, :len(v), 1] = v.imag.astype(np.float32)
    return out


def state_power_w(act, measured: bool = False) -> np.ndarray:
    """상태별 지문의 **분모**로 쓸 그 기기 몫의 전력 (14.167).

    `measured=False` 면 라벨 전력 `target_power_w` 다 — 옛 규약 그대로.

    `measured=True` 면 **실측 전력** `net_power_features[:, 0]` 이다. 이것이 필요한
    이유: `annotator` 가 `target_power_w = where(is_on==1, clean_p, 0)` 로 적는데
    오븐의 `on_state_min_id = 2` 라 **FAN_LIGHT(state 1) 는 is_on=0 -> 0W** 가 된다.
    그래서 `target_power_w > 1.0` 표본만 모으면 그 상태가 **한 사이클도 안 남고**,
    지문이 기기 전체(= 히터)로 되돌아간다. `src/model/companion.py` 머리말이 이미
    이 함정을 적어 두었다 ("FAN_LIGHT 은 `target_power_w = 0` 이라 애초에 안 들어간다").

    ⚠ 그런데 합성기는 `carrier_apps` 로 **그 상태를 활성 전력에 넣는다**
    (`synthesizer.py` 의 `act_p = np.where(live, net_p, 0.0)`, 13.40). 캐시 meta 의
    `carrier_apps: ['oven']` 이 그 증거다. 즉 **자료는 FAN_LIGHT 를 켜짐·14.6W 로
    주는데 사전은 그것을 순저항 히터로 설명**하고 있었다 — 오븐 s1 의 h2/h1 은
    0.0667 인데 사전이 준 값은 히터의 0.0019 로 **35배** 어긋난다.

    ⚠⚠ 이 분모를 **아무 상태에나 갈아 끼우면 안 된다.** 라벨 전력으로 이미 맞춘 칸은
    그대로 두어야 지난 판과 비교가 선다. `harmonic_signatures_by_state` 는 라벨을
    먼저 쓰고, **표본이 모자라 비는 칸에만** 이것으로 다시 시도한다.
    """
    if measured:
        return np.asarray(act.net_power_features)[:, 0].astype(np.float64)
    return np.asarray(act.target_power_w).astype(np.float64)


def harmonic_signatures_by_state(pool, appliances: Sequence[str], n_harm: int = 15,
                                 max_states: int = MAX_STATES, min_cycles: int = 200,
                                 measured_fallback: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """기기 x **상태**별 와트당 고조파 페이저 (K, S, n_harm, 2) 와 쓸 수 있는지 (K, S) bool.

    `harmonic_signatures` 는 기기당 하나라 **한 기기의 상태들이 고조파 모양이 다르면 못 담는다.**
    드라이기가 그 극단이다 — 약풍은 반파(|I2|/|I1| 0.431), 강풍은 순저항(0.0004)인데
    중앙값이 약풍에 앉아 강풍 창에서 없는 h2 를 1.8A 예측하게 만들었다 (13.11).

    상태별 사이클이 `min_cycles` 미만이면 기기 전체 지문으로 되돌린다 — 표본이 얇은 상태를
    억지로 따로 맞추면 그 상태가 잡음을 배운다.

    **두 번 훑는다 (14.167).** 먼저 라벨 전력(`target_power_w`)으로, 그래도 비는 칸만
    실측 전력(`net_power_features[:,0]`)으로 다시 — `state_power_w` 의 설명 참조.
    `measured_fallback=False` 면 옛 동작과 **비트 동일**하다.

    맞춘 칸이 어느 분모에서 왔는지는 `harmonic_signatures_by_state.last_source` 에
    (K, S) int8 로 남긴다: 0 못 맞춤 · 1 라벨 전력 · 2 실측 전력.
    """
    base = harmonic_signatures(pool, appliances, n_harm)          # (K,H,2)
    sig = np.repeat(base[:, None], max_states, axis=1)            # (K,S,H,2)
    used = np.zeros((len(appliances), max_states), dtype=bool)
    src = np.zeros((len(appliances), max_states), dtype=np.int8)
    passes = (False, True) if measured_fallback else (False,)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        for measured in passes:
            by: dict = {}
            for a in acts:
                st = getattr(a, "state_id", None)
                if st is None:
                    continue
                pw = state_power_w(a, measured)
                m0 = pw > 1.0
                for s in np.unique(np.asarray(st)[m0]).astype(int):
                    if not 0 < s < max_states or used[j, s]:
                        continue                 # 이미 맞춘 칸은 **안 건드린다**
                    m = m0 & (np.asarray(st) == s)
                    if m.any():
                        by.setdefault(s, ([], []))
                        by[s][0].append(a.net_harmonics_complex[m])
                        by[s][1].append(pw[m])
            for s, (cs, ps) in by.items():
                if used[j, s]:
                    continue
                c = np.concatenate(cs); p = np.concatenate(ps)[:, None]
                if len(c) < min_cycles:
                    continue
                per_w = c / np.maximum(p, 1e-6)
                sig[j, s, :, 0] = np.median(np.real(per_w), axis=0)
                sig[j, s, :, 1] = np.median(np.imag(per_w), axis=0)
                used[j, s] = True
                src[j, s] = 2 if measured else 1
    harmonic_signatures_by_state.last_source = src
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
