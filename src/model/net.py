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
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.inputs import (
    FINE_CHANNELS, FINE_CYCLES, POWER_SCALE, WIDE_CHANNELS, fine_target_index,
)

MAX_STATES = 5
P_CH_FINE = 30      # 세밀 갈래의 asinh(P/100) 채널 (1.2절)
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
        periodicity: bool = False,
        min_on_w: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.appliances = list(appliances)
        k = len(self.appliances)
        # 물리 프라이어 (12.9.8절). kappa=0 이면 완전히 꺼진다.
        self.prior_kappa = float(prior_kappa)
        self.prior_beta = float(prior_beta)
        # 12.19.4 의 후보 1 / 2. 서로 독립이라 따로 켜서 귀속한다.
        self.wide_summary = bool(wide_summary)     # 광역에도 amax + 창끝 슬라이스
        self.periodicity = bool(periodicity)       # 자기상관 + 교차율
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
            _blk(FINE_CHANNELS, c1, 7, 1), _blk(c1, c1, 7, 2), _blk(c1, c2, 7, 4),
            _blk(c2, c2, 7, 8), _blk(c2, c2, 7, 16),
        ])
        # 타깃 슬라이스를 뽑을 층 (0-based). 0 -> 7사이클, 1 -> 19사이클
        self.tap_layers = (0, 1)
        w1, w2 = int(32 * width), int(64 * width)
        self.wide = nn.Sequential(_blk(WIDE_CHANNELS, w1, 5, 1), _blk(w1, w1, 5, 2), _blk(w1, w2, 5, 4))

        # 전역 평균 + 전역 최대 + 깊은 층 타깃 + 얕은 층 타깃 2개 + 원본 타깃
        # + 광역 평균 + **원시 창 전력 통계 4개**
        trunk_in = c2 * 2 + c2 + (c1 + c1) + FINE_CHANNELS + w2 + WINDOW_STATS
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
        # 전력 혼합에서 state 0(OFF_STANDBY)은 뺀다 — 켜진 상태들만 섞어야 한다.
        on_states = self.state_mask.clone()
        on_states[:, 0] = 0.0
        self.register_buffer("power_mix_mask", on_states)

    def forward(self, fine: torch.Tensor, wide: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.target_pos
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
        z = self.trunk(torch.cat(feats, dim=1))

        o = torch.stack([hd(z) for hd in self.heads], dim=1)   # (B, K, 13)
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
        return {
            # 전력은 on/off 로 게이팅한다. 게이팅이 없으면 꺼진 기기에도 전력이 샌다.
            "power": torch.sigmoid(on_logit) * p_raw,
            "power_raw": p_raw,
            "power_states": p_states,
            "state": state,
            "on_logit": on_logit,
            "plugged_logit": o[..., self.i_on + 1],
            "standby": F.softplus(o[..., self.i_on + 2]),
        }


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


def standby_signatures(pool, appliances: Sequence[str], n_harm: int = 15) -> np.ndarray:
    """기기별 **대기 상태 고조파 페이저** (K, n_harm, 2). 3.4절의 누락 항 ①."""
    sig = np.zeros((len(appliances), n_harm, 2), dtype=np.float32)
    for j, app in enumerate(appliances):
        c = pool.get_standby_profile(app).harmonics_complex
        sig[j, :, 0], sig[j, :, 1] = np.real(c), np.imag(c)
    return sig


def noise_signature(pool, n_harm: int = 15) -> np.ndarray:
    """계측계 자체 고조파 페이저 (n_harm, 2). 3.4절의 누락 항 ②."""
    refs = list(pool.noise_references.values())
    c = np.mean([r.median_phasor for r in refs], axis=0)
    out = np.zeros((n_harm, 2), dtype=np.float32)
    out[:, 0], out[:, 1] = np.real(c), np.imag(c)
    return out


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
