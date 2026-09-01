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
    WIDE_CHANNELS, fine_target_index,
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
        fine_dropout: float = 0.0,
        min_on_w: Optional[Sequence[float]] = None,
        fine_channels: Optional[int] = None,
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
        # 세밀 유래 차원 표식. **연결 순서를 바꾸지 않고** 마스킹만 한다.
        # 순서를 바꾸면 이전 체크포인트가 뒤섞인 입력을 받는다 (실제로 한 번 겪었다).
        fine_flags: List[int] = (
            [1] * self.fine_channels + [1] * c1 + [1] * c1 + [1] * c2 + [1] * c2 + [1] * c2
            + [0] * w2                                        # 광역 평균
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

        # 전력 혼합에서 state 0(OFF_STANDBY)은 뺀다 — 켜진 상태들만 섞어야 한다.
        on_states = self.state_mask.clone()
        on_states[:, 0] = 0.0
        self.register_buffer("power_mix_mask", on_states)

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
