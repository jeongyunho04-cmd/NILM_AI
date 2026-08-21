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

from src.model.inputs import FINE_CHANNELS, FINE_CYCLES, WIDE_CHANNELS, fine_target_index

MAX_STATES = 5


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
    ):
        super().__init__()
        self.appliances = list(appliances)
        k = len(self.appliances)
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

        # 전역 평균 + 전역 최대 + 깊은 층 타깃 + 얕은 층 타깃 2개 + 원본 타깃 + 광역 평균
        trunk_in = c2 * 2 + c2 + (c1 + c1) + FINE_CHANNELS + w2
        h = int(256 * width)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(),
        )
        # 기기별 헤드: power(1) + state(5) + on(1) + plugged(1) + standby(1) = 9
        self.heads = nn.ModuleList([nn.Linear(h, 9) for _ in range(k)])
        for hd in self.heads:
            nn.init.zeros_(hd.bias)
            # on 로짓 바이어스를 양수로. 초기에 sigmoid(on)~0.5 면 전력이 절반으로
            # 눌려 수렴이 느려진다 (2.4절).
            with torch.no_grad():
                hd.bias[6] = on_bias_init

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
        feats.append(self.wide(wide).mean(-1))
        z = self.trunk(torch.cat(feats, dim=1))

        o = torch.stack([hd(z) for hd in self.heads], dim=1)   # (B, K, 9)
        on_logit = o[..., 6]
        p_raw = F.softplus(o[..., 0])
        state = o[..., 1:6].masked_fill(self.state_mask[None] == 0, -1e4)
        return {
            # 전력은 on/off 로 게이팅한다. 게이팅이 없으면 꺼진 기기에도 전력이 샌다.
            "power": torch.sigmoid(on_logit) * p_raw,
            "power_raw": p_raw,
            "state": state,
            "on_logit": on_logit,
            "plugged_logit": o[..., 7],
            "standby": F.softplus(o[..., 8]),
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
