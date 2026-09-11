# -*- coding: utf-8 -*-
"""기기별 2상태 사슬 — 방출 머리 + 전이 머리 + CRF (13.84.24).

```
지금   입력 -> 2갈래 CNN -> 기기별 머리(켜짐·전력) -> 창마다 독립 판정
바꿈   입력 -> 같은 CNN  -> 머리 둘(방출·전이)     -> 사슬 디코딩 -> 파일 전체 상태 열
```

왜 이렇게 바꾸는지 (측정):
- 정상상태 합에서 미니PC 14W 를 읽으려면 형제 75W 를 빼야 하는데 형제 틀의 오차가 더 크다 (13.84.16).
- 차분에서는 형제가 지워진다 — 합성으로 배운 분류기가 실측 사건 30개에서 미니PC **9/11** (13.84.21).
- 참 전이 시각을 주면 test_4 타임라인 **1.000** (창별 0.740).
- 그런데 **딱딱하게 검출하면 무너진다** (정밀도 0.60 -> 0.472). 그래서 검출하지 않고 매 시각 점수만
  내고 Viterbi 가 정한다. 방출이 사건 사이를 붙잡아 거짓 전이 하나가 뒤를 통째로 뒤집는 것을 막는다.
- **따로 배워 사후에 섞으면 안 된다** — 방출이 과확신이라 로그오즈를 1/30 로 줄여야 전이가 이겼고,
  파일-하나-빼기로 계수를 맞춰도 창별과 동점이었다 (13.84.22). 그래서 **같이** 배운다.

전이 점수의 부호 규약: `sw_on[t]` 은 t 에서 OFF->ON 으로 갈 때 더하는 점수,
`sw_off[t]` 은 ON->OFF 로 갈 때 더하는 점수다. 둘 다 **이득**이므로 벌점은 음수로 들어간다.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


def _local_base(x: torch.Tensor, half: int) -> torch.Tensor:
    """(B,T,K) 의 시간축 국소 평균 (반폭 `half`). 가장자리는 복제로 채운다."""
    B, T, K = x.shape
    w = 2 * half + 1
    xp = F.pad(x.permute(0, 2, 1), (half, half), mode="replicate")
    return F.avg_pool1d(xp, kernel_size=w, stride=1).permute(0, 2, 1)


class ChainHeads(nn.Module):
    """방출(z -> 기기별 로짓) + 전이(z 차분·Δ특징 -> 기기별 켜짐/꺼짐 점수).

    `emit_init` 에 v37 의 기기별 머리 가중을 넣으면 방출이 지금 모델에서 출발한다.
    """

    def __init__(self, z_dim: int, d_dim: int, k: int, hidden: int = 128,
                 dz_lag: int = 5, score_norm: int = 0):
        super().__init__()
        self.k = k
        self.dz_lag = dz_lag
        #: 전이 점수를 **국소 기준선**으로 정규화할 반폭(단계). 0 이면 끈다.
        #: 왜: 13.84.25 에서 전이 머리가 test_2 의 참 켜짐을 **파일 상위 0.0%** 로 맞게 짚는데
        #: 절대값이 −0.75 이고 전환 벌점이 −3.97 이라 못 넘었다. 점수 분포가 파일마다 다르다
        #: (p99 가 test_1 −0.57 · test_2 −2.38). 국소 중앙을 빼면 "주변보다 얼마나 사건 같은가" 가
        #: 되어 눈금이 파일에 안 매인다.
        self.score_norm = int(score_norm)
        # 방출 = base_scale * (v37 게이트 로짓) + emit(z).
        # `emit` 을 0 으로 시작하고 base_scale 을 1 로 두면 **초기점이 정확히 v37** 이다 —
        # 사슬이 손해를 끼치는지 이득을 주는지 같은 자리에서 잰다.
        self.emit = nn.Linear(z_dim, k)
        nn.init.zeros_(self.emit.weight)
        nn.init.zeros_(self.emit.bias)
        self.base_scale = nn.Parameter(torch.ones(k))
        self.tr = nn.Sequential(
            nn.Linear(2 * z_dim + d_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * k),          # [켜짐 k, 꺼짐 k]
        )
        # 전환 기본 벌점 — 상태를 바꾸는 것은 기본적으로 비싸다. 학습으로 조정된다.
        self.switch_bias = nn.Parameter(torch.full((k,), -4.0))
        self.emit_scale = nn.Parameter(torch.ones(k))
        # 첫 상태 머리 (13.84.25 의 구조적 한계). 창이 60초라 **파일 앞머리 한 창은 전이를 못 본다** —
        # test_4 의 미니PC 첫 켜짐 52.4초가 격자 밖이었다. 그 구간의 상태는 전이가 아니라 첫 창의
        # 겉모습으로만 정해지므로 전용 머리를 둔다. 0 으로 시작해 옛 동작(사전 0)과 같다.
        self.init_head = nn.Linear(z_dim, k)
        nn.init.zeros_(self.init_head.weight)
        nn.init.zeros_(self.init_head.bias)

    def forward(self, z: torch.Tensor, d: torch.Tensor,
                base: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """z (B,T,Z) · d (B,T,D) · base (B,T,K) -> 방출, 켜짐, 꺼짐 (각 (B,T,K)), 첫상태 (B,K)."""
        em = self.emit(z) * self.emit_scale[None, None]
        if base is not None:
            em = em + self.base_scale[None, None] * base
        L = self.dz_lag
        zp = F.pad(z.transpose(1, 2), (L, L), mode="replicate").transpose(1, 2)
        dz = zp[:, 2 * L:] - zp[:, :-2 * L]            # z(t+L) − z(t−L), (B,T,Z)
        h = self.tr(torch.cat([z, dz, d], dim=-1))
        on, off = h[..., :self.k], h[..., self.k:]
        if self.score_norm > 0:
            on = on - _local_base(on, self.score_norm)
            off = off - _local_base(off, self.score_norm)
        b = self.switch_bias[None, None]
        return em, on + b, off + b, self.init_head(z[:, 0])


def crf_nll(em: torch.Tensor, sw_on: torch.Tensor, sw_off: torch.Tensor,
            y: torch.Tensor, init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """기기별 2상태 선형 사슬의 음의 로그가능도. 기기 축은 서로 독립이다.

    점수 S(y) = Σ_t em[t]·y[t] + Σ_{t: 0->1} sw_on[t] + Σ_{t: 1->0} sw_off[t]
    분배함수는 앞으로 알고리즘으로 정확히 센다.
    """
    B, T, K = em.shape
    # 정답 경로 점수
    yf = y.float()
    gold = (em * yf).sum(1)
    if init is not None:
        gold = gold + init * yf[:, 0]
    prev, cur = yf[:, :-1], yf[:, 1:]
    gold = gold + (sw_on[:, 1:] * (cur * (1 - prev))).sum(1)
    gold = gold + (sw_off[:, 1:] * ((1 - cur) * prev)).sum(1)
    # 분배함수 (B,K,2)
    a0 = torch.zeros(B, K, device=em.device, dtype=em.dtype)
    a1 = em[:, 0] + (init if init is not None else 0.0)
    for t in range(1, T):
        n0 = torch.logaddexp(a0, a1 + sw_off[:, t])
        n1 = torch.logaddexp(a1, a0 + sw_on[:, t]) + em[:, t]
        a0, a1 = n0, n1
    logZ = torch.logaddexp(a0, a1)
    return (logZ - gold).mean()


@torch.no_grad()
def viterbi(em: torch.Tensor, sw_on: torch.Tensor, sw_off: torch.Tensor,
            init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """최적 상태 열 (B,T,K) bool."""
    B, T, K = em.shape
    d0 = torch.zeros(B, K, device=em.device, dtype=em.dtype)
    d1 = em[:, 0] + (init if init is not None else 0.0)
    bk = torch.zeros(B, T, K, 2, dtype=torch.bool, device=em.device)
    for t in range(1, T):
        stay0, flip0 = d0, d1 + sw_off[:, t]
        stay1, flip1 = d1, d0 + sw_on[:, t]
        take0 = flip0 > stay0
        take1 = flip1 > stay1
        bk[:, t, :, 0] = take0                 # 참이면 이전은 상태 1
        bk[:, t, :, 1] = take1                 # 참이면 이전은 상태 0
        d0 = torch.where(take0, flip0, stay0)
        d1 = torch.where(take1, flip1, stay1) + em[:, t]
    path = torch.zeros(B, T, K, dtype=torch.bool, device=em.device)
    cur = d1 > d0
    path[:, T - 1] = cur
    for t in range(T - 1, 0, -1):
        prev_is_flip = torch.where(cur, bk[:, t, :, 1], bk[:, t, :, 0])
        cur = torch.where(prev_is_flip, ~cur, cur)
        path[:, t - 1] = cur
    return path
