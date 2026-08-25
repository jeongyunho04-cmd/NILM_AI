"""추론 후처리 — 물리 전력 상한과 게이트 동기 (12.100~12.102절)

**무엇을 고치는가.** 프로젝터는 격리에서 48.5~49.3W (p5~p95), 최대 49.6W 로
폭이 ±0.5W 인 기기다. 그런데 2단계 단독 모델은 복합 실측에서 중앙 73.5W, 최대
137W 를 붙이고 **창의 74.6%** 가 물리 상한을 넘는다 (12.100.1). 그 초과분은
다른 SMPS 의 몫이고, 그래서 충전기가 전이에서 5/23 밖에 안 맞았다.

**어떻게 고치는가.** 추론 뒤에 상한을 넘는 만큼을 잘라 **다른 SMPS 로 넘긴다.**

    P_proj > cap 이면  초과분을 게이트 가중으로 다른 SMPS 에 분배
    넘겨받아 문턱을 넘은 기기는 게이트도 켠다 (gate_sync)

**세 가지를 측정으로 정했다** (12.102):

- **상한은 프로젝터에만 건다.** 충전기·미니PC 는 상한을 넘는 창이 0.0%/0.4% 라
  걸 이유가 없고, 걸면 **받을 자리가 좁아져 오히려 나빠진다** (44/59 -> 39/59)
- **상한값은 둔감하다.** 55 / 60 / 70W 가 모두 44/59 다. 격리 최대(49.6W)에
  여유를 얹은 55W 를 쓴다 — 튜닝 잔향이 아니라는 근거다
- **초과분은 전부 즉시 넘긴다.** "정상 상태만 넘기고 전이 순간은 남긴다" 를
  시험했으나 **반대로 갔다** (12.101.2) — 재배분은 바로 그 전이 순간에 작동해
  효과를 낸다

**하이브리드에는 걸지 않는다.** 1단계에서 SMPS 를 가져오는 조합은 프로젝터
초과가 44.9% 로 덜하고, 충전기가 이미 20/23 이라 상한이 프로젝터만 깎는다
(41/59 -> 36/59). `run_gate_check --postproc` 는 그래서 기본이 꺼져 있다.
"""
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

#: 이 기기들 사이에서만 전력을 옮긴다 (저항 부하는 건드리지 않는다).
SMPS_GROUP: Tuple[str, ...] = ("beam_projector", "laptop_charger", "minipc")

#: 격리 통전 중 60초 창 평균의 최대값 (2026-08-25 측정).
ISOLATED_MAX_W: Dict[str, float] = {
    "beam_projector": 49.6, "laptop_charger": 70.3, "minipc": 26.9,
}

#: 상한을 거는 기기와 그 값. 격리 최대 + 11%.
CAP_W: Dict[str, float] = {"beam_projector": 55.0}

#: 게이트를 켤 문턱 — 격리 통전 중앙값(48.8 / 51.9 / 14.5W)의 20%.
#: 20% 와 40% 가 튜닝셋에서 각각 미니PC F1 +0.079 / +0.055 였고 홀드아웃에서는
#: 둘 다 −0.01 이다. 이득이 파일에 따라 갈리므로 **기본은 끔** (12.102.2).
GATE_ON_W: Dict[str, float] = {
    "beam_projector": 9.8, "laptop_charger": 10.4, "minipc": 2.9,
}


def apply_postproc(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
                   caps: Optional[Dict[str, float]] = None,
                   gate_sync: bool = False,
                   gate_on_w: Optional[Dict[str, float]] = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """물리 상한을 넘는 예측을 잘라 다른 SMPS 로 넘긴다.

    Args:
        P: (n, K) 기기별 예측 전력 (게이트가 곱해진 값)
        gate: (n, K) 게이트 확률
        apps: 기기 목록 (열 순서)
        caps: {기기: 상한W}. 기본 `CAP_W` (프로젝터만)
        gate_sync: 넘겨받아 문턱을 넘은 기기의 게이트를 켠다
        gate_on_w: 게이트 문턱. 기본 `GATE_ON_W`

    Returns:
        (P_new, gate_new). 총합은 보존된다 — 받을 기기가 있는 한 버리지 않는다.
    """
    caps = CAP_W if caps is None else caps
    thr = GATE_ON_W if gate_on_w is None else gate_on_w
    out = np.array(P, dtype=np.float64, copy=True)
    g = np.array(gate, dtype=np.float64, copy=True)

    recv = [j for j, a in enumerate(apps) if a in SMPS_GROUP]
    capped = [(j, caps[a]) for j, a in enumerate(apps) if a in caps]
    if not capped or not recv:
        return out, g

    for j, lim in capped:
        excess = np.clip(out[:, j] - lim, 0.0, None)
        if not excess.any():
            continue
        out[:, j] -= excess
        others = [k for k in recv if k != j]
        if not others:
            continue
        w = np.clip(g[:, others], 1e-6, None)
        share = w / w.sum(1, keepdims=True)
        out[:, others] += share * excess[:, None]

    if gate_sync:
        for j, a in enumerate(apps):
            if a in SMPS_GROUP and a in thr:
                g[:, j] = np.where(out[:, j] >= thr[a],
                                   np.maximum(g[:, j], 0.5 + 1e-6), g[:, j])
    return out, g
