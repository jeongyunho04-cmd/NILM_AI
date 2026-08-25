"""추론 후처리 — 물리 전력 상한 (12.100절)

프로젝터는 격리에서 **48.5~49.3W (p5~p95), 최대 49.6W** 다. 폭이 ±0.5W 인
기기인데 복합 실측에서 100W 넘게 예측된다 (12.100.1). 그 초과분은 물리적으로
불가능하므로 잘라내고, 잘라낸 만큼을 다른 SMPS 에 넘길 수 있다.

**12.87.2 와 무엇이 다른가.** 그쪽은 같은 사전을 **학습 손실**에 넣어 고조파가
나빠졌다. 여기서는 **추론 뒤**에 자르므로 학습에 되먹임이 없다. 12.85 의 HMM
후처리와 같은 자리(후처리)이나, 그쪽은 게이트의 지속성 사전이었고 이쪽은
전력 크기의 물리 상한이다.

**상한은 관대하게 잡는다.** 격리 최대에 여유를 얹는다 — 실측 전압대가 다르고
(4.3절) 개체차도 있다. 자르는 것이 목적이 아니라 **명백히 불가능한 값**만
막는 것이다.
"""
from typing import Dict, Optional, Sequence
import numpy as np

#: 격리 통전 중 60초 창 평균의 최대값에 여유를 얹은 값 (2026-08-25 측정).
#:     프로젝터 최대 49.6 / 충전기 70.3 / 미니PC 26.9
PHYSICAL_CAP_W: Dict[str, float] = {
    "beam_projector": 55.0,      # 최대 49.6 + 11%
    "laptop_charger": 78.0,      # 최대 70.3 + 11%
    "minipc": 30.0,              # 최대 26.9 + 11%
}


def cap_power(P: np.ndarray, apps: Sequence[str], gate: Optional[np.ndarray] = None,
              caps: Optional[Dict[str, float]] = None,
              redistribute: bool = False) -> np.ndarray:
    """물리 상한을 넘는 예측을 자른다. `redistribute` 면 초과분을 다른 SMPS 로.

    Args:
        P: (n, K) 기기별 예측 전력 (게이트가 이미 곱해진 것)
        apps: 기기 목록 (P 의 열 순서)
        gate: (n, K) 게이트 확률. 재배분 가중에 쓴다
        caps: 기기별 상한. 기본은 `PHYSICAL_CAP_W`
        redistribute: 초과분을 상한에 여유가 있는 다른 SMPS 로 넘긴다.
            **넘길 곳이 없으면 버린다** — 총합이 줄어 잔차가 늘어난다.
            그것이 정직하다. 없는 곳에 억지로 넣으면 오귀속이 는다.
    """
    caps = caps or PHYSICAL_CAP_W
    out = np.array(P, dtype=np.float64, copy=True)
    idx = [(j, a) for j, a in enumerate(apps) if a in caps]
    if not idx:
        return out
    cols = np.array([j for j, _ in idx])
    lim = np.array([caps[a] for _, a in idx], dtype=np.float64)

    sub = out[:, cols]
    excess = np.clip(sub - lim[None, :], 0.0, None)
    sub = np.minimum(sub, lim[None, :])

    if redistribute and excess.sum() > 0:
        head = np.clip(lim[None, :] - sub, 0.0, None)          # 남은 여유
        w = head if gate is None else head * np.clip(gate[:, cols], 1e-6, None)
        tot_ex = excess.sum(1, keepdims=True)
        wsum = w.sum(1, keepdims=True)
        share = np.divide(w, np.where(wsum > 0, wsum, 1.0))
        add = np.minimum(share * tot_ex, head)
        sub = sub + add
    out[:, cols] = sub
    return out


#: 게이트를 켤 문턱 — 격리 통전 중앙값의 40% (2026-08-25 측정 48.8 / 51.9 / 14.5W).
GATE_ON_W: Dict[str, float] = {
    "beam_projector": 19.5, "laptop_charger": 20.8, "minipc": 5.8,
}


def _causal_floor(x: np.ndarray, win: int) -> np.ndarray:
    """과거 `win` 표본 안의 최솟값 — 그 시점까지 **지속된** 성분이다.

    전이 직후에는 0 에 머물다가 창이 다 차면 올라온다. 그래서 "전이 순간은
    건드리지 않고 정상 상태만 옮긴다" 가 된다. 미래를 안 보므로 `run_live`
    에서도 같은 값이 나온다.
    """
    n = len(x)
    out = np.empty_like(x)
    from collections import deque
    dq: "deque[int]" = deque()
    for i in range(n):
        while dq and x[dq[-1]] >= x[i]:
            dq.pop()
        dq.append(i)
        while dq[0] <= i - win:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def cap_power_v2(P: np.ndarray, apps: Sequence[str], gate: np.ndarray,
                 caps: Optional[Dict[str, float]] = None,
                 sustain_win: int = 0, sync_gate: bool = False,
                 gate_on_w: Optional[Dict[str, float]] = None):
    """`cap_power` 의 확장 (12.101절). 지속 성분만 넘기고, 게이트도 맞춘다.

    Args:
        sustain_win: 0 이면 초과분 전부를 넘긴다 (12.100 과 동일).
            >0 이면 **과거 그만큼의 표본에서 지속된 초과분만** 넘긴다.
            전이 순간의 초과분은 원래 기기에 남으므로 그 기기의 전이 귀속을
            깎지 않는다 (12.100.5 가 지목한 손실).
        sync_gate: 넘겨받아 문턱을 넘은 기기의 게이트를 켠다. 전력만 옮기면
            on/off F1 이 안 변한다 (게이트를 안 건드리므로).

    Returns:
        (P_new, gate_new)
    """
    caps = caps or PHYSICAL_CAP_W
    thr = gate_on_w or GATE_ON_W
    out = np.array(P, dtype=np.float64, copy=True)
    g = np.array(gate, dtype=np.float64, copy=True)
    idx = [(j, a) for j, a in enumerate(apps) if a in caps]
    if not idx:
        return out, g
    cols = np.array([j for j, _ in idx])
    lim = np.array([caps[a] for _, a in idx], dtype=np.float64)

    sub = out[:, cols]
    excess = np.clip(sub - lim[None, :], 0.0, None)
    if sustain_win > 0:
        moved = np.stack([_causal_floor(excess[:, k], sustain_win)
                          for k in range(excess.shape[1])], 1)
    else:
        moved = excess
    sub = sub - moved                      # 안 옮긴 초과분은 그대로 둔다

    head = np.clip(lim[None, :] - np.minimum(sub, lim[None, :]), 0.0, None)
    w = head * np.clip(g[:, cols], 1e-6, None)
    tot = moved.sum(1, keepdims=True)
    wsum = w.sum(1, keepdims=True)
    add = np.minimum(np.divide(w, np.where(wsum > 0, wsum, 1.0)) * tot, head)
    sub = sub + add
    out[:, cols] = sub

    if sync_gate:
        for k, (j, a) in enumerate(idx):
            on = sub[:, k] >= thr.get(a, 1e9)
            g[:, j] = np.where(on, np.maximum(g[:, j], 0.5 + 1e-6), g[:, j])
    return out, g
