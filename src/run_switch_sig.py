"""전이 지문 사전 — 단독녹화에서 기기별 **운전 단계와 전이 벡터**를 뽑는다 (12.155)

무엇에 쓰나
----------
사람 타임라인은 *"seq 254 에 드라이기 강 켜짐"* 같은 기록이다. 시각은 ±3초 어긋나고
빠뜨린 것도 있다. 신호에서 정확한 시각과 정체를 찾아야 하는데, 그러려면 **각 기기가
켜지고 꺼지고 단계가 바뀔 때 무엇이 어떻게 변하는지**를 먼저 알아야 한다.

`harmonic_signatures`(3.4절)는 **통전 중 와트당 고조파**라 이 일에 안 맞다. 필요한
것은 전이 자체다 — 드라이기 강↔약은 ΔP 가 크지만 켜짐/꺼짐이 아니고, 선풍기
약/중/강은 ΔP 가 작다. 오븐은 히터가 저 혼자 듀티로 껐다 켜진다.

무엇을 내나
----------
녹화를 0.5초 블록(=`seq` 한 칸)으로 줄이고 **전력 평탄면(plateau)** 으로 자른다.
평탄면 사이의 전이마다 이렇게 적는다:

```
ΔP            전력 계단 (W)
ΔI_h          복소 고조파 계단 (15차, A)
V_rms         그 순간 선전압 — 12.151 의 항등식 보정에 쓴다
plateau 전/후  운전 단계 (W)
```

**항등식을 여기서 쓴다.** `Re(I₁)/P = 1/V₁` 이 정의라(12.151), 전이의 ΔP 는
`Re(ΔI₁)·V` 로 신호에서 직접 나온다. 그래서 사전을 다른 장소에 그대로 쓸 수 있다 —
`sig` 를 `V_녹화/V(t)` 로 옮기면 된다.

쓰는 법
------
    python -X utf8 -m src.run_switch_sig --out results/switch_sig.json
    python -X utf8 -m src.run_switch_sig --stems hotplate_4_new --min-dp 20
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.preprocessing.raw_csv import read_raw_csv

BLOCK = 30                      #: 0.5초 = seq 한 칸
ORDERS = tuple(range(1, 16))
#: 기기별 단독녹화. `_new` 는 **장소 B** 다 (12.155). 장소가 달라도 항등식으로 옮긴다.
ISOLATED: Dict[str, List[str]] = {
    "air_conditioner": ["air_conditioner"],
    "beam_projector": ["beam_projector", "beam_projector_2", "beam_projector_3_fixed"],
    "electiric_kettle": ["electiric_kettle", "electric_kettle_2_fixed",
                         "electric_kettle_3_new"],
    "fan": ["fan_1", "fan_2", "fan_3"],
    "hair_dryer": ["hair_dryer_1", "hair_dryer_2", "hair_dryer_3"],
    "hotplate": ["hotplate_1", "hotplate_2", "hotplate_3_fixed", "hotplate_4_new"],
    "laptop_charger": ["laptop_charger_1", "laptop_charger_2",
                       "laptop_charger_3_fixed", "laptop_charger_4_fixed"],
    "minipc": ["minipc_1", "minipc_2", "minipc_3"],
    "oven": ["oven", "oven_2", "oven_3_fixed"],
}


def to_blocks(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, V, I) 를 0.5초 블록 중앙값으로. I 는 (n, 15) 복소.

    ⚠ **블록을 행 순서가 아니라 `seq` 로 색인한다.** 패킷이 빠진 파일이 있어
    (test_7 은 11개) 행으로 세면 시간축이 그만큼 **줄어든다** — 기존 라벨과
    맞대 보니 test_6/test_7 에서 −3.5초 계통 편차로 나타났다 (12.155).
    전처리(npz)는 빠진 자리를 보간해 채우므로 `t = (seq − seq_lo)·0.5` 가 정본이다.
    빠진 블록은 NaN 으로 두고 앞뒤에서 채운다.
    """
    cols = ["p_w", "vrms"] + [f"ih{h}" for h in ORDERS] + [f"ihdeg{h}" for h in ORDERS]
    d, info = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    seq = d["seq"].to_numpy(np.int64)
    lo, hi = int(seq.min()), int(seq.max())
    nb = hi - lo + 1
    idx = seq - lo

    def agg(x: np.ndarray) -> np.ndarray:
        out = np.full(nb, np.nan)
        o = np.argsort(idx, kind="stable")
        xi, xv = idx[o], x[o]
        b = np.flatnonzero(np.diff(xi)) + 1
        for a_, b_ in zip(np.concatenate([[0], b]), np.concatenate([b, [len(xi)]])):
            out[xi[a_]] = np.median(xv[a_:b_])
        m = np.isnan(out)
        if m.any():                       # 빠진 블록은 선형 보간 (npz 와 같은 처리)
            g = np.flatnonzero(~m)
            out[m] = np.interp(np.flatnonzero(m), g, out[g])
        return out

    P = agg(d["p_w"].to_numpy(np.float64))
    V = agg(d["vrms"].to_numpy(np.float64))
    I = np.stack([
        agg(d[f"ih{h}"].to_numpy(np.float64)
            * np.cos(np.deg2rad(d[f"ihdeg{h}"].to_numpy(np.float64))))
        + 1j * agg(d[f"ih{h}"].to_numpy(np.float64)
                   * np.sin(np.deg2rad(d[f"ihdeg{h}"].to_numpy(np.float64))))
        for h in ORDERS], 1)
    return P, V, I


def to_cycles(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, V, I) 를 **사이클(60Hz) 해상도**로. `seq*30 + cycle` 로 색인한다.

    ⚠ 0.5초 블록으로 줄이면 안 된다. `vh*` 만 30사이클마다 갱신되고 `p_w`/`ih*`/
    `ihdeg*` 는 **매 사이클**이다. 블록으로 보면 1초 안에 일어난 두 사건이 하나로
    뭉쳐 섞인 계단이 되는데, 사람이 스위치를 두 개 누르는 간격은 그보다 훨씬 크다.
    """
    cols = ["p_w", "vrms"] + [f"ih{h}" for h in ORDERS] + [f"ihdeg{h}" for h in ORDERS]
    d, _ = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    seq = d["seq"].to_numpy(np.int64)
    cyc = d["cycle"].to_numpy(np.int64)
    idx = (seq - int(seq.min())) * BLOCK + cyc
    n = int(idx.max()) + 1

    def put(x: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        out[idx] = x
        m = np.isnan(out)
        if m.any():                       # 빠진 사이클은 선형 보간 (npz 와 같은 처리)
            g = np.flatnonzero(~m)
            out[m] = np.interp(np.flatnonzero(m), g, out[g])
        return out

    P = put(d["p_w"].to_numpy(np.float64))
    V = put(d["vrms"].to_numpy(np.float64))
    I = np.stack([
        put(d[f"ih{h}"].to_numpy(np.float64)
            * np.cos(np.deg2rad(d[f"ihdeg{h}"].to_numpy(np.float64))))
        + 1j * put(d[f"ih{h}"].to_numpy(np.float64)
                   * np.sin(np.deg2rad(d[f"ihdeg{h}"].to_numpy(np.float64))))
        for h in ORDERS], 1)
    return P, V, I


def _roll_med(x: np.ndarray, w: int) -> np.ndarray:
    """길이 w 의 이동 중앙값. out[i] 는 x[i:i+w] 의 중앙값이다."""
    if len(x) < w:
        return np.full(len(x), np.median(x))
    sw = np.lib.stride_tricks.sliding_window_view(x, w)
    m = np.median(sw, axis=-1)
    return np.concatenate([m, np.full(len(x) - len(m), m[-1])])


def _step_series(x: np.ndarray, W: int, G: int) -> np.ndarray:
    """앞뒤 창 중앙값의 차 (실수 계열)."""
    n = len(x)
    d = np.zeros(n)
    k = W + G
    if n > W + k:
        pre = _roll_med(x, W)
        d[k:n - W] = pre[k:n - W] - pre[:n - W - k]
    return d


def _pick(score: np.ndarray, W: int, G: int) -> np.ndarray:
    """국소 최대만 남긴다. 반환은 계단의 대표 사이클."""
    out = []
    for b in np.flatnonzero(score >= 1.0):
        lo, hi = max(0, b - W - G), min(len(score), b + W + G + 1)
        if score[b] >= score[lo:hi].max() - 1e-12:
            if not out or b - out[-1] > W + G:
                out.append(int(b))
            elif score[b] > score[out[-1]]:
                out[-1] = int(b)
    return np.array(out, np.int64) - (W + G // 2)


#: 긴 창이 짧은 창의 이 배수 아래로 떨어지면 **되돌아오는 리플**로 본다.
PERSIST = 0.35
#: 지속성 확인용 긴 창 (사이클). 0.75초.
W_LONG = 45


def _persist_gate(short: np.ndarray, long_: np.ndarray) -> np.ndarray:
    """리플 제거. 진짜 on/off 는 준위가 몇 초 유지되지만 리플은 되돌아온다.

    test_5 의 핫플 구간에서 계단이 **0.7초 간격으로 규칙적으로** 잡혔다 —
    337.12 337.72 339.18 339.73 341.03 341.88 … 짧은 창(0.2초)만 보면 매번
    계단이지만 긴 창(0.75초)으로 보면 0 이다 (12.155).
    """
    ok = np.abs(long_) >= PERSIST * np.abs(short)
    ok &= np.sign(long_) == np.sign(short)
    return np.where(ok, short, 0.0)


def edges_p(P: np.ndarray, min_dp: float, W: int = 12, G: int = 6,
            persist: bool = False) -> np.ndarray:
    """전력 채널의 계단 (사이클 해상도).

    `persist=True` 는 **후보를 고를 때만** 쓴다. 되돌아오는 리플을 거른다.
    측정 창의 경계로 쓸 때는 꺼야 한다 — 핫플 듀티처럼 **진짜지만 주기적인**
    전이까지 지우면 창이 그것을 가로질러 중앙값이 무의미해진다 (12.155).
    """
    d = _step_series(P, W, G)
    if persist:
        d = _persist_gate(d, _step_series(P, W_LONG, G))
    return _pick(np.abs(d) / max(min_dp, 1e-9), W, G)


def decorrelate_p(z: np.ndarray, P: np.ndarray) -> np.ndarray:
    """복소 h3 에서 전력에 비례하는 성분을 빼낸다.

    ⚠ **지금 안 쓴다.** 파일 전체로 회귀를 걸면 SMPS 자신의 h3 까지 흡수해서
      계단이 사라졌다 (test_5 에서 후보가 12개 -> 1개). 국소 회귀 + 계단 구간
      제외로 다시 해야 한다. 미결.


    저항 부하는 h3 를 거의 안 내지만, 켜지면 전압이 떨어져 **다른 기기의** h3 를
    움직인다(감쇠). test_5 의 핫플이 그렇다 — 465W 를 켜면 |I₃| 가 0.310 -> 0.285
    로 0.025A 준다. 미니PC 켜짐이 0.043A 라 그 리플과 1.7배밖에 차이가 안 난다.

    그 성분은 **P 에 선형**이므로 `I₃ ~ a·P + b` 를 최소자승으로 풀어 빼면 된다.
    기기 자신의 h3 는 P 와 같은 순간에 계단으로 뛰므로 회귀가 흡수하지 못한다
    (회귀는 전 구간의 평균 기울기만 가져간다).
    """
    A = np.c_[P, np.ones(len(P))]
    out = np.empty_like(z)
    for part in ("real", "imag"):
        y = getattr(z, part)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        if part == "real":
            out = r.astype(np.complex128)
        else:
            out = out + 1j * r
    return out


def edges_i3(I: np.ndarray, min_di3: float, W: int = 12, G: int = 6,
             persist: bool = False, P: np.ndarray = None) -> np.ndarray:
    """**h3 채널**의 계단. 저항 부하는 여기 거의 안 보인다 — test_5 에서 오븐이
    621 -> 1,657W 로 뛸 때 |I₃| 는 0.578 -> 0.587 로 그대로다. 그래서 SMPS 의
    고조파를 잴 때는 **이 채널의 이웃**으로 창을 잘라야 한다 (12.155)."""
    z3 = I[:, 2] if P is None else decorrelate_p(I[:, 2], P)
    zr = _step_series(z3.real, W, G)
    zi = _step_series(z3.imag, W, G)
    d = np.abs(zr + 1j * zi)
    if persist:
        lr = _step_series(I[:, 2].real, W_LONG, G)
        li = _step_series(I[:, 2].imag, W_LONG, G)
        d = np.where(np.abs(lr + 1j * li) >= PERSIST * d, d, 0.0)
    return _pick(d / max(min_di3, 1e-9), W, G)


def edges(P: np.ndarray, I: np.ndarray, min_dp: float, min_di3: float,
          W: int = 12, G: int = 6) -> np.ndarray:
    """두 채널의 합집합."""
    a = set(int(x) for x in edges_p(P, min_dp, W, G, persist=True))
    a |= set(int(x) for x in edges_i3(I, min_di3, W, G, persist=True))
    return np.array(sorted(a), np.int64)


def atten_coef(P: np.ndarray, z3: np.ndarray, b: int,
               half: int = 900, guard: int = 90) -> complex:
    """계단 b 주변에서 **h3 의 전력 비례 성분** `a` 를 잰다 (12.155).

    저항 부하는 h3 를 거의 안 내지만 켜지면 전압을 끌어내려 **다른 기기의** h3 를
    줄인다(감쇠). test_5 의 핫플이 465W 를 켜면 |I₃| 가 0.310 -> 0.285 로 0.025A
    준다 — 미니PC 켜짐(0.043A)과 1.7배밖에 차이가 안 난다.

    `a` 는 그 순간 켜져 있는 기기 구성에 달렸으므로 **국소**로 잰다. 그리고
    계단 자신은 `guard`(1.5초) 로 빼야 한다 — 안 빼면 회귀가 그 계단을 흡수해
    SMPS 의 h3 까지 지운다 (파일 전체로 걸었다가 후보가 12개 -> 1개가 됐다).

    0.5초 블록 **차분**으로 푼다. 차분이라 절편이 없어지고, 듀티 왕복이 표본을
    많이 주므로 조건이 좋다.
    """
    lo, hi = max(0, b - half), min(len(P), b + half)
    n = (hi - lo) // BLOCK * BLOCK
    if n < 8 * BLOCK:
        return 0j
    pb = np.median(P[lo:lo + n].reshape(-1, BLOCK), 1)
    zb = (np.median(z3[lo:lo + n].real.reshape(-1, BLOCK), 1)
          + 1j * np.median(z3[lo:lo + n].imag.reshape(-1, BLOCK), 1))
    dP, dz = np.diff(pb), np.diff(zb)
    k = (b - lo) // BLOCK                     # 계단이 있는 블록
    g = max(1, guard // BLOCK)
    keep = np.ones(len(dP), bool)
    keep[max(0, k - g):k + g + 1] = False
    keep &= np.abs(dP) > 20.0                 # 전력이 실제로 움직인 곳만
    if keep.sum() < 4:
        return 0j
    den = float(np.sum(dP[keep] ** 2))
    if den < 1e-9:
        return 0j
    return complex(np.sum(dP[keep] * dz[keep].real) / den,
                   np.sum(dP[keep] * dz[keep].imag) / den)


def _win(b: int, nb: np.ndarray, n: int, G: int, wmax: int):
    """계단 b 의 앞뒤 창을 **그 채널의 이웃**으로 자른다."""
    left = nb[nb < b - G]
    right = nb[nb > b + G]
    pl = int(left[-1]) + G if len(left) else 0
    pr = int(right[0]) - G if len(right) else n
    return max(pl, b - wmax), b - G, b + G, min(pr, b + wmax)


def measure(P, V, I, ed: np.ndarray, W: int = 12, G: int = 6, wmax: int = 60,
            ed_p: np.ndarray = None, ed_3: np.ndarray = None) -> List[Dict]:
    """계단마다 앞뒤를 잰다. **채널마다 제 이웃으로 창을 자른다.**

    전력은 전력 계단으로, 고조파는 h3 계단으로 자른다. 섞어 쓰면 오븐 히터
    스위칭이 SMPS 의 h3 측정 창을 두 동강 낸다 (12.155).
    """
    out, n = [], len(P)
    ep = np.asarray(ed if ed_p is None else ed_p, np.int64)
    e3 = np.asarray(ed if ed_3 is None else ed_3, np.int64)
    for b in np.asarray(ed, np.int64):
        b = int(b)
        a0, a1, b0, b1 = _win(b, ep, n, G, wmax)
        # 고조파는 **h3 계단 자리**에 맞춘다. 전력 계단과 0.5초까지 어긋날 수 있고
        # (오븐 릴레이와 SMPS 스위치가 같은 순간이 아니다) 어긋난 채로 창을 잡으면
        # 전이 구간이 창 안에 들어와 ΔI 가 0 으로 씻긴다 (12.155).
        anc = b
        if len(e3):
            j = int(np.argmin(np.abs(e3 - b)))
            if abs(int(e3[j]) - b) <= 30:
                anc = int(e3[j])
        c0, c1, d0, d1 = _win(anc, e3, n, G, wmax)
        if min(a1 - a0, b1 - b0, c1 - c0, d1 - d0) < W // 2:
            continue
        pa, pb = float(np.median(P[a0:a1])), float(np.median(P[b0:b1]))
        ia = np.median(I[c0:c1].real, 0) + 1j * np.median(I[c0:c1].imag, 0)
        ib = np.median(I[d0:d1].real, 0) + 1j * np.median(I[d0:d1].imag, 0)
        di = ib - ia
        v = float(np.median(V[a0:b1]))
        dp = pb - pa
        # 감쇠 보정 — h3 에서 **전력에 비례하는 몫**을 뺀다 (12.155)
        a_at = atten_coef(P, I[:, 2], b)
        di = di.copy()
        di[2] = di[2] - a_at * dp
        out.append({"cycle": b, "t_block": b / float(BLOCK),
                    "dp_w": dp, "p_before": pa, "p_after": pb, "v_rms": v,
                    # h3 크기의 앞뒤 값. SMPS 사건의 **방향**은 이것이 정한다 —
                    # 총전력 부호는 오븐·핫플에 오염된다 (12.155).
                    "i3_before": float(abs(ia[2])),
                    "i3_after": float(abs(ia[2] + di[2])),
                    "atten": [a_at.real, a_at.imag],
                    "identity": float(di[0].real * v / dp) if abs(dp) > 1e-9 else float("nan"),
                    "di_re": di.real.tolist(), "di_im": di.imag.tolist(),
                    "win": [int(a1 - a0), int(b1 - b0), int(c1 - c0), int(d1 - d0)]})
    return out


def steps_multi(P: np.ndarray, I: np.ndarray, min_dp: float, min_di3: float = 0.02,
                K: int = 4, g: int = 2) -> List[int]:
    """전력 **과 h3 전류** 두 채널에서 계단을 찾아 합친다.

    전력만 보면 큰 저항 부하에 가린 SMPS 사건을 놓친다 — 오븐 히터가 ±1,100W 로
    듀티를 돌 때 45W 프로젝터 계단은 안 보인다. 그런데 **저항은 h3 에 거의
    아무것도 안 내고**(|u₃| 0.004~0.012) SMPS 는 |I₃|/|I₁| 이 0.90 이다. 그래서
    h3 채널이 정확히 그 반대편을 본다.
    """
    a = set(steps(P, min_dp, K, g))
    a |= set(steps_c(I[:, 2], min_di3, K, g))
    return sorted(a)


def steps_c(z: np.ndarray, min_d: float, K: int = 4, g: int = 2) -> List[int]:
    """복소 계열의 계단. 크기가 아니라 **복소 차의 크기**를 본다 — 위상만 도는
    전이(SMPS 켜짐이 다른 SMPS 위에 얹힐 때)도 잡아야 한다."""
    n = len(z)
    d = np.zeros(n)
    for b in range(K, n - K - g):
        pre = np.median(z[b - K:b].real) + 1j * np.median(z[b - K:b].imag)
        post = np.median(z[b + g:b + g + K].real) + 1j * np.median(z[b + g:b + g + K].imag)
        d[b] = abs(post - pre)
    cand = np.flatnonzero(d >= min_d)
    out = []
    for b in cand:
        lo, hi = max(0, b - K), min(n, b + K + 1)
        if d[b] >= d[lo:hi].max() - 1e-12:
            if not out or b - out[-1] > K:
                out.append(int(b))
            elif d[b] > d[out[-1]]:
                out[-1] = int(b)
    return out


def steps(P: np.ndarray, min_dp: float, K: int = 4, g: int = 2) -> List[int]:
    """앞뒤 창 차분으로 전이 지점을 찾는다. **평탄면 방식은 드리프트에 진다** —
    핫플은 통전 중 저항이 달아오르며 241 -> 474W 로 흐르는데, 그 구간을 하나의
    평탄면으로 묶으려 하면 아예 안 잡힌다.

    `K` 블록(=0.5초 단위) 중앙값을 앞뒤로 비교하고 `g` 블록을 과도 구간으로 비운다.
    같은 계단을 여러 번 잡지 않도록 ±K 안에서 최대만 남긴다.
    """
    n = len(P)
    d = np.zeros(n)
    for b in range(K, n - K - g):
        d[b] = np.median(P[b + g:b + g + K]) - np.median(P[b - K:b])
    cand = np.flatnonzero(np.abs(d) >= min_dp)
    out = []
    for b in cand:
        lo, hi = max(0, b - K), min(n, b + K + 1)
        if np.abs(d[b]) >= np.abs(d[lo:hi]).max() - 1e-12:
            if not out or b - out[-1] > K:
                out.append(int(b))
            elif np.abs(d[b]) > np.abs(d[out[-1]]):
                out[-1] = int(b)
    return out


def transitions(P, V, I, idx: List[int], min_dp: float, K: int = 4, g: int = 2) -> List[Dict]:
    """전이 지점마다 앞뒤 중앙값의 차. 복소 고조파도 같은 창으로 낸다."""
    out = []
    n = len(P)
    for b in idx:
        a0, a1 = max(0, b - K), b
        b0, b1 = min(n, b + g), min(n, b + g + K)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        pa, pb = float(np.median(P[a0:a1])), float(np.median(P[b0:b1]))
        dp = pb - pa
        if abs(dp) < min_dp:
            continue
        ia = np.median(I[a0:a1].real, 0) + 1j * np.median(I[a0:a1].imag, 0)
        ib = np.median(I[b0:b1].real, 0) + 1j * np.median(I[b0:b1].imag, 0)
        di = ib - ia
        v = float(np.median(V[a0:b1]))
        out.append({
            # 전이는 블록 b 와 b+g 사이에서 일어난다. 대표 시각은 그 가운데다 —
            # `b` 를 그대로 쓰면 계통적으로 0.5~1초 이르다 (12.155).
            "block": int(b), "t_block": b + g / 2.0,
            "dp_w": dp, "p_before": pa, "p_after": pb, "v_rms": v,
            # 항등식 검산 — 정의상 1 이어야 한다 (12.151)
            "identity": float(di[0].real * v / dp),
            "di_re": di.real.tolist(), "di_im": di.imag.tolist(),
        })
    return out


def levels(P: np.ndarray, idx: List[int], K: int = 4, g: int = 2) -> List[Dict]:
    """운전 단계 — 전이와 전이 **사이**의 중앙 전력을 모은다. 드리프트가 있어도
    구간 중앙값은 그 단계를 대표한다."""
    n = len(P)
    bounds = [0] + [b + g + 1 for b in idx] + [n]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a2 = max(a, 0); b2 = min(b - g, n)
        if b2 - a2 >= K:
            segs.append((float(np.median(P[a2:b2])), b2 - a2))
    if not segs:
        return []
    v = np.array([x[0] for x in segs]); w = np.array([x[1] for x in segs], float)
    o = np.argsort(v); v, w = v[o], w[o]
    cuts = np.flatnonzero(np.diff(v) > np.maximum(15.0, 0.12 * np.maximum(v[:-1], 1.0))) + 1
    out = []
    for gv, gw in zip(np.split(v, cuts), np.split(w, cuts)):
        out.append({"p_w": float(np.average(gv, weights=gw)), "p_lo": float(gv.min()),
                    "p_hi": float(gv.max()), "blocks": float(gw.sum()),
                    "n_seg": int(len(gv))})
    return sorted(out, key=lambda x: -x["blocks"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None, help="기본은 ISOLATED 전부")
    ap.add_argument("--min-dp", type=float, default=8.0, help="전이로 볼 최소 |ΔP| (W)")
    ap.add_argument("--win", type=int, default=4, help="앞뒤 비교 창 (블록, 0.5초 단위)")
    ap.add_argument("--guard", type=int, default=2, help="과도 구간으로 비울 블록 수")
    ap.add_argument("--out", default="results/switch_sig.json")
    a = ap.parse_args()

    want = {s for s in a.stems} if a.stems else None
    print("=" * 100)
    print("전이 지문 — 단독녹화의 운전 단계와 전이 벡터 (12.155)")
    print("=" * 100)
    doc: Dict[str, Dict] = {}
    for app, stems in ISOLATED.items():
        for st in stems:
            if want is not None and st not in want:
                continue
            if not Path(f"data/{st}.csv").exists():
                continue
            P, V, I = to_blocks(st)
            ix = steps(P, a.min_dp, a.win, a.guard)
            tr = transitions(P, V, I, ix, a.min_dp, a.win, a.guard)
            lv = levels(P, ix, a.win, a.guard)
            doc[st] = {"appliance": app, "blocks": int(len(P)),
                       "v_rms": float(np.median(V)), "levels": lv, "transitions": tr}
            ids = np.array([t["identity"] for t in tr if np.isfinite(t["identity"])])
            print(f"\n■ {app} / {st}   {len(P)*0.5:.0f}s   V {np.median(V):.1f}V   "
                  f"전이 {len(tr)}")
            print("   단계(W): " + "  ".join(
                f"{x['p_w']:.0f}({x['blocks']*0.5:.0f}s)" for x in lv[:6]))
            if len(ids):
                print(f"   항등식 Re(ΔI₁)·V/ΔP 중앙 {np.median(ids):.4f} "
                      f"(p5 {np.percentile(ids,5):.3f} p95 {np.percentile(ids,95):.3f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {a.out}  ({len(doc)}녹화, 전이 {sum(len(v['transitions']) for v in doc.values())}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
