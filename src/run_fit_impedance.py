"""계통 임피던스를 잰다 — **라벨 없이**, 새 장소에서도 (12.149)

무엇에 쓰나
----------
`run_adapt --harm-offset` 의 보정이 `Z_h·I_h` 를 회귀자로 쓴다 (12.148.2). `Z` 는
**그 집 배선의 값**이라 장소가 바뀌면 다시 재야 한다. 실제로 3.2배 다르다:

```
집 (8파일)        R = 1.629 Ω   L = 455 µH   |Z|: h1 1.64 -> h15 3.05 Ω
다른 장소 (test)   R = 0.511 Ω   L = 144 µH   |Z|: h1 0.51 -> h15 0.96 Ω
```

**라벨을 안 쓴다** — `|V_h|` 관측과 복소 `I_h`(`ih*`/`ihdeg*`) 만 있으면 된다.
정답 구간도, 사람 기록도, 격리 녹화도 필요 없다.

두 가지 자
----------
**① 자연 변동법** (`natural`) — 창마다 `V_h = V_src,h − Z_h·I_h` 를 세우고
   `|V_h|` 관측에 맞춘다. 부하가 **연속적으로** 변하면 듣는다.

**② 계단법** (`step`) — 연속한 두 창의 차를 본다:

       Δ|V₁|  ≈  −Re( conj(u)·Z·ΔI₁ )      (V_src 가 지배할 때)

   `Δ|V₁|` 를 `[ReΔI₁, ImΔI₁]` 로 회귀하면 `Z` 가 나온다. 큰 부하를 **반복해서
   켜고 끄면** 듣는다. 전역 위상(`u`)은 미지지만 보정 계수 `A` 가 흡수한다.

둘이 서로 대조다. 집에서 계단법 2.04 Ω vs 자연 변동 1.63 Ω 로 **25% 안에서** 맞는다.

⚠ **`Z` 는 그 정도 정밀도로만 안다.** 파일 집합을 11 -> 8 로 바꾸면 1.95 -> 1.63 으로
   20% 움직인다 (대조 파일이 부하가 커서 `Z` 를 정하는 데 크게 기여한다).

⚠ **물리 제약이 필수다.** 차수마다 `Z_h` 를 자유롭게 두면 h7~h11 에서 **R 이 음수**
   로 나온다 (수동 소자로 불가능). `|V_h|` 폭이 1.3~2.8V 뿐이라 잡음을 맞추는 것이다.
   `Z_h = R + j·h·ωL` 로 묶으면 h1 의 좋은 조건(|V1| 폭 53V)이 전 차수를 정한다.

⚠ **전 파일이 같은 장소여야 한다.** `data/test.csv` 는 다른 장소다 (사용자 확인).

자료가 되는지 먼저 본다
--------------------
새 장소에서 첫 번째로 물을 것은 *"이 녹화로 `Z` 를 잴 수 있나"* 다. 이 도구가
**진단을 먼저 찍는다**:

```
|I₁| 폭        넓을수록 좋다. 집 4.6A / test 7.8A
큰 계단 수      ΔI>0.5A 인 창 전이. 집 1,699개 / test **16개** <- 계단법 불가
적합 RMS       정규화값. 1.0 이면 평균만큼도 설명 못 한 것이다
```

`test.csv` 가 그 반례다 — 전류는 크지만(최대 4.8A) **자주 안 바뀌어서** 계단이
16개뿐이다. **부하가 크다고 되는 게 아니라 자주 바뀌어야 한다.**

몇 개나 있어야 하나 (집 자료를 솎아서 잰 값):

```
계단 수     |Z₁| 중앙    p5~p95        판정
     5      2.930   1.69 ~ 7.39    못 쓴다
    20      1.989   1.49 ~ 5.63
   100      2.351   1.53 ~ 3.54    ±40%
   300      2.146   1.52 ~ 2.85    ±30%
 1,699      2.037   (기준)
```

**수백 개는 있어야 한다.** 큰 부하 하나를 100~300번 토글하는 30분 녹화면 된다.

쓰는 법
------
    python -X utf8 -m src.run_fit_impedance --stems test.2 test3 test_4 …
    python -X utf8 -m src.run_fit_impedance --stems test --out results/z_other.npz
    python -X utf8 -m src.run_fit_impedance --stems … --min-di 0.5
"""
from typing import Dict, List, Sequence, Tuple
import argparse
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

#: 홀수차만 쓴다 — 12.72(전류 짝수차는 레인지 전환 인공물)와 12.147(전압 짝수차가
#: 부하 595배 변화에 불변. 상류인지 채널인지 미결).
ORDERS = (1, 3, 5, 7, 9, 11, 13, 15)
BLOCK = 30          #: `vh*` 가 30사이클(0.5초)마다 갱신된다. `ih*` 는 매 사이클.


def load_blocks(stem: str, orders: Sequence[int] = ORDERS) -> Dict[int, tuple]:
    """차수마다 (복소 I, |V|) 를 30사이클 블록으로. **라벨을 안 쓴다.**"""
    cols = ([f"vh{h}" for h in orders] + [f"ih{h}" for h in orders]
            + [f"ihdeg{h}" for h in orders])
    d = pd.read_csv(f"data/{stem}.csv", usecols=cols)
    n = len(d) // BLOCK * BLOCK
    out = {}
    for h in orders:
        i = (d[f"ih{h}"].values
             * np.exp(1j * np.deg2rad(d[f"ihdeg{h}"].values)))[:n]
        out[h] = (i.reshape(-1, BLOCK).mean(1),
                  d[f"vh{h}"].values[:n].reshape(-1, BLOCK).mean(1))
    return out


def _vsrc(I: np.ndarray, Vm: np.ndarray, Z: complex) -> Tuple[complex, float]:
    """`Z` 고정 -> `V_src` (복소 2변수) 만. 파일x차수마다 독립이라 분리 가능하다."""
    f = lambda q: np.abs((q[0] + 1j * q[1]) - Z * I) - Vm
    r = least_squares(f, [Vm.mean(), 0.0], method="lm", max_nfev=2000)
    return complex(r.x[0], r.x[1]), float(np.sqrt(np.mean(r.fun ** 2))
                                          / max(Vm.std(), 1e-3))


def fit_natural(data: Dict[str, Dict[int, tuple]], orders: Sequence[int] = ORDERS
                ) -> Tuple[float, float, Dict[Tuple[str, int], complex], float]:
    """자연 변동법. `Z_h = R + j·h·ωL` 로 **미지수 둘**. 분리 구조라 빠르다 (~14초)."""
    def outer(q):
        return sum(_vsrc(*data[s][h], q[0] + 1j * h * q[1])[1] ** 2
                   for s in data for h in orders)

    b = minimize(outer, [2.0, 0.05], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 60})
    R, X1 = float(b.x[0]), float(b.x[1])
    VS, rms = {}, []
    for s in data:
        for h in orders:
            v, e = _vsrc(*data[s][h], R + 1j * h * X1)
            VS[(s, h)] = v; rms.append(e)
    return R, X1, VS, float(np.sqrt(np.mean(np.square(rms))))


def fit_step(data: Dict[str, Dict[int, tuple]], min_di: float = 0.5) -> dict:
    """계단법 — 큰 부하의 켜고 끔에서 `|Z₁|` 을 직접 읽는다.

    `Δ|V₁|` 를 `[ReΔI₁, ImΔI₁]` 로 회귀한다. 전역 위상은 미지이나 `|Z|` 는 나오고,
    보정 계수 `A` 가 그 회전을 흡수하므로 실용상 문제가 없다.
    """
    dV, dI = [], []
    for s in data:
        I, V = data[s][1]
        dV.append(np.diff(V)); dI.append(np.diff(I))
    dV = np.concatenate(dV); dI = np.concatenate(dI)
    m = np.abs(dI) > min_di
    out = {"n_all": int(len(dV)), "n_step": int(m.sum()), "min_di": min_di}
    if m.sum() < 20:
        return dict(out, z1=float("nan"), r2=float("nan"))
    X = np.c_[dI[m].real, dI[m].imag]
    b, *_ = np.linalg.lstsq(X, dV[m], rcond=None)
    r2 = 1 - ((dV[m] - X @ b) ** 2).sum() / max(((dV[m] - dV[m].mean()) ** 2).sum(), 1e-12)
    return dict(out, z1=float(abs(b[0] + 1j * b[1])), r2=float(r2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", required=True,
                    help="같은 장소의 파일들. **다른 장소를 섞지 말 것** (Z 를 공유한다)")
    ap.add_argument("--min-di", type=float, default=0.5, help="계단법의 최소 ΔI (A)")
    ap.add_argument("--out", default="", metavar="NPZ", help="R, X1, V_src 저장")
    a = ap.parse_args()

    miss = [s for s in a.stems if not os.path.exists(f"data/{s}.csv")]
    if miss:
        raise SystemExit(f"원자료가 없습니다: {miss}")

    print("=" * 92)
    print(f"계통 임피던스 — 라벨 없이 (파일 {len(a.stems)}개: {', '.join(a.stems)})")
    print("=" * 92)
    data = {s: load_blocks(s) for s in a.stems}

    # ── 진단 먼저 (규칙 41 — 그 축의 변수가 움직이는 자료인가) ────────────
    print(f"\n  [진단] 이 녹화로 잴 수 있나\n")
    print(f"  {'파일':<12s}{'블록':>7s}{'|I₁| 중앙':>10s}{'|I₁| 폭':>10s}"
          f"{'|V₁| 폭':>10s}{f'ΔI>{a.min_di:g}A 계단':>14s}")
    tot_step = 0
    for s in a.stems:
        I, V = data[s][1]
        ns = int((np.abs(np.diff(I)) > a.min_di).sum()); tot_step += ns
        print(f"  {s:<12s}{len(I):>7d}{np.median(np.abs(I)):>10.3f}"
              f"{np.abs(I).max() - np.abs(I).min():>10.3f}"
              f"{V.max() - V.min():>10.2f}{ns:>14d}")
    print(f"  {'합계':<12s}{'':>7s}{'':>10s}{'':>10s}{'':>10s}{tot_step:>14d}")
    if tot_step < 100:
        print(f"\n  ⚠ 계단이 {tot_step}개뿐이다 — **계단법을 못 쓴다.** 수백 개는 있어야")
        print("     ±30% 안에 든다. 큰 부하 하나를 100~300번 토글하는 녹화가 필요하다.")
        print("     자연 변동법은 그래도 될 수 있다 (아래 RMS 를 볼 것).")

    # ── ① 자연 변동법 ────────────────────────────────────────────────
    R, X1, VS, rms = fit_natural(data)
    L = X1 / (2 * np.pi * 60) * 1e6
    print(f"\n  [① 자연 변동법]  물리 제약 Z_h = R + j·h·ωL, 미지수 둘\n")
    print(f"     R  = {R:8.4f} Ω     L = {L:6.0f} µH     정규화 RMS {rms:.3f}")
    print(f"     |Z_h|: " + "  ".join(f"h{h} {abs(R + 1j * h * X1):.2f}" for h in ORDERS))
    if R <= 0:
        print("     ⚠ **R 이 음수다** — 수동 임피던스로 불가능하다. 자료가 모자란다.")
    if rms > 0.95:
        print("     ⚠ RMS 가 1 에 가깝다 — 평균만큼도 설명 못 했다. 부하 변동이 부족하다.")

    # ── ② 계단법 (대조) ──────────────────────────────────────────────
    st = fit_step(data, a.min_di)
    print(f"\n  [② 계단법]  Δ|V₁| ~ [ReΔI₁, ImΔI₁]   — ① 의 대조\n")
    if np.isnan(st["z1"]):
        print(f"     계단 {st['n_step']}개 — **부족하다** (20개 미만)")
    else:
        d = abs(st["z1"] - abs(R + 1j * X1)) / max(abs(R + 1j * X1), 1e-9) * 100
        print(f"     |Z₁| = {st['z1']:.3f} Ω   R² {st['r2']:.3f}   계단 {st['n_step']}개")
        print(f"     ① 의 |Z₁| = {abs(R + 1j * X1):.3f} Ω  ->  **{d:.0f}% 차이**")
        if d > 50:
            print("     ⚠ 두 자가 50% 넘게 다르다. 어느 쪽도 믿지 말 것.")

    # ── 배경 전압 ────────────────────────────────────────────────────
    print(f"\n  [배경 전압 V_src]  — `--harm-offset` 의 회귀자로도 쓴다 (12.149.1)\n")
    print(f"  {'파일':<12s}" + "".join(f"{f'V_src,{h}':>10s}" for h in ORDERS[:5]))
    for s in a.stems:
        print(f"  {s:<12s}" + "".join(f"{abs(VS[(s, h)]):>10.2f}" for h in ORDERS[:5]))

    if a.out:
        np.savez(a.out, R=R, X1=X1, orders=np.array(ORDERS),
                 stems=np.array(a.stems), rms=rms,
                 vsrc=np.array([[VS[(s, h)] for h in ORDERS] for s in a.stems]),
                 step_z1=st["z1"], step_n=st["n_step"], argv=np.array(sys.argv))
        print(f"\n  저장: {a.out}")
    print("\n  ⚠ Z 는 ±25% 정도로만 안다 (두 자의 차, 그리고 파일 집합 바꾸면 20% 이동).")
    print("  ⚠ **다른 장소를 섞지 말 것.** Z 는 그 집 배선의 값이다 (장소 간 3.2배 차이).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
