# -*- coding: utf-8 -*-
"""남은 잔차의 정체 (②, 기준 `results/_criteria_circuit.md` [N1]~[N4]).

    python -X utf8 -m src.run_residual_probe

12.185.23·24·25 로 계측 규약이 정리됐다 (원시 위상 교정 0.44°/차수). 그 위에서 ②의 두
잔차를 **진단**한다 — 새 요소를 시험하는 것이 아니라 잔차가 **어디에·무엇에 비례해** 있는지
본다. 그것이 다음에 무엇을 시도할지 정한다.

    ②a 충전기 19W (LOO 가 다른 점의 2~3배)
    ②b 미니PC 비도통 구간 5.5%

단계
----
    1  [N1] 시간 구조   잔차를 도통 상승·도통 하강·비도통 셋으로 가른다
    2  [N2] 비도통      비도통 잔차를 `dv/dt`(Cx) · 상수(덧셈) · 전력에 회귀
    3  [N3] 도통 에지   측정 펄스가 모델보다 일찍 끝나는가 (저쪽 §12.8 의 −10%)
    4  [N4] 전력 의존   잔차가 전력을 따라가는가 (덧셈 성분이면 저전력에서 상대적으로 크다)
"""
from typing import Dict, List
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import raw_snapshots_of
from src.synthesis.fit_raw import (BAND, background_phasors, load_raw, phasors, sim_current,
                                   wave_from_phasors, F, NS)

IN = "results/_circuit_raw_C.json"
OUT = "results/_residual.json"
DEVS = ("laptop_charger", "beam_projector", "minipc")


def windows(pt, thr=0.15, guard=3):
    """(도통 상승, 도통 하강, 비도통) 마스크. 도통은 반주기마다 하나씩 있다."""
    a = np.abs(pt.i)
    cond = a >= thr * a.max()
    k = np.ones(2 * guard + 1)
    grown = np.convolve(np.r_[cond, cond, cond].astype(float), k, "same")[NS:2 * NS] > 0
    # 반주기 안의 봉우리 기준으로 상승/하강 가르기
    rise = np.zeros(NS, bool)
    fall = np.zeros(NS, bool)
    lab = np.zeros(NS, int)
    n = 0
    prev = False
    for j in range(NS):
        if grown[j] and not prev:
            n += 1
        lab[j] = n if grown[j] else 0
        prev = grown[j]
    for g in range(1, n + 1):
        idx = np.where(lab == g)[0]
        if len(idx) < 3:
            continue
        pk = idx[int(np.argmax(a[idx]))]
        rise[idx[idx <= pk]] = True
        fall[idx[idx > pk]] = True
    return rise, fall, ~grown


def main() -> None:
    canon = json.load(open(IN, encoding="utf-8"))["devices"]
    bg = background_phasors()
    bgw = wave_from_phasors(bg)
    rec: Dict = {}

    print("=" * 104)
    print("[1][N1] 잔차의 시간 구조 — 도통 상승 / 도통 하강 / 비도통 [실측 rms 대비 %]")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'P[W]':>7s} {'전체':>7s} {'상승':>7s} {'하강':>7s} {'비도통':>7s} "
          f"{'표본(상/하/비)':>16s}")
    store = {}
    for dev in DEVS:
        par = tuple(canon[dev]["fit_fixrd"]["par5"])
        for st in raw_snapshots_of(dev):
            pt = load_raw(st, bg=bg)
            x = sim_current(par, pt, match_power=True)
            if x is None:
                continue
            r = x - pt.i
            ri, fa, nc = windows(pt)
            g = lambda m: 100 * np.sqrt(np.mean(r[m] ** 2)) / pt.irms if m.sum() else np.nan
            store[st] = dict(dev=dev, pt=pt, r=r, rise=ri, fall=fa, nc=nc, x=x)
            print(f"  {st:26s} {pt.p_w:7.2f} "
                  f"{100*np.sqrt(np.mean(r**2))/pt.irms:6.2f}% {g(ri):6.2f}% {g(fa):6.2f}% "
                  f"{g(nc):6.2f}% {ri.sum():5d}/{fa.sum():3d}/{nc.sum():3d}")

    print()
    print("=" * 104)
    print("[2][N2] 비도통 잔차 — `Cx·dv/dt` 인가, 덧셈인가")
    print("=" * 104)
    print("  잔차를 [dv/dt, 1(상수), 배경파형] 에 최소제곱. 계수와 설명된 분산을 본다.")
    print(f"  {'스냅샷':26s} {'P[W]':>7s} {'ΔCx[nF]':>9s} {'상수[mA]':>9s} {'배경배수':>8s} "
          f"{'설명 R²':>8s} {'남은[mA]':>9s}")
    n2 = {}
    for st, d in store.items():
        pt, r, m = d["pt"], d["r"], d["nc"]
        if m.sum() < 40:
            continue
        dv = (np.roll(pt.v, -1) - np.roll(pt.v, 1)) / 2.0 * (F * NS)
        A = np.stack([dv[m], np.ones(m.sum()), bgw[m]], 1)
        c, *_ = np.linalg.lstsq(A, r[m], rcond=None)
        pred = A @ c
        r2 = 1 - np.sum((r[m] - pred) ** 2) / max(np.sum((r[m] - r[m].mean()) ** 2), 1e-18)
        left = 1000 * np.sqrt(np.mean((r[m] - pred) ** 2))
        n2[st] = dict(dCx=c[0], const=c[1], bg=c[2], r2=float(r2), left=left)
        print(f"  {st:26s} {pt.p_w:7.2f} {1e9*c[0]:8.2f} {1000*c[1]:8.2f} {c[2]:8.2f} "
              f"{r2:8.3f} {left:9.3f}")

    print()
    print("=" * 104)
    print("[3][N3] 도통 에지 — 측정 펄스가 모델보다 일찍/늦게 끝나는가")
    print("=" * 104)
    print("  '무게중심 차' = (시뮬 펄스 무게중심 − 실측 펄스 무게중심), 표본. 양수 = 시뮬이 늦다")
    print(f"  {'스냅샷':26s} {'P[W]':>7s} {'무게중심차':>10s} {'폭 비':>7s} "
          f"{'상승/하강 잔차비':>14s}")
    n3 = {}
    for st, d in store.items():
        pt, x = d["pt"], d["x"]
        m = d["rise"] | d["fall"]
        t = np.arange(NS)
        ai, ax = np.abs(pt.i) * m, np.abs(x) * m
        ci = float(np.sum(t * ai) / max(np.sum(ai), 1e-12))
        cx = float(np.sum(t * ax) / max(np.sum(ax), 1e-12))
        wi = float(np.sqrt(np.sum((t - ci) ** 2 * ai) / max(np.sum(ai), 1e-12)))
        wx = float(np.sqrt(np.sum((t - cx) ** 2 * ax) / max(np.sum(ax), 1e-12)))
        rr = d["r"]
        a = np.sqrt(np.mean(rr[d["rise"]] ** 2)) if d["rise"].sum() else np.nan
        b = np.sqrt(np.mean(rr[d["fall"]] ** 2)) if d["fall"].sum() else np.nan
        n3[st] = dict(dcent=cx - ci, wratio=wx / max(wi, 1e-12), rf=float(a / b))
        print(f"  {st:26s} {pt.p_w:7.2f} {cx-ci:9.3f} {wx/max(wi,1e-12):7.3f} {a/b:13.3f}")

    print()
    print("=" * 104)
    print("[4][N4] 전력 의존 — 덧셈 성분이면 저전력에서 상대적으로 크다")
    print("=" * 104)
    print(f"  {'기기':16s} {'P[W]':>7s} {'|I1|[mA]':>9s} {'전체잔차%':>9s} {'절대잔차[mA]':>12s} "
          f"{'배경/|I1|':>9s}")
    for dev in DEVS:
        for st in raw_snapshots_of(dev):
            if st not in store:
                continue
            d = store[st]
            pt, r = d["pt"], d["r"]
            i1 = abs(phasors(pt.i, 1)[0])
            print(f"  {dev:16s} {pt.p_w:7.2f} {1000*i1:8.1f} "
                  f"{100*np.sqrt(np.mean(r**2))/pt.irms:8.2f}% {1000*np.sqrt(np.mean(r**2)):11.2f} "
                  f"{abs(bg[0])/i1:9.3f}")

    json.dump({"n2": n2, "n3": n3}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
