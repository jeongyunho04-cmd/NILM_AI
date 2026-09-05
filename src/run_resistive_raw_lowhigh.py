# -*- coding: utf-8 -*-
"""포트 원시에서 LOW / HIGH 경로의 v–i 정렬을 **따로** 잰다.

포트는 순저항이라 `i(t) = v(t)/R` 이 모든 표본에서 성립하고, 파일 안에 `range` 라벨이
표본마다 있다 (영교차 근처 14% 가 LOW). 그러니 **한 녹화로 두 경로를 동시에** 잰다.
회로 모델도 par5 도 필요 없다.
"""
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.synthesis.fit_raw import halfwave, shift_samples, NS

DEG = 360 * 60 / 15360          # 표본 하나 = 1.406°/차수


def best_shift(v, i, mask, grid=np.arange(-6, 4.001, 0.005)):
    """`i` 를 얼마나 옮겨야 `v` 와 가장 닮는가 (mask 안에서만). 반환 (표본, 상관, n)."""
    if mask.sum() < 12:
        return np.nan, np.nan, int(mask.sum())
    best = (-9.0, np.nan)
    for n in grid:
        c = float(np.corrcoef(v[mask], shift_samples(i, -n)[mask])[0, 1])
        if c > best[0]:
            best = (c, float(n))
    return best[1], best[0], int(mask.sum())


print("=" * 96)
print("포트 원시 — LOW / HIGH 경로의 v–i 정렬 (순저항, 모델 없음)")
print("=" * 96)
print("  부호 규약: 음수 = 전류가 전압보다 **빠르다** (i 를 늦춰야 맞는다)")
print(f"  {'스냅샷':24s} {'경로':6s} {'표본':>5s} {'스큐[표본]':>10s} {'°/차수':>9s} {'상관':>9s}")
out = {}
for st in ("raw_elcetric_kettle_1", "raw_elcetric_kettle_2"):
    d = pd.read_csv(f"data/{st}.csv", usecols=["cyc", "n", "i_a", "v_v", "range"])
    nc = int(d["cyc"].max()) + 1
    V = d["v_v"].to_numpy(float).reshape(nc, NS)
    I = d["i_a"].to_numpy(float).reshape(nc, NS)
    R = d["range"].to_numpy(int).reshape(nc, NS)
    v, i = halfwave(np.median(V, 0)), halfwave(np.median(I, 0))
    # 표본별 다수결 라벨 (40주기 중 LOW 가 과반이면 LOW)
    lo = (R == 0).mean(0) > 0.5
    hi = (R != 0).mean(0) > 0.5
    for lab, m in (("전체", np.ones(NS, bool)), ("LOW", lo), ("HIGH", hi)):
        n, c, cnt = best_shift(v, i, m)
        out.setdefault(st, {})[lab] = n
        print(f"  {st:24s} {lab:6s} {cnt:5d} {n:9.3f} {n * DEG:8.3f}° {c:9.6f}")

print()
print("=" * 96)
print("펌웨어 교정값과 견주기")
print("=" * 96)
print(f"  {'':24s} {'측정 °/차수':>12s} {'펌웨어':>9s} {'차':>8s}")
for st in out:
    for lab, cal in (("LOW", 0.44), ("HIGH", 2.62)):
        m = out[st].get(lab, np.nan)
        print(f"  {st:24s} {lab:5s} {abs(m) * DEG:9.3f}° {cal:8.2f}° {abs(m) * DEG - cal:+7.3f}°")
print()
print("  ⚠ LOW 표본은 영교차 근처 14% 뿐이다 — 기울기가 가팔라 타이밍에는 유리하지만")
print("     표본 수가 적어 산포가 크다. HIGH 가 정본 판정이다.")
