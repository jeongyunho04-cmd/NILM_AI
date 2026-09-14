# -*- coding: utf-8 -*-
"""기기·**상태별** 전압 지수를 **실측**에서 잰다 (14.25).

사용자: *"FAN_LIGHT 는 애초에 저항성이 아닐 건데 어떻게 학습한 거야?"* -> *"생성기 상태별
전압 지수부터 고쳐."*

지금 생성기는 지수를 **부하 분류별**로만 준다:
```
grid_simulator._LOAD_EXPONENTS
  RESISTIVE (1.0, 2.0) · SMPS (−1.0, 0.0) · MOTOR (0.7, 0.7) · PASSIVE (1.0, 1.0)
```
그래서 오븐의 **모든** 상태가 `P ∝ V²` 를 받는다. 그런데 FAN_LIGHT 는 팬+조명이라 2 가 아니다
(14.24 가 실측 11블록에서 **1.65** 를 쟀다).

⚠ 14.7 이 "상태마다 재도 기기 안에서는 일치한다" 고 적은 것은 **순환**이었다 — 생성기가
   2.0 을 걸어 둔 **합성 캐시**에서 2.0 을 확인한 것이다. 여기서는 `processed_data/npz`
   원본만 쓴다. [[check-the-ruler-against-a-known-value]]

품질 문턱 (이걸 안 걸면 `oven_2` 처럼 블록 4개로 −2.86 이 나온다):
  · 블록 **>= 8** (60초 블록)
  · 그 녹화 안에서 **전압이 실제로 움직여야** 한다 — `std(log V) >= 0.002` (약 0.2%)
  · `|r| >= 0.5`
  · 통과한 녹화만 **블록 수로 가중 평균**한다

    python -X utf8 src/run_diag_vexp_state.py [--block 3600] [--min-blocks 8]
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=3600, help="블록 길이 (사이클). 3600=60초")
    ap.add_argument("--min-blocks", type=int, default=8)
    ap.add_argument("--min-vstd", type=float, default=0.002)
    ap.add_argument("--min-r", type=float, default=0.5)
    a = ap.parse_args()

    from src.synthesis.grid_simulator import GridSimulator
    from src.preprocessing.file_registry import get_load_class  # noqa: F401
    gs = GridSimulator()

    rows = defaultdict(list)
    skipped = defaultdict(int)
    for p in sorted(glob.glob("processed_data/npz/*.npz")):
        stem = os.path.basename(p)[:-4]
        app = stem.rsplit("_", 1)[0]
        d = np.load(p, allow_pickle=True)
        if "state_id" not in d.files or "voltage_harmonics_complex" not in d.files:
            continue
        P = np.asarray(d["p_denoised_w"])
        st = np.asarray(d["state_id"])
        V = np.abs(np.asarray(d["voltage_harmonics_complex"])[:, 0])
        ok = (np.asarray(d["is_valid"]) == 1) & (np.asarray(d["is_unplugged"]) == 0)
        for s in sorted(set(st.tolist())):
            if s <= 0:
                continue
            m = ok & (st == s) & (P > 0.5) & (V > 150)
            idx = np.nonzero(m)[0]
            nb = len(idx) // a.block
            if nb < a.min_blocks:
                skipped[(app, s)] += 1
                continue
            pb = np.array([P[idx[i * a.block:(i + 1) * a.block]].mean() for i in range(nb)])
            vb = np.array([V[idx[i * a.block:(i + 1) * a.block]].mean() for i in range(nb)])
            lv = np.log(vb / np.median(vb)); lp = np.log(pb / np.median(pb))
            if lv.std() < a.min_vstd:
                skipped[(app, s)] += 1
                continue
            e = float(np.polyfit(lv, lp, 1)[0]); r = float(np.corrcoef(lv, lp)[0, 1])
            if abs(r) < a.min_r:
                skipped[(app, s)] += 1
                continue
            rows[(app, s)].append((stem, nb, float(np.median(pb)), e, r, float(lv.std())))

    print("**실측** 기기·상태별 전압 지수 `P ∝ V^e` — 60초 블록 · 원본 npz")
    print("품질 문턱: 블록>=%d · std(logV)>=%.3f · |r|>=%.1f\n"
          % (a.min_blocks, a.min_vstd, a.min_r))
    print("  %-20s %4s %7s %9s %9s %9s %8s %10s"
          % ("기기", "상태", "녹화", "블록합", "중앙 W", "**e**", "생성기", "차이"))
    table = {}
    for key in sorted(rows):
        app, s = key
        v = rows[key]
        w = np.array([x[1] for x in v], float)
        e = np.array([x[3] for x in v])
        ew = float((e * w).sum() / w.sum())
        cur = gs.power_voltage_exponent(app)
        table[key] = (ew, int(w.sum()), len(v))
        print("  %-20s %4d %7d %9d %9.1f %9.2f %8.1f %+10.2f"
              % (app, s, len(v), int(w.sum()), np.median([x[2] for x in v]), ew, cur, ew - cur))
    if skipped:
        print("\n  문턱에 걸려 버린 (기기, 상태): %s"
              % ", ".join("%s s%d x%d" % (k[0], k[1], n) for k, n in sorted(skipped.items())))
    print("\n  ⚠ 녹화 하나짜리는 값이 흔들린다 — 녹화 수를 같이 보고 쓴다.")
    print("  ⚠ 통과한 것만 덮어쓰고 나머지는 부하 분류 기본값을 그대로 둔다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
