# -*- coding: utf-8 -*-
"""PC3 를 못 맞히는 것이 **공변량이 h7 에서 끊겨서**인가 (14.11).

사용자: *"PC3 가 계통전압 크기면 이미 15차 이상의 고차 전압 고조파까지 측정하고 있잖아"*

맞는 지적이다. `run_diag_driftlaw.py` 의 `VORD = (3, 5, 7)` 이라 공변량이 8개
[dP, d|V1|, ReV3, ImV3, ReV5, ImV5, ReV7, ImV7] 뿐인데, `voltage_harmonics_complex` 에는
**h15 까지 15차**가 들어 있다. h9·h11·h13·h15 를 통째로 안 쓰고 있었다.
13.84.65 가 *"실측 PC3 는 고차 축이고 계통이 만든다"* 고 쟀으니 더 그렇다.

⚠ 공변량을 늘리면 자리 안 R^2 는 **반드시** 오른다. 그래서 **녹화 하나 빼기(LORO)** 로 잰다.
⚠ 성분별로 본다 — 전체 R^2 는 PC1 이 절반이라 PC3 의 개선을 묻어 버린다
  ([[split-by-class-before-taking-a-median]]).

    python -X utf8 src/run_diag_driftv15.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, load, vec

BLOCK = 600                                   # 10초
BANDS = {"laptop_charger": [(19, 28), (28, 40), (40, 55), (55, 70)],
         "beam_projector": [(35, 45), (45, 55)]}
SETS = {"h7 까지 (현행 VORD)": (3, 5, 7),
        "h15 까지 (+h9·11·13·15)": (3, 5, 7, 9, 11, 13, 15),
        "h7 + |V| 고차크기만": None}          # 고차는 크기만 (위상 없이) — 자유도 절반


def build(vord, mag_only=()):
    """(D, C, tags) — 칸(녹화x전력대) 안 편차."""
    Ds, Cs, tg = [], [], []
    for app, bands in BANDS.items():
        vol = {}
        for p in sorted(glob.glob("processed_data/npz/%s_*.npz" % app)):
            d = np.load(p, allow_pickle=True)
            if "voltage_harmonics_complex" in d.files:
                vol[os.path.basename(p)[:-4]] = np.asarray(d["voltage_harmonics_complex"])
        for stem, P, hc, on, off in load(app):
            if stem not in vol:
                continue
            vc = vol[stem]
            for lo, hi in bands:
                idx = np.nonzero(on & (P >= lo) & (P <= hi))[0]
                if len(idx) < 2 * BLOCK:
                    continue
                Y, X = [], []
                for s in range(0, len(idx) - BLOCK + 1, BLOCK):
                    j = idx[s:s + BLOCK]
                    Y.append(vec(hc[j].mean(0)))
                    v = vc[j].mean(0)
                    x = [P[j].mean(), np.abs(v[0])]
                    for h in vord:
                        x += [v[h - 1].real, v[h - 1].imag]
                    for h in mag_only:
                        x += [np.abs(v[h - 1])]
                    X.append(x)
                Y = np.asarray(Y); X = np.asarray(X, float)
                if len(Y) < 2:
                    continue
                Ds.append(Y - Y.mean(0)); Cs.append(X - X.mean(0)); tg += [stem] * len(Y)
    return np.vstack(Ds), np.vstack(Cs), np.asarray(tg)


def loro_resid(D, C, tags):
    """녹화 하나 빼기 예측 잔차."""
    R = np.zeros_like(D)
    for held in np.unique(tags):
        tr, te = tags != held, tags == held
        if tr.sum() < C.shape[1] + 2 or te.sum() < 1:
            R[te] = D[te]
            continue
        sc = C[tr].std(0); sc[sc == 0] = 1
        B = np.linalg.lstsq(C[tr] / sc, D[tr], rcond=None)[0] / sc[:, None]
        R[te] = D[te] - C[te] @ B
    return R


def main():
    # 기준 주성분은 **현행 공변량 표본**에서 한 번만 뽑는다 (자가 비교가 아니게)
    D0, _, tags0 = build((3, 5, 7))
    Vt = np.linalg.svd(D0 - D0.mean(0), full_matrices=False)[2]
    ev = (np.linalg.svd(D0 - D0.mean(0), compute_uv=False) ** 2)
    ev = ev / ev.sum()
    print("표본 %d블록 · 녹화 %d · 성분 설명분산 %s"
          % (len(D0), len(np.unique(tags0)), np.round(np.cumsum(ev)[:3], 3)))
    print("\n**성분별 LORO R^2** — 녹화 하나 빼기라 자유도가 공짜가 아니다")
    print("  %-26s %5s %9s %9s %9s %9s"
          % ("공변량", "개수", "전체", "PC1", "PC2", "PC3"))

    for name, vord in SETS.items():
        if vord is None:
            D, C, tags = build((3, 5, 7), mag_only=(9, 11, 13, 15))
        else:
            D, C, tags = build(vord)
        if len(D) != len(D0):
            print("  %-26s 표본이 달라 건너뜀 (%d)" % (name, len(D)))
            continue
        R = loro_resid(D, C, tags)
        row = "  %-26s %5d" % (name, C.shape[1])
        tot = 1 - (R ** 2).sum() / (D ** 2).sum()
        row += " %8.3f" % tot
        for i in range(3):
            a = D @ Vt[i]; b = R @ Vt[i]
            row += " %8.3f" % (1 - (b ** 2).sum() / max((a ** 2).sum(), 1e-30))
        print(row)

    # ── 고차 전압이 실제로 움직이나 (안 움직이면 공변량을 더해도 소용없다) ──
    print("\n참고 — 블록 사이 **전압 고조파 변동** (칸 안 표준편차, V1 대비 ppm)")
    D, C, tags = build((3, 5, 7, 9, 11, 13, 15))
    names = ["dP", "d|V1|"] + sum([["ReV%d" % h, "ImV%d" % h]
                                   for h in (3, 5, 7, 9, 11, 13, 15)], [])
    v1 = np.abs(C[:, 1]).mean() + 220.0
    for i, n in enumerate(names):
        if i < 2:
            continue
        print("   %-8s %8.1f ppm" % (n, 1e6 * C[:, i].std() / v1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
