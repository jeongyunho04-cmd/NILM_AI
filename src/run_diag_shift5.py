# -*- coding: utf-8 -*-
"""세션 이동의 정체 — **형제 지문 실현값**을 깨끗하게 가른다 (13.84.35).

13.84.35 첫 판의 결함 둘을 고친다.
  ⚠ 실수·허수를 한 회귀에 겹쳐 넣어 환경 R² 가 전부 0.000 으로 나왔었다.
    -> 성분마다 따로 맞추고 설명분산을 합산한다 ([[check-the-ruler-against-a-known-value]]).
  ⚠ 관측 페이저에 형제 아닌 기기가 다 섞여 있었다 (기록안 CV h1 66% 가 그 증거).
    -> **형제 하나만** 켜져 있고 나머지 전부 OFF 인 단계만 쓴다.

그러면 산포를 네 조각으로 가를 수 있다:
    ① 기록 안 · 전력대 맞춤   = 시각 잡음 (바닥)
    ② 기록 안 · 전력대 가로질러 = **동작점** 성분 (13.84.32 가 말한 것)
    ③ 기록 간 · 전력대 맞춤   = **어느 녹화를 뽑았나** + 환경
    ④ ③ 중 환경으로 설명되는 몫 = 회로모델이 닿는 곳

    python -X utf8 src/run_diag_shift5.py [--records 3000] [--refresh]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

SIB = ("laptop_charger", "beam_projector")
ORD = [1, 3, 5, 7, 9, 11, 13]


def extract(cache, nrec, out):
    C = Path(cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    sib = [apps.index(x) for x in SIB]
    raw = np.load(C / "raw.npy", mmap_mode="r")
    yon = np.load(C / "y_on.npy", mmap_mode="r")
    ysb = np.load(C / "y_standby.npy", mmap_mode="r")
    ypw = np.load(C / "y_power.npy", mmap_mode="r")
    zg = np.load(C / "z_grid.npy", mmap_mode="r")
    grid = np.arange(meta["target_offset"], int(meta["record_s"] * 60) - 13 * 60 - 1,
                     int(meta["grid_s"] * 60))
    oi = [o - 1 for o in ORD]
    V, ENV, W, SIBID, E, SBW = [], [], [], [], [], []
    for i in range(min(nrec, len(raw))):
        yo = np.asarray(yon[i])
        # **형제 하나만** 켜져 있고 나머지 전부 OFF 인 단계
        tot = yo.sum(1)
        for s, k in enumerate(sib):
            m = (tot == 1) & yo[:, k]
            if not m.any():
                continue
            if "r" not in dir():
                pass
            r = np.asarray(raw[i]); z = np.asarray(zg[i])[0]
            pw = np.asarray(ypw[i]); sb = np.asarray(ysb[i])
            for t in np.nonzero(m)[0]:
                c = grid[t]; seg = slice(max(0, c - 600), c)
                h = r[0:15, seg] + 1j * r[15:30, seg]
                v = (np.median(h.real, 1) + 1j * np.median(h.imag, 1))[oi]
                V.append(v)
                ENV.append([z[0], z[1], np.median(r[32, seg])]
                           + list(np.median(np.abs(r[33:45, seg]), 1)[:6]))
                W.append(float(pw[t, k])); SBW.append(float(sb[t].sum()))
                SIBID.append(s); E.append(i)
            del r
    np.savez_compressed(out, V=np.asarray(V), ENV=np.asarray(ENV), W=np.asarray(W),
                        SBW=np.asarray(SBW), SIB=np.asarray(SIBID), E=np.asarray(E))


def env_r2(X, yc):
    """복소 목표를 **성분마다 따로** 맞추고 설명분산을 합산한다 (교차검증)."""
    num = den = 0.0
    for y in (yc.real, yc.imag):
        p = cross_val_predict(LinearRegression(), X, y, cv=5)
        num += float(np.sum((y - p) ** 2)); den += float(np.sum((y - y.mean()) ** 2))
    return max(1.0 - num / max(den, 1e-30), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=3000)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    sp = Path(os.environ.get("TEMP", ".")) / ("shift5_%d.npz" % a.records)
    if a.refresh or not sp.exists():
        print("raw.npy 훑는 중 (몇 분)...")
        extract(a.cache, a.records, sp)
    d = np.load(sp)
    V, ENV, W, SB, S, E = d["V"], d["ENV"], d["W"], d["SBW"], d["SIB"], d["E"]
    nz = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)

    for s, nm in enumerate(SIB):
        k = (S == s) & (W > 3)
        if k.sum() < 200:
            print("%s: 표본 %d — 건너뜀" % (nm, k.sum()))
            continue
        v, w, g, e = V[k], W[k], ENV[k], E[k]
        pw = v / w[:, None]                            # 와트당 복소 (그 기기 **혼자**)
        q = np.quantile(w, [0, .34, .67, 1.0])
        band = np.clip(np.digitize(w, q[1:3]), 0, 2)
        print("\n== %s == 표본 %d · 기록 %d · W 중앙 %.0f · 전력대 %s"
              % (nm, k.sum(), len(set(e)), np.median(w),
                 " / ".join("%.0f~%.0f" % (q[i], q[i + 1]) for i in range(3))))
        print("  %-5s %9s %9s %9s | %10s %10s"
              % ("차수", "①기록안", "②동작점", "③기록간", "④환경 R²", "**남는 것**"))
        for j, o in enumerate(ORD):
            mag = np.abs(pw[:, j])
            # ① 기록 안 · 전력대 맞춤
            f1 = []
            for ee in set(e):
                for b in range(3):
                    mm = (e == ee) & (band == b)
                    if mm.sum() >= 4:
                        f1.append(np.std(mag[mm]) / (np.mean(mag[mm]) + 1e-12))
            # ② 기록 안 · 전력대 가로질러 (전력대 중앙들의 산포)
            f2 = []
            for ee in set(e):
                cm = [np.median(mag[(e == ee) & (band == b)])
                      for b in range(3) if ((e == ee) & (band == b)).sum() >= 4]
                if len(cm) >= 2:
                    f2.append(np.std(cm) / (np.mean(cm) + 1e-12))
            # ③ 기록 간 · 전력대 맞춤 (가운데 전력대의 기록별 중앙)
            rb, gb = [], []
            for ee in set(e):
                mm = (e == ee) & (band == 1)
                if mm.sum() >= 4:
                    rb.append(np.median(pw[mm, j].real) + 1j * np.median(pw[mm, j].imag))
                    gb.append(np.median(g[mm], 0))
            rb = np.asarray(rb); gb = np.asarray(gb)
            f3 = float(np.std(np.abs(rb)) / (np.mean(np.abs(rb)) + 1e-12)) if len(rb) > 20 else np.nan
            r2 = env_r2(nz(gb), rb) if len(rb) > 40 else np.nan
            print("  h%-4d %8.1f%% %8.1f%% %8.1f%% | %9.3f %9.1f%%"
                  % (o, 100 * np.median(f1), 100 * np.median(f2), 100 * f3,
                     r2, 100 * f3 * np.sqrt(max(1 - r2, 0)) if f3 == f3 else np.nan))
        print("  ①시각잡음 바닥 · ②같은 녹화 안 동작점 이동 · ③다른 녹화를 뽑은 것(+환경)")
        print("  ④ = ③ 중 회로모델(R,X,V,vh)이 설명하는 몫 · 남는 것 = 회로모델 **밖**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
