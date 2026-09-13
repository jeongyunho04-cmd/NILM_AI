# -*- coding: utf-8 -*-
"""13.84.35 자 대기 + 오염원 가르기.

프로젝터는 46W 고정인데 와트당 h1 기록 간 CV 가 49.5% 였다. 물리로 말이 안 된다.
[[check-the-ruler-against-a-known-value]] — 먼저 **아는 값**(단독 녹화에서 만든 `sig`)에 댄다.
그 다음 오염원 둘을 가른다:
  A) **대기·상시배경** — `tot==1` 은 ON 이 하나라는 뜻이지 다른 기기가 꽂혀 있지 않다는 뜻이 아니다.
     꽂힌 구성이 기록마다 다르면 그 자체가 기록 간 산포를 만든다. 회로모델과 무관하다.
  B) **증분** Δ|I|/ΔW — 기록 안 전력대 차분은 상수 배경을 **양변에서 지운다**.
     이것으로 재서 산포가 줄면 범인은 상수항이다.

    python -X utf8 src/run_diag_shift6.py
"""
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
C = Path("cache/seqraw_v1")


def env_r2(X, yc):
    num = den = 0.0
    for y in (yc.real, yc.imag):
        p = cross_val_predict(LinearRegression(), X, y, cv=5)
        num += float(np.sum((y - p) ** 2)); den += float(np.sum((y - y.mean()) ** 2))
    return max(1.0 - num / max(den, 1e-30), 0.0)


def main():
    d = np.load(Path(os.environ.get("TEMP", ".")) / "shift5_3000.npz")
    V, ENV, W, S, E = d["V"], d["ENV"], d["W"], d["SIB"], d["E"]
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    sib = [apps.index(x) for x in SIB]
    yon = np.load(C / "y_on.npy", mmap_mode="r")
    ypl = np.load(C / "y_plugged.npy", mmap_mode="r")
    ysb = np.load(C / "y_standby.npy", mmap_mode="r")

    # 추출과 **같은 순서**로 단계 인덱스를 되살린다 (raw 를 다시 안 읽는다)
    T = []
    for i in range(len(yon)):
        yo = np.asarray(yon[i]); tot = yo.sum(1)
        for s, k in enumerate(sib):
            m = (tot == 1) & yo[:, k]
            if m.any():
                T.extend(np.nonzero(m)[0])
    T = np.asarray(T)
    assert len(T) == len(E), (len(T), len(E))
    PL = np.zeros((len(T), len(apps)), np.int8)
    SBW = np.zeros((len(T), len(apps)), np.float32)
    pos = 0
    for i in range(len(yon)):
        yo = np.asarray(yon[i]); tot = yo.sum(1)
        need = sum(int(((tot == 1) & yo[:, k]).sum()) for k in sib)
        if not need:
            continue
        pl = np.asarray(ypl[i]); sb = np.asarray(ysb[i])
        for s, k in enumerate(sib):
            idx = np.nonzero((tot == 1) & yo[:, k])[0]
            for t in idx:
                PL[pos] = pl[t]; SBW[pos] = sb[t]; pos += 1
    assert pos == len(E)

    # ── 자 대기: 단독 녹화에서 만든 지문과 견준다 ───────────────────────────
    from src.model.net import harmonic_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="all", carrier_apps=("oven",))
    sg = harmonic_signatures(pool, apps)
    print("자 대기 — 와트당 |I1| (mA/W)")
    print("  %-16s %12s %12s %12s" % ("기기", "단독 녹화 sig", "캐시 중앙", "비"))
    for s, nm in enumerate(SIB):
        k = (S == s) & (W > 20)
        ref = float(np.hypot(sg[apps.index(nm), 0, 0], sg[apps.index(nm), 0, 1])) * 1000
        got = float(np.median(np.abs(V[k, 0] / W[k]))) * 1000
        print("  %-16s %12.2f %12.2f %12.2f" % (nm, ref, got, got / max(ref, 1e-9)))

    print("\nA) 꽂힌 구성·대기가 기록 간 산포를 만드나 (가운데 전력대, 기록별 중앙)")
    print("  %-16s %-5s %9s | %9s %9s %9s"
          % ("기기", "차수", "기록간 CV", "환경만", "+대기·꽂힘", "**늘어난 몫**"))
    for s, nm in enumerate(SIB):
        k = (S == s) & (W > 3)
        v, w, g, e = V[k], W[k], ENV[k], E[k]
        pl, sw = PL[k], SBW[k]
        q = np.quantile(w, [0, .34, .67, 1.0])
        band = np.clip(np.digitize(w, q[1:3]), 0, 2)
        rows = [(ee, (e == ee) & (band == 1)) for ee in set(e)]
        rows = [(ee, m) for ee, m in rows if m.sum() >= 4]
        if len(rows) < 60:
            continue
        G = np.stack([np.median(g[m], 0) for _, m in rows])
        P2 = np.stack([np.r_[pl[m][0], sw[m].sum(1).mean()] for _, m in rows])
        nzf = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
        for j, o in enumerate(ORD):
            rb = np.asarray([np.median((v[m, j] / w[m]).real)
                             + 1j * np.median((v[m, j] / w[m]).imag) for _, m in rows])
            cv = float(np.std(np.abs(rb)) / (np.mean(np.abs(rb)) + 1e-12))
            r1 = env_r2(nzf(G), rb)
            r2 = env_r2(np.c_[nzf(G), nzf(P2)], rb)
            print("  %-16s h%-4d %8.1f%% | %8.3f %9.3f %9.3f"
                  % (nm if j == 0 else "", o, 100 * cv, r1, r2, r2 - r1))

    print("\nB) 증분 Δ|I|/ΔW — 기록 안 전력대 차분이 상수 배경을 지운다 (충전기만)")
    s = 0
    k = (S == s) & (W > 3)
    v, w, e = V[k], W[k], E[k]
    q = np.quantile(w, [0, .34, .67, 1.0])
    band = np.clip(np.digitize(w, q[1:3]), 0, 2)
    print("  %-5s %12s %12s" % ("차수", "와트당 CV", "**증분 CV**"))
    for j, o in enumerate(ORD):
        pw, inc = [], []
        for ee in set(e):
            mlo = (e == ee) & (band == 0); mhi = (e == ee) & (band == 2)
            m1 = (e == ee) & (band == 1)
            if m1.sum() >= 4:
                pw.append(np.abs(np.median((v[m1, j] / w[m1]).real)
                                 + 1j * np.median((v[m1, j] / w[m1]).imag)))
            if mlo.sum() >= 4 and mhi.sum() >= 4:
                dv = ((np.median(v[mhi, j].real) + 1j * np.median(v[mhi, j].imag))
                      - (np.median(v[mlo, j].real) + 1j * np.median(v[mlo, j].imag)))
                dw = np.median(w[mhi]) - np.median(w[mlo])
                if dw > 5:
                    inc.append(abs(dv / dw))
        pw = np.asarray(pw); inc = np.asarray(inc)
        print("  h%-4d %11.1f%% %11.1f%%  (기록 %d / %d)"
              % (o, 100 * pw.std() / (pw.mean() + 1e-12),
                 100 * inc.std() / (inc.mean() + 1e-12), len(pw), len(inc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
