# -*- coding: utf-8 -*-
"""기록마다 **형제 지문이 실제로 어떻게 실현되나** — 세션 이동의 정체 (13.84.35).

13.84.35-ⓒ 에서 기록 간 수준 이동의 **95%** 가 환경(R,X,V,vh)으로 설명 안 됐다.
회로모델은 `(R,X,V)` 의 결정적 함수이므로 **이동은 회로모델이 아니다.** 그럼 무엇인가.

기록마다 형제만 켜진 구간(미니PC 참 OFF)에서 **와트당 페이저**를 뽑는다. 이것이
그 기록에서 손실이 상수로 쓰는 `sig` 가 실제로는 무엇이었는지다. 그 산포를 가른다:
    기록 간 산포  =  [환경으로 설명되는 몫]  +  [어느 녹화·동작점을 뽑았나]
                       회로모델이 닿음            회로모델이 못 닿음
바닥은 **기록 안 산포**다 (같은 기록·같은 녹화·같은 환경에서 시각만 다른 것).

첫 실행은 `raw.npy` 를 훑느라 몇 분 걸린다. 뽑은 것은 scratch 에 적어 두고 다시 쓴다.

    python -X utf8 src/run_diag_shift4.py [--records 1200] [--refresh]
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

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

SIB = ("laptop_charger", "beam_projector")
ORD = [1, 3, 5, 7, 9, 11, 13]        # 1-기반 차수


def extract(cache, nrec, out):
    C = Path(cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    km = apps.index("minipc")
    sib = [apps.index(x) for x in SIB]
    raw = np.load(C / "raw.npy", mmap_mode="r")
    yon = np.load(C / "y_on.npy", mmap_mode="r")
    ypw = np.load(C / "y_power.npy", mmap_mode="r")
    zg = np.load(C / "z_grid.npy", mmap_mode="r")
    grid = np.arange(meta["target_offset"], int(meta["record_s"] * 60) - 13 * 60 - 1,
                     int(meta["grid_s"] * 60))
    oi = [o - 1 for o in ORD]
    V, ENV, SW, MW, Y, E = [], [], [], [], [], []
    for i in range(min(nrec, len(raw))):
        yo = np.asarray(yon[i])
        m = yo[:, sib].sum(1) > 0
        if m.sum() < 6:
            continue
        r = np.asarray(raw[i]); z = np.asarray(zg[i])[0]; pw = np.asarray(ypw[i])
        for t in np.nonzero(m)[0][::6]:
            c = grid[t]; seg = slice(max(0, c - 600), c)
            h = r[0:15, seg] + 1j * r[15:30, seg]
            v = (np.median(h.real, 1) + 1j * np.median(h.imag, 1))[oi]
            V.append(v)
            ENV.append([z[0], z[1], np.median(r[32, seg])]
                       + list(np.median(np.abs(r[33:45, seg]), 1)[:6]))
            SW.append(float(pw[t, sib].sum())); MW.append(float(pw[t, km]))
            Y.append(int(yo[t, km])); E.append(i)
    np.savez_compressed(out, V=np.asarray(V), ENV=np.asarray(ENV), SW=np.asarray(SW),
                        MW=np.asarray(MW), Y=np.asarray(Y), E=np.asarray(E))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=1200)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    sp = Path(os.environ.get("TEMP", ".")) / ("shift4_%d.npz" % a.records)
    if a.refresh or not sp.exists():
        print("raw.npy 훑는 중 (몇 분)...")
        extract(a.cache, a.records, sp)
    d = np.load(sp)
    V, ENV, SW, MW, Y, E = d["V"], d["ENV"], d["SW"], d["MW"], d["Y"], d["E"]
    nz = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
    print("표본 %d · 기록 %d · 형제 W 중앙 %.0f · 미니PC ON %d"
          % (len(Y), len(set(E)), np.median(SW), Y.sum()))

    # ── 형제만 켜진 표본으로 **와트당 페이저**를 뽑는다 ───────────────────────
    m0 = (Y == 0) & (SW > 5)
    pw = V[m0] / SW[m0][:, None]                       # (n, O) 복소 와트당
    Em, ENVm = E[m0], ENV[m0]
    recs = np.array(sorted(set(Em)))
    per_rec, per_env, per_in = [], [], []
    for e in recs:
        k = Em == e
        if k.sum() < 4:
            continue
        per_rec.append(np.median(pw[k].real, 0) + 1j * np.median(pw[k].imag, 0))
        per_env.append(np.median(ENVm[k], 0))
        per_in.append(np.std(np.abs(pw[k]), 0) / (np.abs(per_rec[-1]) + 1e-12))
    R = np.asarray(per_rec); G = np.asarray(per_env); IN = np.asarray(per_in)
    print("   기록 %d개에서 형제 와트당 페이저를 뽑았다 (미니PC 참OFF · 형제 ON)\n" % len(R))

    print("차수별 — 형제 **와트당 지문**이 기록마다 얼마나 다른가")
    print("  %-5s %10s %10s | %12s %12s %10s"
          % ("차수", "기록안 CV", "기록간 CV", "환경 설명 R²", "**남는 CV**", "환경 몫"))
    keep = []
    for j, o in enumerate(ORD):
        mag = np.abs(R[:, j])
        cv_b = float(np.std(mag) / (np.mean(mag) + 1e-12))
        cv_w = float(np.median(IN[:, j]))
        X = nz(G)
        yv = np.concatenate([R[:, j].real, R[:, j].imag])
        XX = np.concatenate([X, X], 0)
        pr = cross_val_predict(LinearRegression(), XX, yv, cv=5)
        ss = 1.0 - np.sum((yv - pr) ** 2) / np.sum((yv - yv.mean()) ** 2)
        ss = max(ss, 0.0)
        res = cv_b * np.sqrt(max(1.0 - ss, 0.0))
        keep.append((o, cv_w, cv_b, ss, res))
        print("  h%-4d %9.1f%% %9.1f%% | %11.3f %11.1f%% %9.1f%%"
              % (o, 100 * cv_w, 100 * cv_b, ss, 100 * res, 100 * ss))
    print("\n  '기록안 CV' = 같은 기록·같은 녹화·같은 환경에서 시각만 다른 것 = **바닥**")
    print("  '환경 설명 R²' = 회로모델이 닿는 몫 (교차검증). '남는 CV' 가 회로모델 **밖**이다")

    # ── 그 실현 지문이 그 기록의 판별 수준을 설명하나 ────────────────────────
    BASE = np.stack([np.log(np.abs(V[:, 0]) + 1e-9), np.angle(V[:, 0]),
                     np.abs(V[:, 1]) / (np.abs(V[:, 0]) + 1e-9),
                     np.abs(V[:, 2]) / (np.abs(V[:, 0]) + 1e-9),
                     np.abs(V[:, 3]) / (np.abs(V[:, 0]) + 1e-9),
                     np.angle(V[:, 1] * np.conj(V[:, 0]))], 1)
    Bn = nz(BASE)
    rng = np.random.RandomState(0); rr = np.array(sorted(set(E))); rng.shuffle(rr)
    tr_r = set(rr[:len(rr) * 3 // 4])
    tr = np.array([x in tr_r for x in E]); te = ~tr
    lr = LogisticRegression(max_iter=6000, C=0.5).fit(Bn[tr], Y[tr])
    d0 = Bn @ lr.coef_[0] + lr.intercept_[0]
    print("\n기록 수준을 무엇이 설명하나 (기록별 판별값 중앙)")
    lvl, F = [], []
    ok = []
    for e, rv, gv in zip(recs[[len(np.nonzero(Em == e)[0]) >= 4 for e in recs]], R, G):
        k = E == e
        if k.sum() >= 8:
            ok.append(True); lvl.append(np.median(d0[k]))
            F.append(list(np.abs(rv) / (np.abs(rv[0]) + 1e-12))[1:] + list(gv)
                     + [np.median(SW[k])])
        else:
            ok.append(False)
    lvl = np.asarray(lvl); F = np.asarray(F)
    no = len(ORD) - 1
    for lbl, cols in (("환경만 (R,X,V,vh)", list(range(no, no + 9))),
                      ("형제 W 만", [F.shape[1] - 1]),
                      ("**실현 지문 모양만** (h3/h1~h13/h1)", list(range(no))),
                      ("실현 지문 + 환경 + W", list(range(F.shape[1])))):
        X = nz(F[:, cols])
        pr = cross_val_predict(LinearRegression(), X, lvl, cv=5)
        r2 = 1.0 - np.sum((lvl - pr) ** 2) / np.sum((lvl - lvl.mean()) ** 2)
        print("   %-36s 교차검증 R² %+.3f" % (lbl, r2))

    # ── 수준을 **라벨 없이** 빼면 순위가 돌아오나 ───────────────────────────
    print("\n기록 수준을 **라벨 없이** 빼면 (그 기록 판별값의 중앙을 0 으로)")
    dd = d0.copy()
    for e in set(E):
        k = E == e
        if k.sum() >= 4:
            dd[k] = d0[k] - np.median(d0[k])
    print("   보정 없음        AUC %.3f" % roc_auc_score(Y[te], d0[te]))
    print("   기록 중앙 빼기   AUC %.3f" % roc_auc_score(Y[te], dd[te]))
    print("   (참고) 기록 안에서만 채점 상한 0.767 · 기록마다 경계 재적합 0.879")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
