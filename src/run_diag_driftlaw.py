# -*- coding: utf-8 -*-
"""표류에 **규칙이 있는가** — 물리 공변량으로 회귀한다 (13.84.56).

사용자: *"회로모델이 있고 표류라고는 하지만 회로가 달라지는 건 아닐 테고, 표류가
3차원이라니 분명히 뭔가 규칙이 있다는 의미다. 해석적으로 해볼 여지가 있어 보인다."*

옳다. 13.84.52 는 표류가 **3차원(94%)** 이라고만 재고 그 셋이 **무엇인지**는 안 물었다.
회로가 고정이면 지문이 움직일 수 있는 길은 **동작점**뿐이다:

    I_h = f(theta_회로 ; P, V_1, V_3, V_5, V_7, ...)

theta 는 안 변한다. 변하는 것은 **부하 전력 P** 와 **계통 전압 파형 V** 다. 그리고 V 는
자료에 이미 있다 — `voltage_harmonics_complex` 가 사이클마다 15차까지, arg(V_1)=0 로.

재는 것:
  1 설명력  표류 16차원의 몇 %를 [dP, d|V1|, dV3, dV5, dV7] 가 설명하나
  2 정체    PC1/PC2/PC3 각각이 어느 공변량과 몇 도인가
  3 부분공간  물리 기저 vs PCA 기저 — 미니PC 를 얼마나 남기고 표류를 얼마나 지우나
  4 전이성  **녹화 하나 빼기**. 사영(지우기) 셋째 길로 **예측 빼기**까지 견준다

핵심은 셋째 길이다. 사영은 16차원 중 k 개를 통째로 버리므로 미니PC 도 같이 깎이지만,
**예측 빼기** `D - X B` 는 덧셈 보정이라 어느 방향도 안 버린다. 미니PC 는 100% 남는다.
그리고 계통 전압은 분해와 무관하게 측정되므로 (외생) 채점 시점에 바로 쓸 수 있다.

    python -X utf8 src/run_diag_driftlaw.py [--block 600]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftdim import ORD, load, vec

#: 회귀에 쓸 계통 전압 차수 (복소 -> Re/Im 두 열)
VORD = (3, 5, 7)
SOURCES = ("laptop_charger", "beam_projector")
BANDS = {"laptop_charger": [(19, 28), (28, 40), (40, 55), (55, 70)],
         "beam_projector": [(35, 45), (45, 55)]}


def cov_names():
    n = ["dP", "d|V1|"]
    for h in VORD:
        n += ["ReV%d" % h, "ImV%d" % h]
    return n


def blocks_xy(P, hc, vc, m, lo, hi, B):
    """전력대 안 B사이클 블록마다 (지문 16차원, 공변량)."""
    idx = np.nonzero(m & (P >= lo) & (P <= hi))[0]
    if len(idx) < B:
        return np.zeros((0, 2 * len(ORD))), np.zeros((0, 2 + 2 * len(VORD)))
    Y, X = [], []
    for s in range(0, len(idx) - B + 1, B):
        j = idx[s:s + B]
        Y.append(vec(hc[j].mean(0)))
        v = vc[j].mean(0)
        x = [P[j].mean(), np.abs(v[0])]
        for h in VORD:
            x += [v[h - 1].real, v[h - 1].imag]
        X.append(x)
    return np.asarray(Y), np.asarray(X, float)


def load_v(app):
    """녹화별 전압 고조파."""
    out = {}
    for p in sorted(glob.glob("processed_data/npz/%s_*.npz" % app)):
        d = np.load(p, allow_pickle=True)
        if "voltage_harmonics_complex" in d.files:
            out[os.path.basename(p)[:-4]] = np.asarray(d["voltage_harmonics_complex"])
    return out


def minipc_sig():
    """미니PC 신호 = ON(8~14W) - 대기 (13.84.51). 녹화 평균."""
    sig = []
    for stem, P, hc, on, off in load("minipc"):
        a = (P >= 8) & (P <= 14) & on
        if a.sum() < 300 or off.sum() < 300:
            continue
        sig.append(vec((np.median(hc[a].real, 0) + 1j * np.median(hc[a].imag, 0))
                       - (np.median(hc[off].real, 0) + 1j * np.median(hc[off].imag, 0))))
    return np.mean(sig, 0)


def fit(X, D):
    """열 표준화한 최소제곱. 반환 B (원래 단위)."""
    sc = X.std(0); sc[sc == 0] = 1
    return np.linalg.lstsq(X / sc, D, rcond=None)[0] / sc[:, None]


def sub_of(X, B, k):
    """물리 모형이 실제로 쓰는 k차원 부분공간 = 적합값의 주성분."""
    Dh = X @ B
    return np.linalg.svd(Dh - Dh.mean(0), full_matrices=False)[2][:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=600, help="블록 길이 (사이클). 600=10초")
    a = ap.parse_args()

    NAMES = cov_names()
    Ys, Xs, tags, cells = [], [], [], []
    for app in SOURCES:
        VH = load_v(app)
        for band in BANDS[app]:
            for stem, P, hc, on, off in load(app):
                if stem not in VH:
                    continue
                Y, X = blocks_xy(P, hc, VH[stem], on, *band, B=a.block)
                if len(Y) < 3:
                    continue
                Ys.append(Y - Y.mean(0)); Xs.append(X - X.mean(0))   # 칸 안에서 중심화
                tags += [stem] * len(Y)
                cells += ["%s|%d" % (stem, band[0])] * len(Y)
    if not Ys:
        print("블록이 없다"); return 1
    D = np.vstack(Ys); X = np.vstack(Xs)
    tags, cells = np.asarray(tags), np.asarray(cells)
    tot = float((D ** 2).sum())
    d0 = float(np.sqrt((D ** 2).sum(1).mean()))
    print("표류 표본 %d블록 (%d사이클=%.0f초) · 녹화 %d개 · 칸 %d개"
          % (len(D), a.block, a.block / 60.0, len(np.unique(tags)), len(np.unique(cells))))
    print("   표류 RMS %.1f mA" % (1000 * d0))
    print("   공변량 표준편차  %s"
          % "  ".join("%s %.3g" % (n, X[:, i].std()) for i, n in enumerate(NAMES)))

    def r2(cols):
        B = fit(X[:, cols], D)
        return 1.0 - float(((D - X[:, cols] @ B) ** 2).sum()) / tot

    # ---- 1 설명력 --------------------------------------------------------
    print("\n[1] 표류 16차원을 물리 공변량이 얼마나 설명하나 (에너지 R^2)")
    print("   %-8s %8s %8s" % ("공변량", "단독", "누적"))
    used = []
    for i, n in enumerate(NAMES):
        used = used + [i]
        print("   %-8s %7.3f  %7.3f" % (n, r2([i]), r2(used)))
    Bfull = fit(X, D)
    R2all = r2(list(range(len(NAMES))))
    EXO = [i for i, n in enumerate(NAMES) if n != "dP"]
    print("   %-8s %8s %7.3f  <- 전부" % ("", "", R2all))
    print("   외생(계통 전압)만 %.3f  ·  dP 만 %.3f   "
          "-- 외생은 분해기의 전력 추정이 **필요 없다**" % (r2(EXO), r2([0])))

    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    ev = np.cumsum(S ** 2 / (S ** 2).sum())
    Dh = X @ Bfull
    sf = np.linalg.svd(Dh - Dh.mean(0), full_matrices=False)[1]
    print("   표류 PCA 누적분산      %s" % "  ".join("PC%d %.2f" % (i + 1, v) for i, v in enumerate(ev[:5])))
    print("   물리 적합값 누적분산   %s" % "  ".join("PC%d %.2f" % (i + 1, v) for i, v in
                                                 enumerate(np.cumsum(sf ** 2 / (sf ** 2).sum())[:5])))

    # ---- 2 PC 의 정체 ----------------------------------------------------
    Bn = Bfull / np.maximum(np.linalg.norm(Bfull, axis=1, keepdims=True), 1e-30)
    print("\n[2] 주성분은 **무엇인가** — PC 와 각 공변량 응답 방향의 각도")
    print("   %-5s %6s | %s" % ("PC", "분산", "".join("%-8s" % n for n in NAMES)))
    for i in range(min(4, len(Vt))):
        ang = [np.degrees(np.arccos(min(1.0, abs(float(Vt[i] @ Bn[j]))))) for j in range(len(NAMES))]
        j = int(np.argmin(ang))
        print("   PC%-3d %5.2f  | %s  -> **%s**"
              % (i + 1, S[i] ** 2 / (S ** 2).sum(),
                 "".join("%5.0f   " % v for v in ang), NAMES[j]))

    # ---- 3 부분공간 견줌 -------------------------------------------------
    ms = minipc_sig(); msn = ms / np.linalg.norm(ms)
    print("\n[3] 같은 자료 안에서 — 물리 기저(적합값 주성분) vs PCA 기저")
    print("   %-4s %-30s %s" % ("k", "물리", "PCA"))
    print("   %-4s %10s %9s %8s %10s %9s %8s"
          % ("", "미니PC", "표류", "SNR", "미니PC", "표류", "SNR"))
    for k in (1, 2, 3, 4, 6):
        row = []
        for V_ in (sub_of(X, Bfull, k), Vt[:k].T):
            s = float(np.linalg.norm(msn - V_ @ (V_.T @ msn)))
            r = float(np.sqrt((((D - (D @ V_) @ V_.T) ** 2).sum(1)).mean())) / d0
            row += [100 * s, 100 * r, s / r]
        print("   %-4d %9.1f%% %8.1f%% %8.2f %9.1f%% %8.1f%% %8.2f" % (k, *row))

    # ---- 4 전이성 --------------------------------------------------------
    print("\n[4] 전이성 — **녹화 하나 빼기**. 본 적 없는 녹화에 통하나")
    print("   길                      표류 남음   미니PC 남음   SNR 배")
    stems = np.unique(tags)

    def loro(make):
        rr, ss = [], []
        for held in stems:
            tr, te = tags != held, tags == held
            if tr.sum() < 12 or te.sum() < 1:
                continue
            r, s = make(X[tr], D[tr], X[te], D[te])
            rr.append(r); ss.append(s)
        return float(np.mean(rr)), float(np.mean(ss))

    def proj(V_, Xte, Dte):
        return (float(np.sqrt((((Dte - (Dte @ V_) @ V_.T) ** 2).sum(1)).mean())) / d0,
                float(np.linalg.norm(msn - V_ @ (V_.T @ msn))))

    for k in (1, 2, 3, 4, 6):
        r, s = loro(lambda Xt, Dt, Xe, De, k=k:
                    proj(np.linalg.svd(Dt - Dt.mean(0), full_matrices=False)[2][:k].T, Xe, De))
        print("   PCA 사영 k=%-2d          %8.1f%% %10.1f%% %8.2f배" % (k, 100 * r, 100 * s, s / r))
    for k in (1, 2, 3, 4, 6):
        r, s = loro(lambda Xt, Dt, Xe, De, k=k: proj(sub_of(Xt, fit(Xt, Dt), k), Xe, De))
        print("   물리 사영 k=%-2d         %8.1f%% %10.1f%% %8.2f배" % (k, 100 * r, 100 * s, s / r))
    for nm, cols in (("전부", list(range(len(NAMES)))), ("외생만", EXO), ("dP만", [0])):
        r, _ = loro(lambda Xt, Dt, Xe, De, c=cols:
                    (float(np.sqrt((((De - Xe[:, c] @ fit(Xt[:, c], Dt)) ** 2).sum(1)).mean())) / d0,
                     1.0))
        print("   **예측 빼기** (%-6s)  %8.1f%% %10.1f%% %8.2f배" % (nm, 100 * r, 100.0, 1.0 / r))

    print("\n   읽는 법 — 예측 빼기는 덧셈 보정이라 **미니PC 를 100% 남긴다**. 같은 표류 감소를")
    print("   내면 사영보다 무조건 낫다. '외생만' 이 서면 분해기의 전력 추정 없이도 걸 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
