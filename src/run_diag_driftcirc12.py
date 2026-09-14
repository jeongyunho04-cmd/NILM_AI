# -*- coding: utf-8 -*-
"""13.84.57 의 야코비를 **v12g (포화 초크)** 로 다시 잰다 (13.84.66).

13.84.57 은 `ΔI = (∂I/∂P)·ΔP − Y·ΔV` 를 `src/synthesis/circuit_sim` 으로 쟀다. 그 토폴로지는
**상수 L** 5모수다 (C_dc, R, L, Cx, rd). 결과: 실측 PC1~PC3 가 해석 8차원 span 안에 0°/3°/13°,
그러나 해석 3차원 대 실측 3차원의 **셋째 주각이 81°** 이고 채점이 붕괴했다 (k=3 에서 0.7885).

그런데 `circuit_model/circuit12.py` (v12g) 는 **포화 초크** `L(i) = L0/(1+(i/Isat)²)` 와
선형 덧셈 경로 G 를 더한 7모수이고, `README_v12.md` 가 *"포화 L 이 고차 출처 · 파형 잔차
6.7% -> 2.3%"* 라고 적었다. 그리고 13.84.65 가 **실측 PC3 는 고차 축이고 계통이 만든다**고
쟀다 (h13 비가 1.26~4.25배, PC3 가 녹화 안 표류 표준편차의 13~20배로 흔들린다).

즉 **81° 는 물리가 아니라 토폴로지 탓일 수 있다.** 그것을 여기서 가른다.

재는 것 (13.84.57 과 **같은 자**로 — 칸·블록·공변량·주각 정의가 같아야 견줄 수 있다):
  [S] 보폭   v12g 에서도 중심차분에 고원이 있나. **먼저 잰다** — 13.84.57 이 여기서 틀렸다
  [A] 절대   시뮬이 실측 지문을 얼마나 맞히나 (v4.3 은 h3~15 에서 18~28% 였다)
  [P] 주각   해석 span 대 실측 표류 PC — **셋째 주각이 81° 에서 내려오나**
  [R] 계급   행정규화 야코비의 특이값. v4.3 은 2.20/1.53/0.892 뒤로 10배 절벽이었다
  [C] 설명력 **맞춘 것 없이** 표류의 몇 %를 지우나

    python -X utf8 src/run_diag_driftcirc12.py [--dev laptop_charger] [--steps]
"""
import argparse
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftcirc import cells, cvec
from src.run_diag_driftdim import ORD, load
from src.run_diag_driftlaw import SOURCES, VORD, cov_names, fit, load_v

OI = [o - 1 for o in ORD]
HMAX = 15


def load12(dev):
    """`circ12_<dev>.pkl` 을 읽는다.

    ⚠ 그 피클은 numpy 2.x 로 절였다. 여기(1.24)서 열려면 `numpy._core` 를 `numpy.core` 로
      돌리는 shim 이 필요하다 — 순수 재명명이라 값은 안 변한다.
    """
    for sub in ("", ".multiarray", ".umath", "._multiarray_umath", ".numeric"):
        try:
            sys.modules["numpy._core" + sub] = __import__("numpy.core" + sub, fromlist=["x"])
        except Exception:
            pass
    with open("circuit_model/circ12_%s.pkl" % dev, "rb") as f:
        return pickle.load(f)


def make_sim(npc):
    from circuit_model import circuit12 as c12

    def sim_I(par, P, V):
        """전압 페이저 (15,) -> 전류 페이저 (15,) complex. 계측 영역 (펌웨어 규약)."""
        H = c12.sim_harmonics(float(P), np.asarray(V, complex), tuple(par), npc=npc)
        return np.full(HMAX, np.nan, complex) if H is None else np.asarray(H)[:HMAX]

    def p_ac(par, P, V):
        """그 동작점의 **교류 입력 전력** — 파형에서 직접 잰다 (전력계가 재는 것)."""
        C, R, L0, Isat, Cx, rd, G = par
        v = c12.wave_from_harmonics(np.asarray(V, complex), npc)
        vs = np.tile(v, c12.NCYC); dt = 1.0 / (c12.F * npc)
        Pm = max(float(P) - G * float(np.mean(v ** 2)), 0.5)
        I = c12._core12(vs, dt, Pm, C, max(R, 1e-3), L0, Isat, Cx, c12.VF, max(rd, 1e-3),
                        vs.max() - c12.VF)
        if not np.all(np.isfinite(I)):
            return np.nan
        ic = I[-npc:] + G * v
        vm = c12.rc_periodic(v, dt, 60e-6); im = c12.rc_periodic(ic, dt, 60e-6)
        return float(np.mean(vm * im))

    return sim_I, p_ac


def solve_P(p_ac, par, V, p_meas, tol=0.02, nit=14):
    """시뮬의 교류 입력 전력이 실측 P 가 되도록 직류 부하 P 를 푼다 (이분법)."""
    lo, hi = 0.3 * p_meas, 1.8 * p_meas
    for _ in range(nit):
        mid = 0.5 * (lo + hi)
        w = p_ac(par, mid, V)
        if not np.isfinite(w):
            return p_meas
        if w < p_meas:
            lo = mid
        else:
            hi = mid
        if abs(w - p_meas) < tol * p_meas:
            break
    return 0.5 * (lo + hi)


def jac(sim_I, par, P, V, dP=0.2, dV=0.1):
    """해석 야코비 (8,16) — 행 순서는 `cov_names()` 와 같다. 자유 파라미터 0개."""
    rows = [(cvec(sim_I(par, P + dP, V)) - cvec(sim_I(par, P - dP, V))) / (2 * dP)]
    for k, phs in [(1, (0.0,))] + [(h, (0.0, np.pi / 2)) for h in VORD]:
        for ph in phs:
            Vp = np.array(V, complex); Vm = np.array(V, complex)
            Vp[k - 1] += dV * np.exp(1j * ph); Vm[k - 1] -= dV * np.exp(1j * ph)
            rows.append((cvec(sim_I(par, P, Vp)) - cvec(sim_I(par, P, Vm))) / (2 * dV))
    return np.asarray(rows)


def pangles(A, B):
    """두 부분공간의 주각(도). 열이 기저 벡터."""
    qa = np.linalg.qr(A)[0]; qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def drift_pcs(CL, k=3):
    """칸마다 평균을 뺀 표류를 쌓아 상위 k 주성분 (16,k)."""
    D = np.vstack([c[1] for c in CL])
    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    return Vt[:k].T, S, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="laptop_charger", choices=list(SOURCES))
    ap.add_argument("--block", type=int, default=600)
    ap.add_argument("--npc", type=int, default=3072,
                    help="주기당 표본. **3072 을 쓴다** — 512/1024/2048 은 3072 대비 야코비 행 "
                         "방향이 최대 6.4/2.4/0.6° 어긋난다 (크기는 0.5%% 안). 셋째 주각을 "
                         "도 단위로 보는 절이라 수렴한 자리에서 재야 한다")
    ap.add_argument("--steps", action="store_true", help="[S] 보폭 고원을 훑는다 (느리다)")
    ap.add_argument("--vh", type=int, default=31,
                    help="[G] 에서 원시 파형으로부터 뽑을 **전압 차수**. 기본 31 — 15 로 자르면 "
                         "예측 각도가 5~10° 에서 14~24° 로 두 배가 된다 (13.84.66 [7]). "
                         "원시는 15.4kHz 라 h128 까지 있다")
    ap.add_argument("--grid", action="store_true",
                    help="[G] **계통 짝 예측** — 13.84.65 의 원시 파형 짝을 자유모수 0 으로 맞힌다")
    a = ap.parse_args()

    d = load12(a.dev)
    par = tuple(d["params"])
    sim_I, p_ac = make_sim(a.npc)
    print("기기 %s · v12g [%s]  C %.1fuF  R %.2f  L0 %.0fuH  Isat %.2fA  Cx %.3fuF  rd %.2f  G %.3fmS"
          % (a.dev, d["date"], 1e6 * par[0], par[1], 1e6 * par[2], par[3],
             1e6 * par[4], par[5], 1e3 * par[6]))

    # ── [G] 계통 짝 예측 — 자유모수 0 ──────────────────────────────────────
    # 주각 놀이는 행을 늘리면 각도가 공짜로 내려간다 (13.84.57 의 자 ②: span 증폭).
    # 여기는 **예측**이라 그 함정이 없다 — 실측 계통차를 시뮬 계통차와 직접 견준다.
    if a.grid:
        from src.run_diag_rawgrid import scan
        R = scan(a.dev, a.vh)
        ev = [x for x in R if x["vh3"] < 1.5]; nt = [x for x in R if x["vh3"] >= 1.5]
        print("")
        print("[G] **계통 짝** — 실측 계통차를 회로가 맞히나 (자유 파라미터 0개)")
        print("   %-26s %10s %10s %8s %9s"
              % ("짝", "실측 |ΔI|", "시뮬 |ΔI|", "크기비", "각도"))
        for e in ev:
            n = min(nt, key=lambda z: abs(z["P"] - e["P"]), default=None)
            if n is None or abs(n["P"] - e["P"]) > 0.15 * max(e["P"], 1.0):
                continue
            Pe = solve_P(p_ac, par, e["V"], e["P"]); Pn = solve_P(p_ac, par, n["V"], n["P"])
            se, sn = sim_I(par, Pe, e["V"]), sim_I(par, Pn, n["V"])
            dm = cvec(n["I"] / n["P"] - e["I"] / e["P"]) * e["P"]
            ds = cvec(sn / n["P"] - se / e["P"]) * e["P"]
            g = np.linalg.norm(ds) / max(np.linalg.norm(dm), 1e-30)
            ct = float(ds @ dm) / max(np.linalg.norm(ds) * np.linalg.norm(dm), 1e-30)
            print("   %-26s %9.1fmA %9.1fmA %8.2f %8.0f°"
                  % ("%s -> %s" % (e["name"][-12:], n["name"][-9:]),
                     1000 * np.linalg.norm(dm), 1000 * np.linalg.norm(ds), g,
                     np.degrees(np.arccos(np.clip(ct, -1, 1)))))
        print("   크기비 1 · 각도 0 이면 회로가 계통차를 그대로 낸다.")
        print("   13.84.65: 이 차이의 75~81%% 가 표류 부분공간 안이고 PC2·PC3 를 12~20배로 흔든다 —")
        print("   그러니 이 표가 곧 **회로가 표류를 설명하나** 의 답이다.")
        return 0

    CL = cells(a.dev, a.block)
    if not CL:
        print("칸이 없다"); return 1
    print("칸 %d개 · 블록 %d개 · npc %d" % (len(CL), sum(len(c[1]) for c in CL), a.npc))

    nm0, Y0, X0, Pm0, Vm0 = CL[0]
    t0 = time.time(); sim_I(par, Pm0, Vm0); warm = time.time() - t0
    print("시뮬 1회 %.2fs (예열 뒤) — 칸당 16회" % warm)

    # ── [S] 보폭 ───────────────────────────────────────────────────────────
    Pd0 = solve_P(p_ac, par, Vm0, Pm0)
    if a.steps:
        print("\n[S] **보폭** — v12g 에서도 중심차분에 고원이 있나 (ImV5 · %s)" % nm0)
        print("   ⚠ 13.84.57 은 여기서 틀려 물리 결론을 뒤집었다. 토폴로지가 바뀌었으니 다시 잰다")
        print("   %10s %12s %10s" % ("dV (V)", "|J| (mA/V)", "직전과 각도"))
        prev = None
        for dv in (4.31, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
            Vp = np.array(Vm0, complex); Vm = np.array(Vm0, complex)
            Vp[4] += 1j * dv; Vm[4] -= 1j * dv
            J = (cvec(sim_I(par, Pd0, Vp)) - cvec(sim_I(par, Pd0, Vm))) / (2 * dv)
            ang = "" if prev is None else "%9.1f°" % np.degrees(np.arccos(np.clip(
                float(J @ prev) / max(np.linalg.norm(J) * np.linalg.norm(prev), 1e-30), -1, 1)))
            print("   %10.3f %12.3f %10s" % (dv, 1000 * np.linalg.norm(J), ang))
            prev = J
        print("   (직전과의 각도가 0 으로 가면 고원이다 — 그 안에서 보폭을 고른다)")

    # ── [A] 절대 지문 ──────────────────────────────────────────────────────
    print("\n[A] **절대** — 시뮬이 실측 지문을 맞히나 (v4.3 은 h3~15 에서 18~28% 였다)")
    print("   %-20s %7s %10s %10s %10s" % ("칸", "P(W)", "|I1|실측", "|I1|시뮬", "h3~15 오차"))
    VH = load_v(a.dev); REC = {st: (P, hc, on) for st, P, hc, on, off in load(a.dev)}
    JS, errs = {}, []
    for nm, Y0, X0, Pm, Vm in CL:
        Pd = solve_P(p_ac, par, Vm, Pm)
        Isim = sim_I(par, Pd, Vm)
        stem, band = nm.split("|"); lo, hi = (int(v) for v in band.split("-"))
        P, hc, on = REC[stem]
        idx = np.nonzero(on & (P >= lo) & (P <= hi))[0]
        Imeas = hc[idx].mean(0)
        e = np.linalg.norm(cvec(Isim)[1:] - cvec(Imeas)[1:]) / np.linalg.norm(cvec(Imeas)[1:])
        errs.append(e)
        print("   %-20s %7.1f %9.1fmA %9.1fmA %9.1f%%"
              % (nm, Pm, 1000 * abs(Imeas[0]), 1000 * abs(Isim[0]), 100 * e))
        JS[nm] = jac(sim_I, par, Pd, Vm)
    print("   중앙 %.1f%%  (v4.3: 18~28%%)" % (100 * np.median(errs)))

    # ── [R] 계급 ───────────────────────────────────────────────────────────
    Jm = np.mean([JS[c[0]] for c in CL], 0)
    Jn = Jm / np.maximum(np.linalg.norm(Jm, axis=1, keepdims=True), 1e-30)
    sv = np.linalg.svd(Jn, compute_uv=False)
    cum = np.cumsum(sv ** 2) / (sv ** 2).sum()
    print("\n[R] **계급** — 행정규화 야코비 8x16 의 특이값 (v4.3: 2.20 1.53 0.892 뒤 10배 절벽)")
    print("   %s" % "  ".join("%.3f" % x for x in sv))
    print("   누적 %s" % "  ".join("%.3f" % x for x in cum[:5]))
    print("   절벽 비 s3/s4 = %.1f배  (v4.3 은 10.4배)" % (sv[2] / max(sv[3], 1e-30)))

    # ── [P] 주각 ───────────────────────────────────────────────────────────
    PC, S, D = drift_pcs(CL, 3)
    Vt = np.linalg.svd(Jn, full_matrices=False)[2]
    print("\n[P] **주각** — 해석 감도 대 실측 표류 (13.84.57 과 같은 정의)")
    ang8 = pangles(Jn.T, PC)
    print("   실측 PC1~PC3 가 해석 **8차원 span** 안에  %s   (v4.3 충전기 0°/3°/13°)"
          % " / ".join("%.0f°" % x for x in ang8))
    for k in (2, 3):
        ak = pangles(Vt[:k].T, PC[:, :k])
        print("   해석 %d차원 대 실측 %d차원          %s%s"
              % (k, k, " / ".join("%.0f°" % x for x in ak),
                 "   <- **13.84.57 은 셋째가 81° 였다**" if k == 3 else ""))
    print("   실측 표류 주성분 설명분산  %s"
          % "  ".join("%.2f" % x for x in np.cumsum(S ** 2)[:3] / (S ** 2).sum()))

    # ── [C] 설명력 ─────────────────────────────────────────────────────────
    X = np.vstack([c[2] for c in CL]); Bf = fit(X, D)
    tot = float((D ** 2).sum())

    def resid(scale_fit):
        num = 0.0
        for nm, Y0, X0, Pm, Vm in CL:
            Pr = X0 @ JS[nm]
            if scale_fit:
                g = float((Y0 * Pr).sum() / max((Pr * Pr).sum(), 1e-30))
                Pr = g * Pr
            num += float(((Y0 - Pr) ** 2).sum())
        return 1.0 - num / tot, (g if scale_fit else 1.0)

    r0 = resid(False)[0]; r1, g1 = resid(True)
    print("\n[C] **설명력** — 맞춘 것 없이 표류를 얼마나 지우나")
    print("   해석 그대로 (자유도 0)            R^2 %+.3f" % r0)
    print("   크기 1개만 다시 맞춤 (자유도 1)   R^2 %+.3f  (배율 %.2f)" % (r1, g1))
    print("   견줌 회귀 자리에서 (자유도 128)   R^2 %+.3f"
          % (1.0 - float(((D - X @ Bf) ** 2).sum()) / tot))
    print("   견줌 13.84.57 v4.3 자유도 1       R^2  +0.37(충전기) / +0.35(프로젝터)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
