# -*- coding: utf-8 -*-
"""표류의 법칙을 **회로에서 계산한다** — 맞추지 않는다 (13.84.57).

13.84.56 이 잰 것: 형제 지문 표류의 58%를 [dP, d|V1|, dV3, dV5, dV7] 가 설명하고
PC1=dP(36°) · PC2=ImV3(14°) · PC3=d|V1|(32°) 다. **규칙이 있다.** 그런데 회귀계수 B 를
녹화마다 다시 맞춰야 하고 (녹화 하나 빼기에서 28%만 지운다) 그러면 새 자리에 못 옮긴다.

회로가 고정이면 B 는 맞출 것이 아니라 **계산할 것**이다:

    ΔI = (∂I/∂P)·ΔP  −  Y·ΔV            (노턴 1차, `circuit_sim.norton`)

Y 는 `I_h` 의 계통 전압 `V_k` 에 대한 감도다. 자유 파라미터가 **0개**다 — 회로 파라미터는
12.184.9 에서 자리 C 파형으로 맞춘 `results/_circuit_params_C.json` 을 그대로 쓴다.

⚠ 12.185.13 [F2]: 선형화는 **V3 를 1% 흔들면** h9 에서 27~32%, h13 에서 47~71% 틀린다 ->
  **생성에는 못 쓴다.** 그러나 여기서 보는 표류는 ΔV3/V1 ≈ 0.035% 로 30배 작고, 그 문서가
  Y 에 허용한 용도가 정확히 "감도 구조 진단" 이다. 그래도 가정하지 않고 [G] 에서 **잰다**.

재는 것:
  [G] 관문   이 섭동 크기에서 선형화가 성립하나 (Y·ΔV 대 참 시뮬 차분)
  [A] 절대   시뮬이 실측 지문을 얼마나 맞히나 (야코비를 믿기 전의 전제)
  [B] 방향   해석 야코비가 13.84.56 의 회귀계수와 몇 도인가
  [C] 설명력 **맞춘 것 없이** 표류의 몇 %를 지우나 — 회귀의 녹화하나빼기 28% 와 견준다

    python -X utf8 src/run_diag_driftcirc.py [--dev laptop_charger] [--gate]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftdim import ORD, load, vec
from src.run_diag_driftlaw import BANDS, SOURCES, VORD, blocks_xy, cov_names, fit, load_v
from src.synthesis.circuit_sim import GUIDE_PARAMS, simulate, to_wave

PARAMS_JSON = "results/_circuit_params_C.json"
HMAX = 15
OI = [o - 1 for o in ORD]


def params(dev, key="fit_siteC_wave"):
    try:
        d = json.load(open(PARAMS_JSON, encoding="utf-8"))[dev]
        return tuple(d.get(key) or d["guide"]), key
    except Exception:
        return tuple(GUIDE_PARAMS[dev]), "guide"


def cvec(z):
    """(15,) complex -> 홀수차 Re/Im 16."""
    z = np.asarray(z)[OI]
    return np.concatenate([z.real, z.imag])


def sim_I(par, P, V):
    """전압 페이저 V (15,) 에서 절대 전류 페이저 (15,) complex."""
    o = simulate(P, *par, vsrc=to_wave(V), h_max=HMAX)
    return o["I"] if o["ok"] else np.full(HMAX, np.nan, complex)


def solve_P(par, V, p_meas, tol=0.02, nit=12):
    """시뮬의 교류 입력 전력이 실측 P 가 되도록 직류 부하 P 를 푼다 (할선법)."""
    lo, hi = 0.3 * p_meas, 1.8 * p_meas
    for _ in range(nit):
        mid = 0.5 * (lo + hi)
        o = simulate(mid, *par, vsrc=to_wave(V), h_max=HMAX)
        if not o["ok"]:
            return p_meas
        if o["p_w"] < p_meas:
            lo = mid
        else:
            hi = mid
        if abs(o["p_w"] - p_meas) < tol * p_meas:
            break
    return 0.5 * (lo + hi)


def jac(par, P, V, dP=0.2, dV=0.1):
    """해석 야코비 (8, 16) — 행 순서는 `cov_names()` 와 같다. 자유 파라미터 0개.

    [보폭] 중심차분의 보폭은 **절대값**이다. 처음에 `relV=0.02` (= 4.3V) 로 뒀더니 ImV5 의
    |J| 가 참값의 0.47배·방향 15.8° 였다 — 그 보폭에서는 도통각이 크게 움직여 할선이
    접선이 아니다. dV 를 4.31 -> 0.001V 로 훑으면 **0.3V 아래가 완전한 고원**이고 수치
    잡음 바닥은 0.001V 에서도 안 보인다 (적분이 매끄럽다). 0.1V 는 고원 한가운데다.
    ⚠ 12.185.13 의 [F2] 불합격은 **1%=2.2V 섭동**에 대한 것이고 여기와 다른 체제다.
    """
    rows = []
    Ip = sim_I(par, P + dP, V); Im = sim_I(par, P - dP, V)
    rows.append((cvec(Ip) - cvec(Im)) / (2 * dP))                    # dI/dP
    for k, phs in [(1, (0.0,))] + [(h, (0.0, np.pi / 2)) for h in VORD]:
        for ph in phs:
            Vp = np.array(V, complex); Vm = np.array(V, complex)
            Vp[k - 1] += dV * np.exp(1j * ph); Vm[k - 1] -= dV * np.exp(1j * ph)
            rows.append((cvec(sim_I(par, P, Vp)) - cvec(sim_I(par, P, Vm))) / (2 * dV))
    return np.asarray(rows)


def cells(dev, block):
    """녹화×전력대 칸마다 (이름, 표류 Y, 공변량 X, 평균 P, 평균 V)."""
    VH = load_v(dev)
    out = []
    for band in BANDS[dev]:
        for stem, P, hc, on, off in load(dev):
            if stem not in VH:
                continue
            Y, X = blocks_xy(P, hc, VH[stem], on, *band, B=block)
            if len(Y) < 3:
                continue
            idx = np.nonzero(on & (P >= band[0]) & (P <= band[1]))[0]
            out.append(("%s|%d-%d" % (stem, *band), Y - Y.mean(0), X - X.mean(0),
                        float(P[idx].mean()), VH[stem][idx].mean(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="laptop_charger", choices=list(SOURCES))
    ap.add_argument("--block", type=int, default=600)
    ap.add_argument("--par", default="fit_siteC_wave")
    ap.add_argument("--gate", action="store_true", help="[G] 선형화 관문도 잰다 (느리다)")
    a = ap.parse_args()

    NAMES = cov_names()
    par, pk = params(a.dev, a.par)
    print("기기 %s · 회로 파라미터 [%s]  C_dc %.1fuF  R %.2f  L %.0fuH  Cx %.3fuF  rd %.2f"
          % (a.dev, pk, 1e6 * par[0], par[1], 1e6 * par[2], 1e6 * par[3], par[4]))

    CL = cells(a.dev, a.block)
    if not CL:
        print("칸이 없다"); return 1
    print("칸 %d개 · 블록 %d개" % (len(CL), sum(len(c[1]) for c in CL)))

    # ── [G] 선형화 관문 ────────────────────────────────────────────────────
    if a.gate:
        nm, Y0, X0, Pm, Vm = CL[0]
        Pd = solve_P(par, Vm, Pm)
        J = jac(par, Pd, Vm)
        I0 = cvec(sim_I(par, Pd, Vm))
        print("\n[G] 선형화 관문 — 실제 섭동 크기에서 ΔI ≈ J·Δx 가 맞나 (%s)" % nm)
        print("   %-8s %10s %12s %12s" % ("공변량", "섭동", "선형 |ΔI|", "참 대비 오차"))
        for i, n in enumerate(NAMES):
            s = float(np.std(X0[:, i])) or 1e-9
            if n == "dP":
                tr = cvec(sim_I(par, Pd + s, Vm)) - I0
            else:
                h = 1 if n == "d|V1|" else int(n[3:])
                Vp = np.array(Vm, complex)
                Vp[h - 1] += s * (1.0 if n.startswith(("d|", "Re")) else 1j)
                tr = cvec(sim_I(par, Pd, Vp)) - I0
            li = J[i] * s
            err = np.linalg.norm(tr - li) / max(np.linalg.norm(tr), 1e-12)
            print("   %-8s %10.4g %9.2f mA %11.1f%%  %s"
                  % (n, s, 1000 * np.linalg.norm(li), 100 * err,
                     "ok" if err < 0.15 else "**비선형**"))
        print("   (12.185.13 의 [F2] 불합격은 1%=2.2V 섭동이다. 여기는 그 30~40분의 1이고,")
        print("    `jac` 의 보폭 주석대로 dV<=0.3V 는 고원이다 — 선형화는 이 체제에서 성립한다)")

    # ── [A] 절대 지문 ──────────────────────────────────────────────────────
    print("\n[A] 절대 지문 — 시뮬이 실측을 맞히나 (야코비를 믿기 전의 전제)")
    print("   %-18s %8s %9s %9s %9s" % ("칸", "P(W)", "|I1|실측", "|I1|시뮬", "h3~15 오차"))
    JS, PD = {}, {}
    for nm, Y0, X0, Pm, Vm in CL:
        Pd = solve_P(par, Vm, Pm); PD[nm] = Pd
        Isim = sim_I(par, Pd, Vm)
        # 실측 칸 평균 지문: 표류를 뺀 원본 평균이 필요하므로 다시 만든다
        stem, band = nm.split("|"); lo, hi = (int(v) for v in band.split("-"))
        VH = load_v(a.dev)
        for st, P, hc, on, off in load(a.dev):
            if st != stem:
                continue
            idx = np.nonzero(on & (P >= lo) & (P <= hi))[0]
            Imeas = hc[idx].mean(0)
        e = np.linalg.norm(cvec(Isim)[1:] - cvec(Imeas)[1:]) / np.linalg.norm(cvec(Imeas)[1:])
        print("   %-18s %8.1f %8.1fmA %8.1fmA %8.1f%%"
              % (nm, Pm, 1000 * abs(Imeas[0]), 1000 * abs(Isim[0]), 100 * e))
        JS[nm] = jac(par, Pd, Vm)

    # ── [B] 방향 — 해석 야코비 vs 회귀 ────────────────────────────────────
    D = np.vstack([c[1] for c in CL]); X = np.vstack([c[2] for c in CL])
    Bf = fit(X, D)
    Jm = np.mean([JS[c[0]] for c in CL], 0)
    print("\n[B] 해석 야코비 vs 13.84.56 의 회귀계수 — 각도")
    print("   %-8s %9s %11s %11s" % ("공변량", "각도", "|J|해석", "|B|회귀"))
    for i, n in enumerate(NAMES):
        u = Jm[i] / max(np.linalg.norm(Jm[i]), 1e-30)
        v = Bf[i] / max(np.linalg.norm(Bf[i]), 1e-30)
        print("   %-8s %8.0f° %9.4g %11.4g" % (n, np.degrees(np.arccos(min(1.0, abs(float(u @ v))))),
                                               np.linalg.norm(Jm[i]), np.linalg.norm(Bf[i])))

    # ── [C] 설명력 — 맞춘 것 없이 ─────────────────────────────────────────
    print("\n[C] **맞춘 것 없이** 표류를 얼마나 지우나 (칸마다 그 칸의 야코비)")
    tot = float((D ** 2).sum())

    def resid(scale_fit):
        num = 0.0
        for nm, Y0, X0, Pm, Vm in CL:
            Pr = X0 @ JS[nm]
            if scale_fit:                      # 크기만 한 개 실수로 다시 맞춘다 (방향은 회로 것)
                g = float((Y0 * Pr).sum() / max((Pr * Pr).sum(), 1e-30))
                Pr = g * Pr
            num += float(((Y0 - Pr) ** 2).sum())
        return 1.0 - num / tot

    print("   해석 그대로 (자유도 0)            R^2 %+.3f" % resid(False))
    print("   크기 1개만 다시 맞춤 (자유도 1)   R^2 %+.3f" % resid(True))
    print("   견줌 회귀 자리에서 (자유도 128)   R^2 %+.3f" %
          (1.0 - float(((D - X @ Bf) ** 2).sum()) / tot))
    print("   견줌 회귀 녹화하나빼기 (13.84.56) R^2  +0.285")

    # 공변량 부분집합별
    print("\n   공변량을 하나씩만 켜서 (해석 그대로)")
    for i, n in enumerate(NAMES):
        num = sum(float(((c[1] - np.outer(c[2][:, i], JS[c[0]][i])) ** 2).sum()) for c in CL)
        print("      %-8s R^2 %+.3f" % (n, 1.0 - num / tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
