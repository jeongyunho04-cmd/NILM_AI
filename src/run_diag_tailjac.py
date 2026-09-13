# -*- coding: utf-8 -*-
"""꼬리(h17~h31)의 기여를 **자유도 0 으로** 잰다 (14.15).

14.11 의 방법 그대로다 — 회귀로 맞추지 않고 **회로로 계산**한다:

    ΔI = (∂I/∂P)·ΔP  +  Σ_h (∂I/∂V_h)·ΔV_h          자유 파라미터 0개

`run_diag_tailpred.py` 의 회귀는 열 16개를 더했는데 꼬리가 60초마다 갱신되어 독립 표본이
녹화당 ~20개뿐이라 과적합했다 (LORO −0.37). 야코비는 맞출 것이 없으니 그 함정이 없다.

  A  dP + h1~h15          14.11 의 승자 (옛 표본에서 R^2 0.654)
  B  A + **h17~h31**      꼬리를 더한 것

⚠ 꼬리가 있는 녹화는 `laptop_charger_3~6` 넷뿐이다 (`vhhi17~31`, 60초 갱신).
⚠ numba/scipy — `scipy.linalg.cython_blas` 를 먼저 import 해야 한다 (14.11 ⑥).

    python -X utf8 src/run_diag_tailjac.py
"""
import csv
import sys

import scipy.linalg.cython_blas          # noqa: F401  ⚠ 먼저
import numba.np.arraymath                # noqa: F401

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, vec

LOW = (1, 3, 5, 7, 9, 11, 13, 15)
HI = (17, 19, 21, 23, 25, 27, 29, 31)
BLOCK = 600
FILES = ("laptop_charger_3", "laptop_charger_4", "laptop_charger_5", "laptop_charger_6")
BANDS = [(19, 28), (28, 40), (40, 55), (55, 70)]
DP, DV = 0.2, 0.1
NPC = 3072


def tail_by_seq(stem):
    with open("data/%s.csv" % stem, newline="") as fh:
        r = csv.reader(fh); hdr = next(r)
        if "vhhi17" not in hdr:
            return None
        ix = {n: i for i, n in enumerate(hdr)}
        cm = [ix["vhhi%d" % h] for h in HI]
        cd = [ix["vhhideg%d" % h] for h in HI]
        cs, cq = ix["vhhi_seq"], ix["seq"]
        out = {}
        for row in r:
            s = int(row[cq])
            if s in out:
                continue
            m = np.array([float(row[c]) for c in cm])
            g = np.array([float(row[c]) for c in cd])
            out[s] = (m * np.exp(1j * np.radians(g))).astype(complex)   # 절대 [V rms]
    return out


def cells():
    """칸별 (stem, band, Y(n,16), P(n), V(n,31) 복소)."""
    out = []
    for stem in FILES:
        tl = tail_by_seq(stem)
        if tl is None:
            continue
        d = np.load("processed_data/npz/%s.npz" % stem, allow_pickle=True)
        hc = np.asarray(d["harmonics_complex"])
        vc = np.asarray(d["voltage_harmonics_complex"])
        P = np.asarray(d["p_denoised_w"]); seq = np.asarray(d["seq"])
        on = np.asarray(d["state_id"]) != 0 if "state_id" in d.files else np.asarray(d["is_on"]) == 1
        ok = on & (np.asarray(d["is_valid"]) == 1) & (np.asarray(d["is_unplugged"]) == 0)
        ks = np.array(sorted(tl)); arr = np.array([tl[k] for k in ks])
        pos = np.clip(np.searchsorted(ks, seq, side="right") - 1, 0, len(ks) - 1)
        tcyc = arr[pos]
        for lo, hi in BANDS:
            idx = np.nonzero(ok & (P >= lo) & (P <= hi))[0]
            if len(idx) < 2 * BLOCK:
                continue
            Y, PP, VV = [], [], []
            for s in range(0, len(idx) - BLOCK + 1, BLOCK):
                j = idx[s:s + BLOCK]
                Y.append(vec(hc[j].mean(0)))
                PP.append(P[j].mean())
                V = np.zeros(31, complex)
                V[:15] = vc[j].mean(0)
                for k, h in enumerate(HI):
                    V[h - 1] = tcyc[j].mean(0)[k]
                VV.append(V)
            if len(Y) < 2:
                continue
            out.append((stem, (lo, hi), np.asarray(Y), np.asarray(PP), np.asarray(VV)))
    return out


def main():
    from src.synthesis import fcm
    import circuit_model.circuit12 as c12

    m = fcm.load_models().get("laptop_charger")
    par = tuple(np.asarray(m.params if hasattr(m, "params") else m, float))
    print("v12g 파라미터 %s" % np.round(par, 4))

    def sim(P, V):
        return np.asarray(c12.sim_harmonics(float(P), np.asarray(V, complex), par, npc=NPC))[:15]

    CS = cells()
    print("칸 %d개 · 블록 %d개 · 녹화 %d"
          % (len(CS), sum(len(c[2]) for c in CS), len(set(c[0] for c in CS))))

    tot = {"meas": 0.0, "A": 0.0, "B": 0.0}
    rows = []
    for stem, band, Y, P, V in CS:
        P0 = float(P.mean()); V0 = V.mean(0)
        # ── 야코비 (자유도 0) ───────────────────────────────────────────
        jP = (vec(sim(P0 + DP, V0)) - vec(sim(P0 - DP, V0))) / (2 * DP)
        jlo, jhi = [], []
        for h in LOW:
            phs = (0.0,) if h == 1 else (0.0, np.pi / 2)
            for ph in phs:
                Vp = V0.copy(); Vm = V0.copy()
                Vp[h - 1] += DV * np.exp(1j * ph); Vm[h - 1] -= DV * np.exp(1j * ph)
                jlo.append((vec(sim(P0, Vp)) - vec(sim(P0, Vm))) / (2 * DV))
        for h in HI:
            for ph in (0.0, np.pi / 2):
                Vp = V0.copy(); Vm = V0.copy()
                Vp[h - 1] += DV * np.exp(1j * ph); Vm[h - 1] -= DV * np.exp(1j * ph)
                jhi.append((vec(sim(P0, Vp)) - vec(sim(P0, Vm))) / (2 * DV))
        Jlo = np.asarray(jlo); Jhi = np.asarray(jhi)

        dY = Y - Y.mean(0); dP = P - P0
        xlo = []
        for h in LOW:
            c = V[:, h - 1] - V0[h - 1]
            xlo.append(c.real)
            if h != 1:
                xlo.append(c.imag)
        Xlo = np.asarray(xlo).T
        xhi = []
        for h in HI:
            c = V[:, h - 1] - V0[h - 1]
            xhi += [c.real, c.imag]
        Xhi = np.asarray(xhi).T

        predA = dP[:, None] * jP[None] + Xlo @ Jlo
        predB = predA + Xhi @ Jhi
        tot["meas"] += float((dY ** 2).sum())
        tot["A"] += float(((dY - predA) ** 2).sum())
        tot["B"] += float(((dY - predB) ** 2).sum())
        rows.append((stem, band, len(Y),
                     1 - ((dY - predA) ** 2).sum() / max((dY ** 2).sum(), 1e-30),
                     1 - ((dY - predB) ** 2).sum() / max((dY ** 2).sum(), 1e-30)))

    print("\n**자유도 0** — 회로가 계산한 ΔI 로 표류를 얼마나 지우나")
    print("  %-20s %-9s %5s %9s %9s %9s" % ("녹화", "전력대", "블록", "A h1~h15", "B +꼬리", "차이"))
    for stem, band, n, a, b in rows:
        print("  %-20s %-9s %5d %9.3f %9.3f %+9.3f" % (stem, "%d-%dW" % band, n, a, b, b - a))
    A = 1 - tot["A"] / tot["meas"]; B = 1 - tot["B"] / tot["meas"]
    print("  %-20s %-9s %5s %9.3f %9.3f %+9.3f" % ("합계", "", "", A, B, B - A))
    print("\n  견줌: 14.11 이 옛 표본(꼬리 없음)에서 h15 까지로 얻은 값 **+0.654**")
    print("  ⚠ 여기는 꼬리가 있는 녹화 넷만이라 표본이 다르다 — A 끼리 견주는 것이 맞다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
