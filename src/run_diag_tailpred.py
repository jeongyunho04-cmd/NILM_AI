# -*- coding: utf-8 -*-
"""꼬리(h17~h31)가 표류를 **더 설명하나** — 자료가 이미 있다 (14.15).

사용자: *"아니 최근의 측정파일에 이미 있을 건데."*  **맞다.** 내가 오래된 파일 둘만 보고
"자료가 없다"고 했다. 최근 녹화 다섯(`laptop_charger_3~6`, `minipc_4`)은 `vhhi17~31` ·
`vhhideg17~31` · `vhhi_seq` 를 갖고 있고 **60초마다 갱신**된다 (19.5분 녹화에 20값).

`vtail.npz` 가 세션당 하나였던 것은 13.73 이 **0.667초짜리 원시 스냅샷**에서 back-fill 했기
때문이다 — 그때는 펌웨어가 아직 안 보냈다. 지금은 보낸다.

재는 것 — 14.11 의 회귀에 꼬리를 더한다:
    A  dP + Re/Im V at h1~h15          (14.11 의 승자)
    B  A + Re/Im V at h17~h31          (+16열)

⚠ 꼬리는 60초마다 갱신되므로 10초 블록 안에서는 **계단 상수**다. 그 블록의 값을 쓴다.
⚠ 녹화가 다섯뿐이라 LORO 가 얇다. 폴드별로 낸다 ([[validate-both-directions]]).
⚠ 공변량을 늘리면 자리 안 R^2 는 반드시 오른다 -> LORO 로만 읽는다.

    python -X utf8 src/run_diag_tailpred.py
"""
import csv
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, vec

HI = (17, 19, 21, 23, 25, 27, 29, 31)
LOW = (1, 3, 5, 7, 9, 11, 13, 15)
BLOCK = 600                       # 10초
FILES = ("laptop_charger_3", "laptop_charger_4", "laptop_charger_5",
         "laptop_charger_6", "minipc_4")
BANDS = {"laptop_charger": [(19, 28), (28, 40), (40, 55), (55, 70)],
         "minipc": [(8, 14)]}


def tail_by_seq(stem):
    """CSV 에서 (seq -> 꼬리 8복소, V1 대비). 없으면 None."""
    path = "data/%s.csv" % stem
    try:
        fh = open(path, newline="")
    except OSError:
        return None
    with fh:
        r = csv.reader(fh)
        hdr = next(r)
        if "vhhi17" not in hdr:
            return None
        ix = {n: i for i, n in enumerate(hdr)}
        cm = [ix["vhhi%d" % h] for h in HI]
        cd = [ix["vhhideg%d" % h] for h in HI]
        cs, cv, cq = ix["vhhi_seq"], ix["vh1"], ix["seq"]
        out = {}
        for row in r:
            s = int(row[cq])
            if s in out:
                continue
            v1 = max(float(row[cv]), 1e-9)
            m = np.array([float(row[c]) for c in cm])
            dg = np.array([float(row[c]) for c in cd])
            out[s] = (m * np.exp(1j * np.radians(dg)) / v1).astype(complex)
    return out


def build():
    """칸(녹화x전력대) 안 편차. (D, C_low, C_hi, tags)"""
    Ds, Cl, Ch, tg = [], [], [], []
    for stem in FILES:
        app = "minipc" if stem.startswith("minipc") else "laptop_charger"
        tl = tail_by_seq(stem)
        if tl is None:
            continue
        d = np.load("processed_data/npz/%s.npz" % stem, allow_pickle=True)
        hc = np.asarray(d["harmonics_complex"])
        vc = np.asarray(d["voltage_harmonics_complex"])
        P = np.asarray(d["p_denoised_w"])
        seq = np.asarray(d["seq"])
        on = np.asarray(d["state_id"]) != 0 if "state_id" in d.files else np.asarray(d["is_on"]) == 1
        ok = on & (np.asarray(d["is_valid"]) == 1) & (np.asarray(d["is_unplugged"]) == 0)
        # 각 사이클의 꼬리 = 그 순간 유효한 vhhi (seq 로 찾고, 없으면 직전 값)
        ks = np.array(sorted(tl))
        arr = np.array([tl[k] for k in ks])
        pos = np.clip(np.searchsorted(ks, seq, side="right") - 1, 0, len(ks) - 1)
        tail_cyc = arr[pos]
        for lo, hi in BANDS[app]:
            idx = np.nonzero(ok & (P >= lo) & (P <= hi))[0]
            if len(idx) < 2 * BLOCK:
                continue
            Y, A, B = [], [], []
            for s in range(0, len(idx) - BLOCK + 1, BLOCK):
                j = idx[s:s + BLOCK]
                Y.append(vec(hc[j].mean(0)))
                v = vc[j].mean(0)
                a = [P[j].mean()]
                for h in LOW:
                    a += [v[h - 1].real, v[h - 1].imag]
                t = tail_cyc[j].mean(0)
                b = []
                for k in range(len(HI)):
                    b += [t[k].real, t[k].imag]
                A.append(a); B.append(b);
            Y = np.asarray(Y); A = np.asarray(A, float); B = np.asarray(B, float)
            if len(Y) < 2:
                continue
            Ds.append(Y - Y.mean(0)); Cl.append(A - A.mean(0)); Ch.append(B - B.mean(0))
            tg += [stem] * len(Y)
    return np.vstack(Ds), np.vstack(Cl), np.vstack(Ch), np.asarray(tg)


def loro(D, C, tags, lam=1e-2):
    R = np.zeros_like(D)
    for h in np.unique(tags):
        tr, te = tags != h, tags == h
        if tr.sum() < 4:
            R[te] = D[te]; continue
        sc = C[tr].std(0); sc[sc == 0] = 1
        X = C[tr] / sc
        B = np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ D[tr]) / sc[:, None]
        R[te] = D[te] - C[te] @ B
    return R


def r2(D, R):
    return 1 - (R ** 2).sum() / max((D ** 2).sum(), 1e-30)


def main():
    D, Cl, Ch, tg = build()
    print("블록 %d · 녹화 %d · %s" % (len(D), len(np.unique(tg)), sorted(set(tg))))
    print("   꼬리 산포 (녹화 안, ppm): %s"
          % " ".join("%.0f" % (1e6 * Ch[:, 2 * k].std()) for k in range(len(HI))))

    CA = Cl
    CB = np.hstack([Cl, Ch])
    print("\n**LORO R^2** — 꼬리를 더하면")
    print("  %-26s %5s %8s" % ("공변량", "열", "LORO R^2"))
    res = {}
    for lam in (1e-3, 1e-2, 1e-1):
        a = r2(D, loro(D, CA, tg, lam)); b = r2(D, loro(D, CB, tg, lam))
        res[lam] = (a, b)
        print("  λ=%-6s A h1~h15 %2d -> %6.3f  |  B +h17~31 %2d -> %6.3f  |  차이 %+.3f"
              % (lam, CA.shape[1], a, CB.shape[1], b, b - a))

    print("\n  녹화별 (λ=0.01):")
    for h in np.unique(tg):
        tr, te = tg != h, tg == h
        vals = []
        for C in (CA, CB):
            sc = C[tr].std(0); sc[sc == 0] = 1
            X = C[tr] / sc
            B = np.linalg.solve(X.T @ X + 0.01 * len(X) * np.eye(X.shape[1]),
                                X.T @ D[tr]) / sc[:, None]
            vals.append(r2(D[te], D[te] - C[te] @ B))
        print("     %-20s A %6.3f  B %6.3f  차이 %+6.3f  (블록 %d)"
              % (h, vals[0], vals[1], vals[1] - vals[0], te.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
