# -*- coding: utf-8 -*-
"""모델의 전압 채널을 **h15 까지** 늘릴 값이 있나 — 재학습 없이 (14.14).

`inputs.VOLT_ORDERS = (1, 3, 5, 7, 9, 11)` 이라 모델은 h13·h15 전압을 **못 본다**.
13.26 이 그 절단점을 고를 때 잰 것은 **Z 식별 R^2** 였다 (h11 까지 0.879 > h15 까지 0.872).
그런데 14.11 이 잰 축은 다르다 — **표류 예측**이고, 거기서는 h9~h15 를 더하니
LORO R^2 가 0.260 -> 0.481 로 뛰었다. 다른 과제로 고른 절단점을 지금 근거로 쓰면
[[derived-limits-outlive-their-reason]] 다.

여기서는 **모델이 실제로 받는 채널 집합**으로 견준다:
    A(현행)  dP + Re/Im V at h1,3,5,7,9,11        = 13개
    B(확장)  dP + Re/Im V at h1,3,5,7,9,11,13,15  = 17개

⚠ 공변량을 늘리면 자리 안 R^2 는 반드시 오른다 -> **녹화 하나 빼기(LORO)** 로 잰다.
⚠ 성분별로 본다 — 전체는 PC1 이 절반이라 묻힌다 ([[split-by-class-before-taking-a-median]]).
⚠ h13·h15 가 h1~h11 로 이미 예측되면 더해도 소용없다. 그것도 같이 잰다.

    python -X utf8 src/run_diag_voltchan.py
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

BLOCK = 600
BANDS = {"laptop_charger": [(19, 28), (28, 40), (40, 55), (55, 70)],
         "beam_projector": [(35, 45), (45, 55)]}
CUR = (1, 3, 5, 7, 9, 11)            # inputs.VOLT_ORDERS — 지금 모델이 보는 것
EXT = (1, 3, 5, 7, 9, 11, 13, 15)    # npz 에 있는 홀수 전부


def build(vord, apps=None):
    """칸(녹화x전력대) 안 편차. (D 지문16, C 공변량, tags)"""
    Ds, Cs, tg = [], [], []
    for app, bands in BANDS.items():
        if apps and app not in apps:
            continue
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
                    x = [P[j].mean()]
                    for h in vord:
                        x += [v[h - 1].real, v[h - 1].imag]
                    X.append(x)
                Y = np.asarray(Y); X = np.asarray(X, float)
                if len(Y) < 2:
                    continue
                Ds.append(Y - Y.mean(0)); Cs.append(X - X.mean(0)); tg += [stem] * len(Y)
    return np.vstack(Ds), np.vstack(Cs), np.asarray(tg)


def loro(D, C, tags):
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


def report(name, apps):
    D0, _, t0 = build(CUR, apps)
    Vt = np.linalg.svd(D0 - D0.mean(0), full_matrices=False)[2]
    print("\n%s" % ("=" * 74))
    print("%s — 블록 %d · 녹화 %d" % (name, len(D0), len(np.unique(t0))))
    print("  %-28s %5s %8s %8s %8s %8s"
          % ("모델이 보는 전압", "채널", "전체", "PC1", "PC2", "PC3"))
    out = {}
    for lbl, vord in (("A 현행  h1~h11", CUR), ("B 확장  h1~h15", EXT)):
        D, C, tg = build(vord, apps)
        R = loro(D, C, tg)
        row = "  %-28s %5d" % (lbl, 2 * len(vord))
        tot = 1 - (R ** 2).sum() / (D ** 2).sum()
        row += " %7.3f" % tot
        for i in range(3):
            a = D @ Vt[i]; b = R @ Vt[i]
            row += " %7.3f" % (1 - (b ** 2).sum() / max((a ** 2).sum(), 1e-30))
        print(row)
        out[lbl] = tot
    d = out["B 확장  h1~h15"] - out["A 현행  h1~h11"]
    print("  %-28s %5s %+7.3f  <- 이만큼이 h13·h15 의 몫이다" % ("차이", "+4", d))

    # h13·h15 가 h1~h11 로 이미 예측되나 (되면 더해도 소용없다)
    _, Ca, ta = build(CUR, apps)
    _, Cb, _ = build(EXT, apps)
    tgt = Cb[:, Ca.shape[1]:]                     # h13·h15 의 Re/Im 4열
    Rr = loro(tgt, Ca, ta)
    r2 = 1 - (Rr ** 2).sum() / max((tgt ** 2).sum(), 1e-30)
    print("  h13·h15 를 h1~h11 로 예측한 LORO R^2 = **%.3f** %s"
          % (r2, "(이미 거의 담겨 있다)" if r2 > 0.8 else "(독립 정보가 있다)"))
    return d


def main():
    d1 = report("SMPS 둘 (충전기 + 프로젝터)", None)
    d2 = report("충전기만", ("laptop_charger",))
    print("\n%s" % ("=" * 74))
    print("읽는 법")
    print("  차이가 +0.05 미만이면  채널을 늘릴 값이 없다 (13.26 의 절단점이 여기서도 옳다)")
    print("  +0.1 이상이면          `VOLT_ORDERS` 를 h15 까지 늘릴 값이 있다")
    print("  ⚠ 늘리면 FINE_CHANNELS 57 -> 61 이라 **캐시를 다시 굽고** 옛 체크포인트와 규약이 어긋난다")
    print("  현재 차이: SMPS둘 %+.3f · 충전기 %+.3f" % (d1, d2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
