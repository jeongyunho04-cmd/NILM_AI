# -*- coding: utf-8 -*-
"""펌웨어 전압 꼬리(h17~31)가 SMPS 전류 고조파를 실제로 움직이나 — 실측 (13.84.36).

사용자: *"이게 펌웨어 바꿔서 얻은 15차 이상 전압 고조파 정보도 포함한 결과인가?"*
답은 아니오였다. 꼬리는 **생성기에만** 들어가고(`vtail=True`) 모델 입력은 `VOLT_ORDERS=(1,3,5,7,9,11)`
에서 끊긴다. 합성에서는 꼬리가 **세션당 상수 7개**뿐이라 연속 변동을 시험할 수 없다.

그래서 원시 CSV 로 간다. 새 펌웨어 세션 E3(96열)은 `vhhi_seq` 가 ~61.7초마다 1 오르고
블록 안에서 꼬리가 정확히 상수다 — **블록 하나 = 꼬리 하나**. 블록마다 같은 동작점에서
와트당 전류 페이저를 뽑아, 그 변동이 꼬리를 따라가는지 본다.

⚠ 시간과 꼬리가 같이 흐르므로 **블록 순번을 대조 예측자**로 같이 낸다. 꼬리가 시간보다
  잘 맞춰야 꼬리 탓이다 ([[validate-both-directions]] · [[check-the-ruler-against-a-known-value]]).

    python -X utf8 src/run_diag_vtail_probe.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import pandas as pd

TAIL = [17, 19, 21, 23, 25, 27, 29, 31]
IORD = [1, 3, 5, 7, 9, 11, 13, 15]
STEMS = {"laptop_charger_3": (55, 80), "laptop_charger_4": (55, 80),
         "laptop_charger_5": (55, 80), "laptop_charger_6": (55, 80),
         "minipc_4": (7, 12)}


def blocks(stem, lo, hi, min_cyc=200):
    cols = (["t_s", "p_w", "vrms", "pll_locked", "vh1", "vhhi_seq"]
            + ["vhhi%d" % h for h in TAIL] + ["vhhideg%d" % h for h in TAIL]
            + ["ih%d" % h for h in IORD] + ["ihdeg%d" % h for h in IORD])
    df = pd.read_csv("data/%s.csv" % stem, usecols=cols, low_memory=False)
    df = df[(df["pll_locked"] > 0.5) & np.isfinite(df["vh1"]) & (df["vh1"] > 0)]
    out = []
    for s, g in df.groupby("vhhi_seq"):
        gp = g[(g["p_w"] >= lo) & (g["p_w"] <= hi)]
        if len(gp) < min_cyc:
            continue
        v1 = float(np.median(g["vh1"]))
        tail = np.array([np.median(g["vhhi%d" % h]) * np.exp(1j * np.deg2rad(np.median(g["vhhideg%d" % h]))) / v1
                         for h in TAIL])
        p = np.median(gp["p_w"])
        cur = np.array([(np.median(gp["ih%d" % h] * np.cos(np.deg2rad(gp["ihdeg%d" % h])))
                         + 1j * np.median(gp["ih%d" % h] * np.sin(np.deg2rad(gp["ihdeg%d" % h])))) / p
                        for h in IORD])
        out.append(dict(stem=stem, seq=float(s), t=float(np.median(g["t_s"])), p=float(p),
                        n=len(gp), tail=tail, cur=cur, vrms=float(np.median(g["vrms"]))))
    return out


def main():
    B = []
    for stem, (lo, hi) in STEMS.items():
        b = blocks(stem, lo, hi)
        B.extend(b)
        if b:
            T = np.stack([x["tail"] for x in b])
            print("  %-18s 블록 %2d · %4.0f~%4.0fW · 꼬리 |h17| %.4f%% (블록간 CV %.0f%%)"
                  % (stem, len(b), min(x["p"] for x in b), max(x["p"] for x in b),
                     100 * np.median(np.abs(T[:, 0])),
                     100 * np.std(np.abs(T[:, 0])) / (np.mean(np.abs(T[:, 0])) + 1e-18)))
    if not B:
        print("블록 없음"); return 1
    print("\n블록 %d개 (기기 2종)" % len(B))
    for dev, pick in (("laptop_charger", lambda x: x["stem"].startswith("laptop")),
                      ("minipc", lambda x: x["stem"].startswith("minipc"))):
        b = [x for x in B if pick(x)]
        if len(b) < 6:
            print("\n== %s == 블록 %d개 — 부족" % (dev, len(b))); continue
        T = np.stack([x["tail"] for x in b])
        Cn = np.stack([x["cur"] for x in b])
        tt = np.asarray([x["t"] for x in b]); pp = np.asarray([x["p"] for x in b])
        print("\n== %s ==  블록 %d · 전력 %0.1f~%0.1fW · V %.1f~%.1fV"
              % (dev, len(b), pp.min(), pp.max(),
                 min(x["vrms"] for x in b), max(x["vrms"] for x in b)))
        print("  꼬리 블록간 산포 (vh1 대비 %%):  %s"
              % "  ".join("h%d %.3f" % (h, 100 * np.std(np.abs(T[:, j])))
                          for j, h in enumerate(TAIL[:4])))
        # 와트당 전류가 블록마다 얼마나 다른가 + 무엇이 그것을 예측하나
        X = np.concatenate([T.real, T.imag], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-18)
        u, s_, vt = np.linalg.svd(X, full_matrices=False)
        pc = u[:, :2] * s_[:2]                       # 꼬리의 주성분 2개
        ctl = np.stack([(tt - tt.mean()) / (tt.std() + 1e-9),
                        (pp - pp.mean()) / (pp.std() + 1e-9)], 1)   # 대조: 시간·전력
        print("  %-5s %11s | %10s %10s %10s"
              % ("차수", "블록간 CV", "꼬리 R²", "시간·전력 R²", "판정"))
        for j, o in enumerate(IORD):
            y = Cn[:, j]
            cv = np.std(np.abs(y)) / (np.mean(np.abs(y)) + 1e-18)

            def loo(Z):
                num = den = 0.0
                for part in (np.real, np.imag):
                    yy = part(y)
                    for i in range(len(yy)):
                        m = np.ones(len(yy), bool); m[i] = False
                        A = np.c_[Z[m], np.ones(m.sum())]
                        w, *_ = np.linalg.lstsq(A, yy[m], rcond=None)
                        num += (yy[i] - np.r_[Z[i], 1.0] @ w) ** 2
                    den += float(np.sum((yy - yy.mean()) ** 2))
                return 1.0 - num / max(den, 1e-30)
            r_t, r_c = loo(pc), loo(ctl)
            print("  h%-4d %10.1f%% | %9.3f %10.3f %10s"
                  % (o, 100 * cv, r_t, r_c, "꼬리" if r_t > r_c + 0.05 and r_t > 0 else "-"))
        print("  (교차검증 R² — 음수는 평균보다 못하다는 뜻이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
