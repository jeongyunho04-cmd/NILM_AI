# -*- coding: utf-8 -*-
"""저항 다중 조합의 커버리지 — 합성 홀드아웃 대 실측 5파일 (13.84.17 ①).

외부 진단 4차: "고전력에서 합계가 초과하는 것은 저항 3~4종 동시 조합이 합성에 사실상 없기 때문".
총전력과 **동시에 켜진 저항 기기 수**의 결합 분포를 겹쳐 본다.

    python -X utf8 src/run_diag_coverage.py [processed_data/holdout60_v32]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.preprocessing import load_nilm_npz

RESIST = ("oven", "hotplate", "electiric_kettle", "hair_dryer", "fan")
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
FS, W, STRIDE = 60, 3600, 15 * 60
BANDS = [(0, 200), (200, 800), (800, 1500), (1500, 2200), (2200, 9999)]


def band(p):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= p < hi:
            return i
    return len(BANDS) - 1


def table(name, tot, nres, n_all):
    print("\n=== %s  (창 %d)" % (name, n_all))
    print("    전력대        전체몫   저항0    저항1    저항2    저항3+")
    for i, (lo, hi) in enumerate(BANDS):
        m = np.array([band(p) == i for p in tot])
        if m.sum() == 0:
            print("    %5d~%-5d  %6.2f%%      –        –        –        –" % (lo, hi, 0.0))
            continue
        nr = np.asarray(nres)[m]
        sh = [100 * (nr == k).mean() for k in (0, 1, 2)] + [100 * (nr >= 3).mean()]
        print("    %5d~%-5d  %6.2f%%  %6.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (lo, hi, 100 * m.mean(), *sh))
    nr = np.asarray(nres)
    print("    저항 동시 수 분포: " + " · ".join(
        "%d %.1f%%" % (k, 100 * (nr == k).mean()) for k in (0, 1, 2, 3))
        + " · 4+ %.2f%%" % (100 * (nr >= 4).mean()))


def main():
    hd = sys.argv[1] if len(sys.argv) > 1 else "processed_data/holdout60_v32"
    meta = json.load(open("%s/meta.json" % hd, encoding="utf-8"))
    apps = meta["appliances"]
    yp = np.load("%s/y_power.npy" % hd)
    yo = np.load("%s/y_on.npy" % hd)
    ri = [apps.index(a) for a in RESIST if a in apps]
    tot_s = yp.sum(1)
    nres_s = (yo[:, ri] > 0.5).sum(1)
    table("합성 홀드아웃 %s" % hd.split("/")[-1], tot_s, nres_s, len(tot_s))

    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    tot_r, nres_r = [], []
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        valid = np.asarray(r["is_valid"]).astype(bool)
        masks = []
        for a in RESIST:
            m = np.zeros(n, bool)
            for t0, t1 in ev[stem]["intervals"].get(a, {}).get("on", []):
                m[int(t0 * FS):int(min(t1 * FS, n))] = True
            masks.append(m)
        M = np.stack(masks)
        for t0 in range(0, n - W + 1, STRIDE):
            tl = slice(t0 + W - 600, t0 + W)
            if not valid[t0:t0 + W].all():
                continue
            tot_r.append(float(P[tl].mean()))
            nres_r.append(int((M[:, tl].mean(1) > 0.5).sum()))
    table("실측 5파일", tot_r, nres_r, len(tot_r))

    print("\n--- 고전력(1500W+) 창에서 저항 3종 이상이 같이 켜진 비율")
    for nm, tot, nres in (("합성", tot_s, nres_s), ("실측", tot_r, nres_r)):
        tot = np.asarray(tot); nres = np.asarray(nres)
        hi = tot >= 1500
        print("    %-4s  1500W+ 창 %5d (%5.2f%%)   그중 저항3+ %5.1f%%   2200W+ 창 %4d (%.2f%%)"
              % (nm, hi.sum(), 100 * hi.mean(),
                 100 * (nres[hi] >= 3).mean() if hi.sum() else 0.0,
                 (tot >= 2200).sum(), 100 * (tot >= 2200).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
