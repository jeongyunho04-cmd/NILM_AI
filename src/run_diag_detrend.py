# -*- coding: utf-8 -*-
"""추세 제거 재측정 v2 — 진단에 유리한 규약으로.

A) 계단 과제: σ 는 **사건에서 20초 넘게 떨어진 정상 구간**에서 재고(진단의 '구간별 σ'),
   계단은 전이 앞뒤 중앙값 차. 추세 제거판의 계단 진폭은 중심 이동평균이라 전이 직후
   ≈ 계단/2 이므로 그렇게 잰다(실측으로도 확인).
B) 상태 과제: stride 5초로 표본을 늘리고, 조건을 (i) 충전기+프로젝터 (ii) 형제 무관 둘 다.
C) ±10초(광역에서만 가능) 와 ±2.5초(세밀에서 가능) 를 나란히.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.preprocessing import load_nilm_npz

FS = 60
FILES = ["test_4", "test_2", "test_1", "test_3"]
SIB = ("laptop_charger", "beam_projector", "fan", "hotplate", "oven", "hair_dryer", "electiric_kettle")
HALVES = {"±10초(광역만)": 10 * FS, "±2.5초(세밀가능)": int(2.5 * FS)}


def movavg(a, half):
    k = 2 * half + 1
    pad = np.pad(a, (half, half), mode="edge")
    c = np.cumsum(np.insert(pad, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def feats(r):
    H = r["harmonics_complex"]
    P = np.asarray(r["power_features"])[:, 0]
    Q = np.asarray(r["power_features"])[:, 1]
    i1 = H[:, 0]
    mag = np.abs(H)
    return {
        "∠I1 °": np.degrees(np.angle(i1)),
        "Q/P": Q / np.maximum(P, 1e-6),
        "|I3|/|I1|": mag[:, 2] / (mag[:, 0] + 1e-12),
    }, P


def masks(spec, n):
    out = {}
    for a, d in spec["intervals"].items():
        m = np.zeros(n, bool)
        for t0, t1 in d.get("on", []):
            m[int(t0 * FS):int(min(t1 * FS, n))] = True
        out[a] = m
    return out


def fisher(a, b):
    if len(a) < 4 or len(b) < 4:
        return float("nan")
    s = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
    return abs(a.mean() - b.mean()) / s


W, STRIDE, TGT = 60 * FS, 5 * FS, 10 * FS
for stem in FILES:
    r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"][stem]
    F, P = feats(r)
    n = len(P)
    valid = np.asarray(r["is_valid"]).astype(bool)
    mk = masks(ev, n)
    y = mk.get("minipc", np.zeros(n, bool))
    all_t = sorted(t for a, d in ev["intervals"].items() for iv in d.get("on", []) for t in iv)
    # 모든 사건에서 20초 넘게 떨어진 정상 표본
    quiet = np.ones(n, bool)
    for t in all_t:
        i = int(t * FS)
        quiet[max(0, i - 20 * FS):min(n, i + 20 * FS)] = False
    quiet &= valid
    cond = mk.get("laptop_charger", np.zeros(n, bool)).copy()
    if "beam_projector" in mk:
        cond &= mk["beam_projector"]
    print("\n=== %s  정상 표본 %.0f%%" % (stem, 100 * quiet.mean()))

    for hname, half in HALVES.items():
        line = []
        for name, x in F.items():
            xd = x - movavg(x, half)
            sr = float(x[quiet].std()) if quiet.sum() > 100 else np.nan
            sd = float(xd[quiet].std()) if quiet.sum() > 100 else np.nan
            steps = []
            for t0, t1 in ev["intervals"].get("minipc", {}).get("on", []):
                for t, sgn in ((t0, +1), (t1, -1)):
                    i0 = int(t * FS)
                    if i0 - 13 * FS < 0 or i0 + 13 * FS > n or not valid[i0 - 13 * FS:i0 + 13 * FS].all():
                        continue
                    steps.append(sgn * (np.median(x[i0 + 3 * FS:i0 + 13 * FS])
                                        - np.median(x[i0 - 13 * FS:i0 - 3 * FS])))
            if not steps:
                continue
            st = float(np.median(np.abs(steps)))
            line.append((name, sr, sd, st, st / (sr + 1e-12), 0.5 * st / (sd + 1e-12)))
        if line:
            print("  A) %s        σ정상원   σ정상추세   계단     d원   d추세제거" % hname)
            for nm, sr, sd, st, d1, d2 in line:
                print("     %-11s %9.4f %9.4f %9.4f %7.2f %8.2f" % (nm, sr, sd, st, d1, d2))

    idx = [t0 for t0 in range(0, n - W + 1, STRIDE) if valid[t0:t0 + W].all()]
    tgt = [slice(t0 + W - TGT, t0 + W) for t0 in idx]
    for cname, sel in (("충전기+프로젝터", cond), ("형제 무관", np.ones(n, bool))):
        keep = [k for k, sl in enumerate(tgt) if sel[sl].all()]
        lab = np.array([bool(y[tgt[k]].all()) for k in keep])
        neg = np.array([not y[tgt[k]].any() for k in keep])
        if lab.sum() < 4 or neg.sum() < 4:
            print("  B) %-14s 표본 부족 (ON %d / OFF %d)" % (cname, lab.sum(), neg.sum()))
            continue
        out = []
        for name, x in F.items():
            row = [name]
            for half in HALVES.values():
                xd = x - movavg(x, half)
                v1 = np.array([x[tgt[k]].mean() for k in keep])
                v2 = np.array([xd[tgt[k]].mean() for k in keep])
                row += [fisher(v1[lab], v1[neg]), fisher(v2[lab], v2[neg])]
            out.append(row)
        print("  B) %-14s ON %d / OFF %d      d원   d추세±10   d원   d추세±2.5"
              % (cname, lab.sum(), neg.sum()))
        for row in out:
            print("     %-11s          %14.2f %9.2f %6.2f %9.2f"
                  % (row[0], row[1], row[2], row[3], row[4]))
