# -*- coding: utf-8 -*-
"""세션 이동을 **실측에서** 잰다 — 합성 결론의 확인 (13.84.35).

합성에서 잰 것: 켜진 기기·전력·꽂힘이 같은 창끼리 관측 차이가
**같은 기록 안 3~4 mA 대 다른 기록 사이 18~23 mA (5~6배)** 였다.
⚠ 합성의 "다른 기록" 은 생성기가 녹화 조각을 새로 뽑는 것이라 구성상 당연하다.
실측에서 같은 모양이 나오는지 봐야 인공물이 아니다.

실측엔 기기별 참값 와트가 없다(13.84.18). 그래서 **구성(참 ON 집합)과 관측 총전력**이
같은 정상 구간끼리만 견준다. 그러면 와트를 몰라도 견줄 수 있다.
  파일 안  = 같은 녹화 세션 안 (바닥)
  파일 간  = 다른 세션 (세션 이동)  — 같은 자리끼리도 따로 낸다 (v_mean 으로 가름)

    python -X utf8 src/run_diag_ss_real.py [--pw 5] [--seg 10]
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from scipy.stats import trim_mean

from src.preprocessing import load_nilm_npz

FS = 60
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
#: 13.84.31 — test_3·test_4 cos +0.982 (같은 자리) · test_2 는 반대. v_mean 도 그렇게 갈린다.
SITE = {"test_1": "D", "test_3": "D", "test_4": "D", "test_5": "D", "test_2": "E"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", type=float, default=5.0, help="총전력 격자 W")
    ap.add_argument("--seg", type=int, default=10, help="정상 구간 길이 s")
    ap.add_argument("--margin", type=int, default=5, help="전이에서 띄울 초")
    a = ap.parse_args()
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]

    segs = []                                     # (파일, 구성, 전력격자, 페이저, 총전력)
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        spec = ev[stem]
        apps = sorted(spec["appliances_present"])
        mk = {}
        for x in apps:
            m = np.zeros(n, bool)
            for t0, t1 in spec["intervals"].get(x, {}).get("on", []):
                m[int(t0 * FS):int(min(t1 * FS, n))] = True
            mk[x] = m
        lab = np.stack([mk[x] for x in apps])
        chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
        for s0, s1 in zip(chg[:-1], chg[1:]):
            s0, s1 = s0 + a.margin * FS, s1 - a.margin * FS
            L = a.seg * FS
            for t in range(s0, s1 - L, L):
                cfg = tuple(x for x in apps if mk[x][t])
                pw = float(trim_mean(P[t:t + L], 0.2))
                v = trim_mean(H[t:t + L].real, 0.2, axis=0) + 1j * trim_mean(H[t:t + L].imag, 0.2, axis=0)
                segs.append((stem, cfg, int(round(pw / a.pw)), v, pw))
    print("정상 구간 %d개 (%ds · 전이에서 %ds 띄움 · 전력 격자 %gW)"
          % (len(segs), a.seg, a.margin, a.pw))

    grp = defaultdict(list)
    for i, (stem, cfg, pb, v, pw) in enumerate(segs):
        grp[(cfg, pb)].append(i)
    oi = [o - 1 for o in ORD]
    buckets = {"in": [], "cross_same": [], "cross_diff": []}
    npair = defaultdict(int)
    for (cfg, pb), idx in grp.items():
        if len(idx) < 2:
            continue
        for i, j in combinations(idx, 2):
            fi, fj = segs[i][0], segs[j][0]
            d = np.abs(segs[i][3][oi] - segs[j][3][oi]) / np.sqrt(2)
            if fi == fj:
                k = "in"
            elif SITE[fi] == SITE[fj]:
                k = "cross_same"
            else:
                k = "cross_diff"
            buckets[k].append(d); npair[k] += 1
    for k in buckets:
        buckets[k] = np.asarray(buckets[k]) if buckets[k] else np.zeros((0, len(ORD)))
    print("짝  파일 안 %d · 파일 간(같은 자리) %d · 파일 간(다른 자리) %d"
          % (npair["in"], npair["cross_same"], npair["cross_diff"]))

    from src.model.net import harmonic_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="all", carrier_apps=("oven",))
    sg = harmonic_signatures(pool, ["minipc"]); del pool

    print("\n%-5s %12s %16s %16s | %10s %8s"
          % ("차수", "파일 안", "파일 간(같은자리)", "파일 간(다른자리)", "미니PC12W", "배"))
    for jj, o in enumerate(ORD):
        row = []
        for k in ("in", "cross_same", "cross_diff"):
            row.append(1000 * np.median(buckets[k][:, jj]) if len(buckets[k]) else np.nan)
        mp = 1000 * abs(sg[0, o - 1, 0] + 1j * sg[0, o - 1, 1]) * 12.0
        print("  h%-4d %10.1f %15.1f %16.1f | %9.1f %7.1f"
              % (o, row[0], row[1], row[2], mp,
                 row[1] / row[0] if row[0] and row[0] == row[0] else np.nan))
    print("\n  '배' = 파일 간(같은 자리) / 파일 안. 합성에서는 5~6배였다.")
    print("  같은 자리끼리도 크면 **자리(환경)가 아니라 세션**이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
