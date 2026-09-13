# -*- coding: utf-8 -*-
"""0단계 (ii) — **실제로 등장하는 활성집합**마다 사전의 조건수 (재설계 계획 B 관문).

계획 B 는 기기별 전력 헤드를 **펼친 NNLS 반복**으로 바꾼다. 그러려면 사전
`D_S = [sig_k]_{k in S}` 가 풀 만해야 한다. 안 풀리면 배분 오차가 조건수만큼 증폭된다.

⚠ **자를 먼저 잰다** (memory: check-conditioning-before-believing-a-fit).
13.84.51 이 잰 `cond[1,P] = 179,850` 은 **다른 양**이다 — 그건 기기 하나의 (1,P)
설계행렬이고, 여기 필요한 건 *동시에 켜진 기기들의* 30차원(Re15+Im15) × |S| 행렬이다.
그 둘을 같은 수로 취급하면 안 된다.

세 가지를 낸다.
  1. test_1~5 의 2초 격자에서 실제로 나타나는 활성집합과 그 빈도
  2. 집합마다 cond(D_S) — 차수별 정규화(`harmonic_scales`) 공간에서
  3. 집합 안에서 **가장 닮은 쌍**과 그 사이각 — 조건수를 누가 망치는지

사전은 `harmonic_signatures` 의 **와트당 상수 페이저**다. 이것이 상한이 아니라
하한임에 주의: 13.84.23 이 SMPS 에서 잔차 17~27% 를 쟀고 원인을 비선형(도통각이
전력에 따라 변함)으로 지목했다. `sig_model.sig(P,V)` 가 그 보정판이다.

    python -X utf8 src/run_diag_dictcond.py
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.net import harmonic_scales, harmonic_signatures
from src.synthesis.segment_pool import SegmentPool

FS = 60
GRID_S = 2.0
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
RESISTIVE = {"electiric_kettle", "oven", "hair_dryer", "hotplate", "fan"}
SMPS = {"laptop_charger", "beam_projector", "minipc"}
SHORT = {"electiric_kettle": "포트", "oven": "오븐", "hair_dryer": "드라이기",
         "hotplate": "핫플", "fan": "선풍", "laptop_charger": "충전기",
         "beam_projector": "프로젝터", "minipc": "미니PC", "air_conditioner": "에어컨"}


def columns(sig, scale):
    """(K,15,2) 와트당 페이저 -> (30,K) 정규화 실수 열."""
    s = np.maximum(scale, 1e-12)[None, :]              # (1,15)
    re = sig[:, :, 0] / s
    im = sig[:, :, 1] / s
    return np.concatenate([re, im], axis=1).T.astype(np.float64)   # (30,K)


def worst_pair(D, names):
    """열 사이 최소 사이각 (도) 과 그 쌍."""
    n = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-30)
    best, pair = 180.0, ("", "")
    for i in range(D.shape[1]):
        for j in range(i + 1, D.shape[1]):
            c = float(np.clip(abs(n[:, i] @ n[:, j]), 0.0, 1.0))
            a = np.degrees(np.arccos(c))
            if a < best:
                best, pair = a, (names[i], names[j])
    return best, pair


def group_of(s):
    r = len(s & RESISTIVE)
    m = len(s & SMPS)
    if r and m:
        return "혼합"
    return "저항성만" if r else ("SMPS만" if m else "-")


def main():
    print("0단계 (ii) — 활성집합별 사전 조건수", flush=True)
    print("=" * 78)
    pool = SegmentPool()
    sig = harmonic_signatures(pool, APPS)
    scale = harmonic_scales(pool, APPS)
    D_all = columns(sig, scale)
    live = [a for j, a in enumerate(APPS) if np.linalg.norm(D_all[:, j]) > 0]
    print("사전: 와트당 상수 페이저 · 30차원(Re15+Im15) · 차수별 정규화")
    print("      열이 살아 있는 기기 %d/%d" % (len(live), len(APPS)))

    ev = load_events()
    sets = Counter()
    for stem in FILES:
        d = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
        n = int(d["is_on"].shape[0])
        apps = sorted(ev[stem]["intervals"].keys())
        on, sc = build_on_off_truth(stem, apps, n, events=ev)
        on, sc = np.asarray(on, bool), np.asarray(sc, bool)
        idx = np.arange(0, n, int(round(GRID_S * FS)))
        for t in idx:
            if not sc[t].all():
                continue                      # uncertain 이 낀 시각은 뺀다
            s = frozenset(a for j, a in enumerate(apps) if on[t, j])
            if s:
                sets[s] += 1

    print()
    print("실제로 등장하는 활성집합 %d 종 (2초 격자 %d 시각)"
          % (len(sets), sum(sets.values())))
    print()
    print("  빈도   |S|  무리        cond(D_S)   최소사이각   가장 닮은 쌍")
    print("  " + "-" * 74)
    rows = []
    for s, cnt in sets.most_common():
        names = sorted(s)
        cols = [APPS.index(a) for a in names]
        D = D_all[:, cols]
        if min(np.linalg.norm(D, axis=0)) <= 0:
            continue
        sv = np.linalg.svd(D, compute_uv=False)
        cond = float(sv[0] / max(sv[-1], 1e-30)) if len(s) > 1 else 1.0
        ang, pair = worst_pair(D, names) if len(s) > 1 else (180.0, ("", ""))
        g = group_of(set(s))
        lbl = "+".join(SHORT.get(a, a) for a in names)
        pr = ("%s~%s" % (SHORT.get(pair[0], pair[0]), SHORT.get(pair[1], pair[1]))
              if pair[0] else "—")
        print("  %5d   %d   %-9s  %9.1f   %7.2f°   %-18s %s"
              % (cnt, len(s), g, cond, ang, pr, lbl))
        rows.append((cnt, len(s), g, cond, ang))

    print()
    print("무리별 요약 (시각 가중)")
    print("  무리        시각      cond 중앙   cond 최대   최소사이각 최소")
    for g in ("저항성만", "SMPS만", "혼합"):
        sel = [r for r in rows if r[2] == g and r[1] > 1]
        if not sel:
            print("  %-9s  (|S|>1 인 집합 없음)" % g)
            continue
        w = np.array([r[0] for r in sel], float)
        c = np.array([r[3] for r in sel], float)
        a = np.array([r[4] for r in sel], float)
        o = np.argsort(c)
        cw = np.cumsum(w[o]) / w.sum()
        med = float(c[o][np.searchsorted(cw, 0.5)])
        print("  %-9s  %6d   %9.1f   %9.1f   %9.2f°"
              % (g, int(w.sum()), med, float(c.max()), float(a.min())))

    print()
    print("판정 기준 (계획 B 관문 — 미리 적는다):")
    print("  cond < 10^2   -> 펼친 NNLS 가 안정. 그대로 간다")
    print("  10^2 ~ 10^4   -> 정규화(사전확률·L2)가 필요. 신경망 제안이 초기값으로 들어가야 한다")
    print("  > 10^4        -> 그 집합에서는 B 가 죽는다. K=0 (지금 헤드) 로 물러선다")
    print()
    print("⚠ 이 사전은 **와트당 상수**다. SMPS 는 도통각이 전력에 따라 변하므로")
    print("  (13.84.23) 실제 조건은 이보다 나쁠 수 있다. sig(P,V) 판으로 재측정할 것.")


if __name__ == "__main__":
    main()
