# -*- coding: utf-8 -*-
"""대기 전류로 **기기가 방에 있는지** 알 수 있나 (13.84.64).

13.84.51 ④: 미니PC 대기는 26.6mA 로 **통전 증분(28mA)과 같은 크기**다. 그리고 전이와 달리
**파일 내내 흐른다** — 580초 파일이면 34,860 사이클이다. 재현성 바닥 9~11% (13.84.36) 는
사이클 하나 기준이니, 그 많은 표본을 평균하면 원리적으로는 아주 작은 양도 잡힌다.

그런데 `src/run_diag_plugged.py` ① 이 보인 것: 모델은 **방에 없는 기기도 꽂혀 있다고 한다**
(test_5 는 SMPS 가 하나도 없는데 충전기 0.765 · 미니PC 0.552). 왜인지 model-free 로 묻는다.

⚠ **식별성이 이 물음의 핵심이다.** 관측은

    m_f = 배경_f + Σ_{있는 k} 대기지문_k

이고 `배경_f` 는 파일마다 **자유**다 (13.84.35 ⑧: 파일 간 3.6~22.6mA — 대기지문과 같은
크기다). 자유 상수 하나와 상수들의 합은 **완전히 축퇴**다. 그래서 세 가지로 나눠 묻는다:

  ① 수준    전부-OFF 구간의 전력·|I1|. 있는 SMPS 수를 따라가나 (축퇴여도 보이면 신호다)
  ② 기준빼기 SMPS 가 하나도 없는 test_5 를 배경 기준으로 삼아 차분한다. NNLS 로 복원
  ③ 식별성  그 차분의 잔차 = 배경 차이의 크기. 복원된 계수와 견줘 **믿을 만한가**
  ④ 방향    대기지문끼리·배경 변동과 몇 도인가. 작으면 원리적으로 못 가른다

    python -X utf8 src/run_diag_presence.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

FS = 60.0
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SMPS = ("beam_projector", "laptop_charger", "minipc")
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
OI = [o - 1 for o in ORD]


def alloff(stem, ev):
    """그 파일에서 **모든 기기가 꺼진** 사이클의 (중앙 페이저, 중앙 전력, 개수).

    ⚠ 라벨이 못 잡은 사건이 섞인다 (test_4 최대 66.7W · test_5 121.4W). 중앙을
    쓰고, 전력 상위 10% 를 떨군다 — 평균을 쓰면 그 몇 개가 끌고 간다.
    """
    r = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
    n = len(r["p_denoised_w"]); t = np.arange(n) / FS
    off = np.asarray(r["is_valid"]) == 1
    for a, v in ev[stem]["intervals"].items():
        for t0, t1 in v.get("on", []):
            off &= ~((t >= t0) & (t < t1))
    P = np.asarray(r["p_denoised_w"])
    if off.sum() < 60:
        return None
    off &= P <= np.percentile(P[off], 90)
    H = np.asarray(r["harmonics_complex"])[off]
    m = np.median(H.real, 0) + 1j * np.median(H.imag, 0)
    return m, float(np.median(P[off])), int(off.sum())


def nnls(A, y, ub=None, iters=500):
    """비음(및 상한) 최소제곱, 사영 경사."""
    x = np.zeros(A.shape[1])
    L = float(np.linalg.norm(A, 2) ** 2) or 1.0
    AtA, Aty = A.T @ A, A.T @ y
    for _ in range(iters):
        x = np.clip(x - (AtA @ x - Aty) / L, 0.0, np.inf if ub is None else ub)
    return x


def ri(c):
    return np.concatenate([np.real(c), np.imag(c)])


def ang(u, v):
    a, b = ri(u), ri(v)
    c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-18)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    sys.path.insert(0, ".")
    from src.model.net import standby_powers, standby_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="all", carrier_apps=("oven",))
    SB = standby_signatures(pool, APPS)                       # (K,15,2)
    SBC = (SB[..., 0] + 1j * SB[..., 1])[:, OI]               # (K,8)
    SBW = standby_powers(pool, APPS)
    del pool

    base = {f: alloff(f, ev) for f in FILES}
    pres = {f: np.array([a in ev[f]["appliances_present"] for a in APPS]) for f in FILES}

    # -- ① 수준 --------------------------------------------------------------
    print("① **수준** — 전부-OFF 구간. SMPS 가 몇 개 있느냐를 따라가나")
    print("   %-8s %7s %9s %10s %10s %10s  %s"
          % ("파일", "사이클", "전력 W", "참 대기합", "|I1| mA", "SMPS 수", "있는 SMPS"))
    for f in FILES:
        m, p, n = base[f]
        s = [a for a in SMPS if a in ev[f]["appliances_present"]]
        print("   %-8s %7d %9.2f %10.2f %10.1f %10d  %s"
              % (f, n, p, float(SBW[pres[f]].sum()), 1000 * abs(m[0]), len(s),
                 " ".join(x[:9] for x in s) or "—"))

    # -- ② 기준 빼기 ---------------------------------------------------------
    print("\n② **기준 빼기** — test_5 (SMPS 없음) 를 배경 기준으로. 차분을 9개 대기지문에 NNLS")
    print("   계수는 [0,1] 로 묶는다 — 있으면 1, 없으면 0 이어야 한다")
    ref = base["test_5"][0][OI]
    A = np.stack([ri(SBC[k]) for k in range(len(APPS))], 1)
    print("   %-8s  %s" % ("파일", "  ".join("%-9s" % a[:9] for a in SMPS)))
    for f in FILES:
        if f == "test_5":
            continue
        y = ri(base[f][0][OI] - ref)
        x = nnls(A, y, ub=1.0)
        cell = []
        for a in SMPS:
            k = APPS.index(a)
            cell.append("%.2f%s" % (x[k], "*" if pres[f][k] else " "))
        rr = np.linalg.norm(y - A @ x) / max(np.linalg.norm(y), 1e-12)
        print("   %-8s  %s   잔차 %5.1f%%   (별 = 실제로 있다)"
              % (f, "  ".join("%-9s" % c for c in cell), 100 * rr))

    # -- ③ 식별성 ------------------------------------------------------------
    print("\n③ **식별성** — 자유 배경의 크기 대 대기지문의 크기. 배경이 크면 ② 는 우연이다")
    dif = []
    for f in FILES:
        for g in FILES:
            if f < g and len(set(ev[f]["appliances_present"]) & set(SMPS)) == \
                         len(set(ev[g]["appliances_present"]) & set(SMPS)):
                dif.append(base[f][0][OI] - base[g][0][OI])
    print("   %-38s %10s" % ("양", "|.| mA (16차원 노름)"))
    for a in SMPS:
        print("   %-38s %10.1f" % ("대기지문  " + a, 1000 * np.linalg.norm(np.abs(SBC[APPS.index(a)]))))
    allp = np.stack([base[f][0][OI] for f in FILES])
    print("   %-38s %10.1f" % ("전부-OFF 페이저의 파일 간 표준편차",
                               1000 * np.linalg.norm(np.std(allp, 0))))
    if dif:
        print("   %-38s %10.1f" % ("**SMPS 구성이 같은** 파일끼리의 차 (n=%d)" % len(dif),
                                   1000 * np.median([np.linalg.norm(np.abs(d)) for d in dif])))
    else:
        print("   %-38s %10s" % ("SMPS 구성이 같은 파일 쌍", "없다 — 배경만 따로 못 잰다"))

    # -- ④ 방향 --------------------------------------------------------------
    print("\n④ **방향** — 대기지문끼리 몇 도인가 (작으면 누구 대기인지 원리적으로 못 가른다)")
    print("   %-30s %8s" % ("쌍", "각도"))
    for i in range(len(SMPS)):
        for j in range(i + 1, len(SMPS)):
            u, v = APPS.index(SMPS[i]), APPS.index(SMPS[j])
            print("   %-30s %7.0f°" % ("%s / %s" % (SMPS[i][:12], SMPS[j][:12]),
                                       ang(SBC[u], SBC[v])))
    print("\n   대기지문 대 **그 기기의 통전 지문** (다르면 대기는 독립된 단서다)")
    from src.model.net import harmonic_signatures
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="all", carrier_apps=("oven",))
    SG = harmonic_signatures(pool, APPS); del pool
    SGC = (SG[..., 0] + 1j * SG[..., 1])[:, OI]
    for a in SMPS:
        k = APPS.index(a)
        print("   %-30s %7.0f°" % (a, ang(SBC[k], SGC[k])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
