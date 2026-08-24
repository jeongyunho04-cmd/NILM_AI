"""
차분 특징 2차 제안 — ΔQ/ΔP 와 Δφ3 (설계 문서 12.55절)
========================================================
12.53.7 이 세운 자("그 차수의 |Δ| 가 복합 널 p95 보다 큰가")를 겨냥해 사용자가
두 가지를 더 제안했다.

    ④ ΔQ/ΔP        무효전력 차분비. ΔP 가 15~48W 로 크고, 저항 부하는 ΔQ=0 이라
                    ΔQ 의 널이 깨끗할 것이라는 논거
    ⑤ Δφ3          **차분 벡터**의 3차 고조파 위상 불변량.
                    |ΔI3| 가 100~190mA 로 널 p95(59mA)를 넘으므로 그 벡터의
                    **방향**은 안정적일 것이라는 논거

기존 `ch31 = asinh(Q/100)` 과 `ch38~45 = 순시 φ_h` 가 있다. 새로운 것은
**차분 형태**이고 ④는 **비율**이다.

[⚠ 파일의 Q 는 제안이 말하는 Q 가 아니다 — 12.55.1]
전처리의 `Q = sqrt(max(0, S^2 - P^2)) * sign` 은 **왜곡전력 D 를 포함한 총 무효분**
이다. SMPS 는 THD 가 커서 D 가 지배하므로 3종 모두 Q/P ~ -1.3 (PF~0.6) 로 같아진다.
제안이 말하는 것은 **기본파 변위 무효전력**이다. 고조파 위상이 전압 기준이므로
(검산: 프로젝터 P 48.64W vs V*Re(I1) 48.41W) 복소 기본파에서 직접 뽑는다:

    Q1 = -V * Im(I1)

이 파일은 **Q1 을 쓴다.** 파일의 Q 로 재면 판별력이 0.16 으로 사라진다.

    python -m src.run_dq_dphi_probe

[⑤ 는 켜짐/꺼짐 방향에 불변이다 — 확인할 것]
    ΔI_h -> −ΔI_h 이면  arg(ΔI3)+π − 3(arg(ΔI1)+π) = φ3 − 2π ≡ φ3
꺼질 때와 켜질 때가 같은 값이어야 한다. 아니면 계산이 틀린 것이다.

[재는 것]
    A 격리 녹화   주장 검증 + 분리비 (12.35.1)
    B 복합 전이   실제 판별력 — 본 시험
    C 널          ΔQ 의 널 p95 (12.53.7 의 자)
    D 저항 3종    사용자 요청 — 저항 부하에도 영향이 있는가
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import load_events
from src.model.realdata import DEFAULT_DIR
from src.run_delta_feature_probe import (DEV_DIR, DEV_FILES, SMPS, load_cplx,
                                         sepratio, transitions)
from src.run_live import KOR

RESIST = ("oven", "hotplate", "electiric_kettle")
ALL = SMPS + RESIST


def phi3(dI: np.ndarray) -> float:
    """차분 복소 벡터의 3차 위상 불변량 (도). arg(ΔI3) − 3·arg(ΔI1)."""
    a = np.angle(dI[2]) - 3.0 * np.angle(dI[0])
    return float(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))


def circ(deg: np.ndarray):
    """원형 평균(도)과 산포(도). R=1 이면 완전히 모여 있다."""
    r = np.radians(deg)
    C, S = np.cos(r).mean(), np.sin(r).mean()
    R = float(np.hypot(C, S))
    mean = float(np.degrees(np.arctan2(S, C)))
    std = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))))
    return mean, std, R


def circ_sep(a: np.ndarray, b: np.ndarray) -> float:
    """각도판 분리비: 원형 평균 간 각거리 / 평균 원형 산포."""
    ma, sa, _ = circ(a)
    mb, sb, _ = circ(b)
    d = abs((ma - mb + 180) % 360 - 180)
    return float(d / max(0.5 * (sa + sb), 1e-9))


def isolated_dq() -> Dict[str, List[dict]]:
    """격리 녹화: ON 블록마다 배경(OFF)을 뺀 ΔP, ΔQ, 복소 고조파."""
    out: Dict[str, List[dict]] = {}
    for app, files in DEV_FILES.items():
        for stem in files:
            f = DEV_DIR / f"{stem}.npz"
            if not f.exists():
                continue
            I, pf = load_cplx(f)
            p = pf[:, 0]
            q = -pf[:, 4] * I[:, 0].imag        # 기본파 변위 무효전력 Q1
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            if on.sum() < 300 or (~on).sum() < 100:
                continue
            bgI = np.median(I[~on].real, 0) + 1j * np.median(I[~on].imag, 0)
            bgp, bgq = float(np.median(p[~on])), float(np.median(q[~on]))
            idx = np.flatnonzero(on)
            for blk in np.array_split(idx, max(1, len(idx) // 600)):
                if len(blk) < 200:
                    continue
                dI = (np.median(I[blk].real, 0) + 1j * np.median(I[blk].imag, 0)) - bgI
                dp = float(np.median(p[blk])) - bgp
                dq = float(np.median(q[blk])) - bgq
                if dp <= 1 or abs(dI[0]) < 1e-4:
                    continue
                out.setdefault(app, []).append({
                    "file": stem, "dP": dp, "dQ": dq, "dQ_dP": dq / dp,
                    "dphi3": phi3(dI), "dI3_mA": float(abs(dI[2]) * 1000)})
    return out


def composite_dq(half=5.0, guard=1.0, snap=6.0, n_null=150):
    """복합 실측: 라벨된 전이의 ΔQ/ΔP·Δφ3, 그리고 널."""
    ev = load_events()
    rows, nulls = [], []
    rng = np.random.default_rng(0)
    for stem in ("test_4", "test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        I, pf = load_cplx(f)
        p = pf[:, 0]
        q = -pf[:, 4] * I[:, 0].imag           # 기본파 변위 무효전력 Q1
        n = len(p)
        t = np.arange(n) / 60.0
        i3 = np.abs(I[:, 2])
        T = transitions(ev, stem, ALL)
        # 저항 전이는 오븐 히터 펄스도 넣는다
        for s0, e0 in (ev[stem]["intervals"].get("oven", {}).get("_heater_pulses") or []):
            T += [(float(s0), "oven", +1), (float(e0), "oven", -1)]
        T.sort()
        tt = np.array([x[0] for x in T]) if T else np.zeros(0)
        for i, (t0, app, sg) in enumerate(T):
            # 창을 이웃 사건 간격에 맞춘다 (12.53.2 의 교훈)
            gap = min(t0 - tt[i - 1] if i > 0 else 99,
                      tt[i + 1] - t0 if i < len(tt) - 1 else 99)
            h = float(np.clip(0.45 * gap, 0.25, half))
            g = min(guard, 0.25 * h)
            sn = min(snap, 0.5 * gap)
            if t0 < h + sn + 1 or t0 > t[-1] - h - sn - 1:
                continue
            best, bd = t0, -1.0
            for c in np.arange(t0 - sn, t0 + sn + 1e-9, 0.25):
                pre = (t >= c - h) & (t <= c - g)
                post = (t >= c + g) & (t <= c + h)
                if pre.sum() < 8 or post.sum() < 8:
                    continue
                d = abs(np.median(i3[post]) - np.median(i3[pre]))
                if d > bd:
                    best, bd = c, d
            pre = (t >= best - h) & (t <= best - g)
            post = (t >= best + g) & (t <= best + h)
            if pre.sum() < 8 or post.sum() < 8:
                continue
            dI = ((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                  - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0)))
            dp = float(np.median(p[post]) - np.median(p[pre]))
            dq = float(np.median(q[post]) - np.median(q[pre]))
            if abs(dp) < 5 or abs(dI[0]) < 1e-4:
                continue
            rows.append({"stem": stem, "t_s": t0, "app": app, "sign": sg,
                         "dP": dp, "dQ": dq, "dQ_dP": dq / dp,
                         "dphi3": phi3(dI), "dI3_mA": float(abs(dI[2]) * 1000)})
        cand = [c for c in np.arange(half + 7, t[-1] - half - 7, 2.0)
                if len(tt) == 0 or np.min(np.abs(tt - c)) > 12]
        for c in rng.choice(cand, min(n_null, len(cand)), replace=False):
            pre = (t >= c - half) & (t <= c - guard)
            post = (t >= c + guard) & (t <= c + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            nulls.append((float(np.median(p[post]) - np.median(p[pre])),
                          float(np.median(q[post]) - np.median(q[pre]))))
    return rows, np.array(nulls)


def main() -> int:
    ap = argparse.ArgumentParser(description="ΔQ1/ΔP · Δφ3 검증 (12.55절)")
    ap.add_argument("--out", default="results/dq_dphi_probe.json")
    a = ap.parse_args()

    print("=" * 98)
    print("[A] 격리 녹화 — 배경을 뺀 순수 기기 값")
    print("=" * 98)
    iso = isolated_dq()
    print(f"  {'기기':<12s}{'블록':>5s}{'ΔP중앙':>9s}{'ΔQ1중앙':>10s}"
          f"{'ΔQ1/ΔP':>20s}{'Δφ3 (도)':>22s}")
    print("  " + "-" * 80)
    for app in ALL:
        v = iso.get(app)
        if not v:
            continue
        dp = np.array([x["dP"] for x in v]); dq = np.array([x["dQ"] for x in v])
        r = np.array([x["dQ_dP"] for x in v]); ph = np.array([x["dphi3"] for x in v])
        m, sd, R = circ(ph)
        print(f"  {KOR.get(app, app):<12s}{len(v):>5d}{np.median(dp):>8.1f}W"
              f"{np.median(dq):>9.1f}v{np.median(r):>13.3f} ±{r.std():<5.3f}"
              f"{m:>13.1f} ±{sd:<5.1f} R={R:.2f}")

    print()
    print("  분리비 (2 미만 겹침).  Δφ3 는 각도판 — 원형 평균 각거리 / 원형 산포")
    for x, y in (("laptop_charger", "beam_projector"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc"), ("hotplate", "oven"),
                 ("hotplate", "electiric_kettle")):
        if x in iso and y in iso:
            A = np.array([r["dQ_dP"] for r in iso[x]]); B = np.array([r["dQ_dP"] for r in iso[y]])
            PA = np.array([r["dphi3"] for r in iso[x]]); PB = np.array([r["dphi3"] for r in iso[y]])
            print(f"    {KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                  f"  ΔQ1/ΔP {sepratio(A,B):6.2f}   Δφ3 {circ_sep(PA,PB):6.2f}")

    print()
    print("=" * 98)
    print("[B] 복합 실측 전이 + [C] 널")
    print("=" * 98)
    rows, nulls = composite_dq()
    print(f"  널 {len(nulls)}개:  |ΔP| 중앙 {np.median(np.abs(nulls[:,0])):7.2f}W"
          f"  p95 {np.percentile(np.abs(nulls[:,0]),95):8.2f}W   |"
          f"  **|ΔQ1| 중앙 {np.median(np.abs(nulls[:,1])):6.2f}var"
          f"  p95 {np.percentile(np.abs(nulls[:,1]),95):7.2f}var**")
    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(r["app"], []).append(r)
    print()
    print(f"  {'기기':<12s}{'n':>4s}{'|ΔP|':>8s}{'|ΔQ1|':>9s}"
          f"{'ΔQ1/ΔP':>20s}{'Δφ3 (도)':>24s}")
    print("  " + "-" * 80)
    for app in ALL:
        v = by.get(app)
        if not v or len(v) < 3:
            continue
        dp = np.abs([x["dP"] for x in v]); dq = np.abs([x["dQ"] for x in v])
        r = np.array([x["dQ_dP"] for x in v]); ph = np.array([x["dphi3"] for x in v])
        m, sd, R = circ(ph)
        print(f"  {KOR.get(app, app):<12s}{len(v):>4d}{np.median(dp):>7.1f}W"
              f"{np.median(dq):>8.2f}v{np.median(r):>13.3f} ±{r.std():<5.3f}"
              f"{m:>14.1f} ±{sd:<5.1f} R={R:.2f}")

    print()
    print("  분리비 (복합)")
    for x, y in (("laptop_charger", "beam_projector"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc"), ("hotplate", "oven")):
        if x in by and y in by and len(by[x]) > 2 and len(by[y]) > 2:
            A = np.array([r["dQ_dP"] for r in by[x]]); B = np.array([r["dQ_dP"] for r in by[y]])
            PA = np.array([r["dphi3"] for r in by[x]]); PB = np.array([r["dphi3"] for r in by[y]])
            sq, sp = sepratio(A, B), circ_sep(PA, PB)
            print(f"    {KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                  f"  ΔQ1/ΔP {sq:6.2f}{'  <<<' if sq>=2 else '     '}"
                  f"   Δφ3 {sp:6.2f}{'  <<<' if sp>=2 else ''}")

    print()
    print("  Δφ3 가 켜짐/꺼짐 방향에 불변인가 (같아야 정상)")
    for app in ALL:
        v = by.get(app)
        if not v:
            continue
        on = np.array([x["dphi3"] for x in v if x["sign"] > 0])
        off = np.array([x["dphi3"] for x in v if x["sign"] < 0])
        if len(on) < 2 or len(off) < 2:
            continue
        print(f"    {KOR.get(app,app):<12s} 켜짐 {circ(on)[0]:>7.1f}°(n={len(on)})"
              f"   꺼짐 {circ(off)[0]:>7.1f}°(n={len(off)})"
              f"   차 {abs((circ(on)[0]-circ(off)[0]+180)%360-180):5.1f}°")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"isolated": iso, "composite": rows,
                                       "null_p_q": nulls.tolist()},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
