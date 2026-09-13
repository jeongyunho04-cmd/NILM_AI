"""
정규화 고조파 지문의 산포 — 지터 크기를 측정으로 정한다 (설계 문서 12.62절)
================================================================================
`DataAugmentor` 가 하는 증강 네 가지 중 **어느 것도 지문을 바꾸지 못한다.**

    공통 전력 배율 (±5%)   ->  |I_k|/|I_1| 을 **정확히** 불변으로 남긴다
    k차 위상 회전 (±4°)    ->  angle(I_k) − k·angle(I_1) 을 **정확히** 불변으로 남긴다
                              (k·θ − k·θ = 0. 순수한 시간 이동이므로 당연하다)
    시간 신축 / 듀티 재타이밍  ->  파형 길이만 바꾼다
    돌입 접점각             ->  첫 2주기만

그래서 합성 창에 들어가는 **정규화된 15차 복소 지문은 원본 녹화의 바이트 그대로의
복사본**이다. 12.58 이 "합성은 어떤 조건에서도 프로젝터↔충전기를 푼다
(F1 0.985~1.000)" 를 발견한 것과 맞물린다 — 지문 벡터가 완벽히 보존되면 그것은
판별이 아니라 **조회**다.

처방은 차수별 독립 지터다 (활성화당 1회, k>=2 만):

    I_k *= (1 + g_k) · exp(i·ψ_k)

**이 스크립트는 g_k · ψ_k 를 얼마로 둘지 측정으로 정하기 위한 것이다.**

[무엇을 재는가]
증강이 불변으로 남기는 그 두 양의 산포를 세 층으로 잰다.

    A. 녹화 내부   같은 녹화 안 활성화 블록끼리       <- 하한. 상태 변동 + 계측 잡음
    B. 녹화 간     녹화별 중심끼리 (기기당 2~3개)     <- 개체·세션 차이. **지터의 표적**
    C. 실측 복합   test_5/6/7 전이 지문 vs 격리 중심   <- 실제로 건너야 하는 격차

지터를 B 에 맞추면 "같은 기종의 다른 개체/다른 세션" 을 흉내내는 것이고,
C 까지 덮으면 실측 도메인을 덮는 것이다. C 는 다른 기기의 동시 움직임과 계측
잡음이 섞여 있어 상한으로 읽어야 한다 (12.53.7 의 널 표와 같은 성질이다).

[왜 k>=2 만 흔드는가]
기본파와 P·Q 를 건드리지 않으면 **전력 라벨이 정확히 유효하게 남는다.**
고조파는 P 에 거의 기여하지 않으므로 물리적으로도 맞다.
그리고 위 두 양의 정의가 1차를 기준으로 하므로, k>=2 만 흔들면
g_k · ψ_k 가 산포에 **1:1 로** 대응한다 — 측정값을 그대로 인수로 쓸 수 있다.

    python -m src.run_fingerprint_spread_probe
    python -m src.run_fingerprint_spread_probe --json results/fp_spread.json
"""
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.run_delta_feature_probe import (DEV_DIR, DEV_FILES, SMPS, composite,
                                         load_cplx, transitions)
from src.model.realdata import DEFAULT_DIR
from src.evaluation.real_events import load_events
from src.run_live import KOR

# 표에 찍을 차수. 2차는 충전기 판별(12.53 ①), 3·5·7 은 SMPS 돌입의 주 성분,
# 9·13 은 12.53 ② 가 쓰는 고차다.
ORDERS = (2, 3, 5, 7, 9, 13)


def signature(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """복소 지문 -> (증강 불변량 2개).

    `a_k = |v_k| / |v_1|`                      공통 배율에 불변
    `p_k = angle(v_k) − k·angle(v_1)`          시간 이동에 불변

    둘 다 k = 2..15 이고 길이 14 다. 차수별 지터 `(1+g_k)·exp(i·ψ_k)` 는
    `a_k` 를 `(1+g_k)` 배 하고 `p_k` 에 `ψ_k` 를 더한다 — 정확히 1:1 이다.
    """
    v = np.asarray(v)
    m1 = max(abs(v[0]), 1e-12)
    ph1 = np.angle(v[0])
    k = np.arange(2, len(v) + 1)
    a = np.abs(v[1:]) / m1
    p = np.angle(v[1:]) - k * ph1
    return a, (p + np.pi) % (2 * np.pi) - np.pi


def rel_spread(a: np.ndarray) -> np.ndarray:
    """진폭비 표본 (n, 14) -> 차수별 상대 산포 (%).

    `g_k` 가 곱셈 인수이므로 로그에서 재야 한다. 그 표준편차가 곧 g_k 의 폭이다.
    """
    a = np.asarray(a, np.float64)
    if len(a) < 2:
        return np.full(a.shape[1], np.nan)
    return 100.0 * np.std(np.log(np.maximum(a, 1e-12)), axis=0, ddof=1)


def circ_spread(p: np.ndarray) -> np.ndarray:
    """위상 표본 (n, 14) -> 차수별 원형 표준편차 (도).

    위상은 ±π 에서 감기므로 산술 표준편차를 쓰면 안 된다 (12.50 의 교훈과 같은
    종류의 실수다). `sqrt(−2 ln R)` 이 원형 표준편차다.
    """
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return np.full(p.shape[1], np.nan)
    R = np.abs(np.mean(np.exp(1j * p), axis=0))
    return np.degrees(np.sqrt(np.maximum(-2.0 * np.log(np.maximum(R, 1e-12)), 0.0)))


def circ_mean(p: np.ndarray) -> np.ndarray:
    return np.angle(np.mean(np.exp(1j * np.asarray(p, np.float64)), axis=0))


def isolated_vectors(min_blk: int = 200, blk_cycles: int = 600) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """격리 녹화 -> {가전: {녹화: [복소 지문, ...]}}.

    `run_delta_feature_probe.isolated` 과 같은 방식으로 ON 구간에서 OFF 배경을
    복소로 빼되, 비율 특징 대신 **복소 벡터 자체**를 남긴다.
    """
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for app, files in DEV_FILES.items():
        for stem in files:
            f = DEV_DIR / f"{stem}.npz"
            if not f.exists():
                continue
            I, pf = load_cplx(f)
            p = pf[:, 0]
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            off = p < np.percentile(p[~on], 60) if (~on).sum() > 100 else ~on
            if on.sum() < blk_cycles // 2 or off.sum() < 100:
                continue
            bg = np.median(I[off].real, 0) + 1j * np.median(I[off].imag, 0)
            idx = np.flatnonzero(on)
            for blk in np.array_split(idx, max(1, len(idx) // blk_cycles)):
                if len(blk) < min_blk:
                    continue
                v = (np.median(I[blk].real, 0) + 1j * np.median(I[blk].imag, 0)) - bg
                if abs(v[0]) < 1e-4:
                    continue
                out.setdefault(app, {}).setdefault(stem, []).append(v)
    return out


def composite_vectors(half: float = 5.0, guard: float = 1.0, snap: float = 6.0
                      ) -> Dict[str, List[np.ndarray]]:
    """복합 실측의 라벨된 SMPS 전이 -> {가전: [복소 차분 벡터, ...]}.

    `run_delta_feature_probe.composite` 과 같은 스냅·창 규칙을 쓴다.
    OFF 전이는 부호를 뒤집어 ON 방향으로 맞춘다 (지문은 방향에 무관하다).
    """
    ev = load_events()
    out: Dict[str, List[np.ndarray]] = {}
    for stem in ("test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        I, pf = load_cplx(f)
        p = pf[:, 0]
        t = np.arange(len(p)) / 60.0
        i3 = np.abs(I[:, 2])
        for t0, app, sign in transitions(ev, stem, SMPS):
            if t0 < half + snap + 1 or t0 > t[-1] - half - snap - 1:
                continue
            best, bd = t0, -1.0
            for c in np.arange(t0 - snap, t0 + snap, 0.25):
                pre = (t >= c - half) & (t <= c - guard)
                post = (t >= c + guard) & (t <= c + half)
                if pre.sum() < 30 or post.sum() < 30:
                    continue
                d = abs(np.median(i3[post]) - np.median(i3[pre]))
                if d > bd:
                    best, bd = c, d
            pre = (t >= best - half) & (t <= best - guard)
            post = (t >= best + guard) & (t <= best + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            dv = ((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                  - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0)))
            dp = float(np.median(p[post]) - np.median(p[pre]))
            if abs(dv[0]) < 1e-3 or abs(dp) < 5:
                continue
            out.setdefault(app, []).append(dv * (1 if sign > 0 else -1))
    return out


def _fmt(row: np.ndarray, cols: Tuple[int, ...], width: int = 8) -> str:
    idx = [c - 2 for c in cols]
    return "".join("     — " if not np.isfinite(row[i]) else f"{row[i]:{width}.1f}"
                   for i in idx)


def main() -> int:
    ap = argparse.ArgumentParser(description="정규화 고조파 지문의 산포 측정")
    ap.add_argument("--apps", default=",".join(SMPS),
                    help="쉼표로 구분. 기본은 SMPS 3종 (남은 실패가 그것이다)")
    ap.add_argument("--all-apps", action="store_true", help="풀에 있는 기기 전부")
    ap.add_argument("--json", default=None, help="표를 JSON 으로도 저장한다")
    a = ap.parse_args()

    iso = isolated_vectors()
    comp = composite_vectors()
    apps = list(iso) if a.all_apps else [x for x in a.apps.split(",") if x in iso]

    hdr = "".join(f"{'k=' + str(k):>8s}" for k in ORDERS)
    print("\n" + "=" * 78)
    print("[지문 산포] 증강이 불변으로 남기는 두 양의 산포")
    print("=" * 78)
    print("  A 녹화 내부 = 같은 녹화 안 블록끼리 (하한)")
    print("  B 녹화 간   = 녹화별 중심끼리      <- 지터의 표적")
    print("  C 실측 복합 = test_5/6/7 전이 지문 vs 격리 중심 (상한, 타 기기 혼입)")

    result: Dict[str, dict] = {}
    for app in apps:
        recs = iso[app]
        per_rec = {r: [signature(v) for v in vs] for r, vs in recs.items()}
        n_blk = {r: len(vs) for r, vs in recs.items()}

        # A. 녹화 내부 — 녹화별로 재고 블록 수로 가중평균한다
        A_a, A_p, wts = [], [], []
        for r, sig in per_rec.items():
            if len(sig) < 2:
                continue
            A_a.append(rel_spread(np.array([s[0] for s in sig])))
            A_p.append(circ_spread(np.array([s[1] for s in sig])))
            wts.append(len(sig))
        if wts:
            w = np.asarray(wts, np.float64) / sum(wts)
            A_amp = np.nansum(np.array(A_a) * w[:, None], axis=0)
            A_ph = np.nansum(np.array(A_p) * w[:, None], axis=0)
        else:
            A_amp = A_ph = np.full(14, np.nan)

        # B. 녹화 간 — 녹화별 중심 (진폭은 기하평균, 위상은 원형평균)
        cen_a, cen_p = [], []
        for r, sig in per_rec.items():
            cen_a.append(np.exp(np.mean(np.log(np.maximum(
                np.array([s[0] for s in sig]), 1e-12)), axis=0)))
            cen_p.append(circ_mean(np.array([s[1] for s in sig])))
        B_amp = rel_spread(np.array(cen_a))
        B_ph = circ_spread(np.array(cen_p))

        # C. 실측 복합 — 격리 전체 중심에서의 편차
        all_a = np.array([s[0] for sig in per_rec.values() for s in sig])
        all_p = np.array([s[1] for sig in per_rec.values() for s in sig])
        ref_a = np.exp(np.mean(np.log(np.maximum(all_a, 1e-12)), axis=0))
        ref_p = circ_mean(all_p)
        cv = comp.get(app, [])
        if len(cv) >= 2:
            cs = [signature(v) for v in cv]
            dev_a = np.array([s[0] for s in cs]) / np.maximum(ref_a, 1e-12)
            dev_p = np.array([s[1] for s in cs]) - ref_p
            # **중앙값으로 읽는다.** RMS 로 재면 전이 하나가 10배 어긋나도
            # log 가 2.3 을 기여해 표를 지배한다 (n=11~16 이라 더욱 그렇다).
            # 6절 1번 규칙 — 비율을 보고하기 전에 표본 수와 이상치 민감도를 본다.
            C_amp = 100.0 * np.median(np.abs(np.log(np.maximum(dev_a, 1e-12))), axis=0)
            dw = (dev_p + np.pi) % (2 * np.pi) - np.pi
            C_ph = np.degrees(np.median(np.abs(dw), axis=0))
        else:
            C_amp = C_ph = np.full(14, np.nan)

        recs_txt = ", ".join(f"{r}({n})" for r, n in sorted(n_blk.items()))
        print(f"\n  {KOR.get(app, app)}  ({app})")
        print(f"    격리 블록 {sum(n_blk.values())}개 / 녹화 {len(n_blk)}개 — {recs_txt}"
              f" | 복합 전이 {len(cv)}개")
        print(f"    {'진폭비 수준 |I_k|/|I_1| (%)':38s}{hdr}")
        print(f"      {'격리 중심':34s}{_fmt(100.0 * ref_a, ORDERS)}")
        print(f"    {'진폭비 상대산포 (%)':38s}{hdr}")
        print(f"      {'A 녹화 내부':34s}{_fmt(A_amp, ORDERS)}")
        print(f"      {'B 녹화 간':34s}{_fmt(B_amp, ORDERS)}")
        print(f"      {'C 실측 복합':34s}{_fmt(C_amp, ORDERS)}")
        print(f"    {'상대위상 ∠I_k − k∠I_1 산포 (도)':38s}{hdr}")
        print(f"      {'A 녹화 내부':34s}{_fmt(A_ph, ORDERS)}")
        print(f"      {'B 녹화 간':34s}{_fmt(B_ph, ORDERS)}")
        print(f"      {'C 실측 복합':34s}{_fmt(C_ph, ORDERS)}")

        result[app] = {
            "n_blocks": n_blk, "n_composite": len(cv),
            "orders": list(range(2, 16)),
            "level_pct": (100.0 * ref_a).tolist(),
            "amp_pct": {"within": A_amp.tolist(), "between": B_amp.tolist(),
                        "composite": C_amp.tolist()},
            "phase_deg": {"within": A_ph.tolist(), "between": B_ph.tolist(),
                          "composite": C_ph.tolist()},
        }

    # 지터 폭 후보 — 차수 2~15 전체의 중앙값으로 요약한다
    print("\n" + "-" * 78)
    print("  지터 폭 후보 (차수 2~15 중앙값. 인수로 쓸 값이다)")
    print("-" * 78)
    ODD = [k - 2 for k in range(3, 16, 2)]     # 3,5,7,9,11,13,15
    EVEN = [k - 2 for k in range(2, 16, 2)]    # 2,4,...,14
    print(f"    {'':16s}{'홀수 g (%)':>12s}{'홀수 ψ (도)':>13s}"
          f"{'짝수 g (%)':>12s}{'짝수 ψ (도)':>13s}")
    for lvl, key in (("A 녹화 내부", "within"), ("B 녹화 간", "between"),
                     ("C 실측 복합", "composite")):
        cells = []
        for sel in (ODD, EVEN):
            am = np.nanmedian([np.nanmedian(np.asarray(v["amp_pct"][key])[sel])
                               for v in result.values()])
            ph = np.nanmedian([np.nanmedian(np.asarray(v["phase_deg"][key])[sel])
                               for v in result.values()])
            cells += [am, ph]
        print(f"    {lvl:16s}{cells[0]:>12.1f}{cells[1]:>13.1f}"
              f"{cells[2]:>12.1f}{cells[3]:>13.1f}")
    print("-" * 78)
    print("  ⚠ 짝수 차수는 계측 바닥이다 — 대칭 비선형 부하는 홀수만 흘린다.")
    print("    수준 줄에서 짝수 차수의 |I_k|/|I_1| 이 얼마나 작은지 확인할 것.")
    print("    **지터 크기는 홀수 열로 정한다.**")
    print("  ⚠ B 는 기기당 녹화가 2~3개다 (n=2 면 산포가 하한이다 — 12.39.6 과 같은 성질).")
    print("    C 는 타 기기의 동시 움직임과 계측 잡음이 섞여 상한이다 — 12.53.7 의")
    print("    복합 널(|ΔI3| p95 58.6mA)과 대조해 읽을 것.")
    print("=" * 78 + "\n")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        print(f"  -> {a.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
