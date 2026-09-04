# -*- coding: utf-8 -*-
"""원시 파형으로 회로 파라미터를 맞춘다 (12.185). 기준은 `results/_criteria_circuit.md` 추가분 [R1]~[R6].

    python -X utf8 -m src.run_raw_fit_probe                    # 전부 (약 3분)
    python -X utf8 -m src.run_raw_fit_probe --devices minipc
    python -X utf8 -m src.run_raw_fit_probe --no-loo           # 교차검증 생략 (빠르게)

단계
----
    1  자료 요약    동작점, 주기 간 산포, h16 이상 바닥, 범위혼합 여부 (재현 바닥 [R1])
    2  Cx 직접측정  비도통 구간 회귀 [R3]
    3  다점 적합    5파라미터 자유 + 가이드 파라미터 대조 [R2]
    4  Cx 고정      직접측정값에 묶고 4개만 [R4]
    5  LOO         동작점 교차검증 [R5]

⚠ 세 가지를 반드시 켠다 (`synthesis.fit_raw` 머리말): 전압 반파 대칭화, 계측 RC 순방향,
다점 동시. 하나라도 빼면 적합이 무너진다 (대칭화를 빼면 RMS 55~95%, 경계 고착).
"""
from typing import Dict, List
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import RAW_SNAPSHOT_FILES, raw_snapshots_of
from src.synthesis import fit_raw as fr
from src.synthesis.fit_raw import (PAR_NAMES, RawPoint, background_phasors, fit, load_raw, loo,
                                  measure_cx, rms_of)

OUT = "results/_circuit_raw_C.json"
OUT_V15 = "results/_circuit_raw_C_v15.json"
#: 가이드 2026-09 개정 §4.2 (원시 파이프라인 결과). 미니PC 는 v5(선로측 X-cap+NTC) 파라미터라
#: 우리 v3 시뮬에서는 재현되지 않는다 — 대조용으로만 적는다.
GUIDE_RAW = {"laptop_charger": (66.3e-6, 5.06, 988e-6, 0.164e-6, 0.43),
             "beam_projector": (53.3e-6, 7.18, 711e-6, 0.240e-6, 0.30),
             "minipc": (36.6e-6, 11.70, 2785e-6, 0.337e-6, 0.01)}
#: §4.3 검산 범위 (C_dc 는 µF/W)
RANGE43 = {"C_dc_per_W": (0.3, 2.0), "R": (1.0, 12.0), "L": (20e-6, 3000e-6),
           "Cx": (0.1e-6, 1.0e-6), "rd": (0.0, 5.0)}
#: 브리지 다이오드 2개의 동적 저항. r_d = n·V_T/I ~ 2×0.026/0.5 = 0.1Ω, 도통 피크에서 0.05Ω.
#: 실리콘 정류 다이오드의 물리 범위는 0.1~0.5Ω 이다 (가이드 §11.4). 미니PC 자유 적합이 준
#: 6.67Ω 은 R 과 맞바뀐 값이고(R 9.88->2.71, rd 1.63->6.67, 도통 중 합은 11.5->9.4 로 불변)
#: 둘은 도통 구간에서 직렬로만 보여 자료가 못 가른다 -> 하나를 물리값에 묶는다.
RD_PHYS = 0.3


def fmt(par5) -> str:
    C, R, L, Cx, rd = par5
    return (f"C_dc {C * 1e6:6.1f}µF  R {R:5.2f}Ω  L {L * 1e6:7.1f}µH  "
            f"Cx {Cx * 1e6:5.3f}µF  rd {rd:5.2f}Ω")


def check43(par5, p_max: float) -> List[str]:
    C, R, L, Cx, rd = par5
    bad = []
    lo, hi = RANGE43["C_dc_per_W"]
    if not lo <= C * 1e6 / p_max <= hi:
        bad.append(f"C_dc/P {C * 1e6 / p_max:.2f}µF/W")
    for name, v in (("R", R), ("L", L), ("Cx", Cx), ("rd", rd)):
        lo, hi = RANGE43[name]
        if not lo <= v <= hi:
            bad.append(f"{name} {v:.4g}")
    return bad


def starts_for(dev: str, cx: float) -> List[tuple]:
    """시작점: 가이드 값 + 직접측정 Cx 를 쓴 물리적 추정 + 흩뿌린 것 (고정 격자, 재현 가능)."""
    c = cx if np.isfinite(cx) and 0.02e-6 < cx < 2e-6 else 0.3e-6
    return [GUIDE_RAW[dev],
            (70e-6, 4.0, 700e-6, c, 0.5),
            (50e-6, 8.0, 1500e-6, c, 1.5),
            (110e-6, 2.0, 200e-6, c, 0.2),
            (35e-6, 12.0, 2500e-6, c, 3.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=list(RAW_SNAPSHOT_FILES))
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--band", type=int, default=fr.BAND)
    ap.add_argument("--no-match-power", action="store_true")
    ap.add_argument("--vband", type=int, default=None,
                    help="소스 전압을 이 차수까지로 자른다 (생성기 규약: 15)")
    ap.add_argument("--no-bg", action="store_true",
                    help="계측계 배경(noise_noselfpower_C, 7.3mA)을 빼지 않는다")
    a = ap.parse_args()
    mp = not a.no_match_power
    bg = None if a.no_bg else background_phasors()
    if bg is not None:
        print(f"계측계 배경 뺌: |I1| {1000 * abs(bg[0]):.1f}mA ∠{np.degrees(np.angle(bg[0])):+.1f}°"
              f"  |I3| {1000 * abs(bg[2]):.2f}mA  (noise_noselfpower_C)")
    rec: Dict = {"band": a.band, "match_power": mp, "subtract_meter_bg": bg is not None,
                 "vband": a.vband, "devices": {}}

    for dev in a.devices:
        stems = raw_snapshots_of(dev)
        if not stems:
            print(f"[{dev}] 원시 스냅샷 없음 — 건너뜀")
            continue
        print("=" * 92)
        print(f"[{dev}]  원시 스냅샷 {len(stems)}개")
        print("=" * 92)
        pts: List[RawPoint] = [load_raw(s, bg=bg, vband=a.vband) for s in stems]
        pts.sort(key=lambda p: p.p_w)
        d: Dict = {"points": {}}

        # ── 1 자료 요약 + 2 Cx 직접측정 ────────────────────────────────────
        print("  [1][2] 자료와 재현 바닥, Cx 직접측정")
        # 바닥: 적합 대상은 40주기 **중앙값** 파형이라 주기 산포는 √n 만큼 줄어든다.
        # h16 이상(oob)은 대역제한 손실에는 안 들어가고 전대역 RMS 에만 관계한다.
        print(f"    {'스냅샷':26s} {'P[W]':>7s} {'Irms':>7s} {'주기산포':>8s} {'중앙값바닥':>10s} "
              f"{'h16+':>7s} {'Cx[µF]':>8s} {'상관':>6s}")
        cxs = []
        for p in pts:
            cx, r, n = measure_cx(p)
            floor = float(p.scatter / np.sqrt(max(p.n_cyc, 1)))
            if np.isfinite(cx):
                cxs.append((cx, r))
            d["points"][p.stem] = {"p_w": p.p_w, "irms": p.irms, "scatter": p.scatter,
                                   "oob": p.oob, "floor": floor, "cx": cx, "cx_corr": r,
                                   "range_mixed": p.range_mixed}
            print(f"    {p.stem:26s} {p.p_w:7.2f} {p.irms:7.4f} {100 * p.scatter:7.1f}% "
                  f"{100 * floor:9.2f}% {100 * p.oob:6.1f}% {cx * 1e6:8.3f} {r:6.3f}"
                  + ("  범위혼합" if p.range_mixed else ""))
        # 상관 가중 중앙값 (신호가 약한 저전력 점을 덜 믿는다)
        good = [c for c, r in cxs if r > 0.5] or [c for c, _ in cxs]
        cx_med = float(np.median(good))
        d["cx_measured"] = cx_med
        print(f"    -> Cx 직접측정 중앙값 {cx_med * 1e6:.3f} µF (상관 0.5 초과 {len(good)}점)"
              f"   가이드 {GUIDE_RAW[dev][3] * 1e6:.3f} µF")

        # ── 3 다점 동시 적합 ───────────────────────────────────────────────
        print("\n  [3] 다점 동시 적합 (5파라미터 자유)")
        st = starts_for(dev, cx_med)
        t0 = time.time()
        r5 = fit(pts, st, band=a.band, match_power=mp)
        p_max = max(p.p_w for p in pts)
        bad = check43(r5.par5, p_max)
        d["fit_free"] = {"par5": list(r5.par5), "loss": r5.loss, "at_bound": r5.at_bound,
                         "out_of_range43": bad, "rms": {k: list(v) for k, v in r5.rms.items()}}
        print(f"    적합  {fmt(r5.par5)}     {time.time() - t0:.0f}s")
        print(f"          C_dc/P {r5.par5[0] * 1e6 / p_max:.2f}µF/W"
              + (f"   ⚠ 경계 {r5.at_bound}" if r5.at_bound else "")
              + (f"   ⚠ §4.3 밖 {bad}" if bad else "   §4.3 범위 안"))
        print(f"    가이드 {fmt(GUIDE_RAW[dev])}")
        gr = {p.stem: rms_of(GUIDE_RAW[dev], p, a.band, mp) for p in pts}
        d["fit_guide_rms"] = {k: list(v) for k, v in gr.items()}
        print(f"    {'스냅샷':26s} {'P[W]':>7s} {'적합 h15':>9s} {'전대역':>8s} "
              f"{'중앙값바닥':>10s} {'가이드 h15':>10s}")
        for p in pts:
            b15, bfull = r5.rms[p.stem]
            g15, _ = gr[p.stem]
            fl = d["points"][p.stem]["floor"]
            print(f"    {p.stem:26s} {p.p_w:7.2f} {100 * b15:8.1f}% {100 * bfull:7.1f}% "
                  f"{100 * fl:9.2f}% {100 * g15:9.1f}%"
                  + ("   범위혼합" if p.range_mixed else ""))

        # ── 4 Cx 고정 ──────────────────────────────────────────────────────
        print("\n  [4] Cx 를 직접측정에 고정하고 4파라미터")
        r4 = fit(pts, st, band=a.band, match_power=mp, fixed={3: cx_med})
        d["fit_fixcx"] = {"par5": list(r4.par5), "loss": r4.loss, "at_bound": r4.at_bound,
                          "rms": {k: list(v) for k, v in r4.rms.items()}}
        dl = 100 * (np.sqrt(r4.loss / r5.loss) - 1)
        print(f"    고정  {fmt(r4.par5)}")
        print(f"    훈련 RMS {100 * np.sqrt(r4.loss):.1f}% vs 자유 {100 * np.sqrt(r5.loss):.1f}%  ({dl:+.1f}%)"
              + (f"   ⚠ 경계 {r4.at_bound}" if r4.at_bound else ""))

        # ── 4b rd 물리값 고정 (가이드 §11.4) ────────────────────────────────
        print(f"\n  [4b] rd 를 물리값 {RD_PHYS}Ω 에 고정 — R–rd 축퇴를 끊는다")
        r_rd = fit(pts, st, band=a.band, match_power=mp, fixed={4: RD_PHYS})
        r_both = fit(pts, st, band=a.band, match_power=mp, fixed={3: cx_med, 4: RD_PHYS})
        d["fit_fixrd"] = {"par5": list(r_rd.par5), "loss": r_rd.loss, "at_bound": r_rd.at_bound,
                          "rms": {k: list(v) for k, v in r_rd.rms.items()}}
        d["fit_fixboth"] = {"par5": list(r_both.par5), "loss": r_both.loss,
                            "at_bound": r_both.at_bound,
                            "rms": {k: list(v) for k, v in r_both.rms.items()}}
        for lab, rr in (("rd 고정", r_rd), ("Cx+rd 고정", r_both)):
            print(f"    {lab:10s} {fmt(rr.par5)}")
            print(f"    {'':10s} 훈련 RMS {100 * np.sqrt(rr.loss):.1f}% "
                  f"(자유 {100 * np.sqrt(r5.loss):.1f}%, {100 * (np.sqrt(rr.loss / r5.loss) - 1):+.1f}%)"
                  + (f"   경계 {rr.at_bound}" if rr.at_bound else ""))

        # ── 5 LOO ──────────────────────────────────────────────────────────
        if not a.no_loo and len(pts) >= 3:
            print("\n  [5] LOO 교차검증 (동작점 하나를 빼고 맞춘 뒤 그 점을 예측)")
            t0 = time.time()
            m5, per5 = loo(pts, st, band=a.band, match_power=mp)
            m4, per4 = loo(pts, st, band=a.band, match_power=mp, fixed={3: cx_med})
            mr, perr = loo(pts, st, band=a.band, match_power=mp, fixed={4: RD_PHYS})
            mb, perb = loo(pts, st, band=a.band, match_power=mp, fixed={3: cx_med, 4: RD_PHYS})
            d["loo"] = {"free": {"mean": m5, "per": per5}, "fixcx": {"mean": m4, "per": per4},
                        "fixrd": {"mean": mr, "per": perr}, "fixboth": {"mean": mb, "per": perb}}
            print(f"    {'스냅샷':26s} {'훈련(자유)':>11s} {'자유':>8s} {'Cx고정':>8s} "
                  f"{'rd고정':>8s} {'둘다':>8s}   <- LOO")
            for p in pts:
                print(f"    {p.stem:26s} {100 * r5.rms[p.stem][0]:10.1f}% "
                      f"{100 * per5[p.stem]:7.1f}% {100 * per4[p.stem]:7.1f}% "
                      f"{100 * perr[p.stem]:7.1f}% {100 * perb[p.stem]:7.1f}%")
            best = min((m5, "자유"), (m4, "Cx고정"), (mr, "rd고정"), (mb, "둘다"))
            print(f"    평균 LOO  자유 {100 * m5:.1f}%  Cx고정 {100 * m4:.1f}%  "
                  f"rd고정 {100 * mr:.1f}%  둘다 {100 * mb:.1f}%   -> {best[1]}   {time.time() - t0:.0f}s")
        rec["devices"][dev] = d
        print()

    out = OUT_V15 if a.vband else OUT
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=float)
    print(f"기록: {OUT}")


if __name__ == "__main__":
    main()
