# -*- coding: utf-8 -*-
"""계측 RC 가 1극이 맞는가 (기준 `results/_criteria_circuit.md` [M1]~[M6]).

    python -X utf8 -m src.run_meas_rc_probe               # 전부 (LOO 포함)
    python -X utf8 -m src.run_meas_rc_probe --no-loo      # 훈련·조합만 (빠르게)
    python -X utf8 -m src.run_meas_rc_probe --scan-fc     # fc 훑기 추가

②의 남은 잔차(충전기 19W LOO 9.2%, 미니PC 비도통 5.5%)에 회로 요소는 넷 다 기각됐다
(12.185.23). 남은 후보 중 **계측 모형**을 본다.

지금 순방향 모델은 전압·전류 양쪽에 같은 1극 `1/(1+jf/1591.55)` 을 쓴다. 그런데 12.184.16 이
LOW 전류 경로는 HIGH 의 ADC 노드 뒤에서 한 단 더 증폭된다고 밝혔다 (60Hz 에서 2.18° = 극 하나).
그러면 **전류 채널의 극이 전압보다 하나 많아야** 한다.

검정력이 높은 이유: 계측 모형은 기기가 아니라 계측기의 성질이라 **세 기기가 공유**해야 하고
([M1]), 극 개수는 정수라 **자유 파라미터가 0** 이다. 순수 지연은 우리 관례에서 상쇄되므로
(v·i 양쪽에 같이 걸리고 모델이 시불변) 여기 걸리는 것은 **모양** 오차뿐이다.

단계
----
    1  회귀      기본 변형이 정본 손실을 재현하는가
    2  재적합    변형마다 기기별 par5 를 다시 맞춘다 ([M2]) — 훈련 손실
    3  LOO       충전기·미니PC ([M3])
    4  조합      유보 자료 6개 ([M3])
    5  차수별    개선이 h9~h15 에 몰리는가 ([M5])
    6  판정      [M1] 공유성 · [M4] 물리
"""
from typing import Dict, List, Tuple
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import (RAW_COMBO_FILES, RAW_COMBO_SOLO,
                                             RAW_SNAPSHOT_FILES, raw_snapshots_of)
from src.synthesis import fit_raw as fr
from src.synthesis.fit_raw import (BAND, background_phasors, fit, load_raw, loo_folds,
                                   phasors, rms_of, sim_current)

IN = "results/_circuit_raw_C.json"
OUT = "results/_meas_rc.json"
RD_PHYS = 0.3
FIX = {4: RD_PHYS}
DEVS = ("laptop_charger", "beam_projector", "minipc")
#: 계측 모형 변형. 값은 `load_raw` 인수. 극 개수는 정수라 자유 파라미터가 0이다.
VARIANTS: Dict[str, dict] = {
    "M0 v1/i1": {},                                   # 지금 (정본)
    "M1 v1/i2": {"i_order": 2},                       # [M5] 사전 예상 — 전류에 극 하나 더
    "M2 v2/i1": {"v_order": 2},
    "M3 v2/i2": {"v_order": 2, "i_order": 2},
    "M6 bg_rc": {"bg_rc": True},                      # 배경 규약만 맞춤 ([M6])
    "M6+M1":    {"bg_rc": True, "i_order": 2},
}
GUIDE_RAW = {"laptop_charger": (66.3e-6, 5.06, 988e-6, 0.164e-6, 0.43),
             "beam_projector": (53.3e-6, 7.18, 711e-6, 0.240e-6, 0.30),
             "minipc": (36.6e-6, 11.70, 2785e-6, 0.337e-6, 0.01)}


def starts_for(dev: str, cx: float) -> List[tuple]:
    c = cx if np.isfinite(cx) and 0.02e-6 < cx < 2e-6 else 0.3e-6
    return [GUIDE_RAW[dev], (70e-6, 4.0, 700e-6, c, 0.5), (50e-6, 8.0, 1500e-6, c, 1.5),
            (110e-6, 2.0, 200e-6, c, 0.2), (35e-6, 12.0, 2500e-6, c, 3.0)]


def fmt5(p) -> str:
    return (f"C {p[0] * 1e6:5.1f}µF R {p[1]:5.2f}Ω L {p[2] * 1e6:6.1f}µH Cx {p[3] * 1e6:5.3f}µF")


def rel(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.sum(np.abs(d) ** 2)) / np.sqrt(np.sum(np.abs(b) ** 2)))


def combo_rms(par: Dict[str, tuple], kw: dict, bg, band: int) -> Dict[str, float]:
    """유보 조합 6개를 이 계측 모형·이 par5 로 재현."""
    out = {}
    for combo, devs in RAW_COMBO_FILES.items():
        parts = [(d, RAW_COMBO_SOLO[combo][d]) for d in devs if d in par]
        if len(parts) < 2:
            continue
        c = load_raw(combo, bg=bg, **kw)
        solos = {d: load_raw(s, bg=bg, **kw) for d, s in parts}
        pw = {d: solos[d].p_w for d, _ in parts}
        if "laptop_charger" in pw:
            pw["laptop_charger"] = c.p_w - sum(v for k, v in pw.items() if k != "laptop_charger")
        tot, ok = np.zeros_like(c.i), True
        for d, _ in parts:
            pt = type(c)(c.stem, c.v, c.i, pw[d], c.vsrc, c.irms, c.n_cyc,
                         c.scatter, c.oob, c.range_mixed, c.i_fc, c.i_order)
            x = sim_current(par[d], pt, match_power=True)
            if x is None:
                ok = False
                break
            tot = tot + x
        out[combo] = rel(phasors(tot, band), phasors(c.i, band)) if ok else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--scan-fc", action="store_true")
    ap.add_argument("--band", type=int, default=fr.BAND)
    a = ap.parse_args()
    canon = json.load(open(IN, encoding="utf-8"))["devices"]
    bg = background_phasors()
    rec: Dict = {"variants": {}}

    # ── 1 회귀 ─────────────────────────────────────────────────────────────
    print("=" * 104)
    print("[1] 회귀 — 기본 변형(M0)이 정본 손실을 재현하는가")
    print("=" * 104)
    for dev in DEVS:
        pts = sorted([load_raw(s, bg=bg) for s in raw_snapshots_of(dev)], key=lambda p: p.p_w)
        L = fr.loss_at(tuple(canon[dev]["fit_fixrd"]["par5"]), {}, pts, a.band)
        r = canon[dev]["fit_fixrd"]["loss"]
        print(f"  {dev:16s} {r:.8f} -> {L:.8f}  ({100 * (L / r - 1):+.4f}%)")

    # ── 2 변형마다 재적합 ──────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("[2][M2] 변형마다 par5 를 다시 맞춘다 — 훈련 RMS")
    print("=" * 104)
    print(f"  {'변형':12s}" + "".join(f"{d[:14]:>16s}" for d in DEVS))
    fits: Dict[str, Dict[str, object]] = {}
    ptsv: Dict[str, Dict[str, list]] = {}
    for name, kw in VARIANTS.items():
        t0 = time.time()
        row, fv, pv = [], {}, {}
        for dev in DEVS:
            pts = sorted([load_raw(s, bg=bg, **kw) for s in raw_snapshots_of(dev)],
                         key=lambda p: p.p_w)
            r = fit(pts, starts_for(dev, canon[dev].get("cx_measured", np.nan)),
                    band=a.band, fixed=FIX)
            fv[dev], pv[dev] = r, pts
            row.append(100 * np.sqrt(r.loss))
        fits[name], ptsv[name] = fv, pv
        print(f"  {name:12s}" + "".join(f"{v:15.2f}%" for v in row) + f"   {time.time() - t0:.0f}s")
    base = {d: 100 * np.sqrt(fits["M0 v1/i1"][d].loss) for d in DEVS}
    print(f"  {'M0 대비':12s}" + "".join(f"{'':>16s}" for _ in DEVS))
    for name in VARIANTS:
        if name == "M0 v1/i1":
            continue
        print(f"  {name:12s}" + "".join(
            f"{100 * (np.sqrt(fits[name][d].loss / fits['M0 v1/i1'][d].loss) - 1):+15.1f}%"
            for d in DEVS))
    print(f"  par5 (M0 / 최선 변형):")
    for dev in DEVS:
        best = min(VARIANTS, key=lambda n: fits[n][dev].loss)
        print(f"    {dev:16s} M0  {fmt5(fits['M0 v1/i1'][dev].par5)}")
        print(f"    {'':16s} {best:9s} {fmt5(fits[best][dev].par5)}")

    # ── 4 조합 (유보) ──────────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("[4][M3] 유보 자료 — 조합 스냅샷 6개")
    print("=" * 104)
    print(f"  {'변형':12s} {'평균':>8s}   " + "  ".join(f"{c[4:18]:>14s}" for c in RAW_COMBO_FILES))
    combos = {}
    for name, kw in VARIANTS.items():
        par = {d: fits[name][d].par5 for d in DEVS}
        cr = combo_rms(par, kw, bg, a.band)
        combos[name] = cr
        m = float(np.nanmean(list(cr.values())))
        print(f"  {name:12s} {100 * m:7.2f}%   "
              + "  ".join(f"{100 * cr[c]:13.2f}%" for c in RAW_COMBO_FILES))

    # ── 5 차수별 ([M5]) ────────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("[5][M5] 개선이 어느 차수에 있는가 — 차수별 잔차 [실측 대비 %]")
    print("=" * 104)
    ODD = [1, 3, 5, 7, 9, 11, 13, 15]
    best_alt = min((n for n in VARIANTS if n != "M0 v1/i1"),
                   key=lambda n: sum(fits[n][d].loss for d in DEVS))
    print(f"  최선 대안 = {best_alt}")
    for dev in DEVS:
        for name in ("M0 v1/i1", best_alt):
            pts, par = ptsv[name][dev], fits[name][dev].par5
            acc = np.zeros(BAND)
            for p in pts:
                x = sim_current(par, p, match_power=True)
                if x is None:
                    continue
                acc += np.abs(phasors(x, BAND) - phasors(p.i, BAND)) / max(
                    np.sqrt(np.mean(np.abs(phasors(p.i, BAND)) ** 2)), 1e-12)
            acc /= len(pts)
            print(f"  {dev[:14]:14s} {name:10s}" + "".join(f"{100 * acc[h - 1]:8.2f}" for h in ODD))

    # ── 3 LOO ──────────────────────────────────────────────────────────────
    if not a.no_loo:
        print()
        print("=" * 104)
        print("[3][M3] LOO — 충전기·미니PC")
        print("=" * 104)
        print(f"  {'변형':12s} {'충전기':>9s} {'미니PC':>9s}   점별(충전기)")
        for name, kw in VARIANTS.items():
            t0, row, det = time.time(), [], ""
            for dev in ("laptop_charger", "minipc"):
                folds = loo_folds(ptsv[name][dev],
                                  starts_for(dev, canon[dev].get("cx_measured", np.nan)),
                                  band=a.band, fixed=FIX)
                per = {p.stem: rms_of(b.par5, p, a.band)[0] for p, _, b in folds}
                row.append(100 * float(np.mean(list(per.values()))))
                if dev == "laptop_charger":
                    det = "  ".join(f"{100 * v:.1f}" for v in per.values())
            rec["variants"].setdefault(name, {})["loo"] = row
            print(f"  {name:12s} {row[0]:8.2f}% {row[1]:8.2f}%   {det}   {time.time() - t0:.0f}s")

    # ── 6 fc 훑기 ([M4]) ───────────────────────────────────────────────────
    if a.scan_fc:
        print()
        print("=" * 104)
        print("[6][M4] fc 훑기 (1극/1극) — 알려진 값 1591.55Hz 에서 얼마나 벗어나나")
        print("=" * 104)
        print(f"  {'fc[Hz]':>8s}" + "".join(f"{d[:14]:>16s}" for d in DEVS))
        for fc in (900, 1200, 1591.55, 2000, 2600, 3500, 5000):
            row = []
            for dev in DEVS:
                pts = sorted([load_raw(s, bg=bg, v_fc=fc, i_fc=fc)
                              for s in raw_snapshots_of(dev)], key=lambda p: p.p_w)
                r = fit(pts, starts_for(dev, canon[dev].get("cx_measured", np.nan)),
                        band=a.band, fixed=FIX)
                row.append(100 * np.sqrt(r.loss))
            print(f"  {fc:8.0f}" + "".join(f"{v:15.2f}%" for v in row))

    # ── 판정 ───────────────────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("[판정] [M1] 공유성 — 세 기기가 **모두** 좋아지는 변형만 채택")
    print("=" * 104)
    for name in VARIANTS:
        if name == "M0 v1/i1":
            continue
        g = [np.sqrt(fits[name][d].loss / fits["M0 v1/i1"][d].loss) - 1 for d in DEVS]
        cm = np.nanmean(list(combos[name].values())) / np.nanmean(list(combos["M0 v1/i1"].values())) - 1
        allbetter = all(x < 0 for x in g)
        print(f"  {name:12s} 훈련 " + " ".join(f"{100 * x:+6.1f}%" for x in g)
              + f"   조합 {100 * cm:+6.1f}%   "
              + ("세 기기 모두 개선" if allbetter else "일부만/전부 악화 -> [M1] 기각"))

    for name in VARIANTS:
        rec["variants"].setdefault(name, {}).update({
            "kwargs": VARIANTS[name],
            "train": {d: float(fits[name][d].loss) for d in DEVS},
            "par5": {d: list(fits[name][d].par5) for d in DEVS},
            "combo": {k: float(v) for k, v in combos[name].items()}})
    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
