# -*- coding: utf-8 -*-
"""SMPS 결합을 생성기에 (①b, 기준 `results/_criteria_circuit.md` [B1]~[B5]).

    python -X utf8 -m src.run_coupling_probe

`synthesis.coupling.CouplingModel` 이 내는 ΔI 를 다섯 가지로 검정한다.

    1 [B1] 재현      ΔI 를 단독 **실측**에 더해 조합 스냅샷 6개를 재현. 12.185.12 [B] 와 견준다
    2 [B2] 소스      원시 전압 18개로 ΔI 를 각각 계산해 산포 (캐시를 소스 하나로 해도 되는가)
    3 [B3] Z         Z 0.5~2.0Ω · L 0~400µH 에서 ΔI 가 얼마나 움직이는가 (= 다양성의 크기)
    4 [B4] 비용      캐시 적중률과 창당 시간
    5 [B5] 안전      단독이면 0 · 저항만이면 항등 · 기본파를 5% 넘게 안 바꾼다
"""
from typing import Dict, List
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import (RAW_COMBO_FILES, RAW_COMBO_SOLO,
                                             RAW_SNAPSHOT_FILES, raw_snapshots_of)
from src.synthesis import coupling as CP
from src.synthesis import fcm
from src.synthesis.fit_raw import BAND, background_phasors, load_raw, phasors

OUT = "results/_coupling.json"
Z_SITE_C = 0.520          #: 장소 C 선로 저항 — 포트 계단 78개 (12.185.14)
ODD = [1, 3, 5, 7, 9, 11, 13, 15]


def rel(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.sum(np.abs(d) ** 2)) / np.sqrt(np.sum(np.abs(b) ** 2)))


def main() -> None:
    bg = background_phasors()
    cm = CP.CouplingModel()
    rec: Dict = {}

    # ── 1 [B1] 재현 ────────────────────────────────────────────────────────
    print("=" * 100)
    print("[1][B1] ΔI 를 단독 실측에 더해 조합 6개를 재현 (Z = 0.520Ω, 장소 C 계단)")
    print("=" * 100)
    print("  ⚠ 전력이 재배분되는 조합(충전기가 든 넷)은 배선 검사가 안 된다 — 전력·텍스처·결합이")
    print("     섞인다. `무배분` 표시가 붙은 둘(raw_beam_minipc_1/2)만 판정에 쓴다.")
    print(f"  {'조합':22s} {'[A]중첩':>8s} {'+결합':>7s} {'+텍스처':>8s} {'+둘다':>7s} {'[B]':>6s}")
    tm = CP.TextureModel()
    b1 = {}
    for combo, devs in RAW_COMBO_FILES.items():
        parts = [(d, RAW_COMBO_SOLO[combo][d]) for d in devs if d in CP.SMPS_DEVICES]
        if len(parts) < 2:
            continue
        c = load_raw(combo, bg=bg)
        solos = {d: load_raw(s, bg=bg) for d, s in parts}
        d0 = parts[0][0]
        pw = {d: solos[d].p_w for d, _ in parts}
        if "laptop_charger" in pw:
            pw["laptop_charger"] = c.p_w - sum(v for k, v in pw.items() if k != "laptop_charger")
        Ic = phasors(c.i, BAND)
        Isum = sum(phasors(solos[d].i, BAND) for d, _ in parts)
        dz = sum(cm.delta(pw, Z_SITE_C).get(d, 0) for d, _ in parts)
        # 텍스처 델타: 각 기기의 단독 녹화 텍스처 -> 조합 녹화 텍스처
        dtex = 0
        for d, stem in parts:
            a_ = tm.models[d].current(pw[d], fcm.source_from_raw(combo))
            b_ = tm.models[d].current(pw[d], fcm.source_from_raw(stem))
            if a_ is not None and b_ is not None:
                dtex = dtex + (a_ - b_)
        clean = abs(pw[d0] - solos[d0].p_w) < 0.1 and all(
            abs(pw[d] - solos[d].p_w) < 0.1 for d, _ in parts)
        b1[combo] = {"naive": rel(Isum, Ic), "z": rel(Isum + dz, Ic),
                     "texture": rel(Isum + dtex, Ic), "both": rel(Isum + dtex + dz, Ic),
                     "powers": pw, "no_realloc": bool(clean)}
        print(f"  {combo:22s} {100 * b1[combo]['naive']:7.2f}% {100 * b1[combo]['z']:6.2f}% "
              f"{100 * b1[combo]['texture']:7.2f}% {100 * b1[combo]['both']:6.2f}%"
              + ("   무배분" if clean else "   (전력 재배분)"))
    g = [v for v in b1.values() if v["no_realloc"]]
    if g:
        print(f"  {'무배분 평균':22s} {100 * np.mean([v['naive'] for v in g]):7.2f}% "
              f"{100 * np.mean([v['z'] for v in g]):6.2f}% "
              f"{100 * np.mean([v['texture'] for v in g]):7.2f}% "
              f"{100 * np.mean([v['both'] for v in g]):6.2f}%"
              f"   12.185.12 [B] 3.6/3.7%")
    rec["b1"] = b1

    # ── 2 [B2] 소스 민감도 ─────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[2][B2] ΔI 를 원시 전압 18개로 각각 계산 — 캐시를 소스 하나로 해도 되는가")
    print("=" * 100)
    stems = [s for d in RAW_SNAPSHOT_FILES for s in raw_snapshots_of(d)] + list(RAW_COMBO_FILES)
    P3 = {"laptop_charger": 33.0, "beam_projector": 47.0, "minipc": 19.0}
    models = fcm.load_models()
    Z = fcm.line_impedance(Z_SITE_C, 0.0, CP.H)
    per = []
    for st in stems:
        try:
            src = fcm.source_from_raw(st)
            combo, _ = CP.forward_per_device(P3, models, src, Z)
            solo = {d: CP.forward_per_device({d: p}, models, src, Z)[0][d] for d, p in P3.items()}
            per.append(sum(combo[d] - solo[d] for d in P3))
        except Exception as e:
            print(f"    {st}: 실패 {e}")
    A = np.array(per)
    mu, sd = A.mean(0), A.std(0)
    print(f"  소스 {len(A)}개.  ΔI 합 [mA]")
    print(f"    {'차수':8s}" + "".join(f"{h:8d}" for h in ODD))
    print(f"    {'|평균|':8s}" + "".join(f"{1000 * abs(mu[h - 1]):8.2f}" for h in ODD))
    print(f"    {'산포':8s}" + "".join(f"{1000 * sd[h - 1]:8.2f}" for h in ODD))
    ratio = sd / np.maximum(np.abs(mu), 1e-12)
    print(f"    {'산포/평균':8s}" + "".join(f"{ratio[h - 1]:8.2f}" for h in ODD)
          + ("   -> 소스 하나로 충분" if np.median(ratio[:8]) < 0.20 else "   -> 소스도 키에 넣어야"))
    rec["b2"] = {"n": len(A), "ratio": list(ratio)}

    # ── 3 [B3] Z 민감도 ────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[3][B3] Z 무작위화가 만드는 다양성 — R 0.5~2.0Ω, L 0~400µH")
    print("=" * 100)
    src = fcm.source_from_raw(CP.DEFAULT_SOURCE)
    base = None
    print(f"  {'R[Ω]':>6s} {'L[µH]':>7s} {'|ΔI| 합[mA]':>12s} {'기준 대비':>10s}   차수별 |ΔI| [mA]")
    zrows = []
    for r_line in (0.5, 0.75, 1.0, 1.5, 2.0):
        for l_line in (0.0, 200e-6, 400e-6):
            Zx = fcm.line_impedance(r_line, l_line, CP.H)
            combo, _ = CP.forward_per_device(P3, models, src, Zx)
            solo = {d: CP.forward_per_device({d: p}, models, src, Zx)[0][d] for d, p in P3.items()}
            s = sum(combo[d] - solo[d] for d in P3)
            if base is None:
                base = s
            zrows.append({"r": r_line, "l": l_line, "norm": float(np.linalg.norm(s)),
                          "rel_to_base": rel(s, base)})
            print(f"  {r_line:6.2f} {1e6 * l_line:7.0f} {1000 * np.linalg.norm(s):11.2f} "
                  f"{100 * rel(s, base):9.1f}%   "
                  + " ".join(f"{1000 * abs(s[h - 1]):5.1f}" for h in (3, 5, 7, 9, 11, 13, 15)))
    rec["b3"] = zrows

    # ── 4 [B4] 비용 ────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[4][B4] 캐시 — 적중률과 시간")
    print("=" * 100)
    cm2 = CP.CouplingModel()
    rng = np.random.default_rng(0)
    t0 = time.time()
    n_call = 400
    for _ in range(n_call):
        p = {"laptop_charger": float(rng.uniform(17, 72)),
             "beam_projector": float(rng.uniform(45, 50)),
             "minipc": float(rng.uniform(7, 27))}
        if rng.random() < 0.4:
            p.pop(list(p)[int(rng.integers(3))])
        cm2.delta(p, float(rng.uniform(0.7, 2.0)), float(rng.uniform(*CP.L_LINE_RANGE)))
    dt = time.time() - t0
    st = cm2.stats()
    print(f"  {n_call}회 호출 {dt:.1f}s  ({1000 * dt / n_call:.1f} ms/회)")
    print(f"  적중 {st['hits']}  누락 {st['misses']}  실패 {st['failures']}  "
          f"적중률 {100 * st['hit_rate']:.0f}%  캐시 {st['size']}칸")
    # 두 번째 통과 (같은 씨앗) — 완전 적중이어야 한다
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(n_call):
        p = {"laptop_charger": float(rng.uniform(17, 72)),
             "beam_projector": float(rng.uniform(45, 50)),
             "minipc": float(rng.uniform(7, 27))}
        if rng.random() < 0.4:
            p.pop(list(p)[int(rng.integers(3))])
        cm2.delta(p, float(rng.uniform(0.7, 2.0)), float(rng.uniform(*CP.L_LINE_RANGE)))
    print(f"  같은 호출 재현 {time.time() - t0:.2f}s  누적 적중률 "
          f"{100 * cm2.stats()['hit_rate']:.0f}%")
    rec["b4"] = {"ms_per_call": 1000 * dt / n_call, **{k: float(v) for k, v in st.items()}}

    # ── 5 [B5] 안전 ────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[5][B5] 안전")
    print("=" * 100)
    one = cm.delta({"minipc": 19.0}, 1.0)
    print(f"  (a) SMPS 하나만: ΔI {'없음 — 통과' if not one else '⚠ 나왔다'}")
    res = cm.delta({"electiric_kettle": 1200.0, "oven": 1100.0}, 1.0)
    print(f"  (b) 저항만:     ΔI {'없음 — 통과' if not res else '⚠ 나왔다'}")
    d3 = cm.delta(P3, Z_SITE_C)
    I1 = {d: models[d].current(p, fcm.source_from_raw(CP.DEFAULT_SOURCE))[0] for d, p in P3.items()}
    print(f"  (c) 기본파 변화:")
    okc = True
    for d in P3:
        r = abs(d3[d][0]) / abs(I1[d])
        okc &= r < 0.05
        print(f"      {d:16s} |ΔI1|/|I1| {100 * r:5.2f}%" + ("" if r < 0.05 else "   ⚠ 5% 초과"))
    print(f"      -> {'통과' if okc else '⚠ 결합이 기본파를 바꾼다 — 전력 정답과 어긋난다'}")
    rec["b5"] = {"solo_empty": not one, "resistive_empty": not res,
                 "d_i1_frac": {d: float(abs(d3[d][0]) / abs(I1[d])) for d in P3}}

    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
