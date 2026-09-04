# -*- coding: utf-8 -*-
"""원시 조합 스냅샷으로 **결합**을 검정한다 (12.185.12).

    python -X utf8 -m src.run_raw_combo_probe

조합 녹화는 단자 전압 V_term 을 직접 재므로 고정점 반복(가이드 §6.1)이 필요 없다. 각 기기를 그
V_term 으로 시뮬해 더한 것이 실측 총전류와 같은지 보면 모델의 합성 능력을 가정 없이 잰다.

  [A] 실측 중첩   단독 녹화 전류의 단순 합 vs 조합 실측  -> 결합 효과의 크기 (모델 무관)
  [B] 모델 합성   각 기기를 **조합의** V_term 으로 시뮬해 합 vs 조합 실측   <- 이것이 판정
  [C] 모델 단독합 각 기기를 **단독 녹화의** V_term 으로 시뮬해 합 (결합 무시)

파라미터는 `results/_circuit_raw_C.json` (run_raw_fit_probe) 의 자유 적합값을 그대로 쓴다 —
조합에 맞춘 자유 파라미터가 하나도 없다.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa
import json
import numpy as np

from src.synthesis.fit_raw import (BAND, background_phasors, downsample, load_raw, phasors,
                                   rc_filter, sim_current)
from src.synthesis.circuit_sim import NPC, simulate

from src.preprocessing.file_registry import RAW_COMBO_FILES, RAW_COMBO_SOLO

PAR = {d: tuple(v["fit_free"]["par5"])
       for d, v in json.load(open("results/_circuit_raw_C.json", encoding="utf-8"))["devices"].items()}
COMBO = {c: [(d, RAW_COMBO_SOLO[c][d]) for d in devs] for c, devs in RAW_COMBO_FILES.items()}
bg = background_phasors()


def rel(a, b):
    """대역제한 상대 RMS (파세발)."""
    d = phasors(a, BAND) - phasors(b, BAND)
    return float(np.sqrt(np.sum(np.abs(d) ** 2)) / np.sqrt(np.sum(np.abs(phasors(b, BAND)) ** 2)))


for combo, parts in COMBO.items():
    parts = [(d, s) for d, s in parts if d in PAR]
    if not parts:
        continue
    c = load_raw(combo, bg=bg)
    solos = {d: load_raw(s, bg=bg) for d, s in parts}
    p_solo = sum(p.p_w for p in solos.values())
    # 전력 배분: 프로젝터·미니PC 는 단독값 고정(정상 부하), 나머지를 충전기에 준다
    # (충전기는 배터리 상태로 변해서 단독 스냅샷 전력과 다르다)
    pw = {d: solos[d].p_w for d, _ in parts}
    if "laptop_charger" in pw:
        pw["laptop_charger"] = c.p_w - sum(v for k, v in pw.items() if k != "laptop_charger")
    else:
        k = c.p_w / p_solo
        pw = {d: v * k for d, v in pw.items()}
    print("=" * 92)
    print(f"[{combo}]  조합 실측 {c.p_w:6.2f}W   단독 합 {p_solo:6.2f}W   배분: "
          + " + ".join(f"{d} {pw[d]:.1f}W" for d, _ in parts))
    # [A] 실측 중첩
    i_sum = sum(solos[d].i for d, _ in parts)
    print(f"   [A] 실측 단순 합 vs 조합 실측          {100 * rel(i_sum, c.i):6.2f}%")
    # [B] 모델을 조합의 V_term 으로
    i_b = np.zeros_like(c.i)
    ok = True
    for d, _ in parts:
        pt = type(c)(c.stem, c.v, c.i, pw[d], c.vsrc, c.irms, c.n_cyc,
                     c.scatter, c.oob, c.range_mixed)
        x = sim_current(PAR[d], pt, match_power=True)
        if x is None:
            ok = False
            break
        i_b = i_b + x
    if ok:
        print(f"   [B] 모델(조합 V_term) 합 vs 조합 실측  {100 * rel(i_b, c.i):6.2f}%")
    # [C] 모델을 각자 단독 V_term 으로
    i_c = np.zeros_like(c.i)
    for d, _ in parts:
        x = sim_current(PAR[d], solos[d], match_power=True)
        i_c = i_c + (x if x is not None else 0)
    print(f"   [C] 모델(단독 V_term) 합 vs 조합 실측  {100 * rel(i_c, c.i):6.2f}%")
    print(f"       참고: 모델 단독 재현 " + "  ".join(
        f"{d} {100 * rel(sim_current(PAR[d], solos[d]), solos[d].i):.1f}%" for d, _ in parts))
    # 차수별
    Ic = phasors(c.i, BAND)
    Is = phasors(i_sum, BAND)
    Ib = phasors(i_b, BAND) if ok else None
    print("       차수          " + "".join(f"{k:8d}" for k in range(1, 16, 2)))
    print("       조합 실측[mA] " + "".join(f"{1000 * abs(Ic[k - 1]):8.2f}" for k in range(1, 16, 2)))
    print("       실측합/조합   " + "".join(f"{abs(Is[k - 1]) / abs(Ic[k - 1]):8.3f}" for k in range(1, 16, 2)))
    if ok:
        print("       모델합/조합   " + "".join(f"{abs(Ib[k - 1]) / abs(Ic[k - 1]):8.3f}" for k in range(1, 16, 2)))
        print("       모델 Δ위상[°] " + "".join(
            f"{np.degrees(np.angle(Ib[k - 1] / Ic[k - 1])):8.1f}" for k in range(1, 16, 2)))
    print()
