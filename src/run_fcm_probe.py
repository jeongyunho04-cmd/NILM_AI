# -*- coding: utf-8 -*-
"""FCM 추출과 결합 생성기 검정 (12.185.13~15). 기준은 `results/_criteria_circuit.md` [F1]~[F4].

    python -X utf8 -m src.run_fcm_probe
    python -X utf8 -m src.run_fcm_probe --stages 3 4

단계
----
    1  FCM 구조     장소 C 전압에서 J, Y. 대각·최대 비대각 (가이드 §5.1 대조)
    2  선형화 한계  I ≈ J − Y·ΔV 가 어디서 깨지는가 (가이드 §5.2 재현 — 우리 모델로)
    3  Z(h) 실측    조합 원시 스냅샷의 차분에서 선로 임피던스 (12.185.14)
    4  forward 검정 고정점이 조합 실측 (I_total, V_term) 을 재현하는가 (12.185.15)
    5  Z 민감도     Z 를 얼마나 정확히 알아야 하는가 (생성기 설계용)
"""
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import RAW_COMBO_FILES, RAW_COMBO_SOLO
from src.synthesis import fcm
from src.synthesis.fcm import (H, HFULL, DeviceModel, forward, line_impedance, load_models,
                              measure_z, odd_only, to_spectrum)
from src.synthesis.fit_raw import background_phasors, load_raw, phasors, rc_filter

OUT = "results/_fcm_C.json"
ODD = list(range(1, H + 1, 2))
#: 12.184.11 의 장소 C 계단 임피던스 (포트 on/off). 3단계에서 이것과 대조한다
Z_C_STEP = 0.45


def _ref(pt):
    """소스 V1 의 위상 [rad] — 펌웨어 관례 `arg(X_h) − h·arg(V_1)` 의 기준."""
    return float(np.angle(phasors(rc_filter(pt.v, inverse=True), H)[0]))


def vtrue(pt) -> np.ndarray:
    """(15,) 참 계통 전압 페이저 — 계측 RC 를 벗기고, **V1 기준 위상**, 홀수차만.

    ⚠ `circuit_sim.simulate` 가 주는 `I`/`V` 는 소스 V1 기준이다 (펌웨어 관례). 절대 시간
    프레임의 페이저와 섞으면 h 차에 `h·∠V1` 이 실린다 — h15 에서 ∠V1 이 2° 만 돼도 30° 다.
    """
    X = phasors(rc_filter(pt.v, inverse=True), H)
    h = np.arange(1, H + 1)
    return odd_only(np.abs(X) * np.exp(1j * (np.angle(X) - h * _ref(pt))))


def itrue(pt) -> np.ndarray:
    """(15,) 참 전류 페이저 — 계측 RC 를 벗기고 홀수차만.

    ⚠ 시뮬은 참 전류를 내는데 원시 표본에는 RC 가 걸려 있다. 벗기지 않고 비교하면 h9 5%,
    h13 11%, h15 15% 를 모델 탓으로 돌리게 된다 (실제로 4단계 첫 판이 그랬다).
    """
    X = phasors(rc_filter(pt.i, inverse=True), H)
    h = np.arange(1, H + 1)
    return odd_only(np.abs(X) * np.exp(1j * (np.angle(X) - h * _ref(pt))))


def power_split(combo: str, solos: Dict, p_total: float) -> Dict[str, float]:
    """프로젝터·미니PC 는 단독값 고정, 나머지를 충전기에 (12.185.12)."""
    pw = {d: solos[d].p_w for d in solos}
    if "laptop_charger" in pw:
        pw["laptop_charger"] = p_total - sum(v for k, v in pw.items() if k != "laptop_charger")
    else:
        k = p_total / sum(pw.values())
        pw = {d: v * k for d, v in pw.items()}
    return pw


def fmt_z(Z) -> str:
    return "  ".join(f"h{h}:{abs(Z[h - 1]):5.2f}∠{np.degrees(np.angle(Z[h - 1])):+4.0f}" for h in ODD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--r-line", type=float, default=None, help="선로 저항 [Ω] (기본: 3단계 실측)")
    ap.add_argument("--l-line", type=float, default=100e-6)
    a = ap.parse_args()

    models = load_models()
    # 생성기 규약(V15)으로 맞춘 파라미터가 있으면 4단계에서 함께 잰다 (12.185.16)
    try:
        models_v15 = load_models("results/_circuit_raw_C_v15.json", key="fit_free")
    except Exception:
        models_v15 = None
    bg = background_phasors()
    print("기기 모델 (원시 적합, 12.185.10):")
    for d, m in models.items():
        C, R, L, Cx, rd = m.par5
        print(f"  {d:15s} C_dc {C * 1e6:5.1f}µF R {R:5.2f}Ω L {L * 1e6:6.0f}µH "
              f"Cx {Cx * 1e6:.3f}µF rd {rd:5.2f}Ω")
    rec: Dict = {}

    # 장소 C 기준 전압 (프로젝터 단독 스냅샷의 실측 파형)
    ref = load_raw("raw_beam_projector_1", bg=bg)
    V_C = vtrue(ref)
    print(f"\n장소 C 기준 전압: V1 {abs(V_C[0]):.1f}V  "
          + "  ".join(f"vh{h} {100 * abs(V_C[h - 1]) / abs(V_C[0]):.2f}%∠{np.degrees(np.angle(V_C[h - 1])):+.0f}°"
                      for h in (3, 5, 7)))

    # ── 1 FCM 구조 ─────────────────────────────────────────────────────────
    if 1 in a.stages:
        print("\n[1] FCM 구조 — |Y_hk|·V1/|I1| (k차 전압 1% 변화가 h차 전류를 I1 의 몇 % 바꾸나)")
        pw = {"laptop_charger": 45.0, "beam_projector": 47.3, "minipc": 18.0}
        rows = {}
        for d, p in pw.items():
            if d not in models:
                continue
            J, Y = models[d].norton(p, V_C)
            Yn = np.abs(Y) * abs(V_C[0]) / abs(J[0])
            M = Yn.copy()
            np.fill_diagonal(M, 0)
            i, k = np.unravel_index(np.argmax(M), M.shape)
            rows[d] = {"diag": [Yn[h - 1, h - 1] for h in ODD], "max_off": [int(i + 1), int(k + 1), M[i, k]],
                       "J1": abs(J[0])}
            print(f"    {d:15s} p={p:5.1f}W  |J1| {abs(J[0]):.3f}A")
            print(f"      대각 h1..h7   " + "  ".join(f"{Yn[h - 1, h - 1]:6.2f}" for h in (1, 3, 5, 7)))
            print(f"      최대 비대각  Y[h={i + 1},k={k + 1}] = {M[i, k]:.2f}")
        rec["stage1"] = rows

    # ── 2 선형화 한계 ──────────────────────────────────────────────────────
    if 2 in a.stages:
        print("\n[2] 선형화 한계 — I ≈ J − Y·ΔV 대 시뮬 (가이드 §5.2 를 우리 모델로)")
        rows = {}
        for d, p in (("laptop_charger", 45.0), ("beam_projector", 47.3), ("minipc", 18.0)):
            if d not in models:
                continue
            J, Y = models[d].norton(p, V_C)
            out = {}
            for lab, mk in (("h3 +1%", lambda V: 0.01), ("h3 +3%", lambda V: 0.03), ("h3 위상 +20°", None)):
                Vp = V_C.copy()
                if mk is None:
                    Vp[2] = V_C[2] * np.exp(1j * np.deg2rad(20))
                else:
                    Vp[2] = V_C[2] + mk(V_C) * abs(V_C[0]) * np.exp(1j * np.angle(V_C[2]))
                Isim = models[d].current(p, Vp)
                Ilin = J - Y @ (Vp - V_C)
                err = np.abs(Ilin - Isim) / np.abs(Isim)
                chg = np.abs(Isim - J) / np.abs(J)
                out[lab] = {"err": err.tolist(), "change": chg.tolist()}
                print(f"    {d:15s} {lab:12s} 변화 h3 {100 * chg[2]:5.1f}% h9 {100 * chg[8]:5.1f}%"
                      f"  |  선형화 오차 h3 {100 * err[2]:5.1f}% h9 {100 * err[8]:5.1f}% h13 {100 * err[12]:5.1f}%")
            rows[d] = out
        rec["stage2"] = rows

    # ── 3 Z(h) 실측 ────────────────────────────────────────────────────────
    r_line = a.r_line
    if 3 in a.stages:
        from src.synthesis.fcm import z_from_steps
        print("\n[3] Z(h) 실측 — 포트 on/off 계단 (12.185.14). ΔI1 6.4A 라 조합 차분(0.14A)보다 45배 낫다")
        Zm, sp, n = z_from_steps()
        print(f"    계단 {n}개")
        print("    차수  " + "".join(f"{h:9d}" for h in ODD))
        print("    |Z|   " + "".join(f"{abs(Zm[h - 1]):9.3f}" for h in ODD) + "  Ω")
        print("    ∠Z    " + "".join(f"{np.degrees(np.angle(Zm[h - 1])):9.1f}" for h in ODD) + "  °")
        print("    산포  " + "".join(f"{100 * sp[h - 1]:8.0f}%" for h in ODD))
        print("    -> h1 만 채택: |Z_1| = %.3f Ω (산포 %.0f%%). h3~h7 은 재현되지만 ∠Z 가 수동" % (abs(Zm[0]), 100 * sp[0]))
        print("       R+jωL 이 낼 수 없는 값이다 (h3 141° = 실수부 음수) — 분모에 HIGH 레인지")
        print("       저항 고조파 인공물(규칙 75), 분자에 상류 부하의 응답이 섞였다. h9 이상은 산포 240~340%.")
        rec["stage3"] = {"Z": [[z.real, z.imag] for z in Zm], "spread": sp.tolist(), "n": n}
        if r_line is None and np.isfinite(abs(Zm[0])):
            r_line = float(abs(Zm[0]))
    if r_line is None:
        r_line = Z_C_STEP                      # 3단계를 건너뛰면 12.184.11 의 계단값
    # ── 4 forward 검정 ─────────────────────────────────────────────────────
    if 4 in a.stages:
        Z = line_impedance(r_line, a.l_line)
        print(f"\n[4] forward() 고정점 검정 — Z = {r_line:.3f}Ω + {a.l_line * 1e6:.0f}µH (12.185.15)")
        print("    조합 실측 (V_term, I_total) 에서 V_src 를 되짚고, 고정점이 그것을 되찾는가")
        print("    소스는 전대역(h≤128)으로 만든다 — 합성은 오프라인이라 소스 전압을 우리가 고른다.")
        print("    대조로 h15 절단 소스 + V15 규약 파라미터도 함께 잰다 (12.185.16).")
        rows = {}
        for combo, devs in RAW_COMBO_FILES.items():
            c = load_raw(combo, bg=bg)
            solos = {d: load_raw(RAW_COMBO_SOLO[combo][d], bg=bg) for d in devs}
            pw = power_split(combo, solos, c.p_w)
            Vt_meas = vtrue(c)
            It_meas = itrue(c)
            # 소스는 **전대역**으로 만든다 (12.185.16/18): 실측 단자 전압 전대역 + Z·I 되짚기.
            # h17+ 는 우리 기기 전류가 없으니 Z 보정 없이 그대로 남는다.
            Vfull = to_spectrum(rc_filter(c.v, inverse=True))
            Vfull = Vfull * np.exp(-1j * np.arange(len(Vfull)) * 0.0)
            Zf = np.zeros(len(Vfull), complex)
            Zf[1:H + 1] = Z
            If = np.zeros(len(Vfull), complex)
            If[1:H + 1] = It_meas
            V_src = Vfull + Zf * If
            try:
                I_p, V_p, delta = forward(pw, models, V_src, Z, n_iter=4)
            except RuntimeError as e:
                print(f"    {combo:22s} {e}")
                continue
            eI = float(np.linalg.norm(I_p - It_meas) / np.linalg.norm(It_meas))
            eV = float(np.linalg.norm(V_p[1:H + 1] - Vt_meas) / np.linalg.norm(Vt_meas))
            # 대조: 소스를 h15 로 자르고 V15 규약 파라미터로 (지금까지의 생성기 규약)
            eI15 = np.nan
            if models_v15 is not None:
                try:
                    V15src = np.zeros(len(V_src), complex)
                    V15src[1:H + 1] = V_src[1:H + 1]
                    I15, _, _ = forward(pw, models_v15, V15src, Z, n_iter=4)
                    eI15 = float(np.linalg.norm(I15 - It_meas) / np.linalg.norm(It_meas))
                except RuntimeError:
                    pass
            rows[combo] = {"err_I_full": eI, "err_I_v15": eI15, "err_V": eV, "delta": delta, "powers": pw}
            print(f"    {combo:22s} 전대역 소스 {100 * eI:5.2f}%   h15 소스 {100 * eI15:5.2f}%   "
                  f"V_term 오차 {100 * eV:5.3f}%   수렴 {delta:.1e}")
        rec["stage4"] = rows

    # ── 5 Z 민감도 ─────────────────────────────────────────────────────────
    if 5 in a.stages:
        print("\n[5] Z 민감도 — 생성기가 Z 를 얼마나 정확히 알아야 하는가")
        c = load_raw("raw_smps3_1", bg=bg)
        solos = {d: load_raw(RAW_COMBO_SOLO["raw_smps3_1"][d], bg=bg) for d in RAW_COMBO_FILES["raw_smps3_1"]}
        pw = power_split("raw_smps3_1", solos, c.p_w)
        Z0 = line_impedance(r_line, a.l_line)
        V_src = odd_only(vtrue(c) + Z0 * itrue(c))
        base = None
        rows = []
        for rl in (0.0, 0.25, r_line, 0.9, 1.5, 3.0):
            for ll in (0.0, a.l_line, 400e-6):
                Z = line_impedance(rl, ll)
                try:
                    I_p, _, _ = forward(pw, models, V_src, Z, n_iter=4)
                except RuntimeError:
                    continue
                if base is None:
                    base = I_p
                d = float(np.linalg.norm(I_p - base) / np.linalg.norm(base))
                rows.append({"R": rl, "L": ll, "rel": d})
        b0 = [r for r in rows if abs(r["R"] - r_line) < 1e-9 and abs(r["L"] - a.l_line) < 1e-12]
        ref_I = b0[0]["rel"] if b0 else 0.0
        print(f"    기준 R=0, L=0 대비 총전류 변화 (기준점 R={r_line:.2f} L={a.l_line * 1e6:.0f}µH 은 {100 * ref_I:.1f}%)")
        for r in rows:
            print(f"      R {r['R']:5.2f}Ω  L {r['L'] * 1e6:5.0f}µH   Δ|I| {100 * r['rel']:5.2f}%")
        rec["stage5"] = rows

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
