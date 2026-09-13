# -*- coding: utf-8 -*-
"""회로 모델 검증 게이트 + 장소 C / 4차 펌웨어 판정 — GPU 없이 30초 (12.184).

    python -X utf8 -m src.run_circuit_gate_probe
    python -X utf8 -m src.run_circuit_gate_probe --delay-ms 0.5     # 새 펌웨어 위상을 되돌리고 비교
    python -X utf8 -m src.run_circuit_gate_probe --no-gate5           # 파형 시뮬 생략

[1] 게이트 1·4  — 가이드 §4.2 파라미터로 `sp_curves`(장소 A) 를 재현하는가. 도통각·THD·손실.
[2] 장소 C 전달비 — 충전기 5C(4차 펌웨어) / 4_fixed(옛 펌웨어) 를 같은 전력의 장소 A 곡선으로 나눈 `T_C`.
[3] 지연 검정   — 5C / 4_fixed 절대 페이저 비. 크기 1 · 위상이 h 에 선형이면 **채널 지연**이다.
[4] 게이트 5    — 5C 의 `vh`+`vhdeg` 로 장소 C 전압 파형을 복원해 시뮬에 넣고, 이상 정현파 대비
                 회전 `T_sim` 을 실측 `T_C` 옆에 놓는다. 차수 부분집합·부호를 바꿔 어느 차수가 미는지 본다.
[5] 짝수차 검정 — `vh2/vh1` 크기·위상 안정성 vs 충전기 `|I2|/|I1|` 위상 안정성 (가이드 §8.2).
[6] 게이트 5 기기·부하 축 — 미니PC 단독 녹화(`minipc_4C`)의 전력대별 T_C 와 시뮬 (12.184.8).

판정 기준은 실행 전에 적었다 (`results/_criteria_circuit.md`).
"""
from typing import Dict
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np
import pandas as pd

from src.preprocessing.raw_csv import detect_raw_format, read_raw_csv
from src.preprocessing.raw_phasors import (apply_phase_delay, phase_delay_fit,
                                           steady_signature, voltage_phasors)
from src.synthesis.circuit_sim import GUIDE_PARAMS, ideal_wave, simulate, to_wave
from src.run_site_voltage_probe import canon_sig, load_curves_canonical  # noqa: E402

ODD = [2, 4, 6, 8, 10, 12, 14]          # h3..h15 의 0-based 인덱스
W = np.r_[1, 1, 1, 1, 1, .8, .8, .6, .6, .45, .45, .35, .35, .3]


def fmt_deg(z):
    return " ".join(f"{np.degrees(np.angle(v)):5.0f}" for v in z)


def fmt_mag(z):
    return " ".join(f"{abs(v):5.2f}" for v in z)


def loss(s, s_meas):
    d = np.r_[(s[1:] - s_meas[1:]).real, (s[1:] - s_meas[1:]).imag] * np.r_[W, W]
    return float(d @ d)


def load_csv(path: str, extra=()):
    """원시값으로 읽는다 (등록부의 위상 복원을 걸지 않는다) — [3] 지연 검정은 원시가 필요하다."""
    cols = ["p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, 16)] \
        + [f"ihdeg{h}" for h in range(1, 16)] + list(extra)
    fmt = detect_raw_format(path)
    if fmt["vhdeg"]:
        cols += [f"vh{h}" for h in range(1, 16)] + [f"vhdeg{h}" for h in range(1, 16)]
    df, info = read_raw_csv(path, usecols=cols, phase_fix=False)
    return df, fmt


def gate14(curves) -> Dict[str, float]:
    print("\n[1] 게이트 1·4 — 가이드 §4.2 파라미터 vs sp_curves (장소 A, 이상 정현파)")
    print("    도통각 기준 30~40° (반주기, 브리지 전류 기준), 손실 < 0.01 이 게이트 4")
    out = {}
    for dev, par in GUIDE_PARAMS.items():
        P = curves[f"{dev}__P"]; S = curves[f"{dev}__S"]; V = curves[f"{dev}__V"]
        ls = []
        print(f"  {dev}")
        for i in range(len(P)):
            r = simulate(P[i], *par, V_rms=float(V[i]))
            sa = S[i] / S[i][0]
            l_ = loss(r["s"], sa) if r["ok"] else np.nan
            ls.append(l_)
            if i in (0, len(P) // 2, len(P) - 1):
                print(f"    P={P[i]:5.1f}  P_sim={r['p_w']:5.1f}  도통각 {r['cond_deg']:5.1f}°  "
                      f"THD {r['thd']:.2f}  손실 {l_:.4f}")
                print(f"      시뮬 ∠h3..15 {fmt_deg(r['s'][ODD])}   |s| {fmt_mag(r['s'][ODD])}")
                print(f"      실측 ∠h3..15 {fmt_deg(sa[ODD])}   |s| {fmt_mag(sa[ODD])}")
        out[dev] = float(np.nanmean(ls))
        print(f"    평균 손실 {out[dev]:.4f}   ({'통과' if out[dev] < 0.01 else '미통과'} — 게이트 4)")
    return out


def site_c_transfer(curves, sig_new, sig_old, delay_ms: float):
    print("\n[2] 장소 C 전달비 T_C = 실측 / 장소 A 곡선(같은 전력)  — 충전기")
    P = curves["laptop_charger__P"]; S = curves["laptop_charger__S"]
    res = {}
    for lab, sg in (("5C 4차펌웨어", sig_new), ("4_fixed 옛펌웨어", sig_old)):
        i = int(np.argmin(np.abs(P - sg["p_w"])))
        sa = S[i] / S[i][0]
        T = sg["s"] / sa
        res[lab] = T
        print(f"  {lab:16s} P={sg['p_w']:5.1f} n={sg['n']:6d} V={sg['vrms']:5.1f} ihdeg1={sg['ihdeg1']:+5.1f}"
              f"  (A 곡선 P={P[i]:.1f})")
        print(f"      ∠T h3..15 {fmt_deg(T[ODD])}   |T| {fmt_mag(T[ODD])}")
    if delay_ms:
        print(f"  (--delay-ms {delay_ms}: 5C 위상을 +{360*60*delay_ms/1000:.1f}°×h 되돌린 값이 위 5C 행이다)")
    return res


def delay_test(sig_new_raw, sig_old):
    print("\n[3] 지연 검정 — 5C(4차, 보정 전) / 4_fixed(옛) 절대 페이저 비")
    R = sig_new_raw["I"] / sig_old["I"]
    f = phase_delay_fit(R)
    print(f"  |비| h1..15 : {fmt_mag(R)}")
    print(f"  ∠비  h1..15 : {fmt_deg(R)}")
    print(f"  홀수차 선형 맞춤: 기울기 {f['slope_deg']:+.2f}°/차수 (= 지연 {f['delay_ms']:+.3f} ms), "
          f"절편 {f['intercept_deg']:+.1f}°, 잔차 {np.abs(f['resid_deg']).max():.1f}° 이내")
    pure = abs(f["resid_deg"]).max() < 2.0 and np.all(np.abs(f["mag"] - 1) < 0.1)
    print("  판정:", "**순수 회전(시간지연 모델)** — 크기 불변·위상 h 선형. 물리(파형)가 아니라 보드의 교정 상태나 "
          "채널 기준이다 (12.184.3: 부하 없는 USER 버튼 교정이 정확히 이 모양을 만든다)"
          if pure else "선형 지연으로 안 설명된다 — 파형 차이가 섞여 있다")
    return f


def gate5(curves, df_new, sig_new, sig_old, delay_ms: float):
    print("\n[4] 게이트 5 (전압축) — 5C 의 vh+vhdeg 파형을 충전기 시뮬에 넣는다")
    Vc, st = voltage_phasors(df_new, sig_new["mask"])
    print(f"  장소 C 전압 (n={st['n']}, vrms {st['vrms']:.1f}):")
    print(f"    |V| h1..15   {' '.join(f'{abs(v):6.2f}' for v in Vc)}")
    print(f"    ∠V  h1..15   {fmt_deg(Vc)}")
    print(f"    창간 산포°    {' '.join(f'{c:5.1f}' for c in st['circ_std_deg'])}")
    par = GUIDE_PARAMS["laptop_charger"]
    Pp = sig_new["p_w"]
    base = simulate(Pp, *par, vsrc=ideal_wave(abs(Vc[0])))
    s0 = base["s"]
    print(f"  이상 정현파 {abs(Vc[0]):.0f}V, P={Pp:.0f}W: ∠h1 {np.degrees(np.angle(base['I'][0])):+5.1f}°")
    P = curves["laptop_charger__P"]; S = curves["laptop_charger__S"]
    i = int(np.argmin(np.abs(P - Pp))); sa = S[i] / S[i][0]
    print(f"  {'실측 T_C (옛펌웨어 4_fixed)':34s} ∠ {fmt_deg((sig_old['s'] / sa)[ODD])}  |T| {fmt_mag((sig_old['s'] / sa)[ODD])}")
    print(f"  {'실측 T_C (5C, 지연 보정 %.2fms)' % delay_ms:34s} ∠ {fmt_deg((sig_new['s'] / sa)[ODD])}  |T| {fmt_mag((sig_new['s'] / sa)[ODD])}")
    cases = [("시뮬 +vhdeg 전 차수", +1, None),
             ("시뮬 +vhdeg 홀수차만", +1, {1, 3, 5, 7, 9, 11, 13, 15}),
             ("시뮬 +vhdeg h1+h3", +1, {1, 3}),
             ("시뮬 +vhdeg h1,3,5,7", +1, {1, 3, 5, 7}),
             ("시뮬 +vhdeg h1+h9..15", +1, {1, 9, 11, 13, 15}),
             ("시뮬 −vhdeg 전 차수 (부호 반전)", -1, None)]
    for lab, sign, orders in cases:
        V = np.array([v if (orders is None or k + 1 in orders) else 0 for k, v in enumerate(Vc)], complex)
        V = np.abs(V) * np.exp(1j * sign * np.angle(V))
        r = simulate(Pp, *par, vsrc=to_wave(V))
        if not r["ok"]:
            print(f"  {lab:34s} 발산"); continue
        T = r["s"] / s0
        print(f"  {lab:34s} ∠ {fmt_deg(T[ODD])}  |T| {fmt_mag(T[ODD])}   ∠h1 {np.degrees(np.angle(r['I'][0])):+5.1f}°")


def minipc_gate5(curves, path: str):
    """[6] 미니PC 단독 녹화(장소 C, 기본 교정)로 게이트 5 의 **기기·부하 축** — 전력대별 실측 T_C 와 시뮬."""
    import os
    if not os.path.exists(path):
        print(f"\n[6] 건너뜀 — {path} 없음"); return
    print(f"\n[6] 게이트 5 (기기·부하 축) — 미니PC {path}")
    df, fmt = load_csv(path)
    if not fmt["vhdeg"]:
        print("  vhdeg 없는 파일 — 건너뜀"); return
    P = curves["minipc__P"]; S = curves["minipc__S"]; par = GUIDE_PARAMS["minipc"]
    Vc = None
    for lo, hi in ((8, 12), (12, 16), (19, 26)):
        sg = steady_signature(df, lo, hi)
        if sg["n"] < 300:
            continue
        if Vc is None:
            Vc, st = voltage_phasors(df, sg["mask"])
            print(f"  전압 (n={st['n']}): |V| h3,5,7,9 {np.round(np.abs(Vc[[2, 4, 6, 8]]), 2)}  ∠ {fmt_deg(Vc[[2, 4, 6, 8]])}")
        i = int(np.argmin(np.abs(P - sg["p_w"]))); sa = S[i] / S[i][0]; T = canon_sig(sg["s"]) / sa
        print(f"  P={sg['p_w']:5.1f}W n={sg['n']:6d} ihdeg1 {sg['ihdeg1']:+5.1f} (A {np.degrees(curves['minipc__ph1_rad'][i]):+5.1f})"
              f"   실측 T_C ∠ {fmt_deg(T[ODD])}  |T| {fmt_mag(T[ODD])}")
        s0 = simulate(sg["p_w"], *par, vsrc=ideal_wave(abs(Vc[0])))["s"]
        for lab, orders in (("시뮬 h1+h3", {1, 3}), ("시뮬 h1,3,5,7", {1, 3, 5, 7}), ("시뮬 전 차수", None)):
            V = np.array([v if (orders is None or k + 1 in orders) else 0 for k, v in enumerate(Vc)], complex)
            r = simulate(sg["p_w"], *par, vsrc=to_wave(V))
            if not r["ok"]:
                print(f"    {lab:12s} 발산"); continue
            Ts = r["s"] / s0
            print(f"    {lab:12s}{'':32s}∠ {fmt_deg(Ts[ODD])}  |T| {fmt_mag(Ts[ODD])}")
    print("  읽기: h1+h3(+5,7) 만 넣은 시뮬이 h3~h11 에서 실측과 10~15° 안이고 부하 추세(경부하일수록 큰 회전)도 따라온다 —"
          " 전압축 기작 확인. h9~h15 전압 위상을 넣으면 나빠진다 (12.184.8)")


def even_test(df_new, sig_new):
    print("\n[5] 짝수차 검정 (가이드 §8.2) — 전압 vh2 는 잠겨 있고 전류 I2 는 떠 있는가")
    m = sig_new["mask"]
    vh1 = df_new["vh1"].to_numpy(np.float64)[m]; vh2 = df_new["vh2"].to_numpy(np.float64)[m]
    z = np.exp(1j * np.deg2rad(df_new["vhdeg2"].to_numpy(np.float64)[m])); r = abs(z.mean())
    cs_v = np.degrees(np.sqrt(max(-2 * np.log(max(r, 1e-12)), 0)))
    m2, cs_i = sig_new["even2"]
    print(f"  vh2/vh1 중앙 {np.median(vh2/vh1)*100:.2f}%  ∠vh2 원형표준편차 {cs_v:.1f}°   |  "
          f"|I2|/|I1| 중앙 {m2*100:.2f}%  ∠I2 원형표준편차 {cs_i:.1f}°")
    print("  판정:", "전압 짝수차는 위상이 잠겨 있고 전류 짝수차는 무작위다 — 정류기가 6V 짝수차 전압에 반응하지 않는다면 "
          "vh2 는 콘센트에 없는 **전압 채널 인공물**이다 (저항 부하로 확정할 것)" if cs_v < 5 and cs_i > 30
          else "잠김/무작위 구분이 뚜렷하지 않다")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curves", default="processed_data/sp_curves.npz")
    ap.add_argument("--site-c-new", default="data/laptop_charger_5C.csv")
    ap.add_argument("--site-c-old", default="data/laptop_charger_4_fixed.csv")
    ap.add_argument("--band", default="66,76", help="정상 구간 전력 [lo,hi) W")
    ap.add_argument("--delay-ms", type=float, default=None,
                    help="새 녹화의 전류 위상 복원 (ms 상당). 기본은 등록부 PHASE_FIX_DEG_PER_ORDER "
                         "(10.8°/차수 = 0.5ms). 0 을 주면 원시 그대로 본다")
    ap.add_argument("--no-gate5", action="store_true")
    ap.add_argument("--minipc", default="data/minipc_4C.csv", help="[6] 미니PC 단독 녹화 (장소 C, vhdeg 필요)")
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.band.split(","))
    curves = load_curves_canonical(a.curves)          # 정본 LOW 규약 (12.184.13)
    if a.delay_ms is None:
        from pathlib import Path
        from src.preprocessing.file_registry import phase_fix_of
        a.delay_ms = phase_fix_of(Path(a.site_c_new).stem) / 360.0 / 60.0 * 1e3

    gate14(curves)

    df_new, fmt_new = load_csv(a.site_c_new)
    df_old, fmt_old = load_csv(a.site_c_old)
    print(f"\n  형식: {a.site_c_new} v{fmt_new['version']} (vhdeg {fmt_new['vhdeg']}) / "
          f"{a.site_c_old} v{fmt_old['version']} (vhdeg {fmt_old['vhdeg']})")
    sig_new_raw = steady_signature(df_new, lo, hi)
    sig_old = steady_signature(df_old, lo, hi)
    if sig_new_raw["n"] == 0 or sig_old["n"] == 0:
        print("정상 구간 표본이 없다 — --band 를 조정하라"); return 1
    f = delay_test(sig_new_raw, sig_old)
    df_c = apply_phase_delay(df_new, a.delay_ms) if a.delay_ms else df_new
    sig_new = steady_signature(df_c, lo, hi) if a.delay_ms else sig_new_raw
    # [2][4] 는 정본 곡선과 비교하므로 서명도 정본 규약으로 (둘 다 LOW 전용 녹화). [3] 은 원시 그대로.
    sig_new = {**sig_new, "s": canon_sig(sig_new["s"])}
    sig_old = {**sig_old, "s": canon_sig(sig_old["s"])}
    site_c_transfer(curves, sig_new, sig_old, a.delay_ms)
    if not a.no_gate5:
        if fmt_new["vhdeg"]:
            gate5(curves, df_new, sig_new, sig_old, a.delay_ms)
        else:
            print("\n[4] 건너뜀 — vhdeg 가 없는 파일이다")
    if fmt_new["vhdeg"]:
        even_test(df_new, sig_new_raw)
    if not a.no_gate5:
        minipc_gate5(curves, a.minipc)
    if not a.delay_ms and abs(f["delay_ms"]) > 0.05:
        print(f"\n다음: --delay-ms {abs(f['delay_ms']):.3f} 로 다시 돌려 보정 뒤의 T_C 를 봐라")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
