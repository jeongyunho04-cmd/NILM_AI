# -*- coding: utf-8 -*-
"""장소 전압 파형 — 포트 전압계 검정, 장소 C 실측 파형 재피팅, 장소 간 전이 (12.184.9).

    python -X utf8 -m src.run_site_voltage_probe            # 전부 (재피팅 포함, 약 2분)
    python -X utf8 -m src.run_site_voltage_probe --no-fit   # [1][2] 와 가이드 파라미터 예측만

[1] 포트 전압계   순저항이면 I_h = V_h/R 이라 `ihdeg_h = vhdeg_h` 다 (펌웨어 위상 교정이 채널 지연을 이미 지운
                 값이므로 되돌리지 않는다 — 포트 기본파 +0.02/+0.16° 가 그 증거). 그러면 vhdeg 없는 옛 녹화라도
                 포트가 켜진 구간의 ihdeg_h 가 그 장소의 전압 위상이다 — **같은 장소·같은 날의 두 저항이 같은 값을
                 주는지**가 방법의 검정이다 (기준: h3~h9 15° 안). 크기도 두 길(전압 채널 vh_h / 전류 |I_h|/|I_1|·V1)로 잰다.
[2] 장소 전압    A(8/25 포트) / B(9/02 포트) 는 [1] 이 서는 만큼만, C 는 minipc_4C 의 vhdeg 로 직접.
[3] 장소 C 재피팅 §4.2 의 5파라미터를 **실측 파형**으로 다시 맞춘다 — minipc_4C (9~30W 5점), laptop_charger_5C
                 (28~72W 9점, 위상 복원 적용). 이상 정현파로 맞춘 가이드 값은 장소 A 의 왜곡을 파라미터에 박고 있었다.
[4] 전이·예측    C 에서 맞춘 파라미터로 (a) 장소 A 곡선(sp_curves)을 이상 정현파로 낼 때의 손실, (b) 예측 T_C 와
                 실측 T_C, (c) 장소 B 는 V_B 가 [1] 을 못 넘으면 참고로만.
판정 기준은 `results/_criteria_circuit.md` (추가분, 결과 전에 적었다).
"""
from typing import Dict, List, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.raw_csv import read_raw_csv
from src.preprocessing.raw_phasors import current_phasors, steady_signature, voltage_phasors
from src.synthesis.circuit_sim import GUIDE_PARAMS, simulate, to_wave

ODD = [2, 4, 6, 8, 10, 12, 14]
H = 15
W = np.r_[1, 1, 1, 1, 1, .8, .8, .6, .6, .45, .45, .35, .35, .3]
WW = np.r_[W, W]

#: 저항 녹화 (stem, 정상 구간 전력 하한). 포트가 1순위, 같은 날 핫플이 대조.
RESISTIVE = {
    "A": [("electric_kettle_2_fixed", 1000.0), ("hotplate_3_fixed", 400.0)],
    "B": [("electric_kettle_3_new", 1000.0), ("hotplate_4_new", 400.0)],
}
#: 장소 C 실측 s(p) 점을 뽑을 전력 대역
SITE_C_FILES = {
    "minipc": ("minipc_4C", [(8.5, 10), (11.5, 13.5), (15, 19), (21, 24), (27, 31)]),
    "laptop_charger": ("laptop_charger_5C", [(70, 74), (66, 70), (60, 66), (52, 58), (46, 52), (40, 46), (35, 40), (30, 35), (25, 30)]),
    "beam_projector": ("beam_projector_4C", [(49.3, 51.0), (47.5, 49.3)]),      # ON 은 49W 고정 — 두 대역은 앞/뒤 구간
}
#: 12.179.3 실측 T_B (단일 SMPS 창 실측 / 합성) — h3..h15 (|T|, ∠°). 참고용
MEASURED_TB = {
    "laptop_charger": [(1.01, 15), (0.92, 25), (0.84, 41), (0.77, 73), (1.21, 110), (2.37, 108), (3.63, 80)],
    "beam_projector": [(0.94, 9), (0.87, 15), (0.78, 26), (0.67, 58), (1.30, 100), (3.52, 77), (3.74, 17)],
}
#: 장소 C 실측 T_C (12.184.2 충전기 70W 옛 펌웨어, 12.184.8 미니PC 9.2W) — 분모는 장소 A sp_curves
MEASURED_TC = {
    "laptop_charger": (69.8, [(0.99, -7), (0.99, -14), (0.99, -19), (0.99, -21), (1.11, -21), (1.42, -27), (1.67, -47)]),
    "minipc": (9.2, [(0.93, -24), (0.90, -43), (0.89, -58), (0.80, -85), (0.74, -94), (0.77, -115), (0.59, -114)]),
}
RANGE_43 = {"C_dc/P": (0.3e-6, 2e-6), "R": (1.0, 3.0), "L": (20e-6, 500e-6), "Cx": (0.1e-6, 1e-6), "rd": (0.3, 3.0)}


def load_curves_canonical(path: str) -> dict:
    """sp_curves.npz 를 **정본 LOW 교정 규약**(2.62°)으로 읽는다 (12.184.13).

    곡선은 전부 LOW 레인지 옛 녹화(0.44°)라 h1 정규화 서명 s_h 가 정본보다 +2.18°×(h−1) 돌아 있다.
    read_raw_csv 가 원본 파일을 정본으로 돌려 주므로 곡선도 같은 규약으로 맞춰야 실측·시뮬 비교가 선다.
    """
    from src.preprocessing.file_registry import LOW_CAL_DEG_CANONICAL, LOW_CAL_DEG_LEGACY
    z = np.load(path)
    shift = np.deg2rad(LOW_CAL_DEG_CANONICAL - LOW_CAL_DEG_LEGACY)
    rot = np.exp(-1j * shift * np.arange(0, H))            # h1 정규화: 차수 h 는 (h−1)×shift
    out = {}
    for k in z.files:
        a = z[k]
        if k.endswith("__S") or k.endswith("__sb_S"):
            a = a * rot
        out[k] = a
    out["_canonical_shift_deg"] = float(np.degrees(shift))
    return out


def canon_sig(s: np.ndarray) -> np.ndarray:
    """옛 규약(LOW 0.44°)으로 읽은 h1 정규화 서명을 정본(2.62°)으로: s_h × e^{−j·2.18°·(h−1)} (LOW 전용 녹화에만)."""
    from src.preprocessing.file_registry import LOW_CAL_DEG_CANONICAL, LOW_CAL_DEG_LEGACY
    return s * np.exp(-1j * np.deg2rad(LOW_CAL_DEG_CANONICAL - LOW_CAL_DEG_LEGACY) * np.arange(0, len(s)))


def fdeg(z):
    return " ".join(f"{np.degrees(np.angle(v)):5.0f}" for v in z)


def fmag(z):
    return " ".join(f"{abs(v):5.2f}" for v in z)


def fpair(meas):
    return "∠ " + " ".join(f"{a:5.0f}" for _, a in meas) + "  |T| " + " ".join(f"{m:5.2f}" for m, _ in meas)


def odd_only(V: np.ndarray) -> np.ndarray:
    out = np.zeros_like(V)
    out[0::2] = V[0::2]
    return out


def ideal(V1: float) -> np.ndarray:
    return np.r_[complex(V1), np.zeros(H - 1)].astype(complex)


# ── [1][2] ───────────────────────────────────────────────────────────────────
def resistor_phasors(stem: str, p_min: float) -> Dict:
    cols = ["p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, H + 1)] \
        + [f"ihdeg{h}" for h in range(1, H + 1)] + [f"vh{h}" for h in range(1, H + 1)]
    df, _ = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    ok = (df["p_w"].to_numpy() > p_min) & (df["over_range"].to_numpy() == 0) & (df["range"].to_numpy() == 1)
    C = current_phasors(df)[ok]
    I = np.median(C.real, 0) + 1j * np.median(C.imag, 0)
    vh = np.array([np.median(df[f"vh{h}"].to_numpy(np.float64)[ok]) for h in range(1, H + 1)])
    return {"stem": stem, "n": int(ok.sum()), "p_w": float(np.median(df["p_w"].to_numpy()[ok])),
            "vrms": float(np.median(df["vrms"].to_numpy()[ok])), "I": I, "vh": vh,
            "V_from_I": np.abs(I) / abs(I[0]) * vh[0], "V": vh * np.exp(1j * np.angle(I))}


def part1_2(save: str) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    print("[1] 포트 전압계 검정 — 같은 장소·같은 날의 두 저항이 같은 전압 위상을 주는가 (기준 h3~h9 15° 안)")
    sites: Dict[str, np.ndarray] = {}
    trusted: Dict[str, bool] = {}
    for site, lst in RESISTIVE.items():
        rs = [resistor_phasors(s, p) for s, p in lst]
        for r in rs:
            print(f"  {site} {r['stem']:24s} n={r['n']:6d} P={r['p_w']:5.0f}W V={r['vrms']:5.1f} ihdeg1={np.degrees(np.angle(r['I'][0])):+5.2f}")
            print(f"      ∠V h3..15 (전류에서)  {fdeg(r['V'][ODD])}")
            print(f"      |V_h|/V1 %  전압채널  {' '.join(f'{100*x/r['vh'][0]:5.2f}' for x in r['vh'][ODD])}   h2 {100*r['vh'][1]/r['vh'][0]:.2f}")
            print(f"                 전류에서  {' '.join(f'{100*x/r['vh'][0]:5.2f}' for x in r['V_from_I'][ODD])}   h2 {100*r['V_from_I'][1]/r['vh'][0]:.2f}")
        d = np.degrees(np.angle(rs[0]["V"][ODD] / rs[1]["V"][ODD]))
        ok = bool(np.all(np.abs(d[:4]) < 15))
        trusted[site] = ok
        print(f"      두 저항의 ∠ 차 h3..15   {' '.join(f'{x:5.0f}' for x in d)}   -> {'통과' if ok else '**불합격** (HIGH 레인지 저항 고조파 0.3~1% 는 분해능·이음매 바닥)'}")
        sites[site] = rs[0]["V"]
    cols = ["p_w", "over_range", "range"] + [f"vh{h}" for h in range(1, H + 1)] + [f"vhdeg{h}" for h in range(1, H + 1)]
    df, _ = read_raw_csv("data/minipc_4C.csv", usecols=cols)
    Vc, st = voltage_phasors(df, (df["p_w"].to_numpy() > 8) & (df["p_w"].to_numpy() < 12))
    sites["C"] = Vc
    trusted["C"] = True
    print("\n[2] 장소 전압 페이저 (홀수차만 쓴다 — 짝수차는 [1] 의 h2 두 길이 6~20배 갈려 전압 채널 인공물)")
    print("         |V1|        " + "".join(f"h{h:<6d}" for h in (3, 5, 7, 9, 11, 13, 15)))
    for s, V in sites.items():
        print(f"  {s}  {abs(V[0]):6.1f}V |V| " + " ".join(f"{abs(V[k]):5.2f}" for k in ODD) + "   ∠ " + fdeg(V[ODD])
              + ("" if trusted[s] else "   (불신)"))
    np.savez(save, **{f"V_{s}": V for s, V in sites.items()}, trusted=json.dumps(trusted))
    print(f"  -> {save}")
    return sites, trusted


# ── [3] ──────────────────────────────────────────────────────────────────────
def site_c_points(dev: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    stem, bands = SITE_C_FILES[dev]
    cols = ["p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, H + 1)] \
        + [f"ihdeg{h}" for h in range(1, H + 1)] + [f"vh{h}" for h in range(1, H + 1)] + [f"vhdeg{h}" for h in range(1, H + 1)]
    df, info = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    pts = []
    for lo, hi in bands:
        sg = steady_signature(df, lo, hi)
        if sg["n"] < 200:
            continue
        V, _ = voltage_phasors(df, sg["mask"])
        pts.append((sg["p_w"], sg["s"], V))
    print(f"  {dev}: {stem} 에서 {len(pts)}점 (P {pts[0][0]:.1f}~{pts[-1][0]:.1f}W, 위상 복원 {info['phase_fix_deg_per_order']}°/차수)")
    return pts


def loss_pts(par, pts) -> float:
    e = 0.0
    for P, s_meas, V in pts:
        r = simulate(P, *par, vsrc=to_wave(odd_only(V)))
        if not r["ok"] or not np.all(np.isfinite(r["s"])):
            return 1e3
        d = np.r_[(r["s"][1:] - s_meas[1:]).real, (r["s"][1:] - s_meas[1:]).imag] * WW
        e += float(d @ d)
    return e / len(pts)


def loss_curve_ideal(par, dev: str, curves) -> float:
    P = curves[f"{dev}__P"]; S = curves[f"{dev}__S"]; Vr = curves[f"{dev}__V"]
    pts = [(float(P[i]), S[i] / S[i][0], ideal(float(Vr[i]))) for i in range(len(P))]
    return loss_pts(par, pts)


LO = np.log(np.array([10e-6, 0.3, 5e-6, 0.01e-6, 0.1]))
HI = np.log(np.array([500e-6, 10.0, 5000e-6, 2e-6, 10.0]))


def refit(dev: str, pts, n_starts: int = 3, seed: int = 0) -> Tuple[Tuple[float, ...], float]:
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    x0 = np.log(np.array(GUIDE_PARAMS[dev]))
    best = (tuple(GUIDE_PARAMS[dev]), loss_pts(GUIDE_PARAMS[dev], pts))

    def f(x):
        return loss_pts(tuple(np.exp(np.clip(x, LO, HI))), pts)
    starts = [x0] + [x0 + rng.normal(0, 0.4, 5) for _ in range(n_starts - 1)]
    for k, s in enumerate(starts):
        r = minimize(f, s, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-6})
        print(f"    start {k}: loss {r.fun:.4f} ({r.nit} it)")
        if r.fun < best[1]:
            best = (tuple(float(v) for v in np.exp(np.clip(r.x, LO, HI))), float(r.fun))
    return best


def check_43(dev: str, par, p_nom: float) -> str:
    c, r, l, cx, rd = par
    bad = []
    if not RANGE_43["C_dc/P"][0] <= c / p_nom <= RANGE_43["C_dc/P"][1]: bad.append(f"C_dc/P {c/p_nom*1e6:.2f}µF/W")
    if not RANGE_43["R"][0] <= r <= RANGE_43["R"][1]: bad.append(f"R {r:.2f}Ω")
    if l > RANGE_43["L"][1]: bad.append(f"L {l*1e6:.0f}µH(초크?)")
    if not RANGE_43["Cx"][0] <= cx <= RANGE_43["Cx"][1]: bad.append(f"Cx {cx*1e6:.2f}µF")
    if not RANGE_43["rd"][0] <= rd <= RANGE_43["rd"][1]: bad.append(f"rd {rd:.2f}Ω")
    return "§4.3 안" if not bad else "§4.3 밖: " + ", ".join(bad)


def part3(curves, out: str) -> Dict[str, Tuple[float, ...]]:
    print("\n[3] 장소 C 재피팅 — 실측 파형(vhdeg)으로 5파라미터를 다시 맞춘다 (게이트 4 장소 C 판: 손실 < 0.01)")
    fitted: Dict[str, Tuple[float, ...]] = dict(GUIDE_PARAMS)
    rows = {}
    for dev in ("minipc", "laptop_charger", "beam_projector"):
        import os
        if not os.path.exists(f"data/{SITE_C_FILES[dev][0]}.csv"):
            continue
        pts = site_c_points(dev)
        g = GUIDE_PARAMS[dev]
        l_g = loss_pts(g, pts)
        print(f"    가이드 파라미터의 장소 C 손실 {l_g:.4f}  (장소 A 이상정현파 손실 {loss_curve_ideal(g, dev, curves):.4f})")
        par, l_fit = refit(dev, pts)
        fitted[dev] = par
        p_nom = {"laptop_charger": 65.0, "minipc": 30.0, "beam_projector": 65.0}[dev]
        print(f"    재피팅 손실 {l_fit:.4f}   C_dc {par[0]*1e6:.1f}µF  R {par[1]:.2f}Ω  L {par[2]*1e6:.0f}µH  Cx {par[3]*1e6:.2f}µF  rd {par[4]:.2f}Ω"
              f"   [{check_43(dev, par, p_nom)}]")
        print(f"      (가이드:            C_dc {g[0]*1e6:.1f}µF  R {g[1]:.2f}Ω  L {g[2]*1e6:.0f}µH  Cx {g[3]*1e6:.2f}µF  rd {g[4]:.2f}Ω)")
        # 점별 잔차: 어느 전력·차수가 남나
        for P, s_meas, V in pts[::max(1, len(pts) // 3)]:
            s = simulate(P, *par, vsrc=to_wave(odd_only(V)))["s"]
            print(f"      P={P:5.1f}  실측 ∠h3..15 {fdeg(s_meas[ODD])} |s| {fmag(s_meas[ODD])}")
            print(f"             시뮬 ∠h3..15 {fdeg(s[ODD])} |s| {fmag(s[ODD])}")
        rows[dev] = {"guide": list(g), "fit_siteC_wave": list(par), "loss_guide_siteC": l_g, "loss_fit_siteC": l_fit,
                     "points": [(float(P), float(abs(V[0]))) for P, _, V in pts]}
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
    print(f"  -> {out}")
    return fitted


# ── [4] ──────────────────────────────────────────────────────────────────────
def part4(sites, trusted, curves, params: Dict[str, Dict[str, Tuple[float, ...]]]):
    print("\n[4a] 전이 — 그 파라미터로 장소 A 곡선(sp_curves)을 **이상 정현파**로 낼 때의 손실 (기준 < 0.02)")
    for dev in ("minipc", "laptop_charger", "beam_projector"):
        print(f"  {dev:15s} " + "   ".join(f"{lab}: {loss_curve_ideal(ps[dev], dev, curves):.4f}" for lab, ps in params.items()))
    print("\n[4b] 예측 T_C = s(V_C)/s(이상 정현파) vs 실측 T_C (분모 = 장소 A 곡선). h3..h15")
    for dev, (P, meas) in MEASURED_TC.items():
        pts = {p: V for p, _, V in site_c_points(dev)}
        Pk = min(pts, key=lambda p: abs(p - P)); Vc = pts[Pk]
        print(f"  {dev} P={P:.0f}W (V_C 는 {Pk:.1f}W 대역의 것)")
        print(f"    실측                     {fpair(meas)}")
        for lab, ps in params.items():
            a = simulate(P, *ps[dev], vsrc=to_wave(ideal(abs(Vc[0]))))["s"]
            b = simulate(P, *ps[dev], vsrc=to_wave(odd_only(Vc)))["s"]
            T = b / a
            print(f"    시뮬 {lab:12s}        ∠ {fdeg(T[ODD])}  |T| {fmag(T[ODD])}")
    import os
    if os.path.exists("data/beam_projector_4C.csv"):
        pts = site_c_points("beam_projector")
        P, s_meas, Vc = pts[0]
        Pa = curves["beam_projector__P"]; Sa = curves["beam_projector__S"]; i = int(np.argmin(np.abs(Pa - P))); sa = Sa[i] / Sa[i][0]
        T = s_meas / sa
        print(f"  beam_projector P={P:.1f}W (분모 장소 A {Pa[i]:.1f}W, 두 쪽 다 정본 규약)")
        print(f"    실측 T_C                   ∠ {fdeg(T[ODD])}  |T| {fmag(T[ODD])}")
        for lab, ps in params.items():
            a = simulate(P, *ps["beam_projector"], vsrc=to_wave(ideal(abs(Vc[0]))))["s"]
            b = simulate(P, *ps["beam_projector"], vsrc=to_wave(odd_only(Vc)))["s"]
            Ts = b / a
            print(f"    시뮬 {lab:12s}        ∠ {fdeg(Ts[ODD])}  |T| {fmag(Ts[ODD])}")
    print("\n[4c] 장소 B (참고 — V_B 는 [1] 을 " + ("통과" if trusted.get("B") else "**못 넘었다**: 포트·핫플이 h9/h13 만 맞고 h3~h7 이 갈린다") + ")")
    VB = sites["B"]; VA = sites["A"]
    for dev, meas in MEASURED_TB.items():
        P = float(curves[f"{dev}__P"][-1]) if dev == "laptop_charger" else float(np.median(curves[f"{dev}__P"]))
        print(f"  {dev} P={P:.0f}W   실측 T_B (12.179.3)   {fpair(meas)}")
        for lab, ps in params.items():
            for vlab, Vb in (("V_B h9,11,13 만", np.array([v if k in (8, 10, 12) or k == 0 else 0 for k, v in enumerate(VB)], complex)),
                             ("V_B 홀수 전부", odd_only(VB))):
                a = simulate(P, *ps[dev], vsrc=to_wave(ideal(abs(Vb[0]))))["s"]
                b = simulate(P, *ps[dev], vsrc=to_wave(Vb))["s"]
                T = b / a
                print(f"    시뮬 {lab:12s} {vlab:15s} ∠ {fdeg(T[ODD])}  |T| {fmag(T[ODD])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curves", default="processed_data/sp_curves.npz")
    ap.add_argument("--save", default="results/_site_voltage.npz")
    ap.add_argument("--params-out", default="results/_circuit_params_C.json")
    ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args()
    curves = load_curves_canonical(a.curves)
    print(f"(sp_curves 를 정본 LOW 규약으로 회전: +{curves['_canonical_shift_deg']:.2f}°×(h−1) 제거)")
    sites, trusted = part1_2(a.save)
    params = {"가이드 §4.2": dict(GUIDE_PARAMS)}
    if not a.no_fit:
        params["C파형 재피팅"] = part3(curves, a.params_out)
    part4(sites, trusted, curves, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
