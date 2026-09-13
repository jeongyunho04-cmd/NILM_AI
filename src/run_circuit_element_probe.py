# -*- coding: utf-8 -*-
"""§3.3 누락 요소 검정 — 장소 C 재피팅 잔차(h13~h15, 중부하)를 과녁으로 요소 하나씩 켜 본다 (12.184.10).

    python -X utf8 -m src.run_circuit_element_probe              # 전부 (약 3분)
    python -X utf8 -m src.run_circuit_element_probe --devices minipc

변형 (각각 장소 C 실측 파형으로 재피팅, 시작점은 12.184.9 의 C 재피팅 값):
    base  5파라미터, V 홀수차 전부                       <- 기준
    E0    5파라미터, V 를 h1~h7 로 제한 (입력 민감도)     E0b h1~h9, E0c h1~h11
    E1    + nvt  지수 다이오드 무릎  v = Vf + rd·i + nvt·ln(1+i/10mA)
    E2    + Rp   초크 병렬 감쇠 저항 (L ∥ Rp)
    E3    + α    직류 부하 지수  i = (P/V0)(V0/v_c)^α  (1 정전력, 0 정전류, −1 저항)
    E4    + k_hi h9 이상 전압 크기 배율 (전압 채널 고차 신뢰도 — 시뮬 인수가 아니라 입력 정형)
    E5    포트(순저항)의 비 c_h = Y_h·R 를 전압 채널 보정으로 걸고 재피팅 (--kettle-check). 12.184.12: 나빠진다 -> 전압 오차 아님
    E6    차수별 복소 채널 보정 14개를 첫 기기에서 피팅, 둘째 기기에 고정해 5개만 재피팅 (교차 검증). 12.184.12: 안 옮는다
지표: 손실(가이드 §4.1), res13_15 = 중부하 점들의 |Δs13|+|Δs15| 평균, §4.3 범위, 예측 T_C(미니PC 9W·충전기 70W)와
충전기 T_B(h9,11,13). 판정 기준은 results/_criteria_circuit.md (추가분). 결과는 설계 12.184.10.
⚠ 실측 T_C/T_B 의 분모는 장소 A 곡선(파형 미상)이고 시뮬의 분모는 이상 정현파다 — 장소 A 가 정현파가 아니므로
(12.184.9 [4a]) 이 비교에는 장소 A 왜곡만큼의 편향이 들어 있다. ④(장소 A vhdeg)가 오기 전엔 방향·추세만 본다.
"""
from typing import Dict, List, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.run_site_voltage_probe import (MEASURED_TB, MEASURED_TC, ODD, WW, check_43, fdeg, fmag, fpair,
                                        ideal, odd_only, site_c_points)
from src.synthesis.circuit_sim import GUIDE_PARAMS, simulate, to_wave

#: 중부하 점 (res13_15 의 과녁): 미니PC 17W 이상, 충전기 45W 이하
MID = {"minipc": lambda P: P >= 17.0, "laptop_charger": lambda P: P <= 45.0}
#: 확장 파라미터의 (log 여부, 하한, 상한, 시작값)
EXTRA = {"nvt": (True, 0.005, 0.5, 0.05), "Rp": (True, 1.0, 500.0, 30.0), "alpha": (False, -1.0, 1.0, 0.5),
         "k_hi": (False, 0.0, 1.5, 0.5),     # k_hi: h9 이상 전압 크기 배율 (전압 채널 고차 신뢰도, 시뮬 인수가 아니라 입력 정형)
         **{f"ck{h}": (True, 0.2, 5.0, 1.0) for h in (3, 5, 7, 9, 11, 13, 15)},     # E6 차수별 크기 보정
         **{f"cp{h}": (False, -180.0, 180.0, 0.0) for h in (3, 5, 7, 9, 11, 13, 15)}}  # E6 차수별 위상 보정 [°]
LO5 = np.log(np.array([10e-6, 0.3, 5e-6, 0.01e-6, 0.1]))
HI5 = np.log(np.array([500e-6, 10.0, 5000e-6, 2e-6, 10.0]))


def limit_orders(V: np.ndarray, hmax: int, k_hi: float = 1.0) -> np.ndarray:
    out = odd_only(V)
    out[hmax:] = 0
    if k_hi != 1.0:
        out[8:] = out[8:] * k_hi          # h9 이상
    return out


KETTLE_CORR = None      # (15,) complex: 포트가 말하는 전압 채널 보정 c_h = Y_h·R (E5). None 이면 안 건다
CHAN_CORR = None        # (15,) complex: 교차 검증용 채널 보정 (E6). None 이면 안 건다


def sim_s(par5, extras: Dict[str, float], P: float, V: np.ndarray):
    ex = {k: v for k, v in extras.items() if k not in ("k_hi",) and not k.startswith("c")}
    V = V.copy()
    if "k_hi" in extras:
        V[8:] = V[8:] * extras["k_hi"]
    if KETTLE_CORR is not None:
        V[2::2] = V[2::2] * KETTLE_CORR[2::2]          # 홀수차 h3.. 에 포트 비를 곱한다 (h1 은 기준)
    if CHAN_CORR is not None:
        V[2::2] = V[2::2] * CHAN_CORR[2::2]
    # E6: 차수별 복소 보정 (ck{h}=크기, cp{h}=위상°) 를 피팅 변수로
    for k in range(2, 15, 2):
        h = k + 1
        if f"ck{h}" in extras:
            V[k] = V[k] * extras[f"ck{h}"] * np.exp(1j * np.deg2rad(extras.get(f"cp{h}", 0.0)))
    r = simulate(P, *par5, vsrc=to_wave(V), **ex)
    return r["s"] if r["ok"] and np.all(np.isfinite(r["s"])) else None


def metrics(par5, extras, pts, dev: str, vmax: int = 15) -> Tuple[float, float]:
    e = 0.0
    res = []
    for P, s_meas, V in pts:
        s = sim_s(par5, extras, P, limit_orders(V, vmax))
        if s is None:
            return 1e3, 1e3
        d = np.r_[(s[1:] - s_meas[1:]).real, (s[1:] - s_meas[1:]).imag] * WW
        e += float(d @ d)
        if MID[dev](P):
            res.append(abs(s[12] - s_meas[12]) + abs(s[14] - s_meas[14]))
    return e / len(pts), float(np.mean(res)) if res else np.nan


def refit(dev: str, pts, x0_5: np.ndarray, extra_names: List[str], vmax: int, n_starts: int = 3, seed: int = 0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    lo = list(LO5); hi = list(HI5); x0 = list(x0_5)
    for n in extra_names:
        islog, a, b, s0 = EXTRA[n]
        lo.append(np.log(a) if islog else a); hi.append(np.log(b) if islog else b); x0.append(np.log(s0) if islog else s0)
    lo = np.array(lo); hi = np.array(hi); x0 = np.array(x0)

    def unpack(x):
        x = np.clip(x, lo, hi)
        par5 = tuple(float(v) for v in np.exp(x[:5]))
        ex = {}
        for i, n in enumerate(extra_names):
            islog = EXTRA[n][0]
            ex[n] = float(np.exp(x[5 + i]) if islog else x[5 + i])
        return par5, ex

    def f(x):
        par5, ex = unpack(x)
        return metrics(par5, ex, pts, dev, vmax)[0]
    best = (None, None, 1e9)
    starts = [x0] + [x0 + rng.normal(0, 0.3, len(x0)) for _ in range(n_starts - 1)]
    # 초기 심플렉스 폭을 명시한다 — 0 에서 시작하는 변수(log 1 = 0, 위상 0°)는 Nelder-Mead 기본 폭(0.00025)으로는 안 움직인다
    step = np.array([0.15] * 5 + [(10.0 if n.startswith("cp") else 0.2 if n in ("alpha", "k_hi") else 0.15) for n in extra_names])
    for s in starts:
        simplex = np.vstack([s] + [s + np.eye(len(s))[i] * step[i] for i in range(len(s))])
        r = minimize(f, s, method="Nelder-Mead", options={"maxiter": 900 if extra_names else 700, "xatol": 1e-3, "fatol": 1e-6,
                                                          "initial_simplex": simplex})
        if r.fun < best[2]:
            p5, ex = unpack(r.x)
            best = (p5, ex, float(r.fun))
    return best


def predictions(dev: str, par5, extras, sites) -> str:
    out = []
    if dev in MEASURED_TC:
        P, meas = MEASURED_TC[dev]
        Vc = sites["C_pts"][dev]
        a = sim_s(par5, extras, P, ideal(abs(Vc[0]))); b = sim_s(par5, extras, P, odd_only(Vc))
        if a is not None and b is not None:
            T = b / a
            out.append(f"      T_C  실측 {fpair(meas)}")
            out.append(f"           시뮬 ∠ {fdeg(T[ODD])}  |T| {fmag(T[ODD])}")
    if dev in MEASURED_TB:
        meas = MEASURED_TB[dev]; VB = sites["V_B"]; P = 68.6 if dev == "laptop_charger" else 48.7
        Vb = np.array([v if k in (0, 8, 10, 12) else 0 for k, v in enumerate(VB)], complex)
        a = sim_s(par5, extras, P, ideal(abs(Vb[0]))); b = sim_s(par5, extras, P, Vb)
        if a is not None and b is not None:
            T = b / a
            out.append(f"      T_B  실측 {fpair(meas)}   (V_B h9,11,13 만)")
            out.append(f"           시뮬 ∠ {fdeg(T[ODD])}  |T| {fmag(T[ODD])}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--devices", nargs="+", default=["minipc", "laptop_charger"])
    ap.add_argument("--params", default="results/_circuit_params_C.json")
    ap.add_argument("--site-voltage", default="results/_site_voltage.npz")
    ap.add_argument("--out", default="results/_circuit_elements_C.json")
    ap.add_argument("--variants", nargs="*", default=None, help="이름 접두로 고른다 (예: base E0b E4)")
    ap.add_argument("--kettle-check", default="results/_vh_check_electric_kettle_4C.npz", help="E5: 포트 비 (run_vh_calib_probe)")
    a = ap.parse_args()
    base = json.load(open(a.params, encoding="utf-8"))
    sv = np.load(a.site_voltage, allow_pickle=True)
    sites = {"V_B": sv["V_B"], "C_pts": {}}
    CORR_NAMES = [f"ck{h}" for h in (3, 5, 7, 9, 11, 13, 15)] + [f"cp{h}" for h in (3, 5, 7, 9, 11, 13, 15)]
    all_variants = [("base", [], 15), ("E0 V h1~h7", [], 7), ("E0b V h1~h9", [], 9), ("E0c V h1~h11", [], 11),
                    ("E1 +nvt", ["nvt"], 15), ("E2 +Rp", ["Rp"], 15), ("E3 +alpha", ["alpha"], 15), ("E4 +k_hi", ["k_hi"], 15),
                    ("E5 포트보정V", [], 15), ("E6 채널보정피팅", CORR_NAMES, 15)]
    variants = [v for v in all_variants if not a.variants or any(v[0].startswith(x) for x in a.variants)]
    report = {}
    global KETTLE_CORR, CHAN_CORR
    kc = None
    if a.kettle_check:
        z = np.load(a.kettle_check)
        kc = z["kettle_Y"] * float(z["kettle_R1"])          # c_h = Y_h·R (h1 = 1)
    for dev in a.devices:
        pts = site_c_points(dev)
        P0, _, V0 = min(pts, key=lambda t: abs(t[0] - (9.2 if dev == "minipc" else 69.8)))
        sites["C_pts"][dev] = V0
        x0 = np.log(np.array(base[dev]["fit_siteC_wave"]))
        p_nom = 65.0 if dev == "laptop_charger" else 30.0
        print(f"\n== {dev}  (중부하 점: {[round(P,1) for P,_,_ in pts if MID[dev](P)]})")
        ref = None
        for lab, names, vmax in variants:
            KETTLE_CORR = kc if lab.startswith("E5") else None
            if not lab.startswith("E6"):
                CHAN_CORR_saved, CHAN_CORR = CHAN_CORR, None
            if lab.startswith("E5") and kc is None:
                continue
            if lab.startswith("E6"):
                if dev == a.devices[0]:
                    p5, ex, l_ = refit(dev, pts, x0, names, vmax, n_starts=2)
                    CHAN_CORR = np.ones(15, complex)
                    for h in (3, 5, 7, 9, 11, 13, 15):
                        CHAN_CORR[h - 1] = ex[f"ck{h}"] * np.exp(1j * np.deg2rad(ex[f"cp{h}"]))
                    ex = {}
                    print(f"    (E6 보정표, {dev} 에서 맞춤)  |c| h3..15 {fmag(CHAN_CORR[ODD])}   ∠c {fdeg(CHAN_CORR[ODD])}")
                    p5, ex, l_ = refit(dev, pts, np.log(np.array(p5)), [], vmax, n_starts=1)
                else:
                    if CHAN_CORR is None:
                        continue
                    p5, ex, l_ = refit(dev, pts, x0, [], vmax)      # 다른 기기: 보정표 고정, 5개만 다시
                    lab = "E6 채널보정(교차)"
            else:
                p5, ex, l_ = refit(dev, pts, x0, names, vmax)
            _, res = metrics(p5, ex, pts, dev, vmax)
            if ref is None:
                ref = (l_, res)
            exs = "  ".join(f"{k}={v:.3g}" for k, v in ex.items())
            print(f"  {lab:12s} 손실 {l_:.4f} ({l_/ref[0]*100:4.0f}%)  res13_15 {res:.3f} ({res/ref[1]*100:4.0f}%)  "
                  f"C_dc {p5[0]*1e6:.1f}µF R {p5[1]:.2f}Ω L {p5[2]*1e6:.0f}µH Cx {p5[3]*1e6:.2f}µF rd {p5[4]:.2f}Ω {exs}  [{check_43(dev, p5, p_nom)}]")
            # 중부하 점의 h13/h15 실측 vs 시뮬
            for P, s_meas, V in pts:
                if MID[dev](P):
                    s = sim_s(p5, ex, P, limit_orders(V, vmax))
                    print(f"      P={P:5.1f} h13 실측 {abs(s_meas[12]):.2f}∠{np.degrees(np.angle(s_meas[12])):4.0f} 시뮬 {abs(s[12]):.2f}∠{np.degrees(np.angle(s[12])):4.0f}"
                          f" | h15 실측 {abs(s_meas[14]):.2f}∠{np.degrees(np.angle(s_meas[14])):4.0f} 시뮬 {abs(s[14]):.2f}∠{np.degrees(np.angle(s[14])):4.0f}")
            print(predictions(dev, p5, ex, sites))
            report.setdefault(dev, {})[lab] = {"par5": list(p5), "extras": ex, "loss": l_, "res13_15": res, "vmax": vmax}
            if not lab.startswith("E6"):
                CHAN_CORR = CHAN_CORR_saved
    import os
    if os.path.exists(a.out):
        old = json.load(open(a.out, encoding="utf-8"))
        for dev, d in report.items():
            old.setdefault(dev, {}).update(d)
        report = old
    json.dump(report, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
