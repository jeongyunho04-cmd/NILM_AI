# -*- coding: utf-8 -*-
"""순저항 부하로 전압 채널을 검정한다 — I_h/V_h = 1/R 이 모든 차수에서 성립하는가 (12.184.11).

    python -X utf8 -m src.run_vh_calib_probe                                  # 장소 C 포트(HIGH) + 드라이기 강 대조
    python -X utf8 -m src.run_vh_calib_probe --low data/bulb_C.csv --low-pmin 40  # ①′ LOW 레인지 순저항이 오면

[원리] 순저항이면 I_h = V_h/R 이므로 `|I_h/V_h|·R = 1`, `ihdeg_h − vhdeg_h = 0` 이 h1~h15 에서 성립해야 한다.
어긋나면 전류 채널이나 전압 채널 중 하나가 틀린 것이다. 어느 쪽인지는 이 검정만으로는 못 가른다 — 단서 둘:
  · 짝수차: 저항 전류의 짝수차는 0 이므로 전압 채널의 짝수차 값 자체가 **전압 채널 인공물**의 크기·위상이다.
  · HIGH 레인지(≥2.9A)는 전류가 LOW/HIGH 두 ADC 경로를 표본마다 이어 붙인 것이라 이음매 인공물(0.3~0.6%)이
    실릴 수 있다. LOW 레인지 순저항(백열전구·납땜인두)은 그것이 없어 전압 채널을 바로 판정한다 (--low).
[출력] 차수별 비 표, 짝수차 인공물 표, 계단에서 Z. results/_vh_check_<stem>.npz
"""
from typing import Dict
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np
import pandas as pd

from src.preprocessing.raw_csv import read_raw_csv
from src.preprocessing.raw_phasors import current_phasors

H = 15
COLS = ["t_s", "p_w", "irms", "vrms", "over_range", "range", "cycle", "pll_locked"] \
    + [f"ih{h}" for h in range(1, H + 1)] + [f"ihdeg{h}" for h in range(1, H + 1)] \
    + [f"vh{h}" for h in range(1, H + 1)] + [f"vhdeg{h}" for h in range(1, H + 1)]
ODD = [2, 4, 6, 8, 10, 12, 14]
EVEN = [1, 3, 5, 7, 9, 11, 13]


def load(path: str):
    df, _ = read_raw_csv(path, usecols=COLS)
    return df


def steady_mask(df, p_min: float, p_max: float = 1e9, rng=None) -> np.ndarray:
    p = df["p_w"].to_numpy(np.float64)
    ps = pd.Series(p).rolling(31, center=True, min_periods=1).median().to_numpy()
    m = (ps > p_min) & (ps < p_max) & (np.abs(p - ps) < 0.1 * ps + 20) \
        & (df["over_range"].to_numpy() == 0) & (df["pll_locked"].to_numpy() == 1)
    if rng is not None:
        m &= df["range"].to_numpy() == rng
    return m


def ratio_table(df, mask, label: str) -> Dict:
    C = current_phasors(df)[mask]
    Vh = np.stack([df[f"vh{h}"].to_numpy(np.float64) * np.exp(1j * np.deg2rad(df[f"vhdeg{h}"].to_numpy(np.float64)))
                   for h in range(1, H + 1)], 1)[mask]
    Y = C / Vh
    Ym = np.median(Y.real, 0) + 1j * np.median(Y.imag, 0)
    R1 = 1.0 / abs(Ym[0])
    scat = np.array([np.degrees(np.sqrt(max(-2 * np.log(max(abs(np.mean(np.exp(1j * np.angle(Y[:, k])))), 1e-12)), 0))) for k in range(H)])
    I_rel = np.median(np.abs(C), 0) / np.median(np.abs(C[:, 0]))
    V_rel = np.median(np.abs(Vh), 0) / np.median(np.abs(Vh[:, 0]))
    print(f"\n[{label}] n={mask.sum()}  I={np.median(df['irms'].to_numpy()[mask]):.2f}A  P={np.median(df['p_w'].to_numpy()[mask]):.0f}W  "
          f"V={np.median(df['vrms'].to_numpy()[mask]):.1f}  R(h1)={R1:.2f}Ω  레인지 {dict(pd.Series(df['range'].to_numpy()[mask]).value_counts())}")
    print("   h          " + " ".join(f"{h:6d}" for h in range(1, H + 1)))
    print("   |Y_h|·R    " + " ".join(f"{abs(v)*R1:6.2f}" for v in Ym) + "    <- 저항이면 1")
    print("   ∠I−∠V °    " + " ".join(f"{np.degrees(np.angle(v)):6.0f}" for v in Ym) + "    <- 저항이면 0")
    print("   ∠ 산포 °    " + " ".join(f"{c:6.0f}" for c in scat))
    print("   |I_h|/I1 % " + " ".join(f"{100*x:6.2f}" for x in I_rel) + "    (전류가 말하는 전압 고조파)")
    print("   |V_h|/V1 % " + " ".join(f"{100*x:6.2f}" for x in V_rel) + "    (전압 채널)")
    # 채널 차이 벡터 (V1 대비 %): 전압채널 − 전류에서 되돌린 값
    Vc = np.median(Vh, 0) if False else (np.median(Vh.real, 0) + 1j * np.median(Vh.imag, 0))
    Vi = (np.median(C.real, 0) + 1j * np.median(C.imag, 0)) * R1          # 전류 -> 전압 (저항 가정)
    diff = np.abs(Vc - Vi) / abs(Vc[0]) * 100
    print("   |채널차|/V1 %" + " ".join(f"{x:6.2f}" for x in diff) + "    (전압채널 − 전류환산)")
    return {"Y": Ym, "R1": R1, "scatter_deg": scat, "I_rel": I_rel, "V_rel": V_rel, "diff_pct": diff, "n": int(mask.sum())}


def even_artifact(t: Dict):
    print("\n[짝수차 = 전압 채널 인공물] 저항 전류의 짝수차는 0 이므로 전압 채널의 짝수차가 곧 인공물이다")
    print("   h          " + " ".join(f"{h+1:6d}" for h in EVEN))
    print("   전압채널 %  " + " ".join(f"{100*t['V_rel'][k]:6.2f}" for k in EVEN))
    print("   전류에서 %  " + " ".join(f"{100*t['I_rel'][k]:6.2f}" for k in EVEN) + "    <- 바닥(잡음)")
    print("   홀수차 채널차 %" + " ".join(f"{t['diff_pct'][k]:6.2f}" for k in ODD) + "   (h3..h15: 두 채널이 서로 다른 만큼)")


def line_impedance(df, p_on: float):
    t = df["t_s"].to_numpy(); p = df["p_w"].to_numpy(np.float64); i = df["irms"].to_numpy(); v = df["vrms"].to_numpy()
    ps = pd.Series(p).rolling(61, center=True, min_periods=1).median().to_numpy()
    on = (ps > 0.5 * p_on).astype(int)
    edges = np.where(np.diff(on) != 0)[0]
    zs = []
    for e in edges:
        a = (t > t[e] - 12) & (t < t[e] - 3) & (np.abs(p - ps) < 0.2 * ps + 20)
        b = (t > t[e] + 3) & (t < t[e] + 12) & (np.abs(p - ps) < 0.2 * ps + 20)
        if a.sum() < 60 or b.sum() < 60:
            continue
        dI = np.median(i[b]) - np.median(i[a]); dV = np.median(v[b]) - np.median(v[a])
        if abs(dI) > 1.0:
            zs.append(-dV / dI)
            print(f"   t={t[e]:6.1f}s  ΔI {dI:+.2f}A  ΔVrms {dV:+.2f}V  Z={-dV/dI:.3f}Ω")
    if zs:
        print(f"   Z 중앙 {np.median(zs):.3f} Ω  ({len(zs)}계단; 장소 A 1.470 / B 0.907)")
    return zs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kettle", default="data/electric_kettle_4C.csv")
    ap.add_argument("--kettle-pmin", type=float, default=1300.0)
    ap.add_argument("--dryer", default="data/hair_dryer_4C.csv", help="대조 (모터 섞임). '' 이면 생략")
    ap.add_argument("--low", default="", help="①′ LOW 레인지 순저항 녹화 (백열전구·납땜인두)")
    ap.add_argument("--low-pmin", type=float, default=30.0)
    a = ap.parse_args()
    out = {}
    df = load(a.kettle)
    t = ratio_table(df, steady_mask(df, a.kettle_pmin), f"포트 {a.kettle} HIGH")
    even_artifact(t)
    print("\n[선로 임피던스] 포트 계단")
    zs = line_impedance(df, a.kettle_pmin)
    out["kettle"] = {**t, "Z": zs}
    if a.dryer:
        dd = load(a.dryer)
        out["dryer"] = ratio_table(dd, steady_mask(dd, 800, 1400), f"드라이기 강 {a.dryer} HIGH (모터 섞임 — 대조)")
    if a.low:
        dl = load(a.low)
        out["low"] = ratio_table(dl, steady_mask(dl, a.low_pmin, rng=0), f"LOW 레인지 순저항 {a.low}")
        print("   -> LOW 레인지는 이음매가 없으므로 위 표의 어긋남은 전압 채널의 것이다. 홀수차 (|Y|·R, ∠) 가 전압 채널 보정표다.")
    stem = a.kettle.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
    np.savez(f"results/_vh_check_{stem}.npz", **{f"{k}_{kk}": np.asarray(vv) for k, d in out.items() for kk, vv in d.items() if kk != "Z"})
    print(f"\n-> results/_vh_check_{stem}.npz")
    print("\n판정: 포트(순저항)에서 두 채널이 h3 는 V1 의 ~3%, h5~h15 는 0.3~0.6% 어긋난다. 짝수차 인공물(h2 2.5%, h4 1.5%, h6 0.5%)이"
          " 전압 채널에 실재하므로 홀수차에도 비슷한 크기의 인공물이 있다고 봐야 하고, HIGH 레인지 전류의 이음매 인공물도 같은 크기다."
          " 어느 쪽인지는 LOW 레인지 순저항(--low)이 가른다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
