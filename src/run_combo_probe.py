# -*- coding: utf-8 -*-
"""SMPS 동시 녹화 검정 — 합의 검정(계단 델타 vs 단독 서명)과 포트 on/off 가 SMPS 고조파에 미치는 것 (12.184.15).

    python -X utf8 -m src.run_combo_probe --file data/test_19.csv          # 계단 델타 vs 단독 서명
    python -X utf8 -m src.run_combo_probe --file data/test_20.csv          # + 포트 창: (총전류 − 포트) vs SMPS 만

[델타 검정] 전력 계단(|ΔP| ≥ 6W)마다 앞·뒤 정상 구간의 절대 페이저 차 Δ_h 를 뽑는다. 다른 기기가 그대로면
Δ_h 는 켜지거나 꺼진 기기의 페이저다. 같은 장소·같은 날의 단독 녹화(충전기 5C, 미니PC 4C, 프로젝터 4C)에서
같은 전력의 절대 페이저를 가져와 |Δ_h/I_ref,h| 와 ∠ 차를 차수별로 놓는다. 1·0° 면 **중첩이 성립**(기기 간
결합 없음), 계통적으로 벗어나면 결합이거나 측정 교차 오염이다. ΔP 로 기기를 고른다 (프로젝터 ≈ 49W,
미니PC 6~16W, 충전기 나머지).
[포트 창] 포트가 켜진 창(HIGH)의 총전류에서 포트 단독 페이저(electric_kettle_4C, 같은 장소·같은 날)를 빼면
SMPS 만 남아야 한다. 포트가 꺼진 창(LOW)의 SMPS 페이저와 차수별로 비교한다 — 차이는 (a) LOW/HIGH 교정값
차(옛 규약이면 −2.18°×h 점프), (b) HIGH 레인지 전류 인공물(12.184.12, 6A 의 0.5~3%), (c) 실제 결합의 합이다.
모든 파일은 read_raw_csv 로 정본 규약으로 읽는다 (옛 녹화는 range==0 사이클 −2.18°×h).
"""
from typing import Dict, List, Tuple
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
ODD = [2, 4, 6, 8, 10, 12, 14]
COLS = ["t_s", "p_w", "irms", "vrms", "over_range", "range", "pll_locked"] \
    + [f"ih{h}" for h in range(1, H + 1)] + [f"ihdeg{h}" for h in range(1, H + 1)]
REF = {"laptop_charger": "laptop_charger_5C", "minipc": "minipc_4C", "beam_projector": "beam_projector_4C",
       "electiric_kettle": "electric_kettle_4C"}


def fdeg(z):
    return " ".join(f"{np.degrees(np.angle(v)):5.0f}" for v in z)


def fmag(z):
    return " ".join(f"{abs(v):5.2f}" for v in z)


def load(stem_or_path: str):
    p = stem_or_path if stem_or_path.endswith(".csv") else f"data/{stem_or_path}.csv"
    df, info = read_raw_csv(p, usecols=COLS)
    return df, info


def med_phasor(C: np.ndarray) -> np.ndarray:
    return np.median(C.real, 0) + 1j * np.median(C.imag, 0)


def ref_phasor(dev: str, P: float, tol: float) -> Tuple[np.ndarray, int, float]:
    """단독 녹화에서 전력 P±tol 의 절대 페이저 중앙값."""
    df, _ = load(REF[dev])
    p = df["p_w"].to_numpy(np.float64)
    ps = pd.Series(p).rolling(31, center=True, min_periods=1).median().to_numpy()
    m = (np.abs(ps - P) <= tol) & (np.abs(p - ps) < 0.1 * ps + 3) & (df["over_range"].to_numpy() == 0)
    if m.sum() < 120:
        return None, int(m.sum()), np.nan
    C = current_phasors(df)[m]
    return med_phasor(C), int(m.sum()), float(np.median(p[m]))


def steps(df, min_dp: float = 6.0, pre=(3.0, 10.0), post=(3.0, 10.0)) -> List[Dict]:
    t = df["t_s"].to_numpy(); p = df["p_w"].to_numpy(np.float64)
    ps = pd.Series(p).rolling(61, center=True, min_periods=1).median().to_numpy()
    d = ps[60:] - ps[:-60]                       # 1초 간격 차
    cand = np.where(np.abs(d) > min_dp)[0] + 30
    out = []
    last = -1e9
    for i in cand:
        if t[i] - last < 8.0:
            continue
        a = (t > t[i] - pre[1]) & (t < t[i] - pre[0]) & (np.abs(p - ps) < 0.1 * ps + 3)
        b = (t > t[i] + post[0]) & (t < t[i] + post[1]) & (np.abs(p - ps) < 0.1 * ps + 3)
        if a.sum() < 120 or b.sum() < 120:
            continue
        if np.std(p[a]) > 0.15 * np.median(p[a]) + 3 or np.std(p[b]) > 0.15 * np.median(p[b]) + 3:
            continue
        out.append({"t": float(t[i]), "a": a, "b": b, "dP": float(np.median(p[b]) - np.median(p[a])),
                    "Pa": float(np.median(p[a])), "Pb": float(np.median(p[b]))})
        last = t[i]
    return out


def classify(dP: float) -> str:
    a = abs(dP)
    if 40 <= a <= 58:
        return "beam_projector"
    if 5 <= a <= 17:
        return "minipc"
    if a > 800:
        return "electiric_kettle"
    return "laptop_charger"


def delta_test(df):
    print("[델타 검정] 계단마다 Δ 페이저 vs 같은 전력의 단독 서명 (|Δ_h / I_ref,h|, ∠차 °) — 1 / 0 이면 중첩 성립")
    C = current_phasors(df)
    rows = []
    for s in steps(df):
        D = med_phasor(C[s["b"]]) - med_phasor(C[s["a"]])
        if s["dP"] < 0:
            D = -D                                 # 꺼짐 계단은 부호를 뒤집어 켜진 기기 페이저로
        dev = classify(s["dP"])
        Pd = abs(s["dP"])
        tol = {"beam_projector": 4.0, "minipc": 2.0, "laptop_charger": 4.0, "electiric_kettle": 60.0}[dev]
        ref, n, Pr = ref_phasor(dev, Pd, tol)
        tag = f"t={s['t']:6.1f}s  {s['Pa']:6.1f} -> {s['Pb']:6.1f}W  ΔP {s['dP']:+6.1f}  -> {dev:15s}"
        if ref is None:
            print(f"  {tag}  단독 참조 없음 (P±{tol}W 표본 {n})"); continue
        R = D / ref
        print(f"  {tag}  참조 P={Pr:.1f}W n={n}")
        print(f"      |Δ/ref| h1..15 {fmag(R)}")
        print(f"      ∠(Δ/ref) °     {fdeg(R)}")
        rows.append((dev, Pd, R))
    return rows


def kettle_test(df):
    p = df["p_w"].to_numpy(np.float64); ps = pd.Series(p).rolling(61, center=True, min_periods=1).median().to_numpy()
    on = ps > 1000; off = (ps > 60) & (ps < 400)
    if on.sum() < 300:
        return
    print("\n[포트 창] SMPS 3종 켜진 채 포트 on/off — (총전류 − 포트 단독) 이 SMPS 만(LOW 창) 과 같은가")
    C = current_phasors(df)
    stable = np.abs(p - ps) < 0.1 * ps + 3
    ov = df["over_range"].to_numpy() == 0
    m_off = off & stable & ov; m_on = on & stable & ov
    S_off = med_phasor(C[m_off])
    tot_on = med_phasor(C[m_on])
    kref, n, Pk = ref_phasor("electiric_kettle", float(np.median(p[m_on]) - np.median(p[m_off])), 80.0)
    print(f"  SMPS 만(LOW) n={m_off.sum()} P={np.median(p[m_off]):.0f}W |I1| {abs(S_off[0]):.3f}A   포트 켜짐(HIGH) n={m_on.sum()} P={np.median(p[m_on]):.0f}W |I1| {abs(tot_on[0]):.2f}A   포트 단독 참조 P={Pk:.0f}W n={n}")
    S_est = tot_on - kref
    print(f"  SMPS 만 (LOW 창)      |I_h| mA h1..15 {' '.join(f'{1e3*abs(v):5.0f}' for v in S_off)}")
    print(f"  총전류 − 포트 단독     |I_h| mA        {' '.join(f'{1e3*abs(v):5.0f}' for v in S_est)}")
    R = S_est / S_off
    print(f"  비 |est/SMPS|           {fmag(R)}")
    print(f"  ∠(est/SMPS) °           {fdeg(R)}")
    print(f"  |차| mA                 {' '.join(f'{1e3*abs(v):5.0f}' for v in (S_est - S_off))}    (SMPS 고조파 대비 %: {' '.join(f'{100*abs(a)/max(abs(b),1e-9):4.0f}' for a, b in zip((S_est - S_off)[ODD], S_off[ODD]))})")
    # 포트 계단 델타 vs 포트 단독
    Dk = []
    for s in steps(df, min_dp=800):
        D = med_phasor(C[s["b"]]) - med_phasor(C[s["a"]])
        if s["dP"] < 0:
            D = -D
        Dk.append(D)
    if Dk:
        D = np.mean(Dk, 0); R = D / kref
        print(f"  포트 계단 Δ / 포트 단독 ({len(Dk)}계단)  |비| {fmag(R)}")
        print(f"                                  ∠  {fdeg(R)}")
        print("  읽기: 포트 계단 Δ 가 포트 단독과 다르면 그 차이 = 포트가 켜지며 SMPS 가 바뀐 것 + HIGH 인공물 변화. 위 표의 |차| 가 SMPS 고조파의"
              " 몇 % 인지가 '포트 켜진 창에서 NILM 이 SMPS 를 얼마나 잃는가' 다.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    df, info = load(a.file)
    print(f"{a.file}: {len(df)} 사이클, LOW 회전 {info['low_cal_shift_deg_per_order']}°/차수, 세션 {info['session']}")
    delta_test(df)
    kettle_test(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
