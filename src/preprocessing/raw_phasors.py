# -*- coding: utf-8 -*-
"""원본 CSV 의 전류·전압 고조파 블록 -> 페이저 (12.184).

4차 펌웨어(2026-09-04)부터 `vhdeg1~15` 가 있어 전압 **파형**을 복원할 수 있다. 여기 모은 것:

    current_phasors(df)            (N,15) complex   ih·e^{j·ihdeg}  [A rms]
    voltage_phasors(df, mask)      (15,) complex    vh 중앙값 · e^{j·원형평균(vhdeg)}  [V rms]
    steady_signature(df, lo, hi)   전력 [lo,hi) 정상 구간의 서명 (정규화·절대 중앙값, 표본수, P 중앙)
    phase_delay_fit(ratio)         두 녹화의 페이저 비가 **순수 시간 지연**인지 (h 에 선형인지)

[관례] 펌웨어: `ihdeg_h = arg(I_h) − h·arg(V_1)`, `vhdeg_h = arg(V_h) − h·arg(V_1)` (vhdeg1 ≡ 0).
순수 시간 지연 τ 는 이 관례에서 상쇄된다 — 전류·전압이 **같이** 지연될 때 이야기다.
전류 채널만 τ 지연되면 `ihdeg_h` 가 `−h·ω·τ` 만큼 돈다: **h 에 선형, 크기 불변**.
`phase_delay_fit` 이 그것을 가른다 (같은 기기·같은 장소·같은 전력의 옛 녹화가 대조군).

[전압 블록] `vh`/`vhdeg` 는 0.5초 창(30사이클)의 공통값이라 30행마다 반복된다.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

H = 15


def current_phasors(df: pd.DataFrame, h_max: int = H) -> np.ndarray:
    """(N, h_max) complex. `ih{h}`·exp(j·`ihdeg{h}`)."""
    return np.stack([df[f"ih{h}"].to_numpy(np.float64)
                     * np.exp(1j * np.deg2rad(df[f"ihdeg{h}"].to_numpy(np.float64)))
                     for h in range(1, h_max + 1)], 1)


def has_voltage_phase(df: pd.DataFrame, h_max: int = H) -> bool:
    return all(f"vhdeg{h}" in df.columns for h in range(1, h_max + 1))


def voltage_phasors(df: pd.DataFrame, mask: Optional[np.ndarray] = None,
                    h_max: int = H) -> Tuple[np.ndarray, Dict]:
    """(h_max,) complex 대표 전압 페이저와 통계.

    크기는 중앙값, 위상은 원형 평균. `vhdeg` 가 없으면 위상 0 (`stats['has_phase']=False`).
    stats: circ_std_deg (h_max,) — 창 사이 위상 산포 [°], n, vrms.
    """
    m = np.ones(len(df), bool) if mask is None else np.asarray(mask, bool)
    mag = np.array([np.median(df[f"vh{h}"].to_numpy(np.float64)[m]) for h in range(1, h_max + 1)])
    stats: Dict = {"n": int(m.sum()), "has_phase": has_voltage_phase(df, h_max),
                   "vrms": float(np.median(df["vrms"].to_numpy(np.float64)[m])) if "vrms" in df else np.nan}
    if stats["has_phase"]:
        ph = np.zeros(h_max)
        cs = np.zeros(h_max)
        for h in range(1, h_max + 1):
            z = np.exp(1j * np.deg2rad(df[f"vhdeg{h}"].to_numpy(np.float64)[m]))
            r = z.mean()
            ph[h - 1] = np.angle(r)
            cs[h - 1] = np.degrees(np.sqrt(max(-2.0 * np.log(max(abs(r), 1e-12)), 0.0)))
        stats["circ_std_deg"] = cs
    else:
        ph = np.zeros(h_max)
        stats["circ_std_deg"] = np.full(h_max, np.nan)
    return mag * np.exp(1j * ph), stats


def steady_signature(df: pd.DataFrame, p_lo: float, p_hi: float,
                     h_max: int = H) -> Dict:
    """전력이 [p_lo, p_hi) 이고 over_range 가 아닌 사이클의 서명.

    반환: s (정규화 중앙값, h1=1), I (절대 페이저 중앙값 [A rms]), n, p_w, vrms, ihdeg1,
          mask (그 사이클), even2 (|I2|/|I1| 중앙값, 위상 원형 표준편차 — 짝수차 검정용).
    """
    p = df["p_w"].to_numpy(np.float64)
    ok = (p >= p_lo) & (p < p_hi)
    if "over_range" in df:
        ok &= df["over_range"].to_numpy() == 0
    if "range" in df:
        ok &= df["range"].to_numpy() == 0
    C = current_phasors(df, h_max)[ok]
    if len(C) == 0:
        return {"n": 0, "mask": ok}
    S = C / C[:, [0]]
    s = np.median(S.real, 0) + 1j * np.median(S.imag, 0)
    I = np.median(C.real, 0) + 1j * np.median(C.imag, 0)
    z2 = np.exp(1j * np.angle(S[:, 1]))
    r2 = abs(z2.mean())
    return {
        "s": s, "I": I, "n": int(ok.sum()), "mask": ok,
        "p_w": float(np.median(p[ok])),
        "vrms": float(np.median(df["vrms"].to_numpy(np.float64)[ok])) if "vrms" in df else np.nan,
        "ihdeg1": float(np.median(df["ihdeg1"].to_numpy(np.float64)[ok])),
        "even2": (float(np.median(np.abs(S[:, 1]))),
                  float(np.degrees(np.sqrt(max(-2.0 * np.log(max(r2, 1e-12)), 0.0))))),
    }


def phase_delay_fit(ratio: np.ndarray, orders: Optional[np.ndarray] = None) -> Dict:
    """페이저 비 `ratio` (h_max,) 의 위상이 `h` 에 선형인가 (= 순수 시간 지연인가).

    홀수차만 맞춘다 (짝수차는 잡음 수준). 반환: slope_deg (°/차수), delay_ms, intercept_deg,
    resid_deg (홀수차 잔차), mag (|ratio| 홀수차).
    """
    h = np.arange(1, len(ratio) + 1) if orders is None else np.asarray(orders)
    odd = h % 2 == 1
    ph = np.unwrap(np.angle(ratio[odd]))
    a, b = np.polyfit(h[odd], ph, 1)
    resid = np.degrees(ph - (a * h[odd] + b))
    slope = np.degrees(a)
    return {"slope_deg": float(slope), "delay_ms": float(-slope / 360.0 / 60.0 * 1e3),
            "intercept_deg": float(np.degrees(b)), "resid_deg": resid,
            "mag": np.abs(ratio[odd]), "orders": h[odd]}


def apply_phase_rotation(df: pd.DataFrame, deg_per_order: float, h_max: int = H,
                         mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """전류 위상을 `+deg_per_order × h` 만큼 돌린 사본. `ihdeg{h}` 만 바꾼다 (있는 열만).

    펌웨어의 위상 교정(−h×delay 회전)이 틀린 채 녹화된 파일을 되돌리는 데 쓴다 (12.184.3).
    `mask` 를 주면 그 행만 돌린다 — 레인지 라벨별 교정값이 다를 때 (`range==0` 사이클만, 12.184.13).
    """
    out = df.copy()
    m = None if mask is None else np.asarray(mask, bool)
    for h in range(1, h_max + 1):
        col = f"ihdeg{h}"
        if col not in out.columns:
            continue
        d = out[col].to_numpy(np.float64).copy()
        if m is None:
            d = d + deg_per_order * h
        else:
            d[m] = d[m] + deg_per_order * h
        out[col] = (d + 180.0) % 360.0 - 180.0
    return out


def apply_phase_delay(df: pd.DataFrame, delay_ms: float, h_max: int = H) -> pd.DataFrame:
    """`apply_phase_rotation` 의 ms 판: 지연 τ 는 60Hz 에서 `360·60·τ` °/차수다."""
    return apply_phase_rotation(df, 360.0 * 60.0 * delay_ms / 1000.0, h_max)
