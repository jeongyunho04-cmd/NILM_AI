# -*- coding: utf-8 -*-
"""원시 파형 스냅샷으로 회로 파라미터를 맞춘다 (12.185, 가이드 2026-09 개정 §10).

핵심 세 가지 — **하나라도 빠지면 적합이 무너진다** (전부 실측으로 확인했다):

  1. **전압 반파 대칭화.** 실측 전압은 +352 / −323 V 로 비대칭이다 (아날로그 전단의 덧셈
     짝수 성분, 가이드 §10.3). 그대로 넣으면 캡이 양의 피크로 충전된 뒤 음 반주기(323V)에서는
     도통 자체가 불가능해 시뮬이 반파 전류를 낸다. 대칭화 없이 맞추면 파형 RMS 가
     55~95% 에서 멈추고 R·L 이 경계(40Ω, 8mH)에 붙는다. 대칭화하면 5~14% 로 내려온다.
     `v_sym = [v(t) − v(t+T/2)]/2` 는 짝수 차수를 정확히 지우고 홀수를 정확히 보존한다.
  2. **계측 RC 를 순방향 모델에.** 세 ADC 입력이 1kΩ·100nF (fc 1591.55Hz) 를 지나므로 실측
     펄스는 둥글고 낮다 — 프로젝터 실측 1.18A/35° 대 필터 없는 시뮬 2.47A/21°. 전압은
     역RC 로 참값을 복원해 소스로 넣고, 시뮬 전류에는 RC 를 걸어서 실측과 비교한다.
     (펌웨어는 2Hz 고조파의 **크기**만 되돌린다. 원시 표본에는 보상이 없다.)
  3. **다점 동시 적합.** 한 동작점으로 맞추면 `rd` 가 그 점을 흡수해 다른 전력에서 틀린다
     (가이드 §10.5). 기기별로 모든 스냅샷을 한 손실에 넣는다.

[대역] 모델은 h15 까지만 뜻이 있다. 실측 전류의 h16 이상은 RMS 의 10~15% 이고 모델이 낼 수
없는 바닥이다. 그래서 손실은 h1~h15 로 자르고 전대역 RMS 는 참고로 함께 적는다.
**대역을 자른 시간영역 적합은 절대 페이저 h1~h15 적합과 같다** (파세발). 원시의 이점은 표현이
아니라 (a) 절대 크기, (b) 세션 위상 오프셋이 없음, (c) 주기별 통계, (d) Cx 직접 측정이다.
가이드 §10.1 의 "고조파로는 안 된다" 는 Z변환 선형 추정기(도통 구간이 필요)에 대한 말이지
적합 일반에 대한 말이 아니다.

[전력] 시뮬의 `P` 는 **직류 부하 전력**이고 측정 `p_w` 는 **교류 입력 전력**이다 (차이는 브리지
Vf·i 와 R·i² 손실, 실측 3~5%). `match_power=True` 면 시뮬 입력 전력이 측정과 같아지도록
P 를 안쪽에서 3회 되풀이해 맞춘다 — 그러면 잔차가 모양만 본다.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.synthesis.circuit_sim import F, NPC, simulate

NS = 256                #: 원시 표본/주기 (15360Hz / 60Hz)
NCYC_SIM = 10           #: 시뮬 주기 수 (8주기면 40주기 값과 1e-13 안에서 같다)
RC_FC_HZ = 1591.55      #: 안티앨리어싱 RC 차단 주파수 (펌웨어 `NILM_RC_FC_HZ`)
BAND = 15               #: 손실을 자르는 차수
#: par5 로그 경계 (C_dc, R, L, Cx, rd)
LO5 = np.log(np.array([5e-6, 0.3, 5e-6, 0.02e-6, 0.02]))
HI5 = np.log(np.array([600e-6, 40.0, 8000e-6, 2e-6, 20.0]))


# ── 신호 처리 ────────────────────────────────────────────────────────────────
def halfwave(x: np.ndarray) -> np.ndarray:
    """반파 대칭화 `[x(t) − x(t+T/2)]/2` — 짝수 차수를 정확히 지우고 홀수를 정확히 보존."""
    return (np.asarray(x, float) - np.roll(x, len(x) // 2)) / 2.0


def rc_filter(x: np.ndarray, inverse: bool = False, fc: float = RC_FC_HZ) -> np.ndarray:
    """1극 저역통과 `1/(1+jf/fc)` 를 걸거나(계측 모사) 벗긴다(참값 복원). 빈 k = k·60Hz."""
    X = np.fft.rfft(x)
    Hh = 1.0 / (1.0 + 1j * np.arange(len(X)) * F / fc)
    return np.fft.irfft(X / Hh if inverse else X * Hh, len(x))


def upsample(x: np.ndarray, npc: int = NPC) -> np.ndarray:
    """대역제한 보간 (FFT 영패딩). 시뮬은 NPC=3072 가 필요하다 (X-cap LC 안정)."""
    X = np.fft.rfft(x)
    Y = np.zeros(npc // 2 + 1, complex)
    Y[:len(X)] = X
    return np.fft.irfft(Y, npc) * (npc / len(x))


def downsample(x: np.ndarray, ns: int = NS) -> np.ndarray:
    X = np.fft.rfft(x)[:ns // 2 + 1]
    return np.fft.irfft(X, ns) * (ns / len(x))


def phasors(x: np.ndarray, band: int = BAND) -> np.ndarray:
    """(band,) complex rms 페이저, cos 기저. `circuit_sim._phasors` 와 같은 규약."""
    X = np.fft.rfft(x)[1:band + 1]
    return X / (len(x) / 2) / np.sqrt(2.0)


# ── 자료점 ───────────────────────────────────────────────────────────────────
@dataclass
class RawPoint:
    stem: str
    v: np.ndarray           # (256,) 대칭화한 실측 전압 [V] (계측 RC 가 걸린 채)
    i: np.ndarray           # (256,) 대칭화한 실측 전류 [A] (계측 RC 가 걸린 채)
    p_w: float              # 교류 입력 전력 [W]
    vsrc: np.ndarray        # (NCYC_SIM*NPC,) 시뮬 소스 = 역RC 로 참값 복원 + 보간 + 반복
    irms: float
    n_cyc: int
    scatter: float          # 주기 간 산포 / rms (재현 바닥)
    oob: float              # h16 이상 성분 / rms (모델이 낼 수 없는 바닥)
    range_mixed: bool


#: 계측계 자체의 배경 전류 (`noise_noselfpower_C`, 기기 없음): 1.70W, |I1| 7.3mA ∠+10.9°.
#: ⚠ minipc_4C / beam_projector_4C 의 "꺼짐" 창(30mA ∠+68°)은 배경이 **아니다** — 어댑터가 꽂힌 채라
#: 그 기기 자신의 X-cap 전류다 (원시로 잰 Cx 로 계산한 28.6·34.1mA 와 맞는다). 그걸 빼면 모델이
#: Cx 를 못 찾는다 (2Hz 적합에서 미니PC Cx 가 하한 0.010µF 로 갔다). 배경은 이 파일 하나로 잰다. 12.185.9
BG_STEM = "noise_noselfpower_C"


def background_phasors(stem: str = BG_STEM, data_dir: str = "data") -> np.ndarray:
    """(15,) complex — 계측계 배경 전류 페이저 [A rms], 홀수차만 (짝수는 인공물)."""
    from src.preprocessing.raw_csv import read_raw_csv
    from src.preprocessing.raw_phasors import steady_signature
    cols = ["p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, 16)]         + [f"ihdeg{h}" for h in range(1, 16)]
    df, _ = read_raw_csv(f"{data_dir}/{stem}.csv", usecols=cols)
    I = steady_signature(df, -1e9, 1e9)["I"]
    out = np.zeros(15, complex)
    out[0::2] = I[0::2]
    return out


def wave_from_phasors(X: np.ndarray, ns: int = NS) -> np.ndarray:
    """(15,) rms 페이저 -> (ns,) 파형. `phasors()` 의 역."""
    Y = np.zeros(ns // 2 + 1, complex)
    Y[1:len(X) + 1] = np.asarray(X, complex) * np.sqrt(2.0) * (ns / 2)
    return np.fft.irfft(Y, ns)


def load_raw(stem: str, data_dir: str = "data", ncyc_sim: int = NCYC_SIM,
             bg: Optional[np.ndarray] = None, vband: Optional[int] = None) -> RawPoint:
    """`bg` 를 주면 그 배경 페이저를 실측 전류에서 뺀다 (`background_phasors()`).

    `vband` 를 주면 시뮬 소스 전압을 그 차수까지로 자른다. **생성기와 규약을 맞추는 스위치다**:
    실시간 입력은 고조파 15개뿐인데 원시 파형에는 h127 까지 있다. 장소 C 전압의 h17 이상은
    V1 의 0.27%(실측)/0.40%(역RC) 로 작아 보이지만, h15 로 자르면 피크가 +0.94V 올라가고
    **도통각이 38.8° -> 36.9° 로 2° 바뀐다**. 그 2° 가 h9~h15 에서 크기 14%·위상 20~37° 로
    증폭된다 (12.185.16). 전체 파형으로 맞춘 파라미터를 V15 생성기에 쓰면 그만큼 어긋난다.
    """
    import pandas as pd
    from src.preprocessing.file_registry import RAW_RANGE_MIXED

    d = pd.read_csv(f"{data_dir}/{stem}.csv", usecols=["cyc", "n", "i_a", "v_v", "range"])
    nc = int(d["cyc"].max()) + 1
    V = d["v_v"].to_numpy(np.float64).reshape(nc, NS)
    I = d["i_a"].to_numpy(np.float64).reshape(nc, NS)
    vm, im = halfwave(np.median(V, 0)), halfwave(np.median(I, 0))
    if bg is not None:
        im = im - wave_from_phasors(bg)
    irms = float(np.sqrt(np.mean(im ** 2)))
    scatter = float(np.sqrt(np.mean(np.std(I, 0) ** 2)) / irms)
    X = phasors(im, NS // 2)
    oob = float(np.sqrt(np.sum(np.abs(X[BAND:]) ** 2)) / irms)
    vt = rc_filter(vm, inverse=True)
    if vband is not None:
        Vf_ = np.fft.rfft(vt)
        Vf_[vband + 1:] = 0
        vt = np.fft.irfft(Vf_, len(vt))
    vsrc = np.tile(upsample(vt), ncyc_sim)
    return RawPoint(stem, vm, im, float(np.mean(vm * im)), vsrc, irms, nc, scatter, oob,
                    stem in RAW_RANGE_MIXED)


# ── Cx 직접 측정 (가이드 §10.2) ──────────────────────────────────────────────
def measure_cx(pt: RawPoint, thr: float = 0.15, guard: int = 3) -> Tuple[float, float, int]:
    """비도통 구간에서 `i ≈ Cx·dv/dt` 회귀. 반환 (Cx [F], 상관, 표본수).

    도통 구간을 `guard` 표본만큼 넓혀 제외한다 (에지 링잉이 회귀를 흔든다).
    LTI 필터는 미분과 교환하므로 RC 가 걸린 채로 재도 정확하다.
    """
    dv = (np.roll(pt.v, -1) - np.roll(pt.v, 1)) / 2.0 * (F * NS)
    for t in (thr, 0.3, 0.5):
        cond = np.abs(pt.i) >= t * np.abs(pt.i).max()
        k = np.ones(2 * guard + 1)
        m = ~(np.convolve(np.r_[cond, cond, cond].astype(float), k, "same")[NS:2 * NS] > 0)
        if m.sum() >= 60:
            break
    if m.sum() < 20:
        return np.nan, np.nan, int(m.sum())
    a, b = dv[m], pt.i[m]
    cx = float(np.sum(a * b) / np.sum(a * a))
    r = float(np.corrcoef(a, b)[0, 1])
    return cx, r, int(m.sum())


# ── 잔차와 적합 ──────────────────────────────────────────────────────────────
def sim_current(par5: Sequence[float], pt: RawPoint, match_power: bool = True,
                n_match: int = 3) -> Optional[np.ndarray]:
    """(256,) 시뮬 전류 — 계측 RC 를 건 뒤 원시와 같은 표본률로."""
    P = pt.p_w
    r = None
    for _ in range(n_match if match_power else 1):
        r = simulate(P, *par5, vsrc=pt.vsrc)
        if not r["ok"] or not np.isfinite(r["p_w"]) or r["p_w"] <= 0:
            return None
        if not match_power:
            break
        P *= pt.p_w / r["p_w"]                      # 시뮬 입력 전력을 측정에 맞춘다
    r = simulate(P, *par5, vsrc=pt.vsrc)
    if not r["ok"]:
        return None
    return rc_filter(downsample(r["i"][-NPC:]))


def point_residual(par5, pt: RawPoint, band: int = BAND, match_power: bool = True) -> np.ndarray:
    """(2·band,) 상대 잔차. `r@r` = (대역제한 상대 RMS)² (파세발)."""
    isim = sim_current(par5, pt, match_power)
    if isim is None:
        return np.full(2 * band, 10.0)
    d = (phasors(isim, band) - phasors(pt.i, band)) / pt.irms
    return np.r_[d.real, d.imag]


def residual(x: np.ndarray, pts: Sequence[RawPoint], band: int = BAND,
             match_power: bool = True, fixed: Optional[Dict[int, float]] = None) -> np.ndarray:
    par5 = unpack(x, fixed)
    w = 1.0 / np.sqrt(len(pts))                     # 손실 = 동작점별 상대 RMS² 의 평균
    return np.concatenate([point_residual(par5, p, band, match_power) * w for p in pts])


def unpack(x: np.ndarray, fixed: Optional[Dict[int, float]] = None) -> Tuple[float, ...]:
    """자유 변수 벡터(로그) -> par5. `fixed` 는 {인덱스: 값} 으로 고정한 것."""
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    par = [0.0] * 5
    for j, i in enumerate(free):
        par[i] = float(np.exp(np.clip(x[j], LO5[i], HI5[i])))
    for i, v in fixed.items():
        par[i] = float(v)
    return tuple(par)


def pack(par5: Sequence[float], fixed: Optional[Dict[int, float]] = None) -> np.ndarray:
    fixed = fixed or {}
    return np.array([np.log(par5[i]) for i in range(5) if i not in fixed], float)


def rms_of(par5, pt: RawPoint, band: int = BAND, match_power: bool = True) -> Tuple[float, float]:
    """(대역제한 상대 RMS, 전대역 상대 RMS). 실패하면 (nan, nan)."""
    isim = sim_current(par5, pt, match_power)
    if isim is None:
        return np.nan, np.nan
    d = (phasors(isim, band) - phasors(pt.i, band)) / pt.irms
    full = float(np.sqrt(np.mean((isim - pt.i) ** 2)) / pt.irms)
    return float(np.sqrt(np.sum(np.abs(d) ** 2))), full


@dataclass
class RawFit:
    par5: Tuple[float, ...]
    loss: float                     # 동작점별 대역제한 상대 RMS² 의 평균
    rms: Dict[str, Tuple[float, float]]
    at_bound: List[str]
    nfev: int


PAR_NAMES = ("C_dc", "R", "L", "Cx", "rd")


def fit(pts: Sequence[RawPoint], starts: Sequence[Sequence[float]], band: int = BAND,
        match_power: bool = True, fixed: Optional[Dict[int, float]] = None,
        max_nfev: int = 400) -> RawFit:
    """다중 시작 TRF 최소제곱. `fixed` 로 파라미터를 물리값에 묶을 수 있다 (식별성)."""
    from scipy.optimize import least_squares
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    lo, hi = LO5[free], HI5[free]
    best, nfev = None, 0
    for s in starts:
        x0 = np.clip(pack(s, fixed), lo, hi)
        try:
            r = least_squares(residual, x0, args=(pts, band, match_power, fixed),
                              bounds=(lo, hi), diff_step=2e-3, xtol=1e-10, ftol=1e-12,
                              max_nfev=max_nfev)
        except Exception:
            continue
        nfev += int(r.nfev)
        if best is None or r.cost < best[1]:
            best = (r.x, float(r.cost))
    if best is None:
        raise RuntimeError("모든 시작점이 실패했다")
    par5 = unpack(best[0], fixed)
    ab = [PAR_NAMES[i] for j, i in enumerate(free)
          if abs(best[0][j] - lo[j]) < 1e-6 or abs(best[0][j] - hi[j]) < 1e-6]
    rms = {p.stem: rms_of(par5, p, band, match_power) for p in pts}
    return RawFit(par5, 2.0 * best[1], rms, ab, nfev)


def loo(pts: Sequence[RawPoint], starts: Sequence[Sequence[float]], **kw) -> Tuple[float, Dict[str, float]]:
    """동작점 하나를 빼고 맞춘 뒤 그 점의 대역제한 RMS. (평균, 점별)"""
    out: Dict[str, float] = {}
    for i, p in enumerate(pts):
        tr = [q for j, q in enumerate(pts) if j != i]
        r = fit(tr, starts, **kw)
        out[p.stem] = rms_of(r.par5, p, kw.get("band", BAND), kw.get("match_power", True))[0]
    return float(np.mean(list(out.values()))), out
