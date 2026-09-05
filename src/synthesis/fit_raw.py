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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.synthesis.circuit_fit import EXTRA_SPEC
from src.synthesis.circuit_sim import F, NPC, VF_DEFAULT, simulate

NS = 256                #: 원시 표본/주기 (15360Hz / 60Hz)
NCYC_SIM = 10           #: 시뮬 주기 수 (8주기면 40주기 값과 1e-13 안에서 같다)
RC_FC_HZ = 1591.55      #: 안티앨리어싱 RC 차단 주파수 (펌웨어 `NILM_RC_FC_HZ`)
BAND = 15               #: 손실을 자르는 차수
#: par5 로그 경계 (C_dc, R, L, Cx, rd)
LO5 = np.log(np.array([5e-6, 0.3, 5e-6, 0.02e-6, 0.02]))
HI5 = np.log(np.array([600e-6, 40.0, 8000e-6, 2e-6, 20.0]))

#: 원시 적합에서 켤 수 있는 확장 요소 — 12.184.10 의 E1·E2·E3 (판정 철회분, 12.185.1).
#: 경계·중립값·두 번째 시작값은 `circuit_fit.EXTRA_SPEC` 하나만 쓴다.
#: `k_hi`(h9↑ 전압 크기 배율)는 뺐다 — 2Hz 전압 채널의 인공물을 재던 **입력 정형**이고,
#: 원시는 전압을 15360Hz 로 직접 재므로 같은 뜻을 갖지 않는다.
#:
#: ⚠ 셋 다 **`rd` 를 물리값에 고정한 위에서만** 뜻이 있다. `nvt` 는 브리지의 비선형 저항이고
#: `rd` 는 그 선형 저항이라 같은 자리에 산다 — `rd` 가 자유로우면 둘이 맞바뀌어(미니PC 의
#: R–rd 축퇴와 같은 종류) 어느 쪽도 판정되지 않는다.
RAW_EXTRAS: Tuple[str, ...] = ("nvt", "Gp", "alpha")

#: 원시 쪽 확장 사양 = `circuit_fit.EXTRA_SPEC` + `Vf`. 브리지 순방향 강하는 원래 상수
#: (1.4V = 다이오드 2개)로 박아 두는데, `nvt`(지수 무릎)·`rd`(선형 저항)·`Vf`(상수)는
#: **같은 브리지의 세 가지 기술**이다. 둘을 고정한 채 셋째만 맞추면 "요소가 필요하다" 는
#: 결론이 그 고정 때문일 수 있다 — 그래서 `Vf` 도 풀 수 있게 둔다 (기본은 고정 1.4V).
SPEC: Dict[str, Tuple[str, float, float, float, float]] = dict(EXTRA_SPEC)
SPEC["Vf"] = ("lin", 0.0, 3.0, VF_DEFAULT, 0.8)
#: 지수항의 기준 전류 [A]. 중립 0.01 은 옛 자리표(`circuit_sim.I0_KNEE`), 물리 포화전류는
#: 1e-9~1e-5. `nvt` 와 로그로 얽혀 있으므로 둘을 같이 풀면 곡선 가족 하나를 맞추는 것이다.
SPEC["i0"] = ("log", 1e-10, 1e-2, 0.01, 1e-6)


# ── 신호 처리 ────────────────────────────────────────────────────────────────
def halfwave(x: np.ndarray) -> np.ndarray:
    """반파 대칭화 `[x(t) − x(t+T/2)]/2` — 짝수 차수를 정확히 지우고 홀수를 정확히 보존."""
    return (np.asarray(x, float) - np.roll(x, len(x) // 2)) / 2.0


def rc_filter(x: np.ndarray, inverse: bool = False, fc: float = RC_FC_HZ,
              order: int = 1) -> np.ndarray:
    """`1/(1+jf/fc)^order` 를 걸거나(계측 모사) 벗긴다(참값 복원). 빈 k = k·60Hz.

    `order` 는 극 개수. 기본 1 은 펌웨어가 아는 안티앨리어싱 RC(1kΩ·100nF) 하나다.
    12.184.16 에 따르면 LOW 전류 경로는 HIGH 의 ADC 노드 뒤에서 한 단 더 증폭되므로
    **전류 채널의 극이 전압보다 하나 많을 수 있다** — `run_meas_rc_probe` 가 그것을 잰다.
    """
    X = np.fft.rfft(x)
    Hh = (1.0 / (1.0 + 1j * np.arange(len(X)) * F / fc)) ** int(order)
    return np.fft.irfft(X / Hh if inverse else X * Hh, len(x))


#: 변환기 버든 저항 [Ω] — 사용자 제공 LTspice 넷리스트 (ADC 앞단).
#: 적합된 코너 `f_c` 는 자화 인덕턴스를 말한다: `L_m = R_burden / (2π·f_c)`.
R_BURDEN_CT = 27.0        #: R8, SCT-013-100 2차 버든
R_BURDEN_ZMPT = 220.0     #: R12, ZMPT-101B 2차 버든


def hp_filter(x: np.ndarray, fc: float, inverse: bool = False) -> np.ndarray:
    """변성기 고역통과 `jf/(jf + fc)` 를 걸거나 벗긴다.

    CT·전압변성기는 직류를 못 통과시킨다 — 자화 인덕턴스 `L_m` 과 버든 `R` 이 만드는
    1차 고역통과이고 코너가 `f_c = R/(2π·L_m)` 이다. 넷리스트는 둘 다 **이상 전류원**으로
    두므로 이 항이 통째로 빠져 있다 (RC·연산증폭기는 넷리스트가 정확히 준다).

    **위상 앞섬이 저차에서 크고 고차에서 사라진다** — 극(RC)이나 순수 지연과 모양이
    다르므로 자료가 셋을 가를 수 있다. f_c=30Hz 면 h1 +26.6° h3 +14.0° h15 +1.9°.
    """
    if fc <= 0.0:
        return np.asarray(x, float)
    X = np.fft.rfft(x)
    f = np.arange(len(X)) * F
    Hh = np.ones(len(X), complex)
    Hh[1:] = (1j * f[1:]) / (1j * f[1:] + fc)
    Hh[0] = 0.0 if not inverse else 0.0       # 직류는 어차피 0 (반파 대칭화)
    if inverse:
        Y = np.zeros_like(X)
        Y[1:] = X[1:] / Hh[1:]
        return np.fft.irfft(Y, len(x))
    return np.fft.irfft(X * Hh, len(x))


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
    #: 시뮬 전류에 걸 계측 RC (전류 채널). 전압 쪽은 `vsrc` 를 만들 때 이미 벗겨 두었다.
    i_fc: float = RC_FC_HZ
    i_order: int = 1
    #: CT(SCT-013-100) 고역통과 코너 [Hz]. 0 이면 끈다. `L_m = 27Ω/(2π·f)`.
    i_hp_fc: float = 0.0
    #: 전류 채널이 전압 채널보다 늦게 표본화되는 양 [원시 표본, 15360Hz]. ADC 채널 스큐.
    #: 넷리스트(사용자 제공 LTspice)로 확인한 바 세 채널의 아날로그 전달함수는 1k·100n 1극으로
    #: 같고 차이가 h15 에서 0.5° 뿐이다 — 그러니 남는 차수 의존 위상은 **아날로그가 아니다**.
    #: 순수 지연은 위상이 f 에 선형이라 우리 대역(h1~h15)에서 극 하나와 모양이 거의 같다
    #: (극 1591.55Hz: h1 −2.16° h15 −29.5° / 지연 91µs: h1 −1.97° h15 −29.5°).
    i_skew_samp: float = 0.0


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


def shift_samples(x: np.ndarray, n: float) -> np.ndarray:
    """대역제한 분수 표본 이동. `n>0` 이면 신호가 **늦어진다**."""
    if n == 0.0:
        return np.asarray(x, float)
    X = np.fft.rfft(x)
    k = np.arange(len(X))
    return np.fft.irfft(X * np.exp(-2j * np.pi * k * n / len(x)), len(x))


def load_raw(stem: str, data_dir: str = "data", ncyc_sim: int = NCYC_SIM,
             bg: Optional[np.ndarray] = None, vband: Optional[int] = None,
             v_fc: float = RC_FC_HZ, v_order: int = 1,
             i_fc: float = RC_FC_HZ, i_order: int = 1,
             bg_rc: bool = False, i_skew_samp: Optional[float] = None,
             v_hp_fc: float = 0.0, i_hp_fc: float = 0.0) -> RawPoint:
    """`bg` 를 주면 그 배경 페이저를 실측 전류에서 뺀다 (`background_phasors()`).

    `vband` 를 주면 시뮬 소스 전압을 그 차수까지로 자른다. **생성기와 규약을 맞추는 스위치다**:
    실시간 입력은 고조파 15개뿐인데 원시 파형에는 h127 까지 있다. 장소 C 전압의 h17 이상은
    V1 의 0.27%(실측)/0.40%(역RC) 로 작아 보이지만, h15 로 자르면 피크가 +0.94V 올라가고
    **도통각이 38.8° -> 36.9° 로 2° 바뀐다**. 그 2° 가 h9~h15 에서 크기 14%·위상 20~37° 로
    증폭된다 (12.185.16). 전체 파형으로 맞춘 파라미터를 V15 생성기에 쓰면 그만큼 어긋난다.
    """
    import pandas as pd
    from src.preprocessing.file_registry import RAW_RANGE_MIXED, RAW_SKEW_SAMP_LOW

    if i_skew_samp is None:
        # 펌웨어 위상 교정은 2Hz 블록에만 걸린다 — 원시에는 채널 어긋남이 남아 있다.
        # 장소 C 포트 원시로 직접 쟀고(∠I₁−∠V₁ = +2.87°) 부호·기전이 확인됐다 (12.185.25).
        i_skew_samp = RAW_SKEW_SAMP_LOW

    d = pd.read_csv(f"{data_dir}/{stem}.csv", usecols=["cyc", "n", "i_a", "v_v", "range"])
    nc = int(d["cyc"].max()) + 1
    V = d["v_v"].to_numpy(np.float64).reshape(nc, NS)
    I = d["i_a"].to_numpy(np.float64).reshape(nc, NS)
    vm, im = halfwave(np.median(V, 0)), halfwave(np.median(I, 0))
    if bg is not None:
        bw = wave_from_phasors(bg)
        # ⚠ 배경은 **2Hz 파일**에서 왔고 펌웨어가 크기를 이미 보상한 값이다. 원시 전류에는
        # 보상이 없으므로 규약이 어긋난다 (h15 에서 |H|=0.87, 13%). `bg_rc=True` 면 배경에도
        # 같은 RC 를 걸어 규약을 맞춘다 ([M6]).
        if bg_rc:
            bw = rc_filter(bw, fc=i_fc, order=i_order)
        im = im - bw
    irms = float(np.sqrt(np.mean(im ** 2)))
    scatter = float(np.sqrt(np.mean(np.std(I, 0) ** 2)) / irms)
    X = phasors(im, NS // 2)
    oob = float(np.sqrt(np.sum(np.abs(X[BAND:]) ** 2)) / irms)
    vt = rc_filter(vm, inverse=True, fc=v_fc, order=v_order)
    if v_hp_fc > 0.0:
        vt = hp_filter(vt, v_hp_fc, inverse=True)   # ZMPT-101B 를 벗겨 참 전압으로
    if vband is not None:
        Vf_ = np.fft.rfft(vt)
        Vf_[vband + 1:] = 0
        vt = np.fft.irfft(Vf_, len(vt))
    vsrc = np.tile(upsample(vt), ncyc_sim)
    return RawPoint(stem, vm, im, float(np.mean(vm * im)), vsrc, irms, nc, scatter, oob,
                    stem in RAW_RANGE_MIXED, float(i_fc), int(i_order),
                    float(i_hp_fc), float(i_skew_samp))


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


# ── 확장 요소 ────────────────────────────────────────────────────────────────
def sim_kwargs(extras: Optional[Dict[str, float]]) -> Dict[str, float]:
    """extras -> `circuit_sim.simulate` 인수. `Gp`(컨덕턴스) 를 `Rp`(저항) 로 뒤집는다.

    중립값(nvt 0 · Gp 0 · alpha 1)이면 빈 dict 에 가까워져 기본 모형과 **정확히** 같다.
    """
    if not extras:
        return {}
    kw = {k: float(v) for k, v in extras.items() if k in ("nvt", "alpha", "Vf", "i0")}
    if float(extras.get("Gp", 0.0)) > 0.0:
        kw["Rp"] = 1.0 / float(extras["Gp"])
    return kw


def neutral_extras(names: Sequence[str] = ()) -> Dict[str, float]:
    """중립값 — 여기서 확장 모형은 기본 5파라미터 모형과 같다."""
    return {n: SPEC[n][3] for n in names}


def warm_extras(names: Sequence[str] = ()) -> Dict[str, float]:
    """두 번째 시작값 — 중립에서 야코비가 0 인 방향(Gp=0 하한, nvt 로그 하한)에 대비해 켠 값."""
    return {n: SPEC[n][4] for n in names}


def in_physical_range(extras: Dict[str, float]) -> List[str]:
    """[E5] 물리 범위 밖인 요소 이름. 밖이면 '요소' 가 아니라 '자리표' 로 읽는다."""
    # nvt 는 브리지 다이오드 **2개**의 n·V_T 합 = 2×(1~2)×0.026 = 0.052~0.104V
    lim = {"nvt": (0.05, 0.10), "Gp": (0.0, 0.05), "alpha": (0.0, 1.0), "Vf": (0.0, 2.1),
           "i0": (1e-9, 1e-5)}
    out = []
    for n, v in extras.items():
        if n not in lim or v == SPEC[n][3]:    # 중립은 '꺼짐' 이라 판정 대상이 아니다
            continue
        lo, hi = lim[n]
        if not lo <= v <= hi:
            out.append(f"{n}={v:.4g}")
    return out


# ── 잔차와 적합 ──────────────────────────────────────────────────────────────
def sim_current(par5: Sequence[float], pt: RawPoint, match_power: bool = True,
                n_match: int = 3, extras: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
    """(256,) 시뮬 전류 — 계측 RC 를 건 뒤 원시와 같은 표본률로."""
    kw = sim_kwargs(extras)
    P = pt.p_w
    r = None
    for _ in range(n_match if match_power else 1):
        r = simulate(P, *par5, vsrc=pt.vsrc, **kw)
        if not r["ok"] or not np.isfinite(r["p_w"]) or r["p_w"] <= 0:
            return None
        if not match_power:
            break
        P *= pt.p_w / r["p_w"]                      # 시뮬 입력 전력을 측정에 맞춘다
    r = simulate(P, *par5, vsrc=pt.vsrc, **kw)
    if not r["ok"]:
        return None
    out = downsample(r["i"][-NPC:])
    if pt.i_hp_fc > 0.0:
        out = hp_filter(out, pt.i_hp_fc)           # CT 를 걸어 계측값으로
    out = rc_filter(out, fc=pt.i_fc, order=pt.i_order)
    return shift_samples(out, pt.i_skew_samp) if pt.i_skew_samp else out


def point_residual(par5, pt: RawPoint, band: int = BAND, match_power: bool = True,
                   extras: Optional[Dict[str, float]] = None) -> np.ndarray:
    """(2·band,) 상대 잔차. `r@r` = (대역제한 상대 RMS)² (파세발)."""
    isim = sim_current(par5, pt, match_power, extras=extras)
    if isim is None:
        return np.full(2 * band, 10.0)
    d = (phasors(isim, band) - phasors(pt.i, band)) / pt.irms
    return np.r_[d.real, d.imag]


def residual(x: np.ndarray, pts: Sequence[RawPoint], band: int = BAND,
             match_power: bool = True, fixed: Optional[Dict[int, float]] = None,
             extras: Sequence[str] = ()) -> np.ndarray:
    par5, ex = unpack(x, fixed, extras)
    w = 1.0 / np.sqrt(len(pts))                     # 손실 = 동작점별 상대 RMS² 의 평균
    return np.concatenate([point_residual(par5, p, band, match_power, ex) * w for p in pts])


def loss_at(par5, extras: Optional[Dict[str, float]], pts: Sequence[RawPoint],
            band: int = BAND, match_power: bool = True) -> float:
    """한 점에서의 손실 (동작점별 상대 RMS² 의 평균). [E2] 단조 검사에 쓴다."""
    w = 1.0 / np.sqrt(len(pts))
    r = np.concatenate([point_residual(par5, p, band, match_power, extras) * w for p in pts])
    return float(r @ r)


def unpack(x: np.ndarray, fixed: Optional[Dict[int, float]] = None,
           extras: Sequence[str] = ()) -> Tuple[Tuple[float, ...], Dict[str, float]]:
    """자유 변수 벡터 -> (par5, extras dict). `fixed` 는 {인덱스: 값} 으로 고정한 것.

    par5 는 로그, extras 는 `EXTRA_SPEC` 의 변환을 따른다. 로그 요소가 하한에 붙으면
    중립값으로 되돌린다 (nvt 1e-4 -> 0: '없음' 을 정확히 표현하기 위해).
    """
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    par = [0.0] * 5
    for j, i in enumerate(free):
        par[i] = float(np.exp(np.clip(x[j], LO5[i], HI5[i])))
    for i, v in fixed.items():
        par[i] = float(v)
    ex: Dict[str, float] = {}
    for j, n in enumerate(extras):
        kind, lo, hi, neutral, _ = SPEC[n]
        v = float(x[len(free) + j])
        if kind == "log":
            v = float(np.exp(np.clip(v, np.log(lo), np.log(hi))))
            if v <= lo * 1.0000001:
                v = neutral
        else:
            v = float(np.clip(v, lo, hi))
        ex[n] = v
    return tuple(par), ex


def pack(par5: Sequence[float], fixed: Optional[Dict[int, float]] = None,
         extras: Optional[Dict[str, float]] = None, names: Sequence[str] = ()) -> np.ndarray:
    fixed = fixed or {}
    extras = extras or {}
    x = [np.log(par5[i]) for i in range(5) if i not in fixed]
    for n in names:
        kind, lo, hi, neutral, _ = SPEC[n]
        v = float(extras.get(n, neutral))
        x.append(np.log(max(v, lo)) if kind == "log" else float(np.clip(v, lo, hi)))
    return np.array(x, float)


def extra_bounds(names: Sequence[str] = ()) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for n in names:
        kind, a, b = SPEC[n][:3]
        lo.append(np.log(a) if kind == "log" else a)
        hi.append(np.log(b) if kind == "log" else b)
    return np.array(lo, float), np.array(hi, float)


def rms_of(par5, pt: RawPoint, band: int = BAND, match_power: bool = True,
           extras: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """(대역제한 상대 RMS, 전대역 상대 RMS). 실패하면 (nan, nan)."""
    isim = sim_current(par5, pt, match_power, extras=extras)
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
    extras: Dict[str, float] = field(default_factory=dict)
    #: [E2] 겹쳐 시작이 기본값보다 나쁜 결과를 막았는가 — True 면 그 줄은 최적화 실패의 기록이다
    guard_fired: bool = False


PAR_NAMES = ("C_dc", "R", "L", "Cx", "rd")


def _norm_start(s) -> Tuple[Sequence[float], Dict[str, float]]:
    """시작점을 (par5, extras) 로 정규화. par5 만 주면 extras 는 중립."""
    if len(s) == 2 and isinstance(s[1], dict):
        return s[0], s[1]
    return s, {}


def fit(pts: Sequence[RawPoint], starts: Sequence, band: int = BAND,
        match_power: bool = True, fixed: Optional[Dict[int, float]] = None,
        max_nfev: int = 400, extras: Sequence[str] = ()) -> RawFit:
    """다중 시작 TRF 최소제곱. `fixed` 로 파라미터를 물리값에 묶을 수 있다 (식별성).

    `extras` 에 이름을 주면 그 확장 요소를 함께 맞춘다 (`RAW_EXTRAS`). 시작점은 par5 만
    주거나 `(par5, extras_dict)` 쌍으로 준다 — 겹쳐 시작은 `fit_nested` 를 쓰라.
    """
    from scipy.optimize import least_squares
    fixed = fixed or {}
    extras = tuple(extras)
    free = [i for i in range(5) if i not in fixed]
    elo, ehi = extra_bounds(extras)
    lo, hi = np.r_[LO5[free], elo], np.r_[HI5[free], ehi]
    best, nfev = None, 0
    for s in starts:
        p0, e0 = _norm_start(s)
        x0 = np.clip(pack(p0, fixed, e0, extras), lo, hi)
        try:
            r = least_squares(residual, x0, args=(pts, band, match_power, fixed, extras),
                              bounds=(lo, hi), diff_step=2e-3, xtol=1e-10, ftol=1e-12,
                              max_nfev=max_nfev)
        except Exception:
            continue
        nfev += int(r.nfev)
        if best is None or r.cost < best[1]:
            best = (r.x, float(r.cost))
    if best is None:
        raise RuntimeError("모든 시작점이 실패했다")
    par5, ex = unpack(best[0], fixed, extras)
    ab = [PAR_NAMES[i] for j, i in enumerate(free)
          if abs(best[0][j] - lo[j]) < 1e-6 or abs(best[0][j] - hi[j]) < 1e-6]
    rms = {p.stem: rms_of(par5, p, band, match_power, ex) for p in pts}
    return RawFit(par5, 2.0 * best[1], rms, ab, nfev, ex)


def fit_nested(pts: Sequence[RawPoint], base_par5: Sequence[float], extras: Sequence[str],
               starts: Sequence = (), band: int = BAND, match_power: bool = True,
               fixed: Optional[Dict[int, float]] = None, max_nfev: int = 400) -> RawFit:
    """기본 모형의 최적점 + **중립** extras 에서 출발한다 — 결과가 기본보다 나쁠 수 없다.

    12.184.10 의 요소 기각이 무효였던 이유가 여기다: 스칼라 최소화(Nelder-Mead)를 흩뿌린
    시작점에서 돌려 확장 모형이 중립값보다 **나쁜** 점에서 멈췄다. 겹친 모형은 기본을 특수
    케이스로 품으므로 그것은 요소가 아니라 최적화의 기록이다 (규칙: [E2]).

    중립에서 야코비가 0 인 방향(Gp=0 은 하한, nvt 는 로그 하한)이 있으므로 켠 시작점
    (`warm_extras`) 도 함께 넣는다 — 넣지 않으면 첫 미분이 0 이라 그 자리에서 못 움직인다.
    """
    extras = tuple(extras)
    st = [(tuple(base_par5), neutral_extras(extras))]
    if warm_extras(extras) != neutral_extras(extras):
        st.append((tuple(base_par5), warm_extras(extras)))
    st.extend(starts)
    r = fit(pts, st, band=band, match_power=match_power, fixed=fixed,
            max_nfev=max_nfev, extras=extras)
    l0 = loss_at(base_par5, neutral_extras(extras), pts, band, match_power)
    if l0 < r.loss:                                 # 단조 보장 — 여기 걸리면 최적화 결함이다
        rms = {p.stem: rms_of(base_par5, p, band, match_power) for p in pts}
        return RawFit(tuple(base_par5), l0, rms, [], r.nfev, neutral_extras(extras), True)
    return r


def loo_folds(pts: Sequence[RawPoint], starts: Sequence,
              **kw) -> List[Tuple[RawPoint, List[RawPoint], RawFit]]:
    """폴드마다 (뺀 점, 훈련점, 기본 적합). 요소를 여럿 재려면 이것을 한 번만 만들어 나눠 쓴다."""
    out = []
    for i, p in enumerate(pts):
        tr = [q for j, q in enumerate(pts) if j != i]
        out.append((p, tr, fit(tr, starts, **kw)))
    return out


def loo(pts: Sequence[RawPoint], starts: Sequence, **kw) -> Tuple[float, Dict[str, float]]:
    """동작점 하나를 빼고 맞춘 뒤 그 점의 대역제한 RMS. (평균, 점별)"""
    band, mp = kw.get("band", BAND), kw.get("match_power", True)
    out = {p.stem: rms_of(r.par5, p, band, mp, r.extras)[0]
           for p, _, r in loo_folds(pts, starts, **kw)}
    return float(np.mean(list(out.values()))), out


def loo_nested(folds: Sequence[Tuple[RawPoint, List[RawPoint], RawFit]], extras: Sequence[str],
               starts: Sequence = (), band: int = BAND, match_power: bool = True,
               fixed: Optional[Dict[int, float]] = None, max_nfev: int = 400
               ) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """확장 요소의 LOO. **폴드마다 다시 맞춘 기본**(`folds`) 에서 겹쳐 시작한다.

    전체 자료로 맞춘 정본 par5 를 겹쳐 시작에 쓰면 그것이 뺀 점까지 보고 정한 값이라
    새어 든다 ([E4]). 반환 (평균, 점별 RMS, 점별 채택된 extras).
    """
    out: Dict[str, float] = {}
    ex_per: Dict[str, Dict[str, float]] = {}
    for p, tr, b in folds:
        r = fit_nested(tr, b.par5, extras, starts=starts, band=band, match_power=match_power,
                       fixed=fixed, max_nfev=max_nfev)
        out[p.stem] = rms_of(r.par5, p, band, match_power, r.extras)[0]
        ex_per[p.stem] = r.extras
    return float(np.mean(list(out.values()))), out, ex_per
