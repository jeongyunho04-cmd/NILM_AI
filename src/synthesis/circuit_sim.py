# -*- coding: utf-8 -*-
"""회로 기반 SMPS 시뮬레이터 — PFC 없는 capacitor-input 정류기 (CIRCUIT_FCM_GUIDE §3, 12.184).

    v_src ──[ R ]──[ L ]──┬──────────────┐
                          │              │
                         C_x         브리지 (Vf + rd·i)
                          │              │
                          │         ┌────┴────┐
                          │        C_dc    정전력 부하 P
                          │         └────┬────┘
                          └──────────────┘

원본은 저장소 루트의 `circuit_sim.py`(사용자 제공, `sim3`)였고 여기로 옮기면서 셋을 더했다:

  1. **`vsrc` 인수** — 이상 정현파 대신 임의 전압 파형을 넣는다. 가이드 §5(FCM 추출)와
     §8.1(게이트 5 를 실측 전압 파형으로)이 이것을 전제하는데 원본에는 없었다.
     4차 펌웨어의 `vh`+`vhdeg` 블록에서 `to_wave()` 로 파형을 복원해 넣는다
     (`src/preprocessing/raw_phasors.py`).
  2. **절대 페이저** — `simulate()` 가 h1 정규화 서명 `s` 와 함께 절대 전류 페이저 `I`(A rms)
     를 돌려준다. 생성기 통합과 노턴 추출은 절대값이 필요하다 (§5 의 `norton` 이 정규화된
     `sim` 을 쓰는 것은 결함이다 — `I1` 도 전압에 따라 변하므로 정규화하면 ∂I/∂V 가 틀린다).
  3. **게이트 1 진단** — 도통각은 브리지 전류 `ib > 0` 인 시간으로 잰다 (선 전류 `iL` 로 재면
     X-cap 링잉이 섞여 60~160° 가 나온다).

[위상 관례] 펌웨어와 같다: `arg(I_h) − h·arg(V_1)`. `to_wave` 가 `arg(V_1) = 0` 인 cos 기저로
파형을 만들므로 그 파형을 넣으면 시뮬 위상과 CSV 의 `ihdeg`/`vhdeg` 가 같은 자다.
(순수 시간 지연은 이 관례에서 상쇄되지만, cos↔sin 기저 차이는 `(h−1)·90°` 로 남는다 —
같은 기저를 쓰는지 저항 부하로 확인하라, 규칙 74.)

[수치] 반음해(semi-implicit) X-cap 노드 + 심플렉틱 오일러 LC. `ω0·dt < 2` 여야 안정한데
프로젝터(L 32µH, Cx 0.51µF)는 `ω0·dt ≈ 1.34` 로 여유가 작다. NPC 를 줄이지 마라.
numba 컴파일 뒤 1회 약 1.4ms (NPC 3072, 24주기).
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:                                   # numba 없으면 순수 파이썬 (약 1000배 느리다)
    def njit(**kw):
        def d(f):
            return f
        return d

F = 60.0        #: 계통 주파수
NPC = 3072      #: 주기당 표본
NCYC = 24       #: 시뮬 주기 수 (앞 20주기는 과도 안정화)
NDFT = 4        #: DFT 에 쓰는 마지막 주기 수
VF_DEFAULT = 1.4  #: 브리지 순방향 강하 (다이오드 2개 직렬, 고정)

#: 가이드 §4.2 의 장소 A 피팅값 (C_dc, R, L, C_x, rd). 출발점일 뿐이다 — rd 가 격자 상한(8Ω),
#: 미니PC R 도 상한(5.95Ω)에 붙어 있고, 이상 정현파로 맞춘 것이라 장소 A 의 전압 왜곡이
#: 파라미터에 박혀 있다 (12.184.2). 실측 파형이 있는 녹화로 다시 맞춰야 한다.
GUIDE_PARAMS: Dict[str, Tuple[float, float, float, float, float]] = {
    "laptop_charger": (53.1e-6, 2.10, 2500e-6, 0.28e-6, 5.04),
    "beam_projector": (116.0e-6, 4.59, 32e-6, 0.51e-6, 8.00),
    "minipc":         (108.0e-6, 5.95, 1000e-6, 0.44e-6, 8.00),
}


I0_KNEE = 0.01      #: 지수 다이오드 무릎의 기준 전류 [A] (E1)
V0_LOAD = 300.0     #: 직류 부하 지수의 기준 전압 [V] (E3)


@njit(cache=True)
def _diode_ib(drive, rd, nvt):
    """도통 전압 여유 `drive` 에서 브리지 전류. nvt>0 이면 v = rd·i + nvt·ln(1+i/I0) 를 뉴턴으로 푼다 (E1)."""
    if drive <= 0.0:
        return 0.0
    if nvt <= 0.0:
        return drive / rd
    ib = drive / (rd + nvt / I0_KNEE)
    for _ in range(4):
        f = rd * ib + nvt * np.log(1.0 + ib / I0_KNEE) - drive
        fp = rd + nvt / (I0_KNEE + ib)
        ib -= f / fp
        if ib < 0.0:
            ib = 0.0
    return ib


@njit(cache=True)
def _core(vsrc, dt, P, C_dc, R, L, Cx, Vf, rd, vc0, nvt, Rp, alpha):
    """시간 적분. 반환 (선 전류 iL, 브리지 전류 ib).

    확장 (기본값이면 원본과 같다): nvt>0 지수 다이오드 무릎(E1), Rp>0 초크 병렬 감쇠(E2),
    alpha 직류 부하 지수(E3: 1 정전력, 0 정전류, −1 저항).
    """
    N = vsrc.size
    iac = np.zeros(N)
    ibr = np.zeros(N)
    vc = vc0
    iL = 0.0          # 초크(L)에 흐르는 전류
    it = 0.0          # 선 전류 = iL + Rp 가지 전류
    vn = vsrc[0]
    b = dt / Cx if Cx > 1e-12 else 0.0
    i_ref = P / V0_LOAD
    ib = 0.0
    for k in range(N):
        # 소스 -> 노드 (R + (L ∥ Rp) 직렬)
        if L > 1e-9:
            if Rp > 0.0:
                vL = (vsrc[k] - vn - R * iL) / (1.0 + R / Rp)
                iL += dt * vL / L
                it = iL + vL / Rp
            else:
                iL += dt * (vsrc[k] - vn - iL * R) / L
                it = iL
        else:
            it = (vsrc[k] - vn) / R
            iL = it
        if Cx > 1e-12:
            # 도통 여부를 현재 vn 으로 판정한 뒤 노드 전압을 반음해로 갱신.
            # 지수 다이오드면 직전 ib 에서 선형화한 rd_eff 로 반음해 계수를 잡는다.
            rd_eff = rd + (nvt / (I0_KNEE + ib) if nvt > 0.0 else 0.0)
            a = dt / (rd_eff * Cx)
            if vn >= 0.0:
                if vn - Vf - vc > 0.0:
                    vn = (vn + b * it + a * (Vf + vc)) / (1.0 + a)
                else:
                    vn = vn + b * it
            else:
                if -vn - Vf - vc > 0.0:
                    vn = (vn + b * it - a * (Vf + vc)) / (1.0 + a)
                else:
                    vn = vn + b * it
            ib = _diode_ib(abs(vn) - Vf - vc, rd, nvt)
        else:
            ib = _diode_ib(abs(vsrc[k] - it * R) - Vf - vc, rd, nvt)
            it = ib if vsrc[k] >= 0.0 else -ib
            iL = it
        # 직류 부하: alpha=1 이면 P/vc (정전력)
        vcc = max(vc, 1.0)
        if alpha == 1.0:
            i_load = P / vcc
        else:
            i_load = i_ref * (V0_LOAD / vcc) ** alpha
        vc += dt * (ib - i_load) / C_dc
        iac[k] = it
        ibr[k] = ib
    return iac, ibr


def ideal_wave(V_rms: float = 222.0, n: int = NCYC * NPC) -> np.ndarray:
    """이상 정현파 `√2·V·sin(ωt)`. (원본 `sim3` 와 같은 기저 — 관례는 DFT 에서 상쇄된다.)"""
    t = np.arange(n) / (F * NPC)
    return np.sqrt(2.0) * V_rms * np.sin(2 * np.pi * F * t)


def to_wave(V: Sequence[complex], n: int = NCYC * NPC) -> np.ndarray:
    """전압 페이저 (15,) complex [V rms, 펌웨어 관례 `arg(V_h) − h·arg(V_1)`] -> 파형.

    `v(t) = Σ_k √2·|V_k|·cos(k·ω·t + ∠V_k)`, 기본파 위상 0 (cos 기저).
    """
    V = np.asarray(V, complex)
    t = np.arange(n) / (F * NPC)
    v = np.zeros(n)
    for k in range(1, len(V) + 1):
        if V[k - 1] == 0:
            continue
        v += np.sqrt(2.0) * np.abs(V[k - 1]) * np.cos(2 * np.pi * F * k * t + np.angle(V[k - 1]))
    return v


def _phasors(x: np.ndarray, h_max: int = 15) -> np.ndarray:
    """마지막 NDFT 주기의 h1..h_max 페이저 (rms, cos 기저)."""
    seg = x[-NDFT * NPC:]
    X = np.fft.rfft(seg)[NDFT:NDFT * (h_max + 1):NDFT]
    return X / (len(seg) / 2) / np.sqrt(2.0)


def simulate(P: float, C_dc: float, R: float, L: float, Cx: float = 0.0, rd: float = 1.0,
             V_rms: float = 222.0, Vf: float = VF_DEFAULT,
             vsrc: Optional[np.ndarray] = None, h_max: int = 15,
             nvt: float = 0.0, Rp: float = 0.0, alpha: float = 1.0) -> Dict:
    """한 동작점을 푼다.

    반환 dict:
      ok        수치가 유한한가
      s         (h_max,) complex  h1 정규화 서명 (펌웨어 관례)
      I         (h_max,) complex  절대 전류 페이저 [A rms], 같은 관례
      V         (h_max,) complex  전압 페이저 [V rms], 같은 관례 (V[0] 는 실수)
      p_w, irms, thd, cond_deg     평균 전력, 전류 rms, THD(h2..h_max / h1), 반주기 도통각[°]
      i, v      마지막 NDFT 주기의 파형
    확장 요소 (12.184.10): nvt 지수 다이오드 무릎 [V], Rp 초크 병렬 감쇠 [Ω] (0=없음), alpha 직류 부하 지수.
    """
    if vsrc is None:
        vsrc = ideal_wave(V_rms)
    vsrc = np.ascontiguousarray(vsrc, dtype=np.float64)
    dt = 1.0 / (F * NPC)
    iL, ib = _core(vsrc, dt, float(P), float(C_dc), max(float(R), 1e-3), float(L), float(Cx),
                   float(Vf), max(float(rd), 1e-3), float(np.max(np.abs(vsrc))) - float(Vf),
                   float(nvt), float(Rp), float(alpha))
    out: Dict = {"ok": bool(np.all(np.isfinite(iL)))}
    if not out["ok"]:
        out["s"] = np.full(h_max, np.nan, complex)
        return out
    Ih = _phasors(iL, h_max)
    Vh = _phasors(vsrc, h_max)
    h = np.arange(1, h_max + 1)
    ref = np.angle(Vh[0])
    I = np.abs(Ih) * np.exp(1j * (np.angle(Ih) - h * ref))
    V = np.abs(Vh) * np.exp(1j * (np.angle(Vh) - h * ref))
    seg_i = iL[-NDFT * NPC:]
    seg_v = vsrc[-NDFT * NPC:]
    seg_b = ib[-NDFT * NPC:]
    out.update({
        "s": I / I[0] if abs(I[0]) > 1e-12 else np.full(h_max, np.nan, complex),
        "I": I, "V": V,
        "p_w": float(np.mean(seg_i * seg_v)),
        "irms": float(np.sqrt(np.mean(seg_i ** 2))),
        "thd": float(np.sqrt(np.sum(np.abs(I[1:]) ** 2)) / max(abs(I[0]), 1e-12)),
        # 브리지 전류가 흐르는 표본의 비율 × 180° = 반주기 도통각
        "cond_deg": float(np.mean(seg_b > 0.0) * 180.0),
        "i": seg_i, "v": seg_v,
    })
    return out


def sim(P, C_dc, R, L, Cx=0.0, rd=1.0, V_rms=222.0, Vf=VF_DEFAULT, vsrc=None) -> np.ndarray:
    """가이드 §3.2 의 `sim` — h1 정규화 서명 (15,) complex. 발산하면 NaN."""
    return simulate(P, C_dc, R, L, Cx, rd, V_rms, Vf, vsrc)["s"]


sim3 = sim   #: 루트 `circuit_sim.py` 의 옛 이름


def fit_loss(par: Sequence[float], points: Sequence[Tuple[float, np.ndarray]], V_rms: float = 222.0,
             vsrc: Optional[np.ndarray] = None) -> float:
    """가이드 §4.1 의 손실 — h2..h15 Re/Im 가중 제곱합의 전력점 평균."""
    W = np.r_[1, 1, 1, 1, 1, .8, .8, .6, .6, .45, .45, .35, .35, .3]
    WW = np.r_[W, W]
    e = 0.0
    for p, s_meas in points:
        s = sim(p, *par, V_rms=V_rms, vsrc=vsrc)
        if not np.isfinite(s[1]):
            return 1e9
        d = (np.r_[s[1:].real, s[1:].imag] - np.r_[s_meas[1:].real, s_meas[1:].imag]) * WW
        e += float(d @ d)
    return e / len(points)


def norton(par: Sequence[float], p: float, V_base: np.ndarray, h_max: int = 15,
           rel: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """노턴 등가 `I(h) = J(h) − Σ_k Y(h,k)·ΔV(k)` 를 유한 차분으로 (가이드 §5, 절대 페이저 판).

    V_base : (15,) complex 동작점 전압 페이저 (펌웨어 관례).
    반환   : J (h_max,) complex [A rms], Y (h_max, h_max) complex [S]. 450회 시뮬, 약 1초.
    """
    base = simulate(p, *par, vsrc=to_wave(V_base))
    J = base["I"]
    Y = np.zeros((h_max, h_max), complex)
    dV = rel * abs(V_base[0])
    for k in range(1, h_max + 1):
        for ph in (0.0, np.pi / 2):                    # 실수부·허수부 각각 흔든다
            Vp = np.array(V_base, complex)
            Vp[k - 1] += dV * np.exp(1j * ph)
            dI = simulate(p, *par, vsrc=to_wave(Vp))["I"] - J
            Y[:, k - 1] += -(dI / (dV * np.exp(1j * ph))) / 2
    return J, Y
