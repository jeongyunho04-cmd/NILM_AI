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
def _diode_ib(drive, rd, nvt, i0, ib0=-1.0, nit=8):
    """도통 전압 여유 `drive` 에서 브리지 전류 — `v = rd·i + nvt·ln(1 + i/i0)` 를 푼다.

    `i0` 는 지수항의 기준 전류다. 옛 기본값 `I0_KNEE = 0.01A` 는 "무릎을 조금 둥글게" 하는
    자리표였고, **Shockley 로 읽으려면 포화전류(1e-9~1e-5A)** 여야 한다. 그때 `rd` 는 빼야
    한다 — 물리 다이오드의 동적 저항 `n·Vt/i` 는 로그항의 **미분**이지 별도 항이 아니다
    (`d/di [nvt·ln(1+i/i0)] = nvt/(i0+i)`). 둘을 같이 두면 다이오드를 두 번 세는 것이고,
    그래서 rd=0.3Ω 을 고정한 채 맞춘 nvt 가 물리값의 1/4~1/2 로 나왔다 (12.185.20).

    뉴턴의 출발점: rd=0 일 때의 해석해 `i0·(e^{drive/nvt} − 1)` 을 쓰고 `drive/rd` 로 자른다.
    둘 다 참 해보다 크므로 f ≥ 0 에서 감쇠 하강하면 단조 수렴한다 (옛 선형 출발점은
    i0 가 작으면 여러 자릿수 아래에서 시작해 4회로 못 올라온다).
    """
    if drive <= 0.0:
        return 0.0
    if nvt <= 0.0:
        return drive / rd
    if ib0 > 0.0:
        ib = ib0                       # 겹쳐 시작 — 노드 뉴턴 안에서는 직전 반복의 해가 가깝다
    else:
        x = drive / nvt
        if x > 40.0:
            x = 40.0
        ib = i0 * (np.exp(x) - 1.0)
    if rd > 1e-12:
        lim = drive / rd
        if ib > lim:
            ib = lim
    for _ in range(nit):
        f = rd * ib + nvt * np.log(1.0 + ib / i0) - drive
        fp = rd + nvt / (i0 + ib)
        step = f / fp
        if step > 0.5 * ib:          # 한 걸음에 절반 아래로 못 내려간다 (감쇠)
            step = 0.5 * ib
        ib -= step
        if ib < 0.0:
            ib = 0.0
        if step * step < 1e-24:
            break
    return ib


@njit(cache=True)
def _core(vsrc, dt, P, C_dc, R, L, Cx, Vf, rd, vc0, nvt, Rp, alpha, i0, nsub):
    """시간 적분. 반환 (선 전류 iL, 브리지 전류 ib).

    확장 (기본값이면 원본과 같다): nvt>0 지수 다이오드 무릎(E1), Rp>0 초크 병렬 감쇠(E2),
    alpha 직류 부하 지수(E3: 1 정전력, 0 정전류, −1 저항).

    `nsub` 는 표본 하나를 몇 조각으로 쪼갤지 (L·Cx 공진이 빠를 때 안정성·정확도. `substeps()` 참조).
    나온 값은 조각 평균이라 nsub=1 이면 원본과 같다.
    """
    N = vsrc.size
    iac = np.zeros(N)
    ibr = np.zeros(N)
    vc = vc0
    iL = 0.0          # 초크(L)에 흐르는 전류
    it = 0.0          # 선 전류 = iL + Rp 가지 전류
    vn = vsrc[0]
    h = dt / nsub
    b = h / Cx if Cx > 1e-12 else 0.0
    i_ref = P / V0_LOAD
    ib = 0.0
    for k in range(N):
        v0 = vsrc[k]
        v1 = vsrc[k + 1] if k + 1 < N else vsrc[k]
        acc_i = 0.0
        acc_b = 0.0
        for j in range(nsub):
            vs = v0 + (v1 - v0) * (j + 0.5) / nsub          # 조각 안의 소스 전압 (선형 보간)
            # 소스 -> 노드 (R + (L ∥ Rp) 직렬)
            if L > 1e-9:
                if Rp > 0.0:
                    vL = (vs - vn - R * iL) / (1.0 + R / Rp)
                    iL += h * vL / L
                    it = iL + vL / Rp
                else:
                    iL += h * (vs - vn - iL * R) / L
                    it = iL
            else:
                it = (vs - vn) / R
                iL = it
            if Cx > 1e-12:
                # 노드 전압을 **음해로** 푼다:  vn = vfree − b·i_b(vn),  vfree = vn + b·it
                # (부호는 vn 의 부호를 따른다. i_b 는 `_diode_ib` 의 브리지 전류.)
                #
                # ⚠ 옛 판은 이것을 `rd_eff` 로 선형화했는데, 선형화 전류를 어디서 잡느냐에
                # 따라 답이 크게 달라진다. 직전 표본의 ib(≈0 도통 직전)에서 잡으면 지수
                # 다이오드에서 rd_eff 가 5×10⁴Ω 까지 뛰어 다이오드 가지가 통째로 무시되고,
                # 예측값에서 잡으면 반대로 과대평가된다. 두 선형화가 nvt 판의 손실을
                # **18배** 갈랐다 — 그 자리에서 적분이 수렴하지 않았다는 뜻이다 (12.185.23).
                # 그래서 선형화를 버리고 스칼라 뉴턴으로 직접 푼다. 단조 함수라 안정하다.
                vfree = vn + b * it
                sgn = 1.0 if vfree >= 0.0 else -1.0
                drive = sgn * vfree - Vf - vc
                if drive <= 0.0:
                    vn = vfree
                    ib = 0.0
                else:
                    # g(u) = u − |vfree| + b·i_b(u − Vf − vc) = 0,  u = |vn| ≥ 0
                    u = sgn * vfree
                    ibn = ib if ib > 0.0 else -1.0     # 직전 표본의 브리지 전류로 시작
                    for _ in range(6):
                        ibn = _diode_ib(u - Vf - vc, rd, nvt, i0, ibn, 3)
                        g = u - sgn * vfree + b * ibn
                        if g <= 0.0:
                            break
                        # dg/du = 1 + b·di/dv,  di/dv = 1/(rd + nvt/(i0+i))
                        dv = rd + (nvt / (i0 + ibn) if nvt > 0.0 else 0.0)
                        gp = 1.0 + b / dv
                        step = g / gp
                        lo = Vf + vc
                        if u - step < lo:
                            step = 0.5 * (u - lo)      # 도통 문턱 아래로 안 내려간다
                        u -= step
                        if step < 1e-9:            # 1nV — 노드 전압의 물리 규모보다 훨씬 아래
                            break
                    ib = _diode_ib(u - Vf - vc, rd, nvt, i0, ibn, 4)
                    vn = sgn * u
            else:
                ib = _diode_ib(abs(vs - it * R) - Vf - vc, rd, nvt, i0)
                it = ib if vs >= 0.0 else -ib
                iL = it
            # 직류 부하: alpha=1 이면 P/vc (정전력)
            vcc = max(vc, 1.0)
            if alpha == 1.0:
                i_load = P / vcc
            else:
                i_load = i_ref * (V0_LOAD / vcc) ** alpha
            vc += h * (ib - i_load) / C_dc
            acc_i += it
            acc_b += ib
        iac[k] = acc_i / nsub
        ibr[k] = acc_b / nsub
    return iac, ibr


def substeps(L: float, Cx: float, dt: float, target: float = 0.25, cap: int = 32) -> int:
    """L·Cx 공진이 한 표본 안에서 몇 라디안 도는지 보고 쪼갤 수를 정한다.

    심플렉틱 오일러는 `ω0·h < 2` 에서만 안정하고 `ω0·h` 가 0.5 를 넘으면 링잉 주파수가
    눈에 띄게 어긋난다. 옛 판은 NPC=3072 고정이라 (프로젝터 L 32µH·Cx 0.51µF 에서 ω0·dt≈1.34)
    여유가 없었고, 적합이 Cx 를 더 줄이면 **발산**해 손실이 1e2 로 튀었다 (12.185.4).
    """
    if L <= 1e-9 or Cx <= 1e-12:
        return 1
    w0 = 1.0 / np.sqrt(L * Cx)
    return int(min(cap, max(1, np.ceil(w0 * dt / target))))


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
             nvt: float = 0.0, Rp: float = 0.0, alpha: float = 1.0,
             i0: float = I0_KNEE, nsub: Optional[int] = None) -> Dict:
    """한 동작점을 푼다.

    반환 dict:
      ok        수치가 유한한가
      s         (h_max,) complex  h1 정규화 서명 (펌웨어 관례)
      I         (h_max,) complex  절대 전류 페이저 [A rms], 같은 관례
      V         (h_max,) complex  전압 페이저 [V rms], 같은 관례 (V[0] 는 실수)
      p_w, irms, thd, cond_deg     평균 전력, 전류 rms, THD(h2..h_max / h1), 반주기 도통각[°]
      i, v      마지막 NDFT 주기의 파형
    확장 요소 (12.184.10): nvt 지수 다이오드 무릎 [V], Rp 초크 병렬 감쇠 [Ω] (0=없음), alpha 직류 부하 지수.
    `i0` 는 지수항의 기준 전류 — 기본 0.01A 는 옛 자리표이고, Shockley 로 읽으려면
    포화전류(1e-9~1e-5A)로 두고 `rd` 를 하한에 묶어야 한다 (`_diode_ib` 주석).
    """
    if vsrc is None:
        vsrc = ideal_wave(V_rms)
    vsrc = np.ascontiguousarray(vsrc, dtype=np.float64)
    dt = 1.0 / (F * NPC)
    ns = substeps(float(L), float(Cx), dt) if nsub is None else int(nsub)
    iL, ib = _core(vsrc, dt, float(P), float(C_dc), max(float(R), 1e-3), float(L), float(Cx),
                   float(Vf), max(float(rd), 1e-3), float(np.max(np.abs(vsrc))) - float(Vf),
                   float(nvt), float(Rp), float(alpha), float(i0), ns)
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
        "cond_deg": float(np.mean(seg_b > 0.0) * 180.0), "nsub": ns,
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
