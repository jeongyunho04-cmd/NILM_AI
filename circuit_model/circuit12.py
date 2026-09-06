"""
토폴로지 v12g — 2026-09-06 확정 표준형 (세 기기 공통)

   v_src ─┬─[R]─[L(i)]─[브리지 Vf + rd·i]─ C_dc ∥ P_main
          Cx  (선로측 X-cap)
          G   (선형 덧셈 경로: 대기 전원·계측 배경.  P_main = P − G·Vrms²)

   L(i) = L0 / (1 + (i/Isat)²)   ← 초크 포화. 6.7% → 2.3% 를 만든 요소.
   NTC 는 R 에 흡수 (충전기는 켠 뒤 10분 동안 3.2 → 2.6Ω). 생성기에서는 범위로 랜덤화.

파라미터 벡터 (pkl 'params'):  (C_dc, R, L0, Isat, Cx, rd, G)
계측 규약: 단일 입력 ADC (LSB 201.4µV), 전압 역RC(τ) → 시뮬 → 전류 RC(τ), τ = 60µs (LOW/HIGH 비 실측 + 스캔 최적).
"""
import numpy as np
from numba import njit

F = 60.0
VF = 1.4
#: 정상상태에 이르기까지 돌리는 주기 수 (13.23). 옛 값은 24 였다 — 초기 vc0 을 전압 꼭짓점에 두므로
#: 과도가 빨리 죽어 그만큼 필요하지 않다. 무작위 60판(기기 3종 x 3~70W x 전압 텍스처)에서 ncyc 64
#: 대비 상대오차의 **최악**은
#:     ncyc  6  3.8e-02 |  8  1.7e-02 | 10  6.8e-03 | 12  2.7e-03 | **16  4.2e-04** | 24  1.0e-05
#: 다. 중앙값은 ncyc 8 에서 이미 3.6e-09 지만 **저전력 꼬리가 늦게 잠긴다** — 회로모델 자체의 실측
#: 오차가 2~7% 이므로 그보다 두 자릿수 아래인 16 을 쓴다.
NCYC = 16


@njit(cache=True)
def _core12(vsrc, dt, P, C_dc, Rs, L0, Isat, Cx, Vf, rd, vc0):
    N = vsrc.size; iac = np.zeros(N); vc = vc0; iL = 0.0
    for k in range(N):
        icx = Cx * (vsrc[k] - vsrc[k - 1]) / dt if k > 0 else 0.0
        drive = abs(vsrc[k]) - Vf - vc
        L = L0 / (1.0 + (iL / Isat) ** 2)
        if drive > 0.0 or iL > 0.0:
            iL += dt * (drive - iL * (Rs + rd)) / L
            if iL < 0.0:
                iL = 0.0
        ib = iL
        vc += dt * (ib - P / max(vc, 1.0)) / C_dc
        iac[k] = (ib if vsrc[k] >= 0.0 else -ib) + icx
    return iac


def rc_periodic(x, dt, tau, inverse=False):
    """주기 신호에 1극 RC (정확, 주파수영역). inverse=True 면 역RC."""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), dt)
    H = 1.0 / (1.0 + 1j * 2 * np.pi * f * tau)
    return np.fft.irfft(X * (1.0 / H if inverse else H), n=len(x))


def sim_wave(P, V, params, npc=256, tau=60e-6, up=12, ncyc=NCYC, deembed=True):
    """
    측정 전압 1주기 파형 V (npc 샘플, 이미 역RC 된 것이면 deembed=False) → 계측 영역 전류 1주기 (npc 샘플).
    params = (C, R, L0, Isat, Cx, rd, G).  P 는 총전력 (덧셈 성분 포함).
    """
    C, R, L0, Isat, Cx, rd, G = params
    dt = 1.0 / (F * npc)
    Vt = rc_periodic(V, dt, tau, inverse=True) if (deembed and tau) else V
    t_lo = np.arange(npc) / npc; t_hi = np.arange(npc * up) / (npc * up)
    Vh = np.interp(t_hi, np.r_[t_lo, 1.0], np.r_[Vt, Vt[0]]); vsrc = np.tile(Vh, ncyc); dth = dt / up
    Pm = P - G * np.mean(Vt ** 2)
    if Pm < 0.5:
        Pm = 0.5
    I = _core12(vsrc, dth, Pm, C, max(R, 1e-3), L0, Isat, Cx, VF, max(rd, 1e-3), vsrc.max() - VF)
    if not np.all(np.isfinite(I)):
        return None
    Ic = I[-npc * up:] + G * Vh
    if tau:
        Ic = rc_periodic(Ic, dth, tau)
    return Ic.reshape(npc, up).mean(1)


def wave_from_harmonics(V15, npc=3072):
    """V15: h1..h15 복소 (RMS, h배 위상 관례 X(h)=|X|e^{j(arg−h·arg V1)}) → 1주기 파형 (cos 기준).

    13.23: 차수마다 cos 를 부르던 것을 **역 FFT 한 번**으로 바꿨다 (15 x npc 개의 cos -> irfft).
    같은 값이다 (무작위 20판에서 최대 차이 6e-13 V) 고 10.4배 빠르다 — 이 함수가 `sim_harmonics`
    시간의 24% 였다.
    """
    X = np.zeros(npc // 2 + 1, dtype=np.complex128)
    X[1:16] = np.sqrt(2.0) * np.asarray(V15, dtype=np.complex128) * (npc / 2.0)
    return np.fft.irfft(X, n=npc)


def harmonics_from_wave(x, vref, nh=15):
    X = np.fft.rfft(x) / len(x) * 2 / np.sqrt(2); Vh = np.fft.rfft(vref)[1]
    h = np.arange(1, nh + 1)
    return np.abs(X[1:nh + 1]) * np.exp(1j * (np.angle(X[1:nh + 1]) - h * np.angle(Vh)))


def sim_harmonics(P, V15, params, tau=60e-6, npc=3072, ncyc=NCYC, measured=True):
    """
    고조파 소스 V15 (참 전압, 계측 영역 아니어야 함) → 전류 h1..h15 (RMS, h배 관례).
    measured=True 면 계측 RC(τ) 를 걸어 펌웨어 ih/ihdeg 와 같은 영역, False 면 회로 참전류 (결합 고정점용).
    """
    C, R, L0, Isat, Cx, rd, G = params
    v = wave_from_harmonics(V15, npc); vsrc = np.tile(v, ncyc); dt = 1.0 / (F * npc)
    Pm = max(P - G * np.mean(v ** 2), 0.5)
    I = _core12(vsrc, dt, Pm, C, max(R, 1e-3), L0, Isat, Cx, VF, max(rd, 1e-3), vsrc.max() - VF)
    if not np.all(np.isfinite(I)):
        return None
    Ic = I[-npc:] + G * v
    if measured and tau:
        Ic = rc_periodic(Ic, dt, tau)
    return harmonics_from_wave(Ic, v)
