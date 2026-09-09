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


def _MEAS_REF_RAD_PER_H(tau):
    """참 전압 → 계측 전압으로 위상 기준을 옮길 때 차수당 되돌릴 각 [rad] (13.68).

    계측 전압은 참 전압에 1극 RC 를 먹은 것이므로 h1 위상이 `−atan(2πFτ)` 만큼 뒤처진다.
    펌웨어 규약이 `−h·arg(V1)` 로 h 배 빼므로 어긋남은 정확히 h 에 비례한다. τ=60µs 에서 1.2958°/h.
    """
    return np.arctan(2 * np.pi * F * tau)


def harmonics_from_wave(x, vref, nh=15):
    X = np.fft.rfft(x) / len(x) * 2 / np.sqrt(2); Vh = np.fft.rfft(vref)[1]
    h = np.arange(1, nh + 1)
    return np.abs(X[1:nh + 1]) * np.exp(1j * (np.angle(X[1:nh + 1]) - h * np.angle(Vh)))


def sim_harmonics(P, V15, params, tau=60e-6, npc=3072, ncyc=NCYC, measured=True):
    """
    고조파 소스 V15 (참 전압, 계측 영역 아니어야 함) → 전류 h1..h15 (RMS, h배 관례).
    measured=True 면 계측 RC(τ) 를 걸어 펌웨어 ih/ihdeg 와 같은 영역, False 면 회로 참전류 (결합 고정점용).

    ⚠ **위상 기준은 계측 전압이다** (2026-09-09, 13.68). 펌웨어 규약은
    `ihdeg_h = arg(I_h) − h·arg(V1)` 이고 그 `V1` 은 **계측** 전압이다 — 계측 전압은 참 전압보다
    `atan(2πFτ)` 만큼 뒤처지므로, 계측 영역 전류를 참 전압 `v` 에 기준하면 **h 에 비례해** 어긋난다
    (τ=60µs 에서 1.2958°/h, h15 에서 19.44°. 크기는 안 변한다). 아래 `_MEAS_REF_RAD_PER_H` 회전이
    그것을 되돌린다. `measured=False` 는 참전류·참전압이라 회전이 없다 (자기 정합).

    근거 ① 순저항 부하의 실측 `ihdeg1` 이 −0.12~+0.06° 다 (포트·핫플·오븐·드라이기). 계측V 기준의
    예측이 0.00°, 참V 기준이면 −1.30° 라 규약이 갈린다. ② `sim_wave` 를 적합하는 `fit12` 는
    처음부터 계측 전압에 기준했다 (`harmonics_from_wave(Is, b['V'])`) — 두 입구가 어긋나 있었다.
    ③ `fit12` 는 `sim_harmonics` 를 안 부르고 위상을 목적함수에 넣은 적도 없다 (`np.abs` 로만 쓴다).
    그래서 **pkl 재적합은 필요 없다**. 원시 22파일 28구간에서 복소 상대오차 중앙값이
    h3 0.132 → 0.074, h5 0.189 → 0.083, h7 0.292 → 0.134, h9 0.352 → 0.173 으로 내렸다.
    h11+ 는 거의 안 내린다 — 거기 남은 것은 `wave_from_harmonics` 의 전압 h15 절단이고
    (통제: 전압을 h15 로 자른 `sim_wave` 가 이 값에 얹힌다), 텍스처 라이브러리에 h17+ 가 없어
    지금은 못 고친다.
    """
    C, R, L0, Isat, Cx, rd, G = params
    v = wave_from_harmonics(V15, npc); vsrc = np.tile(v, ncyc); dt = 1.0 / (F * npc)
    Pm = max(P - G * np.mean(v ** 2), 0.5)
    I = _core12(vsrc, dt, Pm, C, max(R, 1e-3), L0, Isat, Cx, VF, max(rd, 1e-3), vsrc.max() - VF)
    if not np.all(np.isfinite(I)):
        return None
    Ic = I[-npc:] + G * v
    if not (measured and tau):
        return harmonics_from_wave(Ic, v)
    Ic = rc_periodic(Ic, dt, tau)
    # 기준을 참 전압 v 에서 계측 전압 rc_periodic(v, dt, tau) 로 옮긴다. 둘의 h1 위상차가
    # 정확히 atan(2πFτ) 이므로 닫힌 꼴로 돌린다 — 되돌림 FFT 한 번을 아낀다 (같은 값이다,
    # `test_sim_harmonics_references_the_measured_voltage` 가 1e-9 로 묶어 둔다).
    X = harmonics_from_wave(Ic, v)
    return X * np.exp(1j * np.arange(1, len(X) + 1) * _MEAS_REF_RAD_PER_H(tau))
