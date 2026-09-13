# -*- coding: utf-8 -*-
"""사용자가 준 LTspice 넷리스트(ADC 앞단)에서 세 채널의 전달함수를 직접 계산한다.

넷리스트 해독
-------------
바이어스   R2/R3 분압(3.3V -> 1.65V) -> U2 팔로워 -> **N005** = 세 채널 공통 기준
전류       CT(SCT-013-100) -> 버든 R8 27Ω (N003-N005)
           U1 비반전  이득 1 + R9/R10 = 1 + 6.69k/2.7k = 3.478   -> N002
  HIGH     N002 -> R7 1k -> C2 100n -> ADC_high            RC 1극 (fc 1591.55)
  LOW      N002 -> R4 5.04k -> U3 반전 이득 −R5/R4 = −74.8k/5.04k = −14.84 -> N008
           N008 -> R6 1k -> C5 100n -> ADC_low             RC 1극 (fc 1591.55)
전압       ZMPT-101B -> 버든 R12 220Ω (N011-N005)
           U4 비반전  이득 1 + R14/R15 = 1 + 27k/2.2k = 13.27  -> N010
           N010 -> R11 1k -> C4 100n -> ADC_V              RC 1극 (fc 1591.55)

**핵심: LOW 는 N002(U1 출력)에서 갈라진다 — ADC_high 노드(R7·C2 뒤)가 아니다.**
그래서 세 채널이 각각 1k·100n 을 정확히 **하나씩** 지난다.
"""
import numpy as np

GBW = 1.0e6          # MCP6001 이득대역폭 [Hz] (데이터시트 typ 1MHz)
FC_RC = 1.0 / (2 * np.pi * 1e3 * 100e-9)


def opamp_pole(noise_gain: float) -> float:
    """폐루프 −3dB 대역 = GBW / 잡음이득."""
    return GBW / noise_gain


def H(f, poles, gain=1.0):
    y = np.full(np.shape(f), complex(gain))
    for fp in poles:
        y = y / (1.0 + 1j * np.asarray(f) / fp)
    return y


NG_U1 = 1 + 6.69 / 2.7          # 비반전
NG_U3 = 1 + 74.8 / 5.04         # 반전이지만 잡음이득은 1 + Rf/Rin
NG_U4 = 1 + 27.0 / 2.2

print(f"RC 극        {FC_RC:.2f} Hz   (1kΩ · 100nF, 세 채널 공통)")
print(f"U1 잡음이득  {NG_U1:.3f}  -> 폐루프 극 {opamp_pole(NG_U1) / 1e3:7.1f} kHz")
print(f"U3 잡음이득  {NG_U3:.3f}  -> 폐루프 극 {opamp_pole(NG_U3) / 1e3:7.1f} kHz")
print(f"U4 잡음이득  {NG_U4:.3f}  -> 폐루프 극 {opamp_pole(NG_U4) / 1e3:7.1f} kHz")
print()

P_V = [FC_RC, opamp_pole(NG_U4)]
P_HI = [FC_RC, opamp_pole(NG_U1)]
P_LO = [FC_RC, opamp_pole(NG_U1), opamp_pole(NG_U3)]

h = np.arange(1, 16)
f = 60.0 * h
print("차수별 위상 [°] — 기본파 기준으로 재규격화하기 전의 절대 위상")
print(f"  {'h':>3s} {'f[Hz]':>7s} {'전압':>9s} {'HIGH':>9s} {'LOW':>9s} "
      f"{'LOW−전압':>10s} {'LOW−HIGH':>10s}")
for i, hh in enumerate(h):
    pv = np.degrees(np.angle(H(f[i], P_V)))
    ph = np.degrees(np.angle(H(f[i], P_HI)))
    pl = np.degrees(np.angle(H(f[i], P_LO)))
    print(f"  {hh:3d} {f[i]:7.0f} {pv:8.3f}° {ph:8.3f}° {pl:8.3f}° "
          f"{pl - pv:9.3f}° {pl - ph:9.3f}°")

print()
print("우리 관례 `arg(I_h) − h·arg(V_1)` 에서 남는 계측 오차")
print("  (전류 채널 위상 − h × 전압 채널 기본파 위상) − (참값에서의 같은 양)")
pv1 = np.angle(H(60.0, P_V))
for lab, P in (("HIGH", P_HI), ("LOW", P_LO)):
    err = np.degrees(np.angle(H(f, P)) - h * pv1) - np.degrees(
        np.angle(H(f, [FC_RC])) - h * np.angle(H(60.0, [FC_RC])))
    print(f"  {lab:5s} 1극 모형 대비 잔차: " + " ".join(f"h{hh}{v:+.3f}" for hh, v in zip(h[::2], err[::2])))

print()
print("60Hz 에서 LOW−HIGH 위상차 = %.4f°  (12.184.16 이 설명하려 한 2.18° 와 견주라)"
      % (np.degrees(np.angle(H(60.0, P_LO)) - np.angle(H(60.0, P_HI)))))
print("LOW/HIGH 이득비 = %.3f  (교정 상수. 주파수 의존은 U3 극 %.0f kHz 뿐)"
      % (14.84, opamp_pole(NG_U3) / 1e3))
