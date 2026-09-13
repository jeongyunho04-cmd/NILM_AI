#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""원시 파형 캡처(raw_*.csv, 프로토콜 v2)에서 레인지별 위상 교정값을 계산한다.

    python phase_cal.py data/raw_20260905_204158.csv [...]

출력값은 펌웨어 nilm_dsp.h 의 NILM_CAL_DEFAULT_LOW_DEG / _HIGH_DEG 에 그대로
적는 값이다. 즉 C = angle(I1) - angle(V1) 이며, CSV/프레임의 phase_deg 와는
부호가 반대다(그쪽은 angle(V1) - angle(I1)).

[왜 원시 파형이 더 정확한가]
WiFi 로그의 phase_deg 로 재려면 "지금 적용 중인 상수 - 잔차"를 해야 하는데,
그러려면 그 순간 어느 레인지의 상수가 적용됐는지도 알아야 한다. 원시 캡처는
ADC 코드 그대로라 교정이 하나도 안 걸린 절대값을 바로 구할 수 있고, 게다가
한 채널만으로 전 구간을 재구성할 수 있어 "레인지가 안 섞인" 측정이 된다.
주기간 산포가 0.02도 수준으로 나온다.

[주의: 레인지 순도]
LOW 는 피크 약 2.2A 에서 레일에 닿으므로, LOW 값을 재려면 부하가 그보다
작아야 한다(약 1.5A rms 이하. 인두, 작은 히터). HIGH 는 전 구간 유효하지만
CT 위상오차가 전류에 따라 달라지므로, 실제로 HIGH 레인지로 동작할 크기의
부하(2.9A rms 이상)로 재는 것이 맞다.
"""
import sys, numpy as np, pandas as pd

N = 256
LSB = 3.3 / 16384.0
ADC_MID = 8192
CLIP = int(1.55 / 3.3 * 16384.0)                 # 7695
SENS_HIGH = 27.0 * (1.0 + 6800/2700) / 2000.0    # 0.0475 V/A
LOW_TRIM = 1.00580                              # nilm_acq.c 의 NILM_LOW_TRIM
SENS_LOW = SENS_HIGH * (75000/5100) * LOW_TRIM  # 0.70258 V/A
VOLT_SCALE = 309.34     # 2026-09-05 무부하 트림 (그 전 캡처는 307.9)
V_INTERP_F = 0.14549          # 랭크1 -> 랭크2 시간차 / 샘플 주기


def analyse(path):
    d = pd.read_csv(path)
    if "bias" not in d.columns:
        print(f"{path}: v1(차동) 캡처라 이 스크립트로는 못 읽습니다"); return
    nc = len(d) // N
    R = lambda c: np.asarray(d[c], float).reshape(nc, N)
    lo, v, hi, bi = R("low"), R("v"), R("high"), R("bias")
    ofl = ADC_MID + d.off_low.iloc[0]
    ofh = ADC_MID + d.off_high.iloc[0]
    ofv = ADC_MID + d.off_volt.iloc[0]

    b = bi.mean(1, keepdims=True)                        # 주기평균 바이어스(펌웨어와 동일)
    i_lo = ((ADC_MID + (lo - b)) - ofl) * LSB / SENS_LOW    # 랭크1 시각
    i_hi = ((ADC_MID - (hi - b)) - ofh) * LSB / SENS_HIGH   # 랭크2 시각, 부호 반전
    v_r1 = ((ADC_MID + (v - b)) - ofv) * LSB * VOLT_SCALE   # 랭크1 시각
    vn = np.concatenate([v[:, 1:], 2 * v[:, -1:] - v[:, -2:-1]], 1)
    v_r2 = ((ADC_MID + ((v + V_INTERP_F * (vn - v)) - b)) - ofv) * LSB * VOLT_SCALE

    lo_clip = (np.abs(lo - ADC_MID) >= CLIP).any(1)      # 그 주기에 LOW 가 레일에 닿았나
    hi_clip = (np.abs(hi - ADC_MID) >= CLIP).any(1)
    amp = np.abs(np.fft.rfft(i_hi, axis=1)[:, 1]) / (N / 2)
    live = amp > 0.5 * amp.max()                          # 부하가 켜진 주기
    edge = np.zeros(nc, bool)                             # 켜짐/꺼짐 전환 접점
    edge[:-1] |= live[:-1] != live[1:]; edge[1:] |= live[:-1] != live[1:]

    print("=" * 70)
    print(f"{path}   주기 {nc}개  (부하 ON {live.sum()}, 전환접점 {edge.sum()})")
    for name, cur, volt, bad in (("LOW ", i_lo, v_r1, lo_clip),
                                 ("HIGH", i_hi, v_r2, hi_clip)):
        use = live & ~edge & ~bad
        if use.sum() < 3:
            why = "부하가 너무 커서 레일에 닿음" if bad[live & ~edge].any() else "쓸 주기 부족"
            print(f"  {name}: 측정 불가 ({why}, 쓸 수 있는 주기 {use.sum()}개)")
            continue
        I1 = np.fft.rfft(cur[use], axis=1)[:, 1]
        V1 = np.fft.rfft(volt[use], axis=1)[:, 1]
        z = I1 * np.conj(V1)
        C = np.degrees(np.angle(z.sum()))
        ang = np.degrees(np.angle(z))
        irms = np.sqrt((cur[use] ** 2).mean())
        # 그 주기들이 실제로 이 레인지 하나로만 돌았는지
        pure = float(np.mean((np.abs(lo[use] - ADC_MID) < CLIP))) * 100
        print(f"  {name}: C = {C:+7.3f} deg   (주기 {use.sum()}개, 산포 {ang.std():.3f} deg, "
              f"Irms {irms:.3f} A)")
        if name == "HIGH" and pure < 99.0:
            print(f"        주의: 이 주기들은 실제 동작 시 LOW/HIGH 가 섞인다"
                  f"(샘플의 {pure:.0f}% 만 LOW 유효). HIGH 전용 재구성으로 낸 값이라")
            print(f"        위상 자체는 유효하지만, CT 위상오차는 전류에 따라 변하므로")
            print(f"        실제 HIGH 로 라벨될 크기(2.9A rms 이상)로 다시 재는 편이 낫다.")
    print("  -> nilm_dsp.h 의 NILM_CAL_DEFAULT_LOW_DEG / _HIGH_DEG 에 그대로 적는다")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        analyse(p)
