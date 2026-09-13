# -*- coding: utf-8 -*-
"""장소 C 포트 원시 파형 — 순저항이라 `i(t) = v(t)/R` 이 **모든 표본에서** 성립한다.

이 자료가 한 번에 가르는 것 넷:
  1. v–i 표본 정렬   상관 최대 이동량 = 두 채널의 어긋남 (모델 없음, 도통 창 없음)
  2. 채널 전달       T_h = (I_h/I_1)/(V_h/V_1) — 순저항이면 모든 차수에서 1∠0
  3. ①a 의 바닥      절대 |I_h| — 장소 A·B 에서 잰 33.5mA∠+164° 가 여기서도 보이는가
  4. h17+ 텍스처     전류의 h17+ 가 전압의 h17+ 와 맞는가 (계통이냐 전단 인공물이냐)
"""
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.synthesis.fit_raw import halfwave, phasors, shift_samples, rc_filter, NS, F

STEMS = ("raw_elcetric_kettle_1", "raw_elcetric_kettle_2")
ODD = [1, 3, 5, 7, 9, 11, 13, 15]
HI = [17, 19, 21, 23, 25, 27, 29, 31]

print("=" * 100)
print("[1] 자료")
print("=" * 100)
pts = {}
for st in STEMS:
    d = pd.read_csv(f"data/{st}.csv")
    nc = int(d["cyc"].max()) + 1
    V = d["v_v"].to_numpy(float).reshape(nc, NS)
    I = d["i_a"].to_numpy(float).reshape(nc, NS)
    rng = d["range"].to_numpy(int).reshape(nc, NS)
    vm, im = np.median(V, 0), np.median(I, 0)
    vs, is_ = halfwave(vm), halfwave(im)
    p = float(np.mean(vs * is_))
    pts[st] = dict(v=vs, i=is_, vraw=vm, iraw=im, nc=nc, p=p, rng=rng, Vall=V, Iall=I)
    print(f"  {st:26s} 주기 {nc:3d}  P {p:8.1f}W  Irms {np.sqrt(np.mean(is_**2)):6.3f}A  "
          f"피크 {np.abs(is_).max():6.3f}A  HIGH {100*np.mean(rng!=0):5.1f}%  "
          f"Vrms {np.sqrt(np.mean(vs**2)):6.1f}V")
    print(f"  {'':26s} R = Vrms/Irms = {np.sqrt(np.mean(vs**2))/np.sqrt(np.mean(is_**2)):.2f}Ω  "
          f"주기 산포 {100*np.sqrt(np.mean(np.std(I,0)**2))/np.sqrt(np.mean(is_**2)):.1f}%")

print()
print("=" * 100)
print("[2] v–i 표본 정렬 — 순저항이므로 i 와 v 의 상관이 최대인 이동량이 곧 스큐")
print("=" * 100)
print("  (반파 대칭화 전/후 둘 다. 대칭화는 짝수차를 지우므로 전압 채널 인공물을 뺀 판이다)")
GRID = np.arange(-1.5, 1.55, 0.02)
for st in STEMS:
    for lab, v, i in (("원본", pts[st]["vraw"], pts[st]["iraw"]),
                      ("반파대칭", pts[st]["v"], pts[st]["i"])):
        best = (-9, 0)
        for n in GRID:
            c = float(np.corrcoef(v, shift_samples(i, -n))[0, 1])
            if c > best[0]:
                best = (c, float(n))
        print(f"  {st:26s} {lab:8s} 스큐 {best[1]:+6.3f} 표본 = {360*60*best[1]/15360:+6.3f}°/차수"
              f"   상관 {best[0]:.6f}")
# 주기별로도 (산포 확인)
for st in STEMS:
    ns = []
    for c in range(pts[st]["nc"]):
        v, i = halfwave(pts[st]["Vall"][c]), halfwave(pts[st]["Iall"][c])
        best = (-9, 0)
        for n in np.arange(-1.0, 1.02, 0.05):
            cc = float(np.corrcoef(v, shift_samples(i, -n))[0, 1])
            if cc > best[0]:
                best = (cc, float(n))
        ns.append(best[1])
    ns = np.array(ns)
    print(f"  {st:26s} 주기별 중앙값 {np.median(ns):+.3f} 표본  산포 {ns.std():.3f}  (n={len(ns)})")

print()
print("=" * 100)
print("[3] 채널 전달 T_h = (I_h/I_1)/(V_h/V_1) — 순저항이면 1∠0")
print("=" * 100)
for st in STEMS:
    Iph = phasors(pts[st]["i"], 63)
    Vph = phasors(pts[st]["v"], 63)
    T = (Iph / Iph[0]) / (Vph / Vph[0])
    print(f"  [{st}]")
    print("    차수  " + "".join(f"{h:8d}" for h in ODD))
    print("    |T|   " + "".join(f"{abs(T[h-1]):8.3f}" for h in ODD))
    print("    ∠T °  " + "".join(f"{np.degrees(np.angle(T[h-1])):8.2f}" for h in ODD))
    print("    |I|%  " + "".join(f"{100*abs(Iph[h-1]/Iph[0]):8.3f}" for h in ODD))
    print("    |V|%  " + "".join(f"{100*abs(Vph[h-1]/Vph[0]):8.3f}" for h in ODD))
    print("    |I|mA " + "".join(f"{1000*abs(Iph[h-1]):8.1f}" for h in ODD))

print()
print("=" * 100)
print("[4] h17+ — 전류(계통 참값)와 전압 채널이 맞는가")
print("=" * 100)
for st in STEMS:
    Iph = phasors(pts[st]["i"], 63)
    Vph = phasors(pts[st]["v"], 63)
    T = (Iph / Iph[0]) / (Vph / Vph[0])
    print(f"  [{st}]")
    print("    차수  " + "".join(f"{h:8d}" for h in HI))
    print("    |I|%  " + "".join(f"{100*abs(Iph[h-1]/Iph[0]):8.3f}" for h in HI))
    print("    |V|%  " + "".join(f"{100*abs(Vph[h-1]/Vph[0]):8.3f}" for h in HI))
    print("    |T|   " + "".join(f"{abs(T[h-1]):8.3f}" for h in HI))
    print("    ∠T °  " + "".join(f"{np.degrees(np.angle(T[h-1])):8.1f}" for h in HI))

print()
print("=" * 100)
print("[5] 짝수차 — 전압 채널 인공물 확인 (저항 전류의 짝수차는 바닥이어야)")
print("=" * 100)
for st in STEMS:
    Iph = phasors(pts[st]["iraw"], 20)
    Vph = phasors(pts[st]["vraw"], 20)
    print(f"  {st:26s} 차수 " + "".join(f"{h:8d}" for h in (2, 4, 6, 8, 10)))
    print(f"  {'':26s} |I|% " + "".join(f"{100*abs(Iph[h-1]/Iph[0]):8.3f}" for h in (2,4,6,8,10)))
    print(f"  {'':26s} |V|% " + "".join(f"{100*abs(Vph[h-1]/Vph[0]):8.3f}" for h in (2,4,6,8,10)))
