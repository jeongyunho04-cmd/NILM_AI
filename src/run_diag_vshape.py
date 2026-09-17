# -*- coding: utf-8 -*-
"""**전압 고조파(파형 모양)** 가 SMPS 도통 시점을 옮기나 (14.385, 사용자 지적).

14.384 에서 내가 잰 것은 두 가지였는데 **둘 다 이 가설을 못 건드린다**:
```
  ✗ Δ∠I 대 Δ∠V  — 전압의 **위상**을 따라가나. 기울기 −0.016
  ✗ vrms        — 전압의 **크기**. 한 녹화 안에서 부호가 갈림
  ⇒ 안 잰 것: 전압의 **모양** (THD_v · |V3|/|V1| · 평정도)
```
물리: SMPS 정류기는 `v(t) > V_dc` 일 때만 도통한다. 3·5차가 실리면 파형 꼭대기가
**평평해지고**, 그러면 그 문턱을 넘는 **시각이 밀린다**. 전류 펄스가 Δt 밀리면
모든 차수가 `h·ω·Δt` 로 돈다 — 관측된 그 모양이다.

그래서 **파형에서 직접** 도통 시점을 계산해서 k 와 견준다:
```
  v(t) 를 V_1..V_15 로 되살린다 -> 꼭대기 근처에서 0.90·Vpeak 를 넘는 시각 t_on
  ⇒ t_on 이 밀리면 k 도 같이 밀려야 한다 (부호까지)
```
"""
import csv
import glob
import numpy as np

ODD = np.array([3, 5, 7, 9, 11, 13], float)
#: 14.384 가 생 CSV 에서 잰 값
K = {"beam_projector_1.cal2": 0.0, "beam_projector_2": 0.82, "beam_projector_3": 7.57,
     "laptop_charger_1.cal2": 0.0, "laptop_charger_2": -4.54, "laptop_charger_3": -4.95,
     "laptop_charger_4": -5.39, "laptop_charger_5": -7.41, "laptop_charger_6": -13.29,
     "minipc_1.cal2": 0.0, "minipc_2": -0.61, "minipc_3": -1.51, "minipc_4": -2.43}
GRP = {"beam": "빔", "laptop": "충전기", "minipc": "미니PC"}


def vshape(stem):
    """그 녹화의 전압 파형 지표 — (THD_v, |V3|/|V1|, |V5|/|V1|, t_on[us], Vrms)."""
    fs = glob.glob("data/%s.csv" % stem)
    if not fs:
        return None
    VM, VD, VR = [], [], []
    with open(fs[0], encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        hd = next(r)
        c = {n: i for i, n in enumerate(hd)}
        if not all(("vh%d" % h) in c and ("vhdeg%d" % h) in c for h in range(1, 16)):
            return None
        for row in r:
            try:
                VM.append([float(row[c["vh%d" % h]]) for h in range(1, 16)])
                VD.append([float(row[c["vhdeg%d" % h]]) for h in range(1, 16)])
                VR.append(float(row[c["vrms"]]))
            except (ValueError, IndexError, KeyError):
                pass
    VM, VD, VR = np.asarray(VM), np.asarray(VD), np.asarray(VR)
    if len(VM) < 500:
        return None
    #: 녹화 대표 페이저 — 크기는 중앙, 위상은 원형 평균
    mag = np.median(VM, 0)
    ph = np.array([np.angle(np.exp(1j * np.radians(VD[:, i])).mean()) for i in range(15)])
    V = mag * np.exp(1j * ph)
    thd = float(np.sqrt((mag[1:] ** 2).sum()) / max(mag[0], 1e-9))
    #: 파형을 되살려 **0.90·Vpeak 를 넘는 시각**을 잰다
    t = np.linspace(0, 1, 20001)                       # 한 주기를 0~1 로
    v = np.zeros_like(t)
    for i in range(15):
        v += np.abs(V[i]) * np.cos(2 * np.pi * (i + 1) * t + np.angle(V[i]))
    pk = v.max()
    i_pk = int(np.argmax(v))
    seg = v[:i_pk + 1]
    idx = np.where(seg >= 0.90 * pk)[0]
    t_on = float(t[idx[0]]) / 60.0 * 1e6 if len(idx) else np.nan   # us
    return thd, float(mag[2] / max(mag[0], 1e-9)), float(mag[4] / max(mag[0], 1e-9)), \
        t_on, float(np.median(VR))


print("전압 **모양**이 도통 시점을 옮기나 — k 와 견준다\n")
print("%-24s %8s %8s %9s %9s %10s %8s"
      % ("녹화", "k[도/차]", "THD_v", "|V3|/|V1|", "|V5|/|V1|", "t_on[us]", "Vrms"))
rows = []
for s in sorted(K):
    d = vshape(s)
    if d is None:
        print("%-24s (전압 고조파 열 없음)" % s)
        continue
    thd, v3, v5, ton, vr = d
    print("%-24s %8.2f %8.4f %9.4f %9.4f %10.1f %8.1f"
          % (s, K[s], thd, v3, v5, ton, vr))
    rows.append((K[s], thd, v3, v5, ton, vr, s))

a = np.array([r[:6] for r in rows], float)
names = ["THD_v", "|V3|/|V1|", "|V5|/|V1|", "t_on", "vrms"]
print("\n전체 (n=%d)" % len(a))
for j, nm in enumerate(names, start=1):
    m = np.isfinite(a[:, 0]) & np.isfinite(a[:, j])
    if m.sum() > 2 and np.std(a[m, j]) > 0:
        print("  corr(k, %-10s) = **%+.3f**" % (nm, np.corrcoef(a[m, 0], a[m, j])[0, 1]))
print("\n기기별 (교락을 줄인다)")
for g, ko in GRP.items():
    b = np.array([r[:6] for r in rows if r[6].startswith(g)], float)
    if len(b) < 3:
        print("  %-8s n=%d — 못 잰다" % (ko, len(b)))
        continue
    out = []
    for j, nm in enumerate(names, start=1):
        if np.std(b[:, j]) > 0:
            out.append("%s %+.3f" % (nm, np.corrcoef(b[:, 0], b[:, j])[0, 1]))
    print("  %-8s n=%d | %s" % (ko, len(b), " · ".join(out)))
