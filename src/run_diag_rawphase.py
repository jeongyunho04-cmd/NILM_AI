# -*- coding: utf-8 -*-
"""회전이 **계측기에 있나 우리 처리에 있나** — 생 CSV 에서 바로 잰다 (14.384).

사용자 지적: *"펌웨어 adc는 문제가 없을텐데 복합이랑 단독 녹화를 더 분석해봐"*.
그러면 갈라야 할 것은 이것이다:
```
  생 CSV (ihdeg)  ->  numpy_exporter  ->  npz  ->  SegmentPool(net_ 배경 제거)  ->  sig
  ^^^^^^^^^^^^^^                                    ^^^^^^^^^^^^^^^^^^^^^^^^
  계측기가 준 것                                      **우리가 손댄 것**
```
같은 기기의 녹화 둘을 **생 CSV 의 ihdeg 로만** 견준다. 거기에 이미 회전이 있으면
계측 쪽이고, 없으면 **우리 처리가 넣은 것**이다.

⚠ 통전 구간만 본다 (irms 가 그 녹화 중앙의 절반 이상). 꺼진 구간의 위상은 잡음이다.
"""
import csv
import glob
import numpy as np

ODD = np.array([3, 5, 7, 9, 11, 13], float)
GROUPS = {
    "빔": ["beam_projector_1.cal2", "beam_projector_2", "beam_projector_3"],
    "충전기": ["laptop_charger_1.cal2", "laptop_charger_2", "laptop_charger_3",
             "laptop_charger_4", "laptop_charger_5", "laptop_charger_6"],
    "미니PC": ["minipc_1.cal2", "minipc_2", "minipc_3", "minipc_4"],
}


def read(stem):
    fs = glob.glob("data/%s.csv" % stem)
    if not fs:
        return None
    mag, deg, irms, vmag, vdeg = [], [], [], [], []
    with open(fs[0], encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        hd = next(r)
        c = {n: i for i, n in enumerate(hd)}
        need = ["ih%d" % h for h in range(1, 16)] + ["ihdeg%d" % h for h in range(1, 16)]
        if not all(k in c for k in need + ["irms"]):
            return None
        hasv = all(("vh%d" % h) in c and ("vhdeg%d" % h) in c for h in range(1, 16))
        for row in r:
            try:
                mag.append([float(row[c["ih%d" % h]]) for h in range(1, 16)])
                deg.append([float(row[c["ihdeg%d" % h]]) for h in range(1, 16)])
                irms.append(float(row[c["irms"]]))
                if hasv:
                    vmag.append([float(row[c["vh%d" % h]]) for h in range(1, 16)])
                    vdeg.append([float(row[c["vhdeg%d" % h]]) for h in range(1, 16)])
            except (ValueError, IndexError):
                pass
    mag = np.asarray(mag); deg = np.asarray(deg); irms = np.asarray(irms)
    on = irms > 0.5 * np.median(irms[irms > 0]) if (irms > 0).any() else irms > 0
    if on.sum() < 200:
        return None
    I = mag * np.exp(1j * np.radians(deg))
    V = (np.asarray(vmag) * np.exp(1j * np.radians(np.asarray(vdeg)))) if vmag else None
    return I[on], (V[on] if V is not None else None), int(on.sum())


def circang(z):
    return np.array([np.angle(np.exp(1j * np.angle(z[:, int(h) - 1])).mean()) for h in ODD])


def slope(d):
    d = np.degrees(np.unwrap(np.radians(d)))
    k = float((ODD @ d) / (ODD @ ODD))
    r = d - k * ODD
    return k, 1.0 - (r ** 2).sum() / max(((d - d.mean()) ** 2).sum(), 1e-12)


print("생 CSV 의 `ihdeg` 만으로 — 풀·배경제거·정제를 **통과하지 않은** 값\n")
for ko, stems in GROUPS.items():
    got = {}
    for s in stems:
        d = read(s)
        if d is not None:
            got[s] = d
    if len(got) < 2:
        print("  %-8s 읽은 녹화 %d개\n" % (ko, len(got)))
        continue
    fs = list(got)
    ref = fs[0]
    ai = {f: circang(got[f][0]) for f in fs}
    av = {f: (circang(got[f][1]) if got[f][1] is not None else None) for f in fs}
    print("  %-8s 기준 **%s** (%d사이클)" % (ko, ref, got[ref][2]))
    for f in fs[1:]:
        di = np.degrees(np.angle(np.exp(1j * (ai[f] - ai[ref]))))
        k, ss = slope(di)
        vtxt = ""
        if av[f] is not None and av[ref] is not None:
            dv = np.degrees(np.angle(np.exp(1j * (av[f] - av[ref]))))
            kv, sv = slope(dv)
            vtxt = " | **∠V** k=%+6.2f R²=%+.2f" % (kv, sv)
        print("     %-24s ∠I k=%+7.2f 도/차수 (Δt %+7.1f us) R²=%+.3f%s"
              % (f, k, k / 360.0 / 60.0 * 1e6, ss, vtxt))
    print()
