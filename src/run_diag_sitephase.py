# -*- coding: utf-8 -*-
"""**자리별 전압 텍스처 대 SMPS 회전** — 세션 표를 써서 교락을 깬다 (14.386).

사용자 지적: *"전압 텍스처가 크게 다른 자리에서 같은 기기를 녹화한 판 이미 존재해"*.
맞다 — `file_registry.SITE_SESSIONS` 가 그 표다.
```
  D1  vh3 0.70%  vh5 1.70  vh7 0.99  216.5V   충전기_1 · 미니PC_1
  D2  vh3 0.73%  vh5 1.43  vh7 1.13  215.8V   빔_3
  E1  vh3 3.01%  vh5 1.84  vh7 0.13  229.0V   빔_1 · 빔_2 · 충전기_2
  E2  vh3 4.01%  vh5 1.26  vh7 0.30  230.7V   미니PC_2 · 미니PC_3
  E3  vh3 3.42%  vh5 1.08  vh7 0.37  229.5V   충전기_3·4·5·6 · 미니PC_4
```
★ 교락을 깨는 자리가 둘이다:
```
  ① **자리 사이** D vs E — vh3 이 4~6배 갈린다. 그런데 vrms 도 같이 갈린다 (216 vs 229)
  ② ★★ **E 안에서** E1/E2/E3 — vrms 는 229.0/230.7/229.5 로 **1.7V 안**인데
     vh3 은 3.01/4.01/3.42 로 **33% 폭**이다. 여기가 전압 크기를 고정하고
     **모양만** 흔드는 자리다
```
"""
import csv
import glob
import numpy as np

ODD = np.array([3, 5, 7, 9, 11, 13], float)
SESS = {
    "D1": dict(vh3=0.70, vh5=1.70, vh7=0.99, v=216.5),
    "D2": dict(vh3=0.73, vh5=1.43, vh7=1.13, v=215.8),
    "E1": dict(vh3=3.01, vh5=1.84, vh7=0.13, v=229.0),
    "E2": dict(vh3=4.01, vh5=1.26, vh7=0.30, v=230.7),
    "E3": dict(vh3=3.42, vh5=1.08, vh7=0.37, v=229.5),
}
OF = {"laptop_charger_1": "D1", "minipc_1": "D1", "beam_projector_3": "D2",
      "beam_projector_1": "E1", "beam_projector_2": "E1", "laptop_charger_2": "E1",
      "minipc_2": "E2", "minipc_3": "E2",
      "laptop_charger_3": "E3", "laptop_charger_4": "E3", "laptop_charger_5": "E3",
      "laptop_charger_6": "E3", "minipc_4": "E3"}
DEV = {"beam_projector": "빔", "laptop_charger": "충전기", "minipc": "미니PC"}


def ang_of(stem):
    fs = glob.glob("data/%s.csv" % stem) + glob.glob("data/%s.cal2.csv" % stem)
    if not fs:
        return None
    M, D, IR = [], [], []
    with open(fs[0], encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        hd = next(r)
        c = {n: i for i, n in enumerate(hd)}
        for row in r:
            try:
                M.append([float(row[c["ih%d" % h]]) for h in range(1, 16)])
                D.append([float(row[c["ihdeg%d" % h]]) for h in range(1, 16)])
                IR.append(float(row[c["irms"]]))
            except (ValueError, IndexError, KeyError):
                pass
    M, D, IR = np.asarray(M), np.asarray(D), np.asarray(IR)
    on = IR > 0.5 * np.median(IR[IR > 0]) if (IR > 0).any() else IR > 0
    if on.sum() < 300:
        return None
    I = M[on] * np.exp(1j * np.radians(D[on]))
    return np.array([np.angle(np.exp(1j * np.angle(I[:, int(h) - 1])).mean()) for h in ODD]), int(on.sum())


def slope(d):
    d = np.degrees(np.unwrap(np.radians(d)))
    k = float((ODD @ d) / (ODD @ ODD))
    r = d - k * ODD
    return k, 1.0 - (r ** 2).sum() / max(((d - d.mean()) ** 2).sum(), 1e-12)


A = {}
for s in OF:
    a = ang_of(s)
    if a:
        A[s] = a

print("① **같은 기기 · 자리를 건너뛴** 대조 — vh3 이 4~6배 갈린다 (vrms 도 갈린다)\n")
print("%-8s %-22s -> %-22s %9s %9s %9s %8s"
      % ("기기", "기준(세션)", "비교(세션)", "Δvh3[%]", "Δvrms[V]", "k[도/차]", "R²"))
rows = []
for dev, ko in DEV.items():
    ss = [s for s in A if s.startswith(dev)]
    for i in range(len(ss)):
        for j in range(len(ss)):
            if i >= j:
                continue
            a, b = ss[i], ss[j]
            sa, sb = OF[a], OF[b]
            if sa == sb:
                continue
            k, r2 = slope(np.degrees(np.angle(np.exp(1j * (A[b][0] - A[a][0])))))
            dv3 = SESS[sb]["vh3"] - SESS[sa]["vh3"]
            dvr = SESS[sb]["v"] - SESS[sa]["v"]
            rows.append((ko, dv3, dvr, k, r2, sa, sb))
            print("%-8s %-22s -> %-22s %9.2f %9.1f %9.2f %8.3f"
                  % (ko, "%s(%s)" % (a, sa), "%s(%s)" % (b, sb), dv3, dvr, k, r2))

print("\n② ★★ **E 안에서만** — vrms 는 1.7V 안에 고정, vh3 만 3.01~4.01 로 흔들린다\n")
print("%-8s %-22s -> %-22s %9s %9s %9s %8s"
      % ("기기", "기준(세션)", "비교(세션)", "Δvh3[%]", "Δvrms[V]", "k[도/차]", "R²"))
e_rows = []
for dev, ko in DEV.items():
    ss = [s for s in A if s.startswith(dev) and OF[s].startswith("E")]
    for i in range(len(ss)):
        for j in range(len(ss)):
            if i >= j or OF[ss[i]] == OF[ss[j]]:
                continue
            a, b = ss[i], ss[j]
            k, r2 = slope(np.degrees(np.angle(np.exp(1j * (A[b][0] - A[a][0])))))
            dv3 = SESS[OF[b]]["vh3"] - SESS[OF[a]]["vh3"]
            dvr = SESS[OF[b]]["v"] - SESS[OF[a]]["v"]
            e_rows.append((ko, dv3, dvr, k, r2))
            print("%-8s %-22s -> %-22s %9.2f %9.1f %9.2f %8.3f"
                  % (ko, "%s(%s)" % (a, OF[a]), "%s(%s)" % (b, OF[b]), dv3, dvr, k, r2))

print("\n③ **같은 세션 안** 대조 — 전압이 완전히 같다. 여기 k 가 **기기 고유 흔들림**이다\n")
same = []
for dev, ko in DEV.items():
    ss = sorted([s for s in A if s.startswith(dev)])
    for i in range(len(ss)):
        for j in range(len(ss)):
            if i >= j or OF[ss[i]] != OF[ss[j]]:
                continue
            k, r2 = slope(np.degrees(np.angle(np.exp(1j * (A[ss[j]][0] - A[ss[i]][0])))))
            same.append(abs(k))
            print("  %-8s %-20s -> %-20s (%s)  k=%+7.2f  R²=%+.3f"
                  % (ko, ss[i], ss[j], OF[ss[i]], k, r2))

print("\n" + "=" * 96)
a1 = np.array([(r[1], r[3]) for r in rows], float)
if len(a1) > 2:
    print("  ① 자리 건너뜀 n=%d — corr(k, Δvh3) **%+.3f** · 기울기 **%+.2f 도/차수/%%vh3**"
          % (len(a1), np.corrcoef(a1[:, 0], a1[:, 1])[0, 1], np.polyfit(a1[:, 0], a1[:, 1], 1)[0]))
a2 = np.array([(r[1], r[3]) for r in e_rows], float)
if len(a2) > 2:
    print("  ② E 안 (vrms 고정) n=%d — corr(k, Δvh3) **%+.3f** · 기울기 **%+.2f**"
          % (len(a2), np.corrcoef(a2[:, 0], a2[:, 1])[0, 1], np.polyfit(a2[:, 0], a2[:, 1], 1)[0]))
if same:
    print("  ③ 같은 세션 안 |k| — 중앙 **%.2f** · 최대 **%.2f** (n=%d)  <- 전압으로 설명 불가"
          % (float(np.median(same)), float(np.max(same)), len(same)))
