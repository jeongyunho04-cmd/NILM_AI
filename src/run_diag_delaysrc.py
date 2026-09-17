# -*- coding: utf-8 -*-
"""**그 지연이 어디서 오나** — 사전 쪽인가 녹화 쪽인가 (14.383).

14.382 가 잰 것: 격리 지문 대비 실측 위상이 `Δ∠(h) = k·h` 로 어긋나고 R² 0.81~0.99,
빼고 나면 ±1~5도. 이제 **k 의 출처**를 가른다.

```
  ⓐ 사전 쪽   격리 녹화마다 위상 기준이 다르면, 같은 기기의 **녹화 둘을 서로**
              견줘도 k 가 나온다. 나오면 사전이 이미 흔들리는 것이다
  ⓑ 녹화 쪽   트리거(영교차) 타이밍이 어긋나면 **전압 고조파도 같은 k** 로 돈다.
              전류에만 있으면 트리거가 아니다
  ⓒ 주파수    60Hz 가 아니면 고정 창에서 위상이 미끄러진다. `freq` 를 본다
  ⓓ 시간 표류  한 녹화 안에서 k 가 움직이면 클럭 드리프트다
```
"""
import glob
import numpy as np
from src import env_guard  # noqa: F401
from src.model import inputs as _I  # noqa: E402
from src.model.postproc import SMPS_GROUP  # noqa: E402
from src.model.realdata import RealWindows  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
KO = {"beam_projector": "빔", "laptop_charger": "충전기", "minipc": "미니PC"}
ODD = np.array([3, 5, 7, 9, 11, 13], float)


def slope(hs, deg):
    """Δ∠ = k·h 최소자승 (원점 통과) 과 R²."""
    d = np.degrees(np.unwrap(np.radians(np.asarray(deg, float))))
    k = float((hs @ d) / (hs @ hs))
    r = d - k * hs
    ss = 1.0 - (r ** 2).sum() / max(((d - d.mean()) ** 2).sum(), 1e-12)
    return k, ss


def slope_hm1(hs, deg):
    """Δ∠ = k·(h−1) — V₁ 위상을 0 으로 정규화한 계면 이쪽이 맞는 모형이다."""
    d = np.degrees(np.unwrap(np.radians(np.asarray(deg, float))))
    x = hs - 1.0
    k = float((x @ d) / (x @ x))
    r = d - k * x
    ss = 1.0 - (r ** 2).sum() / max(((d - d.mean()) ** 2).sum(), 1e-12)
    return k, ss


print("=" * 100)
print("ⓐ **사전 쪽** — 같은 기기의 격리 녹화끼리 견준다 (기준 = 그 기기 첫 녹화)\n")
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
for a in [x for x in APPS if x in SMPS_GROUP]:
    rec = {}
    for act in pool.appliance_activations.get(a, []):
        hh = np.asarray(act.net_harmonics_ri, np.float64)
        pp = np.asarray(act.target_power_w, np.float64)
        if hh.shape[0] != len(pp):
            continue
        m = pp > 1.0
        if m.sum() < 200:
            continue
        c = (hh[m, :, 0] + 1j * hh[m, :, 1])
        rec.setdefault(act.source_file, []).append(c)
    if len(rec) < 2:
        print("  %-8s 녹화가 %d개라 못 잰다" % (KO[a], len(rec)))
        continue
    files = sorted(rec)
    ref = np.concatenate(rec[files[0]])
    refang = np.array([np.angle(np.exp(1j * np.angle(ref[:, int(h) - 1])).mean())
                       for h in ODD])
    print("  %-8s 녹화 %d개 · 기준 %s" % (KO[a], len(files), files[0][:30]))
    ks = []
    for f in files[1:]:
        cc = np.concatenate(rec[f])
        ang = np.array([np.angle(np.exp(1j * np.angle(cc[:, int(h) - 1])).mean())
                        for h in ODD])
        k, ss = slope(ODD, np.degrees(ang - refang))
        ks.append(k)
        print("     %-32s k=%+6.2f 도/차수 (Δt %+7.1f us) R²=%.3f"
              % (f[:32], k, k / 360.0 / 60.0 * 1e6, ss))
    if ks:
        print("     ⇒ 격리 녹화 사이 k 폭 **%.2f ~ %.2f 도/차수** (σ %.2f)\n"
              % (min(ks), max(ks), float(np.std(ks))))

print("=" * 100)
print("ⓑ **녹화 쪽** — 복합 녹화의 **전압** 고조파도 같은 기울기로 도나\n")
nv = len(_I.VOLT_ORDERS)
VO = np.array([h for h in _I.VOLT_ORDERS if h >= 3], float)
vk = {}
for st in ("test_1", "test_2", "test_3", "test_4", "test_5"):
    z = np.load("processed_data/composite_eval/%s.npz" % st, allow_pickle=True)
    x = RealWindows._to_33ch(z).astype(np.float64)
    ang, mag = [], []
    for s_, h in enumerate(_I.VOLT_ORDERS):
        if h < 3:
            continue
        v = x[33 + s_] + 1j * x[33 + nv + s_]
        ang.append(np.degrees(np.angle(np.exp(1j * np.angle(v)).mean())))
        mag.append(float(np.median(np.abs(v))))
    k, ss = slope(VO, ang)
    k1, ss1 = slope_hm1(VO, ang)
    vk[st] = k
    print("  %-8s ∠V_h = %s" % (st, " ".join("h%d %+6.1f" % (h, a_)
                                             for h, a_ in zip(VO.astype(int), ang))))
    print("  %-8s   k·h 맞춤 k=%+6.2f R²=%.3f  /  k·(h−1) 맞춤 k=%+6.2f R²=%.3f"
          % ("", k, ss, k1, ss1))

print("\n" + "=" * 100)
print("ⓒ **주파수** — 60Hz 가 아니면 고정 창에서 위상이 미끄러진다\n")
for st in ("test_1", "test_2", "test_3", "test_4", "test_5"):
    fs = sorted(glob.glob("data/%s.csv" % st))
    if not fs:
        continue
    import csv
    with open(fs[0], encoding="utf-8", newline="") as fh:
        r = csv.reader(fh)
        hd = next(r)
        if "freq" not in hd:
            print("  %-8s freq 열 없음" % st)
            continue
        i = hd.index("freq")
        v = []
        for row in r:
            try:
                v.append(float(row[i]))
            except (ValueError, IndexError):
                pass
    v = np.asarray(v)
    #: 60Hz 에서 벗어난 만큼 * 한 창 길이 = 위상 미끄러짐
    df = float(np.median(v)) - 60.0
    print("  %-8s f 중앙 **%.4f Hz** (60 대비 %+.4f) · 폭 %.4f · "
          "1주기창에서 h차 미끄러짐 = h * %.3f 도"
          % (st, np.median(v), df, float(v.max() - v.min()), 360.0 * df / 60.0))
