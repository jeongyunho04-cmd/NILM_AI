# -*- coding: utf-8 -*-
"""전류 위상 어긋남이 **그 녹화의 전압 위상**으로 설명되나 (14.383).

여기까지 잰 것:
```
  ⓐ 같은 기기의 격리 녹화끼리도 Δ∠I(h) = k·h 로 어긋난다 (충전기 k 폭 8.4 도/차수)
  ⓑ 복합 녹화의 ∠V_h 에는 그런 기울기가 없다 (R² 음수~0.4)
  ⓒ range·freq·pll·irms 와 상관 없음 · 날짜와도 기기마다 부호가 다름
```
남은 가설: **물리다.** 비선형 부하의 h차 전류 위상은 인가 전압의 h차 위상을 따라간다.
녹화마다 ∠V_h 가 다르면 ∠I_h 도 달라진다 — 그것은 기준 오류가 아니라 **실재**이고,
`V_h` 가 이미 모델 입력(채널 33~48)이므로 **배울 수 있다**.

    Δ∠I_h  대  Δ∠V_h   (같은 기기, 녹화 둘 사이)
    기울기 ~1 이면 전압을 그대로 따라간다 · ~0 이면 전압과 무관한 기준 오류다
"""
import numpy as np
from src import env_guard  # noqa: F401
from src.model.postproc import SMPS_GROUP  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
KO = {"beam_projector": "빔", "laptop_charger": "충전기", "minipc": "미니PC"}
ODD = [3, 5, 7, 9, 11, 13]

pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
print("Δ∠I_h 가 Δ∠V_h 를 따라가나 — 같은 기기의 녹화 쌍\n")
allx, ally = [], []
for a in [x for x in APPS if x in SMPS_GROUP]:
    rec = {}
    for act in pool.appliance_activations.get(a, []):
        vh = getattr(act, "net_voltage_harmonics_complex", None)
        if vh is None:
            continue
        hh = np.asarray(act.net_harmonics_ri, np.float64)
        pp = np.asarray(act.target_power_w, np.float64)
        vh = np.asarray(vh)
        if hh.shape[0] != len(pp) or len(vh) != len(pp):
            continue
        m = pp > 1.0
        if m.sum() < 200:
            continue
        ic = hh[m, :, 0] + 1j * hh[m, :, 1]
        d = rec.setdefault(act.source_file, {"i": [], "v": []})
        d["i"].append(ic)
        d["v"].append(vh[m])
    if len(rec) < 2:
        print("  %-8s 전압을 실은 녹화가 %d개라 못 잰다\n" % (KO[a], len(rec)))
        continue
    files = sorted(rec)
    ang = {}
    for f in files:
        ic = np.concatenate(rec[f]["i"])
        vc = np.concatenate(rec[f]["v"])
        ang[f] = (
            np.array([np.angle(np.exp(1j * np.angle(ic[:, h - 1])).mean()) for h in ODD]),
            np.array([np.angle(np.exp(1j * np.angle(vc[:, h - 1])).mean()) for h in ODD]))
    print("  %-8s 녹화 %d개 — 기준 %s" % (KO[a], len(files), files[0][:28]))
    print("     %-28s %s" % ("녹화", " ".join("%14s" % ("h%d ΔI/ΔV" % h) for h in ODD)))
    for f in files[1:]:
        di = np.degrees(np.angle(np.exp(1j * (ang[f][0] - ang[files[0]][0]))))
        dv = np.degrees(np.angle(np.exp(1j * (ang[f][1] - ang[files[0]][1]))))
        print("     %-28s %s" % (f[:28],
                                 " ".join("%+6.1f/%+6.1f" % (i_, v_) for i_, v_ in zip(di, dv))))
        allx.extend(dv.tolist())
        ally.extend(di.tolist())
    print()

x, y = np.asarray(allx), np.asarray(ally)
m = np.isfinite(x) & np.isfinite(y)
if m.sum() > 3:
    sl = float((x[m] @ y[m]) / (x[m] @ x[m]))
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    print("=" * 84)
    print("  전체 %d 칸 — **기울기 %.3f · 상관 %+.3f**" % (m.sum(), sl, r))
    print("  읽는 법  기울기 ~1 이면 **전류 위상이 전압 위상을 따라간다** (물리, 배울 수 있다)")
    print("           기울기 ~0 이면 **전압과 무관한 기준 오류다** (보정해야 한다)")
