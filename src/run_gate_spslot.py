# -*- coding: utf-8 -*-
"""관문 — **초기값 슬롯표** (14.186). 여섯 줄.

12.9.9 의 `S_STATE` 는 `L_power` 의 **Huber 척도**(p90)다. 척도로는 p90 이 맞다.
그런데 13.84.68 이 같은 값을 **머리 바이어스 초기값**으로 재사용했고, 듀티 기기는
p90 과 중앙이 20% 넘게 벌어진다. 그러면 초기값이 라벨에서 Huber δ(=0.1) **밖**에
떨어져 기울기가 포화되고 300에포크로 못 도착한다.

```
  기기  상태   S_STATE   라벨 중앙   학습값   | 표->라벨 경로의 %  · 남은 오차(Huber 눈금)
  포트  s1    1534.5    1456.8   1468.9 |   **84%**        0.008
  드라이 s2   1022.5     964.1    978.6 |   **75%**        0.014
  드라이 s1    529.2     486.9    506.3 |   **54%**        0.037
  핫플  s2     549.6     454.8    525.6 |   **25%**      **0.129**
  오븐  s2    1357.1    1100.6   1298.9 |   **23%**      **0.146**
```

```
  [1] `table` 이면 표가 **같은 객체**이고 바이어스가 **비트 동일**
  [2] `label` 은 **큰 슬롯(>=300W)만** 바꾼다 — 작은 슬롯은 비트 동일
  [3] 새 값이 세그먼트 풀의 **라벨 중앙값과 일치**한다 (직접 계산)
  [4] 바꾼 슬롯은 라벨과 **실측이 5% 안에서 일치**한다 (오븐 s1 은 어긋나서 안 바꾼다)
  [5] ★ 초기값이 라벨에서 **Huber δ 안**으로 들어온다 (지금 오븐 0.146 · 핫플 0.129)
  [6] `load_state_dict` 가 덮으므로 **학습된 체크포인트에는 영향이 없다**
```

    python -X utf8 -m src.run_gate_spslot
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.model.losses import (MIN_STATE_SCALE_W, S_STATE,  # noqa: E402
                              S_STATE_INIT_LABEL, state_power_init_table)
from src.model.net import NILMNet, state_power_w  # noqa: E402

SH = {"oven": "오븐", "electiric_kettle": "포트", "hair_dryer": "드라이", "fan": "선풍",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "hotplate": "핫플", "air_conditioner": "에어컨"}
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]
DELTA = 0.1                                   # `L_power` 의 Huber δ (12.9.9)


def mk(src):
    torch.manual_seed(0)
    return NILMNet(APPS, NS, fine_channels=57, fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0,
                   state_power_src=src).eval()


def main() -> int:
    ok = True

    # [1] table 이면 비트 동일
    same_obj = state_power_init_table("table") is S_STATE
    a, b = mk("table"), mk("table")
    b.load_state_dict(a.state_dict())
    d = max(float((x - y).abs().max()) for x, y in zip(
        [h.bias for h in a.heads], [h.bias for h in b.heads]))
    print("[1] `table` = `S_STATE` **같은 객체** %s · 바이어스 비트 동일 %.3e  %s"
          % (same_obj, d, "OK" if (same_obj and d == 0.0) else "FAIL"))
    ok &= (same_obj and d == 0.0)

    # [2] label 은 큰 슬롯만
    ml = mk("label")
    moved, kept_small, bad = [], True, []
    for j, ap in enumerate(APPS):
        for sid, w in S_STATE.get(ap, {}).items():
            if not (0 <= sid < ml.n_pow and w > 0 and sid < NS[j]):
                continue
            v0 = float(a.heads[j].bias[sid])
            v1 = float(ml.heads[j].bias[sid])
            if abs(v1 - v0) > 1e-6:
                moved.append((ap, sid, w, state_power_init_table("label")[ap][sid]))
                if w < 300.0:
                    bad.append("%s s%d (%.1fW)" % (SH.get(ap, ap), sid, w))
            elif w >= 300.0 and ap in S_STATE_INIT_LABEL and sid in S_STATE_INIT_LABEL[ap]:
                bad.append("안 바뀐 큰 슬롯 %s s%d" % (SH.get(ap, ap), sid))
    print("[2] `label` 이 바꾼 슬롯 **%d개** — 전부 >=300W  %s%s"
          % (len(moved), "OK" if not bad else "FAIL",
             "" if not bad else "  <- " + " · ".join(bad)))
    for ap, sid, w, v in moved:
        print("      %-8s s%-2d  %8.1f -> %8.1fW  (%+.1f%%)"
              % (SH.get(ap, ap), sid, w, v, 100 * (v / w - 1)))
    ok &= (not bad and len(moved) == sum(len(d_) for d_ in S_STATE_INIT_LABEL.values()))

    # [3][4][5] 풀에서 직접 재서 맞춘다
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    print("\n[3][4][5] 세그먼트 풀에서 직접 — 라벨 중앙 · 실측 중앙 · Huber 거리")
    print("      %-8s %3s %9s %10s %10s %9s %9s"
          % ("기기", "상태", "S_STATE", "라벨중앙", "실측중앙", "표 거리", "새 거리"))
    e3, e4, e5 = [], [], []
    for ap, sid, w, v in moved:
        L, M = [], []
        for act in pool.appliance_activations.get(ap, []):
            st = np.asarray(act.state_id)
            m = st == sid
            if m.any():
                L.append(state_power_w(act, False)[m])
                M.append(state_power_w(act, True)[m])
        lab = float(np.median(np.concatenate(L)))
        mea = float(np.median(np.concatenate(M)))
        d0 = abs(w - lab) / max(w, MIN_STATE_SCALE_W)
        d1 = abs(v - lab) / max(w, MIN_STATE_SCALE_W)
        if abs(v - lab) > 0.15:
            e3.append("%s s%d (%.1f 대 %.1f)" % (SH.get(ap, ap), sid, v, lab))
        if abs(lab - mea) / max(lab, 1.0) > 0.05:
            e4.append("%s s%d (라벨 %.1f 대 실측 %.1f)" % (SH.get(ap, ap), sid, lab, mea))
        if not (d1 < DELTA):
            e5.append("%s s%d (%.3f)" % (SH.get(ap, ap), sid, d1))
        print("      %-8s s%-2d %8.1fW %9.1fW %9.1fW %9.3f %9.3f%s"
              % (SH.get(ap, ap), sid, w, lab, mea, d0, d1,
                 "  <- **δ 밖이었다**" if d0 >= DELTA else ""))
    print("[3] 새 값 = 풀 라벨 중앙값  %s%s"
          % ("OK" if not e3 else "FAIL", "" if not e3 else "  <- " + " · ".join(e3)))
    print("[4] 바꾼 슬롯은 **라벨과 실측이 5%% 안**  %s%s"
          % ("OK" if not e4 else "FAIL", "" if not e4 else "  <- " + " · ".join(e4)))
    print("[5] ★ 새 초기값이 **Huber δ=%.1f 안**  %s%s"
          % (DELTA, "OK" if not e5 else "FAIL", "" if not e5 else "  <- " + " · ".join(e5)))
    #: 안 바꾼 자리도 적는다 — 오븐 s1 은 라벨이 0 이라 일부러 뺐다 (13.40 · 14.167)
    L1 = np.concatenate([state_power_w(x, False)[np.asarray(x.state_id) == 1]
                         for x in pool.appliance_activations.get("oven", [])
                         if (np.asarray(x.state_id) == 1).any()])
    M1 = np.concatenate([state_power_w(x, True)[np.asarray(x.state_id) == 1]
                         for x in pool.appliance_activations.get("oven", [])
                         if (np.asarray(x.state_id) == 1).any()])
    print("      (안 바꾼 자리) 오븐 s1 — 라벨 중앙 **%.1fW** 대 실측 **%.1fW** 로 어긋난다."
          % (float(np.median(L1)), float(np.median(M1))))
    print("      `is_on=0` 이라 `target_power_w=0` 으로 적히는 자리다 (13.40 · 14.167). 건드리지 않는다.")
    ok &= (not e3 and not e4 and not e5)

    # [6] 학습된 체크포인트에는 영향 없다
    sd = a.state_dict()
    ml2 = mk("label")
    ml2.load_state_dict(sd)
    d6 = max(float((x - y).abs().max()) for x, y in zip(
        [h.bias for h in a.heads], [h.bias for h in ml2.heads]))
    print("\n[6] `load_state_dict` 뒤에는 **차이 0** — 학습된 판에 영향 없다 %.3e  %s"
          % (d6, "OK" if d6 == 0.0 else "FAIL"))
    ok &= (d6 == 0.0)

    # 곁들여 — 이 고침이 겨냥하는 간격
    T, Lb = S_STATE, state_power_init_table("label")
    g0 = T["electiric_kettle"][1] - T["oven"][2]
    g1 = Lb["electiric_kettle"][1] - Lb["oven"][2]
    print("\n   겨냥: **오븐↔포트 간격** 표 %.0fW (%.1f%%) -> 새 초기값 %.0fW (%.1f%%)"
          % (g0, 100 * g0 / T["electiric_kettle"][1], g1,
             100 * g1 / Lb["electiric_kettle"][1]))
    print("   (학습된 15판은 170W = 11.6% 에 머물러 있다. 실측 계단은 **352W = 24.6%**)")

    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
