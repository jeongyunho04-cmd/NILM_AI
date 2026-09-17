# -*- coding: utf-8 -*-
"""SMPS 의 병이 **게이트인가 수준인가** (14.407, 사용자 지시 ⓑ).

§61 이 손잡이 여덟을 다 써도 미니PC 가 안 움직인다고 확정했고, §62 의 각도 측정이
*"셋이 거의 평행"* 이라는 내 설명을 **반증했다** (미니PC 는 충전기와 **55.9°** · 빔과
**72.3°** 로 잘 갈린다. 좁은 건 충전기/빔 **22.6°** 한 쌍뿐이다).

그러면 남는 후보가 셋인데 그중 **제일 싸고 결정적인 것**이 이것이다:

```
  power = σ(on_logit) · p_raw          <- SMPS 는 이 식이 그대로다
                                          (조합 머리는 `_PIN` 저항 전용이다)
  참ON 창에서
    p_raw 는 맞는데 power 가 작다   -> **게이트 문제**. 수준 머리는 멀쩡하다
    p_raw 도 작다                  -> **수준 문제**. 배분·사전 쪽이다
```
★ 14.75 가 미니PC 에서 그 모양을 이미 한 번 봤다 — *"전력 머리는 내내 ~10W 로 맞게
  내는데 게이트가 **0.004** 로 곱해 지운다"*. 그것이 **기기 전체**에서도 참인가를 본다.

⚠ 참값은 `real_events.json` 의 그 파일·그 기기 `|ΔP|` 중앙값이다 (사람 라벨이 아니라
  **신호 계단**이고, 14.74 가 `delta_p_w` 를 참값으로 쓰면 3~7W 과소라고 경고했다 —
  그래서 **비율로만** 읽고 절대값으로 판정하지 않는다).

    python -X utf8 -m src.run_diag_gatelevel --ckpt results/cnn_comb_s0.pt
"""
import argparse
import json

import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.evaluation.real_events import build_on_off_truth, load_events  # noqa: E402
from src.evaluation.sealing import is_sealed  # noqa: E402
from src.run_gate_check import forward_file, load_model, sync_even_median  # noqa: E402
from src.run_scorecard import STEMS  # noqa: E402

SMPS = ("minipc", "laptop_charger", "beam_projector")


def true_level(ev, stem, app):
    """그 파일·그 기기의 계단 |ΔP| 중앙값 (없으면 nan)."""
    rows = ev[stem]
    rows = rows if isinstance(rows, list) else rows.get("events", [])
    d = [abs(float(e.get("delta_p_w") or 0)) for e in rows
         if app in str(e.get("appliance", "")) and e.get("delta_p_w")]
    return float(np.median(d)) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    a_ = ap.parse_args()
    sync_even_median(a_.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()

    print("게이트인가 수준인가 (14.407) — `power = σ(on)·p_raw` 를 갈라 본다\n")
    for ck in a_.ckpt:
        m, apps, _ = load_model(ck, dev)
        print("  === %s" % ck)
        print("     %-9s %-8s %5s %8s %8s %8s %8s  %s"
              % ("기기", "파일", "창", "참W", "p_raw", "게이트", "power", "판정"))
        agg = {a: [[], [], []] for a in SMPS}
        for st in STEMS:
            if is_sealed(st):
                continue
            d = forward_file(m, st, dev, stride=30)
            n_cyc = int(ev[st]["cycles"])
            on, sc = build_on_off_truth(st, list(apps), n_cyc, ev)
            t = np.asarray(d["targets"], int).clip(0, n_cyc - 1)
            for app in SMPS:
                j = list(apps).index(app)
                k = on[t, j] & sc[t, j]
                if k.sum() < 30:
                    continue
                tw = true_level(ev, st, app)
                pr = float(np.median(d["p_raw"][k, j]))
                g = float(np.median(d["gate"][k, j]))
                pw = float(np.median(d["p_raw"][k, j] * d["gate"][k, j]))
                if np.isfinite(tw) and tw > 1:
                    agg[app][0].append(pr / tw)
                    agg[app][1].append(g)
                    agg[app][2].append(pw / tw)
                #: 판정 — p_raw 는 맞는데 power 가 작으면 **게이트**다
                v = ("**게이트**" if (np.isfinite(tw) and pr > .7 * tw and pw < .6 * tw)
                     else ("수준" if np.isfinite(tw) and pr < .7 * tw else "—"))
                print("     %-9s %-8s %5d %8.1f %8.1f %8.3f %8.1f  %s"
                      % (app[:9], st, int(k.sum()), tw, pr, g, pw, v))
        print()
        print("     %-9s %10s %10s %10s" % ("", "p_raw/참", "게이트", "power/참"))
        for app in SMPS:
            if not agg[app][0]:
                continue
            print("     %-9s %10.3f %10.3f %10.3f  <- **%s**"
                  % (app[:9], np.median(agg[app][0]), np.median(agg[app][1]),
                     np.median(agg[app][2]),
                     "게이트가 지운다" if np.median(agg[app][0]) > 0.7
                     and np.median(agg[app][2]) < 0.6 else "수준부터 낮다"))
        print()
        del m
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
