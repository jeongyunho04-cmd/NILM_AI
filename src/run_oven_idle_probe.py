"""오븐 FAN_LIGHT 구간에서 `standby` 머리가 얼마를 내는가 (12.164).

오븐은 '안 켜진' 상태가 둘이다 — 미사용 `OFF_STANDBY` 0.4W 와 세션 중 히터 off 인
`FAN_LIGHT` 15.0W. 손실의 `idle = σ(plugged)·(1−σ(on))` 이 구조적으로 뒤쪽
자리인데, 합성의 `gt_plugged` 가 항상 1 이라 두 상태를 가를 신호가 없었다
(12.163.4). 여기서 그 처방이 실제로 닿았는지 잰다.

    python -m src.run_oven_idle_probe --ckpt results/adapt_hwO_s0.pt

`test_4` 만 `_heater_pulses` 를 갖는다 — 오븐 세션 안에서 히터가 통전한 구간이다.
세션 ∖ 펄스 = FAN_LIGHT. 참값 14.2W (격리 측정, 12.162).
"""
from typing import List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.run_gate_check import forward_file, load_model

FAN_LIGHT_TRUE_W = 14.2      # 격리 측정 (12.162)
QUIET_APPS = ("electiric_kettle", "hair_dryer", "hotplate")   # 저항 부하가 없어야 조용하다


def _mask(pairs, n_cycles: int) -> np.ndarray:
    m = np.zeros(n_cycles, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n_cycles)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="오븐 FAN_LIGHT 의 standby 예측")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stem", default="test_4")
    ap.add_argument("--events", default="processed_data/real_events.json")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--absent", default="air_conditioner,fan",
                    help="그 장소에 없는 기기. 이들의 standby 가 오븐 몫을 나눠 갖는다")
    ap.add_argument("--off-stems", default="test_5,test_6,test_7,test_8",
                    help="오븐이 아예 없는 파일. `gt_plugged` 재정의가 여기서는 "
                         "오븐 idle 을 0 으로 눌러야 한다 — 안 그러면 유령이 는다")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ev = json.load(open(a.events, encoding="utf-8"))["files"][a.stem]
    n = int(ev["cycles"])
    iv = ev["intervals"]
    if "_heater_pulses" not in iv.get("oven", {}):
        print(f"!! {a.stem} 에 `_heater_pulses` 가 없다 — 세션과 통전을 못 가른다")
        return 2
    sess = _mask(iv["oven"]["on"], n)
    heat = _mask(iv["oven"]["_heater_pulses"], n)
    other = np.zeros(n, bool)
    for app in QUIET_APPS:
        if app in iv:
            other |= _mask(iv[app].get("on", []), n)

    absent = [x for x in a.absent.split(",") if x]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = {}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        d = forward_file(model, a.stem, dev, stride=a.stride)
        t = d["targets"]
        fan = sess[t] & ~heat[t]
        quiet = fan & ~other[t]
        jo = apps.index("oven")
        rows = {}
        for name, m in (("FAN_LIGHT (세션∖통전)", fan),
                        ("  + 조용한 창", quiet),
                        ("오븐 세션 밖", ~sess[t])):
            if m.sum() < 5:
                continue
            rows[name] = {
                "n": int(m.sum()),
                "standby_oven": float(np.median(d["standby"][m, jo])),
                "idle_oven": float(np.median(d["idle"][m, jo])),
                "gate_oven": float(np.median(d["gate"][m, jo])),
                "soft_w_oven": float(np.median(d["gate"][m, jo] * d["p_raw"][m, jo])),
                "p_observed": float(np.median(d["p_observed"][m])),
                "resid": float(np.median(
                    d["p_observed"][m] - (d["gate"][m] * d["p_raw"][m]).sum(1)
                    - d["standby"][m].sum(1) - d["p_noise"][m])),
            }
            for x in absent:
                rows[name]["standby_" + x] = float(np.median(d["standby"][m, apps.index(x)]))
        # 오븐이 없는 파일 — 여기서는 idle 이 0 이어야 한다
        for stem in [x for x in a.off_stems.split(",") if x]:
            try:
                d2 = forward_file(model, stem, dev, stride=a.stride)
            except Exception as e:                      # 없는 녹화는 조용히 넘긴다
                print(f"  (건너뜀 {stem}: {e})")
                continue
            rows[f"[오븐없음] {stem}"] = {
                "n": int(len(d2["targets"])),
                "standby_oven": float(np.median(d2["standby"][:, jo])),
                "idle_oven": float(np.median(d2["idle"][:, jo])),
                "gate_oven": float(np.median(d2["gate"][:, jo])),
                "soft_w_oven": float(np.median(d2["gate"][:, jo] * d2["p_raw"][:, jo])),
                "p_observed": float(np.median(d2["p_observed"])),
                "resid": float(np.median(
                    d2["p_observed"] - (d2["gate"] * d2["p_raw"]).sum(1)
                    - d2["standby"].sum(1) - d2["p_noise"])),
                **{"standby_" + x: float(np.median(d2["standby"][:, apps.index(x)]))
                   for x in absent},
            }

        tag = ck.split("/")[-1].replace(".pt", "")
        blob[tag] = rows

        print(f"\n=== {tag} | {a.stem} (참값 FAN_LIGHT {FAN_LIGHT_TRUE_W}W) ===")
        hdr = (f"{'구간':24s}{'n':>7s}{'오븐standby':>12s}{'idle':>8s}{'게이트':>8s}"
               f"{'오븐soft':>10s}{'관측W':>9s}{'잔차':>8s}"
               + "".join(f"{x[:8]:>10s}" for x in absent))
        print(hdr)
        for k, v in rows.items():
            print(f"{k:24s}{v['n']:7d}{v['standby_oven']:12.2f}{v['idle_oven']:8.3f}"
                  f"{v['gate_oven']:8.3f}{v['soft_w_oven']:10.1f}"
                  f"{v['p_observed']:9.1f}{v['resid']:8.2f}"
                  + "".join(f"{v['standby_' + x]:10.2f}" for x in absent))

    if a.out:
        json.dump(blob, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
