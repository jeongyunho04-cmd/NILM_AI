"""정밀화 결과를 라벨 형식으로 쓴다 (12.155)

`run_refine_labels` 의 산출물을 `real_events.json` 과 같은 구조로 옮긴다.
**원본은 안 건드리고 새 파일로 낸다** — 비교하고 나서 사람이 바꿔 끼울 것.

무엇이 달라지나
--------------
```
events.delta_p_w     총전력 계단 -> **기기 몫** (h3 로 되돌린 값)
events.t_s           사람 기록 -> 신호 계단 (|Δt| 중앙 0.2~1.5초)
intervals            on/off 를 적분해서 다시 만든다
uncertain            신호에 계단이 없는 항목의 근방 (채점에서 뺀다)
_label_provenance    human_switching_log_signal_refined (새 등급)
```

⚠ **`mode` 는 on/off 가 아니다.** 드라이기 강<->약 은 켜진 채로 단계만 바뀐 것이라
   구간을 끊지 않는다. `events` 에는 남기고 `intervals` 에는 영향을 안 준다.

⚠ **못 맞춘 항목은 시각을 지어내지 않는다.** 사람이 적은 것은 일어났으므로 그
   근방을 `uncertain` 으로 두고 어느 쪽으로도 채점하지 않는다 (12.4절 관례).

    python -X utf8 -m src.run_write_labels --refined results/refined_all.json \\
        --out processed_data/real_events_refined.json
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

#: 못 맞춘 항목 근방을 이만큼 `uncertain` 으로 둔다 (초). 사람 기록의 시각 오차
#: p90 이 5초라 그보다 넉넉히 잡는다.
UNCERTAIN_PAD = 8.0
PROV = "human_switching_log_signal_refined"


def build(stem: str, rows: List[Dict], seq_lo: int, dur_s: float) -> Dict:
    ev, unc = [], {}
    for r in rows:
        if r.get("matched"):
            note = (f"cost {r['cost']:.2f}, Δt {r['dt_s']:+.1f}s, "
                    f"ΔP출처 {r['dp_from']}, 총 {r['dp_total_w']:+.1f}W")
            if r.get("order_violation"):
                note += " ⚠순서위반"
            ev.append({"t_s": round(r["t_s"], 2), "appliance": r["appliance"],
                       "kind": r["kind"], "delta_p_w": round(r["dp_device_w"], 1),
                       "_note": note})
        elif r["kind"] in ("on", "off", "mode"):
            t = (r["seq"] - seq_lo) * 0.5
            unc.setdefault(r["appliance"], []).append(
                [max(0.0, t - UNCERTAIN_PAD), min(dur_s, t + UNCERTAIN_PAD)])
    ev.sort(key=lambda e: e["t_s"])

    # 구간 — on/off 를 적분한다. `mode` 는 상태를 안 바꾼다.
    iv: Dict[str, Dict] = {}
    open_at: Dict[str, float] = {}
    for r in sorted(rows, key=lambda r: r["seq"]):
        a = r["appliance"]
        iv.setdefault(a, {"on": [], "uncertain": []})
        if r["kind"] == "already_on":
            open_at[a] = 0.0
            continue
        if not r.get("matched") or r["kind"] == "mode":
            continue
        if r["kind"] == "on":
            open_at.setdefault(a, r["t_s"])
        elif r["kind"] == "off" and a in open_at:
            iv[a]["on"].append([round(open_at.pop(a), 2), round(r["t_s"], 2)])
    for a, t0 in open_at.items():                 # 녹화 끝까지 켜진 채로 남은 것
        iv.setdefault(a, {"on": [], "uncertain": []})["on"].append(
            [round(t0, 2), round(dur_s, 2)])
    for a, spans in unc.items():
        iv.setdefault(a, {"on": [], "uncertain": []})["uncertain"].extend(
            [[round(x, 2), round(y, 2)] for x, y in spans])
    return {"appliances_present": sorted(iv), "intervals": iv, "events": ev,
            "_label_provenance": PROV}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refined", nargs="+", required=True)
    ap.add_argument("--out", default="processed_data/real_events_refined.json")
    a = ap.parse_args()

    smap = json.load(open("results/seq_time_map.json", encoding="utf-8"))
    files: Dict[str, Dict] = {}
    for path in a.refined:
        R = json.load(open(path, encoding="utf-8"))
        for st, v in R.items():
            npz = Path(f"processed_data/composite_eval/{st}.npz")
            dur = (len(np.load(npz, allow_pickle=True)["t_rel_s"]) / 60.0
                   if npz.exists() else 0.0)
            files[st] = build(st, v["rows"], v.get("seq_lo", 0), dur)
            files[st]["duration_s"] = round(dur, 1)
            # 채점기가 요구하는 필드 (`run_power_check` 는 cycles 로 격자를 만든다)
            z = np.load(npz, allow_pickle=True) if npz.exists() else None
            files[st]["cycles"] = int(len(z["t_rel_s"])) if z is not None else 0
            files[st]["v_mean"] = (round(float(np.mean(z["power_features"][:, 4])), 1)
                                   if z is not None else 0.0)

    doc = {"_comment": [
        "run_write_labels 산출 (12.155). 원본 real_events.json 을 대체하지 않는다.",
        "delta_p_w 는 **그 기기 몫**이다 — 총전력 계단이 아니다. SMPS 는 h3 에서",
        "되돌린 값이라 오븐·핫플 듀티에 오염되지 않는다.",
        "t_s 는 신호 계단이다 (사람 기록 대비 |Δt| 중앙 0.2~1.5초).",
        "uncertain 은 사람이 적었는데 신호에 계단이 없는 자리다. 채점에서 뺀다.",
    ], "_label_provenance_levels": {
        PROV: "사람이 그 자리에서 적은 기록을 신호로 정밀화. 시각과 ΔP 는 신호, "
              "정체는 사람. 물리 적합·부호를 전부 검사했다.",
    }, "files": files}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 92)
    print(f"{'파일':<9}{'사건':>5}{'구간기기':>9}{'불확실':>7}{'ON 총시간(s)':>13}  기기")
    for st, v in sorted(files.items()):
        on = sum(y - x for a_ in v["intervals"].values() for x, y in a_["on"])
        nu = sum(len(a_["uncertain"]) for a_ in v["intervals"].values())
        print(f"{st:<9}{len(v['events']):>5}{len(v['intervals']):>9}{nu:>7}{on:>13.0f}  "
              + ", ".join(k[:8] for k in v["appliances_present"]))
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
