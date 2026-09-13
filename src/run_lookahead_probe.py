"""
룩어헤드 — 모델은 타깃 뒤 몇 초까지 실제로 보는가 (설계 문서 12.44절)
========================================================================
12.9.13 이 `TARGET_LOOKAHEAD` 를 60 -> 360(6초)으로 올린 근거는 이것이다.

    오븐이 꺼질 때 남는 전력이 468W 리플이면 핫플, 1180W 연속이면 포트다.
    그 전이를 창 안에서 보면 갈린다.

그리고 **모델이 안 썼다** 고 적혔다 (실패 창에서 전이 있음 0.008 vs 없음 0.001).
여기서 그 시험을 현재 모델로 다시 하되, **한 지점이 아니라 거리별로** 본다.

    python -m src.run_lookahead_probe --ckpt results/cnn_ov1.pt results/cnn_ov1_s1.pt

[왜 거리별인가 — 이것이 이 시험의 요점이다]
세밀 갈래 깊은 층 타깃 탭의 수용영역은 `1 + 6*(1+2+4+8+16) = 187` 사이클,
곧 **±1.56초**다. 얕은 탭 둘은 ±0.06 / ±0.16초다. 그보다 먼 곳이 헤드에 닿는
경로는 둘뿐이고 **둘 다 시간 위치가 없다**:

    세밀 갈래 전역 평균·최대   10초 안 어딘가라는 것만 안다
    광역 갈래                 `wide_summary` 가 기본 꺼짐이라 60초 평균 하나

즉 "타깃 4초 뒤에 오븐이 꺼진다" 를 표현할 경로가 지금 구조에 없다.

    전이가 ±1.56초 안에 있을 때만 게이트가 반응한다  -> **수용영역이 병목이다**
    6초까지 고르게 반응한다                          -> 룩어헤드를 늘릴 값어치가 있다
    어디서도 반응 없다                               -> 12.9.13 의 결론이 그대로 유효

[판정 기준 (돌리기 전에 적는다)]
대상은 관측 >=1300W & 오븐 히터 통전 & 핫플이 실제로 켜진 창이다 (재현율이 무너지는
바로 그 창, 12.41.6).

1. 거리 구간 (0,1.5] 의 핫플 게이트가 (6,inf) 보다 **0.05 이상** 높으면 반응이 있다
2. (1.5,3] 과 (3,6] 이 (6,inf) 와 구분되지 않으면 **수용영역 밖은 안 보는 것**이다
3. 어느 구간도 안 갈리면 12.9.13 이 맞다 — 룩어헤드를 늘려도 소용없다
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.run_gate_check import forward_file, load_model
from src.run_seed_variance_probe import _mask_from

# 타깃 탭 수용영역 ±1.56s 를 경계로 삼는다
GAP_BINS = [(0.0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9)]
GAP_LAB = ["0~1.5s", "1.5~3s", "3~6s", ">6s"]
FILES = ("test_4", "test_5", "test_6")


def oven_edges(v: dict) -> np.ndarray:
    """오븐 히터 통전 구간의 경계 (사이클). 켜짐·꺼짐 둘 다."""
    e = []
    for s, t in v["intervals"]["oven"].get("_heater_pulses") or []:
        e += [int(float(s) * 60), int(float(t) * 60)]
    return np.array(sorted(e), dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="룩어헤드 반응 거리 (12.44절)")
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_ov1.pt", "results/cnn_ov1_s1.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/lookahead_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()

    print("=" * 94)
    print("[룩어헤드 반응 거리]  타깃 뒤 오븐 전이까지의 거리별로 게이트를 본다")
    print("  대상: 관측 >=1300W & 오븐 히터 통전 & 핫플 실제 ON  (재현율이 무너지는 창)")
    print("  타깃 탭 수용영역 = 187사이클 = ±1.56초.  창 안 룩어헤드 = 6.0초")
    print("=" * 94)

    payload: Dict[str, dict] = {}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        tag = Path(ck).stem
        jk, jh = apps.index("electiric_kettle"), apps.index("hotplate")

        gaps, pasts, g_hp, g_kt, obs = [], [], [], [], []
        for stem in FILES:
            v = ev[stem]
            edges = oven_edges(v)
            if not len(edges):
                continue
            d = forward_file(model, stem, dev, stride=a.stride)
            t = d["targets"]
            n = int(v["cycles"])
            ovm = _mask_from(v["intervals"]["oven"].get("_heater_pulses"), n, t)
            hpm = _mask_from(v["intervals"]["hotplate"].get("on"), n, t)
            m = (d["p_observed"] >= 1300.0) & ovm & hpm
            if not m.any():
                continue
            k = np.searchsorted(edges, t, side="right")
            nxt = edges[k.clip(max=len(edges) - 1)]
            gap = np.where(k >= len(edges), 1e9, (nxt - t) / 60.0)
            prv = edges[(k - 1).clip(min=0)]
            past = np.where(k <= 0, 1e9, (t - prv) / 60.0)
            gaps.append(gap[m]); pasts.append(past[m])
            g_hp.append(d["gate"][m, jh])
            g_kt.append(d["gate"][m, jk])
            obs.append(d["p_observed"][m])
        gaps = np.concatenate(gaps); pasts = np.concatenate(pasts)
        g_hp = np.concatenate(g_hp)
        g_kt = np.concatenate(g_kt); obs = np.concatenate(obs)

        print()
        print(f"  [{tag}]  대상 창 {len(gaps)}개")
        rows = {}
        for side, dist, cap in (("미래", gaps, "다음 전이까지 (창 안 6.0초까지만 보인다)"),
                                ("과거", pasts, "직전 전이로부터 (세밀 갈래는 4.0초까지)")):
            print(f"    -- {side}: {cap}")
            print(f"    {'거리':<10s}{'창':>6s}{'핫플 게이트':>13s}"
                  f"{'핫플 재현율':>12s}{'포트 게이트':>13s}{'포트 점화율':>12s}")
            print("    " + "-" * 68)
            sub = {}
            for (lo, hi), lab in zip(GAP_BINS, GAP_LAB):
                b = (dist > lo) & (dist <= hi) if lo else (dist <= hi)
                if b.sum() < 5:
                    continue
                r = {"n": int(b.sum()),
                     "hp_gate_median": float(np.median(g_hp[b])),
                     "hp_recall": float((g_hp[b] > 0.5).mean()),
                     "kt_gate_median": float(np.median(g_kt[b])),
                     "kt_fire": float((g_kt[b] > 0.5).mean()),
                     "p_obs": float(np.median(obs[b]))}
                sub[lab] = r
                print(f"    {lab:<10s}{r['n']:>6d}{r['hp_gate_median']:>13.3f}"
                      f"{r['hp_recall']:>12.3f}{r['kt_gate_median']:>13.3f}"
                      f"{r['kt_fire']:>12.3f}")
            if "0~1.5s" in sub and ">6s" in sub:
                d_hp = sub["0~1.5s"]["hp_gate_median"] - sub[">6s"]["hp_gate_median"]
                far = [(k, sub[k]["hp_gate_median"] - sub[">6s"]["hp_gate_median"])
                       for k in ("1.5~3s", "3~6s") if k in sub]
                print(f"      수용영역 안 (0~1.5s) − (먼 곳) = {d_hp:+.3f}"
                      f"   |  밖 " + ", ".join(f"{k} {x:+.3f}" for k, x in far))
            rows[side] = sub
        payload[tag] = rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
