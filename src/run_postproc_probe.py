"""
후처리 — 물리적 지속시간 사전확률로 유령 포트를 기각할 수 있는가 (설계 문서 12.43절)
========================================================================================
12.41 이 확정했다. 실측 유령의 75~84% 가 **전기포트 하나**이고, 그것은 관측
>=1300W 창에서 **정격 1233W 를 확신하며** 켜진다. 두 실행의 차이는 크기가 아니라
점화율이다 (16.7% vs 37.1%).

그런데 전기포트는 **한 번 켜지면 끝나는** 기기다 — 동작 지속시간 중앙 9.2초
(0.4절). 60초 창이 줄줄이 이어지도록 1233W 가 계속 켜져 있다는 예측은 그 물리와
안 맞는다. **그것을 학습 없이 기각해 본다.**

    python -m src.run_postproc_probe --ckpt results/cnn_ov1.pt results/cnn_ov1_s1.pt

[규칙]
포트 게이트가 0.5 를 넘는 **연속 구간**의 길이를 재서, 학습 풀의 실제 활성화보다
긴 구간은 통째로 기각한다 (그 구간의 포트 전력을 0 으로). 문턱은 학습 풀에서
직접 잰다 — 손으로 고르지 않는다.

[판정 기준 (돌리기 전에 적는다)]
1. **유령이 내려간다.** 얼마나 내려가는지가 이 규칙의 값어치다.
2. **실행 간 폭이 줄어든다.** 21.78W 가 목표다. 이것이 핵심이다 — 규칙이 두 실행을
   같은 곳으로 모으면 미결정을 물리로 메운 것이다.
3. **핫플 재현율은 안 오른다** (예측). 포트를 지워도 그 전력이 핫플로 가지는
   않는다. 게이트가 기기마다 독립이기 때문이다. 그래서 잔차는 **커진다.**
   이것이 예측대로면 "포트를 지우는 것" 과 "핫플을 살리는 것" 은 별개의 처방이다.

[⚠ 이 시험이 못 재는 것 — 반드시 함께 읽을 것]
**비봉인 6파일 전부에 전기포트가 없다.** 그래서 이 규칙의 **오기각 비용을 잴 수
없다.** 여기서 나오는 이득은 전부 상한이다. 참 포트가 있는 녹화가 생기기 전에는
이 규칙을 운영에 넣으면 안 된다 (12.41.7 ③).
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
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model
from src.run_seed_variance_probe import _mask_from

KETTLE = "electiric_kettle"


def pool_durations(app: str = KETTLE) -> np.ndarray:
    """학습 풀의 실제 활성화 지속시간(초). 문턱을 여기서 뽑는다."""
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    return np.array([a.duration_s for a in pool.appliance_activations[app]], float)


def runs(mask: np.ndarray) -> List[tuple]:
    """불리언 배열 -> [(시작 index, 끝 index+1)]."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> int:
    ap = argparse.ArgumentParser(description="포트 지속시간 후처리 (12.43절)")
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_ov1.pt", "results/cnn_ov1_s1.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="기각 문턱(초). 기본은 학습 풀 분위수에서 뽑는다")
    ap.add_argument("--out", default="results/postproc_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    dt = a.stride / 60.0                        # 예측 격자 간격(초)

    dur = pool_durations()
    q = {k: float(np.percentile(dur, k)) for k in (50, 90, 99)}
    lo_pool, hi_pool = float(dur.min()), float(dur.max())
    print("=" * 88)
    print("[포트 지속시간 후처리]  — 학습 없이 물리 사전확률로 기각한다")
    print("=" * 88)
    print(f"  학습 풀의 전기포트 활성화 {len(dur)}개: "
          f"최소 {lo_pool:.1f}s  중앙 {q[50]:.1f}s  p90 {q[90]:.1f}s  "
          f"최대 {hi_pool:.1f}s")
    print(f"  예측 격자 {dt:.2f}s (stride {a.stride})")

    # (라벨, 하한, 상한) — 이 밖의 길이를 가진 ON 구간을 통째로 기각한다.
    # `> X` 쪽이 사전 기준이었고, `< X` 쪽은 결과를 보고 추가한 **사후 규칙**이다
    # (12.43.2). 둘을 표에서 구분해 읽을 것.
    INF = 1e9
    rules = [("원본", 0.0, INF),
             (f"> {hi_pool:.1f}s 기각", 0.0, hi_pool),        # 사전 기준
             ("> 30.0s 기각", 0.0, 30.0),                     # 사전 기준
             (f"< {lo_pool:.1f}s 기각", lo_pool, INF),        # 사후
             ("< 3.0s 기각", 3.0, INF),                       # 사후
             ("< 1.5s 기각", 1.5, INF),                       # 사후
             (f"풀 밖 기각", lo_pool, hi_pool)]                # 사후
    if a.thresholds:
        rules = [("원본", 0.0, INF)] + [(f"> {t:.1f}s 기각", 0.0, t)
                                        for t in a.thresholds]

    payload: Dict[str, dict] = {
        "pool_duration_s": {"n": len(dur), "min": lo_pool, "max": hi_pool, **q},
        "rules": [[l, lo, h] for l, lo, h in rules]}
    per_ck = {}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        tag = Path(ck).stem
        jk, jh = apps.index(KETTLE), apps.index("hotplate")
        fwd = {s: forward_file(model, s, dev, stride=a.stride) for s in stems}

        # ── 예측된 포트 ON 구간의 길이 분포 ──────────────────────────────
        lens, hi_lens = [], []
        for s in stems:
            d = fwd[s]
            on = d["gate"][:, jk] > 0.5
            for i0, i1 in runs(on):
                lens.append((i1 - i0) * dt)
                if (d["p_observed"][i0:i1] >= 1300.0).mean() > 0.5:
                    hi_lens.append((i1 - i0) * dt)
        lens = np.array(lens) if lens else np.zeros(0)
        hi_lens = np.array(hi_lens) if hi_lens else np.zeros(0)
        print()
        print(f"  [{tag}]  예측된 포트 ON 구간 {len(lens)}개 — "
              f"중앙 {np.median(lens) if len(lens) else 0:.1f}s  "
              f"최대 {lens.max() if len(lens) else 0:.1f}s  |  "
              f">=1300W 구간에 걸친 것 {len(hi_lens)}개 "
              f"(중앙 {np.median(hi_lens) if len(hi_lens) else 0:.1f}s)")

        print(f"    {'규칙':<16s}{'유령W':>9s}{'잔차W':>9s}"
              f"{'포트 점화율':>12s}{'핫플 재현율':>12s}{'기각 구간':>10s}")
        print("    " + "-" * 68)
        rows = {}
        for lab, lo, hi_th in rules:
            g, resid, n_hi, n_fire = [], [], 0, 0
            hp_tp = hp_true = n_rej = 0
            for s in stems:
                d = fwd[s]
                P = (d["gate"] * d["p_raw"]).copy()
                on = d["gate"][:, jk] > 0.5
                for i0, i1 in runs(on):
                    sec = (i1 - i0) * dt
                    if sec < lo or sec > hi_th:
                        P[i0:i1, jk] = 0.0
                        on[i0:i1] = False
                        n_rej += 1
                absent = [j for j, x in enumerate(apps)
                          if x not in ev[s]["appliances_present"]]
                g.append(float(P[:, absent].mean(0).sum()))
                r = P.sum(1) + d["standby"].sum(1) + d["p_noise"] - d["p_observed"]
                resid.append(float(np.abs(r).mean()))
                m = d["p_observed"] >= 1300.0
                if m.any():
                    n_hi += int(m.sum())
                    n_fire += int(on[m].sum())
                    if "hotplate" in ev[s]["appliances_present"]:
                        t = _mask_from(ev[s]["intervals"]["hotplate"].get("on"),
                                       int(ev[s]["cycles"]), d["targets"])[m]
                        hp_tp += int(((d["gate"][m, jh] > 0.5) & t).sum())
                        hp_true += int(t.sum())
            row = {"ghost_w": float(np.mean(g)), "resid_w": float(np.mean(resid)),
                   "fire_rate": n_fire / n_hi if n_hi else float("nan"),
                   "hotplate_recall_hi": hp_tp / hp_true if hp_true else float("nan"),
                   "n_rejected_runs": n_rej}
            rows[lab] = row
            print(f"    {lab:<16s}{row['ghost_w']:>9.2f}{row['resid_w']:>9.2f}"
                  f"{100 * row['fire_rate']:>11.1f}%{row['hotplate_recall_hi']:>12.3f}"
                  f"{n_rej:>10d}")
        per_ck[tag] = rows

        # ── 재귀속이 가능한가 — 유령 창에서 핫플 게이트는 어디 있나 ───────
        gh = []
        for s in stems:
            d = fwd[s]
            m = (d["p_observed"] >= 1300.0) & (d["gate"][:, jk] > 0.5)
            if m.any():
                gh.append(d["gate"][m, jh])
        if gh:
            gh = np.concatenate(gh)
            print(f"    유령 점화 창 {len(gh)}개에서 핫플 게이트: "
                  f"중앙 {np.median(gh):.3f}  "
                  f">0.5 {100 * (gh > 0.5).mean():.1f}%  "
                  f">0.2 {100 * (gh > 0.2).mean():.1f}%  "
                  f">0.05 {100 * (gh > 0.05).mean():.1f}%")
            per_ck[tag]["_hotplate_gate_in_phantom"] = {
                "n": int(len(gh)), "median": float(np.median(gh)),
                "over_0.5": float((gh > 0.5).mean()),
                "over_0.2": float((gh > 0.2).mean()),
                "over_0.05": float((gh > 0.05).mean())}

    payload["per_ckpt"] = per_ck

    # ── 실행 간 폭이 줄었는가 (판정 기준 ②) ──────────────────────────────
    tags = [Path(c).stem for c in a.ckpt]
    if len(tags) == 2:
        print()
        print("  [판정 기준 ②] 규칙이 두 실행을 같은 곳으로 모으는가")
        print(f"    {'규칙':<16s}{tags[0]:>12s}{tags[1]:>12s}{'폭':>9s}")
        print("    " + "-" * 49)
        for key in per_ck[tags[0]]:
            if key.startswith("_"):
                continue
            va = per_ck[tags[0]][key]["ghost_w"]
            vb = per_ck[tags[1]][key]["ghost_w"]
            print(f"    {key:<16s}{va:>12.2f}{vb:>12.2f}{abs(vb - va):>9.2f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
