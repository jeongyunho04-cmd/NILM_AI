"""
후처리 — 합 정합성으로 유령을 끈다 (설계 문서 12.46절)
========================================================
12.45.6 이 연 문이다. 9초 모델은 유령 포트가 점화한 창의 **42.7% 에서 핫플도 함께
켠다** (6초 모델은 12.4%). 대체가 아니라 **중복 계상**이고, 그래서 예측 기기합이
관측을 넘는다. 넘는 만큼은 물리적으로 불가능하다 — 관측이 전부다.

    예측합 = Σ 기기 + 대기 + 계측잡음        이것이 관측 P 를 크게 넘으면
                                            켜져 있다고 한 기기 중 하나는 없는 것이다

    python -m src.run_sumfix_probe --ckpt results/cnn_ov1.pt results/cnn_ov1_s1.pt

[규칙 — 배포 가능해야 한다]
**어느 기기가 없는지 모른다고 가정한다.** 실측 6파일에서는 전기포트가 없다는 것을
알지만 운영에서는 모른다. 그래서 기기 이름을 안 쓰고 이렇게만 한다:

    초과 = 예측합 − 관측
    초과 > margin 인 동안:  켜져 있는(게이트>0.5) 기기 중 **게이트가 가장 낮은** 것을 끈다

[문턱을 손으로 고르지 않는다]
관측 <300W 구간은 모델이 사실상 맞힌다 (핫플 F1 0.96~0.99, 유령 7~8W). 거기서
초과가 얼마나 나는지가 "맞을 때도 이만큼은 넘친다" 의 기준이다. 그 **p95** 를
문턱으로 쓴다. 비교를 위해 sweep 도 함께 찍는다.

[이 규칙은 12.43 과 달리 오기각 비용을 **잴 수 있다**]
12.43 의 지속시간 규칙은 전기포트만 건드리는데 실측 6파일에 포트가 없어 오기각을
못 쟀다. 이 규칙은 기기를 안 가리므로 **핫플·오븐·프로젝터·충전기·미니PC 를 끌 수도
있고, 그것들은 정답이 있다.** 끈 것 중 참으로 켜져 있던 비율을 직접 센다.

[판정 기준 (돌리기 전에 적는다)]
1. 유령이 내려간다
2. **참으로 켜져 있던 기기를 끄는 비율이 5% 미만이다** — 이것이 이 규칙의 안전성이다
3. `>=1300W` 핫플 재현율이 안 떨어진다
4. 잔차가 안 커진다 (초과를 깎는 규칙이므로 원래는 줄어야 한다)
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
from src.run_live import KOR
from src.run_postproc_probe import runs
from src.run_seed_variance_probe import _mask_from

KETTLE = "electiric_kettle"
MIN_DROP_W = 20.0          # 이보다 적게 내는 기기를 꺼도 초과가 안 준다
POOL_KETTLE_MIN_S = 5.58   # 12.43 지속시간 규칙의 문턱 (학습 풀 최소 활성화)


def apply_sumfix(P: np.ndarray, gate: np.ndarray, base: np.ndarray,
                 pobs: np.ndarray, margin: float):
    """초과가 margin 이하가 될 때까지 게이트가 낮은 기기부터 끈다.

    Returns: (수정된 P, 끈 자리 bool (n,K))
    """
    P = P.copy()
    dropped = np.zeros(P.shape, bool)
    over = P.sum(1) + base - pobs
    idx = np.flatnonzero(over > margin)
    for i in idx:
        # 켜져 있고 실제로 전력을 내는 기기만 후보
        cand = np.flatnonzero((gate[i] > 0.5) & (P[i] > MIN_DROP_W))
        if not len(cand):
            continue
        for j in cand[np.argsort(gate[i][cand])]:      # 게이트 낮은 순
            if P[i].sum() + base[i] - pobs[i] <= margin:
                break
            P[i, j] = 0.0
            dropped[i, j] = True
    return P, dropped


def score(P: np.ndarray, d: dict, stem: str, apps: List[str], ev: dict) -> dict:
    absent = [j for j, x in enumerate(apps) if x not in ev[stem]["appliances_present"]]
    base = d["standby"].sum(1) + d["p_noise"]
    return {"ghost": float(P[:, absent].mean(0).sum()),
            "resid": float(np.abs(P.sum(1) + base - d["p_observed"]).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description="합 정합성 후처리 (12.46절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_ov1.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--margins", nargs="*", type=float, default=None,
                    help="기본은 <300W 구간 초과의 p95 + sweep")
    ap.add_argument("--out", default="results/sumfix_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    dt = a.stride / 60.0
    payload: Dict[str, dict] = {}

    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        tag = Path(ck).stem
        jk, jh = apps.index(KETTLE), apps.index("hotplate")
        fwd = {s: forward_file(model, s, dev, stride=a.stride) for s in stems}

        # ── 문턱: 관측 <300W 구간 초과의 p95 ──────────────────────────────
        lo_over = []
        for s in stems:
            d = fwd[s]
            base = d["standby"].sum(1) + d["p_noise"]
            over = (d["gate"] * d["p_raw"]).sum(1) + base - d["p_observed"]
            lo_over.append(over[d["p_observed"] < 300.0])
        lo_over = np.concatenate(lo_over)
        m_auto = float(np.percentile(lo_over, 95))
        margins = a.margins or sorted({round(m_auto, 1), 25.0, 50.0, 100.0, 200.0})

        print("=" * 96)
        print(f"[합 정합성] {tag}   (비봉인 {len(stems)}파일, stride {a.stride})")
        print("=" * 96)
        print(f"  관측 <300W 창 {len(lo_over):,}개에서 예측합 − 관측: "
              f"중앙 {np.median(lo_over):+.1f}W  p95 {m_auto:+.1f}W  "
              f"-> **문턱 {m_auto:.1f}W**")

        print()
        print(f"    {'규칙':<22s}{'유령W':>9s}{'잔차W':>9s}{'포트점화':>10s}"
              f"{'핫플재현':>10s}{'끈 창':>8s}{'참ON 오기각':>12s}")
        print("    " + "-" * 82)
        rows = {}

        def run_variant(lab: str, margin: float, duration_rule: bool):
            g, r, ndrop, nwrong, ntot = [], [], 0, 0, 0
            n_hi = n_fire = hp_tp = hp_true = 0
            for s in stems:
                d = fwd[s]
                P = (d["gate"] * d["p_raw"]).copy()
                gate = d["gate"]
                base = d["standby"].sum(1) + d["p_noise"]
                if duration_rule:
                    on = gate[:, jk] > 0.5
                    for i0, i1 in runs(on):
                        if (i1 - i0) * dt < POOL_KETTLE_MIN_S:
                            P[i0:i1, jk] = 0.0
                if margin is not None:
                    P, dropped = apply_sumfix(P, gate, base, d["p_observed"], margin)
                    ndrop += int(dropped.any(1).sum())
                    # 오기각: 끈 기기가 그 파일에 있고 그 순간 정답이 ON 이었나
                    for j, app in enumerate(apps):
                        m = dropped[:, j]
                        if not m.any():
                            continue
                        ntot += int(m.sum())
                        iv = ev[s]["intervals"].get(app, {})
                        if app not in ev[s]["appliances_present"] or not iv.get("on"):
                            continue
                        key = "_heater_pulses" if app == "oven" and iv.get("_heater_pulses") else "on"
                        truth = _mask_from(iv.get(key), int(ev[s]["cycles"]), d["targets"])
                        nwrong += int((m & truth).sum())
                sc = score(P, d, s, apps, ev)
                g.append(sc["ghost"]); r.append(sc["resid"])
                hi = d["p_observed"] >= 1300.0
                if hi.any():
                    n_hi += int(hi.sum())
                    n_fire += int(((P[hi, jk] > 0) & (gate[hi, jk] > 0.5)).sum())
                    if "hotplate" in ev[s]["appliances_present"]:
                        t = _mask_from(ev[s]["intervals"]["hotplate"].get("on"),
                                       int(ev[s]["cycles"]), d["targets"])[hi]
                        hp_tp += int(((gate[hi, jh] > 0.5) & (P[hi, jh] > 0) & t).sum())
                        hp_true += int(t.sum())
            row = {"ghost_w": float(np.mean(g)), "resid_w": float(np.mean(r)),
                   "fire_rate": n_fire / n_hi if n_hi else float("nan"),
                   "hp_recall_hi": hp_tp / hp_true if hp_true else float("nan"),
                   "n_windows_touched": ndrop, "n_dropped": ntot,
                   "n_dropped_true_on": nwrong,
                   "wrong_rate": nwrong / ntot if ntot else 0.0}
            rows[lab] = row
            print(f"    {lab:<22s}{row['ghost_w']:>9.2f}{row['resid_w']:>9.2f}"
                  f"{100 * row['fire_rate']:>9.1f}%{row['hp_recall_hi']:>10.3f}"
                  f"{ndrop:>8d}"
                  + (f"{nwrong:>6d}/{ntot:<5d}" if ntot else f"{'-':>12s}"))

        run_variant("원본", None, False)
        run_variant(f"지속시간 <{POOL_KETTLE_MIN_S:.1f}s", None, True)
        for m in margins:
            run_variant(f"합 정합성 >{m:.0f}W", m, False)
        run_variant(f"둘 다 (>{m_auto:.0f}W)", m_auto, True)
        payload[tag] = {"margin_auto_w": m_auto, "variants": rows}
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
