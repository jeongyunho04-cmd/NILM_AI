"""
입력 절제 — 모델이 **시간 결**을 실제로 쓰는가 (설계 문서 12.42절)
====================================================================
12.41 이 확정했다. 실행 간 분산은 관측 >=1300W & 오븐 히터 통전 창(전체의 8.9%)에
몰려 있고, 거기서 모델은 **정격 1233W 짜리 유령 포트를 확신하며 켠다.** 점화율만
실행마다 다르다 (16.7% vs 37.1%).

셋 다 순저항이라 순시 고조파로는 축퇴다 (오븐 1156 + 핫플 470 = 1626W 와
포트 1233 + 오븐(낮은 듀티) ~ 1620W 가 관측 1605W 에서 안 갈린다). **남는 판별자는
시간 결뿐이다.**

    포트   한 번 켜지면 끝난다 (중앙 9.2초)
    오븐   10~25초 통전 / 30~40초 휴지
    핫플   릴레이 약 2초 주기 (0.9초 통전 / 1.1초 휴지)

그 결이 모델 입력 안에 있다. 광역 갈래(60초 @2Hz, 12채널)가 오븐 듀티를 겨냥한
것이고, 세밀 갈래 ch36~37(추세 제거 전력 ±0.5초 / ±2.5초)이 핫플 릴레이 대역을
겨냥한 것이다 (0.4절, 1.3절).

**입력을 0 으로 죽여 보고 그 구간의 유령 점화율이 안 변하면, 모델은 그 결을 안
쓰고 있는 것이다.** 학습하지 않는다.

    python -m src.run_ablation_probe --ckpt results/cnn_ov1.pt results/cnn_ov1_s1.pt

[판정 기준 (돌리기 전에 적는다)]
1. **광역을 죽여도 점화율이 ±3%p 안이면 광역은 그 결정에 안 쓰인다.**
   그러면 오븐 듀티를 겨냥한 1.3절 설계가 이 구간에서는 헛돌고 있는 것이다.
2. ch36~37 을 죽여도 핫플 재현율이 안 떨어지면 핫플 판정도 릴레이 결을 안 쓴다.
3. 반대로 크게 흔들리면 그 갈래는 쓰이고 있고, 문제는 **정보 부족이 아니라
   그 정보로도 못 가르는 것**이다. 처방이 갈린다.

> ⚠ **0 으로 죽이는 것은 학습 분포 밖이다.** 반응이 크게 나왔다고 "그 채널이
> 중요하다" 로 바로 읽으면 안 된다. 이 시험이 힘을 갖는 방향은 **안 변할 때**다
> — 분포 밖 입력을 줘도 답이 그대로면 그 입력을 안 보고 있는 것이 확실하다.
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
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
from src.model.realdata import dense_targets
from src.run_gate_check import load_model
from src.run_live import KOR
from src.run_seed_variance_probe import _mask_from

# 추세 제거 전력 (핫플 릴레이 대역). 1.3절 / inputs.RIPPLE_HALF_*
RIPPLE_CH = (36, 37)

VARIANTS: Dict[str, dict] = {
    "원본": {},
    "광역=0": {"zero_wide": True},
    "ch36~37=0": {"zero_fine": RIPPLE_CH},
    "둘 다 0": {"zero_wide": True, "zero_fine": RIPPLE_CH},
}


@torch.no_grad()
def forward_ablated(model, stem: str, dev: str, stride: int,
                    zero_wide: bool = False,
                    zero_fine: Sequence[int] = ()) -> dict:
    """`run_gate_check.forward_file` 과 같되 입력 일부를 0 으로 죽인다."""
    rw = dense_targets(stem, stride=stride)
    G, R, POBS = [], [], []
    for i in range(0, len(rw), 512):
        idx = np.arange(i, min(i + 512, len(rw)))
        f, w, pobs, _oh, _pn = rw.batch(idx)
        f = np.ascontiguousarray(f)
        w = np.ascontiguousarray(w)
        if zero_fine:
            f = f.copy()
            f[:, list(zero_fine)] = 0.0
        if zero_wide:
            w = np.zeros_like(w)
        ft = torch.from_numpy(f).to(dev)
        wt = torch.from_numpy(w).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(ft, wt)
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        R.append(o["power_raw"].float().cpu().numpy())
        POBS.append(pobs)
    return {"gate": np.concatenate(G), "p_raw": np.concatenate(R),
            "p_observed": np.concatenate(POBS), "targets": rw.target_cycle}


def measure(model, apps: List[str], stems: List[str], ev: dict, dev: str,
            stride: int, **abl) -> dict:
    """유령 W, 목표 구간 점화율, 그 구간 핫플 재현율."""
    ghost, n_hi, n_fire, fire_w = [], 0, 0, []
    hp_tp = hp_true = 0
    jk = apps.index("electiric_kettle")
    jh = apps.index("hotplate")
    for stem in stems:
        d = forward_ablated(model, stem, dev, stride, **abl)
        P = d["gate"] * d["p_raw"]
        absent = [j for j, x in enumerate(apps)
                  if x not in ev[stem]["appliances_present"]]
        ghost.append(float(P[:, absent].mean(0).sum()))

        hi = d["p_observed"] >= 1300.0
        if hi.any():
            fire = d["gate"][hi, jk] > 0.5
            n_hi += int(hi.sum())
            n_fire += int(fire.sum())
            fire_w.append(P[hi, jk][fire])
            if "hotplate" in ev[stem]["appliances_present"]:
                truth = _mask_from(ev[stem]["intervals"]["hotplate"].get("on"),
                                   int(ev[stem]["cycles"]), d["targets"])[hi]
                hp_tp += int(((d["gate"][hi, jh] > 0.5) & truth).sum())
                hp_true += int(truth.sum())
    fw = np.concatenate(fire_w) if fire_w else np.zeros(0)
    return {"ghost_w": float(np.mean(ghost)),
            "n_hi": n_hi, "fire_rate": n_fire / n_hi if n_hi else float("nan"),
            "fire_median_w": float(np.median(fw)) if len(fw) else 0.0,
            "hotplate_recall_hi": hp_tp / hp_true if hp_true else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description="입력 절제 — 시간 결을 쓰는가 (12.42절)")
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_ov1.pt", "results/cnn_ov1_s1.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/ablation_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]

    print("=" * 90)
    print(f"[입력 절제] 목표 구간 = 관측 >=1300W ({len(stems)}파일, stride {a.stride})")
    print("=" * 90)
    payload: Dict[str, dict] = {}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        tag = Path(ck).stem
        print()
        print(f"  [{tag}]")
        print(f"    {'변형':<12s}{'유령W':>9s}{'포트 점화율':>13s}"
              f"{'점화 시 중앙W':>14s}{'핫플 재현율(>=1300)':>20s}")
        print("    " + "-" * 68)
        base = None
        rows = {}
        for name, abl in VARIANTS.items():
            r = measure(model, apps, stems, ev, dev, a.stride, **abl)
            rows[name] = r
            if base is None:
                base = r
            d_fire = 100 * (r["fire_rate"] - base["fire_rate"])
            mark = "" if name == "원본" else f"  ({d_fire:+.1f}%p)"
            print(f"    {name:<12s}{r['ghost_w']:>9.2f}"
                  f"{100 * r['fire_rate']:>12.1f}%{r['fire_median_w']:>14.1f}"
                  f"{r['hotplate_recall_hi']:>15.3f}{mark}")
        payload[tag] = rows

    print()
    print(f"    (목표 구간 창 {list(payload.values())[0]['원본']['n_hi']}개)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
