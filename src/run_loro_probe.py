"""
녹화 단위 홀드아웃 — 파형 암기 가설을 직접 친다 (설계 문서 12.16.3절)
=======================================================================
12.16 이 이렇게 좁혔다. 학습 풀의 저항 부하 활성화가 핫플 3개·포트 13개(녹화 1개)
뿐이고 합성기가 그것을 **그대로 재생**하므로, 모델이 *"이 파형이 있는가"* 를 대조해
맞히고 있을 수 있다. 그러면 합성에서 100% 이면서 실측에서 실패하는 것이 설명된다.

**시간축 홀드아웃은 이것을 원리적으로 못 잡는다** — 같은 녹화의 뒤 20% 라 같은 기기·
같은 세션·같은 파형이다. 녹화로 잘라야 한다.

    학습 A (`cnn_v17`)        핫플 활성화 3개 전부 (hotplate_1 x1, hotplate_2 x2)
    학습 B (`cnn_v18_nohp2`)  hotplate_2 제외 -> hotplate_1 x1 만
    평가 1                    hotplate_2 활성화로만 만든 창   <- B 는 처음 본다
    평가 2                    hotplate_1 활성화로만 만든 창   <- 둘 다 봤다 (대조군)

판정:
  암기라면   B 가 평가1에서 A 보다 크게 떨어지고, 평가2 에서는 비슷하다
  일반화라면 B 가 평가1·2 에서 모두 A 와 비슷하다
  표본 부족이면 B 가 평가1·2 **둘 다** 떨어진다 (핫플 데이터가 41% 줄었으므로)

**대조군이 핵심이다.** 평가2 가 없으면 "녹화를 못 봐서" 와 "데이터가 적어서" 가 안 갈린다.

    python -m src.run_loro_probe
"""
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.model.inputs import target_index
from src.run_duty_probe import (
    HEAT_MIN_W, OVEN_HEATER_MIN_W, SMPS, WINDOW_CYCLES, format_inputs, gates_of,
)
from src.run_gate_check import load_model


def build_windows(only_files: Dict[str, List[str]], n: int, ti: int, seed: int,
                  with_smps: bool = True, max_tries: int = 80) -> Optional[dict]:
    """지정한 녹화의 활성화**만** 써서 오븐히터+핫플 창을 만든다."""
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train",
                       only_activation_files=only_files)
    synth = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    active = ["oven", "hotplate"] + (SMPS if with_smps else [])
    np.random.seed(seed)
    X, P, TR = [], [], []
    tries = 0
    while len(X) < n and tries < n * max_tries:
        tries += 1
        smp = synth.synthesize_random_window(
            window_size_cycles=WINDOW_CYCLES, force_active=active,
            force_plugged_all=True, compute_gt_harmonics=False,
            target_biased_placement=True,
        )
        gp = smp.gt_target_power_w
        if gp.get("oven", np.zeros(1))[ti] < OVEN_HEATER_MIN_W:
            continue
        if gp.get("hotplate", np.zeros(1))[ti] < HEAT_MIN_W:
            continue
        on = np.asarray(smp.gt_is_on["hotplate"], np.int8)
        X.append(format_inputs(smp))
        P.append(float(smp.power_features[ti, 0]))
        TR.append(int(np.abs(np.diff(on)).sum()))
    if not X:
        return None
    return {"x": np.stack(X), "p": np.asarray(P), "transitions": np.asarray(TR)}


def main() -> int:
    ap = argparse.ArgumentParser(description="녹화 단위 홀드아웃 (12.16.3절)")
    ap.add_argument("--seen", default="results/cnn_v17.pt",
                    help="hotplate_2 를 **본** 모델")
    ap.add_argument("--unseen", default="results/cnn_v18_nohp2.pt",
                    help="hotplate_2 를 **못 본** 모델")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/loro_probe.json")
    a = ap.parse_args()

    env_guard.verify_numerics()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ti = target_index(WINDOW_CYCLES)

    print("=" * 92)
    print("[녹화 단위 홀드아웃] 파형 암기 가설 — 시간축 홀드아웃이 못 보는 것 (12.16.3절)")
    print("=" * 92)

    evals = {
        "평가1 hotplate_2 (unseen 이 처음 봄)": {"hotplate": ["hotplate_2"]},
        "평가2 hotplate_1 (둘 다 봤음, 대조군)": {"hotplate": ["hotplate_1"]},
    }
    pops = {}
    for lab, only in evals.items():
        w = build_windows(only, a.n, ti, a.seed)
        if w is None:
            print(f"  {lab}: 창을 못 만들었습니다"); continue
        pops[lab] = w
        print(f"  {lab:38s} {len(w['x']):4d}창  P 중앙 {np.median(w['p']):7.1f}W  "
              f"60초 전이 중앙 {np.median(w['transitions']):.0f}회")

    models = {}
    for key, path in (("seen", a.seen), ("unseen", a.unseen)):
        if not Path(path).exists():
            print(f"\n  {path} 없음 — 건너뜁니다")
            continue
        models[key] = (path,) + load_model(path, dev)

    print()
    print(f"  {'평가셋':38s}" + "".join(f"{k:>26s}" for k in models))
    print(f"  {'':38s}" + "".join(f"{'핫플검출':>10s}{'포트오검':>10s}{'오븐':>7s}"
                                  for _ in models))
    payload: Dict[str, dict] = {"models": {k: v[0] for k, v in models.items()}}
    for lab, w in pops.items():
        cells, row = "", {}
        for key, (path, model, apps, _ck) in models.items():
            jo, jh, jk = (apps.index("oven"), apps.index("hotplate"),
                          apps.index("electiric_kettle"))
            g = gates_of(model, w["x"], dev)
            r = {"hotplate_on": float((g[:, jh] > 0.5).mean()),
                 "kettle_on": float((g[:, jk] > 0.5).mean()),
                 "oven_on": float((g[:, jo] > 0.5).mean()),
                 "hotplate_gate_median": float(np.median(g[:, jh]))}
            row[key] = r
            cells += f"{r['hotplate_on']:>10.3f}{r['kettle_on']:>10.3f}{r['oven_on']:>7.3f}"
        print(f"  {lab:38s}{cells}")
        payload[lab] = row

    if "seen" in models and "unseen" in models and len(payload) >= 3:
        print()
        print("  [판정] 핫플 검출률, seen - unseen")
        drops = {}
        for lab in pops:
            d = payload[lab]["seen"]["hotplate_on"] - payload[lab]["unseen"]["hotplate_on"]
            drops[lab] = d
            print(f"    {lab:38s} {d:+.3f}")
        labs = list(pops)
        if len(labs) == 2:
            unseen_drop, control_drop = drops[labs[0]], drops[labs[1]]
            print()
            if unseen_drop > 0.05 and unseen_drop > 2 * max(control_drop, 0.01):
                v = "**암기 확인** — 못 본 녹화에서만 떨어진다"
            elif max(unseen_drop, control_drop) <= 0.05:
                v = "**암기 아님** — 녹화를 안 봐도 같은 성능이다"
            elif control_drop > 0.05 and unseen_drop > 0.05:
                v = "**표본 부족** — 둘 다 떨어졌다. 녹화 정체가 아니라 데이터 양의 문제다"
            else:
                v = "판정 보류 — 패턴이 셋 중 어디에도 안 맞는다"
            print(f"  {v}")
            payload["verdict"] = v

    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
