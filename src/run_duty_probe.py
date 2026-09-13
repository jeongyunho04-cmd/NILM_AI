"""
듀티 프로브 — 모델이 시간 구조로 저항 부하를 가를 줄 아는가 (12.15.4절)
=========================================================================
12.15 가 실측 실패를 여기까지 좁혔다.

    관측 1,608W 를 설명하는 두 가설
      A  오븐 히터(1,140W) + 핫플레이트(468W, 릴레이 통전 0.90초)
      B  전기포트 한 대 (정격 1,533W)

    전력으로 동점이고 (포트 정격이 관측을 그대로 설명한다)
    고조파로도 동점이다 (가설B 의 I3 구간 [0.2787, 0.3527] 이 가설A 의 0.2811 을 덮는다)

**남는 판별자는 시간 구조뿐이다.** 핫플은 0.9초씩 끊어 통전하고 포트는 켜지면 연속이다.

여기서는 **학습하지 않는다.** 두 가설을 합성으로 만들어 - 정답을 아는 채로 -
전력을 맞춘 뒤 모델에 직접 물어본다. 합성 홀드아웃에는 이 상황이 1.06% 뿐이라
(12.9.12절) 기존 지표로는 보이지 않는다.

    python -m src.run_duty_probe --ckpt results/hedge_0.2.pt

[두 축을 함께 본다]
  저항만        A/B 를 저항 부하만으로 만든다. 판별의 상한이다
  저항+SMPS     프로젝터+충전기를 함께 켠다. `test_4` 의 실제 상황이고,
                12.15.3 이 "SMPS 기저가 저항 고조파를 덮는다" 고 한 것을 검증한다

[판정]
  ① 저항만에서 갈린다  -> 모델은 듀티를 쓸 줄 안다
     ②-1 SMPS 를 넣으면 무너진다 -> 12.15.3 이 맞다. 처방은 SMPS 바탕 보정
     ②-2 SMPS 를 넣어도 갈린다   -> 합성-실측 격차 문제. 처방은 sim-to-real
  ③ 저항만에서도 못 가른다 -> 목적함수 문제. seq2point 가 시간 구조를 요구하지 않는다
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

from src.model.inputs import FINE_CYCLES, build_inputs, target_index
from src.run_gate_check import load_model

WINDOW_CYCLES = 3600
HEAT_MIN_W = 300.0            # 저항 부하가 "통전 중" 이라고 볼 하한
OVEN_HEATER_MIN_W = 800.0     # 오븐이 팬/조명이 아니라 히터 상태
SMPS = ["beam_projector", "laptop_charger"]   # test_4 의 저부하 동반자


def format_inputs(sample) -> np.ndarray:
    """합성 샘플 -> (33, W). `dataset.NILMBatchGenerator._format_inputs` 와 같은 배열."""
    hr = np.asarray(sample.harmonics_ri, np.float32)          # (N,15,2)
    pf = np.asarray(sample.power_features, np.float32)        # (N,6)
    x = np.empty((33, hr.shape[0]), np.float32)
    x[0:15] = hr[:, :, 0].T
    x[15:30] = hr[:, :, 1].T
    x[30], x[31], x[32] = pf[:, 0], pf[:, 1], pf[:, 4]
    return x


def duty_transitions(sample, app: str) -> int:
    """세밀 갈래(뒤 10초) 안에서 이 기기가 껐다 켜진 횟수.

    모델이 볼 수 있는 시간 구조의 양이다. 포트는 0, 핫플은 5회 근처가 나와야 한다.
    """
    on = np.asarray(sample.gt_is_on.get(app, np.zeros(1)), np.int8)
    if on.size < FINE_CYCLES:
        return 0
    seg = on[-FINE_CYCLES:]
    return int(np.abs(np.diff(seg)).sum())


def build_population(synth, kind: str, with_smps: bool, n: int, ti: int,
                     rng_seed: int, max_tries: int = 60) -> Optional[dict]:
    """가설 A 또는 B 의 창을 n 개 만든다. 타깃 시점 조건을 만족할 때까지 다시 뽑는다."""
    np.random.seed(rng_seed)
    resistive = ["oven", "hotplate"] if kind == "A" else ["electiric_kettle"]
    active = resistive + (SMPS if with_smps else [])
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
        if kind == "A":
            if gp.get("oven", np.zeros(1))[ti] < OVEN_HEATER_MIN_W:
                continue
            if gp.get("hotplate", np.zeros(1))[ti] < HEAT_MIN_W:
                continue
        else:
            if gp.get("electiric_kettle", np.zeros(1))[ti] < OVEN_HEATER_MIN_W:
                continue
        X.append(format_inputs(smp))
        P.append(float(smp.power_features[ti, 0]))
        TR.append(duty_transitions(smp, "hotplate" if kind == "A" else "electiric_kettle"))
    if not X:
        return None
    return {"x": np.stack(X), "p": np.asarray(P), "transitions": np.asarray(TR),
            "tries": tries}


@torch.no_grad()
def gates_of(model, x: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    G = []
    for i in range(0, len(x), batch):
        fine, wide = build_inputs(x[i:i + batch])
        f = torch.from_numpy(np.ascontiguousarray(fine)).to(dev)
        w = torch.from_numpy(np.ascontiguousarray(wide)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(f, w)
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
    return np.concatenate(G)


def main() -> int:
    ap = argparse.ArgumentParser(description="듀티 프로브 (12.15.4절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_v15.pt", "results/hedge_0.2.pt"])
    ap.add_argument("--n", type=int, default=400, help="가설당 창 개수")
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    help="전력 정합 구간 (W). 생략하면 두 모집단의 겹치는 구간에서 자동")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/duty_probe.json")
    a = ap.parse_args()

    env_guard.verify_numerics()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ti = target_index(WINDOW_CYCLES)

    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    synth = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)

    print("=" * 92)
    print("[듀티 프로브] 오븐히터+핫플(듀티)  vs  전기포트(연속) — 전력을 맞추고 묻는다")
    print("=" * 92)

    pops: Dict[str, dict] = {}
    for with_smps in (False, True):
        lab = "저항+SMPS" if with_smps else "저항만"
        for kind in ("A", "B"):
            key = f"{lab}/{kind}"
            p = build_population(synth, kind, with_smps, a.n, ti, a.seed + (1 if kind == "B" else 0))
            if p is None:
                print(f"  {key}: 창을 못 만들었습니다"); continue
            pops[key] = p
            what = "오븐히터+핫플" if kind == "A" else "포트"
            print(f"  {key:16s} {what:12s} {len(p['x']):4d}창  "
                  f"P 중앙 {np.median(p['p']):7.1f}W [{np.percentile(p['p'],10):.0f}, "
                  f"{np.percentile(p['p'],90):.0f}]  세밀창 내 전이 중앙 {np.median(p['transitions']):.0f}회")

    payload: Dict[str, dict] = {"populations": {
        k: {"n": len(v["x"]), "p_median": float(np.median(v["p"])),
            "transitions_median": float(np.median(v["transitions"]))} for k, v in pops.items()}}

    for ckpt in a.ckpt:
        model, apps, ck = load_model(ckpt, dev)
        jo, jh, jk = apps.index("oven"), apps.index("hotplate"), apps.index("electiric_kettle")
        tag = Path(ckpt).stem
        print("\n" + "=" * 92)
        print(f"[{tag}] stage {ck.get('stage', 1)}")
        print("=" * 92)
        res: Dict[str, dict] = {}

        for lab in ("저항만", "저항+SMPS"):
            ka, kb = f"{lab}/A", f"{lab}/B"
            if ka not in pops or kb not in pops:
                continue
            A, B = pops[ka], pops[kb]
            # 전력 정합 - 두 모집단이 겹치는 구간만 비교한다
            lo, hi = (a.band if a.band else
                      (max(np.percentile(A["p"], 5), np.percentile(B["p"], 5)),
                       min(np.percentile(A["p"], 95), np.percentile(B["p"], 95))))
            ma, mb = (A["p"] >= lo) & (A["p"] <= hi), (B["p"] >= lo) & (B["p"] <= hi)
            if ma.sum() < 20 or mb.sum() < 20:
                print(f"\n  [{lab}] 정합 구간 [{lo:.0f}, {hi:.0f}]W 표본 부족 "
                      f"(A {ma.sum()}, B {mb.sum()})")
                continue
            ga, gb = gates_of(model, A["x"][ma], dev), gates_of(model, B["x"][mb], dev)
            r = {
                "band_w": [float(lo), float(hi)], "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                "p_a": float(np.median(A["p"][ma])), "p_b": float(np.median(B["p"][mb])),
                "A_hotplate_on": float((ga[:, jh] > 0.5).mean()),
                "A_oven_on": float((ga[:, jo] > 0.5).mean()),
                "A_kettle_on": float((ga[:, jk] > 0.5).mean()),
                "B_kettle_on": float((gb[:, jk] > 0.5).mean()),
                "B_hotplate_on": float((gb[:, jh] > 0.5).mean()),
                "B_oven_on": float((gb[:, jo] > 0.5).mean()),
            }
            # 두 가설을 가르는 능력: 포트 게이트가 A 와 B 에서 얼마나 벌어지는가
            r["kettle_gate_separation"] = float(np.median(gb[:, jk]) - np.median(ga[:, jk]))
            r["hotplate_gate_separation"] = float(np.median(ga[:, jh]) - np.median(gb[:, jh]))
            res[lab] = r

            print(f"\n  [{lab}]  전력 정합 [{lo:.0f}, {hi:.0f}]W   "
                  f"A {ma.sum()}창(중앙 {r['p_a']:.0f}W) / B {mb.sum()}창(중앙 {r['p_b']:.0f}W)")
            print(f"    {'':22s}{'오븐':>9s}{'핫플':>9s}{'포트':>9s}")
            print(f"    가설A (오븐+핫플)    {r['A_oven_on']:>9.3f}{r['A_hotplate_on']:>9.3f}"
                  f"{r['A_kettle_on']:>9.3f}   <- 핫플이 높고 포트가 낮아야 정답")
            print(f"    가설B (포트)         {r['B_oven_on']:>9.3f}{r['B_hotplate_on']:>9.3f}"
                  f"{r['B_kettle_on']:>9.3f}   <- 포트가 높고 핫플이 낮아야 정답")
            print(f"    판별력  포트 게이트 격차 {r['kettle_gate_separation']:+.3f}   "
                  f"핫플 게이트 격차 {r['hotplate_gate_separation']:+.3f}")

            # 가설A 안에서: 세밀창 전이 횟수가 많을수록 핫플을 잘 잡는가
            tr = A["transitions"][ma]
            bins = [(0, 0), (1, 2), (3, 4), (5, 99)]
            rows = []
            for blo, bhi in bins:
                m = (tr >= blo) & (tr <= bhi)
                if m.sum() < 10:
                    continue
                rows.append({"transitions": f"{blo}-{bhi}" if blo != bhi else str(blo),
                             "n": int(m.sum()),
                             "hotplate_on": float((ga[m, jh] > 0.5).mean()),
                             "kettle_on": float((ga[m, jk] > 0.5).mean())})
            if len(rows) >= 2:
                r["by_transitions"] = rows
                print(f"    가설A 를 세밀창 내 핫플 전이 횟수로 나누면 "
                      f"(듀티를 쓴다면 위→아래로 올라야 한다):")
                for w in rows:
                    print(f"      전이 {w['transitions']:>5s}회  n={w['n']:>4d}  "
                          f"핫플 검출 {w['hotplate_on']:.3f}   포트 오검 {w['kettle_on']:.3f}")
        payload[tag] = res

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
