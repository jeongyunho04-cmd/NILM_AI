"""
게이트 진단 — 소프트 게이트가 물리적으로 불가능한 전력을 내고 있는가 (12.9.14절)
==================================================================================
`net.py` 의 출력은 `power = sigmoid(on_logit) * p_raw` 다. 이 형태는 정격 1233W 인
전기포트에서 `sigma=0.381` 로 **469W** 를 내는 것을 허용한다. 그런 물리적 상태는 없다.

12.9.14 가 실측 실패를 이렇게 분해했다:

    전기포트    p_raw 1233W (정격 1290W)  x  sigmoid(on) 0.381  ->  469W
    핫플레이트  p_raw  397W (정격  468W)  x  sigmoid(on) 0.004  ->    2W

**크기는 둘 다 정확히 안다. 어느 쪽인지 결정을 못 한다.** 12.12.3 의 처방(헤지 벌점)은
손실 항이라 적응한 창에서만 듣는다 — 12.14.2 가 그것을 확인했다 (홀드아웃 핫플 F1 은
`w_hedge` 와 무관하게 0.63).

여기서는 **학습하지 않는다.** 저장된 가중치를 그대로 두고 추론에서만 게이트를
이진화해(`1[sigma>0.5] * p_raw`) 무엇이 바뀌는지 본다. 구조를 바꿔 재학습할
가치가 있는지 5분에 판정하기 위한 것이다.

    python -m src.run_gate_check --ckpt results/hedge_0.2.pt

[돌리기 전 예측 - 12.9.5 의 규율]
소프트 게이트로 학습된 모델이므로, 이진화하면 469W 짜리 유령 포트는 사라지지만
**핫플이 그것을 받아가지는 못하고 잔차만 커질** 것이다. 그렇게 나오면 "표현 공간이
오답을 허용한다" 는 진단은 맞고 해법은 재학습이다. 만약 핫플이 실제로 받아가면
재학습 없이 끝난다.
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

from src.evaluation.real_events import load_events, score_absent, score_on_off
from src.evaluation.sealing import is_sealed
from src.model.net import NILMNet, appliance_state_counts
from src.model.realdata import dense_targets, upsample_to_cycles
from src.run_baseline import S_I

# 헤지로 판정한다 - 이 밖이면 "결정했다", 안이면 "망설였다"
HEDGE_LO, HEDGE_HI = 0.05, 0.95


def load_model(ckpt_path: str, dev: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    apps = ck["appliances"]
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, apps, ck


@torch.no_grad()
def forward_file(model, stem: str, dev: str, stride: int = 30) -> dict:
    """파일 하나를 촘촘히 훑어 게이트와 원시 전력을 그대로 돌려준다."""
    rw = dense_targets(stem, stride=stride)
    G, R, SB, PN, POBS = [], [], [], [], []
    for i in range(0, len(rw), 512):
        idx = np.arange(i, min(i + 512, len(rw)))
        f, w, pobs, oh, pn = rw.batch(idx)
        ft = torch.from_numpy(np.ascontiguousarray(f)).to(dev)
        wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(ft, wt)
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        R.append(o["power_raw"].float().cpu().numpy())
        SB.append(o["standby"].float().cpu().numpy())
        PN.append(pn); POBS.append(pobs)
    return {"gate": np.concatenate(G), "p_raw": np.concatenate(R),
            "standby": np.concatenate(SB), "p_noise": np.concatenate(PN),
            "p_observed": np.concatenate(POBS), "targets": rw.target_cycle}


def gated(d: dict, hard: bool) -> np.ndarray:
    g = (d["gate"] > 0.5).astype(np.float32) if hard else d["gate"]
    return g * d["p_raw"]


def score_one(d: dict, P: np.ndarray, stem: str, apps: List[str], ev: dict) -> dict:
    resid = P.sum(1) + d["standby"].sum(1) + d["p_noise"] - d["p_observed"]
    n_cycles = int(ev[stem]["cycles"])
    on_c = upsample_to_cycles(d["gate"] > 0.5, d["targets"], n_cycles)
    ab = score_absent(P, stem, apps, pred_on=d["gate"] > 0.5, s_i=S_I, events=ev)
    f1 = score_on_off(on_c, stem, apps, events=ev)
    vals = [v["f1"] for v in f1.values() if v["n_true_on"] > 0]
    return {"absent_sum_w": ab["absent_sum_w"], "absent_fa_rel_max":
            max([v["fa_rel"] for v in ab["absent"].values()
                 if v["fa_rel"] == v["fa_rel"]] or [float("nan")]),
            "residual_abs_w": float(np.abs(resid).mean()),
            "on_off_f1_mean": float(np.mean(vals)) if vals else float("nan"),
            "per_app_f1": f1}


def hedge_report(d: dict, apps: List[str]) -> dict:
    """망설인 게이트가 만드는 전력이 얼마나 되는가."""
    g, p = d["gate"], d["p_raw"]
    soft = g * p
    hedging = (g > HEDGE_LO) & (g < HEDGE_HI)
    total = float(soft.sum())
    out = {"hedged_window_share": float(hedging.any(1).mean()),
           "hedged_power_share": float(soft[hedging].sum() / total) if total > 0 else 0.0}
    per = {}
    for j, a in enumerate(apps):
        m = hedging[:, j]
        if m.sum() == 0:
            continue
        per[a] = {"n": int(m.sum()), "share_of_windows": float(m.mean()),
                  "mean_gate": float(g[m, j].mean()),
                  "mean_emitted_w": float(soft[m, j].mean()),
                  "mean_rated_w": float(p[m, j].mean())}
    out["per_appliance"] = per
    return out


def oven_on_breakdown(d: dict, apps: List[str], ev: dict) -> dict:
    """`test_4` 의 오븐 히터 통전 구간에서 핫플/포트가 어떻게 갈리는가.

    12.9.5 가 "남은 실패 - 하나로 좁혀졌다" 고 한 바로 그 구간이다.
    """
    iv = ev["test_4"]["intervals"]
    n = int(ev["test_4"]["cycles"])

    def mask(pairs):
        m = np.zeros(n, bool)
        for s, e in pairs:
            m[int(s * 60):int(e * 60)] = True
        return m

    ov = mask(iv["oven"]["_heater_pulses"])[d["targets"]]
    hp = mask(iv["hotplate"]["on"])[d["targets"]]
    jh, jk, jo = (apps.index("hotplate"), apps.index("electiric_kettle"), apps.index("oven"))
    out = {}
    for name, m in (("오븐ON+핫플통전", ov & hp), ("오븐ON+핫플휴지", ov & ~hp),
                    ("오븐OFF+핫플통전", ~ov & hp)):
        if m.sum() < 5:
            continue
        out[name] = {
            "n": int(m.sum()),
            "hotplate_gate": float(np.median(d["gate"][m, jh])),
            "hotplate_soft_w": float(np.median(d["gate"][m, jh] * d["p_raw"][m, jh])),
            "hotplate_hard_w": float(np.median((d["gate"][m, jh] > 0.5) * d["p_raw"][m, jh])),
            "hotplate_recall": float((d["gate"][m, jh] > 0.5).mean()),
            "kettle_gate": float(np.median(d["gate"][m, jk])),
            "kettle_soft_w": float(np.median(d["gate"][m, jk] * d["p_raw"][m, jk])),
            "kettle_hard_w": float(np.median((d["gate"][m, jk] > 0.5) * d["p_raw"][m, jk])),
            "kettle_fa": float((d["gate"][m, jk] > 0.5).mean()),
            "oven_gate": float(np.median(d["gate"][m, jo])),
            "oven_soft_w": float(np.median(d["gate"][m, jo] * d["p_raw"][m, jo])),
            "p_observed": float(np.median(d["p_observed"][m])),
            "pred_total_w": float(np.median((d["gate"][m] * d["p_raw"][m]).sum(1))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="게이트 진단 — 소프트 vs 하드 (12.9.14절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_v15.pt", "results/hedge_0.2.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/gate_check.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    payload: Dict[str, dict] = {}

    for ckpt in a.ckpt:
        model, apps, ck = load_model(ckpt, dev)
        tag = Path(ckpt).stem
        print("=" * 88)
        print(f"[{tag}] stage {ck.get('stage', 1)} | {', '.join(stems)}")
        print("=" * 88)

        per_file, rows = {}, {"soft": [], "hard": []}
        for stem in stems:
            d = forward_file(model, stem, dev, stride=a.stride)
            s_soft = score_one(d, gated(d, False), stem, apps, ev)
            s_hard = score_one(d, gated(d, True), stem, apps, ev)
            per_file[stem] = {"soft": s_soft, "hard": s_hard, "hedge": hedge_report(d, apps)}
            rows["soft"].append(s_soft); rows["hard"].append(s_hard)
            if stem == "test_4":
                per_file[stem]["oven_on"] = oven_on_breakdown(d, apps, ev)

        K = [("absent_sum_w", "오귀속W"), ("absent_fa_rel_max", "최악FA"),
             ("residual_abs_w", "잔차W"), ("on_off_f1_mean", "on/off F1")]
        print(f"\n  {'':10s}" + "".join(f"{l:>24s}" for _, l in K))
        print(f"  {'':10s}" + "".join(f"{'소프트':>11s}{'하드':>13s}" for _ in K))
        for stem in stems:
            c = per_file[stem]
            print(f"  {stem:10s}" + "".join(
                f"{c['soft'][k]:>11.3f}{c['hard'][k]:>13.3f}" for k, _ in K))
        print(f"  {'평균':10s}" + "".join(
            f"{np.mean([r[k] for r in rows['soft']]):>11.3f}"
            f"{np.mean([r[k] for r in rows['hard']]):>13.3f}" for k, _ in K))

        print("\n  [망설임] 게이트가 (0.05, 0.95) 안에 있는 창")
        for stem in stems:
            h = per_file[stem]["hedge"]
            print(f"    {stem:8s} 창의 {100*h['hedged_window_share']:5.1f}% 에서 발생, "
                  f"예측 전력의 {100*h['hedged_power_share']:5.1f}% 를 만든다")
            for app, v in sorted(h["per_appliance"].items(),
                                 key=lambda x: -x[1]["mean_emitted_w"] * x[1]["share_of_windows"])[:3]:
                print(f"        {app:18s} 창 {100*v['share_of_windows']:5.1f}%  "
                      f"게이트 {v['mean_gate']:.3f}  정격 {v['mean_rated_w']:7.1f}W "
                      f"-> {v['mean_emitted_w']:7.1f}W 를 낸다")

        ob = per_file.get("test_4", {}).get("oven_on")
        if ob:
            print("\n  [test_4 오븐 히터 통전 구간] — 12.9.5 가 지목한 남은 실패")
            print(f"    {'구간':18s}{'n':>6s}{'오븐':>16s}{'핫플':>16s}{'포트':>16s}"
                  f"{'예측합':>10s}{'관측P':>10s}")
            for name, v in ob.items():
                print(f"    {name:18s}{v['n']:>6d}"
                      f"{v['oven_gate']:>7.3f}{v['oven_soft_w']:>8.1f}W"
                      f"{v['hotplate_gate']:>7.3f}{v['hotplate_soft_w']:>8.1f}W"
                      f"{v['kettle_gate']:>7.3f}{v['kettle_soft_w']:>8.1f}W"
                      f"{v['pred_total_w']:>9.1f}W{v['p_observed']:>9.1f}W")
        payload[tag] = per_file
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
