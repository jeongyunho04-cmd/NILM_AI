"""
Phase 5 — 2단계 실측 준지도 적응 (설계 문서 4.2절)
====================================================
1단계(합성 사전학습)가 끝난 모델을 **기기별 라벨 없이** 실측 복합 부하에 맞춘다.

    L_adapt = L_cons(실측) + L_harm(실측)  +  λ · L_전체(합성)

고치려는 것은 0.6절의 sim-to-real 편차다. 합성 미니PC 17.6W vs 실측 28.4W 처럼
**저부하 기기가 체계적으로 17~34% 낮게 예측되는 것**을 실측 총전력으로 끌어올린다.
적응 전 v11 은 실측 총전력을 평균 −13.9W 로 과소 예측한다.

    python -m src.run_adapt --init results/cnn_v11.pt --tag adapt_v1

[왜 두 항을 함께 걸어야 하는가]
`L_cons` 는 합이 얼마나 모자란지만 알고 **어느 기기인지 모른다.** 실측에는 기기별
라벨이 아예 없으므로 `L_harm` 없이는 배분이 미결정인 채 아무 기기에나 붙는다.
이 두 항이 2단계의 분해를 통째로 떠받친다 (3.3·3.4·4.2절).

[왜 합성 손실을 함께 유지하는가]
파괴적 망각 방지다. λ=0.5 로 1단계 손실 전체를 계속 건다. 학습률은 1단계의 1/10.

[봉인]
`test.csv` 는 최종 평가 전용이라 적응에도 채점에도 쓰지 않는다 (4.3절).
`RealWindows` 가 목록에서 빼고, 채점도 봉인 해제 없이 도는 파일만 한다.
"""
from pathlib import Path
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation import load_holdout, resistive_confusion, score_appliances, summarize, total_power_residual
from src.evaluation.real_events import load_events, score_events, score_on_off
from src.evaluation.sealing import is_sealed
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
    noise_signature, standby_signatures,
)
from src.model.realdata import RealWindows, dense_targets, upsample_to_cycles
from src.model.traincache import CachedWindows
from src.run_baseline import LOW_LOAD, S_I
from src.run_train_cnn import evaluate, prepare_holdout_inputs, report, to_targets

HOLDOUT_DIR = "processed_data/holdout60"


def real_targets(b, dev):
    fine, wide, pobs, oh, pn = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in b]
    return fine, wide, {"p_observed": pobs, "obs_harm": oh, "p_noise": pn}


@torch.no_grad()
def score_real_files(model, apps, dev, stride: int = 30) -> dict:
    """실측 파일별 채점 — 총전력 잔차 / on-off F1 / 이벤트 ΔP.

    12.4절: 실측에는 기기별 정답이 없으므로 MAE·FA·RE 는 못 낸다. 낼 수 있는 것만 낸다.
    창을 사이클마다 만들면 파일당 9만 창이라, 0.5초 간격 예측을 계단 보간한다.
    """
    model.eval()
    ev = load_events()
    out = {}
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        rw = dense_targets(stem, stride=stride)
        P, ON = [], []
        for i in range(0, len(rw), 512):
            f, w, tg = real_targets(rw.batch(np.arange(i, min(i + 512, len(rw)))), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(f, w)
            P.append(o["power"].float().cpu().numpy())
            ON.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            resid = (o["power"].sum(1) + o["standby"].sum(1) + tg["p_noise"]
                     - tg["p_observed"]).float().cpu().numpy()
            out.setdefault(stem + "_resid", []).append(resid)
        P, ON = np.concatenate(P), np.concatenate(ON)
        r = np.concatenate(out.pop(stem + "_resid"))
        n_cycles = int(ev[stem]["cycles"])
        pc = upsample_to_cycles(P, rw.target_cycle, n_cycles)
        oc = upsample_to_cycles(ON > 0.5, rw.target_cycle, n_cycles)
        out[stem] = {
            "n_windows": len(rw),
            "residual_mean_w": float(r.mean()), "residual_abs_w": float(np.abs(r).mean()),
            "on_off": score_on_off(oc, stem, apps, events=ev),
            "events": score_events(pc, stem, apps, events=ev),
        }
    model.train()
    return out


def summarize_real(rs: dict) -> dict:
    f1 = [v["f1"] for s in rs.values() for v in s["on_off"].values() if v["n_true_on"] > 0]
    er = [abs(e["error_rel"]) for s in rs.values() for e in s["events"] if e["error_rel"] == e["error_rel"]]
    n = sum(s["n_windows"] for s in rs.values())
    return {
        "residual_mean_w": float(np.mean([s["residual_mean_w"] for s in rs.values()])),
        "residual_abs_w": float(np.mean([s["residual_abs_w"] for s in rs.values()])),
        "on_off_f1_mean": float(np.mean(f1)) if f1 else float("nan"),
        "event_abs_rel_mean": float(np.mean(er)) if er else float("nan"),
        "n_scored_windows": int(n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 — 2단계 실측 준지도 적응 (4.2절)")
    ap.add_argument("--init", default="results/cnn_v11.pt", help="1단계 체크포인트")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256, help="실측/합성 각각의 배치 크기")
    ap.add_argument("--lr", type=float, default=3e-5, help="1단계(3e-4)의 1/10")
    ap.add_argument("--lam", type=float, default=0.5, help="합성 replay 가중 (4.2절)")
    ap.add_argument("--w-cons", type=float, default=0.4, help="실측 보존 손실 (4.2절)")
    ap.add_argument("--w-harm", type=float, default=0.1, help="실측 고조파 제약")
    ap.add_argument("--w-over", type=float, default=0.0)
    ap.add_argument("--real-stride", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--cache", default="cache/train60")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="adapt")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    env_guard.verify_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 84)
    print("[Phase 5] 2단계 실측 준지도 적응 — 기기별 라벨 없이 (4.2절)")
    print("=" * 84)

    hs = load_holdout(HOLDOUT_DIR)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    hs.X = np.zeros((len(prep[0]), 1, 1), np.float32)

    rw = RealWindows(stride=a.real_stride)
    print(rw.describe())
    print(f"합성 replay: {a.cache} | λ={a.lam} | w_cons={a.w_cons} w_harm={a.w_harm}")

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, sb, nz, hsc = (harmonic_signatures(pool, apps), standby_signatures(pool, apps),
                        noise_signature(pool), harmonic_scales(pool, apps))
    del pool

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5)).to(dev)
    model.load_state_dict(ck["model"])
    print(f"1단계 체크포인트: {a.init} (ep{ck.get('epoch')}, width {ck.get('width')})")

    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig), standby_sig=torch.from_numpy(sb),
        noise_sig=torch.from_numpy(nz), harm_scale=torch.from_numpy(hsc),
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cache = CachedWindows(a.cache)
    rng = np.random.default_rng(a.seed)

    def snapshot(tag: str) -> dict:
        pred, onp = evaluate(model, prep, dev)
        sc, summ, cm, resid = report(pred, onp, hs)
        rs = score_real_files(model, apps, dev)
        rsum = summarize_real(rs)
        print(f"\n  [{tag}] 합성 홀드아웃  MAE {summ['mae_w_mean']:.3f}W  F1 {summ['f1_mean']:.4f}  "
              f"저항3종 {cm['accuracy']:.3f}  잔차 {resid['mean_abs_w']:.1f}W")
        print(f"  {'':{len(tag)+4}s}실측          잔차 평균 {rsum['residual_mean_w']:+.2f}W  "
              f"절대 {rsum['residual_abs_w']:.2f}W  on/off F1 {rsum['on_off_f1_mean']:.3f}  "
              f"이벤트 |상대오차| {rsum['event_abs_rel_mean']:.3f}")
        return {"tag": tag, "synth": {"mae": summ["mae_w_mean"], "f1": summ["f1_mean"],
                                      "resistive_acc": cm["accuracy"],
                                      "resid_abs": resid["mean_abs_w"]},
                "real": rsum, "real_detail": rs,
                "summary": summ, "resistive_confusion": cm, "total_power_residual": resid,
                "per_appliance": [x.as_row() for x in sc]}

    hist = [snapshot("적응 전")]
    t0 = time.time()
    agg, nb = {}, 0
    for step in range(1, a.steps + 1):
        ridx = rng.choice(len(rw), a.batch, replace=len(rw) < a.batch)
        rf, rwd, rtg = real_targets(rw.batch(ridx), dev)
        sidx = np.sort(rng.choice(len(cache), a.batch, replace=False))
        sb_ = tuple(torch.from_numpy(x) for x in cache.batch(sidx))
        sf, swd, stg = to_targets(sb_, dev)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            rp = crit.unlabeled(model(rf, rwd), rtg, w_cons=a.w_cons,
                                w_harm=a.w_harm, w_over=a.w_over)
            sp = crit(model(sf, swd), stg)
            loss = rp["total"] + a.lam * sp["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for k, v in (("real_cons", rp["cons"]), ("real_harm", rp["harm"]),
                     ("synth_total", sp["total"]), ("loss", loss)):
            d = v.detach()
            agg[k] = d if k not in agg else agg[k] + d
        nb += 1

        if step % a.eval_every == 0 or step == a.steps:
            m = {k: float(v) / nb for k, v in agg.items()}
            print(f"\n  step {step:>5d}/{a.steps}  loss {m['loss']:.4f} "
                  f"(실측 cons {m['real_cons']:.2f} harm {m['real_harm']:.3f} / "
                  f"합성 {m['synth_total']:.4f})  [{time.time()-t0:.0f}s]", flush=True)
            hist.append(snapshot(f"step {step}"))
            agg, nb = {}, 0

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "appliances": apps,
                "width": ck.get("width", 1.0), "epoch": a.steps,
                "prior_kappa": ck.get("prior_kappa", 0.0),
                "prior_beta": ck.get("prior_beta", 0.5),
                "select": "final", "stage": 2, "init": a.init},
               out / f"{a.tag}.pt")

    first, last = hist[0], hist[-1]
    print(f"\n{'=' * 84}\n적응 전 -> 후\n{'=' * 84}")
    print(f"  {'':26s}{'전':>12s}{'후':>12s}")
    for lab, kk in [("실측 잔차 평균 (W)", ("real", "residual_mean_w")),
                    ("실측 잔차 절대 (W)", ("real", "residual_abs_w")),
                    ("실측 on/off F1", ("real", "on_off_f1_mean")),
                    ("실측 이벤트 |상대오차|", ("real", "event_abs_rel_mean")),
                    ("합성 MAE (W)", ("synth", "mae")),
                    ("합성 F1", ("synth", "f1")),
                    ("합성 저항3종", ("synth", "resistive_acc")),
                    ("합성 잔차 (W)", ("synth", "resid_abs"))]:
        g, k = kk
        print(f"  {lab:26s}{first[g][k]:>12.4f}{last[g][k]:>12.4f}")

    payload = {"phase": "5-adapt", "config": vars(a),
               "holdout": {k: hs.meta[k] for k in ("n_windows", "content_sha256", "target_index")},
               "history": [{k: v for k, v in h.items() if k != "real_detail"} for h in hist],
               "real_detail": last["real_detail"],
               "per_appliance": last["per_appliance"], "summary": last["summary"],
               "resistive_confusion": last["resistive_confusion"],
               "total_power_residual": last["total_power_residual"],
               "elapsed_s": round(time.time() - t0, 1)}
    (out / f"{a.tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n완료 {time.time()-t0:.0f}s | {out / f'{a.tag}.json'}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
