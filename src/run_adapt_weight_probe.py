"""적응 손실이 **어느 창에서 나오는가** — 고부하가 먹고 있는지 잰다 (12.94절)

12.31 이래 운영은 하이브리드다. 그 이유가 "적응이 SMPS 게이트를 망가뜨린다" 인데,
**왜** 망가지는지는 안 쟀다. 후보 가설 하나:

    L_cons = |Σ P̂ + Σ Ŝ + P_noise − P_관측| 의 **절대 와트 평균**이다.
    핫플·오븐이 켜진 창은 오차가 수십 W 이고 SMPS 만 있는 창은 1W 규모다.
    창을 균등 추첨하므로(run_adapt 의 rng.choice) **기울기는 고부하 창이 쥔다.**

그 가설을 재는 도구다. 창을 관측 총전력으로 층화해 각 층이 손실 합의 몇 %를
내는지 본다. `--ckpt` 는 적응 **직전** 모델(1단계)을 주는 것이 기본 용법이다 —
기울기 균형이 문제되는 시점이 그때다.

    python -m src.run_adapt_weight_probe --ckpt results/cnn_ze1.pt
    python -m src.run_adapt_weight_probe --ckpt results/cnn_ze1.pt --ckpt2 results/adapt_ze1.pt

> **주의.** 여기서 나오는 것은 **손실 기여**이지 기울기 크기가 아니다. 둘은
> 비례하지 않는다 (|·| 의 기울기는 부호뿐이다). 그래서 `--grad` 로 실제
> 기울기 노름도 층별로 잰다. 손실 기여만 보고 "고부하가 학습을 먹는다" 로
> 결론내면 12.79 의 반복이다 (산술을 보고 원인이라 불렀다).
"""
from typing import Dict, List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.model.inputs import LEGACY_FINE_CHANNELS
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.net import (NILMNet, appliance_state_counts, harmonic_scales,
                           harmonic_signatures, noise_signature, standby_signatures)
from src.model.realdata import RealWindows
from src.run_adapt import real_targets
from src.run_baseline import S_I

#: 관측 총전력 층. SMPS 3종은 다 켜도 150W 를 넘지 않는다 (프로젝터 48 + 충전기 45 + 미니PC 19).
BINS = [(0, 150, "SMPS 만 (<150W)"), (150, 600, "중간 (150~600W)"),
        (600, 1500, "고부하 (600~1500W)"), (1500, 1e9, "최고부하 (>1500W)")]


def load(ckpt: str, apps, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    model.load_state_dict(ck["model"])
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="적응 손실이 어느 창에서 나오는지 잰다")
    ap.add_argument("--ckpt", default="results/cnn_ze1.pt", help="적응 직전(1단계) 모델")
    ap.add_argument("--ckpt2", default="", help="비교용 두 번째 모델 (예: 적응 후)")
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-harm", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--holdout-real", default="test_8")
    ap.add_argument("--grad", action="store_true",
                    help="층별 기울기 노름도 잰다 (창 하나씩 역전파. 느리다)")
    ap.add_argument("--grad-n", type=int, default=40, help="층마다 기울기를 잴 창 수")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    held = [x.strip() for x in a.holdout_real.split(",") if x.strip()]
    rw = RealWindows(stride=a.stride, exclude=held or None)

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    apps = sorted(pool.get_appliance_types())
    sig, sbs, nz, hsc = (harmonic_signatures(pool, apps), standby_signatures(pool, apps),
                         noise_signature(pool), harmonic_scales(pool, apps))
    del pool

    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig), standby_sig=torch.from_numpy(sbs),
        noise_sig=torch.from_numpy(nz), harm_scale=torch.from_numpy(hsc),
        harm_odd_only=True, weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)

    P = rw.p_observed
    print("=" * 92)
    print(f"[적응 손실의 출처] 실측 {len(rw):,}창 | w_cons={a.w_cons} w_harm={a.w_harm}")
    print("=" * 92)

    models = [(a.ckpt, load(a.ckpt, apps, dev))]
    if a.ckpt2:
        models.append((a.ckpt2, load(a.ckpt2, apps, dev)))

    for name, model in models:
        model.eval()
        cons_w, harm_w = np.zeros(len(rw)), np.zeros(len(rw))
        with torch.no_grad():
            for s in range(0, len(rw), 256):
                idx = np.arange(s, min(s + 256, len(rw)))
                rf, rwd, rtg = real_targets(rw.batch(idx), dev)
                out = model(rf, rwd)
                recon = out["power"].sum(1) + out["standby"].sum(1) + rtg["p_noise"]
                cons_w[idx] = (recon - rtg["p_observed"]).abs().float().cpu().numpy()
                pred = torch.einsum("bk,khc->bhc", out["power"], crit.sig)
                idle = (torch.sigmoid(out["plugged_logit"])
                        * (1.0 - torch.sigmoid(out["on_logit"])))
                pred = pred + torch.einsum("bk,khc->bhc", idle, crit.standby_sig)
                pred = pred + crit.noise_sig[None]
                err = (pred - rtg["obs_harm"]).abs() / crit.harm_scale[None, :, None]
                m = crit.harm_mask[None, :, None]
                harm_w[idx] = ((err * m).mean((1, 2)) / m.mean().clamp(min=1e-6)
                               ).float().cpu().numpy()

        tot = a.w_cons * cons_w + a.w_harm * harm_w
        print(f"\n  {name}")
        print(f"  {'층':22s}{'창':>8s}{'비중':>7s}{'|cons|W':>10s}"
              f"{'harm':>8s}{'손실 합 비중':>13s}{'창당 손실':>11s}")
        for lo, hi, lab in BINS:
            m = (P >= lo) & (P < hi)
            if not m.any():
                print(f"  {lab:22s}{0:>8d}")
                continue
            print(f"  {lab:22s}{m.sum():>8d}{100*m.mean():>6.0f}%{cons_w[m].mean():>10.2f}"
                  f"{harm_w[m].mean():>8.3f}{100*tot[m].sum()/tot.sum():>12.0f}%"
                  f"{tot[m].mean():>11.3f}")
        print(f"  {'전체':22s}{len(rw):>8d}{100:>6.0f}%{cons_w.mean():>10.2f}"
              f"{harm_w.mean():>8.3f}{100:>12.0f}%{tot.mean():>11.3f}")

        if a.grad:
            print(f"\n  [층별 기울기 노름] 층마다 {a.grad_n}창, 창 하나씩 역전파")
            rng = np.random.default_rng(0)
            for lo, hi, lab in BINS:
                m = np.flatnonzero((P >= lo) & (P < hi))
                if not len(m):
                    continue
                pick = rng.choice(m, min(a.grad_n, len(m)), replace=False)
                norms = []
                for i in pick:
                    model.zero_grad(set_to_none=True)
                    rf, rwd, rtg = real_targets(rw.batch(np.array([i])), dev)
                    rp = crit.unlabeled(model(rf, rwd), rtg, w_cons=a.w_cons,
                                        w_harm=a.w_harm, w_over=0.0, w_hedge=0.0)
                    rp["total"].backward()
                    g = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                       for p in model.parameters() if p.grad is not None))
                    norms.append(float(g))
                norms = np.array(norms)
                print(f"  {lab:22s}‖∇‖ 평균 {norms.mean():8.4f}  중앙 {np.median(norms):8.4f}")
            model.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
