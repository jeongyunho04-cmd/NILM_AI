"""손실이 **정답을 원하는가** — 학습 없이 직접 계산한다 (13.58.1)
=================================================================
처방 전에 손실을 직접 계산해 정답 배분과 오답 배분을 견주는 규율이다. 13.49 에서
이렇게 해서 "손실이 그 답을 원한 것이 아니라 경로가 막힌 것" 을 갈랐고, 13.58 에서는
반대로 "손실이 오답을 실제로 선호한다" 를 확인했다.

    A  모델이 실제로 내는 배분
    B  참 배분 (`TRUE_W`, 13.52 의 조합 차분)

  B 가 모든 항에서 낮으면  -> 손실은 정답을 원하는데 못 간다 (**경로 문제**, 13.49 형)
  A 가 어떤 항에서 낮으면  -> **그 항이 오답을 선호한다** (13.58 형)

⚠ 바꾸는 것은 `out["power"]` 뿐이다. 관문·나머지 기기는 그대로 두므로 **배분만 바꾼
  대조**다. `_harm_pred_active` 가 `power/power_raw` 를 유효 게이트로 쓰므로 고조파
  예측도 그에 맞춰 따라간다.

표 끝에 **차수 묶음별** 정규화 오차도 낸다. 13.58.4 에서 h1~h7 은 참값이 이기고
h9~h15 는 모델 배분이 이겼다 — 전 차수를 뭉쳐 보면 그 반전이 안 보인다.

    python -m src.run_loss_compare --ckpt results/adapt_v24b_z_s0.pt
"""
from typing import Sequence
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.net import (appliance_state_counts, harmonic_scales, harmonic_signatures,  # noqa: F401
                           harmonic_signatures_by_state, noise_signature, standby_signatures)
from src.model.realdata import dense_targets
from src.run_baseline import S_I
from src.run_gate_check import load_model

SMPS = ("beam_projector", "laptop_charger", "minipc")
BIG = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
#: 자리 D 의 SMPS 전용 창에서 셋 다 켜졌을 때의 참 전력 (13.52 조합 차분)
TRUE_W = {"beam_projector": 43.0, "laptop_charger": 39.0, "minipc": 8.0}


def build_loss(apps: Sequence[str], dev: str, harm_weight: str = "inv_h2") -> NILMLoss:
    """`run_adapt` 와 같은 지문·척도로 손실을 짓는다."""
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig_state, _ = harmonic_signatures_by_state(pool, apps)
    return NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(harmonic_signatures(pool, apps)),
        standby_sig=torch.from_numpy(standby_signatures(pool, apps)),
        noise_sig=torch.from_numpy(noise_signature(pool)),
        harm_scale=torch.from_numpy(harmonic_scales(pool, apps)),
        signatures_state=torch.from_numpy(sig_state),
        harm_even_magnitude=True, harm_weight=harm_weight,
        smps_group=[apps.index(x) for x in SMPS if x in apps],
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stems", nargs="+", default=["test_3", "test_4"])
    ap.add_argument("--harm-weight", default="inv_h2")
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-harm", type=float, default=4.0)
    ap.add_argument("--w-hedge", type=float, default=0.2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, apps, _ = load_model(a.ckpt, dev)
    crit = build_loss(apps, dev, a.harm_weight)
    js = [apps.index(x) for x in SMPS]
    jb = [apps.index(x) for x in BIG]
    ev = load_events()
    kw = dict(w_cons=a.w_cons, w_harm=a.w_harm, w_hedge=a.w_hedge, w_over=0.0, w_pref=0.0)

    print(f"{a.ckpt}\n자리 D · SMPS 전용 · 셋 다 켜진 창.  "
          f"A = 모델 배분, B = 참 배분 "
          f"({'/'.join(f'{v:.0f}' for v in TRUE_W.values())}W)\n")
    print(f"  {'파일':8s}{'배분':6s}{'프로':>7s}{'충전':>7s}{'미니':>7s}"
          f"{'harm':>9s}{'cons':>9s}{'hedge':>9s}{'total':>9s}")
    for stem in a.stems:
        rw = dense_targets(stem, stride=30,
                           site_transfer=getattr(m, "site_transfer", None))
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        keep = np.flatnonzero((~on[t][:, jb].any(1)) & on[t][:, js].all(1)
                              & sc[t][:, js].all(1))
        if len(keep) < 20:
            continue
        acc = {}
        for i in range(0, len(keep), 256):
            idx = keep[i:i + 256]
            f, w, pobs, oh, pn = rw.batch(idx)
            o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
            tgt = {"p_observed": torch.from_numpy(pobs).float().to(dev),
                   "obs_harm": torch.from_numpy(oh).float().to(dev),
                   "p_noise": torch.from_numpy(pn).float().to(dev),
                   "harm_offset": None}
            outB = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in o.items()}
            for j, nm in zip(js, SMPS):
                outB["power"][:, j] = TRUE_W[nm]
            for tag, oo in (("A", o), ("B", outB)):
                parts = crit.unlabeled(oo, tgt, **kw)
                d = acc.setdefault(tag, {})
                for k in ("harm", "cons", "hedge", "total"):
                    if k in parts:
                        d[k] = d.get(k, 0.0) + float(parts[k]) * len(idx)
                d["n"] = d.get("n", 0) + len(idx)
                d["w"] = d.get("w", np.zeros(3)) + oo["power"][:, js].float().cpu().numpy().sum(0)
        for tag in ("A", "B"):
            d = acc[tag]
            n = d["n"]
            print(f"  {stem if tag == 'A' else '':8s}{tag:6s}"
                  + "".join(f"{v / n:7.1f}" for v in d["w"])
                  + "".join(f"{d.get(k, 0.0) / n:9.4f}"
                            for k in ("harm", "cons", "hedge", "total")))
    print("\n  B 가 모든 항에서 낮으면 손실은 정답을 원한다 (경로 문제).")
    print("  A 가 어떤 항에서 낮으면 **그 항이 오답을 선호**한다.")

    # ── 차수 묶음별 (13.58.4) ────────────────────────────────────────────────
    # 손실의 harm 은 `harm_weight`·`harm_scale`·상태 지문이 얽혀 있어 차수를 잘라
    # 보기 어렵다. 여기서는 **고정 지문 x 참/모델 전력**의 차수별 정규화 오차를
    # 직접 계산한다. 13.58.4 가 이걸로 h1~h7 과 h9~h15 의 반전을 찾았다.
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    nzs = noise_signature(pool)
    S = {x: sig[apps.index(x), :, 0] + 1j * sig[apps.index(x), :, 1] for x in SMPS}
    OH = []
    for stem in a.stems:
        rw = dense_targets(stem, stride=30, site_transfer=None)
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        k = np.flatnonzero((~on[t][:, jb].any(1)) & on[t][:, js].all(1)
                           & sc[t][:, js].all(1))
        for i in range(0, len(k), 512):
            OH.append(rw.batch(k[i:i + 512])[3])
    if not OH:
        return
    obs = np.concatenate(OH)
    obs = obs[..., 0] + 1j * obs[..., 1]
    scale = np.median(np.abs(obs), 0) + 1e-9
    def _err(alloc, orders):
        pred = nzs[:, 0] + 1j * nzs[:, 1]
        for x, wv in alloc.items():
            pred = pred + wv * S[x]
        c = [h - 1 for h in orders]
        return float(np.mean(np.abs(obs[:, c] - pred[c]) / scale[c]))

    A_alloc = {x: float(acc["A"]["w"][i] / acc["A"]["n"]) for i, x in enumerate(SMPS)}
    print("\n  차수 묶음별 정규화 오차 (고정 지문 x 전력, 낮을수록 좋다)")
    print(f"  {'차수 묶음':16s}{'A 모델':>10s}{'B 참값':>10s}{'승자':>10s}")
    for nm, o in (("h1 만", [1]), ("h1~h5", [1, 3, 5]), ("h1~h7", [1, 3, 5, 7]),
                  ("h9~h15", [9, 11, 13, 15]), ("전 홀수차", [1, 3, 5, 7, 9, 11, 13, 15])):
        ea, eb = _err(A_alloc, o), _err(TRUE_W, o)
        print(f"  {nm:16s}{ea:10.4f}{eb:10.4f}{('B 참값' if eb < ea else 'A 모델'):>10s}")
    print("  ⚠ 저차와 고차의 승자가 다르면 `harm_weight` 가 어느 쪽을 보는지 확인할 것"
          " — inv_h2 는 h15 를 h1 대비 225배 누른다 (13.58.4).")


if __name__ == "__main__":
    main()
