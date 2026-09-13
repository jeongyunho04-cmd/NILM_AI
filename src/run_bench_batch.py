# -*- coding: utf-8 -*-
"""주 루프의 **창/초 천장**이 배치 크기로 올라가나 (13.84.63).

측정 (972758/972759, 2026-09-12): 48코어를 할당받아 **37.4개만** 쓰고, 워커 46개가 각각
76%(Ada)·65%(A10) 만 돌며, GPU 는 34~42% 다. 그런데 두 판의 **주 프로세스가 각각 한 코어의
94%** 에 붙어 있다. 워커가 만들어 놓고 기다린다 — **주 루프가 천장이다.**

워커당 처리량도 이미 꺾였다:
    972281  워커 18   4,180 창/초 = 232/워커   413초/epoch
    972758  워커 23   4,655 창/초 = 202/워커   371초/epoch

가설: 주 루프 시간은 `t = a + b·W` 다 (W = 배치당 창 수). `a` 는 반복마다 치르는 고정 비용
(파이썬 오버헤드·커널 실행·옵티마이저 스텝), `b` 는 창에 비례하는 비용(H2D 복사·GPU 연산).
`a` 가 크면 배치를 키워 **같은 코어로** 천장을 올릴 수 있다. `b` 가 지배하면 소용없다.

이 탐침은 **자료가 필요 없다** — 무작위 텐서로 학습 스텝만 돌려 `a`, `b` 를 가른다.
그래서 로컬에서 잰다. GPU 가 약하면 `b` 가 과대평가되므로 **결론은 보수적**이다
(로컬에서 이득이 보이면 HPC 에서는 더 크다).

⚠ 계산으로 공짜여도 **학습으로는 공짜가 아니다** — 배치를 2배로 하면 epoch 당 기울기 스텝이
  절반이 된다. 짝지은 A/B 로 쓰려면 학습률이나 스텝 수를 맞춰야 한다.

    python -X utf8 src/run_bench_batch.py [--ck results/seq_h38_base.pt] [--chunk 64]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, crf_nll
from src.model.lossbuild import build_loss
from src.model.losses import LossWeights
from src.model.inputs import WIDE_CHANNELS
from src.run_gate_check import load_model

#: ⚠ 채널 수는 **모델에서 끌어온다.** `build_inputs` 의 독스트링 "(B,36,600)/(B,12,W/30)" 은
#: 옛 판이고, v37 계열은 세밀 57(`net.FINE_CHANNELS`) · 광역 47(`inputs.WIDE_CHANNELS`)이다.
FINE_W = 600
H = 15


def make_batch(B, T, K, wide_w, dev, gen, fine_c, wide_c, ddim):
    """학습기가 워커에서 받아 오는 것과 **같은 모양**의 무작위 배치 (CPU, 고정 메모리)."""
    f = lambda *s: torch.randn(*s, generator=gen).pin_memory()
    bt = {
        "fine": f(B, T, fine_c, FINE_W),
        "wide": f(B, T, wide_c, wide_w),
        "dfeat": f(B, T, ddim),      # ⚠ N_FEAT 가 아니라 체크포인트의 ddim 이다
        "y_on": (torch.rand(B, T, K, generator=gen) > 0.5).to(torch.uint8).pin_memory(),
        "y_power": (torch.rand(B, T, K, generator=gen) * 40).pin_memory(),
        "y_plugged": (torch.rand(B, T, K, generator=gen) > 0.5).to(torch.uint8).pin_memory(),
        "y_standby": (torch.rand(B, T, K, generator=gen) * 2).pin_memory(),
        "y_state": (torch.rand(B, T, K, generator=gen) * 3).to(torch.int16).pin_memory(),
        "obs_harm": f(B, T, H, 2),
        "p_noise": (torch.rand(B, T, generator=gen) * 2).pin_memory(),
        "p_observed": (torch.rand(B, T, generator=gen) * 200 + 50).pin_memory(),
        "z_grid": (torch.rand(B, T, 2, generator=gen) + 0.5).pin_memory(),
    }
    return bt


def step(bt, model, heads, crit, opt, dev, K):
    """`run_train_seq` 의 배치 본체 그대로 (vswap·bghead 없음 — 둘 다 선택 항이다)."""
    B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
    fine = bt["fine"].to(dev, non_blocking=True).flatten(0, 1)
    wide = bt["wide"].to(dev, non_blocking=True).flatten(0, 1)
    o = model(fine, wide)
    z = o["z"].reshape(B, T, -1)
    em, on, off, ini = heads(z, bt["dfeat"].to(dev, non_blocking=True),
                             o["on_logit"].reshape(B, T, K))
    crf = crf_nll(em, on, off, bt["y_on"].to(dev).bool(), ini)
    tg = {"y_power": bt["y_power"].to(dev).flatten(0, 1),
          "y_on": bt["y_on"].to(dev).float().flatten(0, 1),
          "y_plugged": bt["y_plugged"].to(dev).float().flatten(0, 1),
          "y_standby": bt["y_standby"].to(dev).flatten(0, 1),
          "y_state": bt["y_state"].to(dev).long().flatten(0, 1),
          "obs_harm": bt["obs_harm"].to(dev).flatten(0, 1),
          "p_noise": bt["p_noise"].to(dev).flatten(0, 1),
          "p_observed": bt["p_observed"].to(dev).flatten(0, 1),
          "harm_offset": None,
          "log_z": torch.log(bt["z_grid"].to(dev)[..., 0].clamp(min=1e-3)).flatten(0, 1)}
    loss = crit(o, tg)["total"] + 1.0 * crf
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="results/seq_h38_base.pt")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--rpb", type=int, nargs="+", default=[2, 3, 6, 12, 24])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ck, map_location=dev, weights_only=False)
    apps = list(ck["appliances"]); K = len(apps)
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.train()
    heads = ChainHeads(ck["zdim"], ck["ddim"], K, hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.train()
    crit = build_loss(apps, dev, weights=LossWeights(harm=0.1, cons=0.0, over=0.0, z=0.0),
                      verbose=False)
    opt = torch.optim.AdamW([{"params": heads.parameters(), "lr": 1e-4},
                             {"params": model.parameters(), "lr": 1e-4}], weight_decay=0.01)
    wide_w = max(2, int(ck["meta"]["window_cycles"]) // 30)   # build_wide 가 30사이클로 요약한다
    print("%s · chunk %d · 기기 %d · 세밀채널 %d · wide %dx%d · 반복 %d (예열 %d)"
          % (torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", a.chunk, K,
             model.fine_channels, WIDE_CHANNELS, wide_w, a.iters, a.warmup))
    print("\n   %-5s %8s %11s %12s %12s" % ("rpb", "창/배치", "초/반복", "창/초", "주루프 CPU"))

    gen = torch.Generator().manual_seed(0)
    rows = []
    for rpb in a.rpb:
        W = rpb * a.chunk
        try:
            bt = make_batch(rpb, a.chunk, K, wide_w, dev, gen, model.fine_channels, WIDE_CHANNELS, ck["ddim"])
            for _ in range(a.warmup):
                step(bt, model, heads, crit, opt, dev, K)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0, c0 = time.perf_counter(), time.process_time()
            for _ in range(a.iters):
                step(bt, model, heads, crit, opt, dev, K)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / a.iters
            dc = (time.process_time() - c0) / a.iters
        except RuntimeError as e:
            msg = str(e)
            kind = "메모리 부족" if "out of memory" in msg.lower() else "오류"
            print("   %-5d %8d   %s — %s" % (rpb, W, kind, msg.split(chr(10))[0][:70]))
            if "out of memory" not in msg.lower():
                raise
            break
        rows.append((W, dt))
        print("   %-5d %8d %10.4fs %12.0f %11.1f%%" % (rpb, W, dt, W / dt, 100 * dc / dt))
        del bt
        if dev == "cuda":
            torch.cuda.empty_cache()

    if len(rows) >= 2:
        Wv = np.array([r[0] for r in rows], float)
        tv = np.array([r[1] for r in rows], float)
        b, aa = np.polyfit(Wv, tv, 1)
        print("\n   맞춤  t = %.1f ms + %.4f ms/창" % (1000 * aa, 1000 * b))
        print("   고정 비용이 배치당 차지하는 몫  %s"
              % "  ".join("W=%d %.0f%%" % (w, 100 * aa / (aa + b * w)) for w in Wv))
        ceil = 1.0 / b if b > 0 else float("inf")
        print("   창/초 천장 (배치 -> 무한대)  %.0f   ·   지금 쓰는 W=384 에서 %.0f"
              % (ceil, 384 / (aa + b * 384)))
        print("\n   읽는 법 — 고정 비용 몫이 크면 배치를 키워 **같은 코어로** 천장이 오른다.")
        print("   HPC 실측은 판당 4,655 창/초 (W=384) 다. 위 천장과 견줘라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
