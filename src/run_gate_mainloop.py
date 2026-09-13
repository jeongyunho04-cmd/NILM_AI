# -*- coding: utf-8 -*-
"""주 루프 목 넓히기 관문 — 값이 같은가, 얼마나 빨라지는가 (13.84.63).

측정 (972758/972759, 2026-09-12): 48코어를 받아 **37.4개만** 쓰고 워커 46개가 각각
76%(Ada)·65%(A10) 만 돈다. GPU 는 34~42%. 그런데 두 판의 **주 프로세스가 각각 한 코어의
94%** 다 — 깔때기 목이 `N 워커 -> IPC 큐 -> 주 프로세스 1개` 이고 우리는 목에 걸려 있다.
창끼리 의존이 있어서가 아니라 **소비자가 하나**여서다.

배치당 주 프로세스 CPU 77.6ms 를 뜯으면:
  · H2D 복사 62MB       약 10ms   창 수에 비례. `pin_memory` 가 꺼져 있었다
  · `crf_nll` 순+역전파  약 30ms   **배치 크기와 무관** (B 2->48 에서 30.7->29.8ms)

`crf_nll` 은 T 를 파이썬 루프로 돌며 `(B,K) = 6x9 = 54개` 텐서에 커널을 수백 번 날린다.
계산은 없고 실행 오버헤드만 남는다. 그 크기면 **CPU 가 빠르다.**

고친 것 둘:
  ① `crf_nll(..., on_cpu=True)` — 기본값. `--no-crf-cpu` 로 옛 동작
  ③ `DataLoader(pin_memory=True)` + `.to(dev, non_blocking=True)`
(②"배치 키우기" 는 **안 넣었다** — epoch 당 기울기 스텝이 줄어 학습이 달라진다)

    python -X utf8 src/run_gate_mainloop.py [--rpb 6] [--chunk 64]
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
from src.model.inputs import WIDE_CHANNELS
from src.model.lossbuild import build_loss
from src.model.losses import LossWeights
from src.run_bench_batch import FINE_W, H, make_batch
from src.run_gate_check import load_model

TOL_VAL, TOL_GRAD = 1e-5, 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="results/seq_h38_base.pt")
    ap.add_argument("--rpb", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA 가 없다 — 이 관문은 GPU 에서만 뜻이 있다"); return 1
    dev = "cuda"

    # ── [1] 항등 — CPU 경로가 GPU 경로와 같은 값·기울기를 내나 ────────────────
    print("[1] 항등 — `crf_nll` 의 CPU 경로가 GPU 경로와 같은가")
    bad = 0
    for B, T, K in ((a.rpb, a.chunk, 9), (2, 16, 9), (12, 32, 5)):
        g = torch.Generator(device=dev).manual_seed(0)
        mk = lambda *s: torch.randn(*s, device=dev, generator=g, requires_grad=True)
        em, on, off = mk(B, T, K), mk(B, T, K), mk(B, T, K)
        ini = mk(B, K)
        y = torch.rand(B, T, K, device=dev, generator=g) > 0.5
        v1 = crf_nll(em, on, off, y, ini, on_cpu=False)
        v1.backward(); g1 = [x.grad.clone() for x in (em, on, off, ini)]
        for x in (em, on, off, ini):
            x.grad = None
        v2 = crf_nll(em, on, off, y, ini, on_cpu=True)
        v2.backward(); g2 = [x.grad.clone() for x in (em, on, off, ini)]
        dv = abs(float(v1) - float(v2)) / max(abs(float(v1)), 1e-12)
        dg = max(float((p - q).abs().max() / q.abs().max().clamp(min=1e-12))
                 for p, q in zip(g1, g2))
        ok = (dv < TOL_VAL) and (dg < TOL_GRAD) and (v2.device.type == dev)
        bad += (not ok)
        print("   B=%-3d T=%-4d K=%-2d  값 상대차 %.2e · 기울기 상대차 %.2e · 반환장치 %s  %s"
              % (B, T, K, dv, dg, v2.device.type, "**같다**" if ok else "!! 다르다 !!"))

    # ── [2] 속도 — 학습 스텝 전체 ────────────────────────────────────────────
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
    wide_w = max(2, int(ck["meta"]["window_cycles"]) // 30)
    gen = torch.Generator().manual_seed(0)

    def step(bt, crf_cpu, nb):
        B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
        fine = bt["fine"].to(dev, non_blocking=nb).flatten(0, 1)
        wide = bt["wide"].to(dev, non_blocking=nb).flatten(0, 1)
        o = model(fine, wide)
        z = o["z"].reshape(B, T, -1)
        em, on, off, ini = heads(z, bt["dfeat"].to(dev, non_blocking=nb),
                                 o["on_logit"].reshape(B, T, K))
        crf = crf_nll(em, on, off,
                      bt["y_on"].bool() if crf_cpu else bt["y_on"].to(dev).bool(),
                      ini, on_cpu=crf_cpu)
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

    W = a.rpb * a.chunk
    print("\n[2] 속도 — 학습 스텝 전체 (창 %d/배치 · %s)" % (W, torch.cuda.get_device_name(0)))
    print("   %-28s %11s %11s %11s" % ("설정", "초/반복", "창/초", "주루프 CPU"))
    res = {}
    for lbl, crf_cpu, pin in (("옛: CRF GPU · 고정X", False, False),
                              ("① CRF CPU 만", True, False),
                              ("③ 고정메모리+비차단 만", False, True),
                              ("**①+③ (새 기본값)**", True, True)):
        bt = make_batch(a.rpb, a.chunk, K, wide_w, dev, gen, model.fine_channels,
                        WIDE_CHANNELS, ck["ddim"])
        if not pin:                      # 고정 메모리를 **끈** 판을 만든다
            bt = {k: v.clone() for k, v in bt.items()}
        for _ in range(4):
            step(bt, crf_cpu, pin)
        torch.cuda.synchronize()
        t0, c0 = time.perf_counter(), time.process_time()
        for _ in range(a.iters):
            step(bt, crf_cpu, pin)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.iters
        dc = (time.process_time() - c0) / a.iters
        res[lbl] = dt
        print("   %-28s %9.2fms %11.0f %10.1f%%" % (lbl, 1000 * dt, W / dt, 100 * dc / dt))
        del bt; torch.cuda.empty_cache()

    base = res["옛: CRF GPU · 고정X"]
    for lbl, v in res.items():
        if lbl != "옛: CRF GPU · 고정X":
            print("   %-28s 처리량 %+.1f%%" % (lbl, 100 * (base / v - 1)))

    print("\n   ⚠ 이 GPU(로컬)는 HPC 보다 훨씬 느리다 — 창당 GPU 비용이 부풀어 **이득이")
    print("   과소평가**된다. HPC 는 GPU 34~42% 라 주 루프 몫이 더 크다.")
    if bad:
        print("\n[판정] !! 항등이 깨졌다 — 적용하면 안 된다"); return 1
    print("\n[판정] **합격** — 값·기울기가 같고 반환 장치도 유지된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
