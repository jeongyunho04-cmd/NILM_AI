# -*- coding: utf-8 -*-
"""학습된 `--gate-free-power` 판이 **정말로** 떨어져 있나 (14.152). 다섯 자리를 본다.

```
  [1] 값      power == power_raw          (곱이 빠졌나)
  [2] 혼합    mix_0 이 0 이 아니고 Σmix = 1  (state 0 이 혼합에 들어왔나)
  [3] 슬롯    p_states[...,0] == 0         (OFF 상태의 전력이 0 인가)
  [4] 경사    ∂Σpower/∂on_logit == 0       (기준선은 0.5~3.3, 분해판은 grad 가 **None**)
  [5] 프라이어 κ 8 과 0 에서 power 비트 동일  (기준선은 0.9~10.1W 움직인다)
```

    python -X utf8 src/run_gate_gfcheck.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa
import torch
from src.model.realdata import dense_targets
from src.run_gate_check import load_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
rw = dense_targets("test_2", stride=30)
f, w, *_ = rw.batch(np.arange(64))
F = torch.from_numpy(np.ascontiguousarray(f)).to(dev)
W = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
for ck in ("results/cnn_pcap_s0.pt", "results/cnn_gfp_s0.pt",
           "results/cnn_hv2_s0.pt", "results/cnn_gfv2_s0.pt"):
    m = load_model(ck, dev)[0]; m.eval()
    gf = bool(getattr(m, "gate_free_power", False))
    o = m(F, W)
    # [1] 값: power == power_raw 인가 (곱이 빠졌나)
    d1 = float((o["power"] - o["power_raw"]).abs().max())
    # [2] 혼합에 state 0 이 들어 있나 (mix 합 1 이고 0번이 0 이 아닌가)
    mx = o["power_mix"].float()
    m0 = float(mx[..., 0].max())
    ms = float(mx.sum(-1).mean())
    # [3] state 0 의 전력 슬롯이 0 인가
    p0 = float(o["power_states"].float()[..., 0].abs().max())
    # [4] 경사: ∂Σpower/∂on_logit 이 0 인가
    o2 = m(F, W)
    ol = o2["on_logit"]
    ol.retain_grad()
    o2["power"].sum().backward()
    g = 0.0 if ol.grad is None else float(ol.grad.abs().max())
    # [5] 프라이어를 껐다 켰을 때 power 가 같은가
    k = float(m.prior_kappa); m.prior_kappa = 0.0
    with torch.no_grad():
        d5 = float((m(F, W)["power"] - o["power"]).abs().max())
    m.prior_kappa = k
    print("%-16s gate_free=%-5s | power−p_raw %8.2e | mix_0 최대 %.4f · Σmix %.4f"
          " | p_states[0] 최대 %.2e | ∂power/∂on 최대 %8.2e | κ 0/8 차 %8.2e"
          % (ck.split("/")[-1].replace(".pt", ""), gf, d1, m0, ms, p0, g, d5))
