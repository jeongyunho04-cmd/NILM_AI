# -*- coding: utf-8 -*-
"""손실 항마다 **전압 지수를 어느 쪽으로 미는가** (14.20).

14.19 가 세운 가설: `L_harm` 의 지문 `sig_k` 가 전압을 모르는 **상수**라 `power ∝ I`
(지수 1)를 밀고, 전력 손실은 지수 2 를 밀어 0.65~1.0 의 타협이 된다.

⚠ **노름만 재면 안 된다** ([[loss-share-is-not-gradient-share]] — `L_harm` 은 값이 80% 인데
게이트 기울기는 1% 였다). **부호**가 요점이다. 항마다:

    g = ∂L_항/∂p̂                       (경사하강은 p̂ 를 −g 쪽으로 민다)
    **지수 압력** E = −Σ_w,k g·p̂·(log V_w − 평균)  / Σ|g·p̂|·std(log V)

  `·p̂` 를 곱해 `∂/∂log p̂` 단위로 바꾼다. **E > 0 이면 그 항이 지수를 올리려 한다.**
  저항 기기의 켜진 창만 본다.

    python -X utf8 src/run_diag_gradexp.py [--ckpt results/cnn_vexp_ctl.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import V_CENTER, V_SPAN, build_inputs
from src.run_gate_check import load_model

RES = ("hair_dryer", "electiric_kettle", "hotplate", "oven")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()

    from src.model.losses import NILMLoss, LossWeights, build_state_scales
    from src.model.net import (harmonic_scales, harmonic_signatures,
                               noise_signature, standby_signatures)
    from src.run_baseline import S_I
    from src.synthesis.genopts import build_synthesizer, resolve

    d = Path(a.holdout)
    apps = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(a.n, len(X))
    raw = np.asarray(X[:n])
    ti = raw.shape[-1] - 60                       # 타깃 = 끝 −1초
    obs = np.stack([raw[:, 0:15, ti], raw[:, 15:30, ti]], axis=-1).astype(np.float32)

    tg = {}
    for k, f in (("y_power", "y_power"), ("y_on", "y_on"), ("y_plugged", "y_plugged"),
                 ("y_standby", "y_standby"), ("y_state", "y_state"),
                 ("p_noise", "p_noise"), ("p_observed", "p_observed")):
        tg[k] = np.asarray(np.load(d / (f + ".npy"), mmap_mode="r"))[:n]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps_m = load_model(a.ckpt, dev)[:2]
    model.eval()
    syn = build_synthesizer(resolve("v32"), "processed_data/npz", "train")
    pool = syn.pool
    sig = harmonic_signatures(pool, apps_m)
    try:
        from src.model.net import harmonic_signatures_by_state
        sig_state = harmonic_signatures_by_state(pool, apps_m)[0]
    except Exception as e:
        print("  (상태별 지문을 못 만들었다: %s)" % e); sig_state = None
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps_m], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(standby_signatures(pool, apps_m)),
        noise_sig=torch.from_numpy(noise_signature(pool)),
        harm_scale=torch.from_numpy(harmonic_scales(pool, apps_m)),
        harm_even_magnitude=True,
        # ⚠ 학습과 **같게** — s_state 를 안 주면 전력 손실이 과소평가된다
        s_state=build_state_scales(apps_m, [S_I[x] for x in apps_m]),
        signatures_state=torch.from_numpy(sig_state) if sig_state is not None else None,
        weights=LossWeights(harm=0.1),
    ).to(dev)

    fi, wi = build_inputs(raw)
    V = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER
    lv = np.log(V / np.median(V))

    col = [apps.index(x) for x in apps_m]
    T = {}
    for k, v in tg.items():
        arr = np.asarray(v[:, col] if v.ndim == 2 else v)
        arr = arr.astype(np.int64 if k == "y_state" else np.float32)
        T[k] = torch.from_numpy(arr).to(dev)
    T["obs_harm"] = torch.from_numpy(obs).to(dev)

    out = model(torch.from_numpy(fi).to(dev), torch.from_numpy(wi).to(dev))
    out = {k: (v.detach().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point()
               else v) for k, v in out.items()}
    p_hat = out["power"]

    res_k = [apps_m.index(x) for x in RES if x in apps_m]
    onm = (T["y_on"][:, res_k] > 0.5) & (T["y_power"][:, res_k] > 5)
    lvt = torch.from_numpy(lv).float().to(dev)[:, None]

    print("체크포인트 %s · 창 %d · 저항 켜진 칸 %d" % (a.ckpt, n, int(onm.sum())))
    print("\n**항별 지수 압력** — E>0 이면 그 항이 전압 지수를 **올리려** 한다")
    print("  %-12s %14s %14s %10s" % ("항", "‖∂L/∂log p̂‖", "지수 압력 E", "부호"))

    parts = crit(out, T)
    rows = []
    for name in ("power", "harm", "cons", "state_power"):
        if name not in parts or not torch.is_tensor(parts[name]) or not parts[name].requires_grad:
            continue
        g = torch.autograd.grad(parts[name], p_hat, retain_graph=True, allow_unused=True)[0]
        if g is None:
            continue
        gl = (g * p_hat.detach())[:, res_k]                    # ∂L/∂log p̂
        num = -(gl * lvt).masked_fill(~onm, 0.0).sum()
        den = gl.abs().masked_fill(~onm, 0.0).sum() * float(lv.std()) + 1e-30
        E = float(num / den)
        rows.append((name, float(gl.masked_fill(~onm, 0.0).norm()), E))
        print("  %-12s %14.5f %14.3f %10s"
              % (name, rows[-1][1], E, "**올림**" if E > 0.02 else ("**내림**" if E < -0.02 else "중립")))

    print("\n  가중치를 곱한 실제 기여 (학습 설정 power 1.0 / harm 0.1)")
    w = {"power": 1.0, "harm": 0.1, "cons": 0.0, "state_power": 0.0}
    for nm, gn, E in rows:
        print("   %-12s 가중 노름 %10.5f · 가중 압력 %+9.4f" % (nm, w.get(nm, 0) * gn, w.get(nm, 0) * gn * E))
    print("\n  두 항의 가중 압력 부호가 반대면 **14.19 가설이 맞다**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
