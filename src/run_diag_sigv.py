# -*- coding: utf-8 -*-
"""`L_harm` 의 지문을 **전압으로 고치면** 지수가 2 로 가나 (14.22).

14.21 이 측정을 고쳤다 — `y_state` 로 묶으면(전력으로 안 고름) 참값 지수가 **정확히 2.0**
(1.97~2.13) 이고 모델은 **1.0** (0.68~1.09) 이다. 딱 `V` 하나가 빠졌다:
    물리   P = V·I        모델   P = c·I
그 상수의 출처가 `L_harm` 이다 — `obs_harm ≈ Σ power_k·sig_k`, `sig_k = median(I/P)` 는
**상수**라 뒤집으면 `power ∝ I` = **지수 1** 이다.

⚠ 14.20 이 이것을 "반증" 했다고 적었는데 **그 반증이 틀렸다.** 수렴점에서 `L_harm` 의 지수
압력이 +0.012 였던 것은 "무관심" 이 아니라 **이미 이긴 자리라 잔여 압력이 없는 것**이다.
이긴 항의 압력을 최적점에서 재면 항상 0 이다. [[static-landscape-does-not-predict-training]]

**반증 가능한 예측**: `sig` 를 `sig · (V_CENTER/V)` 로 고치면 `L_harm` 이 지수 **2** 를 원하고,
헤드를 재적합하면 지수가 1 -> 2 로 가야 한다. **안 가면 이 설명도 틀렸다.**

    python -X utf8 src/run_diag_sigv.py [--steps 600]
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

WATCH = ("hair_dryer", "electiric_kettle", "hotplate", "oven")


def exps_by_state(pred, yp, yo, ys, V, apps, apps_m):
    """**상태 라벨**로 묶어 잰다 — 전력으로 고르지 않으므로 선택 치우침이 없다."""
    out = {}
    for a_ in WATCH:
        k, km = apps.index(a_), apps_m.index(a_)
        for st in sorted(set(ys[:, k].tolist())):
            if st <= 0:
                continue
            m = (yo[:, k] == 1) & (ys[:, k] == st) & (yp[:, k] > 5) & (pred[:, km] > 5)
            if m.sum() < 150:
                continue
            lv = np.log(V[m] / np.median(V[m]))
            et = np.polyfit(lv, np.log(yp[m, k] / np.median(yp[m, k])), 1)[0]
            ep = np.polyfit(lv, np.log(pred[m, km] / np.median(pred[m, km])), 1)[0]
            out[(a_, st)] = (int(m.sum()), et, ep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    from src.model import losses as L
    from src.model.losses import NILMLoss, LossWeights, build_state_scales
    from src.model.net import (harmonic_scales, harmonic_signatures,
                               harmonic_signatures_by_state, noise_signature,
                               standby_signatures)
    from src.run_baseline import S_I
    from src.synthesis.genopts import build_synthesizer, resolve

    d = Path(a.holdout)
    apps = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    raw = np.asarray(np.load(d / "X.npy", mmap_mode="r"))
    n = len(raw); ti = raw.shape[-1] - 60
    obs = np.stack([raw[:, 0:15, ti], raw[:, 15:30, ti]], axis=-1).astype(np.float32)
    tg = {k: np.asarray(np.load(d / (k + ".npy"), mmap_mode="r"))
          for k in ("y_power", "y_on", "y_plugged", "y_standby", "y_state",
                    "p_noise", "p_observed")}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fi, wi = build_inputs(raw)
    V = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER
    yp, yo, ys = tg["y_power"], tg["y_on"], tg["y_state"]

    _m, apps_m = load_model(a.ckpt, dev)[:2]
    col = [apps.index(x) for x in apps_m]
    pool = build_synthesizer(resolve("v32"), "processed_data/npz", "train").pool
    sig_state = harmonic_signatures_by_state(pool, apps_m)[0]

    T = {}
    for k, v in tg.items():
        arr = np.asarray(v[:, col] if v.ndim == 2 else v)
        T[k] = torch.from_numpy(arr.astype(np.int64 if k == "y_state" else np.float32)).to(dev)
    T["obs_harm"] = torch.from_numpy(obs).to(dev)
    T["vrel"] = torch.from_numpy((V / V_CENTER).astype(np.float32)).to(dev)
    FI = torch.from_numpy(fi).to(dev); WI = torch.from_numpy(wi).to(dev)

    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]
    print("창 %d (학습 %d / 검정 %d) · 몸통 얼림 · 헤드만 %d스텝" % (n, len(tr), len(te), a.steps))

    # ── `sig` 를 전압으로 고치는 덧댐 ────────────────────────────────────
    orig = L.NILMLoss._harm_pred_active

    def patched(self, out, power, site_idx=None):
        base = orig(self, out, power, site_idx)
        vr = getattr(self, "_vrel", None)
        if vr is None:
            return base
        # I/P ∝ 1/V  ->  같은 전력이 만드는 전류가 전압에 반비례한다
        return base / vr[:, None, None]

    base_res = None
    res = {}
    for name, fix, wh in (("전력만 (wh 0)", False, 0.0),
                          ("고침 없음 (wh .1)", False, 0.1),
                          ("**sig 고침 (wh .1)**", True, 0.1)):
        model, _ = load_model(a.ckpt, dev)[:2]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.heads.parameters():
            p.requires_grad_(True)
        crit = NILMLoss(
            s_i=torch.tensor([S_I[x] for x in apps_m], dtype=torch.float32),
            signatures=torch.from_numpy(harmonic_signatures(pool, apps_m)),
            standby_sig=torch.from_numpy(standby_signatures(pool, apps_m)),
            noise_sig=torch.from_numpy(noise_signature(pool)),
            harm_scale=torch.from_numpy(harmonic_scales(pool, apps_m)),
            harm_even_magnitude=True,
            s_state=build_state_scales(apps_m, [S_I[x] for x in apps_m]),
            signatures_state=torch.from_numpy(sig_state),
            weights=LossWeights(harm=wh),
        ).to(dev)
        L.NILMLoss._harm_pred_active = patched if fix else orig

        if base_res is None:
            with torch.no_grad():
                p0 = torch.cat([model(FI[i:i + 256], WI[i:i + 256])["power"].float().cpu()
                                for i in range(0, n, 256)]).numpy()
            base_res = exps_by_state(p0[te], yp[te], yo[te], ys[te], V[te], apps, apps_m)

        opt = torch.optim.AdamW(model.heads.parameters(), lr=a.lr)
        trt = torch.from_numpy(tr).to(dev)
        for st in range(a.steps):
            b = trt[torch.randint(0, len(trt), (256,), device=dev)]
            crit._vrel = T["vrel"][b] if fix else None
            out = model(FI[b], WI[b])
            parts = crit(out, {k: v[b] for k, v in T.items() if k != "vrel"})
            opt.zero_grad(); parts["total"].backward(); opt.step()
        crit._vrel = None
        with torch.no_grad():
            pn = torch.cat([model(FI[i:i + 256], WI[i:i + 256])["power"].float().cpu()
                            for i in range(0, n, 256)]).numpy()
        res[name] = exps_by_state(pn[te], yp[te], yo[te], ys[te], V[te], apps, apps_m)
        print("   %s 끝" % name)
    L.NILMLoss._harm_pred_active = orig

    print("\n**검정 절반** 지수 — 상태 라벨로 묶어 잼 (선택 치우침 없음)")
    print("  %-18s %5s %6s %8s %8s %11s %12s %14s"
          % ("기기", "상태", "창", "참값 e", "원래", "전력만", "고침없음", "**sig 고침**"))
    for key in sorted(base_res):
        a_, st = key
        nn_, et, e0 = base_res[key]
        r0 = res["전력만 (wh 0)"].get(key, (0, 0, float("nan")))[2]
        r1 = res["고침 없음 (wh .1)"].get(key, (0, 0, float("nan")))[2]
        r2 = res["**sig 고침 (wh .1)**"].get(key, (0, 0, float("nan")))[2]
        print("  %-18s %5d %6d %8.2f %8.2f %11.2f %12.2f %14.2f"
              % (a_, st, nn_, et, e0, r0, r1, r2))
    print("\n  **예측**: 마지막 열이 2 에 가까워야 이 설명이 맞다. 안 가면 이것도 틀렸다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
