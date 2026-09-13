# -*- coding: utf-8 -*-
"""희석 가설을 **로컬에서** 가른다 — `w_harm` 을 바꿔 헤드만 재적합 (14.21).

사용자: *"잠깐만 갑자기 뭐하는 거야. 원인도 안 찾고 그냥 돌린다고?"*

**맞다.** 14.16 에 *"다음 판은 설계와 관문을 먼저 만들고 학습 전에 반증할 수 있는 예측을
적는다"* 고 적어 놓고, 14.20 의 **추론**만으로 GPU 짝 비교를 걸려 했다. 게다가 14.20 ③ 에
*"수렴한 한 점에서 잰 값이라 궤적을 말해 주지 않는다"* 는 단서를 내가 직접 달았다.

여기서 **재학습 없이** 가른다. 몸통을 얼리고(파라미터 requires_grad=False) **헤드만**
진짜 손실로 재적합한다 — forward 는 진짜 모델 것이라 물리 프라이어·어텐션까지 그대로다.

  ⓐ w_harm 0.00   14.19 ① 이 이미 쟀다: 지수 0.79~1.14 -> 1.56~1.94
  ⓑ w_harm 0.02
  ⓒ w_harm 0.10   (본선 값)

**반증 가능한 예측**: 희석이 기전이면 지수가 `w_harm` 에 대해 **단조 감소**하고
ⓒ 는 원래 헤드(0.79~1.14) 근처로 끌려 내려간다.
  -> 그렇게 안 되면 **희석 가설이 틀렸고** GPU 짝 비교는 낭비다.

    python -X utf8 src/run_diag_wharm_local.py [--steps 600]
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


def exps(pred, yp, V, apps, apps_m):
    out = {}
    for a_ in RES:
        k, km = apps.index(a_), apps_m.index(a_)
        on = (yp[:, k] > 5) & (pred[:, km] > 5)
        if on.sum() < 150:
            continue
        lw = np.log(yp[on, k]); hist, e = np.histogram(lw, bins=40)
        bi = int(np.argmax(hist)); c = 0.5 * (e[bi] + e[bi + 1])
        m = on & (np.abs(np.log(np.maximum(yp[:, k], 1e-9)) - c) < 0.12)
        if m.sum() < 100:
            continue
        lv = np.log(V[m] / np.median(V[m]))
        et = np.polyfit(lv, np.log(yp[m, k] / np.median(yp[m, k])), 1)[0]
        ep = np.polyfit(lv, np.log(pred[m, km] / np.median(pred[m, km])), 1)[0]
        out[a_] = (int(m.sum()), et, ep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    from src.model.losses import NILMLoss, LossWeights, build_state_scales
    from src.model.net import (harmonic_scales, harmonic_signatures,
                               harmonic_signatures_by_state, noise_signature,
                               standby_signatures)
    from src.run_baseline import S_I
    from src.synthesis.genopts import build_synthesizer, resolve

    d = Path(a.holdout)
    apps = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(a.n, len(X))
    raw = np.asarray(X[:n])
    ti = raw.shape[-1] - 60
    obs = np.stack([raw[:, 0:15, ti], raw[:, 15:30, ti]], axis=-1).astype(np.float32)
    tg = {k: np.asarray(np.load(d / (k + ".npy"), mmap_mode="r"))[:n]
          for k in ("y_power", "y_on", "y_plugged", "y_standby", "y_state",
                    "p_noise", "p_observed")}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fi, wi = build_inputs(raw)
    V = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER
    yp = tg["y_power"]

    _m0, apps_m = load_model(a.ckpt, dev)[:2]
    col = [apps.index(x) for x in apps_m]
    syn = build_synthesizer(resolve("v32"), "processed_data/npz", "train")
    pool = syn.pool
    sig_state = harmonic_signatures_by_state(pool, apps_m)[0]

    T = {}
    for k, v in tg.items():
        arr = np.asarray(v[:, col] if v.ndim == 2 else v)
        T[k] = torch.from_numpy(arr.astype(np.int64 if k == "y_state" else np.float32)).to(dev)
    T["obs_harm"] = torch.from_numpy(obs).to(dev)
    FI = torch.from_numpy(fi).to(dev); WI = torch.from_numpy(wi).to(dev)

    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]
    print("창 %d (학습 %d / 검정 %d) · 몸통 얼림 · 헤드만 %d스텝" % (n, len(tr), len(te), a.steps))

    base = None
    rows = {}
    for wh in (0.00, 0.02, 0.10):
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
        if base is None:
            with torch.no_grad():
                p0 = torch.cat([model(FI[i:i + 256], WI[i:i + 256])["power"].float().cpu()
                                for i in range(0, n, 256)]).numpy()
            base = exps(p0[te], yp[te], V[te], apps, apps_m)

        opt = torch.optim.AdamW(model.heads.parameters(), lr=a.lr)
        trt = torch.from_numpy(tr).to(dev)
        for st in range(a.steps):
            b = trt[torch.randint(0, len(trt), (256,), device=dev)]
            out = model(FI[b], WI[b])
            parts = crit(out, {k: v[b] for k, v in T.items()})
            loss = parts["total"]
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pn = torch.cat([model(FI[i:i + 256], WI[i:i + 256])["power"].float().cpu()
                            for i in range(0, n, 256)]).numpy()
        rows[wh] = exps(pn[te], yp[te], V[te], apps, apps_m)
        print("   w_harm %.2f 끝" % wh)

    print("\n**검정 절반** 지수 — 몸통 얼림 · 헤드만 재적합")
    print("  %-18s %7s %8s %9s %9s %9s %9s"
          % ("기기", "창", "참값 e", "원래", "wh 0.00", "wh 0.02", "wh 0.10"))
    for a_ in RES:
        if a_ not in base:
            continue
        nn_, et, e0 = base[a_]
        cells = [rows[w].get(a_, (0, 0, float("nan")))[2] for w in (0.00, 0.02, 0.10)]
        print("  %-18s %7d %8.2f %9.2f %9.2f %9.2f %9.2f"
              % (a_, nn_, et, e0, cells[0], cells[1], cells[2]))
    print("\n  **예측**: 희석이 기전이면 w_harm 이 오를수록 지수가 **단조 감소**하고")
    print("           0.10 은 '원래' 근처로 끌려 내려간다.")
    print("           그렇게 안 되면 희석 가설이 틀렸고 GPU 짝 비교는 낭비다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
