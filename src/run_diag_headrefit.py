# -*- coding: utf-8 -*-
"""몸통을 얼리고 **전력 헤드만** 다시 맞추면 지수가 복원되나 (14.19).

14.18 이 좁혀 놓은 것: 몸통에 전압(R² 0.970)도 곱항(0.707)도 있고, 얼린 몸통 위 **능선 탐침
하나**가 참값 지수를 복원한다(1.51 대 참값 1.49). 그런데 학습된 헤드는 0.95 다.
남은 후보는 **학습이 그 최적점에 안 간다** 하나뿐이다.

여기서 둘을 가른다 — 몸통을 얼리고 헤드만 경사하강으로 다시 맞춘다:
  · 복원되면   원인은 **학습 일정/초기점**이다 (본 학습에서 일찍 굳었다)
  · 안 되면    **전력 목적함수 자체**가 그 방향을 안 민다 -> 손실 설계 문제다

⚠ 능선 탐침(14.18)과 다른 점: 탐침은 기기·봉우리마다 **따로** 맞춘 자유 선형이고,
  여기는 **진짜 헤드 구조**(`p_raw = Σ softmax(state)·softplus(p_states)`)를 **모든 창에
  하나로** 맞춘다. 그 구조가 제약이면 여기서 드러난다.
⚠ 홀드아웃을 절반 나눠 학습/검정한다. 일반화를 보는 것이 아니라 **도달 가능성**을 본다.

    python -X utf8 src/run_diag_headrefit.py [--ckpt results/cnn_vexp_ctl.pt] [--steps 600]
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
import torch.nn.functional as F

from src.model.inputs import V_CENTER, V_SPAN, build_inputs
from src.run_gate_check import load_model

RES = ("hair_dryer", "electiric_kettle", "hotplate", "oven")


def exponents(pred, yp, V, apps, apps_m, tag):
    out = []
    for a_ in RES:
        k, km = apps.index(a_), apps_m.index(a_)
        on = (yp[:, k] > 5) & (pred[:, km] > 5)
        if on.sum() < 200:
            continue
        lw = np.log(yp[on, k]); hist, e = np.histogram(lw, bins=40)
        c = 0.5 * (e[int(np.argmax(hist))] + e[int(np.argmax(hist)) + 1])
        m = on & (np.abs(np.log(np.maximum(yp[:, k], 1e-9)) - c) < 0.12)
        if m.sum() < 120:
            continue
        lv = np.log(V[m] / np.median(V[m]))
        et = np.polyfit(lv, np.log(yp[m, k] / np.median(yp[m, k])), 1)[0]
        ep = np.polyfit(lv, np.log(pred[m, km] / np.median(pred[m, km])), 1)[0]
        out.append((a_, int(m.sum()), et, ep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()

    d = Path(a.holdout)
    apps = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    raw = np.asarray(np.load(d / "X.npy", mmap_mode="r"))
    yp = np.asarray(np.load(d / "y_power.npy", mmap_mode="r"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps_m = load_model(a.ckpt, dev)[:2]
    model.eval()
    fi, wi = build_inputs(raw)
    V = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER

    # ── 몸통 표현을 한 번만 뽑아 둔다 (얼린다) ──────────────────────────
    fe = []
    h = model.trunk.register_forward_hook(
        lambda m, i, o: fe.append(o.detach().float().cpu()))
    base = []
    with torch.no_grad():
        for i in range(0, len(fi), 256):
            o = model(torch.from_numpy(fi[i:i + 256]).to(dev),
                      torch.from_numpy(wi[i:i + 256]).to(dev))
            base.append(o["power"].float().cpu().numpy())
    h.remove()
    Z = torch.cat(fe)
    base = np.concatenate(base)
    print("체크포인트 %s · 창 %d · 몸통 %d차원 (얼림)" % (a.ckpt, len(Z), Z.shape[1]))

    n = len(Z); idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]

    Y = torch.from_numpy(yp[:, [apps.index(x) for x in apps_m]]).float()
    s_i = model.__dict__.get("s_i", None)
    # 손실 규모 정규화: 기기별 참값 중앙(켜진 창)으로 나눈다 (`s_i` 대용)
    sc = torch.tensor([max(float(np.median(yp[yp[:, apps.index(x)] > 5, apps.index(x)]))
                           if (yp[:, apps.index(x)] > 5).any() else 1.0, 1.0)
                       for x in apps_m]).float()

    print("\n헤드만 다시 맞춘다 — 몸통 얼림 · 전력 Huber(pred/s, y/s) · %d스텝" % a.steps)
    heads = torch.nn.ModuleList([torch.nn.Linear(Z.shape[1], model.heads[0].out_features)
                                 for _ in model.heads]).to(dev)
    for j, hd in enumerate(heads):          # 학습된 헤드에서 출발 (미세조정)
        hd.load_state_dict(model.heads[j].state_dict())
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr)
    mask = model.power_mix_mask.to(dev)
    Ztr, Ytr, sct = Z[tr].to(dev), Y[tr].to(dev), sc.to(dev)
    B = 512
    for st in range(a.steps):
        b = torch.randint(0, len(Ztr), (B,), device=dev)
        z = Ztr[b]
        o = torch.stack([hd(z) for hd in heads], dim=1)           # (B,K,·)
        S = model.heads[0].out_features
        ns = (S - 3) // 2
        p_states = F.softplus(o[..., 0:ns])
        state = o[..., ns:2 * ns]
        mix = state.masked_fill(mask[None] == 0, -1e4).softmax(-1)
        p_raw = (mix * p_states).sum(-1)
        on_logit = o[..., 2 * ns]
        pw = torch.sigmoid(on_logit) * p_raw
        dd = (pw - Ytr[b]) / sct[None]
        loss = torch.where(dd.abs() <= 1.0, 0.5 * dd * dd, dd.abs() - 0.5).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (st + 1) % 200 == 0:
            print("   step %4d  loss %.5f" % (st + 1, float(loss)))

    with torch.no_grad():
        o = torch.stack([hd(Z.to(dev)) for hd in heads], dim=1)
        S = model.heads[0].out_features; ns = (S - 3) // 2
        p_states = F.softplus(o[..., 0:ns])
        mix = o[..., ns:2 * ns].masked_fill(mask[None] == 0, -1e4).softmax(-1)
        new = (torch.sigmoid(o[..., 2 * ns]) * (mix * p_states).sum(-1)).cpu().numpy()

    print("\n**검정 절반**에서 지수 (같은 봉우리 마스크)")
    print("  %-18s %7s %9s %9s %11s" % ("기기", "창", "참값 e", "원래 헤드", "다시맞춘 헤드"))
    b0 = {x[0]: x for x in exponents(base[te], yp[te], V[te], apps, apps_m, "before")}
    b1 = {x[0]: x for x in exponents(new[te], yp[te], V[te], apps, apps_m, "after")}
    for a_ in RES:
        if a_ in b0 and a_ in b1:
            print("  %-18s %7d %9.2f %9.2f %11.2f"
                  % (a_, b0[a_][1], b0[a_][2], b0[a_][3], b1[a_][3]))
    print("\n  복원되면 -> 원인은 **학습 일정/초기점** (본 학습에서 일찍 굳었다)")
    print("  안 되면  -> **전력 목적함수가 그 방향을 안 민다** (손실 설계 문제)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
