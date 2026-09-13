# -*- coding: utf-8 -*-
"""사슬 학습 — 방출·전이 머리를 **같이** 배운다 (13.84.24).

얼린 v37 몸통 위에서 `ChainHeads` 를 CRF 손실로 학습하고, 합성 홀드아웃과 **실측 5파일**에서
창별 판정(v37 그대로)과 나란히 채점한다.

초기점이 정확히 v37 이다 — `emit` 가중이 0, `base_scale` 이 1 이라 방출이 v37 게이트 로짓
그대로이고, 전환 벌점이 −4 라 사슬이 거의 상수 열을 낸다. 그래서 **학습 전 사슬 점수는
창별 점수와 거의 같아야 한다** (아래 0 epoch 줄로 확인한다).

    python -X utf8 src/run_train_chain.py --cache cache/seq_v1 --epochs 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, crf_nll, viterbi
from src.model.transition import N_FEAT, feat_at
from src.run_gate_check import forward_file, load_model

FS = 60
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
PRE = 13


def real_sequences(ckpt: str, dev: str, grid_s: float):
    """실측 파일마다 (z, base_logit, dfeat, y, apps) 를 학습과 **같은 격자**로 만든다."""
    model, apps = load_model(ckpt, dev)[:2]
    model.eval()
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    from src.preprocessing import load_nilm_npz
    out = {}
    stride = int(grid_s * FS)
    for stem in FILES:
        d = forward_file(model, stem, dev, stride=stride)
        t = d["targets"] / FS
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        ok = (d["targets"] >= PRE * FS + 1) & (d["targets"] < n - PRE * FS - 1)
        t = t[ok]
        df = np.zeros((len(t), N_FEAT + 2), np.float32)
        for j, tt in enumerate(t):
            f_, dp_ = feat_at(H, P, int(round(tt * FS)))
            df[j, :N_FEAT] = f_
            df[j, -2] = np.sign(dp_)
            df[j, -1] = np.log10(abs(dp_) + 1e-3)
        y = np.zeros((len(t), len(apps)), np.int8)
        for k, a in enumerate(apps):
            for t0, t1 in ev[stem]["intervals"].get(a, {}).get("on", []):
                y[(t >= t0) & (t < t1), k] = 1
        present = np.array([a in ev[stem]["appliances_present"] for a in apps])
        out[stem] = dict(z=d["z"][ok], base=d["on_logit"][ok],
                         dfeat=df, y=y, present=present, t=t)
    return out, apps


def score_real(heads, real, dev, use_chain: bool):
    """기기·파일 평균 시간 정확도. `use_chain=False` 면 방출 문턱 0 (= v37 창별)."""
    accs, mp = [], {}
    with torch.no_grad():
        for stem, d in real.items():
            z = torch.from_numpy(d["z"]).float()[None].to(dev)
            df = torch.from_numpy(d["dfeat"]).float()[None].to(dev)
            bs = torch.from_numpy(d["base"]).float()[None].to(dev)
            em, on, off = heads(z, df, bs)
            p = viterbi(em, on, off) if use_chain else (em > 0)
            p = p[0].cpu().numpy()
            y = d["y"].astype(bool)
            for k in np.nonzero(d["present"])[0]:
                a = (p[:, k] == y[:, k]).mean()
                accs.append(a)
                mp.setdefault(stem, {})[k] = a
    return float(np.mean(accs)), mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seq_v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--score-norm", type=int, default=0, help="전이 점수 국소 정규화 반폭(단계). 0 이면 끔")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="chain_v1")
    a = ap.parse_args()

    c = Path(a.cache)
    meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    K = len(apps)
    Z = np.load(c / "z.npy", mmap_mode="r")
    G = np.load(c / "gate_logit.npy", mmap_mode="r")
    D = np.load(c / "dfeat.npy", mmap_mode="r")
    Y = np.load(c / "y.npy", mmap_mode="r")
    n_rec, T, zdim = Z.shape
    n_ho = max(1, int(n_rec * a.holdout_frac))
    idx = np.random.RandomState(0).permutation(n_rec)
    tr_i, ho_i = idx[n_ho:], idx[:n_ho]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[chain] 기록 %d (학습 %d · 홀드아웃 %d) · 단계 %d · z %d · Δ %d · 기기 %d · %s"
          % (n_rec, len(tr_i), len(ho_i), T, zdim, D.shape[2], K, dev))

    torch.manual_seed(a.seed)
    heads = ChainHeads(zdim, D.shape[2], K, hidden=a.hidden, score_norm=a.score_norm).to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epochs * (len(tr_i) // a.batch)))

    real, apps_r = real_sequences(meta["ckpt"], dev, meta["grid_s"])
    assert apps_r == apps, "기기 열 순서가 캐시와 다르다"

    def batch(ii):
        z = torch.from_numpy(np.asarray(Z[ii])).float().to(dev)
        g = torch.from_numpy(np.asarray(G[ii])).float().to(dev)
        d = torch.from_numpy(np.asarray(D[ii])).float().to(dev)
        y = torch.from_numpy(np.asarray(Y[ii])).bool().to(dev)
        return z, g, d, y

    def eval_syn():
        accs_c, accs_w = [], []
        with torch.no_grad():
            for s in range(0, len(ho_i), a.batch):
                z, g, d, y = batch(np.sort(ho_i[s:s + a.batch]))
                em, on, off = heads(z, d, g)
                accs_c.append(float((viterbi(em, on, off) == y).float().mean()))
                accs_w.append(float(((g > 0) == y).float().mean()))
        return float(np.mean(accs_c)), float(np.mean(accs_w))

    base_r, _ = score_real(heads, real, dev, use_chain=False)
    ch_r, _ = score_real(heads, real, dev, use_chain=True)
    sc, sw = eval_syn()
    print("  epoch  0 (학습 전)  합성 사슬 %.4f / 창별 %.4f · 실측 사슬 %.4f / 창별 %.4f"
          % (sc, sw, ch_r, base_r))

    t0 = time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        heads.train()
        perm = np.random.RandomState(1000 * a.seed + ep).permutation(tr_i)
        tot = 0.0
        nb = 0
        for s in range(0, len(perm) - a.batch + 1, a.batch):
            z, g, d, y = batch(np.sort(perm[s:s + a.batch]))
            em, on, off = heads(z, d, g)
            loss = crf_nll(em, on, off, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        if ep % 5 == 0 or ep == a.epochs:
            heads.eval()
            sc, sw = eval_syn()
            ch_r, mp = score_real(heads, real, dev, use_chain=True)
            hist.append(ch_r)
            print("  epoch %2d  NLL %8.3f · 합성 사슬 %.4f / 창별 %.4f · 실측 사슬 %.4f / 창별 %.4f  [%.0f초]"
                  % (ep, tot / max(nb, 1), sc, sw, ch_r, base_r, time.time() - t0))

    heads.eval()
    ch_r, mp = score_real(heads, real, dev, use_chain=True)
    _, mp0 = score_real(heads, real, dev, use_chain=False)
    if len(hist) >= 3:
        h = np.asarray(hist[len(hist) // 2:])
        print("")
        print("실측 사슬 (학습 후반 평가 %d회): 중앙 %.4f · 최소 %.4f · 최대 %.4f · 창별 %.4f"
              % (len(h), float(np.median(h)), float(h.min()), float(h.max()), base_r))
    km = apps.index("minipc")
    print("\n미니PC 시간 정확도 (실측)")
    print("  파일      창별(v37)   사슬")
    for stem in FILES:
        if km in mp.get(stem, {}):
            print("  %-8s %9.3f %8.3f" % (stem, mp0[stem][km], mp[stem][km]))
    out = Path("results/%s.pt" % a.tag)
    torch.save({"heads": heads.state_dict(), "appliances": apps, "meta": meta,
                "zdim": zdim, "ddim": int(D.shape[2]), "hidden": a.hidden,
                "score_norm": a.score_norm}, out)
    print("\n저장 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
