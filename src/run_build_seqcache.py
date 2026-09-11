# -*- coding: utf-8 -*-
"""사슬 학습용 시퀀스 캐시 — 얼린 몸통 판 (13.84.24).

연속 기록을 합성하고, 결정 격자마다
    z        (몸통 표현, 256d)         <- 방출·전이 머리가 얹힌다
    on_logit (v37 의 창별 게이트)       <- 대조군이자 방출 초기값
    dfeat    (전이 차분 특징, 32d)      <- 13.84.21 에서 미니PC 9/11 을 낸 그 축
    y        (기기별 참 상태)
를 담는다. **원시를 안 담는다** — 얼린 몸통으로 구조만 먼저 가른다(1.4GB 대 32GB).
끝까지 같이 학습할 때는 원시를 담는 판이 따로 필요하다.

    python -X utf8 src/run_build_seqcache.py --out cache/seq_v1 --records 2000
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

from src.model.inputs import VOLT_ORDERS, build_inputs
from src.model.transition import N_FEAT, feat_at
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.sequence import make_record, states, to_raw45
from src.synthesis.synthesizer import LoadSynthesizer

FS = 60
W_CYC = 3600
LOOK = 360
TGT_OFF = W_CYC - 1 - LOOK          # 창 시작에서 라벨 시점까지 = 3239
PRE, POST = 13, 3


dfeat_at = feat_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/seq_v1")
    ap.add_argument("--records", type=int, default=2000)
    ap.add_argument("--record-s", type=float, default=300.0)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--ckpt", default="results/cnn_v37.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps = load_model(a.ckpt, dev)[:2]
    model.eval()
    K = len(apps)
    N = int(a.record_s * FS)
    step = int(a.grid_s * FS)
    grid = np.arange(TGT_OFF, N - PRE * FS - 1, step)
    T = len(grid)
    print("[seqcache] 기록 %d개 x %.0f초 · 격자 %.1f초 -> 단계 %d/기록 · 총 %d단계"
          % (a.records, a.record_s, a.grid_s, T, a.records * T))

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # 몸통 폭은 첫 배치에서 알아낸다
    zdim = None
    mm = {}
    t0 = time.time()
    for i in range(a.records):
        rng = np.random.RandomState(a.seed * 1000003 + i)
        smp = make_record(gen, N, rng)
        X = to_raw45(smp, VOLT_ORDERS)                 # (45, N)
        S = states(smp, apps)                          # (K, N)  듀티 구멍 메움
        H = (X[0:15] + 1j * X[15:30]).T                # (N, 15)
        P = X[30]
        win = np.stack([X[:, c - TGT_OFF:c - TGT_OFF + W_CYC] for c in grid])
        fine, wide = build_inputs(win)
        with torch.no_grad():
            o = model(torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev))
            z = o["z"].float().cpu().numpy()
            gl = o["on_logit"].float().cpu().numpy()
        df = np.zeros((T, N_FEAT + 2), np.float32)
        for j, c in enumerate(grid):
            f_, dp_ = dfeat_at(H, P, int(c))
            df[j, :len(f_)] = f_
            df[j, -2] = np.sign(dp_)
            df[j, -1] = np.log10(abs(dp_) + 1e-3)
        y = S[:, grid].T.astype(np.int8)               # (T, K)
        if zdim is None:
            zdim = z.shape[1]
            shapes = {"z": (a.records, T, zdim), "gate_logit": (a.records, T, K),
                      "dfeat": (a.records, T, df.shape[1]), "y": (a.records, T, K)}
            dt = {"z": np.float32, "gate_logit": np.float32, "dfeat": np.float32, "y": np.int8}
            mm = {k: np.lib.format.open_memmap(out / (k + ".npy"), mode="w+",
                                               dtype=dt[k], shape=s) for k, s in shapes.items()}
            tot = sum(int(np.prod(s)) * np.dtype(dt[k]).itemsize for k, s in shapes.items())
            print("[seqcache] z 차원 %d · 예상 %.2f GB" % (zdim, tot / 1e9))
        mm["z"][i] = z
        mm["gate_logit"][i] = gl
        mm["dfeat"][i] = df
        mm["y"][i] = y
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print("  %d/%d  (%.2f초/기록, 남은 %.0f분)"
                  % (i + 1, a.records, el / (i + 1), (a.records - i - 1) * el / (i + 1) / 60))
    meta = dict(records=a.records, record_s=a.record_s, grid_s=a.grid_s, steps=T,
                appliances=apps, ckpt=a.ckpt, seed=a.seed, zdim=int(zdim),
                build_seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[seqcache] 끝 %.1f분 -> %s" % ((time.time() - t0) / 60, out))
    # 라벨 요약
    Y = np.asarray(mm["y"])
    print("  기기별 ON 몫 · 전이 수")
    for k, ap_ in enumerate(apps):
        tr = int((np.diff(Y[:, :, k].astype(np.int8), axis=1) != 0).sum())
        print("    %-18s %5.1f%%  %6d" % (ap_, 100 * Y[:, :, k].mean(), tr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
