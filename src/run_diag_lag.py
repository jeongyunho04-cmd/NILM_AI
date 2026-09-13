# -*- coding: utf-8 -*-
"""전이 뒤에 잔차가 얼마나 오래 남나 — **지속 오차** (2026-09-13).

사용자: *"전력이 크게 변화할때마다 잔차가 크게 늘어나고 뒤에도 지속적으로 영향을
준다고. 이전의 모델은 이런증상이 없었어."*

기제 후보: 사슬은 전력을 **직접** 안 건드린다 (`--no-chain` 으로 확인: 신원·잔차가
사슬 유무에 불변). 그런데 **CRF 기울기는 몸통 `z` 로 흘러든다.** CRF 는 시간적으로
매끄러운 상태열을 보상하므로 `z` 가 매끄러워지고, 그 `z` 를 읽는 전력 머리가 전이
뒤에 **늦게 따라간다**. 그러면 사슬이 전력을 안 건드려도 증상이 나온다.

    잰다: |잔차| 를 **마지막 전이로부터 흐른 시간**으로 묶는다.
          전이 크기(|ΔP|)로도 갈라, 큰 변화에서만 그런지 본다.

    python -X utf8 src/run_diag_lag.py
    python -X utf8 src/run_diag_lag.py --ckpt results/cnn_v37.pt results/seq_fix_ctl.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_diag_rollback import predict
from src.run_train_seq import real_windows

#: 전이 뒤 경과시간 구간 (초). 격자가 2초다.
BINS = (0, 2, 4, 8, 16, 32, 64, 10 ** 9)
#: "큰 변화" 문턱 (W). 전이 시각 앞뒤 관측 총전력 차.
BIG_W = 200.0


def edges(y):
    """(T,K) bool -> 전이가 일어난 시각 인덱스 (어느 기기든)."""
    ch = (y[1:] != y[:-1]).any(1)
    return np.flatnonzero(ch) + 1


def since(T, ev):
    """각 시각이 **마지막 전이로부터** 몇 스텝 지났나. 전이 전이면 큰 값."""
    out = np.full(T, 10 ** 9, np.int64)
    if len(ev) == 0:
        return out
    last = -10 ** 9
    j = 0
    for t in range(T):
        while j < len(ev) and ev[j] <= t:
            last = ev[j]; j += 1
        if last > -10 ** 8:
            out[t] = t - last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_v37.pt", "results/seq_fix_ctl.pt",
                             "results/seq_fix_attn.pt"])
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)

    print("전이 뒤 |잔차| — **지속 오차** (격자 %.0f초)" % a.grid_s, flush=True)
    print("=" * 78)
    res = {}
    for path in a.ckpt:
        _, pr = predict(path, cache, dev)
        bins_all = {i: [] for i in range(len(BINS) - 1)}
        bins_big = {i: [] for i in range(len(BINS) - 1)}
        for stem, d in cache.items():
            base = float(d["p_base"])
            if not np.isfinite(base):
                continue
            on, pw = pr[stem]
            y = d["y"].astype(bool)
            r = np.abs(pw.sum(1) + base - d["p_obs"])
            ev = edges(y)
            sc = since(len(r), ev) * a.grid_s          # 초
            # 그 전이가 **큰 변화**였나 — 전이 앞뒤 관측 총전력 차
            big = np.zeros(len(r), bool)
            if len(ev):
                mark = np.zeros(len(r), bool)
                for e in ev:
                    lo = max(0, e - 2); hi = min(len(r) - 1, e + 2)
                    if abs(float(d["p_obs"][hi]) - float(d["p_obs"][lo])) >= BIG_W:
                        mark[e] = True
                # 각 시각의 '마지막 전이' 가 큰 변화였는지 전파
                lastbig = False
                k = 0
                for t in range(len(r)):
                    while k < len(ev) and ev[k] <= t:
                        lastbig = mark[ev[k]]; k += 1
                    big[t] = lastbig
            for i in range(len(BINS) - 1):
                m = (sc >= BINS[i]) & (sc < BINS[i + 1])
                if m.any():
                    bins_all[i] += list(r[m])
                    mb = m & big
                    if mb.any():
                        bins_big[i] += list(r[mb])
        res[path] = (bins_all, bins_big)

    tags = [p.split("/")[-1][:-3][:13] for p in a.ckpt]
    lab = ["%g~%gs" % (BINS[i], BINS[i + 1]) if BINS[i + 1] < 10 ** 8
           else "%g초 이상" % BINS[i] for i in range(len(BINS) - 1)]
    for nm, idx in (("전이 뒤 경과시간별 |잔차| 평균 W — **모든 전이**", 0),
                    ("같은 것 — **큰 변화(|ΔP|>=%.0fW) 뒤에만**" % BIG_W, 1)):
        print()
        print(nm)
        print("  %-12s%s" % ("경과", "".join("%15s" % t for t in tags)))
        for i in range(len(BINS) - 1):
            line = "  %-12s" % lab[i]
            for p in a.ckpt:
                v = res[p][idx][i]
                line += "%14.1fW" % np.mean(v) if len(v) >= 10 else "%15s" % "-"
            print(line)
        line = "  %-12s" % "표본"
        for p in a.ckpt:
            line += "%15d" % sum(len(res[p][idx][i]) for i in range(len(BINS) - 1))
        print(line)

    print()
    print("읽는 법: 0~2초 칸이 크고 뒤가 빠르게 내려가면 **전이 순간의 오차**다.")
    print("         뒤 칸들이 계속 높으면 **지속 오차** — 전이가 남긴 영향이 안 풀린다.")


if __name__ == "__main__":
    main()
