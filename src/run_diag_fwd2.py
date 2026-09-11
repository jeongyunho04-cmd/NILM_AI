# -*- coding: utf-8 -*-
"""남는 전류의 정체 — **동작점**인가 **어느 녹화를 뽑았나**인가 (13.84.35).

13.84.35 에서 참값을 그대로 넣어도 h11~h15 에서 미니PC 전체 신호만큼(1.0~3.1배)이
남고 기록마다 CV 92~96% 로 움직였다. 환경은 그 중 1~7% 만 설명한다.

여기서 그 남는 양을 **고칠 수 있는 것**과 **못 고칠 것**으로 가른다. 사전을 바꿔 가며
참값 와트로 최소제곱하고 (기록을 갈라 채점) 남는 양이 얼마나 주는지 본다:
  ① 지금             기기당 지문 하나        (= `L_harm`)
  ② **전력 구간별**   기기x전력대 지문        <- 다음 실험이 하려는 것. **이득 상한**
  ③ 기록별 재적합     기록마다 지문 다시      <- 세션 보정의 상한 (실행 불가, 참고선)
② 가 ① 과 별로 안 다르면 다음 실험은 접어야 한다. ③ 만 크게 내려가면 남는 것은
**어느 녹화를 뽑았나** 이고 손실로는 못 고친다 ([[validate-forward-model-on-held-out-combos]]).

    python -X utf8 src/run_diag_fwd2.py [--records 1500]
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

ORD = [1, 3, 5, 7, 9, 11, 13, 15]
NB = 3


def design(ypw, idle, bands=None):
    """(M, C) 실수 설계. 열 = 기기(x전력대) + 대기 지시자 + 상수."""
    n, k = ypw.shape
    if bands is None:
        A = [ypw]
    else:
        cols = []
        for j in range(k):
            b = bands[:, j]
            for q in range(NB):
                cols.append(ypw[:, j] * (b == q))
        A = [np.stack(cols, 1)]
    return np.concatenate(A + [idle, np.ones((n, 1), np.float32)], 1)


def fit_eval(A, y, tr, te):
    """실·허 각각 최소제곱. 남는 |잔차| 중앙 (mA)."""
    out = np.zeros(te.sum(), complex)
    for part, get in ((0, np.real), (1, np.imag)):
        w, *_ = np.linalg.lstsq(A[tr], get(y)[tr], rcond=None)
        r = get(y)[te] - A[te] @ w
        out = out + (r if part == 0 else 1j * r)
    return float(np.median(np.abs(out))) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=1500)
    a = ap.parse_args()
    C = Path(a.cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    N = min(a.records, meta["records"])
    obs = np.asarray(np.load(C / "obs_harm.npy", mmap_mode="r")[:N])
    yon = np.asarray(np.load(C / "y_on.npy", mmap_mode="r")[:N]).astype(bool)
    ypl = np.asarray(np.load(C / "y_plugged.npy", mmap_mode="r")[:N]).astype(bool)
    ypw = np.asarray(np.load(C / "y_power.npy", mmap_mode="r")[:N]).astype(np.float64)
    T = yon.shape[1]
    O = obs[..., 0] + 1j * obs[..., 1]
    idle = (ypl & ~yon).astype(np.float64)
    rec = np.repeat(np.arange(N), T)
    Y = O.reshape(-1, O.shape[-1])
    W = ypw.reshape(-1, ypw.shape[-1])
    I = idle.reshape(-1, idle.shape[-1])

    # 기기별 전력대 (참값 와트의 3분위). OFF 는 0 이라 어느 대든 기여가 0 이다.
    bands = np.zeros_like(W, dtype=np.int8)
    for j in range(W.shape[1]):
        v = W[:, j]; m = v > 1
        if m.sum() > 100:
            q = np.quantile(v[m], [1 / 3, 2 / 3])
            bands[:, j] = np.digitize(v, q)
    A1 = design(W, I)
    A2 = design(W, I, bands)
    rng = np.random.RandomState(0)
    rr = np.arange(N); rng.shuffle(rr)
    tr_r = set(rr[:N * 3 // 4].tolist())
    tr = np.array([r in tr_r for r in rec]); te = ~tr
    print("표본 %d · 기록 %d (학습 %d / 채점 %d) · 열 ① %d  ② %d"
          % (len(Y), N, tr.sum(), te.sum(), A1.shape[1], A2.shape[1]))

    print("\n남는 전류 중앙 (mA) — 기록을 갈라 채점")
    print("  %-5s %10s %12s %8s | %12s %8s | %10s"
          % ("차수", "① 지금", "② 전력구간별", "줄임", "③ 기록별", "줄임", "미니PC12W"))
    from src.model.net import harmonic_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sg = harmonic_signatures(pool, apps); del pool
    km = apps.index("minipc")
    for o in ORD:
        j = o - 1
        y = Y[:, j]
        r1 = fit_eval(A1, y, tr, te)
        r2 = fit_eval(A2, y, tr, te)
        # ③ 기록별 재적합 — 그 기록 앞절반으로 맞추고 뒷절반 채점
        res3 = []
        for i in rr[N * 3 // 4:]:
            s0 = i * T
            sl = slice(s0, s0 + T)
            Ai = A1[sl]; yi = y[sl]
            h = T // 2
            keep = np.abs(Ai).sum(0) > 0
            if keep.sum() < 2 or h <= keep.sum() + 2:
                continue
            rr2 = np.zeros(T - h, complex)
            for part, get in ((0, np.real), (1, np.imag)):
                w, *_ = np.linalg.lstsq(Ai[:h][:, keep], get(yi)[:h], rcond=None)
                e = get(yi)[h:] - Ai[h:][:, keep] @ w
                rr2 = rr2 + (e if part == 0 else 1j * e)
            res3.append(np.median(np.abs(rr2)))
        r3 = float(np.median(res3)) * 1000 if res3 else float("nan")
        mp = 1000 * abs(sg[km, j, 0] + 1j * sg[km, j, 1]) * 12.0
        print("  h%-4d %9.1f %11.1f %7.0f%% | %11.1f %7.0f%% | %9.1f"
              % (o, r1, r2, 100 * (r2 - r1) / max(r1, 1e-9),
                 r3, 100 * (r3 - r1) / max(r1, 1e-9), mp))
    print("\n  ② 가 ① 과 비슷하면 전력 의존 지문은 헛수고다.")
    print("  ③ 만 크게 내려가면 남는 것은 **어느 녹화를 뽑았나** 이고 손실로는 못 고친다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
