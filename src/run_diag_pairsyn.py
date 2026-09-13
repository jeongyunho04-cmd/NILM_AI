# -*- coding: utf-8 -*-
"""짝 점수가 **가짜 전이를 거르는가** — 합성에서 제대로 센다 (13.84.43).

13.84.42 는 실측에서 짝 점수 AUC 0.819 · 켜짐 −ΔP 0.924 를 냈는데 **거짓 전이가 2개뿐**이라
못 닫았다. 13.84.33 과 같은 수를 쓴다 — 실측은 표본이 모자라고 합성은 무제한이다.

합성 기록에 사슬을 돌려 Viterbi 전이를 모으고 참값 상태열로 참/거짓을 갈라 셋의 AUC 를 낸다.
  ① 꺼짐 짝점수   `|Δ_on + Δ_off| / (|Δ_on| + |Δ_off|)`  (0 = 완벽히 반대 = 짝이 맞다)
  ② 켜짐 −ΔP      켜졌다는데 전력이 안 올랐으면 가짜다
  ③ 꺼짐 ΔP       (실측에서는 AUC 0.456 으로 무의미했다 — 대조로 같이 낸다)

⚠ 캐시 읽기·입력 만들기는 **학습기의 `SeqChunks` 를 그대로 쓴다.** 따로 구현하면 규약이
  어긋난다 ([[match-the-scoring-convention-before-comparing]]).
⚠ 합성 Δ 는 생성기가 만든 것이라 실측보다 깨끗할 수 있다. 여기 값은 **상한**이고
  13.84.42 의 실측값과 자릿수를 대조해야 한다.

    python -X utf8 src/run_diag_pairsyn.py --cache /dev/shm/$USER/seqraw_small --ck results/seq_h38_base.pt
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.run_gate_check import load_model
from src.run_train_seq import FS, PRE, SeqChunks, collate

ORD = [1, 3, 5, 7, 9, 11, 13, 15]


def pair_score(d_on, d_off):
    n = np.linalg.norm(d_on) + np.linalg.norm(d_off)
    return float(np.linalg.norm(d_on + d_off) / n) if n > 0 else np.nan


def auc(pos, neg, nmin=3):
    """pos 가 neg 보다 **작을수록** 좋은 점수. 0.5 = 무의미."""
    pos = np.asarray(pos, float); pos = pos[~np.isnan(pos)]
    neg = np.asarray(neg, float); neg = neg[~np.isnan(neg)]
    if len(pos) < nmin or len(neg) < nmin:
        return np.nan, len(pos), len(neg)
    w = (pos[:, None] < neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(w / (len(pos) * len(neg))), len(pos), len(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ck", default="results/seq_h38_base.pt")
    ap.add_argument("--records", type=int, default=300)
    ap.add_argument("--tol-steps", type=int, default=5,
                    help="모델 전이를 참 전이로 볼 허용 격자 수 (격자 2초면 ±10초)")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    apps = meta["appliances"]
    grid_s = float(meta["grid_s"])
    N = int(meta["steps_raw"]) if "steps_raw" in meta else None
    raw_mm = np.load(os.path.join(a.cache, "raw.npy"), mmap_mode="r")
    N = raw_mm.shape[-1]
    grid = np.arange(meta["target_offset"], N - PRE * FS - 1, int(grid_s * FS))
    nrec = min(a.records, raw_mm.shape[0])
    print("캐시 %s · 생성기 %s · 기록 %d/%d · 단계 %d · 격자 %.1f초"
          % (a.cache, meta.get("gen"), nrec, raw_mm.shape[0], len(grid), grid_s))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ck, map_location=dev, weights_only=False)
    assert list(ck["appliances"]) == list(apps), "기기 목록이 다르다"
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()

    # 기록 하나를 통째로 — chunk 를 전체 길이로 준다. 자리 고정으로 재현된다.
    ds = SeqChunks(a.cache, np.arange(nrec), len(grid), meta["target_offset"],
                   meta["window_cycles"], grid, fixed_start=0)

    box = {k: ([], []) for k in ("pair", "on_dp", "off_dp")}
    per_app = {j: {k: ([], []) for k in ("pair", "on_dp")} for j in range(len(apps))}
    nflip = [0, 0]
    win = 8 * FS
    mg = 3 * FS

    with torch.no_grad():
        for r in range(nrec):
            b = collate([ds[r]])
            fine = b["fine"].to(dev).flatten(0, 1)
            wide = b["wide"].to(dev).flatten(0, 1)
            Z, GL = [], []
            for i in range(0, len(fine), 512):
                o = model(fine[i:i + 512], wide[i:i + 512])
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            z = torch.cat(Z)[None]
            gl = torch.cat(GL)[None]
            em, on, off, ini = heads(z, b["dfeat"].to(dev), gl)
            path = viterbi(em, on, off, ini)[0].cpu().numpy()
            y = b["y_on"][0].numpy().astype(bool)

            raw = np.asarray(raw_mm[r])
            H = (raw[0:15] + 1j * raw[15:30]).T[:, [o - 1 for o in ORD]]
            P = raw[30]
            for k in range(len(apps)):
                lo = None
                for i in np.nonzero(np.diff(path[:, k].astype(np.int8)))[0] + 1:
                    c = int(grid[i])
                    if c - mg - win < 0 or c + mg + win > len(P):
                        continue
                    d = H[c + mg:c + mg + win].mean(0) - H[c - mg - win:c - mg].mean(0)
                    dp = float(P[c + mg:c + mg + win].mean() - P[c - mg - win:c - mg].mean())
                    lo_i, hi_i = max(0, i - a.tol_steps), min(len(y), i + a.tol_steps + 1)
                    ok = bool((np.diff(y[lo_i:hi_i, k].astype(np.int8)) != 0).any())
                    nflip[0 if ok else 1] += 1
                    s = 0 if ok else 1
                    if path[i, k]:
                        lo = d
                        box["on_dp"][s].append(-dp)
                        per_app[k]["on_dp"][s].append(-dp)
                    else:
                        box["off_dp"][s].append(dp)
                        if lo is not None:
                            v = pair_score(lo, d)
                            box["pair"][s].append(v)
                            per_app[k]["pair"][s].append(v)
            if (r + 1) % 50 == 0:
                print("  %d/%d 기록 · 전이 %d (거짓 %d)"
                      % (r + 1, nrec, sum(nflip), nflip[1]), flush=True)

    print("\n전이 %d개 (참 %d · 거짓 %d)" % (sum(nflip), nflip[0], nflip[1]))
    print("\n전체 — 점수가 작을수록 '참답다'")
    print("   %-18s %8s %8s %10s %10s %8s" % ("", "참", "거짓", "참 중앙", "거짓 중앙", "AUC"))
    for lbl, key in (("꺼짐 · 짝점수", "pair"), ("켜짐 · −ΔP (W)", "on_dp"),
                     ("꺼짐 · ΔP (W)", "off_dp")):
        g, b_ = box[key]
        v, ng, nb = auc(g, b_)
        print("   %-18s %8d %8d %10.2f %10.2f %8.3f"
              % (lbl, ng, nb, np.median(g) if ng else np.nan,
                 np.median(b_) if nb else np.nan, v))

    print("\n기기별")
    print("   %-18s %7s %7s %9s | %7s %7s %9s"
          % ("기기", "참", "거짓", "짝 AUC", "참", "거짓", "켜짐ΔP AUC"))
    for k, app in enumerate(apps):
        pv, pg, pb = auc(*per_app[k]["pair"])
        dv, dg, db = auc(*per_app[k]["on_dp"])
        print("   %-18s %7d %7d %9.3f | %7d %7d %9.3f" % (app, pg, pb, pv, dg, db, dv))

    print("\n⚠ 합성 Δ 는 생성기가 만든 것이라 실측보다 깨끗할 수 있다 — **상한**으로 읽어라.")
    print("  견줌: 13.84.42 실측 (짝 0.819 · 켜짐 −ΔP 0.924, 거짓 2개뿐이라 못 닫음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
