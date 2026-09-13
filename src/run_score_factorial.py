# -*- coding: utf-8 -*-
"""2x2 요인설계를 **짝지어** 채점한다 — 주효과와 상호작용까지 (13.84.61).

972758/972759 는 칸 넷이다: 생성기(**실측 표류** `v36dp` / 손 맞춤 `v36p`) x 가림(유지 / 해제).
네 숫자를 눈으로 견주면 주효과와 상호작용을 틀리기 쉬워서 여기서 계산한다.

⚠ **짝지어라.** 13.84.62 가 가르친 것: 머릿수(평균) 하나로 판정하면 과적합을 못 본다.
   `--keep-last 4` 가 남긴 `_ep9~12` 를 다 채점해 **epoch 산포**를 같이 낸다. 어느 칸이
   이기더라도 산포 안이면 이긴 게 아니다.

⚠ 채점 규약은 `run_score_seq.score` 를 **그대로** 쓴다 — 그 파일에 있는 기기만 세고 기기별
   정확도를 평균한다. 다른 규약을 쓰면 0.929 가 0.961 로 보인다
   ([[match-the-scoring-convention-before-comparing]]).

    python -X utf8 src/run_score_factorial.py                      # 최종 + ep9~12
    python -X utf8 src/run_score_factorial.py --epochs ""          # 최종만
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_score_seq import FILES, score

GEN = ("v36dp", "v36p")          # 실측 표류 / 손 맞춤
MASK = ("mask", "open")          # 가림 유지 / 해제
REF = ("results/seq_h38_base.pt", 0.9493, 0.8482)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--epochs", default="9 10 11 12",
                    help="짝지을 epoch 들. 빈 문자열이면 최종 체크포인트만")
    ap.add_argument("--ref", default=REF[0])
    ap.add_argument("--gen", nargs=2, default=list(GEN), metavar=("A", "B"),
                    help="첫째 인자의 두 수준 (파일명 `seq_<gen>_<mask>.pt`)")
    ap.add_argument("--mask", nargs=2, default=list(MASK), metavar=("A", "B"),
                    help="둘째 인자의 두 수준")
    a = ap.parse_args()
    gens, masks = tuple(a.gen), tuple(a.mask)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = [""] + ["_ep" + e for e in a.epochs.split()] if a.epochs else [""]

    cache, res = {}, {}
    missing = []
    for g, m, e in itertools.product(gens, masks, eps):
        p = os.path.join(a.dir, "seq_%s_%s%s.pt" % (g, m, e))
        if not os.path.exists(p):
            missing.append(os.path.basename(p)); continue
        c, w, rows = score(p, dev, cache)
        res[(g, m, e)] = (c, w, rows)
        print("  %-34s 사슬 %.4f · 창별 %.4f" % (os.path.basename(p), c, w), flush=True)
    if missing:
        print("\n  ⚠ 없는 체크포인트 %d개: %s" % (len(missing), " ".join(missing[:8])))
    if not res:
        print("채점할 것이 없다 — `scp gate1_External:/home1/zetin348/NILM_AI/results/seq_v36*.pt results/`")
        return 1

    rc = rw = None
    if os.path.exists(a.ref):
        rc, rw, _ = score(a.ref, dev, cache)
        print("\n  견줌 %-28s 사슬 %.4f · 창별 %.4f" % (os.path.basename(a.ref), rc, rw))

    def cell(g, m, i):
        v = [res[k][i] for k in res if k[0] == g and k[1] == m]
        return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)

    for i, lbl in ((0, "사슬"), (1, "창별")):
        print("\n=== %s 정확도 — 칸 평균 ± epoch 표준편차 (n) ===" % lbl)
        print("   %-16s %22s %22s" % ("", masks[0], masks[1]))
        for g in gens:
            row = "   %-16s" % ("%s (%s)" % (g, {"v36dp": "실측 표류",
                                                          "v36p": "손 맞춤"}.get(g, "")))
            for m in masks:
                mu, sd, n = cell(g, m, i)
                row += "   %9.4f ±%.4f(%d)" % (mu, sd, n)
            print(row)
        gm = {(g, m): cell(g, m, i)[0] for g in gens for m in masks}
        if not any(np.isnan(list(gm.values()))):
            eg = np.mean([gm[(gens[0], m)] for m in masks]) - np.mean([gm[(gens[1], m)] for m in masks])
            em = np.mean([gm[(g, masks[0])] for g in gens]) - np.mean([gm[(g, masks[1])] for g in gens])
            ix = (gm[(gens[0], masks[0])] - gm[(gens[0], masks[1])]) \
                - (gm[(gens[1], masks[0])] - gm[(gens[1], masks[1])])
            sd = np.nanmean([cell(g, m, i)[1] for g in gens for m in masks])
            print("   주효과 %-6s (%s − %s)  %+.4f" % ("첫째", gens[0], gens[1], eg))
            print("   주효과 %-6s (%s − %s)  %+.4f" % ("둘째", masks[0], masks[1], em))
            print("   상호작용                           %+.4f" % ix)
            print("   ⚠ 견줌: 칸 안 epoch 표준편차 평균 %.4f — 효과가 이보다 작으면 **없는 것**이다" % sd)
        r = rc if i == 0 else rw
        if r is not None:
            best = max(gm, key=gm.get)
            print("   최고 칸 %s/%s %.4f  ·  기준선 %.4f  ·  차 %+.4f"
                  % (best[0], best[1], gm[best], r, gm[best] - r))

    # ── 파일별 미니PC — 머릿수가 가리는 것을 본다 (13.84.62) ──────────────────
    print("\n=== 미니PC 시간 정확도, 파일별 (사슬) — 칸마다 epoch 중앙 ===")
    print("   %-8s %s" % ("파일", "".join("  %-14s" % ("%s/%s" % (g, m))
                                          for g in gens for m in masks)))
    for stem in FILES:
        line = "   %-8s" % stem
        ok = False
        for g in gens:
            for m in masks:
                v = [res[k][2][stem][1] for k in res
                     if k[0] == g and k[1] == m and stem in res[k][2]]
                line += "  %14.3f" % np.median(v) if v else "  %14s" % "—"
                ok |= bool(v)
        if ok:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
