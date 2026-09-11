# -*- coding: utf-8 -*-
"""세션 이동의 **정체** — 회로모델이 아니면 무엇인가 (13.84.35).

사용자: *"회로모델로 SMPS 가 환경에 따라 어떻게 바뀌는지 대부분 시뮬레이션 했는데
어디서 이 정도의 세션 이동이 생기는지 모르겠다."*

논리: 합성에서 회로모델은 `(R, X, V)` 의 **결정적 함수**다. 이동이 회로모델 탓이라면
`R,X,V` 를 특징에 주면 되돌려야 하는데 13.84.33 에서 0.648 -> 0.646 이었다.
그러니 이동은 회로모델이 **아니다**. 기록이 뽑는 나머지를 가른다.

이 판이 가르는 셋:
  ⓐ **수준 이동인가 기울기 이동인가** — 기록마다 절편 하나만 다시 맞춰 본다(1자유도).
     절편만으로 대부분 회복되면 이동은 **오프셋**이고, 오프셋을 예측하면 끝난다.
  ⓑ **동작점으로 설명되는가** — 13.84.32 가 와트당 고차가 47~109% 변한다고 했다.
     기본특징 x 전력 교차항(맹목: log|I1| · 신탁: 형제 참값 W)을 건다.
     13.84.33 의 교차항 시험은 기본특징 x **환경**만 걸었지 x **전력**은 안 걸었다.
  ⓒ **기록의 경계 어긋남이 무엇을 따라가나** — 기록별 절편을 환경·전력에 회귀한다.

    python -X utf8 src/run_diag_shift3.py [--records 1200]
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

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

SIB = ("laptop_charger", "beam_projector")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=1200)
    a = ap.parse_args()

    C = Path(a.cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    km = apps.index("minipc")
    sib = [apps.index(x) for x in SIB]
    raw = np.load(C / "raw.npy", mmap_mode="r")
    yon = np.load(C / "y_on.npy", mmap_mode="r")
    ypw = np.load(C / "y_power.npy", mmap_mode="r")
    zg = np.load(C / "z_grid.npy", mmap_mode="r")
    grid = np.arange(meta["target_offset"], int(meta["record_s"] * 60) - 13 * 60 - 1,
                     int(meta["grid_s"] * 60))

    BASE, ENV, SW, Y, E = [], [], [], [], []
    for i in range(min(a.records, len(raw))):
        yo = np.asarray(yon[i])
        m = yo[:, sib].sum(1) > 0
        if m.sum() < 6:
            continue
        r = np.asarray(raw[i]); z = np.asarray(zg[i])[0]; pw = np.asarray(ypw[i])
        for t in np.nonzero(m)[0][::6]:
            c = grid[t]; seg = slice(max(0, c - 600), c)
            h = r[0:15, seg] + 1j * r[15:30, seg]
            v = np.median(h.real, 1) + 1j * np.median(h.imag, 1)
            i1 = abs(v[0]) + 1e-9
            BASE.append([np.log(i1), np.angle(v[0]), abs(v[2]) / i1,
                         abs(v[4]) / i1, abs(v[6]) / i1, np.angle(v[2] * np.conj(v[0]))])
            vh = np.median(np.abs(r[33:45, seg]), 1)[:6]
            ENV.append([z[0], z[1], np.median(r[32, seg])] + list(vh))
            SW.append(float(pw[t, sib].sum()))            # 형제 참값 와트 (신탁)
            Y.append(int(yo[t, km])); E.append(i)
    BASE = np.asarray(BASE); ENV = np.asarray(ENV); SW = np.asarray(SW)
    Y = np.asarray(Y); E = np.asarray(E)
    nz = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
    Bn, En = nz(BASE), nz(ENV)
    SWn = nz(SW[:, None])
    print("표본 %d (미니PC ON %d / OFF %d) · 기록 %d개 · 형제 W 중앙 %.0f (p10~p90 %.0f~%.0f)"
          % (len(Y), Y.sum(), (1 - Y).sum(), len(set(E)),
             np.median(SW), np.percentile(SW, 10), np.percentile(SW, 90)))

    recs = np.array(sorted(set(E))); np.random.RandomState(0).shuffle(recs)
    tr_r = set(recs[:len(recs) * 3 // 4])
    tr = np.array([e in tr_r for e in E]); te = ~tr

    def fit(Xs, inter_with=None):
        X = np.concatenate(Xs, 1)
        if inter_with is not None:
            X = np.concatenate([X] + [Bn * inter_with[:, [j]]
                                      for j in range(inter_with.shape[1])], 1)
        lr = LogisticRegression(max_iter=6000, C=0.5).fit(X[tr], Y[tr])
        d = X @ np.r_[lr.coef_[0]] + lr.intercept_[0]
        return roc_auc_score(Y[te], d[te]), d

    print("\nⓑ 경계를 **무엇에 따라 기울이면** 회복되나 (환경 밖 채점)")
    auc0, d0 = fit([Bn])
    print("   %-46s AUC %.3f" % ("① 기본 특징만 (지금)", auc0))
    for nm, w in (("② x 환경 (R,X,V,vh)  — 13.84.33 재현", En),
                  ("③ x **전력** log|I1|  (맹목, 쓸 수 있음)", Bn[:, [0]]),
                  ("④ x **형제 참값 W**   (신탁, 상한)", SWn),
                  ("⑤ x 전력 + 환경 둘 다", np.concatenate([Bn[:, [0]], En], 1))):
        print("   %-46s AUC %.3f" % (nm, fit([Bn], w)[0]))

    print("\nⓐ 이동이 **수준**인가 **기울기**인가 (채점 기록만, 앞절반 학습 뒷절반 채점)")
    lo, hi, base_in = [], [], []
    for e in sorted(set(E[te])):
        m = np.nonzero(E == e)[0]
        if len(m) < 12 or len(set(Y[m])) < 2:
            continue
        h = len(m) // 2
        p, q = m[:h], m[h:]
        if len(set(Y[p])) < 2 or len(set(Y[q])) < 2:
            continue
        base_in.append(roc_auc_score(Y[q], d0[q]))
        # 절편만: 전역 판별값 d0 에 그 기록 앞절반이 정하는 상수를 더한다
        # -> 순위는 안 변하므로 AUC 로는 못 본다. 대신 **전역 문턱** 기준 정확도로 잰다.
        thr = np.median(d0[tr])
        off = -np.median(d0[p])                       # 그 기록의 수준을 전역 중앙에 맞춘다
        lo.append(((d0[q] + off + thr > thr) == (Y[q] > 0)).mean())
        hi.append(((d0[q] > thr) == (Y[q] > 0)).mean())
    print("   기록 %d개 · 전역 경계 그대로 기록 안 AUC   %.3f" % (len(base_in), np.median(base_in)))
    print("   전역 문턱 정확도  보정 없음 %.3f  ->  **기록별 절편만 맞춤 %.3f**"
          % (np.median(hi), np.median(lo)))

    print("\nⓒ 기록의 경계 어긋남은 **무엇을 따라가나** (기록별 판별값 중앙 = 그 기록의 수준)")
    rid, lvl, feats = [], [], []
    for e in sorted(set(E)):
        m = np.nonzero(E == e)[0]
        if len(m) < 8:
            continue
        rid.append(e); lvl.append(np.median(d0[m]))
        feats.append([np.median(SW[m]), np.median(BASE[m, 0])] + list(np.median(ENV[m], 0)))
    lvl = np.asarray(lvl); F = np.asarray(feats)
    nm = ["형제 참값 W", "log|I1|", "R", "X", "V실효", "vh1", "vh3", "vh5", "vh7", "vh9", "vh11"]
    print("   기록 %d개 · 수준 산포 sd %.3f" % (len(lvl), lvl.std()))
    print("   %-14s %8s %8s" % ("설명변수", "단독 r", "단독 R²"))
    for j, x in enumerate(nm):
        r_ = np.corrcoef(F[:, j], lvl)[0, 1]
        print("   %-14s %+8.3f %8.3f" % (x, r_, r_ ** 2))
    for lbl, cols in (("환경만 (R,X,V,vh)", list(range(2, 11))),
                      ("전력만 (형제W, log|I1|)", [0, 1]),
                      ("둘 다", list(range(11)))):
        g = LinearRegression().fit(nz(F[:, cols]), lvl)
        print("   %-22s 합동 R² %.3f" % (lbl, g.score(nz(F[:, cols]), lvl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
