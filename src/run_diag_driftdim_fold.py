# -*- coding: utf-8 -*-
"""13.84.52 ③ 의 LORO 이득 1.20배가 **0 과 구분되나** — 녹화별로 갈라 본다.

평균 하나는 줄 하나가 끌고 갈 수 있다 ([[validate-both-directions]], [[one-seed-cannot-rank-small-classes]]).

    python -X utf8 src/run_diag_driftdim_fold.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim_audit import P16, build, minipc_sig


def main():
    D, tags, sizes = build(10 * 60)
    m_sig = minipc_sig()
    s0 = np.linalg.norm(m_sig)
    d0 = np.sqrt((D ** 2).sum(1).mean())
    stems = np.unique(tags)

    print("LORO 녹화별 SNR 배 (10초 블록 · **통합/통합** 자) — 표본 %d · 녹화 %d"
          % (len(D), len(stems)))
    print("  %-26s %6s %9s %9s %9s" % ("빼 둔 녹화", "블록", "k=1", "k=2", "k=3"))
    rows = {k: [] for k in (1, 2, 3)}
    for held in stems:
        tr, te = D[tags != held], D[tags == held]
        line = "  %-26s %6d" % (held[:26], len(te))
        for k in (1, 2, 3):
            _, _, V2 = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
            Pm = np.eye(P16) - V2[:k].T @ V2[:k]
            s_ = np.linalg.norm(Pm @ m_sig)
            d_ = np.sqrt(((te @ Pm.T) ** 2).sum(1).mean())
            r = (s_ / d_) / (s0 / d0)
            rows[k].append((r, len(te)))
            line += " %8.2f배" % r
        print(line)
    print()
    for k in (1, 2, 3):
        r = np.array([x[0] for x in rows[k]])
        w = np.array([x[1] for x in rows[k]], float)
        print("  k=%d  단순평균 %.2f배 · **블록가중 %.2f배** · 범위 %.2f~%.2f · 1.0 미만 %d/%d"
              % (k, r.mean(), float((r * w).sum() / w.sum()), r.min(), r.max(),
                 int((r < 1.0).sum()), len(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
