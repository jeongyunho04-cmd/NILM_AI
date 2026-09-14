# -*- coding: utf-8 -*-
"""생성기가 표류를 **이미 만들고 있나** — 처방 전에, 짝을 맞춰 잰다 (14.12).

생성기는 이미 회로로 전압 응답을 넣는다 (13.2 텍스처 델타):

    I(생성) = I(녹화 재생) + [I_sim(합성 텍스처) − I_sim(녹화 텍스처)] + [결합 델타]

창마다 텍스처 하나를 뽑으므로 **창 사이에는 전압이 바뀐다**. 그러면 표류가 이미 생긴다.
`smps_drift` 를 더하기 전에 **얼마나 생기는지** 재야 한다
([[check-the-generator-can-make-the-failing-window]]).

⚠⚠ **자를 맞춘다** ([[match-the-scoring-convention-before-comparing]]).
  라이브러리 전체(24파일·vrms 207~231V)의 산포를 실측 **한 녹화 안** 표류와 견주면 안 된다.
  둘을 따로 낸다:
    [안]   같은 파일의 텍스처끼리   <->  실측 같은 녹화의 10초 블록끼리
    [사이] 파일 평균끼리            <->  실측 녹화 평균끼리

    python -X utf8 src/run_diag_driftsyn.py
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, blocks, load, vec
from src.synthesis.coupling import SmpsCircuit
from src.synthesis.vtexture import default_library

APP = "laptop_charger"
BANDS = [(19, 28), (28, 40), (40, 55), (55, 70)]
POWERS = (23.0, 33.0, 47.0, 63.0)
V1 = 222.0
BLOCK = 600                       # 10초


def real_drift():
    """실측 — [안] 녹화 안 블록 편차 · [사이] 녹화 평균 편차. 같은 전력대 구조."""
    within, between = [], []
    for lo, hi in BANDS:
        means = []
        for stem, P, hc, on, off in load(APP):
            B = blocks(P, hc, on, lo, hi, B=BLOCK)
            if len(B) < 2:
                continue
            within.append(B - B.mean(0))
            means.append(B.mean(0))
        if len(means) >= 2:
            M = np.asarray(means)
            between.append(M - M.mean(0))
    return np.vstack(within), np.vstack(between)


def syn_drift(lib, circ):
    """합성 — 텍스처 델타. [안] 같은 파일 텍스처끼리 · [사이] 파일 평균끼리."""
    stems = lib.stems()
    ref = next((s for s in stems if APP in s), stems[0])
    rel_rec = lib.file_rel_full(ref)
    r_id = lib.file_id(ref)
    within, between = [], []
    for P in POWERS:
        by_file = defaultdict(list)
        for t in lib.textures:
            d = circ.texture_delta(APP, float(P), t.source_rel_full(), t.id,
                                   rel_rec, r_id, V1, None)
            if d is not None:
                by_file[t.stem].append(vec(np.asarray(d)[:15]))
        means = []
        for stem, v in by_file.items():
            A = np.asarray(v)
            if len(A) >= 2:
                within.append(A - A.mean(0))
            means.append(A.mean(0))
        if len(means) >= 2:
            M = np.asarray(means)
            between.append(M - M.mean(0))
    return np.vstack(within), np.vstack(between)


def rms(D):
    return float(np.sqrt((D ** 2).sum(1).mean()))


def per_order(D):
    n = len(ORD)
    v = (D ** 2).mean(0)
    return np.sqrt(v[:n] + v[n:])


def main():
    lib = default_library()
    circ = SmpsCircuit()
    print("텍스처 라이브러리: %s" % lib.describe())

    rw, rb = real_drift()
    sw, sb = syn_drift(lib, circ)

    print("\n%s" % ("=" * 74))
    print("**짝 맞춘 비교** — 합성 텍스처 델타 대 실측 표류")
    print("=" * 74)
    print("  %-28s %12s %12s %8s" % ("", "합성", "실측", "비"))
    for lbl, S, R in (("[안]  한 녹화/파일 안", sw, rw),
                      ("[사이] 녹화/파일 사이", sb, rb)):
        print("  %-28s %9.1f mA %9.1f mA %7.2f배   (표본 %d/%d)"
              % (lbl, 1000 * rms(S), 1000 * rms(R), rms(S) / max(rms(R), 1e-12),
                 len(S), len(R)))

    print("\n  차수별 [안] (mA)")
    ps, pr = per_order(sw), per_order(rw)
    print("  %-6s %10s %10s %8s" % ("차수", "합성", "실측", "비"))
    for i, o in enumerate(ORD):
        print("  h%-5d %8.2f %10.2f %8.2f" % (o, 1000 * ps[i], 1000 * pr[i],
                                              ps[i] / max(pr[i], 1e-12)))

    # 방향이 맞나
    bas = np.load("results/drift_basis.npz", allow_pickle=True)
    Vb = np.asarray(bas["V"], float)
    Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
    for lbl, D in (("[안]", sw), ("[사이]", sb)):
        if len(D) < 5:
            continue
        U = np.linalg.svd(D - D.mean(0), full_matrices=False)
        ev = np.cumsum(U[1] ** 2 / (U[1] ** 2).sum())
        s = np.linalg.svd(Vb @ U[2][:3].T, compute_uv=False)
        cap = ((D @ Vb.T) ** 2).sum() / (D ** 2).sum()
        print("\n  %s 합성 표류 차원 %s · 실측기저와 주각 %s · 실측기저 안 %.3f (무작위 0.188)"
              % (lbl, np.round(ev[:3], 3),
                 " ".join("%.0f°" % np.degrees(np.arccos(np.clip(x, 0, 1))) for x in s), cap))
    print("\n  실측 [안] 차원 %s"
          % np.round(np.cumsum(np.linalg.svd(rw - rw.mean(0), compute_uv=False) ** 2
                               / (np.linalg.svd(rw - rw.mean(0), compute_uv=False) ** 2).sum())[:3], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
