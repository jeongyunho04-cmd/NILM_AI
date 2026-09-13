# -*- coding: utf-8 -*-
"""표류 부분공간이 **실측 복합 부하에서도 성립하나** — 조합 전이 검정.

13.84.52 는 부분공간을 **충전기 단독 녹화**에서만 배웠고 LORO 로 *녹화* 전이만 쟀다.
실제 배치는 충전기가 **다른 기기와 같이** 켜진 상태다. 그 조합에서도 같은 3차원인지는
아무도 안 쟀다 ([[validate-forward-model-on-held-out-combos]]).

⚠ `composite_eval` 에는 기기별 라벨이 없다 (metadata: *"단일 가전이 아니므로 상태 라벨과
  세그먼트 풀 사용 금지"*). 그래서 표류를 **구성이 안 바뀐 이웃 블록의 차분**으로 뽑는다 —
  시간이 붙어 있고 총전력이 1% 안에서 같으면 기기 구성이 바뀌었을 리 없다.
  차분은 편차의 √2 배라 나눠 준다.

재는 것:
  ① 실측 복합 표류도 **3차원인가**
  ② 충전기에서 배운 3차원이 실측 복합 표류를 **얼마나 담는가** (귀무: 무작위 3차원 = 3/16)
  ③ 두 부분공간의 **주각** (principal angles)
  ④ ⚠ 13.84.60 확인 — PC1 이 **미니PC-형제 판별축**과 몇 도인가
     (13.84.52 ② 는 'ON−대기' 축에 대 87° 라 적었는데 그건 과제 축이 아니다)

    python -X utf8 src/run_diag_driftreal.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, load, vec

BLOCK = 10 * 60          # 10초 = drift_basis.npz 의 block
P_TOL = 0.01             # 이웃 블록 총전력 차 1% 이내여야 같은 구성으로 본다
P16 = 2 * len(ORD)
RNG = np.random.default_rng(0)


def composite_drift():
    """실측 복합 파일에서 구성이 안 바뀐 이웃 블록 차분. (n,16)"""
    out, tags, kept, tot = [], [], 0, 0
    for p in sorted(glob.glob("processed_data/composite_eval/*.npz")):
        d = np.load(p, allow_pickle=True)
        hc = np.asarray(d["harmonics_complex"])
        P = np.asarray(d["p_denoised_w"])
        ok = (np.asarray(d["is_valid"]) == 1) & (np.asarray(d["is_unplugged"]) == 0)
        if "is_segment_seam" in d.files:
            ok &= np.asarray(d["is_segment_seam"]) == 0
        n = len(P) // BLOCK
        B = np.array([vec(hc[i * BLOCK:(i + 1) * BLOCK].mean(0)) for i in range(n)])
        pb = np.array([P[i * BLOCK:(i + 1) * BLOCK].mean() for i in range(n)])
        okb = np.array([ok[i * BLOCK:(i + 1) * BLOCK].all() for i in range(n)])
        for i in range(n - 1):
            tot += 1
            if not (okb[i] and okb[i + 1]) or pb[i] < 20:
                continue
            if abs(pb[i + 1] - pb[i]) / max(pb[i], 1e-9) > P_TOL:
                continue
            out.append((B[i + 1] - B[i]) / np.sqrt(2.0))
            tags.append(os.path.basename(p)[:-4])
            kept += 1
    print("   이웃 블록 %d쌍 중 **구성 불변 %d쌍** (총전력 1%% 이내 · 20W 이상)" % (tot, kept))
    return np.asarray(out), np.asarray(tags)


def captured(D, V):
    """D 의 분산 중 span(V) 안에 있는 몫."""
    num = ((D @ V.T) ** 2).sum()
    den = (D ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def sig_perwatt(app, lo, hi):
    """기기의 와트당 지문 (ON 대역 − 대기)."""
    acc = []
    for stem, P, hc, on, off in load(app):
        a = (P >= lo) & (P <= hi) & on
        if a.sum() < 300 or off.sum() < 300:
            continue
        v = (np.median(hc[a].real, 0) + 1j * np.median(hc[a].imag, 0)) \
            - (np.median(hc[off].real, 0) + 1j * np.median(hc[off].imag, 0))
        acc.append(vec(v) / max(np.median(P[a]), 1e-9))
    return np.asarray(acc).mean(0) if acc else None


def ang(a, b):
    c = abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return np.degrees(np.arccos(np.clip(c, 0, 1))), c


def main():
    bas = np.load("results/drift_basis.npz", allow_pickle=True)
    V = np.asarray(bas["V"], float)                 # (3,16) 충전기 단독에서 배운 것
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    print("=" * 78)
    print("표류 부분공간의 **조합 전이** 검정")
    print("=" * 78)
    print("기저 drift_basis.npz — k=%s · block=%s · 블록 %s · 설명분산 %s"
          % (bas["k"], bas["block"], bas["n_blocks"], np.round(bas["explained"], 3)))

    print("\n실측 복합 (`composite_eval`, 기기 라벨 없음):")
    D, tags = composite_drift()
    if len(D) < 10:
        print("   표본 부족"); return 1
    print("   표류 크기 RMS %.1f mA · 파일 %s"
          % (1000 * np.sqrt((D ** 2).sum(1).mean()), sorted(set(tags))))

    # ── ① 실측 복합 표류의 차원 ────────────────────────────────────────────
    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    ev = np.cumsum(S ** 2 / (S ** 2).sum())
    null = []
    for _ in range(400):
        Z = RNG.normal(size=D.shape)
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        null.append(np.cumsum(s ** 2 / (s ** 2).sum()))
    null = np.asarray(null)
    print("\n① **실측 복합 표류도 3차원인가**")
    print("   %-8s %8s %20s" % ("", "실측", "등방잡음 중앙(p5~p95)"))
    for i in (0, 1, 2, 5):
        lo, md, hi = np.percentile(null[:, i], [5, 50, 95])
        print("   PC1~%-2d %9.2f %11.2f (%.2f~%.2f)  %s"
              % (i + 1, ev[i], md, lo, hi, "**초과**" if ev[i] > hi else "아니오"))

    # ── ② 충전기 기저가 실측 복합을 담는가 ─────────────────────────────────
    cap = captured(D - D.mean(0), V)
    rnd = []
    for _ in range(2000):
        Q, _ = np.linalg.qr(RNG.normal(size=(P16, 3)))
        rnd.append(captured(D - D.mean(0), Q.T))
    rnd = np.asarray(rnd)
    own = captured(D - D.mean(0), Vt[:3])
    print("\n② **충전기 단독에서 배운 3차원이 실측 복합 표류를 담는 몫**")
    print("   충전기 기저      %.3f" % cap)
    print("   무작위 3차원     %.3f  (p5~p95 %.3f~%.3f · 이론 3/16=%.3f)"
          % (rnd.mean(), np.percentile(rnd, 5), np.percentile(rnd, 95), 3 / 16))
    print("   실측 제 기저     %.3f  <- 천장" % own)
    p = float((rnd >= cap).mean())
    print("   -> p=%.3f %s" % (p, "**무작위와 구분 안 됨 = 조합 전이 실패**" if p > 0.05
                               else "무작위보다 유의하게 많이 담는다"))
    print("   담는 몫을 천장으로 정규화: %.1f%% (무작위 %.1f%%)"
          % (100 * cap / own, 100 * rnd.mean() / own))

    # ── ③ 주각 ─────────────────────────────────────────────────────────────
    s = np.linalg.svd(V @ Vt[:3].T, compute_uv=False)
    print("\n③ 두 3차원 부분공간의 **주각**: %s"
          % " · ".join("%.0f°" % np.degrees(np.arccos(np.clip(x, 0, 1))) for x in s))
    print("   (0° 면 같은 부분공간 · 90° 면 직교. 무작위 3차원끼리는 대개 55~80°)")

    # ── ④ 판별축 — 13.84.60 확인 ───────────────────────────────────────────
    mp = sig_perwatt("minipc", 8, 14)
    ch = sig_perwatt("laptop_charger", 19, 70)
    print("\n④ **어느 축에 대는가가 결론을 뒤집는다**")
    if mp is not None and ch is not None:
        disc = mp / np.linalg.norm(mp) - ch / np.linalg.norm(ch)   # 미니PC−형제 판별축
        onstb = mp
        for lbl, ax in (("미니PC ON−대기 (13.84.52 ②가 쓴 축)", onstb),
                        ("미니PC−형제 **판별축** (과제 축)", disc)):
            a1, _ = ang(V[0], ax)
            print("   %-38s PC1 %5.1f°" % (lbl, a1))
            print("      %-35s PC2 %5.1f° · PC3 %5.1f°" % ("", ang(V[1], ax)[0], ang(V[2], ax)[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
