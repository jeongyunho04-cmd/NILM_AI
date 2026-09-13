# -*- coding: utf-8 -*-
"""13.84.52 의 주장 넷을 **귀무값에 대 본다**.

13.84.52 는 `run_diag_driftdim.py` 로 "표류는 3차원이고 으뜸 모드가 미니PC 와 87° 직교라
1.7배 넘을 수 있다" 고 적었다. 그 절 자체가 ④에서 [[check-the-ruler-against-a-known-value]] 를
인용하는데, **② 의 87° 에는 그 자를 안 댔다.** 여기서 댄다.

  ⓐ 87° 가 뜻이 있나  16차원에서 무작위 두 벡터의 각도는 90° ± 14.5° 다. 87° 는 0.2σ 다
  ⓑ PCA 스펙트럼이 뜻이 있나  같은 모양의 등방 잡음과 **같은 군 구조**로 견준다
  ⓒ 군 크기  `B - B.mean(0)` 는 블록 2개짜리 군에서 ±d 를 만든다 — 인공 저차원이다
  ⓓ ④ 의 SNR 배  분모가 녹화별 RMS 의 **평균**인데 기준선은 **통합** RMS 다 (다른 추정량)

    python -X utf8 src/run_diag_driftdim_audit.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
from src.run_diag_driftdim import ORD, blocks, load, vec

BANDS = [(19, 28), (28, 40), (40, 55), (55, 70)]
P16 = 2 * len(ORD)
RNG = np.random.default_rng(0)


def build(B):
    """`run_diag_driftdim.main` 과 **같은** 표본. 군 크기도 같이 돌려준다."""
    devs, tags, sizes = [], [], []
    for band in BANDS:
        for stem, P, hc, on, off in load("laptop_charger"):
            Bl = blocks(P, hc, on, *band, B=B)
            if len(Bl) < 2:
                continue
            devs.append(Bl - Bl.mean(0))
            tags += [stem] * len(Bl)
            sizes.append((stem, band, len(Bl)))
    return np.vstack(devs), np.asarray(tags), sizes


def minipc_sig():
    sig = []
    for stem, P, hc, on, off in load("minipc"):
        a = (P >= 8) & (P <= 14) & on
        if a.sum() < 300 or off.sum() < 300:
            continue
        v = (np.median(hc[a].real, 0) + 1j * np.median(hc[a].imag, 0)) \
            - (np.median(hc[off].real, 0) + 1j * np.median(hc[off].imag, 0))
        sig.append(vec(v))
    return np.asarray(sig).mean(0)


def ang(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return np.degrees(np.arccos(np.clip(abs(c), 0, 1))), abs(c)


def main():
    print("=" * 78)
    print("13.84.52 감사 — 귀무값에 대 본다")
    print("=" * 78)

    # ── ⓐ 16차원에서 '직교' 의 귀무값 ────────────────────────────────────
    n = 200000
    x = RNG.normal(size=(n, P16)); y = RNG.normal(size=(n, P16))
    c = np.abs((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))
    th = np.degrees(np.arccos(c))
    print("\nⓐ **16차원에서 무작위 두 방향의 각도** (n=%d)" % n)
    print("   중앙값 %.1f° · p5~p95 %.1f~%.1f° · |cos| 표준편차 %.3f (이론 1/√16=%.3f)"
          % (np.median(th), np.percentile(th, 5), np.percentile(th, 95), c.std(), 1 / np.sqrt(P16)))
    for lbl, deg in (("PC1", 87.0), ("PC2", 48.0), ("PC3", 63.0)):
        cc = abs(np.cos(np.radians(deg)))
        p = float((c >= cc).mean())
        print("   %s %4.0f°  |cos| %.3f  ->  %.2fσ · 무작위가 이만큼 가까울 확률 **p=%.3f**"
              % (lbl, deg, cc, cc / (1 / np.sqrt(P16)), p))

    # ── 표본 ─────────────────────────────────────────────────────────────
    for B in (60 * 60, 10 * 60):
        lab = "%d초" % (B // 60)
        D, tags, sizes = build(B)
        m_sig = minipc_sig()
        print("\n" + "=" * 78)
        print("블록 %s — 표본 %d · 녹화 %d · 군 %d개" % (lab, len(D), len(np.unique(tags)), len(sizes)))

        # ⓒ 군 크기
        cnt = np.array([s[2] for s in sizes])
        eff = int((cnt - 1).sum())
        print("\nⓒ **군(녹화x전력대)별 블록 수** %s" % np.sort(cnt)[::-1].tolist())
        print("   군 2개짜리: %d개 — 이 군의 편차는 ±d 로 **완전 공선**이다 (인공 rank-1)"
              % int((cnt == 2).sum()))
        print("   자유도: 표본 %d 인데 **유효 %d** (군마다 평균을 빼서 1 씩 잃는다), 차원 %d"
              % (len(D), eff, P16))

        # ⓑ PCA 스펙트럼 대 귀무
        U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
        ev = np.cumsum(S ** 2 / (S ** 2).sum())
        null = []
        for _ in range(200):
            Z = []
            for _st, _bd, m in sizes:
                g = RNG.normal(size=(m, P16))
                Z.append(g - g.mean(0))            # **같은 군 구조**로 만든다
            Z = np.vstack(Z)
            s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
            null.append(np.cumsum(s ** 2 / (s ** 2).sum()))
        null = np.asarray(null)
        print("\nⓑ **PCA 누적 설명분산 — 실제 대 등방잡음(같은 군 구조)**")
        print("   %-6s %8s %18s %8s" % ("", "실제", "귀무 중앙(p5~p95)", "초과?"))
        for i in (0, 1, 2, 5):
            lo, md, hi = np.percentile(null[:, i], [5, 50, 95])
            print("   PC1~%-2d %8.2f %8.2f (%.2f~%.2f) %8s"
                  % (i + 1, ev[i], md, lo, hi, "예" if ev[i] > hi else "**아니오**"))

        # ⓐ 실제 각도
        print("\n   실제 PC 대 미니PC 각도:")
        for i in range(3):
            a, cc = ang(Vt[i], m_sig)
            p = float((c >= cc).mean())
            print("      PC%d  %5.1f°  |cos| %.3f  p=%.3f %s"
                  % (i + 1, a, cc, p, "<- 무작위와 구분 안 됨" if p > 0.05 else ""))

        # ⓓ k=1 수치 재계산 + ④ 추정량
        print("\nⓓ **k=1 재계산** (13.84.52 ③: '미니PC 1.5% 손실 · 표류 42% 제거')")
        s0 = np.linalg.norm(m_sig); d0 = np.sqrt((D ** 2).sum(1).mean())
        V = Vt[:1].T
        Pm = np.eye(P16) - V @ V.T
        s1 = np.linalg.norm(Pm @ m_sig); d1 = np.sqrt(((D @ Pm.T) ** 2).sum(1).mean())
        print("   미니PC %.1f -> %.1f mA  = **%.2f%% 손실** (기록 1.5%%)"
              % (1000 * s0, 1000 * s1, 100 * (1 - s1 / s0)))
        print("   표류   %.1f -> %.1f mA  = **%.1f%% 제거** (기록 42%%) · 분산으로는 %.1f%%"
              % (1000 * d0, 1000 * d1, 100 * (1 - d1 / d0), 100 * (1 - (d1 / d0) ** 2)))
        print("   SNR 배 %.2f  <- '42%%' 는 이 1.41 을 옮겨 적은 것으로 보인다" % ((s1 / d1) / (s0 / d0)))

        print("\n   **④ 의 SNR 배가 추정량을 섞는다** — 분모는 녹화별 RMS 의 평균,")
        print("   기준선은 통합 RMS 다. 같은 자로 다시 잰다:")
        stems = np.unique(tags)
        print("   %-3s %10s %12s %12s" % ("k", "기록값(섞임)", "통합/통합", "차이"))
        for k in (1, 2, 3):
            sl, dl_mean, te_all = [], [], []
            for held in stems:
                tr, te = D[tags != held], D[tags == held]
                if len(tr) < k + 1 or len(te) < 1:
                    continue
                _, _, V2 = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
                Pm2 = np.eye(P16) - V2[:k].T @ V2[:k]
                sl.append(np.linalg.norm(Pm2 @ m_sig))
                dl_mean.append(np.sqrt(((te @ Pm2.T) ** 2).sum(1).mean()))
                te_all.append(((te @ Pm2.T) ** 2).sum(1))
            mixed = (np.mean(sl) / np.mean(dl_mean)) / (s0 / d0)
            pooled = (np.mean(sl) / np.sqrt(np.concatenate(te_all).mean())) / (s0 / d0)
            print("   %-3d %8.2f배 %10.2f배 %10.1f%%"
                  % (k, mixed, pooled, 100 * (mixed / pooled - 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
