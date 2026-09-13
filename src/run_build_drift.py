# -*- coding: utf-8 -*-
"""형제 지문 **표류 부분공간**을 한 번 만들어 굳힌다 (13.84.52 -> results/drift_basis.npz).

13.84.52 가 잰 것: 충전기 지문 표류는 16차원(홀수차 8개 Re/Im) 중 **3차원이 94~96%** 이고,
으뜸 모드(PC1, 분산 50%)는 미니PC 신호와 **87° 로 거의 직교**다. 그 방향을 지우면 미니PC 를
1.5% 만 잃고 표류가 42% 준다 (녹화 하나 빼기 전이 1.70배).

⚠ 해석적으로는 못 쓴다 — "차수 비례 회전" 가설은 PC1 과 53° 빗나가고 이득이 1.00배였다
  (13.84.52 ④). **자료에서 추정해야 한다.**
⚠ 미니PC 녹화는 **안 쓴다.** 표적 기기로 기저를 만들면 자기 신호를 지우는 자명한 축퇴다.

    python -X utf8 src/run_build_drift.py [--block 600] [--k 3]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftdim import ORD, blocks, load, vec

#: 기저를 만들 기기 — **형제만**. 미니PC 는 절대 넣지 않는다.
SOURCES = ("laptop_charger", "beam_projector")
BANDS = {"laptop_charger": [(19, 28), (28, 40), (40, 55), (55, 70)],
         "beam_projector": [(35, 45), (45, 55)]}
OUT = "results/drift_basis.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=600, help="블록 길이 (사이클). 600=10초")
    ap.add_argument("--k", type=int, default=3, help="남길 주성분 수")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    D, tags, apps = [], [], []
    for app in SOURCES:
        G = load(app)
        for band in BANDS[app]:
            for stem, P, hc, on, off in G:
                B = blocks(P, hc, on, *band, B=a.block)
                if len(B) < 2:
                    continue
                D.append(B - B.mean(0))
                tags += [stem] * len(B)
                apps += [app] * len(B)
    if not D:
        print("블록이 없다"); return 1
    D = np.vstack(D)
    tags, apps = np.asarray(tags), np.asarray(apps)
    print("표류 표본 %d블록 (%d사이클=%.0f초) · 녹화 %d개 · 기기 %s"
          % (len(D), a.block, a.block / 60.0, len(np.unique(tags)),
             " · ".join("%s %d" % (x, int((apps == x).sum())) for x in np.unique(apps))))

    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    ev = np.cumsum(S ** 2 / (S ** 2).sum())
    print("   누적 설명분산  %s" % "  ".join("PC%d %.2f" % (i + 1, ev[i]) for i in range(min(5, len(ev)))))
    V = Vt[:a.k]                                     # (k, 16)

    # ── 관문 — 미니PC 를 지우지 않는가 ([[augmentation-can-erase-the-discriminant]]) ──
    MP = load("minipc")
    sig = []
    for stem, P, hc, on, off in MP:
        m = (P >= 8) & (P <= 14) & on
        if m.sum() < 300 or off.sum() < 300:
            continue
        sig.append(vec((np.median(hc[m].real, 0) + 1j * np.median(hc[m].imag, 0))
                       - (np.median(hc[off].real, 0) + 1j * np.median(hc[off].imag, 0))))
    m_sig = np.mean(sig, 0)
    Pm = np.eye(V.shape[1]) - V.T @ V
    keep = float(np.linalg.norm(Pm @ m_sig) / np.linalg.norm(m_sig))
    dr = float(np.sqrt(((D @ Pm.T) ** 2).sum(1).mean()) / np.sqrt((D ** 2).sum(1).mean()))
    print("   k=%d 사영 뒤 — 미니PC %.1f%% 남음 · 표류 %.1f%% 남음 · SNR %.2f배"
          % (a.k, 100 * keep, 100 * dr, keep / dr))
    ang = [float(np.degrees(np.arccos(min(1.0, abs(
        float(V[i] @ (m_sig / np.linalg.norm(m_sig)))))))) for i in range(a.k)]
    print("   미니PC 와의 각도  %s" % "  ".join("PC%d %.0f°" % (i + 1, v) for i, v in enumerate(ang)))
    if keep < 0.5:
        print("   ⚠ 미니PC 를 절반 넘게 지운다 — k 를 줄여라"); return 1

    # 성분 표준편차 — 증강이 이 폭으로 뽑는다 (13.84.61). 없으면 생성기가 하드코딩 값을 쓴다.
    comp_sd = np.array([float(np.std(D @ V[i])) for i in range(a.k)])
    print("   성분 표준편차  %s (mA) · 3σ 합성 %.1f mA"
          % ("  ".join("PC%d %.1f" % (i + 1, 1000 * v) for i, v in enumerate(comp_sd)),
             1000 * 3 * float(np.linalg.norm(comp_sd))))
    np.savez(a.out, V=V, orders=np.array(ORD), block=a.block, k=a.k,
             explained=ev[:a.k], minipc_keep=keep, drift_keep=dr, comp_sd=comp_sd,
             sources=np.array(SOURCES), n_blocks=len(D))
    print("   -> %s 에 저장 (V %s · 차수 %s)" % (a.out, V.shape, ORD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
