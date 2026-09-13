# -*- coding: utf-8 -*-
"""형제 지문 표류를 **넘을 수 있는가** — 표류의 차원을 재고 사영해 본다 (13.84.52).

사용자: *"지문 표류가 막는다고. 이걸 극복할 방법은 없는 거야?"*

아직 아무도 **표류가 무엇인지** 안 쟀다. 백색잡음이면 평균으로 줄지만 (1분 블록이 3,600
사이클인데 9~11% 니 백색이 아니다), **낮은 차원의 구조**면 그 부분공간을 사영해 없앨 수 있다.
13.84.35 ⑦ 의 단서: 차수끼리 r(h11,h13) = +0.92, 앞뒤 절반 일치 0.52~0.57 — 한 realization 이
천천히 움직인다.

재는 것:
  ① 차원    충전기 표류의 주성분 스펙트럼. 몇 개가 몇 %를 설명하나
  ② 사영    상위 k 개를 없애면 **미니PC 신호가 얼마나 남는가**
  ③ SNR     사영 전후의 (미니PC 신호 / 남은 표류). 이것이 오르면 넘을 수 있다
  ④ 전이성  충전기에서 배운 부분공간이 **다른 녹화**에도 통하나 (안 통하면 신탁이다)

⚠ 미니PC 신호는 `I_ON − I_대기` 다 (13.84.51). 절대 ON 이 아니다.

    python -X utf8 src/run_diag_driftdim.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

ORD = [1, 3, 5, 7, 9, 11, 13, 15]
BLOCK = int(__import__("os").environ.get("DD_BLOCK", 60 * 60))   # 환경변수로 바꾼다


def vec(c):
    """(…,15) complex -> 홀수차 Re/Im 16차원 실수."""
    z = c[..., [o - 1 for o in ORD]]
    return np.concatenate([z.real, z.imag], axis=-1)


def load(app):
    """녹화별 (stem, P, hc, state0마스크)."""
    out = []
    for p in sorted(glob.glob("processed_data/npz/%s_*.npz" % app)):
        d = np.load(p, allow_pickle=True)
        if "harmonics_complex" not in d.files:
            continue
        hc = np.asarray(d["harmonics_complex"])
        P = np.asarray(d["p_denoised_w"]) if "p_denoised_w" in d.files \
            else np.asarray(d["power_features"])[:, 0]
        on = np.asarray(d["state_id"]) != 0 if "state_id" in d.files \
            else np.asarray(d["is_on"]) == 1
        ok = np.ones(len(P), bool)
        for k, w in (("is_valid", 1), ("is_unplugged", 0)):
            if k in d.files:
                ok &= np.asarray(d[k]) == w
        out.append((os.path.basename(p)[:-4], P, hc, on & ok, (~on) & ok))
    return out


def blocks(P, hc, m, lo, hi, B=None):
    """전력대 안에서 1분 블록마다 평균 페이저. (n_block, 16)"""
    B = int(B or BLOCK)
    idx = np.nonzero(m & (P >= lo) & (P <= hi))[0]
    if len(idx) < B:
        return np.zeros((0, 2 * len(ORD)))
    out = []
    for s in range(0, len(idx) - B + 1, B):
        j = idx[s:s + B]
        out.append(vec(hc[j].mean(0)))
    return np.asarray(out)


def main():
    CH = load("laptop_charger")
    MP = load("minipc")

    # ── 표류 표본: 녹화 안 블록끼리의 편차 (13.84.36 이 잰 그 양) ──────────
    devs, tags = [], []
    for band in [(19, 28), (28, 40), (40, 55), (55, 70)]:
        for stem, P, hc, on, off in CH:
            B = blocks(P, hc, on, *band)
            if len(B) < 2:
                continue
            devs.append(B - B.mean(0))
            tags += [stem] * len(B)
    if not devs:
        print("블록이 모자란다"); return 1
    D = np.vstack(devs)
    tags = np.asarray(tags)
    print("충전기 표류 표본 %d블록 (1분) · 녹화 %d개" % (len(D), len(np.unique(tags))))
    rms = float(np.sqrt((D ** 2).sum(1).mean()))
    print("   표류 크기 RMS %.1f mA" % (1000 * rms))

    # ── ① 차원 ─────────────────────────────────────────────────────────────
    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    print("\n① 표류의 주성분 — 누적 설명분산")
    print("   %s" % "".join("  PC%-4d" % (i + 1) for i in range(6)))
    print("   %s" % "".join("  %5.2f " % v for v in np.cumsum(ev)[:6]))

    # ── 미니PC 신호 = ON − 대기 (13.84.51) ─────────────────────────────────
    sig = []
    for stem, P, hc, on, off in MP:
        a = (P >= 8) & (P <= 14) & on
        b = off
        if a.sum() < 300 or b.sum() < 300:
            continue
        v = (np.median(hc[a].real, 0) + 1j * np.median(hc[a].imag, 0)) \
            - (np.median(hc[b].real, 0) + 1j * np.median(hc[b].imag, 0))
        sig.append(vec(v))
    sig = np.asarray(sig)
    m_sig = sig.mean(0)
    print("\n미니PC 신호 (ON 8~14W − 대기) · 녹화 %d개" % len(sig))
    print("   크기 %.1f mA · 녹화 간 산포 %.1f mA"
          % (1000 * np.linalg.norm(m_sig), 1000 * np.sqrt(((sig - m_sig) ** 2).sum(1).mean())))

    # ── ②③ 사영 ────────────────────────────────────────────────────────────
    print("\n②③ 상위 k 개 표류 방향을 **없애면**")
    print("   %-4s %12s %12s %12s %10s" % ("k", "미니PC 남음", "표류 남음", "SNR", "SNR 배"))
    base = None
    for k in (0, 1, 2, 3, 4, 6, 8):
        V = Vt[:k].T if k else np.zeros((D.shape[1], 0))
        Pm = np.eye(D.shape[1]) - (V @ V.T if k else 0)
        s_left = float(np.linalg.norm(Pm @ m_sig))
        d_left = float(np.sqrt(((D @ Pm.T) ** 2).sum(1).mean()))
        snr = s_left / d_left if d_left > 0 else np.inf
        if base is None:
            base = snr
        print("   %-4d %9.1f mA %9.1f mA %12.3f %9.2f배"
              % (k, 1000 * s_left, 1000 * d_left, snr, snr / base))

    # ── ④ 전이성 — 다른 녹화에서 배운 부분공간이 통하나 ─────────────────────
    print("\n④ 전이성 — **다른 녹화**에서 배운 부분공간으로 사영 (녹화 하나 빼기)")
    print("   %-4s %12s %12s %10s" % ("k", "미니PC 남음", "표류 남음", "SNR 배"))
    stems = np.unique(tags)
    for k in (1, 2, 3, 4):
        sl, dl = [], []
        for held in stems:
            tr = D[tags != held]
            te = D[tags == held]
            if len(tr) < k + 1 or len(te) < 1:
                continue
            _, _, Vt2 = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
            V = Vt2[:k].T
            Pm = np.eye(D.shape[1]) - V @ V.T
            sl.append(np.linalg.norm(Pm @ m_sig))
            dl.append(np.sqrt(((te @ Pm.T) ** 2).sum(1).mean()))
        if not sl:
            continue
        s_, d_ = float(np.mean(sl)), float(np.mean(dl))
        d0 = float(np.sqrt((D ** 2).sum(1).mean()))
        s0 = float(np.linalg.norm(m_sig))
        print("   %-4d %9.1f mA %9.1f mA %9.2f배"
              % (k, 1000 * s_, 1000 * d_, (s_ / d_) / (s0 / d0)))
    print("\n   SNR 배가 1 근처면 사영이 미니PC 도 같이 지운다 = 못 넘는다.")
    print("   뚜렷이 크면 표류가 낮은 차원이고 **넘을 수 있다**.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
