"""`s(p)` 곡선의 주장을 **우리 실측으로** 독립 검증한다 (12.166.1).

    python -m src.run_sp_verify

두 가지를 잰다:
  ① 서명이 부하 의존인가 — SVD 차원, 같은 기기 vs 다른 기기의 각도, THD 단조성
  ② 상시 배경이 실재하는가 — '모든 기기 OFF' 창의 P·|I1|·k·THD 와 곡선과의 각도
"""
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.model.net import noise_signature
from src.model.realdata import dense_targets
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.sp_curves import BACKGROUND, load_curves

SMPS = ("laptop_charger", "minipc", "beam_projector")


def ri(s):
    return np.r_[s[1:].real, s[1:].imag]


def ang(a, b):
    c = abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")

    # ── ① 서명이 부하 의존인가 ──────────────────────────────────────────
    X, LBL, PW = [], [], []
    for a in SMPS:
        for act in pool.appliance_activations.get(a, []):
            m = np.asarray(act.is_on).astype(bool)
            if not m.any():
                continue
            c = act.net_harmonics_complex[m]
            p = np.asarray(act.net_power_features)[m, 0]
            ok = (p > 3.0) & (np.abs(c[:, 0]) > 1e-4)
            if ok.sum() < 30:
                continue
            s = c[ok] / c[ok][:, [0]]
            X.append(np.c_[s[:, 1:].real, s[:, 1:].imag])
            LBL += [a] * int(ok.sum())
            PW.append(p[ok])
    X = np.vstack(X); LBL = np.array(LBL); PW = np.concatenate(PW)
    print(f"[①] 표본 {len(X):,}개 "
          + ", ".join(f"{a} {int((LBL == a).sum()):,}" for a in SMPS))

    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    print(f"  SVD 설명력  PC1 {ev[0]:.1%}  PC2 {ev[1]:.1%}  PC3 {ev[2]:.1%}"
          f"   (1~2차원 다양체)")

    def sig_at(app, p, w=0.15):
        m = (LBL == app) & (np.abs(PW / p - 1) < w)
        return X[m].mean(0) if m.sum() >= 20 else None

    print("  각도 (28차원 h1정규화 서명)")
    for a, pa, b, pb, tag in (
            ("laptop_charger", 29, "minipc", 19, "다른 기기"),
            ("laptop_charger", 47, "minipc", 27, "다른 기기"),
            ("laptop_charger", 68, "beam_projector", 49, "다른 기기"),
            ("laptop_charger", 17, "laptop_charger", 69, "**같은 기기**"),
            ("minipc", 10, "minipc", 27, "**같은 기기**")):
        sa, sb = sig_at(a, pa), sig_at(b, pb)
        lab = f"{a} {pa}W ↔ {b} {pb}W ({tag})"
        print(f"    {lab:56s} " + (f"{ang(sa, sb):5.1f}°" if sa is not None
                                   and sb is not None else "표본 부족"))

    print("  THD vs 전력 (캡 입력이면 단조 감소)")
    for a in SMPS:
        m = LBL == a
        if m.sum() < 200:
            continue
        q = np.quantile(PW[m], [0.1, 0.5, 0.9])
        row = []
        for lo, hi in zip(q[:-1], q[1:]):
            sel = m & (PW >= lo) & (PW < hi)
            if sel.sum() >= 20:
                row.append(f"{np.sqrt((X[sel] ** 2).sum(1)).mean():.2f}"
                           f"({(lo + hi) / 2:.0f}W)")
        print(f"    {a:16s} " + " -> ".join(row))

    # ── ② 상시 배경이 실재하는가 ────────────────────────────────────────
    nz = noise_signature(pool); n_c = nz[:, 0] + 1j * nz[:, 1]
    del pool
    bg = load_curves().get(BACKGROUND)
    print(f"\n[②] '모든 기기 OFF' 창의 잔여 성분")
    if bg is None:
        print("  곡선 파일이 없다 — processed_data/sp_curves.npz")
        return 0
    ev_files = {}
    for f in ("processed_data/real_events.json",
              "processed_data/real_events_refined.json"):
        ev_files.update(json.load(open(f, encoding="utf-8"))["files"])
    print(f"{'파일':11s}{'창':>6s}{'P':>8s}{'|I1|':>9s}{'k':>7s}{'THD':>7s}"
          f"{'곡선과 각도':>12s}{'noise와 각도':>13s}")
    for stem in ("test_5", "test_7", "test_8", "test_15"):
        ev = ev_files.get(stem)
        if not ev:
            continue
        n = int(ev["cycles"])
        anyon = np.zeros(n, bool)
        for _, d in ev["intervals"].items():
            anyon |= _mask(d.get("on", []), n) | _mask(d.get("uncertain", []), n)
        rw = dense_targets(stem, stride=30)
        H = np.concatenate([rw.batch(np.arange(i, min(i + 512, len(rw))))[3]
                            for i in range(0, len(rw), 512)])
        P = np.concatenate([rw.batch(np.arange(i, min(i + 512, len(rw))))[2]
                            for i in range(0, len(rw), 512)])
        V = rw.v_observed.astype(float)
        m = ~anyon[rw.target_cycle]
        if m.sum() < 20:
            print(f"{stem:11s}{int(m.sum()):6d}   (표본 부족)")
            continue
        c = H[m, :, 0] + 1j * H[m, :, 1]
        cm = np.median(c.real, 0) + 1j * np.median(c.imag, 0)
        p, v = float(np.median(P[m])), float(np.median(V[m]))
        s = cm / cm[0]
        print(f"{stem:11s}{int(m.sum()):6d}{p:8.2f}{abs(cm[0]):9.4f}"
              f"{abs(cm[0]) * v / max(p, 1e-9):7.2f}"
              f"{np.sqrt((np.abs(s[1:]) ** 2).sum()):7.2f}"
              f"{ang(ri(s), ri(bg.signature(max(p, bg.p_min)))):12.1f}"
              f"{ang(ri(s), ri(n_c / n_c[0])):13.1f}")
    print(f"  참고: 배경 5W |I1| {abs(bg.current(5.0, 224)[0]):.4f} A  vs  "
          f"미니PC 9.5W |I1| {abs(load_curves()['minipc'].current(9.5, 224)[0]):.4f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
