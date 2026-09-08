"""SMPS 배분은 **무엇으로 가릴 수 있나** — 특징별 판별력 (13.62)
=================================================================
사용자: *"smps끼리의 배분을 어떻게 결정하는거야. 결정할수 있는 방법을 찾고 그걸
배우도록 모델을 유도해보는게 낫지 않을까?"*

지금까지는 손실의 **구멍**을 막아 왔다 (13.60·13.61). 그 전에 물어야 할 것이 있다:
그 배분이 신호에서 **원리상 결정 가능한가.** 안 되면 어떤 손실 항을 손봐도 못 간다.

    d′(특징, A->B) = |특징이 10W 맞바뀜에 움직이는 양| / (그 특징의 창간 산포)

10W 를 A 에서 B 로 옮겼을 때 그 특징이 잡음보다 얼마나 크게 움직이나. **d′ < 1 이면
그 특징으로는 못 가른다** — 창 하나로는. 창을 N 개 모으면 √N 만큼 좋아지므로
`창 N 개 누적` 열도 같이 낸다.

⚠ 순방향 모형은 **현장 지문**(13.59.7, 조합 차분으로 푼 것)을 쓴다. 격리 지문으로
  재면 전이 오차가 판별력으로 잘못 들어간다.

⚠ 산포는 관측의 창간 표준편차다. 여기에는 기기 자체의 전력 요동도 들어 있어
  **잡음을 과대평가**한다 — 그래서 이 표의 d′ 는 **하한**이다. 관측 총전력을
  회귀로 뺀 잔차 산포도 같이 낸다 (그쪽이 상한에 가깝다).

    python -m src.run_smps_identify
    python -m src.run_smps_identify --swap-w 5 --harm-weight inv_h2
"""
import argparse
import itertools
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import dense_targets
from src.run_baseline import S_I
from src.run_loss_compare import SMPS, BIG

ORD = (1, 3, 5, 7, 9, 11, 13, 15)
#: 13.59.7 에서 조합 차분으로 다시 잰 참 ON 전력 (대기 되돌림 포함)
P_TRUE = {"beam_projector": 44.64, "laptop_charger": 38.87, "minipc": 8.93}


def in_situ_signatures(apps, stems, js, jb, ev):
    """복합 파일의 켜짐 조합에서 현장 지문을 푼다 (13.59.7 과 같은 풀개)."""
    from src.model.net import standby_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sb = standby_signatures(pool, apps)
    del pool
    sbc = sb[:, :, 0] + 1j * sb[:, :, 1]
    rows, qrows = {}, {}
    for stem in stems:
        rw = dense_targets(stem, stride=30, site_transfer=None)
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        base = (~on[t][:, jb].any(1)) & sc[t][:, js].all(1)
        OH = np.concatenate([rw.batch(np.arange(i, min(i + 1024, len(rw))))[3]
                             for i in range(0, len(rw), 1024)])
        OH = OH[..., 0] + 1j * OH[..., 1]
        Q = np.asarray(rw.q_observed, np.float64)
        P = np.concatenate([rw.batch(np.arange(i, min(i + 1024, len(rw))))[2]
                            for i in range(0, len(rw), 1024)])
        pat = on[t][:, js].astype(int)
        for combo in itertools.product([0, 1], repeat=3):
            m = base & (pat == np.array(combo)).all(1)
            if m.sum() >= 20:
                rows.setdefault(combo, []).append(OH[m])
                qrows.setdefault(combo, []).append(np.stack([Q[m], P[m]], 1))
    med = {c: (lambda x: np.median(x.real, 0) + 1j * np.median(x.imag, 0))(
        np.concatenate(v)) for c, v in rows.items()}
    qmed = {c: np.median(np.concatenate(v), 0) for c, v in qrows.items()}
    n = {c: len(np.concatenate(v)) for c, v in rows.items()}
    combos = [c for c in sorted(med, reverse=True) if c != (0, 0, 0)]
    A = np.asarray([[c[i] for i in range(3)] for c in combos], float)
    w = np.sqrt(np.asarray([n[c] for c in combos], float))
    off, qoff = med[(0, 0, 0)], qmed[(0, 0, 0)]
    sig = np.zeros((3, 15), complex)
    for h in range(15):
        b = np.asarray([med[c][h] - off[h] for c in combos])
        x, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        sig[:, h] = [(x[i] + sbc[js[i], h]) / P_TRUE[SMPS[i]] for i in range(3)]
    bq = np.asarray([qmed[c][0] - qoff[0] for c in combos])
    xq, *_ = np.linalg.lstsq(A * w[:, None], bq * w, rcond=None)
    qp = np.asarray([xq[i] / P_TRUE[SMPS[i]] for i in range(3)])
    return sig, qp, med, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", default=["test_3", "test_4", "test_5"])
    ap.add_argument("--swap-w", type=float, default=10.0, help="맞바꿀 전력 W")
    ap.add_argument("--harm-weight", default="inv_h2", choices=("off", "inv_h", "inv_h2"))
    a = ap.parse_args()
    apps = list(S_I)
    js = [apps.index(x) for x in SMPS]
    jb = [apps.index(x) for x in BIG if x in apps]
    ev = load_events()
    sig, qp, med, ncomb = in_situ_signatures(apps, a.stems, js, jb, ev)

    # ── 판정 창의 관측 산포 ──────────────────────────────────────────────
    OH, Q, P = [], [], []
    for stem in a.stems:
        rw = dense_targets(stem, stride=30, site_transfer=None)
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        k = np.flatnonzero((~on[t][:, jb].any(1)) & on[t][:, js].all(1)
                           & sc[t][:, js].all(1))
        if not len(k):
            continue
        for i in range(0, len(k), 512):
            idx = k[i:i + 512]
            _f, _w, pobs, oh, _pn = rw.batch(idx)
            OH.append(oh); P.append(pobs)
        Q.append(np.asarray(rw.q_observed, np.float64)[k])
    obs = np.concatenate(OH); obs = obs[..., 0] + 1j * obs[..., 1]
    q = np.concatenate(Q); p = np.concatenate(P)
    print(f"자리 D · SMPS 전용 · 셋 다 켜진 창 {len(obs)}개")
    print(f"현장 지문 (조합 {len(ncomb)}개로 푼 것) · 맞바뀜 {a.swap_w:.0f}W\n")

    def resid_std(x):
        """관측 총전력을 회귀로 뺀 잔차의 표준편차 (기기 요동을 일부 제거)."""
        X = np.stack([np.ones_like(p), p], 1)
        c, *_ = np.linalg.lstsq(X, x, rcond=None)
        return np.std(x - X @ c, axis=0)

    hh = np.arange(1, 16, dtype=float)
    wmap = {"off": np.ones(15), "inv_h": 1.0 / hh, "inv_h2": 1.0 / hh ** 2}
    wh = wmap[a.harm_weight] / wmap[a.harm_weight].max()

    for i, j in itertools.combinations(range(3), 2):
        na, nb = SMPS[i][:8], SMPS[j][:8]
        print(f"=== {na} <-> {nb}  ({a.swap_w:.0f}W 이동) ===")
        print(f"  {'특징':12s}{'변화':>11s}{'산포':>11s}{'잔차산포':>11s}"
              f"{'d′':>8s}{'d′(잔차)':>10s}{'가중 뒤 d′':>12s}")
        rows = []
        for h in ORD:
            d = a.swap_w * (sig[i, h - 1] - sig[j, h - 1])
            s_raw = float(np.std(np.abs(obs[:, h - 1])))
            s_res = float(resid_std(np.abs(obs[:, h - 1])))
            dm = float(abs(d)) * 1000.0
            dp = dm / max(s_raw * 1000.0, 1e-9)
            dr = dm / max(s_res * 1000.0, 1e-9)
            rows.append((h, dm, s_raw * 1000, s_res * 1000, dp, dr, dp * wh[h - 1]))
        for h, dm, sr, ss, dp, dr, dw in rows:
            print(f"  {'|I' + str(h) + '| mA':12s}{dm:11.3f}{sr:11.3f}{ss:11.3f}"
                  f"{dp:8.3f}{dr:10.3f}{dw:12.4f}")
        dq = a.swap_w * abs(qp[i] - qp[j])
        sq_raw, sq_res = float(np.std(q)), float(resid_std(q))
        print(f"  {'Q VAR':12s}{dq:11.3f}{sq_raw:11.3f}{sq_res:11.3f}"
              f"{dq / max(sq_raw, 1e-9):8.3f}{dq / max(sq_res, 1e-9):10.3f}"
              f"{'—':>12s}")
        print(f"  {'P W':12s}{0.0:11.3f}{float(np.std(p)):11.3f}"
              f"{'—':>11s}{0.0:8.3f}{'—':>10s}{'—':>12s}   <- 맞바뀜은 합을 안 바꾼다")
        best = max(rows, key=lambda r: r[4])
        bw = max(rows, key=lambda r: r[6])
        print(f"  최고 d′ h{best[0]} {best[4]:.3f}  ->  창 {len(obs)}개 누적 "
              f"{best[4] * np.sqrt(len(obs)):.1f}")
        print(f"  가중({a.harm_weight}) 뒤 최고는 h{bw[0]} {bw[6]:.4f} "
              f"-> 누적 {bw[6] * np.sqrt(len(obs)):.1f}\n")
    print("  d′ < 1 이면 창 하나로는 못 가른다. 창을 모으면 √N 로 자란다 —")
    print("  그래서 **창 누적** 열이 실제로 결정 가능한지를 말한다.")
    print("  ⚠ 누적이 커도 순방향 모형 오차가 그만큼이면 못 쓴다 (13.59.7 의 전이 오차).")


if __name__ == "__main__":
    main()
