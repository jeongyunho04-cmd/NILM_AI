"""배분의 **두 번째 판별자**를 찾는다 — 재학습 없이 (12.133)

왜
--
12.132.1 이 실측 `L_harm` 을 끄는 것이 그것을 고치려던 25판 전부보다 낫다고
쟀다. 그런데 그것은 진단이지 처방이 아니다 — 고조파를 통째로 버리면 새 기기
앞에서 배울 것이 없다. 그리고 12.132 가 근본 원인을 **판별자가 하나뿐인 것**
으로 짚었다 (저항은 등가저항 `R=V^2/P` 라는 둘째 판별자가 있어서 조건수가
21.7 로 더 나쁜데도 배분이 된다).

그래서 둘을 묻는다:

    (A) 고조파 **안에서** 신뢰할 자리가 따로 있는가
        차수별 모델오차가 8배 갈린다 (h1 0.079 ~ h14 1.696, 12.123.1).
        그런데 `L_harm` 은 `harm_scale` 정규화 뒤 15차수를 같은 무게로 본다.
        **차수 부분집합마다 12.122.2 의 자를 다시 풀면** 재가중이 통할지 나온다

    (B) 고조파 **밖에** 쓸 것이 있는가 — 무효전력 Q
        `L_cons` 는 P 만 본다. **Q 제약이 손실 어디에도 없다** (입력 채널 31 에
        있는데 안 쓴다). Q 는 60Hz 에서 가산이므로 `Σ Q_i = Q관측` 이 성립하고,
        저항의 등가저항처럼 **기기 고유값**이면 둘째 판별자가 된다

판정
----
(A) 어떤 차수 부분집합이 **모델오차/판별신호 < 1** 을 만들면 재가중이 통한다.
    전부 1 을 넘으면 고조파 안에는 답이 없다 (12.122.2 의 벽이 차수와 무관)
(B) Q/W 의 d' 가 고조파 d' 를 넘으면 둘째 판별자다. 규칙 23 — 분모를 같이 낸다

    python -m src.run_second_judge_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse
import glob
import itertools
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import HUMAN_ON_DEFAULT_STEMS, SMPS_APPLIANCES
from src.preprocessing import classify_file, load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW

H = 15


# ─────────────────────────────────────────────────────────────────────────
# (A) 차수 부분집합마다 12.122.2 의 자
# ─────────────────────────────────────────────────────────────────────────
def windows(apps: Sequence[str], nz: np.ndarray) -> List[dict]:
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out = []
    for stem in HUMAN_ON_DEFAULT_STEMS:
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        for k in range(0, len(i) - WINDOW, WINDOW):
            sl = i[k:k + WINDOW]
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if len(sup):
                out.append({"sup": sup, "y": hc[sl].mean(0)[:H] - NZ})
    return out


def solve(sig: np.ndarray, idx: Sequence[int], y: np.ndarray,
          orders: Sequence[int]) -> float:
    """지정한 차수만 써서 NNLS. 잔차는 그 차수들에서만 잰다."""
    o = np.asarray(orders)
    b = np.concatenate([y[o].real, y[o].imag])
    if not len(idx):
        return float(np.linalg.norm(b))
    A = np.array([np.concatenate([sig[j, o, 0], sig[j, o, 1]]) for j in idx]).T
    _, r = nnls(A, b)
    return float(r)


# ─────────────────────────────────────────────────────────────────────────
# (B) 무효전력 Q — 기기 고유값인가
# ─────────────────────────────────────────────────────────────────────────
def q_per_watt(app_filter=None) -> Dict[str, np.ndarray]:
    """격리 녹화의 통전 구간에서 기기별 Q/P (60초 창 평균)."""
    out: Dict[str, List[float]] = {}
    for f in sorted(glob.glob("processed_data/npz/*.npz")):
        try:
            app = classify_file(f).appliance_type
        except Exception:
            continue
        if not app or (app_filter and app not in app_filter):
            continue
        z = load_nilm_npz(f)
        pf = np.asarray(z["power_features"])            # [p, q, s, pf, vrms, thd_i]
        p = np.asarray(z["p_denoised_w"])
        m = (np.asarray(z["is_on"]).astype(bool)
             & np.asarray(z["is_valid"]).astype(bool) & (p > 1.0))
        i = np.flatnonzero(m)
        W = 3600
        for k in range(0, len(i) - W, W // 4):
            s = i[k:k + W]
            if s[-1] - s[0] > W * 1.5:
                continue
            pm = float(p[s].mean())
            if pm < 1.0:
                continue
            out.setdefault(app, []).append(float(pf[s, 1].mean()) / pm)
    return {k: np.array(v) for k, v in out.items() if len(v) >= 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt")
    ap.add_argument("--out", default="results/second_judge.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.net import harmonic_signatures, harmonic_scales, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG = harmonic_signatures(pool, apps); HSC = harmonic_scales(pool, apps)
    NZ = noise_signature(pool)
    del pool

    print("=" * 90)
    print("배분의 두 번째 판별자 — 고조파 안에 있는가, 밖에 있는가")
    print("=" * 90)

    # ── (A) 차수 부분집합 ─────────────────────────────────────────────
    wins = windows(apps, NZ)
    allj = list(range(len(apps)))
    ODD = [0, 2, 4, 6, 8, 10, 12, 14]
    SUBSETS = [
        ("전 15차 (현행)", list(range(H))),
        ("홀수차 전부", ODD),
        ("짝수차 전부", [1, 3, 5, 7, 9, 11, 13]),
        ("h1 만", [0]),
        ("h1,h3", [0, 2]),
        ("h1~h5 홀수", [0, 2, 4]),
        ("h1~h9 홀수", [0, 2, 4, 6, 8]),
        ("h3~h9 홀수 (h1 뺌)", [2, 4, 6, 8]),
        ("h9~h15 홀수 (고차만)", [8, 10, 12, 14]),
    ]
    print(f"\n[A] 차수 부분집합마다 12.122.2 의 자 — 창 {len(wins)}개, 모델도 손실도 없다\n")
    print(f"  {'차수 집합':<22s}{'차원':>5s}{'자유':>9s}{'정답':>9s}"
          f"{'판별신호':>10s}{'**모델오차/판별신호**':>22s}")
    print("  " + "-" * 78)
    rowsA = {}
    for name, o in SUBSETS:
        fr = [solve(SIG, allj, w["y"], o) for w in wins]
        tr = [solve(SIG, list(w["sup"]), w["y"], o) for w in wins]
        f, t = float(np.mean(fr)), float(np.mean(tr))
        ratio = t / max(t - f, 1e-12)
        rowsA[name] = {"free": f, "true": t, "ratio": ratio, "n_dim": 2 * len(o)}
        mark = "  <- 1 아래!" if 0 < ratio < 1 else ""
        print(f"  {name:<22s}{2 * len(o):>5d}{f:>9.4f}{t:>9.4f}"
              f"{t - f:>10.4f}{ratio:>18.2f}배{mark}")
    print("\n  1 아래인 집합이 있으면 **재가중으로 고칠 수 있다**.")
    print("  전부 1 을 넘으면 12.122.2 의 벽은 차수와 무관하고, 고조파 안에는 답이 없다.")

    # ── (B) 무효전력 Q ────────────────────────────────────────────────
    Q = q_per_watt()
    print(f"\n[B] 무효전력 Q/P — 기기 고유값인가 (격리 60초 창)\n")
    print(f"  {'기기':<18s}{'창':>5s}{'p5':>9s}{'중앙':>9s}{'p95':>9s}"
          f"{'폭/|중앙|':>11s}")
    print("  " + "-" * 62)
    stat = {}
    for app in sorted(Q):
        v = Q[app]
        lo, mid, hi = np.percentile(v, [5, 50, 95])
        stat[app] = (float(lo), float(mid), float(hi))
        mark = "  <- SMPS" if app in SMPS_APPLIANCES else ""
        print(f"  {app:<18s}{len(v):>5d}{lo:>9.3f}{mid:>9.3f}{hi:>9.3f}"
              f"{(hi - lo) / max(abs(mid), 1e-9):>11.3f}{mark}")

    print(f"\n  SMPS 쌍의 판별력 d' — 분모는 두 기기 창간 산포의 평균 (규칙 23)\n")
    print(f"  {'쌍':<34s}{'Q/P d′':>10s}{'고조파 d′ (와트당)':>20s}")
    print("  " + "-" * 66)
    # 고조파 d' 는 12.124 와 같은 자 — 와트당 지문 차이 / 창간 산포
    from src.run_state_sig_probe import median_signature_scatter
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SD = median_signature_scatter(pool, apps); del pool
    sm = [x for x in SMPS_APPLIANCES if x in apps and x in Q]
    outB = {}
    for x, y in itertools.combinations(sm, 2):
        vx, vy = Q[x], Q[y]
        sx, sy = np.std(vx), np.std(vy)
        dq = abs(np.median(vx) - np.median(vy)) / max((sx + sy) / 2, 1e-9)
        i, j = apps.index(x), apps.index(y)
        ci = SIG[i, :, 0] + 1j * SIG[i, :, 1]
        cj = SIG[j, :, 0] + 1j * SIG[j, :, 1]
        dh = float(np.median(np.abs(ci - cj) / np.maximum((SD[i] + SD[j]) / 2, 1e-9)))
        outB[f"{x}|{y}"] = {"d_q": float(dq), "d_harm": dh}
        print(f"  {x[:15]} vs {y[:15]:<16s}{dq:>10.2f}{dh:>20.2f}")

    print("\n  d′ 가 크면 갈린다. Q 쪽이 크면 **둘째 판별자**이고, `L_cons` 에")
    print("  Q 항을 더하는 것이 저항의 `resistive_match` 에 해당한다.")
    print("\n  ⚠ 규칙 1 — 이것은 격리 통계다. 복합에서 Q 가 가산인지는 따로 봐야 한다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"order_subsets": rowsA, "q_per_watt": stat, "smps_pairs": outB,
         "n_windows": len(wins), "_config": {"argv": sys.argv, "args": vars(a)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
