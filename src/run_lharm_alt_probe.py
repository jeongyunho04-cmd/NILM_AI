"""`L_harm` 의 결함을 고칠 것인가, 대신할 것을 넣을 것인가 (12.135)

무엇이 확인됐나
-------------
```
12.122.2   실측에서 `L_harm` 의 최소는 **오답 쪽**이다 (모델오차/판별신호 2.23)
12.132.1   그 항을 끄면 프로젝터 −18.9W. 고치려던 25판 전부보다 낫다
12.133(A)  그 비가 **차수에 단조**다 — h1 만 1.08, 고차만 3.68
```

마지막 것이 결함의 정체를 짚는다. `L_harm` 은 `harm_scale` 로 15차수를 균등화해
놓고 **신뢰도가 3.4배 다른 것을 똑같이 믿는다.** 두 길이 열린다:

    (A) 결함을 없앤다 — 차수별 신뢰도로 **가중**한다
        `harm_mask` 가 이미 차수별 가중 자리다. 상한이 얼마인지 먼저 잰다

    (B) 대신할 것을 넣는다 — **전이(轉移)**
        지금까지 실패한 것의 공통점은 **기기별 상수**를 요구한 것이다
        (`sig_i` 근사, `qp_i` 근사+비가산). 전이는 다르다 — 켜짐/꺼짐 순간의
        Δ 는 **한 기기에 귀속**되고 그 기기의 지문을 자료에서 직접 관측한다

무엇을 재면 갈리는가
-----------------
    (A) 차수 가중으로 도달 가능한 **최선의 비**. 1 아래로 못 가면 가중은 상한이 있다
    (B) 전이에서 관측한 Δ지문이 정상상태 지문보다 **기기를 잘 가르는가** (규칙 23)

    python -m src.run_lharm_alt_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse
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
from src.preprocessing import load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW
from src.run_summarize_gate import human_stems

H = 15


# ─────────────────────────────────────────────────────────────────────────
# (A) 차수 가중의 상한
# ─────────────────────────────────────────────────────────────────────────
def load_windows(apps: Sequence[str], nz: np.ndarray) -> List[dict]:
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


def ratio_for_weights(sig: np.ndarray, wins: List[dict], allj: List[int],
                      w: np.ndarray) -> float:
    """차수 가중 `w` 를 준 좌표계에서 모델오차/판별신호."""
    s = np.sqrt(np.maximum(w, 0))[None, :]          # 잔차 노름에 들어가는 것은 sqrt
    fr, tr = [], []
    for win in wins:
        y = win["y"] * s[0]
        b = np.concatenate([y.real, y.imag])
        for idx, acc in ((allj, fr), (list(win["sup"]), tr)):
            A = np.array([np.concatenate([sig[j, :H, 0] * s[0],
                                          sig[j, :H, 1] * s[0]]) for j in idx]).T
            acc.append(nnls(A, b)[1])
    f, t = float(np.mean(fr)), float(np.mean(tr))
    return t / max(t - f, 1e-12)


# ─────────────────────────────────────────────────────────────────────────
# (B) 전이의 판별력
# ─────────────────────────────────────────────────────────────────────────
def transition_signatures(apps: Sequence[str], pre_s: float = 5.0,
                          post_s: float = 5.0, guard_s: float = 1.0
                          ) -> Dict[str, List[np.ndarray]]:
    """정답 ON 전이에서 관측한 **와트당 Δ 지문**.

    전이 순간에는 그 기기 하나만 변하므로 `Δ고조파 / ΔP` 가 **다른 기기와
    무관하게** 그 기기의 지문이다. 상수를 가정하지 않고 자료에서 읽는다.

    ⚠ 다른 기기가 같이 바뀐 전이는 버린다 — 그러면 귀속이 깨진다.
    """
    ev = load_events()
    out: Dict[str, List[np.ndarray]] = {}
    C = 60      # 사이클/초
    for stem in human_stems(ev):
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        pf = np.asarray(z["power_features"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        iv = ev[stem].get("intervals", {})
        for app, spec in iv.items():
            if app not in apps:
                continue
            j = apps.index(app)
            for (s0, _s1) in spec.get("on", []):
                t = int(round(float(s0) * C))
                a0, a1 = t - int((pre_s + guard_s) * C), t - int(guard_s * C)
                b0, b1 = t + int(guard_s * C), t + int((post_s + guard_s) * C)
                if a0 < 0 or b1 >= len(hc):
                    continue
                if not (ok[a0:a1].all() and ok[b0:b1].all()):
                    continue
                # 다른 기기가 이 창에서 바뀌면 버린다
                oth = [k for k in range(len(apps)) if k != j]
                if np.any(on[a0:a1, oth].mean(0) != on[b0:b1, oth].mean(0)):
                    continue
                dP = float(pf[b0:b1, 0].mean() - pf[a0:a1, 0].mean())
                if dP < 5.0:                       # 너무 작은 전이는 잡음
                    continue
                dI = hc[b0:b1].mean(0)[:H] - hc[a0:a1].mean(0)[:H]
                out.setdefault(app, []).append(dI / dP)
    return out


def dprime(A: np.ndarray, B: np.ndarray) -> float:
    """두 표본군의 판별력 — |중앙 차| / 군내 산포 평균 (규칙 23)."""
    ma, mb = np.median(A, 0), np.median(B, 0)
    sa = 1.4826 * np.median(np.abs(A - ma[None]), 0)
    sb = 1.4826 * np.median(np.abs(B - mb[None]), 0)
    d = np.abs(ma - mb) / np.maximum((sa + sb) / 2, 1e-12)
    return float(np.median(d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt")
    ap.add_argument("--out", default="results/lharm_alt.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG = harmonic_signatures(pool, apps)
    NZ = noise_signature(pool)
    del pool
    from src.run_state_sig_probe import median_signature_scatter
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SD = median_signature_scatter(pool, apps)
    del pool

    print("=" * 88)
    print("L_harm — 결함을 고칠 것인가, 대신할 것을 넣을 것인가")
    print("=" * 88)

    # ── (A) 차수 가중의 상한 ──────────────────────────────────────────
    wins = load_windows(apps, NZ)
    allj = list(range(len(apps)))
    print(f"\n[A] 차수 가중으로 도달 가능한 최선 — 창 {len(wins)}개\n")
    CAND = [
        ("균등 (현행)", np.ones(H)),
        ("홀수차만", np.array([1.0 if h % 2 == 0 else 0.0 for h in range(H)])),
        ("1/tau_h", None),                        # 아래에서 채운다
        ("저차 강조 1/h", np.array([1.0 / (h + 1) for h in range(H)])),
        ("저차 강조 1/h^2", np.array([1.0 / (h + 1) ** 2 for h in range(H)])),
        ("h1,h3 만", np.array([1.0, 0, 1.0] + [0.0] * 12)),
    ]
    from src.model.losses import HARM_DEADZONE_PROFILE
    tau = np.array(HARM_DEADZONE_PROFILE[:H])
    CAND[2] = ("1/tau_h", 1.0 / np.maximum(tau, 1e-6))
    print(f"  {'가중':<22s}{'유효차원':>10s}{'모델오차/판별신호':>20s}")
    print("  " + "-" * 56)
    rowsA = {}
    for name, w in CAND:
        w = w / max(w.max(), 1e-12)
        r = ratio_for_weights(SIG, wins, allj, w)
        eff = float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)) * 2
        rowsA[name] = {"ratio": r, "eff_dim": eff}
        print(f"  {name:<22s}{eff:>10.1f}{r:>19.2f}배")

    # ── (B) 전이의 판별력 ─────────────────────────────────────────────
    T = transition_signatures(apps)
    print(f"\n[B] 전이에서 읽은 와트당 Δ지문 — 기기별 표본 수")
    print("  " + ", ".join(f"{k} {len(v)}" for k, v in sorted(T.items())) or "  (없음)")
    sm = [x for x in SMPS_APPLIANCES if x in T and len(T[x]) >= 3]
    if len(sm) >= 2:
        print(f"\n  SMPS 쌍의 판별력 d′ — **분모는 각각의 전이 간 산포** (규칙 23)\n")
        print(f"  {'쌍':<36s}{'전이 d′':>10s}{'정상상태 d′':>14s}{'배':>8s}")
        print("  " + "-" * 70)
        outB = {}
        for x, y in itertools.combinations(sm, 2):
            A = np.concatenate([np.stack([v.real for v in T[x]]),
                                np.stack([v.imag for v in T[x]])], 1)
            B = np.concatenate([np.stack([v.real for v in T[y]]),
                                np.stack([v.imag for v in T[y]])], 1)
            dt = dprime(A, B)
            i, j = apps.index(x), apps.index(y)
            ci = SIG[i, :, 0] + 1j * SIG[i, :, 1]
            cj = SIG[j, :, 0] + 1j * SIG[j, :, 1]
            ds = float(np.median(np.abs(ci - cj)
                                 / np.maximum((SD[i] + SD[j]) / 2, 1e-9)))
            outB[f"{x}|{y}"] = {"d_trans": dt, "d_steady": ds}
            print(f"  {x[:16]} vs {y[:16]:<18s}{dt:>10.2f}{ds:>14.2f}"
                  f"{dt / max(ds, 1e-9):>8.1f}x")
    else:
        outB = {}
        print("  ⚠ SMPS 전이 표본이 부족하다 — 판정 불가")

    print("\n  [읽는 법] (A) 가 1 아래로 못 가면 차수 가중에는 상한이 있다.")
    print("            (B) 가 정상상태보다 크면 전이가 둘째 판별자다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"weights": rowsA, "transition_dprime": outB,
         "n_transitions": {k: len(v) for k, v in T.items()},
         "_config": {"argv": sys.argv, "args": vars(a)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
