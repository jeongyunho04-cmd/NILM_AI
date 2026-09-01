"""상태 조건 지문이 배분을 살리는가 — **재학습 없이** 판정한다 (12.124)

배경
----
12.122.3 이 오배분의 기제를 특정했다. `harmonic_signatures` 는 통전 구간
(`P > 0.5 x steady_p90`)의 **중앙값 하나**를 낸다. 그런데

    프로젝터와의 와트당 형상 차이       h3     h5     h7     h9    h11
      충전기 고전력 (지문이 여기)      0.9%   0.3%   3.8%   6.7%  22.8%
      충전기 저전력 (문턱에 잘림)      6.6%  15.5%  29.8%  53.7% 126.2%

**지문이 하필 둘이 같아지는 상태의 것이다.** 문턱 아래 표본이 충전기 15.3%,
미니PC 46.0% 다.

그래서 "상태별 지문" 이 배분을 고칠 후보다. 12.122.18 이 상태가 지문 변동의
42~45% 를 설명한다고 쟀고, 규칙 31 이 **`sig(state)` 는 관측에서 결정되므로
자유도가 아니다** 로 이 길만 열어 뒀다 (`sig(P)` 는 순환이라 닫힘).

이 스크립트가 재는 것
--------------------
(1) **판별력 d'** — 프로젝터 vs 충전기/미니PC. 중앙값 지문 vs 상태별 지문.
    규칙 23 — 분모(기기내 산포)를 반드시 같이 낸다.

(2) **12.122.2 의 자를 다시 푼다.** 사람 라벨 5파일 55창에서

        자유    min_{P>=0} ‖y − ΣP·sig‖              9종 전부
        정답    같은 것을 사람이 켰다고 적은 기기로만

    를 지문 변형마다 풀고 **모델오차 / 판별신호** 를 낸다. 12.122.2 가
    현행 지문에서 2.8배로 쟀고, **이 값이 1 아래로 안 내려가면 고조파
    잔차를 줄이는 어떤 절차도 오답으로 간다.**

(3) ⚠ **규칙 31 가드.** 상태별 지문은 상태를 고를 자유를 준다. 그래서
    정답만 좋아지는 게 아니라 **자유해도 같이 좋아진다.** 둘을 나란히 내고,
    자유해가 더 좋아지면 그것은 개선이 아니라 식별성 손실이다.

    python -m src.run_state_sig_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse
import itertools
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import HUMAN_ON_DEFAULT_STEMS, SMPS_APPLIANCES
from src.preprocessing import load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW

H = 15
ON_STATES = (1, 2)          # 0 은 OFF/대기 — standby_sig 가 따로 맡는다


# ─────────────────────────────────────────────────────────────────────────
# 지문 만들기
# ─────────────────────────────────────────────────────────────────────────
def per_state_signatures(pool, appliances: Sequence[str], *, threshold: bool = True
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K, S, H, 2) 상태별 와트당 지문, (K, S) 표본수, (K, S, H) 기기내 산포.

    `threshold=False` 면 `harmonic_signatures` 의 `P > 0.5 x steady` 문턱을
    걷는다 — 12.122.3 이 지목한 그 문턱이다. 상태로 가르면 저전력 상태가
    자기 지문을 갖게 되므로 문턱의 원래 목적(부수 상태 오염 방지)은
    상태 분리가 대신한다.
    """
    K, S = len(appliances), max(ON_STATES) + 1
    sig = np.zeros((K, S, H, 2), np.float32)
    n = np.zeros((K, S), np.int64)
    sd = np.zeros((K, S, H), np.float32)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app) if threshold else 0.0
        for st in ON_STATES:
            cs, ps = [], []
            for a in acts:
                m = (a.state_id == st) & (a.target_power_w > max(thr, 1.0))
                if m.any():
                    cs.append(a.net_harmonics_complex[m])
                    ps.append(a.target_power_w[m])
            if not cs:
                continue
            per_w = np.concatenate(cs) / np.maximum(np.concatenate(ps)[:, None], 1e-6)
            sig[j, st, :, 0] = np.median(per_w.real, 0)
            sig[j, st, :, 1] = np.median(per_w.imag, 0)
            n[j, st] = len(per_w)
            # 기기내 산포 = 판별력의 **분모** (규칙 23). 복소 편차의 로버스트 폭.
            dev = per_w - (sig[j, st, :, 0] + 1j * sig[j, st, :, 1])[None]
            sd[j, st] = 1.4826 * np.median(np.abs(dev), 0)
    return sig, n, sd


def median_signature_scatter(pool, appliances: Sequence[str]) -> np.ndarray:
    """현행 중앙값 지문의 기기내 산포 (K, H) — d' 의 분모."""
    out = np.zeros((len(appliances), H), np.float32)
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        cs, ps = [], []
        for a in acts:
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                cs.append(a.net_harmonics_complex[m]); ps.append(a.target_power_w[m])
        if not cs:
            continue
        per_w = np.concatenate(cs) / np.maximum(np.concatenate(ps)[:, None], 1e-6)
        med = np.median(per_w.real, 0) + 1j * np.median(per_w.imag, 0)
        out[j] = 1.4826 * np.median(np.abs(per_w - med[None]), 0)
    return out


# ─────────────────────────────────────────────────────────────────────────
# 창 모으기 (run_fit_insitu_sig.collect 와 같은 55창, 전 15차)
# ─────────────────────────────────────────────────────────────────────────
def collect_windows(apps: Sequence[str], nz: np.ndarray) -> List[dict]:
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out: List[dict] = []
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
            if not len(sup):
                continue
            out.append({"stem": stem, "sup": sup, "y": hc[sl].mean(0)[:H] - NZ})
    return out


# ─────────────────────────────────────────────────────────────────────────
# NNLS — 12.122.2 의 (6)절
# ─────────────────────────────────────────────────────────────────────────
def _design(cols: Sequence[np.ndarray]) -> np.ndarray:
    """복소 열들을 (2H, n) 실수 설계행렬로."""
    return np.array([np.concatenate([c.real, c.imag]) for c in cols]).T


def solve_flat(sig2: np.ndarray, idx: Sequence[int], b: np.ndarray) -> float:
    """지문 하나짜리 (K, H, 2)."""
    if not len(idx):
        return float(np.linalg.norm(b))
    A = _design([sig2[j, :, 0] + 1j * sig2[j, :, 1] for j in idx])
    _, r = nnls(A, b)
    return float(r)


def solve_state(sigS: np.ndarray, idx: Sequence[int], b: np.ndarray,
                free_states: Sequence[int]) -> float:
    """상태별 지문 (K, S, H, 2). `free_states` 기기만 상태를 **고를 수 있다**.

    ⚠ 이것은 **오라클 상한**이다. 상태를 고르는 자유가 있으므로 정답도
    자유해도 같이 좋아진다 — 그래서 둘을 나란히 봐야 한다 (규칙 31).
    """
    if not len(idx):
        return float(np.linalg.norm(b))
    idx = list(idx)
    choosable = [j for j in idx if j in free_states]
    fixed = [j for j in idx if j not in free_states]
    best = np.inf
    for combo in itertools.product(ON_STATES, repeat=len(choosable)):
        cols = []
        for j in fixed:
            st = ON_STATES[-1] if np.any(sigS[j, ON_STATES[-1]]) else ON_STATES[0]
            cols.append(sigS[j, st, :, 0] + 1j * sigS[j, st, :, 1])
        for j, st in zip(choosable, combo):
            if not np.any(sigS[j, st]):
                break
            cols.append(sigS[j, st, :, 0] + 1j * sigS[j, st, :, 1])
        else:
            _, r = nnls(_design(cols), b)
            best = min(best, float(r))
    return best if np.isfinite(best) else solve_flat(sigS[:, ON_STATES[-1]], idx, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt", help="기기 목록용")
    ap.add_argument("--sig-insitu", default="results/sig_insitu.npz")
    ap.add_argument("--out", default="results/state_sig_probe.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.net import harmonic_signatures, harmonic_scales, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG_MED = harmonic_signatures(pool, apps)
    HSC = harmonic_scales(pool, apps)
    NZ = noise_signature(pool)
    SD_MED = median_signature_scatter(pool, apps)
    SIG_ST, N_ST, SD_ST = per_state_signatures(pool, apps, threshold=True)
    SIG_ST_NT, N_ST_NT, SD_ST_NT = per_state_signatures(pool, apps, threshold=False)
    del pool

    SIG_INS = np.asarray(np.load(a.sig_insitu, allow_pickle=True)["sig"], np.float32)
    pj = apps.index("beam_projector")
    smps = [apps.index(x) for x in SMPS_APPLIANCES if x in apps]

    print("=" * 88)
    print("상태 조건 지문 진단 — 12.122.3 이 지목한 기제를 직접 친다")
    print("=" * 88)

    # ── (1) 판별력 d' ─────────────────────────────────────────────────
    print("\n[1] 프로젝터와의 판별력 d' = |sig_i − sig_pj| / 두 기기 산포의 평균")
    print("    (규칙 23 — 분모 없이 재면 결론이 뒤집힌다)\n")
    ODD = np.arange(0, H, 2)
    hdr = "  " + " ".join(f"h{h+1:<5d}" for h in ODD)
    for name in ("laptop_charger", "minipc"):
        if name not in apps:
            continue
        j = apps.index(name)
        print(f"  --- {name} vs beam_projector ---")
        med_i = SIG_MED[j, :, 0] + 1j * SIG_MED[j, :, 1]
        med_p = SIG_MED[pj, :, 0] + 1j * SIG_MED[pj, :, 1]
        d_med = np.abs(med_i - med_p) / np.maximum((SD_MED[j] + SD_MED[pj]) / 2, 1e-9)
        print(hdr)
        print("  현행 중앙값 " + " ".join(f"{v:<6.2f}" for v in d_med[ODD]))
        for st in ON_STATES:
            if N_ST[j, st] == 0:
                continue
            stp = ON_STATES[-1] if N_ST[pj, ON_STATES[-1]] else ON_STATES[0]
            si = SIG_ST[j, st, :, 0] + 1j * SIG_ST[j, st, :, 1]
            sp = SIG_ST[pj, stp, :, 0] + 1j * SIG_ST[pj, stp, :, 1]
            d = np.abs(si - sp) / np.maximum((SD_ST[j, st] + SD_ST[pj, stp]) / 2, 1e-9)
            print(f"  상태{st} (n={N_ST[j, st]:>6d}) " + " ".join(f"{v:<6.2f}" for v in d[ODD]))
        print()

    # ── (2)(3) 12.122.2 의 자 ─────────────────────────────────────────
    wins = collect_windows(apps, NZ)
    print(f"[2] 12.122.2 의 자 — 사람 라벨 5파일 {len(wins)}창, 모델도 손실도 없다\n")
    VARIANTS = [
        ("현행 격리 중앙값", lambda idx, b: solve_flat(SIG_MED, idx, b)),
        ("in-situ r0.1", lambda idx, b: solve_flat(SIG_INS, idx, b)),
        ("상태별 (문턱 유지)", lambda idx, b: solve_state(SIG_ST, idx, b, smps)),
        ("상태별 (문턱 해제)", lambda idx, b: solve_state(SIG_ST_NT, idx, b, smps)),
    ]
    allj = list(range(len(apps)))
    rows: Dict[str, dict] = {}
    for name, fn in VARIANTS:
        free, true = [], []
        for w in wins:
            b = np.concatenate([w["y"].real, w["y"].imag])
            free.append(fn(allj, b))
            true.append(fn(list(w["sup"]), b))
        f, t = float(np.mean(free)), float(np.mean(true))
        rows[name] = {"free": f, "true": t, "signal": t - f,
                      "ratio": t / max(t - f, 1e-12)}
        print(f"  {name:<20s} 자유 {f:.4f}   정답 {t:.4f}   "
              f"판별신호 {t - f:.4f}   **모델오차/판별신호 {t / max(t - f, 1e-12):>6.2f}배**")

    base = rows["현행 격리 중앙값"]
    print("\n[3] 규칙 31 가드 — 자유해가 정답보다 더 좋아졌는가")
    print(f"  {'변형':<20s}{'Δ정답':>10s}{'Δ자유':>10s}{'판정':>34s}")
    print("  " + "-" * 76)
    for name, r in rows.items():
        if name == "현행 격리 중앙값":
            continue
        dt, df = r["true"] - base["true"], r["free"] - base["free"]
        if dt >= -1e-6:
            v = "정답잔차가 안 줄었다"
        elif df <= dt:
            v = "**식별성 손실** — 자유해가 더 줄었다"
        else:
            v = "정답이 더 줄었다 — 판별에 유리"
        print(f"  {name:<20s}{dt:>+10.4f}{df:>+10.4f}{v:>34s}")

    print("\n  **모델오차/판별신호 가 1 아래로 안 내려가면** 고조파 잔차를 줄이는")
    print("  어떤 절차도(손실·NNLS·후처리) 오답으로 간다 — 12.122.2 의 벽이다.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"variants": rows, "n_windows": len(wins),
         "_config": {"argv": sys.argv, "args": vars(a)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
