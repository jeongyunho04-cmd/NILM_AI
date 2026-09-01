"""
in-situ 지문 적합 — **복합 파일 자체에서** `sig` 를 푼다 (12.122.11, 2026-09-01)
===============================================================================
`net.harmonic_signatures` 는 **격리 녹화**에서 와트당 페이저를 만든다. 12.122.10 이
그 지문의 전력·전압 의존을 모형화해 봤으나 복합으로 전이되지 않았다 — 격리와
복합의 계통 전압대가 다르고(겹침 0~31%), 배분은 하나도 안 고쳐졌다.

**세 번째 길이 있다: 복합 파일 자체에서 푼다.** 사람 스위칭 로그가 ON 집합을
주므로 지문과 전력을 교대로 풀 수 있다 (bilinear ALS).

    서로 다른 ON 구성 16개, 창 56개
    식 1,680  vs  미지수 413 (지문 270 + 전력 143)      4:1 과결정

[앵커가 결정적이다]
앵커 없이 돌리면 ALS 가 **편향된 초기해에 지문을 맞춰 자기강화한다** — 고정
지문의 NNLS 가 프로젝터로 몰리니 지문도 그쪽으로 끌려가고, 프로젝터가 26.8W 로
과잉 하향된다 (참 47). 아는 물리를 걸어야 한다:

    ① 프로젝터가 켜져 있으면 전력이 46.9W 로 확정이다 (12.122.7 의 참값표)
    ② 저항이 없는 파일(test_7/8/13)은 P관측 = SMPS 합이다

[검증 — 파일 단위 홀드아웃]
자기 자료에 맞춘 것이 아님을 보여야 한다. 한 파일을 빼고 적합해 그 파일로 잰다.

```
55창 LOFO                고정 지문(격리)   in-situ
자유잔차                    0.1375       0.1049   -24%
정답잔차                    0.1953       0.1843    -6%
오차/판별신호                  3.38         2.32   -31%
자유해 프로젝터W (참 47)         83.9         33.8
```

**배분 오차가 +79% -> −28% 로 2.8배 준다.** 다만 순방향 모델 자체는 6%만
좋아지고, `test_6` 은 오히려 나빠진다 (비 1.74 -> 4.66).

    python -m src.run_fit_insitu_sig                    # 적합 + LOFO 보고
    python -m src.run_fit_insitu_sig --out results/sig_insitu.npz

[규칙 1 — 이것은 실측에서 적합했다]
격리 통계가 아니다. 그래서 규칙 1 을 피한다. 다만 **사람 라벨이 있는 5파일에만**
의존하므로 그 파일들의 구성 다양성이 곧 이 지문의 한계다.

[규칙 14 — 안 잡는 것]
홀수차만 푼다 (짝수차는 12.72 가 계측 인공물로 확정). 그리고 창 60초 평균이라
전이 순간의 지문 변화는 안 들어간다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.power_ref import REFERENCE_W
from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import HUMAN_ON_DEFAULT_STEMS
from src.preprocessing import load_nilm_npz
from src.preprocessing.file_registry import NOISE_FLOOR_EXTERNAL_W

EVAL_DIR = "processed_data/composite_eval"
WINDOW = 3600
ODD = np.arange(0, 15, 2)

#: 저항이 없어 P관측 = SMPS 합이 성립하는 파일 (앵커 ②)
SMPS_ONLY_STEMS = ("test_7", "test_8", "test_13")

#: 안 켜진 기기를 사전값으로 끌어당기는 릿지. 0 이면 그 기기 지문이 폭주한다.
RIDGE = 1e-3


def collect(apps: Sequence[str], stems: Sequence[str], hs: np.ndarray,
            nz: np.ndarray) -> List[dict]:
    ev = load_events()
    pj = apps.index("beam_projector")
    NZ = nz[:, 0] + 1j * nz[:, 1]
    out: List[dict] = []
    for stem in stems:
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"]); pf = np.asarray(z["power_features"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        for k in range(0, len(i) - WINDOW, WINDOW):
            sl = i[k:k + WINDOW]
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if not len(sup):
                continue
            out.append({
                "stem": stem, "sup": sup,
                "y": (hc[sl].mean(0) - NZ)[ODD] / hs[ODD],
                "pobs": float(pf[sl, 0].mean()),
                "pj_on": bool(on[sl, pj].mean() > 0.5),
                "smps_only": stem in SMPS_ONLY_STEMS,
            })
    return out


def _cols(S: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return np.array([np.concatenate([S[j][ODD].real, S[j][ODD].imag]) for j in idx]).T


def solve_powers(S: np.ndarray, w: dict, pj: int, anchor: bool = True):
    """지문 고정, 전력을 푼다. `anchor` 면 프로젝터를 고정하고 총전력 식을 붙인다."""
    sup = list(w["sup"])
    b = np.concatenate([w["y"].real, w["y"].imag])
    if not anchor:
        x, r = nnls(_cols(S, sup), b)
        return np.asarray(x), r
    fixed: Dict[int, float] = {}
    if pj in sup and "beam_projector" in REFERENCE_W:
        fixed[pj] = REFERENCE_W["beam_projector"][0]
        b = b - fixed[pj] * _cols(S, [pj])[:, 0]
    free = [j for j in sup if j not in fixed]
    out = np.zeros(len(sup))
    for j, v in fixed.items():
        out[sup.index(j)] = v
    if not free:
        return out, float(np.linalg.norm(b))
    Af = _cols(S, free)
    if w["smps_only"]:
        # 총전력 식 한 줄. 가중은 고조파 잔차와 규모를 맞춘다
        g = np.linalg.norm(Af) / max(np.sqrt(len(free)) * 50.0, 1e-9)
        Af = np.vstack([Af, g * np.ones((1, len(free)))])
        b = np.concatenate([b, [g * (w["pobs"] - NOISE_FLOOR_EXTERNAL_W - sum(fixed.values()))]])
    xf, r = nnls(Af, b)
    for j, v in zip(free, xf):
        out[sup.index(j)] = v
    return out, r


#: 홀수차 인덱스 — 노름을 재는 자리
_ODD = np.arange(0, 15, 2)


def renorm(S: np.ndarray, prior: np.ndarray,
           update: Optional[Sequence[int]] = None) -> np.ndarray:
    """각 기기의 와트당 노름을 **격리값으로 되돌린다.** 형상만 남기고 척도를 뺀다.

    [왜 필요한가 — 척도 부정성]
    ALS 에서 `sig_i x c` 와 `P_i / c` 는 같은 곱을 낸다. 프로젝터는 전력이
    46.9W 로 고정돼 척도가 잡히지만 **충전기·미니PC 는 앵커가 없다.**
    그래서 2026-09-01 첫 적합에서 충전기 지문이 x0.69 로 줄었고,
    12.120.3 대로 **가장 싼 배수구가 프로젝터에서 충전기로 바뀌었다:**

        와트당 노름 (손실 좌표계)   격리      in-situ
        beam_projector          0.1215   0.1508   1위 -> 2위
        laptop_charger          0.1354   0.0931   2위 -> **1위**

    결과는 `test.2` 유령 0.16 -> 22.94W, `test3` 0.20 -> 24.60W —
    **전부 충전기였다.** 적합 집합 밖 파일에서만 터진 이유도 이것이다.

    [왜 노름을 격리값으로 묶어도 되는가]
    12.122.3 이 복합의 단독 구간을 격리 지문과 맞대어 **크기가 5~8% 안에서
    전이된다**고 쟀다. 못 믿을 것은 형상(어느 차수에 얼마)이지 척도가 아니다.
    """
    out = S.copy()
    idx = range(len(S)) if update is None else update
    for j in idx:
        a = np.linalg.norm(S[j][_ODD])
        b = np.linalg.norm(prior[j][_ODD])
        if a > 1e-12 and b > 1e-12:
            out[j] = S[j] * (b / a)
    return out


def solve_sigs(P: List[np.ndarray], wins: List[dict], prior: np.ndarray,
               K: int, ridge: float = RIDGE,
               update: Optional[Sequence[int]] = None) -> np.ndarray:
    """전력 고정, 지문을 차수마다 복소 최소자승으로 푼다.

    `update` 를 주면 그 열만 갱신하고 나머지는 사전값 그대로 둔다.
    **저항 3종은 h>=3 지문이 거의 영이라 이 계에서 영방향이다** — 릿지로도
    못 잡아서 잔차를 통째로 흡수한다 (2026-09-01 에 핫플 x11.8, 오븐 x6.9).
    거기다 오븐 라벨이 세션 단위라(12.119) 전력 자체가 틀려서 지문이 그것을
    보상한다. 겨냥은 SMPS 배분이므로 **SMPS 열만 푸는 것이 기본이다.**
    """
    S = prior.copy()
    for oi in range(len(ODD)):
        A = np.zeros((len(wins), K))
        b = np.zeros(len(wins), complex)
        for r, (w, p) in enumerate(zip(wins, P)):
            A[r, w["sup"]] = p
            b[r] = w["y"][oi]
        G = A.T @ A
        lam = ridge * max(G.diagonal().mean(), 1e-12)
        sol = np.linalg.solve(G + lam * np.eye(K), A.T @ b + lam * prior[:, ODD[oi]])
        if update is None:
            S[:, ODD[oi]] = sol
        else:
            S[update, ODD[oi]] = sol[list(update)]
    return S


def fit(wins: List[dict], prior: np.ndarray, pj: int, K: int,
        n_iter: int = 25, anchor: bool = True,
        update: Optional[Sequence[int]] = None,
        keep_norm: bool = True, ridge: float = RIDGE) -> np.ndarray:
    S = prior.copy()
    P = [solve_powers(S, w, pj, anchor)[0] for w in wins]
    for _ in range(n_iter):
        S = solve_sigs(P, wins, prior, K, ridge=ridge, update=update)
        if keep_norm:
            S = renorm(S, prior, update)
        P = [solve_powers(S, w, pj, anchor)[0] for w in wins]
    return S


def residual(S: np.ndarray, w: dict, K: int, free: bool):
    sup = list(range(K)) if free else list(w["sup"])
    b = np.concatenate([w["y"].real, w["y"].imag])
    x, r = nnls(_cols(S, sup), b)
    full = np.zeros(K); full[sup] = x
    return r / max(np.linalg.norm(b), 1e-12), full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/adapt_ovh.pt", help="기기 목록용")
    ap.add_argument("--stems", nargs="+", default=list(HUMAN_ON_DEFAULT_STEMS))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--no-anchor", action="store_true",
                    help="앵커를 끈다. **자기강화로 무너지는 것을 보는 절제용**")
    ap.add_argument("--all-appliances", action="store_true",
                    help="9종 전부 갱신한다. 기본은 SMPS 3종만 — 저항은 h>=3 이 "
                         "거의 영이라 잔차를 흡수해 x7~12 로 튄다")
    ap.add_argument("--ridge", type=float, default=RIDGE,
                    help="지문을 격리 사전값으로 끌어당기는 수축. 크면 형상이 덜 "
                         "움직인다. **적합 집합 밖 파일의 유령이 이것으로 조절된다**")
    ap.add_argument("--free-norm", action="store_true",
                    help="노름을 격리값에 묶지 않는다. **척도 부정성으로 무너지는 것을 "
                         "보는 절제용** — 충전기 지문이 x0.69 로 줄어 가장 싼 배수구가 "
                         "되고 적합 집합 밖 파일에서 유령이 22~25W 튄다")
    ap.add_argument("--out", default="results/sig_insitu.npz")
    a = ap.parse_args()

    apps = torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"]
    K = len(apps); pj = apps.index("beam_projector")
    from src.model.net import harmonic_scales, harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    S0 = harmonic_signatures(pool, apps); hs = harmonic_scales(pool, apps)
    nz = noise_signature(pool)
    del pool
    prior = np.array([(S0[j, :, 0] + 1j * S0[j, :, 1]) / hs for j in range(K)])

    from src.model.realdata import SMPS_APPLIANCES
    upd = None if a.all_appliances else [apps.index(x) for x in SMPS_APPLIANCES if x in apps]
    wins = collect(apps, a.stems, hs, nz)
    print("=" * 84)
    print(f"in-situ 지문 적합 — 창 {len(wins)}개, {len(a.stems)}파일, "
          f"앵커 {'끔' if a.no_anchor else '켬'}, "
          f"갱신 {'9종 전부' if a.all_appliances else 'SMPS 3종만'}, "
          f"노름 {'자유' if a.free_norm else '격리값 고정'}")
    print("=" * 84)

    # ── 파일 단위 홀드아웃 검증 ──────────────────────────────────────────
    print(f"\n{'홀드아웃':10s}{'창':>5s} | {'고정 지문(격리)':^24s} | {'in-situ (LOFO)':^24s}")
    print(f"{'':10s}{'':>5s} | {'자유':>8s}{'정답':>8s}{'비':>6s} | {'자유':>8s}{'정답':>8s}{'비':>6s}")
    print("-" * 76)
    agg = {k: [] for k in ("f0", "t0", "f1", "t1", "pj0", "pj1")}
    for held in a.stems:
        tr = [w for w in wins if w["stem"] != held]
        te = [w for w in wins if w["stem"] == held]
        if not te or not tr:
            continue
        S = fit(tr, prior, pj, K, a.iters, not a.no_anchor, upd,
                keep_norm=not a.free_norm, ridge=a.ridge)
        loc = {k: [] for k in ("f0", "t0", "f1", "t1")}
        for w in te:
            f0, x0 = residual(prior, w, K, True); t0, _ = residual(prior, w, K, False)
            f1, x1 = residual(S, w, K, True);     t1, _ = residual(S, w, K, False)
            for k, v in (("f0", f0), ("t0", t0), ("f1", f1), ("t1", t1)):
                loc[k].append(v); agg[k].append(v)
            if w["pj_on"]:
                agg["pj0"].append(x0[pj]); agg["pj1"].append(x1[pj])
        m = {k: np.mean(v) for k, v in loc.items()}
        print(f"{held:10s}{len(te):>5d} | {m['f0']:8.4f}{m['t0']:8.4f}"
              f"{m['t0'] / m['f0']:6.2f} | {m['f1']:8.4f}{m['t1']:8.4f}"
              f"{m['t1'] / m['f1']:6.2f}")
    F0, T0, F1, T1 = (np.mean(agg[k]) for k in ("f0", "t0", "f1", "t1"))
    print("-" * 76)
    print(f"{'전체':10s}{len(agg['f0']):>5d} | {F0:8.4f}{T0:8.4f}{T0 / F0:6.2f}"
          f" | {F1:8.4f}{T1:8.4f}{T1 / F1:6.2f}")
    r0, r1 = T0 / max(T0 - F0, 1e-9), T1 / max(T1 - F1, 1e-9)
    print(f"\n  오차/판별신호            고정 {r0:.2f}  ->  in-situ {r1:.2f}")
    print(f"  자유해 프로젝터W (참 {REFERENCE_W['beam_projector'][0]:.0f})  "
          f"고정 {np.mean(agg['pj0']):.1f}  ->  in-situ {np.mean(agg['pj1']):.1f}")

    # ── 전 파일로 적합해 저장 (이것이 산출물) ────────────────────────────
    S = fit(wins, prior, pj, K, a.iters, not a.no_anchor, upd,
            keep_norm=not a.free_norm, ridge=a.ridge)
    sig = np.zeros_like(S0)
    for j in range(K):
        c = S[j] * hs                      # 손실 좌표계 -> 와트당 원단위
        sig[j, :, 0], sig[j, :, 1] = c.real, c.imag
    # **짝수차는 격리 지문 그대로 둔다** — 홀수차만 풀었다 (12.72)
    sig[:, 1::2, :] = S0[:, 1::2, :]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, sig=sig, appliances=np.array(apps),
             stems=np.array(a.stems), n_windows=len(wins),
             lofo={"resid_free_fixed": F0, "resid_true_fixed": T0,
                   "resid_free_insitu": F1, "resid_true_insitu": T1,
                   "ratio_fixed": r0, "ratio_insitu": r1}.__repr__())
    print(f"\n저장: {a.out}")

    print(f"\n{'기기':18s}{'격리 |sig|':>12s}{'in-situ |sig|':>15s}{'비':>8s}   (h=3)")
    for j, app in enumerate(apps):
        a3 = abs(S0[j, 2, 0] + 1j * S0[j, 2, 1]); b3 = abs(sig[j, 2, 0] + 1j * sig[j, 2, 1])
        print(f"{app:18s}{a3 * 1e3:12.4f}{b3 * 1e3:15.4f}{b3 / max(a3, 1e-12):8.2f}")


if __name__ == "__main__":
    main()
