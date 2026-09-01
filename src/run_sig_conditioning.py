"""
지문 조건화 진단 — SMPS 3종 (SMPS_PLAN 4.1절, 2026-08-31 개정)
=================================================================
`SMPS_PLAN` 은 이 스크립트가 다섯 가지를 낸다고 적었다. 그중 ② 의 **자가 틀렸다.**

    계획이 적은 ②   ‖sig_i − sig_j‖                     <- 분모가 없다
    실제로 필요한 ②  ‖sig_i − sig_j‖ / 창간 산포          <- d'

**둘이 정반대 결론을 낸다.** 분모 없이 재면 차수별로 8~24배 갈려서
*"균등 무게가 어긋나 있다 -> 4.2 로 간다"* 가 나오고, 분모를 넣으면 h=3~15 가
1.1배로 평평해서 *"이 방향을 닫는다"* 가 나온다. 기기 안의 창간 산포가 기기
사이의 차이와 **같은 속도로** 커지기 때문이다:

    프로젝터 vs 충전기, 와트당 (60초 창, 격리 녹화)
      차수          h1     h3     h5     h7     h9    h11    h13    h15
      기기간 차이  14.5%  14.5%  23.9%  35.1%  47.5%  69.2% 108.9% 169.4%
      기기내 산포   2.5%   6.4%  10.7%  16.4%  23.1%  33.9%  61.2%  72.8%
      ------------------------------------------------------------------
      d'           4.59   1.61   1.58   1.55   1.48   1.47   1.41   1.52

그래서 **두 자를 나란히 찍는다.** 어느 쪽으로 읽었는지가 결론을 가르므로
하나만 내면 안 된다.

    python -m src.run_sig_conditioning

내는 것 (SMPS_PLAN 4.1 의 다섯 항목):

    (1) harm_scale     차수별 정규화가 무게를 실제로 어떻게 바꾸는가
    (2) 차수별 판별력   **분모 없는 것과 있는 것을 나란히**. 크기만 / 복소 둘 다
    (3) kappa          차수 부분집합별 조건수
    (4) leak W/W       저항 1W 오예측이 SMPS 에 붙이는 W — 해석 + **실측 in-situ**
    (5) 제안 무게       (2) 에 비례하는 harm_mask 후보

[규칙 1 — 이것은 격리 통계다]
(1)~(3) 과 (4) 의 앞쪽은 격리 녹화에서 만든 지문의 **대수적 성질**이다.
"모델이 무엇을 써야 한다" 를 여기서 정하면 안 된다. (4) 의 in-situ 만 실측이고,
판정은 전부 실측 재채점으로 해야 한다.

[규칙 14 — 안 잰 것]
이 스크립트는 `harmonic_signatures` 가 만든 **중앙값 지문**을 쓴다. 창마다의
지문 변동은 (2) 의 분모에만 들어가고 (3)(4) 에는 안 들어간다. 즉 kappa 와
leak 은 **지문이 정확하다는 가정 아래의 값**이라 낙관 쪽이다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import glob
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.net import harmonic_scales, harmonic_signatures, noise_signature, standby_signatures
from src.model.realdata import SMPS_APPLIANCES
from src.preprocessing import load_nilm_npz

NPZ_DIR = "processed_data/npz"
EVAL_DIR = "processed_data/composite_eval"
WINDOW_CYCLES = 3600                  # 60초 @ 60Hz — 모델의 창과 같다
PAIRS = (("beam_projector", "laptop_charger"),
         ("minipc", "beam_projector"),
         ("minipc", "laptop_charger"))
ODD = np.arange(0, 15, 2)             # 0-based -> 1,3,5..15 차


# ── 공통 ────────────────────────────────────────────────────────────────────
def complex_sig(sig: np.ndarray) -> np.ndarray:
    """(K,H,2) -> (K,H) 복소 와트당 지문."""
    return sig[:, :, 0] + 1j * sig[:, :, 1]


def loss_basis(S: np.ndarray, hs: np.ndarray, cols: Sequence[int],
               orders: np.ndarray) -> np.ndarray:
    """손실 단위 실수 설계행렬 (2*len(orders), len(cols)).

    `L_harm` 이 보는 좌표계 그대로다 — `|pred−obs| / harm_scale`.
    """
    out = []
    for j in cols:
        v = S[j][orders] / hs[orders]
        out.append(np.concatenate([v.real, v.imag]))
    return np.asarray(out).T


def isolated_windows(app: str, window: int = WINDOW_CYCLES) -> np.ndarray:
    """격리 녹화에서 60초 창마다의 **와트당** 복소 지문 (n, 15).

    같은 기기의 녹화 여러 개를 다 쓴다 — 세션 간 표류가 산포에 들어가야
    배포에서 볼 산포가 된다 (규칙 5).
    """
    C: List[np.ndarray] = []
    for f in sorted(glob.glob(str(Path(NPZ_DIR) / f"{app}*.npz"))):
        z = load_nilm_npz(f)
        hc = np.asarray(z["harmonics_complex"])
        on = np.asarray(z["is_on"]).astype(bool)
        ok = np.asarray(z["is_valid"]).astype(bool)
        p = np.asarray(z["p_denoised_w"])
        idx = np.flatnonzero(on & ok & (p > 1.0))
        for k in range(0, len(idx) - window, window):
            s = idx[k:k + window]
            C.append(hc[s].mean(0) / p[s].mean())
    return np.asarray(C)


# ── (1) harm_scale ──────────────────────────────────────────────────────────
def sec_harm_scale(hs: np.ndarray) -> None:
    print("\n" + "=" * 88)
    print("(1) harm_scale — 차수별 정규화가 무게를 어떻게 바꾸는가")
    print("=" * 88)
    print("  `err = |pred−obs| / harm_scale[h]` 라 **가중치는 1/harm_scale 이다.**")
    print("  harm_mask 는 전부 1 이지만 실효 무게는 균등이 아니다.\n")
    print("  차수 " + "".join(f"{h:>8d}" for h in range(1, 16)))
    print("  A    " + "".join(f"{x * 1e3:8.2f}" for x in hs) + "   (mA)")
    print("  1/A  " + "".join(f"{hs[0] / x:8.2f}" for x in hs) + "   (h=1 대비 실효 무게)")
    ev, od = hs[1::2], hs[0::2]
    print(f"\n  홀수차 {od.min() * 1e3:.1f}~{od.max() * 1e3:.1f} mA  vs  "
          f"짝수차 {ev.min() * 1e3:.1f}~{ev.max() * 1e3:.1f} mA "
          f"-> 짝수차 오차가 {od.mean() / ev.mean():.0f}배 증폭된다")
    print("  ⚠ 12.72 가 짝수차를 **계측 인공물**로 확정했고 같은 기기 녹화 사이에서")
    print("    1.3~1.8배 흔들린다고 쟀다. 그 자리에 가장 큰 무게가 걸려 있다.")
    print("    `--harm-odd-only` 가 이미 있는데 단일 변수로 잰 적이 없다 (12.75.5).")


# ── (2) 차수별 판별력 ────────────────────────────────────────────────────────
def sec_discrimination(S: np.ndarray, hs: np.ndarray, apps: List[str],
                       win: Dict[str, np.ndarray]) -> Dict[str, dict]:
    print("\n" + "=" * 88)
    print("(2) 차수별 판별력 — **분모 없는 자와 있는 자를 나란히**")
    print("=" * 88)
    idx = {a: i for i, a in enumerate(apps)}
    res: Dict[str, dict] = {}

    print("\n  [2a] 계획이 적은 자: ‖sig_i−sig_j‖ / harm_scale   (분모 없음, 홀수차)")
    print("       " + "".join(f"{'h' + str(h):>8s}" for h in range(1, 16, 2)) + "   max/min")
    for a, b in PAIRS:
        d = (np.abs(S[idx[a]] - S[idx[b]]) / hs)[ODD] * 1e3
        res.setdefault(f"{a}|{b}", {})["naive"] = d.tolist()
        print(f"  {a[:8]:>8s}/{b[:8]:<8s}" + "".join(f"{x:8.2f}" for x in d)
              + f"   {d.max() / d.min():6.2f}")

    print("\n  [2b] 분모를 넣은 자: d' = |Δ와트당| / 창간 산포   (60초 창, 격리 녹화)")
    print("       " + "".join(f"{'h' + str(h):>8s}" for h in range(1, 16, 2)) + "   max/min")
    for a, b in PAIRS:
        A, B = win.get(a), win.get(b)
        if A is None or B is None or len(A) < 3 or len(B) < 3:
            print(f"  {a[:8]:>8s}/{b[:8]:<8s}   창이 모자라 못 잰다")
            continue
        num = np.abs(A.mean(0) - B.mean(0))
        den = np.sqrt((A.real.var(0) + A.imag.var(0) + B.real.var(0) + B.imag.var(0)) / 2)
        d = (num / np.maximum(den, 1e-15))[ODD]
        res.setdefault(f"{a}|{b}", {})["dprime"] = d.tolist()
        print(f"  {a[:8]:>8s}/{b[:8]:<8s}" + "".join(f"{x:8.2f}" for x in d)
              + f"   {d.max() / d.min():6.2f}   (n={len(A)},{len(B)})")

    print("\n  [2c] 왜 갈리는가 — 분자와 분모가 같은 속도로 큰다")
    print("       " + "".join(f"{'h' + str(h):>8s}" for h in range(1, 16, 2)))
    a, b = PAIRS[0]
    A, B = win.get(a), win.get(b)
    if A is not None and B is not None and len(A) >= 3 and len(B) >= 3:
        num = np.abs(A.mean(0) - B.mean(0)) / ((np.abs(A.mean(0)) + np.abs(B.mean(0))) / 2)
        print("  기기간 차이" + "".join(f"{x * 100:7.1f}%" for x in num[ODD]))
        for nm, C in ((a, A), (b, B)):
            r = np.sqrt(C.real.var(0) + C.imag.var(0)) / np.maximum(np.abs(C.mean(0)), 1e-15)
            print(f"  {nm[:9]:>9s} 내" + "".join(f"{x * 100:7.1f}%" for x in r[ODD]))
    return res


# ── (3) kappa ───────────────────────────────────────────────────────────────
def sec_kappa(S: np.ndarray, hs: np.ndarray, apps: List[str]) -> Dict[str, float]:
    print("\n" + "=" * 88)
    print("(3) kappa — 차수 부분집합별 조건수 (SMPS 3종, 손실 단위)")
    print("=" * 88)
    idx = {a: i for i, a in enumerate(apps)}
    cols = [idx[a] for a in SMPS_APPLIANCES]
    subsets = {
        "h=1..15 전부": np.arange(15),
        "홀수차 1,3..15": ODD,
        "홀수차 h>=3": np.arange(2, 15, 2),
        "홀수차 h>=5": np.arange(4, 15, 2),
        "h=3,5,7 만": np.array([2, 4, 6]),
        "h=9..15 만": np.array([8, 10, 12, 14]),
    }
    out: Dict[str, float] = {}
    print(f"\n  {'차수 집합':16s}{'kappa':>10s}{'sigma_min':>12s}   해석")
    for lbl, o in subsets.items():
        A = loss_basis(S, hs, cols, o)
        s = np.linalg.svd(A, compute_uv=False)
        k = float(s[0] / s[-1])
        out[lbl] = k
        note = "잘 조건화" if k < 20 else ("주의" if k < 100 else "**거의 특이**")
        print(f"  {lbl:16s}{k:10.2f}{s[-1]:12.5f}   {note}")
    print("\n  기준: kappa 20 아래면 식별가능성 문제가 아니다 (12.87.3 을 정정).")
    print("  ⚠ kappa 는 **상대** 오차를 증폭한다. 관측 오차의 크기를 안 재면")
    print("    '풀린다' 고 말할 수 없다 — (4) 의 in-situ 가 그 크기를 준다.")
    return out


# ── (4) leak W/W ────────────────────────────────────────────────────────────
def sec_leak(S: np.ndarray, hs: np.ndarray, apps: List[str],
             sb: np.ndarray, nz: np.ndarray, orders: np.ndarray,
             insitu: bool = True) -> Dict[str, dict]:
    print("\n" + "=" * 88)
    print("(4) leak W/W — 저항 오예측이 SMPS 에 붙이는 W")
    print("=" * 88)
    idx = {a: i for i, a in enumerate(apps)}
    scols = [idx[a] for a in SMPS_APPLIANCES]
    A = loss_basis(S, hs, scols, orders)
    res_apps = [a for a in apps if a not in SMPS_APPLIANCES]
    out: Dict[str, dict] = {"analytic": {}, "insitu": {}}

    print("\n  [4a] 해석 — 저항 지문 1W 를 SMPS 기저에 NNLS 사영")
    print(f"       {'저항 기기':18s}{'프로젝터':>10s}{'충전기':>10s}{'미니PC':>10s}"
          f"{'합':>8s}{'잔차':>8s}")
    for r in res_apps:
        v = S[idx[r]][orders] / hs[orders]
        y = np.concatenate([v.real, v.imag])
        x, _ = nnls(A, y)
        rel = np.linalg.norm(y - A @ x) / max(np.linalg.norm(y), 1e-15)
        out["analytic"][r] = {"w_per_w": x.tolist(), "resid_rel": float(rel)}
        print(f"       {r:18s}{x[0]:10.3f}{x[1]:10.3f}{x[2]:10.3f}"
              f"{x.sum():8.3f}{rel:8.3f}")
    print("\n  판정선 (SMPS_PLAN 4.1): 어디서도 0.05 아래면 2.3 의 정정이 맞다.")
    print("  ⚠ 부호와 크기를 같이 볼 것 — 저항이 1.2kW 급이라 5% 오차면")
    print("    0.05 W/W 도 3W 다. 미니PC 전 대역이 7.6~26.7W 다.")

    if insitu:
        _leak_insitu(S, hs, sb, nz, apps, orders, out)
    return out


def _leak_insitu(S, hs, sb, nz, apps, orders, out) -> None:
    """저항 **전용** 실측 파일에서 나오는 유령 SMPS 전력.

    `test_9 / test_11 / test_12` 는 정답상 SMPS 가 0종이다. 관측 고조파를 9종
    기저에 NNLS 로 풀면 SMPS 열에 붙는 W 가 **관측·지문 오차가 만드는 유령의
    바닥**이다. 모델이 안 끼므로 이것은 순수 계측·지문 항이다.
    """
    print("\n  [4b] in-situ — 저항 전용 실측 파일에서 나오는 유령 SMPS W")
    print("       (정답상 SMPS 0종. NNLS 가 SMPS 열에 붙이는 만큼이 유령 바닥이다)")
    ev = load_events()
    stems = [s for s in ("test_9", "test_11", "test_12") if s in ev]
    if not stems:
        print("       대상 파일이 없다")
        return
    idx = {a: i for i, a in enumerate(apps)}
    scols = [idx[a] for a in SMPS_APPLIANCES]
    Aall = loss_basis(S, hs, list(range(len(apps))), orders)
    print(f"\n       {'파일':10s}{'P 관측':>10s}{'유령 SMPS':>12s}{'/P':>8s}"
          f"{'프로젝터':>10s}{'충전기':>10s}{'미니PC':>10s}")
    for stem in stems:
        z = load_nilm_npz(str(Path(EVAL_DIR) / f"{stem}.npz"))
        hc = np.asarray(z["harmonics_complex"])
        pf = np.asarray(z["power_features"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        i = np.flatnonzero(ok)
        X, P = [], []
        for k in range(0, len(i) - WINDOW_CYCLES, WINDOW_CYCLES):
            s = i[k:k + WINDOW_CYCLES]
            y = hc[s].mean(0)
            # 계측계 페이저를 뺀다. 대기 전류는 어느 기기가 꽂혀 있는지 모르므로 안 뺀다
            y = y - (nz[:, 0] + 1j * nz[:, 1])
            v = (y[orders] / hs[orders])
            x, _ = nnls(Aall, np.concatenate([v.real, v.imag]))
            X.append(x)
            P.append(pf[s, 0].mean())
        if not X:
            continue
        X, P = np.asarray(X), np.asarray(P)
        per = X[:, scols].mean(0)                 # 기기별 유령 W 평균
        G = X[:, scols].sum(1)                    # 창별 SMPS 합계
        print(f"       {stem:10s}{P.mean():10.1f}{G.mean():12.2f}"
              f"{G.mean() / max(P.mean(), 1e-9):8.4f}"
              + "".join(f"{x:10.2f}" for x in per))
        out["insitu"][stem] = {
            "p_observed_mean_w": float(P.mean()),
            "ghost_smps_w_mean": float(G.mean()),
            "ghost_smps_w_max": float(G.max()),
            "ghost_per_observed": float(G.mean() / max(P.mean(), 1e-9)),
            "per_appliance_w": {a: float(v) for a, v in zip(SMPS_APPLIANCES, per)},
            "n_windows": int(len(G))}
    print("\n       ⚠ 이 값이 미니PC 대역(7.6~26.7W)과 견줄 만하면, kappa 8 은")
    print("         '잘 조건화' 여도 **가장 작은 기기는 오차 바닥 아래**다.")


# ── (6) 식별가능성 in-situ ──────────────────────────────────────────────────
def sec_identifiability(S, hs, nz, apps, orders) -> Dict[str, dict]:
    """**12.87.3 을 모델 없이 직접 시험한다.**

    사람 라벨이 있는 SMPS 파일에서 창마다 두 번 푼다:

        자유       min_{P>=0} ‖y − A·P‖            9종 전부 열어 놓고
        정답 support  같은 것을 **사람이 켰다고 적은 기기로만**

    자유 쪽 잔차가 더 작은 것은 당연하다(자유도가 크다). 물어야 할 것은
    **정답이 관측을 얼마나 못 설명하는가** 다. 정답 쪽 잔차가 크게 나쁘면
    관측이 틀린 답을 **적극적으로 선호**한다는 뜻이고, 그러면 고조파 오차를
    줄이는 어떤 방법도(손실이든 NNLS든) 같은 자리로 간다.

    모델도 손실도 안 낀다. 지문 기하와 관측만 있다.
    """
    print("\n" + "=" * 88)
    print("(6) 식별가능성 in-situ — 관측이 정답을 더 좋아하는가 (12.87.3 직접 시험)")
    print("=" * 88)
    ev = load_events()
    idx = {a: i for i, a in enumerate(apps)}
    scols = [idx[a] for a in SMPS_APPLIANCES]
    A = loss_basis(S, hs, list(range(len(apps))), orders)
    out: Dict[str, dict] = {}
    print(f"\n       {'파일':9s}{'창':>4s}  {'자유 잔차':>9s} {'배분(P/C/M)':>16s}"
          f"  {'정답 잔차':>9s} {'배분(P/C/M)':>16s}  {'비':>5s}")
    for stem in ("test_7", "test_8", "test_13", "test_5", "test_6"):
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        RU, RT, XU, XT = [], [], [], []
        for k in range(0, len(i) - WINDOW_CYCLES, WINDOW_CYCLES):
            sl = i[k:k + WINDOW_CYCLES]
            y = hc[sl].mean(0) - (nz[:, 0] + 1j * nz[:, 1])
            v = y[orders] / hs[orders]
            b = np.concatenate([v.real, v.imag])
            nb = max(float(np.linalg.norm(b)), 1e-12)
            xu, ru = nnls(A, b)
            RU.append(ru / nb); XU.append(xu)
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if not len(sup):
                continue
            xt, rt = nnls(A[:, sup], b)
            full = np.zeros(len(apps)); full[sup] = xt
            RT.append(rt / nb); XT.append(full)
        if not RU or not RT:
            continue
        RU, RT = np.asarray(RU), np.asarray(RT)
        XU, XT = np.asarray(XU), np.asarray(XT)
        fu = "/".join(f"{v:.0f}" for v in XU[:, scols].mean(0))
        ft = "/".join(f"{v:.0f}" for v in XT[:, scols].mean(0))
        ratio = float(RT.mean() / max(RU.mean(), 1e-12))
        print(f"       {stem:9s}{len(RU):>4d}  {RU.mean():9.4f} {fu:>16s}"
              f"  {RT.mean():9.4f} {ft:>16s}  {ratio:5.2f}x")
        out[stem] = {"resid_free": float(RU.mean()), "resid_true_support": float(RT.mean()),
                     "ratio": ratio, "n_windows": int(len(RU)),
                     "alloc_free_w": dict(zip(SMPS_APPLIANCES, XU[:, scols].mean(0).tolist())),
                     "alloc_true_w": dict(zip(SMPS_APPLIANCES, XT[:, scols].mean(0).tolist()))}
    print("\n  읽는 법:")
    print("    비가 1 에 가깝다   -> 관측이 두 답을 못 가른다. 진짜 미결정이다 (12.87.3 맞음)")
    print("    비가 1 보다 크다   -> **관측이 틀린 답을 더 좋아한다.** 고조파 오차를 줄이는")
    print("                        어떤 방법도 같은 자리로 간다 — 손실 무게 문제가 아니다")
    print("    자유 잔차가 크다   -> 지문·관측 불일치가 그만큼이다. 배분 차이보다 크면")
    print("                        무엇을 고르든 그 오차 안이다")
    return out


# ── (5) 제안 무게 ───────────────────────────────────────────────────────────
def sec_weights(disc: Dict[str, dict]) -> Optional[List[float]]:
    print("\n" + "=" * 88)
    print("(5) 제안 무게 — 그리고 그것을 돌려야 하는지")
    print("=" * 88)
    dp = [v["dprime"] for v in disc.values() if "dprime" in v]
    if not dp:
        print("  d' 를 못 재서 제안하지 않는다.")
        return None
    worst = np.min(np.asarray(dp), axis=0)          # 최악 쌍 기준 (12.36 의 자)
    spread = float(worst.max() / max(worst.min(), 1e-9))
    naive = [v["naive"] for v in disc.values() if "naive" in v]
    nspread = float(np.max([max(x) / min(x) for x in naive])) if naive else float("nan")

    print(f"\n  최악 쌍 d' (홀수차)  " + "".join(f"{x:7.2f}" for x in worst))
    print(f"  분모 없는 자의 폭    {nspread:.2f} 배")
    print(f"  분모 있는 자의 폭    {spread:.2f} 배        <- **이쪽이 판정**")

    print("\n  판정 (SMPS_PLAN 4.1 — 돌리기 전에 적은 것):")
    if spread >= 3.0:
        w = (worst / worst.mean())
        full = np.zeros(15)
        full[ODD] = w
        print(f"  -> 3배 이상 갈린다. **4.2 로 간다.** 제안 harm_w (홀수차):")
        print("     " + "  ".join(f"h{h}={x:.2f}" for h, x in zip(range(1, 16, 2), w)))
        return full.tolist()
    print(f"  -> **평평하다 ({spread:.2f}배 < 3배). 이 방향을 닫는다.**")
    print("     4.2(L_harm 차수 무게)를 돌리지 않는다. SMPS_PLAN 7절이 적은")
    print("     *\"4.1 의 ②가 평평하다 -> 3절의 진단이 죽는다\"* 그 경우다.")
    print("\n  대신 남는 손실 축은 (1) 이 가리킨 곳이다:")
    print("     `--harm-odd-only` — 확정된 계측 인공물(12.72)에서 무게를 뺀다.")
    print("     플래그가 이미 있고(run_adapt.py), 단일 변수로 잰 적이 없다(12.75.5).")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/adapt_ovh.pt",
                    help="기기 목록을 읽을 체크포인트 (가중치는 안 쓴다)")
    ap.add_argument("--no-insitu", action="store_true", help="(4b) 실측 부분을 건너뛴다")
    ap.add_argument("--orders", default="odd", choices=("odd", "all"),
                    help="leak/kappa 를 어느 차수로 풀지. 기본 홀수차")
    ap.add_argument("--out", default="results/sig_conditioning.json")
    a = ap.parse_args()

    apps = torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"]
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir=NPZ_DIR, time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb = standby_signatures(pool, apps)
    nz = noise_signature(pool)
    hs = harmonic_scales(pool, apps)
    del pool
    S = complex_sig(sig)
    orders = ODD if a.orders == "odd" else np.arange(15)

    print("=" * 88)
    print("지문 조건화 진단 — SMPS 3종 (SMPS_PLAN 4.1절, 개정판)")
    print("=" * 88)
    print(f"  기기 {len(apps)}종, 15차, 창 {WINDOW_CYCLES} 사이클(60초)")
    print(f"  leak/kappa 차수: {a.orders}")

    sec_harm_scale(hs)
    win = {app: isolated_windows(app) for app in SMPS_APPLIANCES}
    disc = sec_discrimination(S, hs, apps, win)
    kap = sec_kappa(S, hs, apps)
    leak = sec_leak(S, hs, apps, sb, nz, orders, insitu=not a.no_insitu)
    ident = {} if a.no_insitu else sec_identifiability(S, hs, nz, apps, orders)
    proposed = sec_weights(disc)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"appliances": apps, "harm_scale": hs.tolist(), "orders": a.orders,
         "discrimination": disc, "kappa": kap, "leak": leak,
         "identifiability": ident,
         "proposed_harm_w": proposed,
         "isolated_windows": {k: int(len(v)) for k, v in win.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
