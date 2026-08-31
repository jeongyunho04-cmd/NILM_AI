"""추론 후처리 — 물리 전력 상한과 게이트 동기 (12.100~12.102절)

**무엇을 고치는가.** 프로젝터는 격리에서 48.5~49.3W (p5~p95), 최대 49.6W 로
폭이 ±0.5W 인 기기다. 그런데 2단계 단독 모델은 복합 실측에서 중앙 73.5W, 최대
137W 를 붙이고 **창의 74.6%** 가 물리 상한을 넘는다 (12.100.1). 그 초과분은
다른 SMPS 의 몫이고, 그래서 충전기가 전이에서 5/23 밖에 안 맞았다.

**어떻게 고치는가.** 추론 뒤에 상한을 넘는 만큼을 잘라 **다른 SMPS 로 넘긴다.**

    P_proj > cap 이면  초과분을 게이트 가중으로 다른 SMPS 에 분배
    넘겨받아 문턱을 넘은 기기는 게이트도 켠다 (gate_sync)

**세 가지를 측정으로 정했다** (12.102):

- **상한은 프로젝터에만 건다.** 충전기·미니PC 는 상한을 넘는 창이 0.0%/0.4% 라
  걸 이유가 없고, 걸면 **받을 자리가 좁아져 오히려 나빠진다** (44/59 -> 39/59)
- **상한값은 둔감하다.** 55 / 60 / 70W 가 모두 44/59 다. 격리 최대(49.6W)에
  여유를 얹은 55W 를 쓴다 — 튜닝 잔향이 아니라는 근거다
- **초과분은 전부 즉시 넘긴다.** "정상 상태만 넘기고 전이 순간은 남긴다" 를
  시험했으나 **반대로 갔다** (12.101.2) — 재배분은 바로 그 전이 순간에 작동해
  효과를 낸다

**하이브리드에는 걸지 않는다.** 1단계에서 SMPS 를 가져오는 조합은 프로젝터
초과가 44.9% 로 덜하고, 충전기가 이미 20/23 이라 상한이 프로젝터만 깎는다
(41/59 -> 36/59). `run_gate_check --postproc` 는 그래서 기본이 꺼져 있다.
"""
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

#: 이 기기들 사이에서만 전력을 옮긴다 (저항 부하는 건드리지 않는다).
SMPS_GROUP: Tuple[str, ...] = ("beam_projector", "laptop_charger", "minipc")

#: 격리 통전 중 60초 창 평균의 최대값 (2026-08-25 측정).
ISOLATED_MAX_W: Dict[str, float] = {
    "beam_projector": 49.6, "laptop_charger": 70.3, "minipc": 26.9,
}

#: 상한을 거는 기기와 그 값. 격리 최대 + 11%.
CAP_W: Dict[str, float] = {"beam_projector": 55.0}

#: 게이트를 켤 문턱 — 격리 통전 중앙값(48.8 / 51.9 / 14.5W)의 20%.
#: 20% 와 40% 가 튜닝셋에서 각각 미니PC F1 +0.079 / +0.055 였고 홀드아웃에서는
#: 둘 다 −0.01 이다. 이득이 파일에 따라 갈리므로 **기본은 끔** (12.102.2).
GATE_ON_W: Dict[str, float] = {
    "beam_projector": 9.8, "laptop_charger": 10.4, "minipc": 2.9,
}


def apply_postproc(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
                   caps: Optional[Dict[str, float]] = None,
                   gate_sync: bool = False,
                   gate_on_w: Optional[Dict[str, float]] = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """물리 상한을 넘는 예측을 잘라 다른 SMPS 로 넘긴다.

    Args:
        P: (n, K) 기기별 예측 전력 (게이트가 곱해진 값)
        gate: (n, K) 게이트 확률
        apps: 기기 목록 (열 순서)
        caps: {기기: 상한W}. 기본 `CAP_W` (프로젝터만)
        gate_sync: 넘겨받아 문턱을 넘은 기기의 게이트를 켠다
        gate_on_w: 게이트 문턱. 기본 `GATE_ON_W`

    Returns:
        (P_new, gate_new). 총합은 보존된다 — 받을 기기가 있는 한 버리지 않는다.
    """
    caps = CAP_W if caps is None else caps
    thr = GATE_ON_W if gate_on_w is None else gate_on_w
    out = np.array(P, dtype=np.float64, copy=True)
    g = np.array(gate, dtype=np.float64, copy=True)

    recv = [j for j, a in enumerate(apps) if a in SMPS_GROUP]
    capped = [(j, caps[a]) for j, a in enumerate(apps) if a in caps]
    if not capped or not recv:
        return out, g

    for j, lim in capped:
        excess = np.clip(out[:, j] - lim, 0.0, None)
        if not excess.any():
            continue
        out[:, j] -= excess
        others = [k for k in recv if k != j]
        if not others:
            continue
        w = np.clip(g[:, others], 1e-6, None)
        share = w / w.sum(1, keepdims=True)
        out[:, others] += share * excess[:, None]

    if gate_sync:
        for j, a in enumerate(apps):
            if a in SMPS_GROUP and a in thr:
                g[:, j] = np.where(out[:, j] >= thr[a],
                                   np.maximum(g[:, j], 0.5 + 1e-6), g[:, j])
    return out, g


#: 잔차 흡수 비율 (12.104). 기본파를 빼고 유사도를 재면 1.0 에서도 전이 귀속이
#: 45/59 로 유지되고 잔차만 8.88 -> 6.50W 로 준다. **1.0 을 쓴다.**
#: (기본파를 넣었을 때는 1.0 에서 전이가 43/59 로 떨어졌다 — 저항 잔차가
#:  SMPS 로 새면서 전이 Δ 를 흔들었기 때문이다.)
ABSORB_FRAC = 1.0

#: 잔차를 받을 최소 게이트. 꺼져 있다고 본 기기에는 안 넘긴다 (유령 방지).
ABSORB_MIN_GATE = 0.1


def absorb_residual(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
                    standby: np.ndarray, p_noise: np.ndarray, p_observed: np.ndarray,
                    obs_harm: np.ndarray, sig: np.ndarray, standby_sig: np.ndarray,
                    noise_sig: np.ndarray, frac: float = ABSORB_FRAC,
                    min_gate: float = ABSORB_MIN_GATE,
                    odd_only: bool = True, skip_fundamental: bool = True) -> np.ndarray:
    """총전력 잔차를 **고조파 잔차가 닮은 SMPS 에** 나눠 준다 (12.104).

    `L_cons` 를 추론 시점에 거는 것과 같은데, **누구에게 줄지를 고조파가 정한다**
    는 점이 다르다 (12.5 가 경고한 "합만 보고 아무에게나 붙이는" 것을 피한다).

        resid_h = 관측 고조파 − (Σ P̂·sig + 대기 + 계측계)
        w_k     = max(0, cos(resid_h, sig_k))        게이트가 낮은 기기는 0
        P̂_k    += w_k / Σw · (관측 총전력 − 예측 총합) · frac

    **안전장치가 둘이다.** ① 게이트가 `min_gate` 아래인 기기는 안 받는다.
    ② 유사도를 **3차 이상**으로만 잰다 — 기본파를 넣으면 코사인이 1차에 지배되어
    저항 부하의 잔차까지 SMPS 를 닮은 것으로 보인다. 실측에서는 `test_9`(저항만)가
    25.65 -> 25.35W 로 거의 안 변하고 `test_5` 는 12.25 -> 3.52W 다 (frac=1.0).
    """
    out = np.array(P, dtype=np.float64, copy=True)
    if frac <= 0:
        return out
    cols = [j for j, a in enumerate(apps) if a in SMPS_GROUP]
    if not cols:
        return out
    hh = list(range(0, sig.shape[1], 2)) if odd_only else list(range(sig.shape[1]))
    if skip_fundamental and len(hh) > 1:
        # **기본파를 뺀다.** 넣으면 코사인이 1차에 지배되어 저항 부하의 잔차까지
        # SMPS 를 닮은 것으로 보인다 (단위 시험: 오븐 모양 20W 중 12.3W 가 SMPS 로
        # 흘렀다). 3차 이상만 쓰면 크기가 아니라 **모양**으로 갈린다.
        hh = hh[1:]

    total = out.sum(1) + standby.sum(1) + p_noise
    resid = (p_observed - total) * float(frac)

    pred_h = (np.einsum("nk,khc->nhc", out, sig)
              + np.einsum("nk,khc->nhc", standby, standby_sig) + noise_sig[None])
    R = (obs_harm - pred_h)[:, hh, :]
    Rc = R[:, :, 0] + 1j * R[:, :, 1]
    rn = np.linalg.norm(Rc, axis=1)

    W = np.zeros((len(out), len(cols)))
    for i, j in enumerate(cols):
        s = sig[j][hh, 0] + 1j * sig[j][hh, 1]
        W[:, i] = np.clip(np.real(Rc @ np.conj(s)) / (np.linalg.norm(s) * rn + 1e-12),
                          0.0, None)
    W = np.where(gate[:, cols] >= min_gate, W, 0.0)
    tw = W.sum(1, keepdims=True)
    ok = tw[:, 0] > 0
    add = np.zeros_like(W)
    add[ok] = W[ok] / tw[ok] * resid[ok, None]
    out[:, cols] = np.clip(out[:, cols] + add, 0.0, None)
    return out


# ── 저항 부하 정합 (12.112) ──────────────────────────────────────────────
#: 격리 실측에서 잰 등가저항 (V^2 / P, 통전 중 중앙값).
#: **기기 고유값이다** — 같은 기기의 서로 다른 녹화에서 0.1~1.3% 안에 든다.
#:     포트   35.79 / 35.85            오븐 40.41 / 40.90 / 40.45
#:     드라이기 54.48 / 54.57 / 53.89   핫플 101.76 / 101.82 / 101.90
#: 니크롬선 저항이라 전압·개체와 무관하게 재현된다. 고조파로는 0.596%p 밖에
#: 안 갈리는 저항 3종이 **저항값으로는 13~180% 갈린다** (0.2절의 그 문제).
RESISTIVE_OHM: Dict[str, float] = {
    "electiric_kettle": 35.8, "oven": 40.6, "hair_dryer": 54.3, "hotplate": 101.8,
}

#: 드라이기 약풍은 반파 정류라 전력이 절반이고 겉보기 저항이 2배(102.6Ω)다.
#: 핫플(101.8Ω)과 겹치므로 **짝수차로 가른다** — 반파는 |I2|/|I1| 0.40,
#: 핫플은 0.00 이다 (12.109.2).
HALFWAVE_OHM: Dict[str, float] = {"hair_dryer": 108.6}
HALFWAVE_I2_MIN = 0.15


def resistive_match(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
                    p_observed: np.ndarray, v_rms: np.ndarray,
                    standby: np.ndarray, p_noise: np.ndarray,
                    obs_harm: Optional[np.ndarray] = None,
                    tol: float = 0.05, min_w: float = 150.0,
                    cand_gate_min: float = 0.0, margin: float = 2.0,
                    snap: bool = False,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """관측 전력·전압에 **맞는 저항 조합**을 골라 재배정한다 (12.112).

    저항 부하는 니크롬선이라 `P = V^2 / R` 이고 `R` 이 기기 고유값이다. 그래서
    창마다 다음을 푼다:

        G_필요 = (관측 P − 비저항 예측 − 대기 − 계측) / V^2
        16개 조합(저항 4종 on/off) 중 Σ(1/R_i) 가 G_필요 에 가장 가까운 것

    **왜 후처리인가.** 모델은 P 와 V 를 따로 받지만 `V^2/P` 를 명시적으로 만들지
    않고, 복합에서는 개별 기기의 P 를 모른다. 반면 후처리는 **합에서 역산**할 수
    있다 — 저항 성분만 남기면 컨덕턴스는 더해지므로 조합을 셀 수 있다.

    Args:
        tol: 상대 오차가 이 값을 넘으면 손대지 않는다 (설명 못 하는 창)
        min_w: 저항 성분이 이보다 작으면 손대지 않는다 (전부 꺼진 창)
        obs_harm: (n,15,2). 주면 드라이기 약(반파)을 짝수차로 가른다
        cand_gate_min: 후보 조합에 넣을 최소 게이트 (12.117 의 B). 0 이면 무제한.
            **이미 켜진 기기(`cur`)는 문턱과 무관하게 남는다** — 12.112 가
            경고한 "게이트로 후보를 좁히면 고쳐야 할 맞바꿈을 놓친다" 를 피한다
            (`test3` 오븐 게이트 0.09). 겨냥은 게이트가 **바닥**인 기기를
            맞바꿈으로 켜는 것이다 (`test_9` 드라이기).
        snap: 조합이 이미 맞을 때(`best == cur`)도 전력을 `V^2/R` 로 맞춘다
            (12.117 의 A). 개수도 신원도 안 바뀌므로 규칙 18 과 충돌하지 않는다 —
            **같은 집합**이다. 겨냥은 `test3` 처럼 조합은 맞는데 전력이 모자라
            그 차이가 문턱 아래 게이트로 새는 창이다.
    """
    out = np.array(P, dtype=np.float64, copy=True)
    g = np.array(gate, dtype=np.float64, copy=True)
    cols = [j for j, a in enumerate(apps) if a in RESISTIVE_OHM]
    if not cols:
        return out, g
    names = [apps[j] for j in cols]
    v2 = np.maximum(np.asarray(v_rms, dtype=np.float64), 1.0) ** 2

    # 저항 성분 = 관측 − (비저항 예측 + 대기 + 계측)
    other = out.sum(1) - out[:, cols].sum(1)
    p_res = np.asarray(p_observed, np.float64) - other - standby.sum(1) - p_noise

    # 반파(드라이기 약) 판정용 짝수차
    half = np.zeros(len(out), bool)
    if obs_harm is not None:
        h = np.asarray(obs_harm, np.float64)
        i1 = np.hypot(h[:, 0, 0], h[:, 0, 1])
        i2 = np.hypot(h[:, 1, 0], h[:, 1, 1])
        half = (i2 / np.maximum(i1, 1e-9)) > HALFWAVE_I2_MIN

    # 16개 조합의 컨덕턴스를 미리 만든다
    import itertools
    combos = []
    for k in range(len(cols) + 1):
        for pick in itertools.combinations(range(len(cols)), k):
            combos.append(pick)

    for i in range(len(out)):
        if p_res[i] < min_w:
            continue
        g_need = p_res[i] / v2[i]
        ohm = dict(RESISTIVE_OHM)
        if half[i]:
            ohm.update(HALFWAVE_OHM)      # 드라이기를 반파 저항으로 본다

        # **개수는 그대로 두고 맞바꿈만 한다.** 기기를 늘리게 두면 정합기가 없는
        # 기기를 발명한다 — 제한 없이 돌렸을 때 test_9(드라이기 없는 파일)의 유령이
        # 3.94 -> 86.98W 로 터졌다. 반대로 게이트로만 후보를 좁히면 고쳐야 할
        # 맞바꿈을 못 한다 — test3 의 오븐 게이트가 0.09 라 후보에서 빠진다.
        # 물리 정합이 잘하는 일은 **누구인지 바꾸는 것**이지 몇 대인지 정하는 것이
        # 아니다. 몇 대인지는 모델이 안다.
        cur = tuple(k for k, j in enumerate(cols) if g[i, j] > 0.5)
        if not cur:
            continue

        # 후보 제한 (12.117.3). 게이트가 바닥인 기기는 맞바꿈으로 켜지 못한다.
        # **이미 켜진 것은 무조건 남긴다** — 안 그러면 `cur` 자체가 후보에서
        # 빠져 `cur_err` 이 inf 가 되고 아무 맞바꿈이나 통과한다.
        allow = None
        if cand_gate_min > 0:
            allow = {k for k, j in enumerate(cols)
                     if g[i, j] >= cand_gate_min} | set(cur)

        best, best_err, cur_err = None, np.inf, np.inf
        for pick in combos:
            if len(pick) != len(cur):
                continue
            if allow is not None and not set(pick) <= allow:
                continue
            gg = sum(1.0 / ohm[names[k]] for k in pick)
            if gg <= 0:
                continue
            err = abs(gg - g_need) / max(g_need, 1e-9)
            if pick == cur:
                cur_err = err
            if err < best_err:
                best, best_err = pick, err
        # **모델의 조합보다 확실히 나을 때만 바꾼다** (margin 배 이상).
        # 다만 `best == cur` — 조합이 이미 맞는 경우 — 는 맞바꿈이 아니라 **스냅**이다.
        # margin 조건은 `cur_err == best_err` 라 항상 걸리므로 따로 가른다 (12.117.2).
        if best is None or best_err > tol:
            continue
        if tuple(best) == tuple(cur):
            if not snap:
                continue
        elif best_err * margin > cur_err:
            continue
        for k, j in enumerate(cols):
            on = k in best
            out[i, j] = v2[i] / ohm[names[k]] if on else 0.0
            g[i, j] = max(g[i, j], 0.5 + 1e-6) if on else min(g[i, j], 0.5 - 1e-6)
    return out, g
