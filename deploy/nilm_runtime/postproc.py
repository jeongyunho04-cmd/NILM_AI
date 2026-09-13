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


# ── 프로젝터 전력 스냅 (2026-08-31, SMPS_PLAN 4.3절) ─────────────────────────
# **지금 것은 한 방향이다.** `apply_postproc` 은 상한(55W)을 *넘는* 만큼만 넘긴다.
# 그런데 12.120.1 이 재 보니 창의 **87.1%** 가 상한에 붙어 있고, 격리 통전값은
# 47.4W (p5~p95 46.4~48.3, 폭 1.9W) 다 — 상한이 격리값보다 7.6W 높아서 그
# 사이의 과대예측은 통과한다.
#
# 스냅은 세 가지가 다르다:
#     ① 값     55W(상한) -> 47.4W(격리 중앙)
#     ② 방향   한 방향 -> **양방향**. 모델이 40W 로 낮게 붙였으면 되받아 온다
#     ③ 상대   게이트 가중 / 3차 이상 고조파 코사인 중 고른다
#
# ⚠ **12.102 의 "52W 로 내렸더니 42/59 로 나빠졌다" 와 다른 조작이다** —
#   그것은 clip 이고 이것은 snap 이다. 다만 그 결과가 이 항목의 가장 강한 반증
#   후보이므로 **값 둔감성 검사**(47.4 / 48.5 / 50.0 이 같은 방향인가)를 반드시
#   같이 돌린다. 한 점만 좋으면 튜닝 잔향이다.
#
# ⚠ **양방향은 총합을 보존하지 않는다.** 낮게 붙은 것을 끌어올리면 그만큼을
#   남에게서 뺏어 와야 하는데, 뺏길 기기가 이미 0 이면 못 뺏는다. 그래서
#   `take_from` 이 실제로 뺀 만큼만 준다 — 총합은 절대 늘지 않는다.
#
# ⚠ **프로젝터는 상태가 여럿이다** (OPERATING_POINT 1절, 상태수 3). 단일 상수
#   스냅은 그 상태를 뭉갠다. `state_targets` 를 주면 모델의 상태 argmax 에 맞는
#   격리 중앙값으로 스냅한다 — 그쪽이 물리적으로 옳다.
#
#: 격리 통전 중앙값. **상한이 아니라 중앙이다.**
#
# 2026-09-01: 47.4 -> 46.9. 47.4 는 12.102 시절 `SNAP_SWEEP_W` 의 첫 후보였고
# **튜닝으로 고른 값**이었다. 46.9 는 `power_ref.REFERENCE_W` 가 쓰는 것과 같은
# 측정값이다 — `recompute_reference` 가 격리 녹화 3개의 60초 창 95개에서 낸
# 중앙값(p5 44.2 / p95 47.8, 폭/중앙 0.077). **채점 기준과 스냅 목표의 출처를
# 하나로 묶는다.** 결과 차는 0.5W 로 미미하고, 바꾸는 이유는 성능이 아니라 계보다.
#
# ⚠ 프로젝터 저전력 상태(WARMUP_COOLDOWN 3.7~4.4W, 통전의 3.1%)를 이 고정 목표가
#   끌어올린다. 60초 창에서는 묻힌다 — 격리 95창 중 warmup 이 절반을 넘는 창이
#   **0개**, 10%를 넘는 창이 2개(평균 41.8~42.4W)다. 창 최소가 41.8W 이므로
#   **최악의 스냅 오차가 +5.1W** 로 유계다 (2026-09-01 확인). 창이 짧아지면
#   이 가정이 깨지므로 `state_targets` 로 가야 한다.
SNAP_TARGET_W: Dict[str, float] = {"beam_projector": 46.9}

#: 값 둔감성 검사용 후보. 한 점만 좋으면 튜닝 잔향이다 (12.102 가 상한에 쓴 검사).
SNAP_SWEEP_W: Tuple[float, ...] = (47.4, 48.5, 50.0)


def snap_power(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
               targets: Optional[Dict[str, float]] = None,
               bidirectional: bool = True,
               share: str = "gate",
               obs_harm: Optional[np.ndarray] = None,
               sig: Optional[np.ndarray] = None,
               state_targets: Optional[Dict[str, Sequence[float]]] = None,
               state: Optional[np.ndarray] = None,
               min_gate: float = 0.5,
               redistribute: bool = True,
               ) -> Tuple[np.ndarray, np.ndarray]:
    """켜져 있다고 본 기기의 전력을 격리 중앙값으로 **양방향** 스냅한다.

    Args:
        P: (n, K) 예측 전력   gate: (n, K) 게이트 확률
        targets: {기기: 목표W}. 기본 `SNAP_TARGET_W`
        bidirectional: False 면 초과분만 넘긴다 (지금 동작과 같은 방향)
        share: "gate" 게이트 가중 | "harm" 3차 이상 고조파 코사인
        obs_harm/sig: share="harm" 일 때 필요 (n,15,2) / (K,15,2)
        state_targets: {기기: [상태별 목표W]}. 주면 `state` argmax 로 고른다
        state: (n, K, S) 상태 로짓
        min_gate: 이 게이트 아래면 손대지 않는다 — 꺼진 기기를 켜지 않는다 (규칙 18)
        redistribute: False 면 **깎기만 하고 남에게 안 준다.** 총합 보존이 깨지는
            대신 유령이 안 는다. 12.122.4 가 스냅을 반증한 이유가 재배분이었는지
            스냅 자체였는지 가른다 (규칙 3)

    Returns:
        (P_new, gate_new). **개수도 신원도 안 바꾼다** — 규칙 18 을 지킨다.
        총합은 늘지 않는다 (뺏을 수 있는 만큼만 준다).
    """
    tgt = SNAP_TARGET_W if targets is None else targets
    out = np.array(P, dtype=np.float64, copy=True)
    g = np.array(gate, dtype=np.float64, copy=True)
    recv = [j for j, a in enumerate(apps) if a in SMPS_GROUP]
    snapped = [(j, a) for j, a in enumerate(apps) if a in tgt]
    if not snapped or len(recv) < 2:
        return out, g

    for j, app in snapped:
        others = [k for k in recv if k != j]
        if not others:
            continue
        # 목표값 — 상태별이 있으면 그것을 쓴다
        if state_targets and app in state_targets and state is not None:
            lv = np.asarray(state_targets[app], dtype=np.float64)
            si = np.asarray(state)[:, j, :len(lv)].argmax(1)
            aim = lv[si]
        else:
            aim = np.full(len(out), float(tgt[app]))

        live = g[:, j] >= min_gate            # 켜져 있다고 본 창만
        delta = np.where(live, out[:, j] - aim, 0.0)
        if not bidirectional:
            delta = np.clip(delta, 0.0, None)

        # 나눌 비중
        if share == "harm":
            if obs_harm is None or sig is None:
                raise ValueError("share='harm' 은 obs_harm 과 sig 가 필요합니다")
            w = _harmonic_affinity(obs_harm, sig, others)
        else:
            w = np.clip(g[:, others], 1e-6, None)
        ssum = w.sum(1, keepdims=True)
        w = np.divide(w, ssum, out=np.zeros_like(w), where=ssum > 0)

        give = np.clip(delta, 0.0, None)       # 프로젝터가 내놓는 몫
        take = np.clip(-delta, 0.0, None)      # 프로젝터가 받아 오는 몫
        if not redistribute:
            # 깎기만 한다. 받아 오지도 않는다 — 줄 사람이 없으므로.
            out[:, j] -= give
            continue
        # **받아 올 때는 남이 가진 만큼까지만.** 총합을 늘리지 않는다.
        avail = (out[:, others] * w).sum(1)
        take = np.minimum(take, avail)

        out[:, j] += take - give
        out[:, others] += w * give[:, None] - w * take[:, None]
        out = np.clip(out, 0.0, None)
    return out, g


def _harmonic_affinity(obs_harm: np.ndarray, sig: np.ndarray,
                       cols: Sequence[int]) -> np.ndarray:
    """관측 고조파와 각 기기 지문의 **3차 이상** 코사인 (n, len(cols)).

    `absorb_residual` 이 잔차를 기기에 사영할 때 쓰는 기계와 같다 — 기본파를
    빼는 이유도 같다 (12.104: 기본파를 넣으면 저항 잔차가 SMPS 로 샌다).
    게이트가 이미 편향돼 있으면(충전기 재현 0.641) 게이트 가중은 그 편향대로
    나눈다. 관측 자체로 나누면 그 고리를 끊는다.
    """
    o = np.asarray(obs_harm, dtype=np.float64)
    y = (o[:, 2:, 0] + 1j * o[:, 2:, 1])                  # h>=3
    yn = np.linalg.norm(y, axis=1)
    W = np.zeros((len(o), len(cols)))
    for c, j in enumerate(cols):
        v = sig[j, 2:, 0] + 1j * sig[j, 2:, 1]
        vn = np.linalg.norm(v)
        W[:, c] = np.clip(np.real(y @ np.conj(v)) / (yn * vn + 1e-12), 0.0, None)
    return W


#: 잔차 흡수 비율 (12.104). 기본파를 빼고 유사도를 재면 1.0 에서도 전이 귀속이
#: 45/59 로 유지되고 잔차만 8.88 -> 6.50W 로 준다. **1.0 을 쓴다.**
#: (기본파를 넣었을 때는 1.0 에서 전이가 43/59 로 떨어졌다 — 저항 잔차가
#:  SMPS 로 새면서 전이 Δ 를 흔들었기 때문이다.)
ABSORB_FRAC = 1.0

#: 잔차를 받을 최소 게이트. 꺼져 있다고 본 기기에는 안 넘긴다 (유령 방지).
ABSORB_MIN_GATE = 0.1


#: **흡수가 넘겨서는 안 되는 기기별 천장** (12.149.4).
#:
#: `CAP_W` 는 프로젝터 하나뿐이라, 12.149.3 으로 프로젝터를 막고 나니 같은 일이
#: 미니PC 에서 났다 — `test_4` 에서 흡수가 미니PC 를 **98.5W** 까지 밀었다
#: (격리 실측 상한 29.8W 의 3.3배). 상한이 한 기기에만 있으면 오차는 옆으로 간다
#: (규칙 29).
#:
#: 값은 **격리 녹화의 사이클 최대**다 (2026-09-02 측정, `processed_data/npz`,
#: `is_on & is_valid & p>1W`). 60초 창 최대가 아니라 사이클 최대를 쓰는 이유:
#: 창 평균은 짧은 첨두를 뭉개므로 천장으로 쓰면 실재하는 값을 자른다.
#:
#:     기기              60초창 최대   사이클 최대   ISOLATED_MAX_W (2026-08-25)
#:     beam_projector       47.9       52.0        49.6   <- CAP_W 55.0 이 그 위다
#:     laptop_charger       68.8       84.6        70.3
#:     minipc               25.6       29.8        26.9
#:
#: ⚠ 위의 `ISOLATED_MAX_W` 와 **다른 통계다.** 그쪽은 60초 창 *평균*의 최대이고
#:   (재현 47.9 / 68.8 / 25.6 으로 재측정과 맞는다), 이쪽은 *사이클* 최대다.
#:   창 평균을 천장으로 쓰면 실재하는 첨두를 자르므로 여기서는 못 쓴다.
#:
#: 프로젝터는 `CAP_W` 의 55.0 을 그대로 둔다 — **더 느슨한 쪽을 남긴다.** 이 표는
#: 흡수를 막는 안전장치이지 예측을 조이는 장치가 아니다.
ABSORB_CAP_W: Dict[str, float] = {
    "beam_projector": 55.0, "laptop_charger": 84.6, "minipc": 29.8,
}


#: 게이트 정합 문턱 (12.149). **자유 파라미터가 아니다** — 채점(`score_on_off`)과
#: 화면(`run_live.render`)이 이미 쓰는 그 0.5 이고, 요구는 하나다:
#: **"OFF 라고 보고한 기기는 0W 를 낸다."**
SQUELCH_TAU = 0.5


def squelch(P: np.ndarray, gate: np.ndarray, tau: float = SQUELCH_TAU) -> np.ndarray:
    """게이트가 `tau` 아래인 기기의 전력을 0 으로 만든다 (12.149).

    `P̂ = σ(on)·p_raw` 는 σ 가 아무리 작아도 0 이 아니다. 그래서 **꺼졌다고
    보고한 기기가 와트를 낸다.** 12.149 가 `adapt_zi_s0` 에서 잰 것:

        에어컨    σ 0.0080 x p_raw  592W = 4.92W   (게이트>0.5 인 창 0.01%)
        전기포트  σ 0.0048 x p_raw  973W = 5.89W   (이쪽은 `resistive_match` 가 지운다)

    유령8 6.78W 의 82% 가 이 문턱 아래 누설이고, 하드 게이트로 보면 0.83W 다.
    `w_hedge` 는 이것을 못 본다 — 이진 엔트로피 H(0.008)=0.046 이라 σ=0.008 은
    이미 "결정했다" 인데 와트로는 4.9W 다. **손실에 와트를 보는 항이 없다.**

    ⚠ **깎기만 한다.** 빠진 와트는 `absorb_residual` 이 고조파가 닮은 기기로
      돌려보낸다 — 안 그러면 잔차가 4.52 -> 8.65W 로 커진다 (규칙 29: 오차는
      사라지지 않고 옮겨 다닌다). 둘은 **같이 써야 한다.**
    """
    out = np.array(P, dtype=np.float64, copy=True)
    out[np.asarray(gate) < float(tau)] = 0.0
    return out


#: `norton_offset` 이 계수를 한 번만 읽도록 하는 캐시.
_NORTON_CACHE: Dict[str, tuple] = {}


def norton_offset(obs_harm: np.ndarray, coef_npz: str) -> np.ndarray:
    """관측 고조파에서 계통 임피던스 보정을 만든다 (12.148.2, 12.152).

    `realdata.harmonic_offset` 은 원자료 CSV 의 `ih*`/`ihdeg*` 를 읽는다. 그것은
    **적합 경로의 편의**였다 — 추론에는 `obs_harm` 의 Re/Im 이 곧 복소 전류라
    CSV 가 필요 없다. 두 경로를 `test_7` 에서 견주면 **cos 1.0000, 비 1.0000**
    이다 (2026-09-02 확인). 그래서 실시간에서도 같은 보정을 쓸 수 있다.

        V_h = V_src,h − Z_h·I_h,  Z_h = R + j·h·ωL      (12.148.2)
        보정 = ((1, Re(Z·I), Im(Z·I), ...) − mu)/sd @ B

    Args:
        obs_harm: (n, H, 2) 관측 고조파 Re/Im (A)
        coef_npz: `run_norton_probe --save-coef` 의 산출물
    Returns:
        (n, H, 2) 창별 보정. 짝수차는 0 이다 (12.72 + 12.147).
    """
    if coef_npz not in _NORTON_CACHE:
        z = np.load(coef_npz, allow_pickle=True)
        _NORTON_CACHE[coef_npz] = (
            np.asarray(z["coef"], np.float64), np.asarray(z["mu"], np.float64),
            np.asarray(z["sd"], np.float64), np.asarray(z["orders"], int),
            np.asarray(z["zi_orders"], int), float(z["R"]), float(z["X1"]))
    B, mu, sd, odd, zo, R, X1 = _NORTON_CACHE[coef_npz]
    o = np.asarray(obs_harm, dtype=np.float64)
    cc = []
    for h in zo:
        zi = (R + 1j * h * X1) * (o[:, h - 1, 0] + 1j * o[:, h - 1, 1])
        cc += [zi.real, zi.imag]
    y = ((np.c_[np.ones(len(o)), np.array(cc).T] - mu) / sd) @ B
    out = np.zeros((len(o), o.shape[1], 2), np.float64)
    k = len(odd)
    out[:, odd, 0] = y[:, :k]
    out[:, odd, 1] = y[:, k:]
    return out


def absorb_residual(P: np.ndarray, gate: np.ndarray, apps: Sequence[str],
                    standby: np.ndarray, p_noise: np.ndarray, p_observed: np.ndarray,
                    obs_harm: np.ndarray, sig: np.ndarray, standby_sig: np.ndarray,
                    noise_sig: np.ndarray, frac: float = ABSORB_FRAC,
                    min_gate: float = ABSORB_MIN_GATE,
                    odd_only: bool = True, skip_fundamental: bool = True,
                    exclude: Optional[Sequence[str]] = None,
                    caps: Optional[Dict[str, float]] = None,
                    harm_offset: Optional[np.ndarray] = None,
                    mode: str = "cos",
                    limit_by_harm: bool = False,
                    qp: Optional[np.ndarray] = None,
                    noise_q: float = 0.0,
                    q_observed: Optional[np.ndarray] = None,
                    w_q: float = 3.0) -> np.ndarray:
    """총전력 잔차를 **고조파 잔차가 닮은 SMPS 에** 나눠 준다 (12.104).

    `L_cons` 를 추론 시점에 거는 것과 같은데, **누구에게 줄지를 고조파가 정한다**
    는 점이 다르다 (12.5 가 경고한 "합만 보고 아무에게나 붙이는" 것을 피한다).

        resid_h = 관측 고조파 − (Σ P̂·sig + 대기 + 계측계)
        w_k     = max(0, cos(resid_h, sig_k))        게이트가 낮은 기기는 0
        P̂_k    += w_k / Σw · (관측 총전력 − 예측 총합) · frac

    **안전장치가 셋이다.** ① 게이트가 `min_gate` 아래인 기기는 안 받는다.
    ② 유사도를 **3차 이상**으로만 잰다 — 기본파를 넣으면 코사인이 1차에 지배되어
    저항 부하의 잔차까지 SMPS 를 닮은 것으로 보인다. 실측에서는 `test_9`(저항만)가
    25.65 -> 25.35W 로 거의 안 변하고 `test_5` 는 12.25 -> 3.52W 다 (frac=1.0).
    ③ **물리 천장(`caps`, 기본 `ABSORB_CAP_W`)을 넘겨 주지 않는다** (12.149.3/.4).
       ③이 없던 판이 프로젝터를 창의 32.7% 에서 55W 위로 올렸고(최대 344W),
       프로젝터만 막았더니 미니PC 가 98.5W(격리 상한 29.8W)까지 올라갔다.
    """
    out = np.array(P, dtype=np.float64, copy=True)
    if frac <= 0:
        return out
    # ── 못 박은 기기는 안 받는다 (2026-09-02, 12.149.2) ────────────────────
    # `snap_power` 는 **"이 기기의 전력은 격리 측정으로 안다"** 는 주장이다.
    # 거기에 잔차를 더하면 그 주장과 모순이고, 실제로 스냅이 세운 프로젝터를
    # 다시 밀어 올린다 (중앙|오차| 0.00 -> 3.72W). 스냅 대상은 빼고 나눈다.
    skip = set(exclude or ())
    cols = [j for j, a in enumerate(apps) if a in SMPS_GROUP and a not in skip]
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
    # ── 계통 임피던스 보정 (2026-09-02, 12.152) ──────────────────────────
    # 손실은 `pred += harm_offset` 을 하는데(12.148) 흡수는 안 했다. 그래서
    # 흡수는 **보정 안 된 잔차**를 보고 배분을 정했다. 넣으면 그 잔차가
    # 3차 이상 노름 중앙 101.9 -> 60.5 mA (−41%) 로 줄고 방향이 바뀐다
    # (보정 전후 cos 중앙 −0.06 — 사실상 다른 벡터다).
    if harm_offset is not None:
        pred_h = pred_h + np.asarray(harm_offset, dtype=np.float64)
    R = (obs_harm - pred_h)[:, hh, :]
    Rc = R[:, :, 0] + 1j * R[:, :, 1]
    rn = np.linalg.norm(Rc, axis=1)

    cw = ABSORB_CAP_W if caps is None else caps
    lim = np.full(len(cols), np.inf)
    for i, j in enumerate(cols):
        if apps[j] in cw:
            lim[i] = float(cw[apps[j]])

    SS = np.array([sig[j][hh, 0] + 1j * sig[j][hh, 1] for j in cols])   # (C, Hh)
    if mode == "pq":
        # ── 무효전력을 방정식으로 넣는다 (2026-09-02, 12.153) ──────────────
        # 12.152 이 막힌 자리: 총전력 잔차 6W 는 실재하고 배분돼야 하는데
        # **미지수 셋에 방정식이 하나**(Σx = 6W)뿐이라 고조파가 방향만 준다.
        # 고조파로 크기까지 정하면 흡수가 꺼진다 (잔차 6.62 = 흡수 끈 것과 같음).
        #
        # 12.133 이 잰 **두 번째 판별자**가 그 자리를 채운다. SMPS 쌍 d′ 이
        # Q/P 2.31~4.64 vs 고조파 0.91~1.85 로 2.2~2.5배 낫고, 충전기·미니PC 는
        # **전력 자체보다 Q/P 가 훨씬 안정**하다 (폭/중앙 0.156/0.289 vs
        # 0.722/1.162). 그래서 식이 둘이 된다:
        #
        #     Σ x_k            = 총전력 잔차     (하드. 6W 는 다 나간다)
        #     Σ (Q/P)_k · x_k  = 무효 잔차       (연성. **새로 쓰는 것**)
        #     Σ x_k · sig_k    ~ 고조파 잔차     (연성. 남은 한 자유도)
        #
        # 12.153 이 잰 것: 프로젝터 |오차| 중앙 9.44 -> 7.34W, **p90 20.45 ->
        # 12.46W**. 6W 를 그대로 다 배분하면서 꼬리가 8W 준다.
        #
        # ⚠ 이 `Q` 는 기본파 무효분이 아니라 `sign(phase)·sqrt(S²−P²)` 다.
        #   가산성은 **경험적 근거**다 (12.133: 66창 중 62창). 물리 유도가 아니다.
        if qp is None or q_observed is None:
            raise ValueError("mode='pq' 는 qp 와 q_observed 가 필요합니다")
        qq = np.asarray(qp, dtype=np.float64)
        qc = qq[cols]
        room0 = np.clip(lim[None, :] - out[:, cols], 0.0, None) * (gate[:, cols] >= min_gate)
        resid_q = (np.asarray(q_observed, dtype=np.float64)
                   - ((out * qq[None]).sum(1) + (standby * qq[None]).sum(1) + float(noise_q)))
        Ah = np.concatenate([SS.T.real, SS.T.imag], axis=0)        # (2Hh, C)
        bh = np.concatenate([Rc.real, Rc.imag], axis=1)            # (n, 2Hh)
        sh = np.maximum(np.linalg.norm(bh, axis=1), 1e-6)
        sq = np.maximum(np.abs(resid_q), 1e-6)
        # 정규방정식을 창별 스칼라 배율로 조립한다 (열이 셋뿐이라 싸다)
        Hm = Ah.T @ Ah                                             # (C, C)
        Qm = np.outer(qc, qc)
        M = (Hm[None] / (sh ** 2)[:, None, None]
             + (w_q ** 2) * Qm[None] / (sq ** 2)[:, None, None])
        v = ((bh @ Ah) / (sh ** 2)[:, None]
             + (w_q ** 2) * (resid_q / sq ** 2)[:, None] * qc[None])
        L = np.maximum(np.trace(M, axis1=1, axis2=2), 1e-9)[:, None]
        nlive = np.maximum((room0 > 0).sum(1, keepdims=True), 1)
        x = np.clip(np.maximum(resid, 0.0)[:, None] / nlive, 0.0, room0)
        for _ in range(150):
            x = np.clip(x - (np.einsum("nij,nj->ni", M, x) - v) / L, 0.0, room0)
            tot = x.sum(1, keepdims=True)
            ok2 = (tot[:, 0] > 1e-9) & (resid > 0)
            x[ok2] = np.clip(x[ok2] * (resid[ok2, None] / tot[ok2]), 0.0, room0[ok2])
        W = x
    elif mode == "nnls":
        # ── "이 잔차를 만들려면 각 기기가 몇 W 인가" (12.152) ──────────────
        # 코사인은 지문마다 **독립**으로 닮음을 재고 그 비로 나눈다. SMPS 3종
        # 지문이 11.9도 안에 몰려 있어(12.145) 셋이 비슷해지고, 배분이 사실상
        # 균등해진다. NNLS 는 셋을 **같이** 풀어 겹침을 처리한다.
        Am = np.concatenate([SS.T.real, SS.T.imag], axis=0)             # (2Hh, C)
        bm = np.concatenate([Rc.real, Rc.imag], axis=1).T               # (2Hh, n)
        L = float(np.linalg.norm(Am, 2) ** 2) + 1e-9
        hi = np.clip(lim[None, :] - out[:, cols], 0.0, None)
        X = np.zeros((len(out), len(cols)))
        for _ in range(200):
            X = np.clip(X - ((Am @ X.T - bm).T @ Am) / L, 0.0, hi)
        W = X
    else:
        W = np.zeros((len(out), len(cols)))
        for i in range(len(cols)):
            W[:, i] = np.clip(
                np.real(Rc @ np.conj(SS[i])) / (np.linalg.norm(SS[i]) * rn + 1e-12),
                0.0, None)
    W = np.where(gate[:, cols] >= min_gate, W, 0.0)

    # ── 물리 상한을 지키며 나눈다 (2026-09-02, 12.149.3) ──────────────────
    # **처음에는 한 번에 나눴고 그것이 상한을 뚫었다.** `apply_postproc` 의
    # 55W 상한(`CAP_W`)은 "프로젝터는 이보다 못 낸다" 는 물리 제약인데,
    # 흡수가 그 뒤에 돌면서 창의 32.7% 에서 그것을 넘겼다 (최대 344W).
    # 상한은 후처리 순서로 지킬 수 없다 — **흡수 자체가 알아야 한다.**
    #
    # 그래서 수도관 채우기(water-filling)로 나눈다: 여유(`cap − 현재`)만큼만
    # 받고, 넘친 몫은 아직 여유가 있는 기기끼리 다시 나눈다. 받을 기기가
    # 다 차면 남은 잔차는 **그대로 잔차로 둔다** — 없는 곳에 만들지 않는다.
    if mode == "pq":
        # `x` 가 이미 제약을 만족하는 **배분량**이다 (합 = 잔차, 천장 안).
        out[:, cols] = np.clip(out[:, cols] + W, 0.0, None)
        return out

    add = np.zeros_like(W)
    room = np.clip(lim[None, :] - out[:, cols], 0.0, None)   # (n, C)
    left = resid.copy()
    if limit_by_harm:
        # ── 증거보다 많이 주지 않는다 (12.152) ────────────────────────────
        # 12.152 이 잰 것: 총전력 잔차 중앙 +5.96W 인데 **고조파가 지지하는
        # 것은 1.18W** 다 (비 0.01, 0.8~1.25 안에 드는 창이 5.2%). 지금 구조는
        # 총전력 잔차를 *반드시* 배분하므로 넘치는 몫이 프로젝터로 가고,
        # 그래서 프로젝터 |오차| 중앙이 6.85 -> 9.58W 로 나빠진다.
        # 여기서는 **고조파가 지지하는 만큼만** 주고 남는 것은 잔차로 둔다.
        cap_h = W.sum(1)
        left = np.sign(left) * np.minimum(np.abs(left), cap_h)
    live = W > 0
    for _ in range(len(cols) + 1):
        w = np.where(live, W, 0.0)
        tw = w.sum(1)
        ok = (tw > 0) & (np.abs(left) > 1e-9)
        if not ok.any():
            break
        share = np.zeros_like(w)
        share[ok] = w[ok] / tw[ok, None] * left[ok, None]
        # 음의 잔차(과대예측)는 상한과 무관하다. 양수 쪽만 여유로 자른다.
        take = np.where(share > 0, np.minimum(share, room - add), share)
        add += take
        left = left - take.sum(1)
        live = live & ((room - add) > 1e-9)
        if not live.any():
            break
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
