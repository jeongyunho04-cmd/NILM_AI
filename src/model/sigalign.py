# -*- coding: utf-8 -*-
"""사전을 굽기 전에 **녹화별 위상 회전**을 맞춘다 (14.395, §52 (가)).

무엇을 고치나 — `sig = median(I_h/P)` 는 그 기기의 **모든 녹화**를 섞어 복소 중앙값을
낸다. 그런데 녹화마다 SMPS 의 정류 도통 타이밍이 달라 페이저가 `h 비례`로 돌아 있다
(§47.4·§52.1). 돌아 있는 페이저들의 복소 중앙값은 **크기를 잃는다**:

```
  충전기 coh(=|중앙 페이저| / 중앙 |페이저|)   h13 **0.820** · h15 **0.599**
  ⇒ `sig[충전기, 고차]` 가 40% 작다 -> `L_harm` 이 h15 크기를 맞추려고
    **충전기 와트를 올린다** (§46.2 가 잰 +4.1%/+20.8% 가 그 방향이다)
```

고침은 **녹화마다 회전 하나**를 빼고 나서 중앙값을 내는 것이다. 참 배분에서도
못 줄이던 잔차가 판별 차수(h9·h11·h13)에서 이렇게 내려간다 (§52.2):

```
  충전기 0.593 -> **0.416** (−30%) · 빔 0.136 -> **0.108** (−21%)
  미니PC 0.292 -> **0.270** (−7.5%)
  ⚠ 대조군 오븐 0.475 -> 0.461 (**−3.0%**) — **공짜 이득이 아니다**
```

⚠⚠ 규약 셋. 셋 다 재서 정했고, 안 지키면 **오히려 나빠진다** (§52.3):

```
  ① **h1 은 안 돌린다**  순수 지연이면 h1 도 k·1 돌아야 하는데 h1 의 녹화 간 변동은
     그 회전보다 훨씬 작다 (충전기 irr **0.032**). 돌리면 없는 회전을 만들어
     **0.060 으로 두 배** 나빠진다
  ② **전체 위상은 안 옮긴다**  녹화별 회전에서 **중앙값을 뺀다**. 안 그러면 기기마다
     위상이 제멋대로 옮겨져 `Σ_k P_k·sig_k` 의 **기기 간 상대 위상**이 깨진다
     [[fix-the-phase-reference-before-comparing-phasors]]
  ③ **회전은 기기의 성질이지 상태의 성질이 아니다**  그래서 표를 **기기마다 한 번**
     내서 `harmonic_signatures` 와 `harmonic_signatures_by_state` 가 **같은 표**를 쓴다.
     따로 내면 두 입구의 위상 기준이 갈린다
     [[pin-the-two-entry-points-against-each-other]]
```
"""
from typing import Dict, Sequence, Tuple

import numpy as np

#: 회전을 적합하는 차수 — **h1 을 빼고**(규약 ①) **h13·h15 도 뺀다**(규약 ④).
#:
#: ⚠⚠ 규약 ④ 는 관문 [4]가 잡아서 생겼다. 처음엔 h15 까지 넣었는데 충전기 `coh_h13`
#: 이 **0.820 -> 0.747 로 나빠졌다**. 각도를 펴 보니 지연 모형이 **h11 까지만 맞는다**:
#: ```
#:   충전기 녹화1  측정  12.5  20.1  27.6  33.4  42.7 | **77.8  121.4**
#:               k·h  17.4  29.0  40.5  52.1  63.7 |   75.3   86.9
#:   충전기 녹화3  측정  -2.4  -3.4  -4.9  -5.8  -6.1 | **+6.7  +33.5**   <- k≈0 인데도 튄다
#:   ⇒ h13·h15 의 튐은 **회전이 아니다.** 녹화마다 다른 방향으로 도는 게 아니라
#:     **모두 같은 쪽으로** 끌린다 = 공통 가산항
#: ```
#: 13.84.38 이 그 정체를 이미 쟀다 — **세션 배경**이 파일 간 3.6~22.6mA 인데
#: *"h15 에서는 미니PC 12W(5.7mA)보다 크다"*. 신호보다 큰 배경이 얹힌 자리에서
#: 위상차를 지연으로 읽으면 **없는 지연을 만든다**. 그 항은 `noise_sig` + `harm_offset`
#: 이 맡는다 — 여기서 건드릴 것이 아니다.
#: ⇒ **적합은 신호가 배경 위에 있는 h3~h11 에서만 한다.** 회전 자체는 전 차수에 건다
#:   (지연이면 그게 맞다). [[check-conditioning-before-believing-a-fit]]
FIT_ORDERS = (3, 5, 7, 9, 11)
#: ★ **채택 문턱.** 지연 모형이 실제로 맞는 기기에만 건다. 쓸기로 갈린 것 (14.395):
#:   SMPS   미니PC **0.84** · 충전기 **0.89** · 빔 **0.98**
#:   저항   오븐 **0.14** · 핫플 **0.02**          <- 6배 간격. 0.5 는 한가운데다
#: ⚠ 이 자가 없으면 저항의 h3+ 가 거의 0 이라 **잡음에 k 를 맞춰** 사전을 망친다
#:   (관문 [5]가 오븐·핫플 **+5.1%** 로 잡았다). [[measure-separability-before-building-a-separator]]
MIN_R2 = 0.5
#: 녹화 하나가 회전을 낼 수 있는 최소 사이클. `harmonic_signatures_by_state` 의 문턱과 같다.
MIN_CYCLES = 200


def rec_of(act) -> str:
    """활성화가 어느 녹화에서 왔나. `run_diag_sigfloor`(13.84.31)와 같은 규약."""
    return str(getattr(act, "source_file", getattr(act, "recording", "?")))


def med_phasor(z: np.ndarray) -> np.ndarray:
    """실·허를 **따로** 중앙값 — `harmonic_signatures` 가 쓰는 그 식."""
    return np.median(z.real, 0) + 1j * np.median(z.imag, 0)


def fit_k(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
    """`Δ∠(h) = k·h` 를 **절편 없이** 적합해 `(도/차수, R²)` 를 낸다.

    ⚠ 절편을 두지 않는 것이 핵심이다 — 절편이 있으면 `h 비례` 구조가 풀려
    아무 위상차나 흡수하고, 그러면 **없는 회전을 만든다**.

    ⚠ **크기로 가중한다** (`w_h = |ref_h|`). 이것이 "파형을 시간축에서 맞춘다" 의
    선형화다 — 신호가 큰 차수가 지연을 정한다. 안 하면 신호가 작은 고차의 잡음이
    적합을 끌어간다 (오븐 R² 0.36 -> **0.14** 로 내려가 대조군이 제대로 걸러진다).
    """
    hs, ds, ws = [], [], []
    for h in FIT_ORDERS:
        i = h - 1
        if i < len(ref) and abs(ref[i]) > 0 and abs(cur[i]) > 0:
            hs.append(float(h))
            ds.append(np.degrees(np.angle(cur[i] / ref[i])))
            ws.append(float(abs(ref[i])))
    if len(hs) < 4:
        return 0.0, 0.0
    hs, ds, ws = np.asarray(hs), np.asarray(ds), np.asarray(ws)
    k = float((ws * hs * ds).sum() / max((ws * hs * hs).sum(), 1e-12))
    r = ds - k * hs
    r2 = float(1 - (ws * r ** 2).sum()
               / max((ws * (ds - np.average(ds, weights=ws)) ** 2).sum(), 1e-12))
    return k, r2


def rotation_vector(n_harm: int) -> np.ndarray:
    """곱할 차수 벡터 — **h1 자리는 0** 이다 (규약 ①)."""
    v = np.arange(1, n_harm + 1, dtype=np.float64)
    v[0] = 0.0
    return v


def recording_rotations(pool, appliances: Sequence[str], n_harm: int = 15
                        ) -> Dict[str, Dict[str, float]]:
    """기기마다 `{녹화: 빼야 할 회전(도/차수)}` 를 낸다. **중앙값이 이미 빠져 있다**.

    ⚠ 뽑는 사이클은 `harmonic_signatures` 와 **글자 그대로 같다**
    (`target_power_w > max(0.5·steady_p90, 1.0)`) — 통전 구간이다. 선택 규칙이
    갈리면 두 입구가 또 갈린다.
    """
    out: Dict[str, Dict[str, float]] = {}
    for app in appliances:
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        by: Dict[str, list] = {}
        for a in acts:
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                by.setdefault(rec_of(a), []).append(
                    np.asarray(a.net_harmonics_complex)[m]
                    / np.maximum(np.asarray(a.target_power_w)[m], 1e-6)[:, None])
        if len(by) < 2:
            continue                       # 녹화가 하나면 맞출 것이 없다 (**항등**)
        packed = {r: np.concatenate(v) for r, v in by.items()}
        ref = med_phasor(np.concatenate([packed[r] for r in sorted(packed)]))
        fits = {r: (fit_k(ref, med_phasor(z)) if len(z) >= MIN_CYCLES else (0.0, 0.0))
                for r, z in packed.items()}
        #: ★ 규약 ④ — **지연 모형이 맞는 기기에만** 건다. 안 맞으면 통째로 항등이다
        #:   (아무 회전도 안 준다 -> 그 기기의 사전은 **비트 동일**).
        if float(np.median([v[1] for v in fits.values()])) < MIN_R2:
            continue
        ks = {r: v[0] for r, v in fits.items()}
        km = float(np.median(list(ks.values())))          # 규약 ② — 전체 위상 보존
        out[app] = {r: (k - km) for r, k in ks.items()}
    return out


def rotation_report(pool, appliances: Sequence[str]) -> Dict[str, tuple]:
    """기기마다 `(R² 중앙, k 표준편차, 녹화 수, 채택했나)` — 관문과 로그가 읽는다."""
    rep: Dict[str, tuple] = {}
    got = recording_rotations(pool, appliances)
    for app in appliances:
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 0.5 * pool.get_steady_power_w(app)
        by: Dict[str, list] = {}
        for a in acts:
            m = a.target_power_w > max(thr, 1.0)
            if m.any():
                by.setdefault(rec_of(a), []).append(
                    np.asarray(a.net_harmonics_complex)[m]
                    / np.maximum(np.asarray(a.target_power_w)[m], 1e-6)[:, None])
        if len(by) < 2:
            rep[app] = (float("nan"), 0.0, len(by), False)
            continue
        packed = {r: np.concatenate(v) for r, v in by.items()}
        ref = med_phasor(np.concatenate([packed[r] for r in sorted(packed)]))
        fits = [fit_k(ref, med_phasor(z)) if len(z) >= MIN_CYCLES else (0.0, 0.0)
                for z in packed.values()]
        rep[app] = (float(np.median([v[1] for v in fits])),
                    float(np.std([v[0] for v in fits])), len(by), app in got)
    return rep


def derotate(per_w: np.ndarray, deg_per_order: float, n_harm: int = 0) -> np.ndarray:
    """한 녹화의 와트당 페이저 (N, H) 를 `deg_per_order` 만큼 **되돌린다**."""
    if deg_per_order == 0.0:
        return per_w
    v = rotation_vector(n_harm or per_w.shape[1])
    return per_w * np.exp(-1j * np.radians(deg_per_order) * v)[None]
