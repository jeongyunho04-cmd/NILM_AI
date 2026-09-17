# -*- coding: utf-8 -*-
"""녹화별 위상 회전 — **측정 도구다. 학습에는 쓰지 마라** (14.395 -> 14.396 반증).

⚠⚠⚠ **결론부터: 사전을 녹화로 정렬하는 것은 틀렸다 (§54).** 이 모듈은 그것을 재느라
지었고, (나)`sig(P)` 를 지을 때 쓸 적합 도구(`fit_k`·`derotate`)만 남긴다.

무엇을 잘못 봤나 — §51.3 은 이렇게 읽었다:
```
  `sig = median(I_h/P)` 가 녹화를 섞으니 **돌아 있는 페이저의 복소 중앙값이 크기를 잃는다**
  (충전기 coh h15 **0.599**) -> `L_harm` 이 고차 크기를 맞추려고 **와트를 올린다**
```
**틀렸다.** 줄어듦은 버그가 아니라 **올바른 접기**다. 손실이 사전으로 와트를 되풀 때
`P* = Re(z·conj(sig))/|sig|²` 인데, `sig = ρ·s` (ρ = 회전 산포가 주는 줄어듦) 이면
`E[P*] = P·E[cosθ]/ρ = P` 로 **편향이 없다**. 길이를 되찾으면 오히려 `E[P*] = ρ·P` 로
**낮게** 치우친다. 실측이 그대로다 (되푼 와트, 참값 1.000):
```
  차수        h3     h5     h7     h9    h11    h13    h15
  지금 sig  1.006  0.990  0.968  0.928  0.852  0.761  **0.789**
  정렬 sig  0.992  0.972  0.951  0.932  0.888  0.798  **0.660**   <- 더 나쁘다
```

⚠⚠ 그리고 **내 관문이 그것을 가렸다.** 관문은 `irr(정렬한 사이클, 정렬한 sig)` 를 쟀는데
**학습 자료의 사이클은 안 돌아간다.** 사이클을 그대로 두고 재면 부호가 뒤집힌다:
```
  칸              관문의 자(오라클)   **올바른 자**
  충전기 s2        −33.6%            **−1.1%**
  미니PC s2         −4.3%            **+5.4%**
  빔 s2            −22.4%            **+11.3%**
  빔 s1              —               **+113.4%**
```
⇒ 관문이 잰 것은 *"창마다 녹화를 알 수 있다면 얼마나 줄어드나"* 라는 **오라클**이었다.
  추론에서 녹화 정체성은 **관측할 수 없다.** §47.5 가 이미 적어 두었다 —
  *"회전은 조각마다 무작위라 **배울 수 있는 함수가 아니다**"*.

★ **(나)는 살아남는다** — 부하 회전은 **창의 전력에서 계산할 수 있다**. 같은 올바른 자로:
```
  충전기 s2 −2.6% · 미니PC s2 −5.6% · **충전기 s1 −44.4%** (전력 폭 10~35W)
  빔·저항은 전력 폭이 없어 `a = 0` (저절로 항등)
```
관련: [[gate-what-the-consumer-reads]] · [[dont-ablate-to-zero-to-find-a-cause]]
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


def cell_rotation_ok(per_w: np.ndarray, recs: np.ndarray, sizes: np.ndarray) -> bool:
    """이 **칸**(기기 x 상태)에서 지연 모형이 쓸 만한가 — `MIN_R2` 로 판정한다.

    ⚠⚠ 왜 필요한가 (14.395b). 회전 **값**은 녹화의 성질이라 기기마다 한 번 내는 것이
    맞다(규약 ③). 그런데 *그 모형이 맞는지* 는 **칸마다 다르다** — 신호 대 배경이
    상태마다 다르기 때문이다. 관문이 s2 만 보고 있어서 못 봤던 것:

    ```
      미니PC **s1** (중앙 8.7W)  칸 R² **0.18**  -> 회전을 걸면 irr **+21.4% 악화**
      미니PC  s2   (19.2W)      칸 R²  0.84   -> −4.3%
      충전기 s1 0.97 · s2 0.89  ·  빔 s1 0.96 · s2 0.98   (넷 다 −21~−36% 개선)
    ```
    ⇒ 저전력 상태는 h3~h11 이 배경에 묻혀 적합이 **잡음을 맞춘다**. 저항을 거른
      그 문턱이 이 칸도 그대로 거른다 — **새 자를 만들지 않는다**.
    """
    off, blocks = 0, {}
    for r, n in zip(recs, sizes):
        blocks.setdefault(str(r), []).append(per_w[off:off + int(n)])
        off += int(n)
    packed = {r: np.concatenate(v) for r, v in blocks.items()}
    packed = {r: z for r, z in packed.items() if len(z) >= MIN_CYCLES}
    if len(packed) < 2:
        return False
    ref = med_phasor(np.concatenate([packed[r] for r in sorted(packed)]))
    r2 = [fit_k(ref, med_phasor(z))[1] for z in packed.values()]
    return float(np.median(r2)) >= MIN_R2
