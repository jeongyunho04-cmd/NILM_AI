# -*- coding: utf-8 -*-
"""① 프레임별 물리 분해 · ② 창 안 중앙 프레임 정밀화 (14.318, 사용자 제안 ①②).

**아직 모델에 안 붙였다.** 순수 함수로만 두고 관문(`run_gate_physdecomp`)이 참값으로
잰다 — 이득이 확인되기 전에 배선하지 않는다.

① 무엇을 푸나 — 프레임마다 **같은 최소제곱 층**:
```
  y(τ) = [Re I_h, Im I_h]_{h=1..15}                     (30,)  관측 고조파 전류
  A(τ) = [ V_h , j·V_h , T_m,r,h ]                      (30,K)
  θ̂(τ) = (AᵀW²A + Λ)⁻¹ (AᵀW²y + Λθ_prior) = [Ĝ_sum, B̂_sum, α̂]
```
* 저항 몫은 **모든 차수에서 같은 실수 G** 로 전압을 따라간다 -> 열 하나
* 비저항 몫(SMPS·모터)은 **기기·상태별 템플릿**이 가져간다 -> 열 M 개
* 기기별 G 는 **안 푼다.** 식 하나에 미지수 여럿이라 식별이 안 된다. 합 하나만 낸다

⚠⚠ **정정 (사용자 지적)** — 전압 고조파는 **15차수까지 측정된다.**
  `VOLT_ORDERS = (1,3,5,7,9,11)` 은 `to_raw45` 의 **피처 배치** 선택이지 측정 한계가 아니다.
  내가 `raw45` 를 보고 "측정이 6차수"라고 적었던 것은 틀렸다.
  잰 것 (`processed_data/npz` 의 `voltage_harmonics_complex`):
```
    h1 228.05V · h3 6.38 · h5 4.14 · h7 0.213 · h9 2.46 · h11 0.231
    **h13 0.336 · h15 0.396**   <- h7·h11 과 같은 급. 피처가 버린 것이다
    짝수차 0.010~0.100V = V1 의 4e-5~4e-4 — 사실상 0
```
  ⇒ 정정이 실제로 여는 것은 **h13·h15 두 차수**다. 짝수차의 관측 전류는 전압이 진짜로
    없으므로 선형 G 로는 설명할 수 없고 비선형 몫이 맞다. 지금 구현은 `raw45` 를 쓰므로
    6차수뿐이다 — 15차수를 쓰려면 원시 배치를 넓혀야 한다 (**아직 안 했다**).

⚠ **모터에 열을 준다** (사용자 지적 "선풍기나 에어컨만 보완하면"). 스펙의
  `A = [V, jV, T_SMPS]` 에는 선풍기·에어컨이 갈 곳이 없는데 에어컨은 311±200W 로 작지 않다.
  열이 없으면 `Ĝ_sum` 에 **틀린 전압법칙으로 흡수**된다. 그래서 `T` 를 "비저항 전부"로 넓혔다.

⚠ **템플릿은 와트당 상수 페이저다** (`harmonic_signatures_by_state`). 13.84.23 이 SMPS 에서
  잔차 17~27% 를 쟀고 원인을 **도통각이 전력에 따라 변하는 비선형**으로 지목했다.
  그 잔차는 갈 곳이 G·B 밖에 없으므로 **`Ĝ_sum` 을 오염시킨다.** 관문 [1]·[3] 이 그것을 잰다.
  `sig_model.sig(P,V)` 가 보정판인데, 먼저 상수판으로 재서 **보정이 필요한지**를 본다.

② 왜 한 프레임으로 부족한가 — `Ĝ_sum(τ)` 한 칸은 잡음이 크다. 창 안에서 **중앙과 같은
상태인 프레임만** 모아 다시 푼다:
```
  e(τ) = |ΔP|/σ_P + |Δ∠I₁|/σ_φ     전이 강도 (단순 특징 — 순환 방지)
  E(τ) = 중앙 t 와 τ 사이 e 의 누적합
  w(τ) = exp(−E(τ)²/2κ²)            전이를 넘으면 0 에 가까워 안 섞인다
  θ̂_c  = argmin Σ_τ w(τ)‖W(y−Aθ)‖² + ‖Λ(θ−θ_prior)‖²
```
⚠ 스펙은 *"전이가 없으면 창 안 전압 흔들림으로 복소 ZIP 이 저절로 작동한다"* 고 하는데,
  **창 안 전압이 σ/μ 0.227%(세밀 10초)·0.955%(광역 60초) 밖에 안 움직인다** (창 **사이**는
  4.296%). 정임피던스와 정전력의 서명 차이가 전력의 0.45% 라 조건수가 거기서 죽는다.
  ⇒ ZIP 분리는 **기대하지 않는다.** ②는 "같은 상태 프레임을 모아 잡음을 줄이는 것" 으로만 쓴다.
  저항↔비저항 분리는 ZIP 이 아니라 **고조파 모양**이 한다.

전부 미분 가능하다 (`argmin` 이 닫힌 형태고 비음수 투영은 ReLU 다). `kappa` 는 학습
가능한 값으로 둘 수 있다. 실시간이면 `center=None` 로 **끝 프레임 기준**(과거만)을 쓴다.

## ★ 작동 구성 (14.319~320, 측정으로 확정)

```
  A = [ V_h , j·V_h , T_SMPS(FCM 셋) , T_에어컨 ]   · 전압 **15차수** · NNLS
  저항만 **−1.9%** · 에어컨 켜짐 **+6.5%** · 선풍기 켜짐 **+1.1%**
```
기준은 `g_ref = Re(I₁V₁*)/|V₁|² − (비저항 전력)/|V₁|²` 다 — 모델을 안 쓴다.

**열을 하나씩 넣고 뺀 표** (저항만 / 에어컨 켜짐 / 선풍기 켜짐):
```
  G·B 만                −0.1  ·   —    ·   —      물리 자체는 **정확하다**
  + SMPS(FCM) 셋        −1.1  · +47.7  ·  +1.9    G 가 에어컨을 통째로 먹는다
  + **에어컨**           −1.9  ·  +6.5  ·  +1.1    ★ 7배 개선. **에어컨 열은 필수다**
  + 에어컨 + **선풍기**   −18.5 · −12.7  · −15.8    선풍기가 전부 망친다
  L1 희소성(3e-3)        효과 **없음** (6.5 -> 6.5) — 지렛대가 아니다
```

**왜 에어컨은 되고 선풍기는 안 되나 — 단독 녹화가 답한다**
```
  같은 전압에서 **순수 저항의 h3/h1 = V3/V1 = 0.028** 이 기준선이다

  에어컨   비h1 몫 28~36% · h3/h1 0.58~0.60 (저항의 **21배**) · THD 62~74%
          -> 인버터 정류 앞단. **강하게 갈린다.** 열을 줘야 한다
  선풍기   s1 약풍 비h1 1.75% · s2 0.62% · **s3 강풍 0.14% · h3/h1 0.033 · PF 0.998**
          -> 강풍 선풍기는 **전기적으로 저항이다.** `fan_s3` 가 G 열과 7.5도인 것은
             모델 결함이 아니라 **물리**다. 회로모델도 L1 도 템플릿도 못 고친다
          -> 처방: 열을 **주지 않고** G 가 먹게 둔다. 대가는 37.6W = 약 1~2% 다
```
⇒ 사용자 가설 *"선풍기나 에어컨만 보완하면"* 은 **반만 맞다** — 에어컨은 보완하면 되고
  (그리고 그것이 제일 큰 이득이다), 선풍기는 **보완할 수 없는 것**이라 포기가 답이다.

## 아직 안 한 것
```
  ⓐ 모델 배선 — 채널로 낼지 ⑤ 사전대조까지 갈지 미정
  ⓑ ② 정밀화를 **작동 구성**에서 다시 재기 (전이 차단은 통과, 잡음 감소는 미측정)
  ⓒ FCM 은 합에서 안 보인다 (SMPS ~100W 대 저항 1,300W). minipc 지문이 전력에 따라
     46.8% 움직이므로 **SMPS 자체를 읽을 때**는 필요하다 — 저항 추정에는 −1.1 대 −1.1
```
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from src.model.inputs import N_HARM, VOLT_IM0, VOLT_ORDERS, VOLT_RE0

#: 기본 릿지. `theta_prior` 가 0 이면 순수 수축이다.
LAM_G = 1e-3        #: G·B 열 — 물리량이라 약하게
LAM_T = 1e-2        #: 템플릿 열 — 서로 닮아서 더 세게 (cond 을 여기서 잡는다)
KAPPA = 6.0         #: ② 의 전이 누적 폭 (문턱 뺀 e 의 누적 단위)
E_THR = 6.0         #: 전이 강도의 문턱 — 중앙절대차의 이 배를 넘어야 '전이'다
NN_ITERS = 60       #: 비음수 투영 경사의 반복 수


def build_design(raw: np.ndarray, templates: np.ndarray,
                 volt_orders=None, volt_re0: int = VOLT_RE0) -> np.ndarray:
    """`A(τ)` 를 짓는다. `raw` (B,45,T) · `templates` (M,15,2) -> (B,T,30,2+M).

    열 배치:  0 = Ĝ_sum · 1 = B̂_sum · 2.. = 템플릿 α̂ (와트 단위)
    """
    #: 14.319 — 차수는 **인자**다. 측정은 15차수까지 있고 `VOLT_ORDERS` 6차수는
    #  `to_raw45` 의 피처 배치 선택이었다 (사용자 지적).
    vo = tuple(VOLT_ORDERS) if volt_orders is None else tuple(volt_orders)
    nv = len(vo)
    b, _, t = raw.shape
    m = len(templates)
    A = np.zeros((b, t, 2 * N_HARM, 2 + m), dtype=np.float64)
    for s, h in enumerate(vo):
        j = h - 1
        vr = raw[:, volt_re0 + s, :]                       # (B,T)
        vi = raw[:, volt_re0 + nv + s, :]
        A[:, :, j, 0] = vr                                 # Re I = G·Re V
        A[:, :, N_HARM + j, 0] = vi                        # Im I = G·Im V
        A[:, :, j, 1] = -vi                                # Re I = −B·Im V
        A[:, :, N_HARM + j, 1] = vr                        # Im I = +B·Re V
    A[:, :, :N_HARM, 2:] = templates[None, None, :, :, 0].transpose(0, 1, 3, 2)
    A[:, :, N_HARM:, 2:] = templates[None, None, :, :, 1].transpose(0, 1, 3, 2)
    return A


def observed(raw: np.ndarray) -> np.ndarray:
    """`y(τ)` (B,T,30) — 관측 고조파 전류 [Re 15 · Im 15]."""
    return np.concatenate([raw[:, 0:N_HARM, :], raw[:, N_HARM:2 * N_HARM, :]],
                          axis=1).transpose(0, 2, 1).astype(np.float64)


def _colnorm(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """열마다 **가중 노름**을 낸다 (0 이면 1 로 둔다).

    ⚠⚠ 이게 없으면 끝난다 — `V₁ ~ 215V` 인 G 열과 `~0.004 mA/W` 인 템플릿 열이
      **5자릿수** 차이라 `cond(H) = 1.9e8` 이 나오고 릿지가 열마다 다르게 걸린다.
      첫 판이 그래서 잔차 0.61 · 참값 회수 오차 49.7% 였다 (14.318).
      [[check-conditioning-before-believing-a-fit]]
    """
    n = np.sqrt(np.einsum("...kc,k,...kc->...c", A, w ** 2, A))
    return np.where(n > 1e-30, n, 1.0)


def _nnsolve(H: np.ndarray, g: np.ndarray, nn: np.ndarray,
             iters: int = NN_ITERS, l1: float = 0.0) -> np.ndarray:
    """`nn` 이 True 인 좌표에 **비음수**를 걸고 푼다 (투영 경사, 벡터화).

    ⚠⚠ **이게 빠져서 다 틀렸다** (14.318). 음수 α̂ 를 허용하면 템플릿 열이 저항 전류를
      **상쇄·증폭**하는 데 쓰인다 — 모터가 **꺼진** 창에서도 `Ĝ_sum` 오차가 73% 였다
      (모터가 켜진 창 68.6% 와 사실상 같다). 즉 공선 때문이 아니라 **부호 제약이 없어서**다.
      기기는 음의 전력을 못 먹는다.
    ⚠ `run_diag_dictcond` 머리글이 이미 *"계획 B 는 펼친 **NNLS** 반복"* 이라고 적어 뒀다.
      내가 평범한 릿지로 짓고 그 단어를 흘렸다 ([[search-the-log-before-proposing]]).

    투영 경사: `θ <- P(θ − η(Hθ − g))`, `η = 1/λ_max(H)`. ReLU 투영이라 거의 어디서나
    미분 가능하고 (사용자의 "전부 미분 가능" 요건), 반복 수가 고정이라 배치로 돈다.
    """
    th = np.linalg.solve(H, g[..., None])[..., 0]           # 제약 없는 해에서 출발
    if not nn.any():
        return th
    th = np.where(nn, np.maximum(th, 0.0), th)
    eta = 1.0 / np.maximum(np.linalg.norm(H, ord=2, axis=(-2, -1), keepdims=True), 1e-30)
    #: 14.320 — `l1` 은 **비음수 좌표에만** 건다. 근위 경사의 soft-threshold 인데
    #  비음수와 겹치면 그냥 `max(0, x − ηλ)` 다. 꺼진 기기의 α̂ 를 **정확히 0** 으로
    #  만든다 — 안 그러면 안 켜진 열이 저항 전류를 조금씩 훔친다 (에어컨 −0.8%p).
    for _ in range(iters):
        grad = np.einsum("...cd,...d->...c", H, th) - g
        th = th - eta[..., 0] * grad
        if l1 > 0.0:
            th = np.where(nn, np.maximum(th - eta[..., 0] * l1, 0.0), th)
        else:
            th = np.where(nn, np.maximum(th, 0.0), th)
    return th


def _solve(A: np.ndarray, y: np.ndarray, w2: np.ndarray, lam: np.ndarray,
           prior: np.ndarray, nn: np.ndarray, fw: Optional[np.ndarray] = None,
           l1: float = 0.0
           ) -> Tuple[np.ndarray, np.ndarray]:
    """가중 릿지. `fw` 가 있으면 프레임을 그 무게로 **합쳐** 하나를 푼다 (②).

    열을 **단위 노름으로 정규화해서** 풀고 마지막에 되돌린다 — 릿지가 모든 열에
    같은 세기로 걸려야 `lam_g`·`lam_t` 의 뜻이 산다.

    돌려주는 것: `theta`(원래 단위) · `H`(정규화된 정규방정식 — cond 을 밖에서 잰다)
    """
    w = np.sqrt(w2)
    if fw is None:
        sc = _colnorm(A, w)                                  # (B,T,C)
        An = A / sc[..., None, :]
        Aw = An * w2[None, None, :, None]
        H = np.einsum("btkc,btkd->btcd", An, Aw) + lam
        g = np.einsum("btkc,btk->btc", Aw, y) + prior * np.diagonal(lam, 0, -2, -1)
        th = _nnsolve(H, g, nn, l1=l1)
        return th / sc, H
    Af = A * np.sqrt(fw)[..., None, None]                    # 프레임 무게를 행에 실어
    sc = _colnorm(Af.reshape(len(A), -1, A.shape[-1]),
                  np.tile(w, A.shape[1]))                    # (B,C)
    An = A / sc[:, None, None, :]
    Aw = An * w2[None, None, :, None]
    H = np.einsum("bt,btkc,btkd->bcd", fw, An, Aw) + lam[:, 0]
    g = np.einsum("bt,btkc,btk->bc", fw, Aw, y) + prior * np.diagonal(lam[:, 0], 0, -2, -1)
    th = _nnsolve(H, g, nn, l1=l1)
    return th / sc, H


def decompose(raw: np.ndarray, templates: np.ndarray, hscale: np.ndarray,
              lam_g: float = LAM_G, lam_t: float = LAM_T
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """① 프레임별. -> `theta` (B,T,2+M) · `resid` (B,T) · `cond` (B,T) · `A`.

    `resid` 는 **상대**다 — `‖W(y−Aθ̂)‖ / ‖Wy‖`. 절대값은 I1 이 지배해 못 읽는다.
    """
    A = build_design(raw, templates)
    y = observed(raw)
    w = np.concatenate([1.0 / np.maximum(hscale, 1e-9)] * 2)        # (30,) 차수별 같은 무게
    w2 = w ** 2
    k = A.shape[-1]
    lam = np.zeros((k, k))
    lam[0, 0] = lam[1, 1] = lam_g
    for i in range(2, k):
        lam[i, i] = lam_t
    #: G(컨덕턴스)와 템플릿 진폭은 **>= 0**. B(서셉턴스)만 부호가 자유롭다
    #  (용량성이면 음수다). 이 한 줄이 [3] 을 72% -> 아래 표로 바꾼다.
    nn = np.ones(k, bool)
    nn[1] = False
    th, H = _solve(A, y, w2, lam, np.zeros(k), nn)
    r = y - np.einsum("btkc,btc->btk", A, th)
    resid = (np.linalg.norm(r * w, axis=-1)
             / np.maximum(np.linalg.norm(y * w, axis=-1), 1e-12))
    return th, resid, np.linalg.cond(H), A


def refine(raw: np.ndarray, templates: np.ndarray, hscale: np.ndarray,
           center: Optional[int] = None, kappa: float = KAPPA,
           lam_g: float = LAM_G, lam_t: float = LAM_T) -> Tuple[np.ndarray, np.ndarray]:
    """② 중앙(또는 끝) 프레임 정밀화. -> `theta_c` (B,2+M) · `w` (B,T) 프레임 무게.

    `center=None` 이면 **끝 프레임**(과거만) — 실시간용이고 지연이 안 는다.
    """
    b, _, t = raw.shape
    c = (t - 1) if center is None else int(center)
    p = raw[:, 30, :].astype(np.float64)
    z1 = raw[:, 0, :].astype(np.float64) + 1j * raw[:, N_HARM, :].astype(np.float64)
    ph = np.unwrap(np.angle(z1 + 1e-12), axis=-1)
    sp = np.maximum(np.median(np.abs(np.diff(p, axis=-1)), -1, keepdims=True), 1.0)
    sf = np.maximum(np.median(np.abs(np.diff(ph, axis=-1)), -1, keepdims=True), 1e-3)
    #: ⚠⚠ **문턱을 뺀다.** 첫 판은 중앙값으로만 나눠서 **잡음 프레임도 ~2씩 쌓였고**,
    #  60프레임이면 E~120 이라 κ=3 에서 exp(−120²/18)=0 — 유효 프레임이 3.5/120 이었다.
    #  `e` 는 "전이 강도" 라야 하므로 조용한 프레임은 **정확히 0** 이어야 한다.
    e = np.zeros((b, t))
    e[:, 1:] = np.maximum(np.abs(np.diff(p, axis=-1)) / sp - E_THR, 0.0) \
        + np.maximum(np.abs(np.diff(ph, axis=-1)) / sf - E_THR, 0.0)
    #: 중앙에서 **양쪽으로** 누적한다 — 전이를 넘은 프레임만 커진다.
    E = np.zeros((b, t))
    E[:, c + 1:] = np.cumsum(e[:, c + 1:], axis=-1)
    if c > 0:
        E[:, :c] = np.cumsum(e[:, 1:c + 1][:, ::-1], axis=-1)[:, ::-1]
    fw = np.exp(-(E ** 2) / (2.0 * kappa ** 2))
    A = build_design(raw, templates)
    y = observed(raw)
    w = np.concatenate([1.0 / np.maximum(hscale, 1e-9)] * 2)
    k = A.shape[-1]
    lam = np.zeros((b, 1, k, k))
    lam[:, :, 0, 0] = lam[:, :, 1, 1] = lam_g
    for i in range(2, k):
        lam[:, :, i, i] = lam_t
    nn = np.ones(k, bool)
    nn[1] = False
    th, _ = _solve(A, y, w ** 2, lam, np.zeros(k), nn, fw=fw)
    return th, fw


def resistive_templates(apps: Sequence[str], sig_state: np.ndarray, usable: np.ndarray,
                        resistive: Sequence[str]) -> Tuple[np.ndarray, list]:
    """비저항 기기·상태의 지문만 모아 템플릿 열로 만든다. -> (M,15,2) · 이름표."""
    T, names = [], []
    for k, a in enumerate(apps):
        if a in resistive:
            continue
        for s in range(sig_state.shape[1]):
            if not usable[k, s]:
                continue
            v = sig_state[k, s]
            if not np.isfinite(v).all() or np.abs(v).max() <= 0:
                continue
            T.append(v)
            names.append("%s_s%d" % (a, s))
    return (np.stack(T).astype(np.float64) if T else np.zeros((0, N_HARM, 2))), names
