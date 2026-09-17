# -*- coding: utf-8 -*-
"""**컨덕턴스 예산** — 저항 부하의 판정·크기를 물리로 고친다 (14.335).

지금까지 스크래치에만 있던 것(`veto.py`·`pinb.py`·`triple.py`·`floorsep*.py`)을
옮긴 것이다. 학습이 필요 없고, 한 사이클(0.39ms)만 보며, **지연이 0** 이다.

## 네 칸

```
                판정 (게이트 단자 o[10])              크기 (슬롯 단자 o[0:5])
  위 (과잉)   **거부권** Ĝ < G_k − 여유 -> 지운다      **천장**  P <= G_k·V²
              14.322: C포트오탐 32 -> 24 · 6/6 · p=0.031 · A·B·D 씨앗차 전부 0
  아래 (추락) **바닥**  조합이 k 를 요구 -> 세운다      **고정**  P := G_{k,s}·V²
              14.334: §12.5 유령 띠 안 회수 97.9~99.2%  14.327: 총잔차 6.4 -> 5.5
                      헛세움 1/5,715 · 드라이ON 745칸 0   포트 단독창 96 -> 65W
```

윗줄+크기는 §35.1 이 이미 쟀다 (거부권+고정 = 합 60 -> 52). 빠졌던 칸이 **바닥**이다.

## 왜 바닥이 필요한가 — §12.3

```
  포트 `p_raw` 는 여덟 체크포인트 어디서나 **1,398~1,438W** 다 (실측 1,432W).
  무너지는 것은 **게이트 하나**고 (0.002 ~ 0.986), 유령창과의 스피어만이 **−0.976** 이다.
  포트는 상태가 하나라 게이트가 곧 진폭이고, 떨어진 와트를 오븐이 받는다.
  ⇒ "갑자기 추락" 과 "오탐" 은 두 병이 아니라 **한 축의 양끝**이다 (사용자 지적).
```
크기를 묶는 처치(천장·고정)는 §12.2 가 적은 대로 **다른 단자**라 판정 자에 못 닿는다.
바닥·거부권이 닿는 자리가 그 자리다.

## ⚠ 표는 **|V₁| 기준**이다

`Ĝ` 는 분해기가 `I_h = G·V_h` 로 푸는 양이라 **기본파 전압**에 대해 정의된다.
`power_features[:,4]`(vrms)로 재면 THD 만큼(0.25%) 어긋난다. 아래 표는 전부
`|voltage_harmonics_complex[:,0]|` 로 쟀다 — 14.322·14.327 의 측정이 나온 판과 같다.

## ⚠⚠ 표 밖에서 여유를 먹는 것 둘 (14.337)

문턱을 정하는 것은 이웃 간격이 아니라 **기기별 식별 여유** — "k 가 든 조합" 과
"k 가 없는 조합" 사이의 최소 거리다:
```
  포트 **0.635** · 드라이 0.497 · 핫플 0.497 · 오븐 2.590 mS
```
그런데 후보 표에 **없는** 것이 둘 실린다:
```
  선풍기   강풍은 **전기적으로 저항**이다 (PF 0.998 · h3/h1 0.033 대 저항 0.028).
           템플릿이 못 가져가고 Ĝ 에 **그대로 실린다** — 캐시 2만창에서 확인 (14.338):
             선풍기 전력   21~25W(G 0.43)  25~32W(0.60)  32~50W(0.75)
             Ĝ 잔차 중앙   **+0.226**      **+0.411**    **+0.583** mS
           = 명목 G 의 53~78% 가 Ĝ 로 샌다. 포트 식별 여유 0.635 와 같은 급이다
  에어컨   ⚠ **내 예측이 틀렸다.** §34 의 "+6.5%" 를 편향으로 읽고 28mS 에서 +1.83 mS 가
           실린다고 적었는데, 캐시 2만창(에어컨 813창)에서 재니 **편향이 없다**:
             에어컨 W  5~100  100~300  300~500  500~900   (꺼짐 −0.173)
             잔차 중앙 −0.015  −0.059   −0.383   −0.119 mS
           대신 **σ 가 0.265 -> 0.749 (2.8배)** 로 벌어진다. 즉 **잡음**으로 온다
```
★ 방향이 다르다 — 이걸 갈라 둬야 한다:
```
  선풍기  **편향**이고 위쪽이다 -> 거부권·바닥이 **덜 선다** (안 하는 쪽. 보수적)
  에어컨  **잡음**이고 방향이 없다 -> 원리상 양쪽으로 틀릴 수 있다
```
그런데 캐시에서 재니 에어컨 구간에서도 **헛세움이 0.000%** 이고 회수만 43.8% -> 10.0% 로
떨어진다. 판정식이 **여유 비교**(`lo1 + 여유 < lo0`)라서 대칭 잡음은 주로 **회수**를 깎지
정밀도를 안 깎는다. 그래도 "안전하다"가 아니라 **"이 자료에서는 안 터졌다"** 로 적어 둔다.

## ⚠ 거부권과 바닥은 **동시에 못 선다**

⚠⚠ 14.345 — 처음에 *"구조적으로 배타적"* 이라 적었는데 **틀렸다.** 오븐을 바닥에 넣자
`run_gate_gbudget_cache` [6] 이 18만 칸 중 **3칸**에서 둘이 같이 서는 것을 잡았다.
거부권 문턱 바로 **아래 좁은 띠**에서 가능하다 — 예: `Ĝ=22.9` 면 거부권이 서고
(22.9 < 24.957−2), 동시에 오븐을 넣은 조합이 더 맞는다 (없이 19.389 dist 3.51 ·
넣어 24.957 dist 2.06). 포트는 켜짐 상태가 하나라 **우연히** 안 겹쳤을 뿐이다.
⇒ `apply` 가 `fm &= ~vm` 로 **배타성을 강제한다. 거부권이 이긴다** — 거부권은 씨앗
  여섯으로 검증된 처치고(14.322 p=0.031) 바닥은 새 칸이다. 모순이면 보수적인 쪽이다.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import itertools

import numpy as np

#: 상태별 컨덕턴스 (mS). 단독 녹화(`processed_data/npz`)의 통전 구간 중앙값이다.
#:
#: 재는 법 — `is_on & (state_id == s) & (P > 100)` 의 `1e3·P/|V₁|²` 중앙, 녹화 간 중앙.
#: 가장자리 3사이클을 깎아도 **0.03% 안**이라 (중앙값이 릴레이 과도에 둔하다) 안 깎는다.
#: 14.333 이 잰 산포: 가장자리 깎은 CV **0.26~0.84%** · 같은 기기 두 녹화 **0.08~0.26%**.
#:
#: ⚠ **니크롬선 상태만 들어 있다.** 오븐 s1(FAN_LIGHT 14.6W)·핫플 s1(ARMED_IDLE 0.6W)은
#:   팬·표시등이라 `P > 100` 문턱에서 자연히 빠진다. 저항이 아니므로 고정하면 안 된다.
#: ⚠ **드라이기는 반드시 두 상태로 갈라 둔다.** 14.327 이 하나로 뭉갰다가 전력오차가
#:   146 -> **628W** 로 터졌다. s1 은 반파(9.45)이고 핫플(9.94)과 0.50mS 차다.
#: ⚠ `postproc.RESISTIVE_OHM` 을 **대체하지 않는다.** 그 dict 는 `resistive_match` 등
#:   열 곳이 보는 다른 물건이다 (12.112). 여기는 예산 경로 전용이다.
PIN_MS: Dict[str, Dict[int, float]] = {
    "electiric_kettle": {1: 28.122},
    "hair_dryer":       {1: 9.446, 2: 18.814},     # s1 반파 · s2 전파
    "hotplate":         {2: 9.943},
    "oven":             {2: 24.957},
}

#: 예산이 다루는 기기 (표의 열쇠와 같다 — 순서를 고정해 둔다).
RESISTIVE: Tuple[str, ...] = ("electiric_kettle", "hair_dryer", "hotplate", "oven")

#: ★ **바닥을 거는 기기.** 지금은 포트 하나다.
#:
#: 근거 — 14.334 가 실측 5파일에서 잰 것:
#: ```
#:   §12.5 유령 띠 안 회수   test_2 **97.9%** · test_5 **99.2%**
#:   포트ON·드라이ON  98.61% 회수  ·  포트OFF·드라이ON 745칸에서 헛세움 **0.00%**
#:   전체 헛세움 **1 / 5,715** (0.02%)
#: ```
#: ⚠⚠ **14.344 에서 오븐을 넣었다.** 14.334 는 *"오븐에는 안 건다"* 고 적었는데, 그 근거는
#:   14.325 의 `Ĝ_sum` 단독 **오븐 재현 44.9%** 였다. 그건 **라벨** 기준이라 듀티가 섞인
#:   수다 (오븐은 라벨 ON 의 54~71%가 비통전). 바닥이 하는 일은 *"모델이 껐는데 예산이
#:   요구한다"* 이고 그건 다른 물음이다. 새 바닥 `cnn_v49base` 씨앗 여섯에서 잰 것:
#: ```
#:   사중 + 총량 고정 · 바닥=포트      A 8 B13 C16 D2  합 40
#:   사중 + 총량 고정 · 바닥=포트+오븐  A10 B**3** C16 D2  합 **31**  6/6 p=0.031
#: ```
#:   **B미탐 13 -> 3.** 대가는 A유령 8 -> 10 이고, 그 자는 쌍안정이라 얇다
#:   ([[nilm-oven-ghost-ruler-is-two-episodes]]).
FLOOR_APPS: Tuple[str, ...] = ("electiric_kettle", "oven")

VETO_MARGIN_MS = 2.0      #: 14.322 — 0.5~3.0mS 에서 결과가 같았다. 가운데를 쓴다
FLOOR_MARGIN_MS = 1.0     #: 14.334 — 1.0 에서 회수 70.9%/띠 안 97.9~99.2% · 헛세움 0.02%
CLAIM_W = 300.0           #: "모델이 이 기기를 켰다" 로 볼 전력 문턱 (14.322 와 같다)
#: ★ 14.353 — **추론 운영점.** 조합 머리의 초과 주장 꺾기 (`net.comb_over`).
#: 14.352 가 같은 체크포인트 여섯에 이 항만 켜서 잰 값 (실측 5,382창):
#:   조합 단독 합 15 · |r| 8.7W · 1.8%자 0.54/0.46
#:   ★ 꺾기 0.05  합 **8** · |r| 9.2W · 1.8%자 **0.13/0.11** · C포트오탐 9 -> **2**
#:   꺾기+거부권  합 8 (같다) — **꺾기가 거부권을 삼킨다. 후처리가 필요 없다**
#: 0.05 와 0.20 이 같은 값이고 0.01 은 되레 나쁘다(합 9). 평평한 자리를 골랐다.
#: ⚠ `comb_over` 키가 **없는** 옛 체크포인트에만 얹는다 — 키가 있는 판은 그 값을 따른다.
COMB_OVER_OP = 0.05


# ── 표에서 파생 ──────────────────────────────────────────────────────────────
def app_states(app: str) -> List[Tuple[int, float]]:
    """`[(state_id, G_mS), …]` — 그 기기의 통전 상태들. 표에 없으면 빈 목록."""
    return sorted(PIN_MS.get(app, {}).items())


def max_ms(app: str) -> float:
    """그 기기가 낼 수 있는 **최대** 컨덕턴스."""
    d = PIN_MS.get(app)
    return max(d.values()) if d else 0.0


def min_ms(app: str) -> float:
    """그 기기가 켜져 있다면 **적어도** 내는 컨덕턴스. **거부권 문턱이 이것이다.**

    ⚠⚠ 14.338 — 처음에 `max_ms` 를 썼다가 캐시 관문이 잡았다. 거부권의 주장은
      *"이 기기는 **아예** 못 켜져 있다"* 이므로 **제일 작은 켜짐 상태**로 재야 한다.
      `max_ms` 를 쓰면 드라이기가 **반파**(9.446)로 켜져 있을 때 `Ĝ < 18.814 − 2` 가
      성립해 **참으로 켜진 기기를 지운다** — 캐시에서 참 통전 4,643창의 **22.2%** 였다.
      (실측 관문이 이걸 못 잡은 까닭: 거기서는 라벨로 재는데 반파 구간이 적었다.)
    """
    d = PIN_MS.get(app)
    return min(d.values()) if d else 0.0


def _cols(apps: Sequence[str], subset: Iterable[str]) -> List[int]:
    s = set(subset)
    return [j for j, a in enumerate(apps) if a in s]


def _combos(others: Sequence[str]) -> np.ndarray:
    """다른 저항들이 낼 수 있는 **총 컨덕턴스**의 모든 값 (OFF 포함, mS)."""
    opts = [[0.0] + [g for _, g in app_states(a)] for a in others]
    if not opts:
        return np.zeros(1)
    return np.array(sorted({sum(c) for c in itertools.product(*opts)}))


# ── ⓐ 거부권 (위·판정) ───────────────────────────────────────────────────────
def veto_mask(power: np.ndarray, g_ms: np.ndarray, apps: Sequence[str],
              margin: float = VETO_MARGIN_MS, claim_w: float = CLAIM_W) -> np.ndarray:
    """모델이 켰다는데 예산이 **못 대는** 자리 (N,K) bool.

    14.322: *"조합을 맞히는 것(58%)이 아니라 불가능한 주장을 거부하는 것 — 훨씬 약한 요구"*.
    """
    power = np.asarray(power, np.float64)
    g = np.asarray(g_ms, np.float64)
    out = np.zeros(power.shape, bool)
    for j, a in enumerate(apps):
        if a not in PIN_MS:
            continue
        out[:, j] = (power[:, j] >= claim_w) & (g < min_ms(a) - margin)
    return out


# ── ⓑ 바닥 (아래·판정) ───────────────────────────────────────────────────────
def floor_mask(power: np.ndarray, g_ms: np.ndarray, apps: Sequence[str],
               margin: float = FLOOR_MARGIN_MS, claim_w: float = CLAIM_W,
               floor_apps: Sequence[str] = FLOOR_APPS,
               allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """모델이 껐는데 예산이 **k 없이는 설명이 안 되는** 자리 (N,K) bool.

    ```
      lo0 = min_S |Ĝ − ΣG(S)|            k 를 뺀 조합 중 최선
      lo1 = min_S |Ĝ − (G_k + ΣG(S))|    k 를 넣은 조합 중 최선
      세운다  <=>  lo1 + 여유 < lo0
    ```

    Args:
        allowed: `(N,K)` bool. 다른 저항의 후보를 **제한**한다 (예: 모델 주장).
            `None` 이면 **전부 허용** — 설명할 길이 많아지므로 **보수적**이고
            라벨도 모델도 안 쓴다. 14.334 가 잰 것이 이 판이다.

    ⚠ 제한판은 회수가 99.2% 로 오르지만 헛세움이 1 -> **10창**이 되고, 그 10창이
      **포트가 아예 없는 녹화**(test_1/3/4)에서 난다. 새 유령을 만드는 종류다.
      기본을 `None`(전부 허용)으로 두는 까닭이다.
    """
    power = np.asarray(power, np.float64)
    g = np.asarray(g_ms, np.float64)
    out = np.zeros(power.shape, bool)
    for j, a in enumerate(apps):
        if a not in floor_apps or a not in PIN_MS:
            continue
        gk = max_ms(a)
        others = [b for b in RESISTIVE if b != a]
        oc = _cols(apps, others)
        quiet = power[:, j] < claim_w                      # 모델이 안 켠 자리만
        if not quiet.any():
            continue
        idx = np.nonzero(quiet)[0]
        if allowed is None:
            base = _combos(others)
            lo0 = np.abs(g[idx, None] - base[None, :]).min(1)
            lo1 = np.abs(g[idx, None] - (gk + base)[None, :]).min(1)
        else:
            al = np.asarray(allowed, bool)
            lo0 = np.empty(len(idx))
            lo1 = np.empty(len(idx))
            for t, i in enumerate(idx):
                sub = [b for b, c in zip(others, oc) if al[i, c]]
                base = _combos(sub)
                lo0[t] = np.abs(g[i] - base).min()
                lo1[t] = np.abs(g[i] - (gk + base)).min()
        out[idx, j] = (lo1 + margin) < lo0
    return out


# ── ⓒ 고정 (크기) ────────────────────────────────────────────────────────────
def pin_power(power: np.ndarray, v1: np.ndarray, apps: Sequence[str],
              on_mask: np.ndarray) -> np.ndarray:
    """켜졌다고 본 자리의 전력을 `G_{k,s}·|V₁|²` 로 바꾼다. `power` 를 복사해 돌려준다.

    상태는 **모델 예측에 가장 가까운 것**을 고른다 (14.327). 상태를 안 고르고
    뭉개면 드라이기에서 전력오차가 146 -> 628W 로 터진다.
    """
    q = np.array(power, np.float64, copy=True)
    v = np.asarray(v1, np.float64)
    for j, a in enumerate(apps):
        st = app_states(a)
        if not st:
            continue
        m = np.asarray(on_mask, bool)[:, j]
        if not m.any():
            continue
        cand = np.array([g for _, g in st])[None, :] * 1e-3 * v[m][:, None] ** 2
        q[m, j] = cand[np.arange(int(m.sum())),
                       np.argmin(np.abs(cand - q[m, j][:, None]), 1)]
    return q


# ── ⓓ 총량 고정 (크기 · 사용자 제안 14.344) ─────────────────────────────────
def pin_total(power: np.ndarray, g_ms: np.ndarray, v1: np.ndarray, apps: Sequence[str],
              fill: bool = True, fill_min_w: float = 100.0) -> np.ndarray:
    """저항 **총 전력**을 `Ĝ·|V₁|²` 로 묶는다. 배분(비율)은 모델 것을 그대로 쓴다.

    사용자: *"잘못된걸 거부하면 거기에 뭘 채워놔야지 그냥 비워놓아 버리면 어떡하니"*.
    맞다 — 거부권만 걸면 test_5 총잔차가 **20.7 -> 39.5W** 로 터진다. 211~216초에 포트
    1,100W 를 지웠는데 참 주인인 오븐(Ĝ=24.91 ≈ 24.957)은 18W 로 눌린 채였다.

    **기기별 고정(`pin_power`)과 다른 물건이다.** 저쪽은 조합을 **정해야** 하고 이쪽은
    **안 정해도** 된다. 그리고 Ĝ 의 강점이 여기 있다 (14.343, 캐시 2만창):
    ```
      계량기  저항 **총 전력** 오차 중앙 **−5.4W** · σ 17.3W · 상대 **−0.39%**
      분류기  조합 정확 Ĝ 단독 84.5% · Ĝ+반파 94.3% · 모델 94.9% · Ĝ+모델 **99.7%**
      ⇒ **총량은 물리, 배분은 모델.**
    ```
    ⚠ **거부권 뒤에 와야 한다.** 거부권 없이 총량만 맞추면 오탐이 유일한 주장일 때
      재정규화가 **그놈을 키운다** — 실측에서 C오탐 20 -> **22** (14.344).

    Args:
        fill: 거부 뒤 아무도 안 남았는데 예산이 `fill_min_w` 넘게 있으면 **조합 맞춤**으로
            채운다. 실측에서는 한 번도 안 걸렸고(구멍이 안 생긴다) 캐시의 총량보존 맞바꿈
            에서는 272창이 걸렸다. 구멍 방지용 안전판이라 기본을 켜 둔다.
    """
    q = np.array(power, np.float64, copy=True)
    cols = [j for j, a in enumerate(apps) if a in PIN_MS]
    if not cols:
        return q
    tot = np.asarray(g_ms, np.float64) * 1e-3 * np.asarray(v1, np.float64) ** 2
    s = q[:, cols].sum(1)
    hot = s > 1e-9
    sh = np.zeros_like(q[:, cols])
    sh[hot] = q[hot][:, cols] / s[hot][:, None]
    q[:, cols] = sh * tot[:, None]
    if fill:
        empty = (~hot) & (tot > fill_min_w)
        if empty.any():
            #: 조합 맞춤 — 후보 총합이 Ĝ 에 제일 가까운 것을 고른다
            names = [apps[j] for j in cols]
            opts = [[(0.0, None)] + [(gg, s_) for s_, gg in app_states(a)] for a in names]
            combos = list(itertools.product(*opts))
            cg = np.array([sum(y[0] for y in c) for c in combos])
            cv = np.array([[y[0] for y in c] for c in combos])
            j = np.argmin(np.abs(np.asarray(g_ms, np.float64)[empty, None] - cg[None, :]), 1)
            idx = np.nonzero(empty)[0]
            q[idx[:, None], np.array(cols)[None, :]] = (
                cv[j] * 1e-3 * np.asarray(v1, np.float64)[empty, None] ** 2)
    return q


# ── 넷을 한 번에 ─────────────────────────────────────────────────────────────
def apply(power: np.ndarray, g_ms: np.ndarray, v1: np.ndarray, apps: Sequence[str],
          *, veto: bool = True, floor: bool = True, pin: bool = True,
          total: bool = False,
          veto_margin: float = VETO_MARGIN_MS, floor_margin: float = FLOOR_MARGIN_MS,
          claim_w: float = CLAIM_W, floor_apps: Sequence[str] = FLOOR_APPS,
          allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """거부권 -> 바닥 -> 고정 -> **총량 고정**. 전부 끄면 입력 그대로다 (비트 동일).

    ★ 14.344 — 실측 5,382창 · `cnn_v49base` 씨앗 여섯:
    ```
      팔                              A유령 B미탐 C오탐 D미탐   합 | 총잔차|r| | 짝검정
      바닥                             12   18   20    6   56 |  12.2W  |  —
      veto+floor+pin                   8   18   16    2   44 |  14.2W  | 6/6 p=0.031
      ★ **+ total** (바닥=포트+오븐)     10    3   16    2 **31**| **11.0W**| 6/6 p=0.031
                                          씨앗차 [−15,−26,−10,−15,−28,−33]
    ```
    ⚠ **디코더 없이** 31 이다 — 한 사이클 · 0.39ms · 지연 0 · 인과적.
    ⚠ 셋이 각자 다른 칸을 고친다: 거부권 C · **총량 B** · 바닥 D.

    Returns:
        `(power, info)` — `info` 에 `veto`·`floor`·`on` 마스크가 들어 있다.
    """
    q = np.array(power, np.float64, copy=True)
    vm = np.zeros(q.shape, bool)
    fm = np.zeros(q.shape, bool)
    if veto:
        vm = veto_mask(q, g_ms, apps, veto_margin, claim_w)
        q[vm] = 0.0
    if floor:
        fm = floor_mask(q, g_ms, apps, floor_margin, claim_w, floor_apps, allowed)
        #: ⚠⚠ 14.345 — **배타성을 코드로 만든다.** "구조적으로 배타적" 이라고 적어 뒀는데
        #  `run_gate_gbudget_cache` [6] 이 반증했다 — 오븐을 바닥에 넣자 18만 칸 중 **3칸**
        #  에서 둘이 같이 섰다. 거부권 문턱(`Ĝ < min_ms−여유`) 바로 **아래 좁은 띠**에서,
        #  오븐을 넣은 조합이 더 잘 맞을 수 있다 (예: Ĝ=22.9 -> 없이 19.389 dist 3.51 ·
        #  넣어 24.957 dist 2.06). 포트는 상태가 하나라 우연히 안 겹쳤을 뿐이다.
        #  ⇒ **거부권이 이긴다.** 거부권은 씨앗 여섯으로 검증된 처치이고(14.322 p=0.031)
        #    바닥은 새 칸이다. 모순이 나면 보수적인 쪽을 남긴다.
        fm &= ~vm
    on = ((q >= claim_w) | fm)
    for j, a in enumerate(apps):
        if a not in PIN_MS:
            on[:, j] = False
    if pin:
        q = pin_power(q, v1, apps, on)
    elif floor:
        #: 고정을 안 쓰면 바닥이 세운 자리에 크기가 없다. §12.3 이 잰 대로 모델의
        #: `p_raw` 는 멀쩡하지만(1,398~1,438W) 게이트가 곱해져 0 이 되어 있다.
        #: 그 경우 호출자가 `info["floor"]` 로 직접 채워야 한다 — 여기서 짓지 않는다.
        pass
    if total:
        #: ★ 마지막이다 — 거부·바닥·고정이 정한 **비율** 위에 물리가 **총량**을 씌운다
        q = pin_total(q, g_ms, v1, apps)
    if not (veto or floor or pin or total):
        q = np.array(power, np.float64, copy=True)
    return q, {"veto": vm, "floor": fm, "on": on}


# ── Ĝ 를 뽑는 배선 (녹화당 한 번 짓고 사이클마다 푼다) ────────────────────────
class Budget:
    """`physdecomp` 배선을 한 곳에 모은다. 스크래치 넷이 같은 25줄을 베끼고 있었다.

    ⚠ **템플릿에 에어컨을 반드시 넣는다** (14.319). 없으면 에어컨 전류가 틀린
      전압법칙으로 `Ĝ_sum` 에 흡수되어 **+47.7%** 가 된다. 넣으면 +6.5% 다.
    ⚠ 전압은 **15차수 전부**를 쓴다 (`VOLT_ORDERS` 8차수는 모델 **입력 배치**이고
      계측기는 15차까지 준다 — 14.318 의 정정).
    """

    V15 = tuple(range(1, 16))
    FCM = ("beam_projector", "laptop_charger", "minipc")
    FCM_RANGES = {"beam_projector": (35., 55.), "laptop_charger": (10., 75.),
                  "minipc": (5., 30.)}

    def __init__(self, apps: Sequence[str], v_ref: np.ndarray, pool=None,
                 volt_re0: int = 33, n_harm: int = 15, volt_orders=None,
                 motor_cols: bool = False, sigs=None):
        #: 14.338 — **차수는 인자다.** 실측 npz 는 15차수를 다 주지만 **학습 캐시의
        #  세밀 갈래에는 홀수 8차수뿐**이다 (`VOLT_ORDERS`). 캐시 위에서 이 모듈을
        #  쓰려면 여기를 못 박으면 안 된다.
        #  ⚠ Ĝ 자체는 차수를 줄여도 거의 안 변한다 (6차수 0.111 대 15차수 0.107 mS, 14.331).
        self.vo = tuple(self.V15 if volt_orders is None else volt_orders)
        from src.model.fcmtab import build_table, fcm_devices, template
        from src.model.net import harmonic_signatures_by_state
        import src.model.physdecomp as PD

        #: ★ 14.378 — `sigs=(sig, us)` 를 주면 **풀을 안 연다.** 배포 묶음에는
        #: `SegmentPool` 이 없어서 구운 표(`runtime_tables.npz`)로 들어온다.
        #: ⚠ 안 주면 **옛 경로 그대로**다 (연구 쪽은 비트 동일).
        if sigs is not None:
            sig, us = sigs
            sig = np.asarray(sig, np.float64)
            us = np.asarray(us, bool)
            if sig.shape[0] != len(apps):
                raise ValueError("구운 지문의 기기 수가 다르다: %d != %d"
                                 % (sig.shape[0], len(apps)))
        else:
            if pool is None:
                from src.synthesis.segment_pool import SegmentPool
                pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
            sig, us = harmonic_signatures_by_state(pool, list(apps))
        tab = build_table(fcm_devices(), self.FCM_RANGES, v_ref, n_p=10)
        T = [np.stack([c.real, c.imag], -1)
             for c in (template(tab[d], 20., 1.0) for d in self.FCM)]
        #: ★ 14.359 — **기둥을 가질 기기**. 에어컨은 처음부터 있었고 선풍기는 없었다.
        #: `motor_cols` 로 켠다 (기본 옛 동작 = 에어컨만 = **비트 동일**).
        #:
        #: 왜 선풍기인가 — 풀에서 잰 값이 그 이유다:
        #: ```
        #:   fan s1 21.5W  h3/h1 0.114 · THD 0.147 · ∠I1 **+31.3°**
        #:   fan s2 29.4W  h3/h1 0.079 · THD 0.086 · ∠I1 +21.6°
        #:   fan s3 37.6W  h3/h1 0.034 · THD **0.038** · ∠I1 **+3.2°**
        #:   (저항 넷은 THD 0.021~0.038 · ∠I1 −0.1~+1.0°)
        #: ```
        #: s3 은 **전기적으로 저항과 구분이 안 된다.** 기둥이 없으면 그 전류가 갈 데가
        #: `G` 기둥뿐이라 Ĝ 가 부풀고, §37 이 잰 *"선풍기 명목의 53~78%가 Ĝ 로 샌다"* 가
        #: 그것이다. 37.6W 는 216V 에서 **0.8 mS** — 포트 식별 여유 0.635 보다 크다.
        #: ⚠ Ĝ 는 지금 조합 머리의 대들보라 **좋아지는지 관문으로 재고** 켠다.
        #: ⚠ 선풍기 지문은 상태별 전력 폭이 **1.01배**(오차 ≤0.5%)라 템플릿으로 이상적이다.
        self.motor_apps = tuple(["air_conditioner"]
                                + (["fan"] if motor_cols else []))
        self.col_src = [("fcm", d, None) for d in self.FCM]
        for _a in self.motor_apps:
            if _a not in list(apps):
                continue
            _k = list(apps).index(_a)
            for s in range(sig.shape[1]):
                if us[_k, s] and np.abs(sig[_k, s]).max() > 0:
                    T.append(sig[_k, s].astype(np.float64))
                    self.col_src.append(("sig", _a, int(s)))
        self.T = np.stack(T)
        k = 2 + len(self.T)
        self.lam = np.zeros((k, k))
        self.lam[0, 0] = self.lam[1, 1] = PD.LAM_G
        for i in range(2, k):
            self.lam[i, i] = PD.LAM_T
        self.nn = np.ones(k, bool)
        self.nn[1] = False                       # 서셉턴스는 부호가 자유롭다
        self.w = np.ones(2 * n_harm)
        self.k = k
        self.volt_re0 = int(volt_re0)

    def g_sum(self, raw: np.ndarray) -> np.ndarray:
        """`raw` (B, C, N) -> `Ĝ_sum` (B, N) **mS**."""
        import src.model.physdecomp as PD
        A = PD.build_design(raw, self.T, volt_orders=self.vo, volt_re0=self.volt_re0)
        y = PD.observed(raw)
        th, _ = PD._solve(A, y, self.w ** 2, self.lam, np.zeros(self.k), self.nn)
        return 1e3 * th[..., 0]


class LiveGhat:
    """실시간 갈래의 **Ĝ 추정기** (14.378). `run_plot_real.solve_ghat` 의 온라인 판.

    연구 갈래는 녹화 파일 전체의 전압 중앙값으로 기준을 잡는데 실시간에는 미래가
    없다. 그래서 **지금 창의 중앙값**을 쓴다. 잰 차이:
    ```
      test_1 창 60개  Ĝ 차 중앙 **0.0029 mS** · 최대 0.0079
      자    Ĝ 잡음 σ 0.265 mS · 제일 좁은 식별 여유(포트) 0.635 mS
      ⇒ 여유의 **1.2%** 다. 표를 창마다 다시 지을 이유가 없다 (한 번에 1.18s)
    ```
    ⚠ 자리가 바뀌면(다른 건물) 다시 지어야 한다 — `|V₁|` 이 `rebuild_rel` 넘게
      움직이면 다시 짓는다. 한 녹화 안의 표류는 **1.9%** 였다.
    ⚠ `sigs` 를 주면 `SegmentPool` 을 안 연다 (배포 묶음은 풀이 없다).
    """

    def __init__(self, apps: Sequence[str], sigs=None, volt_orders=None,
                 volt_re0: int = 33, rebuild_rel: float = 0.05):
        self.apps = list(apps)
        self.sigs = sigs
        self.volt_re0 = int(volt_re0)
        if volt_orders is None:
            from src.model.inputs import VOLT_ORDERS as _VO
            volt_orders = _VO
        self.vo = tuple(volt_orders)
        self.rebuild_rel = float(rebuild_rel)
        self.bud: Optional["Budget"] = None
        self.v1_ref = 0.0
        self.n_build = 0
        self.last = float("nan")

    def v_ref(self, win: np.ndarray) -> np.ndarray:
        """(1, C, N) 창 -> 15차 기준 전압 페이저 (그 창의 중앙값)."""
        nv, r0 = len(self.vo), self.volt_re0
        med = (np.median(win[0, r0:r0 + nv], 1) + 1j * np.median(win[0, r0 + nv:r0 + 2 * nv], 1))
        v15 = np.zeros(15, complex)
        for s_, h in enumerate(self.vo):
            v15[h - 1] = med[s_]
        return v15

    def of(self, win: np.ndarray, cycle: int) -> float:
        """창의 `cycle` 번째 사이클에서 `Ĝ` [mS]."""
        v15 = self.v_ref(win)
        v1 = float(abs(v15[0]))
        if self.bud is None or abs(v1 - self.v1_ref) > self.rebuild_rel * max(self.v1_ref, 1.0):
            self.bud = Budget(self.apps, v15, volt_re0=self.volt_re0,
                              volt_orders=self.vo, sigs=self.sigs)
            self.v1_ref = v1
            self.n_build += 1
        tc = np.array([int(cycle)], np.int64)
        self.last = float(self.bud.g_sum(np.asarray(win, np.float64)[:, :, tc])[0, 0])
        return self.last
