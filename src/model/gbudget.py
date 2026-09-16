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
           템플릿이 못 가져가고 Ĝ 에 그대로 실린다 -> **0.432~0.765 mS** (21.3~37.7W @222V)
           포트 식별 여유 0.635 **보다 크다**
  에어컨   §34.319 가 잰 잔차 편향 **+6.5%** -> 28 mS 에서 **+1.83 mS**
           ⚠⚠ **실측 5파일에 에어컨이 한 칸도 없다.** 오늘 잰 헛세움 1/5,715 에
              에어컨 구간이 **0칸**이다 — 그 구간은 **안 재 본 것**이지 안전한 게 아니다
```
★ 다만 **방향이 우리 편이다** — 둘 다 Ĝ 를 **위로** 민다. 그러면 거부권은 덜 서고
  (안 지운다) 바닥도 덜 선다 (안 세운다). **틀리는 쪽이 아니라 안 하는 쪽**이다.
  실제로 바닥 회수가 70.9% 에 그친 녹화(test_2)가 **선풍기가 83% 켜진** 녹화다.

## ⚠ 거부권과 바닥은 **동시에 못 선다**

거부권은 `Ĝ < G_k − 여유`, 바닥은 `k 를 넣은 조합이 더 맞을 때`다. `Ĝ` 가 `G_k` 보다
여유만큼 아래면 `k` 를 넣는 순간 과설명이라 바닥이 안 선다. 구조적으로 배타적이지만
**단언하지 않는다** — `run_gate_gbudget` [4] 가 실측 전 창에서 확인한다.
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
#: ⚠ 오븐에는 **안 건다** — 14.325 가 `Ĝ_sum` 단독 오븐 재현을 **44.9%** 로 쟀다
#:   (포트는 99.7%). 오븐은 듀티로 꺼져 있는 시간이 라벨 ON 의 54~71% 다 (14.328).
FLOOR_APPS: Tuple[str, ...] = ("electiric_kettle",)

VETO_MARGIN_MS = 2.0      #: 14.322 — 0.5~3.0mS 에서 결과가 같았다. 가운데를 쓴다
FLOOR_MARGIN_MS = 1.0     #: 14.334 — 1.0 에서 회수 70.9%/띠 안 97.9~99.2% · 헛세움 0.02%
CLAIM_W = 300.0           #: "모델이 이 기기를 켰다" 로 볼 전력 문턱 (14.322 와 같다)


# ── 표에서 파생 ──────────────────────────────────────────────────────────────
def app_states(app: str) -> List[Tuple[int, float]]:
    """`[(state_id, G_mS), …]` — 그 기기의 통전 상태들. 표에 없으면 빈 목록."""
    return sorted(PIN_MS.get(app, {}).items())


def max_ms(app: str) -> float:
    """그 기기가 낼 수 있는 **최대** 컨덕턴스. 거부권 문턱이 이것을 쓴다."""
    d = PIN_MS.get(app)
    return max(d.values()) if d else 0.0


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
        out[:, j] = (power[:, j] >= claim_w) & (g < max_ms(a) - margin)
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


# ── 넷을 한 번에 ─────────────────────────────────────────────────────────────
def apply(power: np.ndarray, g_ms: np.ndarray, v1: np.ndarray, apps: Sequence[str],
          *, veto: bool = True, floor: bool = True, pin: bool = True,
          veto_margin: float = VETO_MARGIN_MS, floor_margin: float = FLOOR_MARGIN_MS,
          claim_w: float = CLAIM_W, floor_apps: Sequence[str] = FLOOR_APPS,
          allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """거부권 -> 바닥 -> 고정. 셋 다 끄면 **입력을 그대로** 돌려준다 (비트 동일).

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
    if not (veto or floor or pin):
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
                 volt_re0: int = 33, n_harm: int = 15):
        from src.model.fcmtab import build_table, fcm_devices, template
        from src.model.net import harmonic_signatures_by_state
        import src.model.physdecomp as PD

        if pool is None:
            from src.synthesis.segment_pool import SegmentPool
            pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        sig, us = harmonic_signatures_by_state(pool, list(apps))
        tab = build_table(fcm_devices(), self.FCM_RANGES, v_ref, n_p=10)
        T = [np.stack([c.real, c.imag], -1)
             for c in (template(tab[d], 20., 1.0) for d in self.FCM)]
        ka = list(apps).index("air_conditioner")
        for s in range(sig.shape[1]):
            if us[ka, s] and np.abs(sig[ka, s]).max() > 0:
                T.append(sig[ka, s].astype(np.float64))
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
        A = PD.build_design(raw, self.T, volt_orders=self.V15, volt_re0=self.volt_re0)
        y = PD.observed(raw)
        th, _ = PD._solve(A, y, self.w ** 2, self.lam, np.zeros(self.k), self.nn)
        return 1e3 * th[..., 0]
