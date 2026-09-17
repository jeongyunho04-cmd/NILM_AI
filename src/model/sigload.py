# -*- coding: utf-8 -*-
"""(나) **부하 의존 위상** — `sig_k,s,h(P) = sig · exp(−j·h·a_k·(ln P − ln P_ref))` (14.402).

무엇을 고치나. `L_harm` 의 사전은 `sig_state[기기, 상태, 차수]` — **상수 페이저 하나**다.
그런데 SMPS 는 통전 상태가 하나뿐인데 그 **상태 안**에서 부하가 위상을 돌린다.
물리: 정류기는 `v(t) > V_dc` 에서만 도통한다. 부하가 가벼우면 벌크 커패시터가 덜
방전돼 `V_dc` 가 높고 **도통 창이 좁아지고 뒤로 밀린다** — 전류 펄스가 Δt 밀리면
모든 차수가 `h·ω·Δt` 로 돈다.

★ **배우는 게 아니라 잰다.** 파라미터가 기기당 **하나**(a_k)뿐이고 `sig` 와 같은 규약으로
  풀에서 적합해 굽는다 -> 자유도를 안 늘리므로 §46 의 맞바꿈이 **원리상 반복 불가**.

⚠⚠ **왜 이 축만 여는가** — 전압 고조파도 후보였지만 §47.4b 가 **못 가른다**고 쟀다:
```
  녹화 사이 corr(k, THD_v) −0.52 · corr(k, |V3|/|V1|) −0.53 · corr(k, vrms) −0.54
  ⇒ 셋이 같이 움직인다. 한 녹화 **안**에서는 |V3|/|V1| 폭이 0.0006~0.0014 라
    (계통 고조파는 세션 안에서 상수) 갈라 볼 수가 없다
```
반면 전력은 §57 이 **부호 시험 + 토막 고정효과 + 대조군**으로 갈랐다:
```
  부호 시험 (표류면 오름 구간에서 부호가 뒤집혀야 한다)
    충전기 내림 +9.75 / 오름 **+9.79** (오름 구간 corr(lnP,t) **−0.042**)
    미니PC  내림 +4.31 / 오름 **+5.12**
  토막 고정효과 (10초보다 느린 **어떤 표류든** 흡수)
    충전기 +5.93~+7.85 · 미니PC +3.25~+4.62   — 부호·자릿수 유지
  ✅ 대조군 오븐 **+0.4%p** · 핫플 −0.3%p — 자유도가 만든 이득이 아니다
```

⚠ 사전 표는 **안 바꾼다.** 회전은 창의 전력에 달렸으므로 **손실 forward 에서** 건다
  (`_apply_pow_gain_state` 와 같은 자리). 그래서 `gbudget`·`sig_site`·진단이 쓰는
  `harmonic_signatures_by_state` 는 **손댈 필요가 없다** — 그것들은 손실이 아니다.
⚠ 추론에서 `P` 는 **모델의 예측**이다 (되먹임). `--pow-sig-instate` 와 같은 고리이고
  그래서 **둘을 같이 켜면 안 된다** (분모를 공유한다 — 14.172 꼴).
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: ★ **잰 값** (14.398~399 · §56~§57). 주기 단위(60Hz) 적합 · 공선 배제 · 대조군 통과.
#:   단위는 **도/차수/ln W**.
#:
#: ⚠⚠ **칸(기기 x 상태) 단위다. 기기 단위로 두면 안 된다.**
#:   처음에 `a` 를 기기 속성으로 두고 충전기 s1(+10.73) 대 s2(+10.99) 가 같은 것을
#:   근거로 삼았다. **충전기엔 맞고 미니PC엔 틀렸다** — 관문 [5]가 잡았다:
#: ```
#:     미니PC s1 에 s2 의 a(+4.1)를 걸었더니 잔차가 **+45.3% 나빠졌다**
#:     §56 이 이미 쟀다 — 미니PC s1 은 자기 적합이 a=−1.85 **R² 0.07 (잡음)** 이고
#:     칸 자체 이득도 −2.0% 뿐이다. 중앙 8.7W 라 h3~h11 이 배경에 묻힌다
#: ```
#:   ⇒ **칸마다 재고, 자기 적합이 선 칸에만 건다.** 저항을 거른 그 규율이다
#:     ([[measure-separability-before-building-a-separator]] · §54.5 ③)
#:
#: ```
#:   칸                       a       근거
#:   laptop_charger s1    **+10.73**  주기 49,953 · 차수별 a_h 평평함 **0.05**
#:   laptop_charger s2    **+10.99**  주기 102,055 · 평평함 0.14
#:   minipc s2             **+4.10**  주기 106,679 · 평평함 0.33 · 시간과 corr **+0.12**
#:   ─ 뺀 칸 ────────────────────────────────────────────────────────
#:   minipc s1              (0)       R² **0.07** = 잡음 · 걸면 **+45.3% 악화**
#:   beam_projector s1·s2   (0)       전력 폭 x1.05 · §56 고유 몫 −0.6%p
#:   저항 넷 · 에어컨        (0)       §56 대조군 (오븐 +0.4%p · 핫플 −0.3%p)
#: ```
#: ⚠ 에어컨 s3 는 고유 몫 −33.9%p 로 크지만 **녹화가 하나**다. 다음 후보로 둔다 (§57.3).
LOAD_ROT_A_CELL: Dict[Tuple[str, int], float] = {
    ("laptop_charger", 1): 10.73,
    ("laptop_charger", 2): 10.99,
    ("minipc", 2):          4.10,
}
#: 전력 폭이 이보다 좁으면 `a` 를 못 잰다 — `run_diag_chgrev` 가 쓰는 그 문턱
MIN_WIDTH = 1.15


def state_prefs(pool, appliances: Sequence[str], max_states: int = 5,
                min_cycles: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """칸마다 `ln P_ref` (K,S) 와 쓸 수 있는지 (K,S). **`harmonic_signatures_by_state`
    와 같은 규약으로 모은다** (`target_power_w > 1.0` · 그 상태).

    ⚠ 기준을 칸마다 잡는 것이 핵심이다. 기기 하나로 잡으면 상태 사이 전력 차이가
    회전으로 들어가 **상태 지문을 두 번 세는** 꼴이 된다 (§52.3 ②의 같은 함정).
    """
    from src.model.net import state_power_w
    K = len(appliances)
    lpref = np.zeros((K, max_states), np.float32)
    ok = np.zeros((K, max_states), bool)
    for j, app in enumerate(appliances):
        by: Dict[int, List[np.ndarray]] = {}
        for a in pool.appliance_activations.get(app, []):
            st = getattr(a, "state_id", None)
            if st is None:
                continue
            pw = state_power_w(a, False)
            m0 = pw > 1.0
            for s in np.unique(np.asarray(st)[m0]).astype(int):
                if not 0 < s < max_states:
                    continue
                m = m0 & (np.asarray(st) == s)
                if m.any():
                    by.setdefault(int(s), []).append(pw[m])
        for s, vs in by.items():
            p = np.concatenate(vs)
            if len(p) < min_cycles:
                continue
            lpref[j, s] = float(np.log(max(np.median(p), 1e-9)))
            #: 폭이 좁은 칸은 회전을 걸어도 `ln P − ln P_ref ≈ 0` 이라 저절로 항등이지만,
            #: **명시로 끈다** — 그래야 관문이 "이 칸은 안 건드린다"를 확인할 수 있다
            ok[j, s] = (np.percentile(p, 95) / max(np.percentile(p, 5), 1e-9)) >= MIN_WIDTH
    return lpref, ok


def load_rot_table(pool, appliances: Sequence[str], max_states: int = 5
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(a (K,S), ln P_ref (K,S), 걸리나 (K,S))`.

    `a` 는 **칸 단위 상수표**에서 온다 (재서 굽는다 — 배우지 않는다).
    칸이 켜지려면 **셋 다** 참이어야 한다: 표에 있다 · 표본이 있다 · 전력 폭이 있다.
    """
    lpref, wide = state_prefs(pool, appliances, max_states)
    a = np.zeros((len(appliances), max_states), np.float32)
    for (app, s), v in LOAD_ROT_A_CELL.items():
        if app in appliances and 0 < s < max_states:
            a[list(appliances).index(app), s] = float(v)
    on = wide & (a != 0.0)
    a = a * on                       # 꺼진 칸은 **정확히 0** 이어야 한다
    return a, lpref, on
