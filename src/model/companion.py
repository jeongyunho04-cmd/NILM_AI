"""동반 부하 — 기기가 **주 상태가 아닐 때도 계속 흐르는** 몫 (2026-09-03, 12.156).

[무엇이 빠져 있었나]
오븐은 `OFF_STANDBY / FAN_LIGHT / HEATING` 세 상태다. 그런데 라벨이 FAN_LIGHT 을
`is_on=0, target_power_w=0` 으로 적는다 — 조명과 컨벡션 팬의 14.2W 가 **어느 기기
몫도 아니게** 배경으로 흘러간다. 그래서 `net.standby_signatures` 가 집는
`get_standby_profile` 은 OFF_STANDBY 의 0.40W / 6.44mA 이고, **오븐이 존재하는
시간의 52~73% 를 차지하는 상태가 순방향 모형 어디에도 없다.**

`harmonic_signatures` 도 그것을 의도적으로 뺀다 (그 독스트링: *"팬/조명 같은 저전력
부수 상태가 지문을 오염시키지 않게 한다"*). 와트당 지문을 깨끗하게 두려는 판단이고
그 자체로는 맞다 — 다만 **그러면 그 상태를 담을 자리가 따로 있어야 한다.**

[왜 이것이 배분을 고치는가]
포트와 오븐은 `L_harm` 안에서 **전력 크기를 빼면 축퇴**다. 1377W 포트와 1141W 오븐의
판별 기여 18.27 중 h1 이 17.38(97.6%)이고, 모양(h2~h8)은 다 합쳐 0.886(4.9%)뿐이다.
와트당 지문이 h1 1.3%, h3 11%, h5 4.2%, h7 3.9% 밖에 안 갈리기 때문이다. 그래서
압력이 어디서 오든 이 축으로 미끄러진다 — 12.155.6 의 반파 채널이 포트 1,209W 를
**장소 B 에 없는 오븐**에게 넘긴 것이 그것이고, 12.155.4 의 장소 B 재적응이 포트를
0.935 -> 0.629 로 떨어뜨린 것도 같은 자리다.

동반 부하는 **크기가 아니라 신원**을 요구한다: 오븐이 켜졌다면 어딘가에 64mA /
|u3| 0.058 의 SMPS 전류가 같이 흐르고 있어야 한다. 포트에는 그런 상태가 없다
(격리에서 3~60W 구간 표본이 **0개**다 — OFF 아니면 통전이다).

[⚠ `(1−σ(on))` 을 곱하지 않는다 — `standby` 항과 다른 점]
팬·조명은 히터와 **동시에** 돈다. 격리 실측:

    HEATING 의 |I3| 중 팬/조명 몫    58% / 40% / 27%   (녹화 3개)
    HEATING − FAN_LIGHT 의 |u3|      0.0010 / 0.0022 / 0.0028   <- 순수 니크롬
    등가저항  전체 40.6Ω  ->  히터만 41.2Ω               <- 병렬 14W 만큼의 차

빼고 남은 잔차가 순수 저항다우므로 가산 분해가 성립한다. 그래서 `σ(plugged)` 로만
건다 — 히터가 꺼진 주기에도, 켜진 주기에도 같은 값이 흐른다.

[⚠ 이중 계상은 안 난다]
`harmonic_signatures` 는 `target_power_w > 0.5·p90` 인 표본만 쓰는데 FAN_LIGHT 은
`target_power_w = 0` 이라 애초에 안 들어간다. HEATING 표본의 `net_harmonics_complex`
에는 팬/조명 전류가 섞여 있지만, 그것은 **와트당 상수**로 희석되어 들어간 것이고
여기서 더하는 것은 **전력에 무관한 가산 상수**다. 겨냥이 다르다 — 전자는 크기,
후자는 신원이다. 실제 겹침은 오븐 h1 의 1.2% (64mA / 5344mA) 라 무시할 수 있다.

[규칙 14 — 상수의 재현성]
녹화 3개(`oven`, `oven_2`, `oven_3_fixed`)에서 OFF_STANDBY 을 배경으로 뺀 값:

    P        14.48 / 14.20 / 14.21 W
    |I1|     64.49 / 63.86 / 64.26 mA        **폭/중앙 0.010**
    |u3|     0.0581 / 0.0571 / 0.0564

`REFERENCE_W` 의 채택 문턱(폭/중앙 0.10)을 열 배 여유로 통과한다. 인수인계
12.155 의 남은 것 [1] 이 적은 *"오븐은 히터가 꺼진 동안 14W SMPS 부하가 남는다
(|u₃| 0.06)"* 와 같은 값이다.

[표에 **안 넣은** 기기]
동반 부하는 "주 상태와 무관하게 흐르는 몫" 이라야 한다. 다음은 아니다:

    포트         3~60W 표본 0개. OFF 아니면 통전 — 중간 상태 자체가 없다
    핫플         OFF_STANDBY 0.41W 뿐. 릴레이 휴지는 히터의 듀티지 별도 부하가 아니다
    드라이기     강/약은 같은 히터의 도통각 차이다 (12.109.2 반파). 동반 부하가 아니다
"""
from typing import Dict, List, Sequence, Tuple
import json

import numpy as np

#: 동반 부하를 가진 기기 -> 그 상태 이름. 라벨의 `state_distribution` 키다.
COMPANION_STATE: Dict[str, str] = {"oven": "FAN_LIGHT"}

#: 배경으로 뺄 상태 (그 기기의 완전 정지 상태).
BASELINE_STATE: Dict[str, str] = {"oven": "OFF_STANDBY"}

#: 상수를 채택할 재현성 문턱. `power_ref.REFERENCE_W` 의 0.10 과 같은 성격이다.
SPREAD_MAX = 0.10


def _recordings(app: str, npz_dir: str) -> List[str]:
    from pathlib import Path
    out = []
    for f in sorted(Path(npz_dir).glob("*.npz")):
        try:
            md = json.loads(str(np.load(f, allow_pickle=True)["metadata_json"]))
        except Exception:
            continue
        if md.get("appliance_type") == app:
            out.append(str(f))
    return out


def companion_constants(appliances: Sequence[str], npz_dir: str = "processed_data/npz",
                        n_harm: int = 15) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(K,H,2) 페이저 / (K,) 전력(W) / 실제로 값이 들어간 기기 이름.

    격리 녹화마다 `동반상태 − 정지상태` 의 중앙 페이저를 구하고, 녹화들의 중앙을
    쓴다. 재현성이 `SPREAD_MAX` 를 넘으면 **그 기기를 뺀다** (규칙 14).
    """
    K = len(appliances)
    sig = np.zeros((K, n_harm, 2), np.float32)
    pw = np.zeros(K, np.float32)
    used: List[str] = []
    for j, app in enumerate(appliances):
        st = COMPANION_STATE.get(app)
        if st is None:
            continue
        nets, pws = [], []
        for f in _recordings(app, npz_dir):
            z = np.load(f, allow_pickle=True)
            md = json.loads(str(z["metadata_json"]))
            ids = {v["state_id"]: k for k, v in md["state_distribution"].items()}
            try:
                sc = next(s for s, n in ids.items() if n == st)
                sb = next(s for s, n in ids.items() if n == BASELINE_STATE[app])
            except StopIteration:
                continue
            sid, ok = z["state_id"], z["is_valid"].astype(bool)
            c, p = z["harmonics_complex"], z["power_features"][:, 0]
            mc, mb = ok & (sid == sc), ok & (sid == sb)
            if mc.sum() < 200 or mb.sum() < 50:
                continue
            cc = np.median(c[mc].real, 0) + 1j * np.median(c[mc].imag, 0)
            cb = np.median(c[mb].real, 0) + 1j * np.median(c[mb].imag, 0)
            nets.append(cc - cb)
            pws.append(float(np.median(p[mc]) - np.median(p[mb])))
        if len(nets) < 2:
            continue
        arr = np.array(nets)
        m = np.median(arr.real, 0) + 1j * np.median(arr.imag, 0)
        a1 = np.abs(arr[:, 0])
        spread = float((a1.max() - a1.min()) / max(abs(m[0]), 1e-9))
        if spread > SPREAD_MAX:            # 규칙 14 — 재현 안 되면 안 쓴다
            continue
        h = min(n_harm, len(m))
        sig[j, :h, 0], sig[j, :h, 1] = m[:h].real, m[:h].imag
        pw[j] = float(np.median(pws))
        used.append(app)
    return sig, pw, used


def standby_operating_signatures(pool, appliances, n_harm: int = 15):
    """동작 중 주 상태가 아닐 때의 **net 페이저**와 전력 (12.163).

    `net.standby_signatures` 는 `get_standby_profile` 을 쓰는데, 그것은 오븐의
    경우 `OFF_STANDBY`(0.40W / 6.44mA) 다. 그런데 **합성이 실제로 넣는 값은
    다르다** — `synthesizer` 가 activation 휴지 구간의 `gt_standby_p` 를
    `net_power_features[:, 0]` 에서 가져오므로 **FAN_LIGHT 의 15.02W** 다.

    그래서 전력과 고조파가 서로 다른 상태를 가리킨다:

        y_standby     15.0 W        <- FAN_LIGHT  (맞다)
        standby_sig   6.44 mA       <- OFF_STANDBY (틀리다)
        실제           15.0 W / 67.4 mA

    **10배 어긋난다.** `L_harm` 의 `idle · standby_sig` 가 전력이 요구하는 것의
    1/10 만 설명하므로 모델이 타협한다 — 실측에서 오븐 standby 예측이 5.37W 다.

    여기서는 **합성이 넣는 것과 같은 자**를 쓴다: activation 안에서 `is_on=0` 인
    사이클의 `net_harmonics_complex` 중앙. 그러면 전력과 고조파가 같은 상태를
    가리킨다.

    Returns:
        (K,H,2) 페이저, (K,) 전력W, 값이 들어간 기기 이름. 동작 중 휴지가
        거의 없는 기기(휴지 비율 < `MIN_IDLE_FRAC`)는 건드리지 않는다 — 그런
        기기는 `OFF_STANDBY` 이 맞다.
    """
    K = len(appliances)
    sig = np.zeros((K, n_harm, 2), np.float32)
    pw = np.zeros(K, np.float32)
    used = []
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        cs, ps, n_tot, n_idle = [], [], 0, 0
        for a in acts:
            m = ~np.asarray(a.is_on).astype(bool)
            n_tot += len(m); n_idle += int(m.sum())
            if m.any():
                cs.append(a.net_harmonics_complex[m])
                ps.append(np.asarray(a.net_power_features)[m, 0])
        if not cs or n_tot == 0:
            continue
        if n_idle / n_tot < MIN_IDLE_FRAC:      # 휴지가 드물면 안 건드린다
            continue
        c = np.concatenate(cs); p = np.concatenate(ps)
        med = np.median(c.real, 0) + 1j * np.median(c.imag, 0)
        h = min(n_harm, len(med))
        sig[j, :h, 0], sig[j, :h, 1] = med[:h].real, med[:h].imag
        pw[j] = float(np.median(p))
        used.append(app)
    return sig, pw, used


#: 동작 중 휴지가 이 비율을 넘는 기기만 위 함수가 다룬다. 오븐 68%, 핫플 57%.
MIN_IDLE_FRAC = 0.20
