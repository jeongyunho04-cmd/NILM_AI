# -*- coding: utf-8 -*-
"""긴 연속 기록 합성 — 사슬 구조 학습용 (13.84.24).

지금 캐시는 **독립 창** 30만 개다. 사슬은 전이가 여럿 든 연속 기록이 필요하다.
`LoadSynthesizer.synthesize_scenario` 가 이미 일정(schedule)을 받아 타임라인을 만들고
`synthesize_random_window` 도 내부에서 그걸 부르므로, **전압 환경·잡음·회로 모형이 똑같이** 걸린다.
여기서 새로 하는 일은 무작위 일정을 만드는 것뿐이다.

라벨은 공짜다 — `gt_is_on` 이 사이클마다 기기별 참 상태를 준다.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .synthesizer import ApplianceSchedule, LoadSynthesizer, SyntheticLoadSample

from src.preprocessing.file_registry import get_usage_probability

#: 기록마다 그 기기를 넣을 확률. **측정 사용률을 쓰면 안 된다** — 포트 0.006 · 드라이기 0.004 ·
#: 핫플 0.021 이라 기록이 SMPS 투성이가 되고 저항 기기를 사실상 못 배운다(13.84.22 의 사건 표본에서
#: 같은 함정을 밟았다). 캐시의 `recipe_mix` 와 같은 생각으로 **여기서 정한다**.
SEQ_MIX: Dict[str, float] = {
    "laptop_charger": 0.55, "minipc": 0.55, "beam_projector": 0.45,   # SMPS 3종 — 가르는 것이 목표
    "fan": 0.35, "hotplate": 0.35, "oven": 0.35,
    "electiric_kettle": 0.30, "hair_dryer": 0.30, "air_conditioner": 0.30,
}


def random_schedules(apps: Sequence[str], n_cycles: int, rng: np.random.RandomState,
                     min_on_s: float = 20.0, max_on_s: float = 240.0,
                     min_off_s: float = 15.0, max_off_s: float = 180.0,
                     probs: Optional[Dict[str, float]] = None,
                     ) -> List[ApplianceSchedule]:
    """기기마다 켜짐 구간을 여러 개 흩뿌린다. 창 밖에서 시작하는 것도 허용한다.

    ⚠ 한 기기의 구간끼리 겹치면 안 된다 — 겹치면 `gt_is_on` 이 한 덩어리가 되어
      전이 라벨이 사라진다. 그래서 꺼짐 간격을 반드시 둔다.
    """
    FS = 60
    out: List[ApplianceSchedule] = []
    pr = SEQ_MIX if probs is None else probs
    for a in apps:
        if rng.rand() > float(pr.get(a, 0.35)):
            continue
        # 창 앞에서 이미 켜져 있었을 수도 있다 (음수 시작 허용)
        t = int(rng.randint(-int(max_on_s * FS), int(min_off_s * FS) + 1))
        while t < n_cycles:
            dur = int(rng.uniform(min_on_s, max_on_s) * FS)
            out.append(ApplianceSchedule(a, start_cycle=t, duration_cycles=dur))
            t += dur + int(rng.uniform(min_off_s, max_off_s) * FS)
    return out


def make_record(gen: LoadSynthesizer, n_cycles: int, rng: np.random.RandomState,
                apps: Optional[Sequence[str]] = None,
                plugged_prob: float = 0.7,
                probs: Optional[Dict[str, float]] = None) -> SyntheticLoadSample:
    """무작위 일정으로 연속 기록 하나를 합성한다. 기기 확률은  가 정한다."""
    names = list(apps or gen.known_appliances)
    sch = random_schedules(names, n_cycles, rng, probs=probs)
    plugged = {a: bool(rng.rand() < plugged_prob) for a in gen.known_appliances}
    for s in sch:
        plugged[s.appliance_type] = True
    env = gen.grid_sim.sample_environment()
    return gen.synthesize_scenario(
        total_duration_cycles=n_cycles,
        schedules=sch,
        plugged_in_appliances=plugged,
        include_noise=True,
        simulate_voltage_drop=True,
        voltage_environment=env,
        compute_gt_harmonics=False,
    )


def to_raw45(smp: SyntheticLoadSample, volt_orders: Sequence[int]) -> np.ndarray:
    """`build_inputs` 가 먹는 원시 45채널 (45, N) 로 편다 — 실측 npz 와 같은 배치."""
    r = smp.harmonics_ri[:, :, 0].T
    i_ = smp.harmonics_ri[:, :, 1].T
    pf = np.asarray(smp.power_features)
    vh = np.asarray(smp.voltage_harmonics_complex)[:, [h - 1 for h in volt_orders]]
    return np.concatenate([r, i_, pf[:, 0:1].T, pf[:, 1:2].T, pf[:, 4:5].T,
                           vh.real.T, vh.imag.T], axis=0).astype(np.float32)


def _close_gaps(m: np.ndarray, max_gap: int) -> np.ndarray:
    """`max_gap` 사이클 이하의 꺼짐 구멍을 메운다 (형태학적 닫기)."""
    if not m.any() or max_gap <= 0:
        return m
    out = m.copy()
    d = np.diff(np.r_[0, m.view(np.int8), 0])
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    for e, s in zip(ends[:-1], starts[1:]):          # 구간 사이의 구멍
        if s - e <= max_gap:
            out[e:s] = True
    return out


#: 듀티 휴지를 메울 길이. 실측 통전 공백 661개 중 **99.5%가 5초 이하**다 (12절 측정).
DUTY_CLOSE_S = 5.0


def states(smp: SyntheticLoadSample, apps: Sequence[str], close_duty: bool = True) -> np.ndarray:
    """(K, N) bool 참 상태.

    ⚠ `close_duty` 가 참이면 5초 이하 꺼짐 구멍을 메운다. **핫플 휴지도 ON 이다**
    ([[hotplate-duty-pause-is-on]], 13.82) — 안 메우면 릴레이 토글이 전이 라벨이 되어
    사슬 학습에서 핫플 하나가 전이의 97% 를 차지한다(40 기록에서 2102 대 나머지 총 337).
    """
    S = np.stack([np.asarray(smp.gt_is_on[a]).astype(bool)
                  if a in smp.gt_is_on else np.zeros(smp.duration_cycles, bool)
                  for a in apps])
    if close_duty:
        g = int(DUTY_CLOSE_S * 60)
        S = np.stack([_close_gaps(S[k], g) for k in range(S.shape[0])])
    return S
