"""
NILM 가전 상태 분류 및 전이 이벤트 탐지 모듈 (State Classifier & Event Detector)
==================================================================================
채터링(Chattering / 빈번한 상태 깜빡임)을 방지하기 위해
경계 히스테리시스(Deadband)와 Dwell-time(최소 지속 시간) 상태 머신을 함께 적용하여
안정적인 다중 동작 상태를 라벨링하고, 상태 전이 시점의 급변량과 이벤트 유형을 탐지합니다.

[채터링을 막는 두 겹의 방어]
1. 경계 히스테리시스 (hysteresis_w):
   전력이 상태 경계 근처에서 미세하게 오르내릴 때 상태가 왔다 갔다 하는 것을 막는다.
   현재 상태를 벗어나려면 경계를 hysteresis_w/2 만큼 확실히 넘어서야 한다.
   예) 선풍기 1단(10~27W) 운전 중 27W 근방에서 떨리면, 27.75W 를 넘어야 2단으로 인정.
2. Dwell-time 상태 머신 (min_state_duration_s):
   경계를 넘었더라도 새 상태가 최소 시간 이상 유지되어야 전이를 확정한다.

이 두 값은 state_definitions.py 에 오래 전부터 정의되어 있었으나 실제로는
어디에서도 읽히지 않아 히스테리시스가 사실상 미구현 상태였다. 여기서 실제로 적용한다.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .state_definitions import ApplianceStateConfig, StateRule, get_appliance_config


@dataclass
class TransitionEvent:
    """가전의 동작 상태가 전이된 단일 이벤트 데이터 구조."""
    sample_idx: int                 # 발생 사이클 인덱스
    t_s: float                      # 발생 상대 시간 (초)
    from_state_id: int              # 이전 상태 ID
    from_state_name: str            # 이전 상태명
    to_state_id: int                # 전이된 상태 ID
    to_state_name: str              # 전이된 상태명
    event_type: str                 # "ON", "OFF", "MODE_CHANGE"
    p_before_w: float               # 전이 전 정상 상태 평균 전력 (W)
    p_after_w: float                # 전이 후 정상 상태 평균 전력 (W)
    delta_p_w: float                # 유효전력 급변량 (Delta P)
    delta_q_var: float              # 무효전력 급변량 (Delta Q)
    prev_state_duration_s: float    # 이전 상태가 유지된 총 시간 (초)
    at_segment_seam: bool = False   # 타임라인 이어붙인 자리에서 발생했는가(가짜 이벤트 의심)


class StateClassifier:
    """가전제품의 소비 전력을 기반으로 다중 상태를 판정하고 이벤트를 탐지하는 클래스."""

    def __init__(self, config: ApplianceStateConfig, sampling_hz: float = 60.0):
        self.config = config
        self.sampling_hz = sampling_hz
        self.min_dwell_samples = max(1, int(config.min_state_duration_s * sampling_hz))
        self.state_map = {s.state_id: s.name for s in config.states}

        # 상태를 전력 하한 기준으로 정렬해 두면 경계 배열로 이진 탐색이 가능하다.
        self._sorted_states: List[StateRule] = sorted(config.states, key=lambda s: s.p_min)
        self._state_by_id: Dict[int, StateRule] = {s.state_id: s for s in config.states}
        # 인접 상태 사이의 경계 전력값 (상태 개수 - 1 개)
        self._boundaries = np.array([s.p_min for s in self._sorted_states[1:]], dtype=float)
        self._ids_in_order = np.array([s.state_id for s in self._sorted_states], dtype=int)

        # 히스테리시스 반폭. 경계를 이만큼 넘어서야 상태 변경을 인정한다.
        self.deadband_w = max(0.0, float(config.hysteresis_w)) / 2.0

        # on_threshold_w 와 OFF 상태 상한이 어긋나 있으면 라벨 의미가 모호해진다.
        off_state = self._sorted_states[0]
        self.on_threshold_w = float(config.on_threshold_w)
        self._threshold_consistent = abs(off_state.p_max - self.on_threshold_w) < 1e-6

    def _raw_state_lookup(self, p_val: float) -> int:
        """순수 전력값 구간에 매칭되는 후보 상태 ID를 반환합니다 (히스테리시스 없음)."""
        pos = int(np.searchsorted(self._boundaries, p_val, side="right"))
        return int(self._ids_in_order[min(pos, len(self._ids_in_order) - 1)])

    def _lookup_all(self, p_values: np.ndarray) -> np.ndarray:
        """전력 배열 전체에 대한 후보 상태를 한 번에 계산합니다."""
        pos = np.searchsorted(self._boundaries, p_values, side="right")
        pos = np.clip(pos, 0, len(self._ids_in_order) - 1)
        return self._ids_in_order[pos]

    def _holds_current_state(self, p_val: float, current_state: int) -> bool:
        """히스테리시스 데드밴드 안이라 현재 상태를 계속 유지해야 하는지 판정합니다.

        현재 상태의 전력 구간을 양쪽으로 deadband_w 만큼 넓혀 두고,
        그 확장 구간 안에 있으면 경계를 넘었더라도 아직 상태가 바뀌지 않은 것으로 본다.
        """
        if self.deadband_w <= 0.0:
            return False
        cur = self._state_by_id.get(current_state)
        if cur is None:
            return False
        return (cur.p_min - self.deadband_w) <= p_val < (cur.p_max + self.deadband_w)

    def classify_series(
        self,
        p_series: np.ndarray,
        q_series: Optional[np.ndarray] = None,
        t_series: Optional[np.ndarray] = None,
        seam_flags: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[TransitionEvent]]:
        """시계열 전력 데이터를 상태 ID와 이진 온/오프 라벨로 분류하고 이벤트를 추출합니다.

        Args:
            seam_flags: (N,) 타임라인 이어붙인 자리 표시. 그 자리의 전이는 가짜로 표시된다.

        Returns:
            state_ids: (N,) int 배열 (다중 상태 클래스 ID)
            is_on: (N,) int 배열 (0: OFF, 1: ON)
            events: 탐지된 TransitionEvent 리스트
        """
        n = len(p_series)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int), []

        if q_series is None:
            q_series = np.zeros(n, dtype=float)
        if t_series is None:
            t_series = np.arange(n) / self.sampling_hz
        seam_set = set(np.where(seam_flags > 0)[0].tolist()) if seam_flags is not None else set()

        # 1. 단발성 스파이크 방지용 1초 롤링 미디언
        p_smooth = pd.Series(p_series).rolling(
            window=int(self.sampling_hz), center=True, min_periods=1
        ).median().values

        # 2. 히스테리시스 없는 순간 후보 상태 (벡터 연산으로 한 번에)
        base_candidates = self._lookup_all(p_smooth)

        # 3. 히스테리시스 데드밴드 + Dwell-time 상태 머신
        state_ids = np.zeros(n, dtype=int)
        current_state = int(base_candidates[0])
        pending_state = current_state
        pending_count = 0
        state_start_idx = 0

        events: List[TransitionEvent] = []
        window = int(self.sampling_hz * 0.5)

        for i in range(n):
            c_state = int(base_candidates[i])

            # 데드밴드 안이면 경계를 넘었어도 현재 상태를 유지한다.
            if c_state != current_state and self._holds_current_state(p_smooth[i], current_state):
                c_state = current_state

            if c_state == current_state:
                pending_state = current_state
                pending_count = 0
            elif c_state == pending_state:
                pending_count += 1
                if pending_count >= self.min_dwell_samples:
                    # 최소 유지 시간을 통과하여 상태 전이 확정
                    prev_state = current_state
                    current_state = pending_state
                    transition_idx = i - pending_count + 1

                    prev_duration = (transition_idx - state_start_idx) / self.sampling_hz

                    idx_pre_start = max(0, transition_idx - window)
                    idx_post_end = min(n, transition_idx + window)

                    p_before = (
                        float(np.median(p_series[idx_pre_start:transition_idx]))
                        if transition_idx > idx_pre_start else float(p_series[transition_idx])
                    )
                    p_after = (
                        float(np.median(p_series[transition_idx:idx_post_end]))
                        if idx_post_end > transition_idx else float(p_series[transition_idx])
                    )
                    q_before = (
                        float(np.median(q_series[idx_pre_start:transition_idx]))
                        if transition_idx > idx_pre_start else float(q_series[transition_idx])
                    )
                    q_after = (
                        float(np.median(q_series[transition_idx:idx_post_end]))
                        if idx_post_end > transition_idx else float(q_series[transition_idx])
                    )

                    if prev_state == 0 and current_state > 0:
                        event_type = "ON"
                    elif prev_state > 0 and current_state == 0:
                        event_type = "OFF"
                    else:
                        event_type = "MODE_CHANGE"

                    # 이어붙인 경계 근처(±0.5초)에서 난 전이는 실제 사건이 아닐 수 있다.
                    near_seam = any(
                        abs(transition_idx - s) <= window for s in seam_set
                    ) if seam_set else False

                    events.append(TransitionEvent(
                        sample_idx=int(transition_idx),
                        t_s=round(float(t_series[transition_idx]), 3),
                        from_state_id=int(prev_state),
                        from_state_name=self.state_map.get(prev_state, f"STATE_{prev_state}"),
                        to_state_id=int(current_state),
                        to_state_name=self.state_map.get(current_state, f"STATE_{current_state}"),
                        event_type=event_type,
                        p_before_w=round(p_before, 2),
                        p_after_w=round(p_after, 2),
                        delta_p_w=round(p_after - p_before, 2),
                        delta_q_var=round(q_after - q_before, 2),
                        prev_state_duration_s=round(prev_duration, 2),
                        at_segment_seam=bool(near_seam),
                    ))

                    state_ids[transition_idx:i + 1] = current_state
                    state_start_idx = transition_idx
                    pending_count = 0
            else:
                pending_state = c_state
                pending_count = 1

            state_ids[i] = current_state

        # 이진 is_on 라벨: 상태 ID가 0보다 크면 1, 0이면 0.
        # **단, `on_state_min_id` 가 있으면 그 ID 이상만 ON 이다** (12.111).
        # 오븐이 그 경우다 — 팬/조명(16.5W)이 `on_threshold_w=10` 을 넘어서
        # 히터가 꺼진 25분 내내 ON 으로 잡혔다. 그 라벨로 학습하면 "오븐 ON 인데
        # 전력 0W" 인 창이 절반이 되고, 모델이 저항 부하를 전부 오븐으로 읽는다
        # (12.110.2). 핫플레이트는 휴지 전력이 문턱 아래라 원래부터 통전만 ON 이다.
        min_id = getattr(self.config, "on_state_min_id", None)
        thr = 1 if min_id is None else int(min_id)
        is_on = np.where(state_ids >= thr, 1, 0).astype(int)

        return state_ids, is_on, events
