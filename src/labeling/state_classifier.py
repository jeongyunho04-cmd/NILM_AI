"""
NILM 가전 상태 분류 및 전이 이벤트 탐지 모듈 (State Classifier & Event Detector)
==================================================================================
채터링(Chattering / 빈번한 상태 깜빡임)을 방지하기 위해
Dwell-time(최소 지속 시간) 상태 머신 필터를 적용하여 안정적인 다중 동작 상태를 라벨링하고,
상태가 전이되는 시점의 급변량(Delta P, Delta Q)과 이벤트 유형을 탐지합니다.

[주요 기능]
1. 1초 미디언 스무딩 + Dwell-time 상태 머신:
   - 과도 과전압이나 순간 스파이크(1초 미만)로 인해 기기 동작 모드가 오판정되지 않도록,
     새로운 상태가 일정 시간(min_dwell_samples) 이상 유지될 때만 상태 천이를 확정.
2. 이벤트 어노테이션:
   - "ON" (0 -> >0), "OFF" (>0 -> 0), "MODE_CHANGE" (>0 -> >0)
   - 전이 시점의 Delta P (W), Delta Q (VAR), 이전 상태 유지 시간(초) 자동 기록.
"""
from dataclasses import asdict, dataclass
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


class StateClassifier:
    """가전제품의 소비 전력을 기반으로 다중 상태를 판정하고 이벤트를 탐지하는 클래스."""

    def __init__(self, config: ApplianceStateConfig, sampling_hz: float = 60.0):
        self.config = config
        self.sampling_hz = sampling_hz
        self.min_dwell_samples = max(1, int(config.min_state_duration_s * sampling_hz))
        self.state_map = {s.state_id: s.name for s in config.states}

    def _raw_state_lookup(self, p_val: float) -> int:
        """순수 전력값 구간에 매칭되는 후보 상태 ID를 반환합니다."""
        for state in self.config.states:
            if state.p_min <= p_val < state.p_max:
                return state.state_id
        return self.config.states[-1].state_id

    def classify_series(
        self,
        p_series: np.ndarray,
        q_series: Optional[np.ndarray] = None,
        t_series: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[TransitionEvent]]:
        """시계열 전력 데이터를 상태 ID와 이진 온/오프 라벨로 분류하고 이벤트를 추출합니다.

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

        # 1. 단발성 스파이크 방지용 1초 롤링 미디언 계산
        p_smooth = pd.Series(p_series).rolling(window=int(self.sampling_hz), center=True, min_periods=1).median().values

        # 2. 순간 후보 상태 매핑
        candidate_states = np.array([self._raw_state_lookup(p) for p in p_smooth], dtype=int)

        # 3. 채터링 방지 Dwell-time 상태 머신 (일정 시간 지속 시 상태 전이 확정)
        state_ids = np.zeros(n, dtype=int)
        current_state = candidate_states[0]
        pending_state = current_state
        pending_count = 0
        state_start_idx = 0

        events: List[TransitionEvent] = []

        for i in range(n):
            c_state = candidate_states[i]

            if c_state == current_state:
                pending_state = current_state
                pending_count = 0
            else:
                if c_state == pending_state:
                    pending_count += 1
                    if pending_count >= self.min_dwell_samples:
                        # 최소 유지 시간을 통과하여 상태 전이 확정!
                        prev_state = current_state
                        current_state = pending_state
                        transition_idx = i - pending_count + 1

                        prev_duration = (transition_idx - state_start_idx) / self.sampling_hz
                        
                        # 전이 전/후의 안정 상태 전력(0.5초 미디언) 계산
                        window = int(self.sampling_hz * 0.5)
                        idx_pre_start = max(0, transition_idx - window)
                        idx_post_end = min(n, transition_idx + window)
                        
                        p_before = float(np.median(p_series[idx_pre_start:transition_idx])) if transition_idx > idx_pre_start else p_series[transition_idx]
                        p_after = float(np.median(p_series[transition_idx:idx_post_end])) if idx_post_end > transition_idx else p_series[transition_idx]
                        q_before = float(np.median(q_series[idx_pre_start:transition_idx])) if transition_idx > idx_pre_start else q_series[transition_idx]
                        q_after = float(np.median(q_series[transition_idx:idx_post_end])) if idx_post_end > transition_idx else q_series[transition_idx]

                        # 이벤트 유형 분류
                        if prev_state == 0 and current_state > 0:
                            event_type = "ON"
                        elif prev_state > 0 and current_state == 0:
                            event_type = "OFF"
                        else:
                            event_type = "MODE_CHANGE"

                        event = TransitionEvent(
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
                        )
                        events.append(event)

                        state_ids[transition_idx:i + 1] = current_state
                        state_start_idx = transition_idx
                        pending_count = 0
                else:
                    pending_state = c_state
                    pending_count = 1

            state_ids[i] = current_state

        # 이진 is_on 라벨: 상태 ID가 0보다 크면 1, 0이면 0
        is_on = np.where(state_ids > 0, 1, 0).astype(int)

        return state_ids, is_on, events
