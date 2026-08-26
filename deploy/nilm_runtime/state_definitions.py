"""
State Definitions and Configuration for NILM AI Appliances
Defines operational states, multi-state class IDs, nominal power ranges,
hysteresis thresholds, and minimum duration constraints for each appliance.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StateRule:
    state_id: int
    name: str
    p_min: float
    p_max: float
    description: str
    nominal_w: Optional[float] = None


@dataclass
class ApplianceStateConfig:
    appliance_type: str
    korean_name: str
    on_threshold_w: float
    states: List[StateRule]
    min_state_duration_s: float = 0.5  # Minimum time in seconds to confirm state transition (anti-chatter)
    hysteresis_w: float = 2.0  # Power deadband around threshold boundaries


# ── Appliance Specific State Configurations ─────────────────────────────────

STATE_CONFIGURATIONS: Dict[str, ApplianceStateConfig] = {
    "air_conditioner": ApplianceStateConfig(
        appliance_type="air_conditioner",
        korean_name="에어컨",
        on_threshold_w=10.0,
        min_state_duration_s=1.0,
        hysteresis_w=5.0,
        states=[
            StateRule(0, "OFF_STANDBY", 0.0, 10.0, "대기 상태 또는 전원 꺼짐", nominal_w=2.0),
            StateRule(1, "FAN_ONLY", 10.0, 80.0, "송풍/팬 단독 동작 모드", nominal_w=25.0),
            StateRule(2, "COOLING_LOW", 80.0, 350.0, "인버터 컴프레서 저부하 냉방", nominal_w=200.0),
            StateRule(3, "COOLING_MID", 350.0, 600.0, "인버터 컴프레서 중부하 냉방", nominal_w=450.0),
            StateRule(4, "COOLING_HIGH", 600.0, 3000.0, "인버터 컴프레서 고부하/터보 냉방", nominal_w=750.0),
        ],
    ),
    "beam_projector": ApplianceStateConfig(
        appliance_type="beam_projector",
        korean_name="빔프로젝터",
        on_threshold_w=4.0,
        min_state_duration_s=0.5,
        hysteresis_w=1.5,
        states=[
            StateRule(0, "OFF_STANDBY", 0.0, 4.0, "대기 전원 상태", nominal_w=1.0),
            StateRule(1, "WARMUP_COOLDOWN", 4.0, 30.0, "부팅 예열 또는 전원 끈 후 팬 냉각", nominal_w=8.0),
            StateRule(2, "LAMP_ON", 30.0, 500.0, "램프 점등 및 정상 화면 투사", nominal_w=48.0),
        ],
    ),
    "electiric_kettle": ApplianceStateConfig(
        appliance_type="electiric_kettle",
        korean_name="전기포트",
        on_threshold_w=50.0,
        min_state_duration_s=0.3,
        hysteresis_w=10.0,
        states=[
            StateRule(0, "OFF", 0.0, 50.0, "전원 꺼짐 / 대기 상태", nominal_w=1.5),
            StateRule(1, "BOILING_HEATING", 50.0, 3000.0, "히터 발열 및 물 끓임 동작", nominal_w=1260.0),
        ],
    ),
    "fan": ApplianceStateConfig(
        appliance_type="fan",
        korean_name="선풍기",
        on_threshold_w=10.0,
        min_state_duration_s=0.5,
        hysteresis_w=1.5,
        states=[
            StateRule(0, "OFF", 0.0, 10.0, "전원 꺼짐", nominal_w=1.5),
            StateRule(1, "SPEED_LOW", 10.0, 27.0, "1단 미풍 운전", nominal_w=23.5),
            StateRule(2, "SPEED_MID", 27.0, 36.0, "2단 약풍 운전", nominal_w=31.0),
            StateRule(3, "SPEED_HIGH", 36.0, 150.0, "3단 강풍 운전", nominal_w=40.5),
        ],
    ),
    "hair_dryer": ApplianceStateConfig(
        appliance_type="hair_dryer",
        korean_name="헤어드라이기",
        on_threshold_w=30.0,
        min_state_duration_s=0.3,
        hysteresis_w=15.0,
        states=[
            StateRule(0, "OFF", 0.0, 30.0, "전원 꺼짐", nominal_w=1.6),
            StateRule(1, "LOW_HEAT_COOL", 30.0, 750.0, "냉풍 또는 1단 온풍/저열 모드", nominal_w=510.0),
            StateRule(2, "HIGH_HEAT_TURBO", 750.0, 3000.0, "2단 열풍/터보 고열 모드", nominal_w=1000.0),
        ],
    ),
    "hotplate": ApplianceStateConfig(
        appliance_type="hotplate",
        korean_name="핫플레이트",
        on_threshold_w=50.0,
        min_state_duration_s=0.3,
        hysteresis_w=10.0,
        states=[
            StateRule(0, "OFF_STANDBY", 0.0, 50.0, "대기 상태 또는 서모스탯 OFF 주기", nominal_w=2.0),
            StateRule(1, "HEATING_ACTIVE", 50.0, 3000.0, "히터 통전 발열 주기", nominal_w=465.0),
        ],
    ),
    "laptop_charger": ApplianceStateConfig(
        appliance_type="laptop_charger",
        korean_name="노트북충전기",
        on_threshold_w=10.0,
        min_state_duration_s=0.5,
        hysteresis_w=2.0,
        states=[
            StateRule(0, "IDLE_UNPLUGGED", 0.0, 10.0, "미연결 무부하 또는 완충 대기", nominal_w=1.5),
            StateRule(1, "TRICKLE_IDLE_CHARGE", 10.0, 35.0, "저속 트리클 충전 / 유휴 상태 충전", nominal_w=30.0),
            StateRule(2, "FAST_ACTIVE_CHARGE", 35.0, 500.0, "고속 충전 / CPU 부하 동작 중 충전", nominal_w=60.0),
        ],
    ),
    "minipc": ApplianceStateConfig(
        appliance_type="minipc",
        korean_name="미니PC",
        on_threshold_w=5.0,
        min_state_duration_s=0.5,
        hysteresis_w=1.0,
        states=[
            StateRule(0, "OFF_STANDBY", 0.0, 5.0, "전원 꺼짐 또는 절전 모드", nominal_w=1.5),
            StateRule(1, "IDLE", 5.0, 13.0, "바탕화면 대기 및 경부하 상태", nominal_w=9.8),
            StateRule(2, "ACTIVE_LOAD", 13.0, 200.0, "프로그램 실행 및 CPU 연산 부하", nominal_w=18.5),
        ],
    ),
    # 측정 대상 오븐은 목표 온도와 시간만 설정하는 모델이라, 히터 조합을 나누는
    # 조작부가 없다. 실측 35.8분에서도 전력은 2.0W(대기) / 16.5W(팬·조명) /
    # 1156.5W(히터) 세 곳에만 모여 있었고, 그 사이 100~800W 구간은 전체를 합쳐
    # 1.05초뿐인 On/Off 전환 램프였다.
    # 이전에는 여기에 MEDIUM_HEAT(단일 히터 400W)가 정의되어 있었으나 관측 0분이었다.
    # 학습 예시가 하나도 없는 클래스를 예측 대상으로 두면 출력 차원만 늘리고
    # 모델이 채울 수 없는 자리를 만들 뿐이라 제거한다.
    "oven": ApplianceStateConfig(
        appliance_type="oven",
        korean_name="오븐",
        on_threshold_w=10.0,
        min_state_duration_s=0.5,
        hysteresis_w=5.0,
        states=[
            StateRule(0, "OFF_STANDBY", 0.0, 10.0, "대기 전원 상태", nominal_w=2.0),
            StateRule(1, "FAN_LIGHT", 10.0, 100.0, "히터 꺼짐 주기 - 조명등 및 컨벡션 팬만 동작", nominal_w=16.5),
            StateRule(2, "HEATING", 100.0, 3000.0, "서모스탯 통전 - 히터 가열", nominal_w=1157.0),
        ],
    ),
    "noise_noselfpower": ApplianceStateConfig(
        appliance_type="noise_noselfpower",
        korean_name="무부하노이즈(외부전원)",
        on_threshold_w=9999.0,
        states=[StateRule(0, "BASELINE_NOISE", 0.0, 9999.0, "무부하 센서 노이즈 바닥", nominal_w=1.4)],
    ),
    "noise_selfpower": ApplianceStateConfig(
        appliance_type="noise_selfpower",
        korean_name="무부하노이즈(자체전원)",
        on_threshold_w=9999.0,
        states=[StateRule(0, "BASELINE_NOISE", 0.0, 9999.0, "보드 자체 소비전력 노이즈 바닥", nominal_w=2.37)],
    ),
}


def get_appliance_config(appliance_type: str) -> ApplianceStateConfig:
    """Returns the state configuration for a given appliance type."""
    if appliance_type in STATE_CONFIGURATIONS:
        return STATE_CONFIGURATIONS[appliance_type]
    # Fallback generic 2-state appliance (OFF / ON)
    return ApplianceStateConfig(
        appliance_type=appliance_type,
        korean_name=appliance_type,
        on_threshold_w=10.0,
        states=[
            StateRule(0, "OFF", 0.0, 10.0, "전원 꺼짐", nominal_w=0.0),
            StateRule(1, "ON", 10.0, 10000.0, "동작 중", nominal_w=100.0),
        ],
    )
