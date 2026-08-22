"""
NILM 원본 파일 역할 레지스트리 (File Role Registry)
====================================================
data/ 안의 모든 CSV가 "어떤 성격의 측정인지"를 결정하는 단일 출처(Single Source of Truth).

[역할이 필요한 이유]
과거에는 PreprocessingPipeline.DEVICE_MAP 에 등록되지 않은 파일이 들어오면
파일 이름(stem)을 그대로 가전 종류로 삼고 일반 OFF/ON(10W) 설정으로 라벨링했다.
그 결과 test.csv(에어컨+드라이기+충전기+선풍기 동시 운전)나 nilm_YYYYMMDD_HHMMSS.csv 같은
'복합 부하 실측 파일'이 "test 라는 이름의 단일 가전"으로 둔갑해 세그먼트 풀에 섞여 들어갔고,
합성 데이터의 정답(Ground Truth)이 근본적으로 오염되었다.

이 모듈은 파일을 4가지 역할로 명확히 나누고, 분류할 수 없는 파일은
조용히 넘어가지 않고 즉시 오류를 내어 사람이 등록하도록 강제한다.

[역할 정의]
1. DEVICE         : 단일 가전 단독 측정. 학습/합성용 세그먼트 풀에 들어간다.
2. NOISE          : 무부하 기준 노이즈. 계측 보드 자체 소비 전력의 기준값.
3. COMPOSITE_EVAL : 여러 가전이 동시에 돌아간 실측. 검증 전용이며 세그먼트 풀 진입 금지.
4. UNKNOWN        : 미등록. 기본적으로 오류를 발생시킨다.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import re


class FileRole(str, Enum):
    """원본 CSV 파일의 측정 성격."""
    DEVICE = "device"                   # 단일 가전 단독 측정 -> 합성 세그먼트 풀 사용 가능
    NOISE = "noise"                     # 무부하 기준 노이즈 -> 배경 노이즈 풀로만 사용
    COMPOSITE_EVAL = "composite_eval"   # 복합 부하 실측 -> 검증 전용, 세그먼트 풀 진입 금지
    UNKNOWN = "unknown"                 # 미등록 -> 처리 거부


class LoadClass(str, Enum):
    """전압 변동에 대한 물리적 응답 특성에 따른 부하 분류.

    GridSimulator 가 전압 변화 시 전류/전력을 어떻게 변형할지 결정하는 기준이며,
    이전에는 grid_simulator.py 와 synthesizer.py 두 곳에 문자열 리스트로 중복 하드코딩되어
    한쪽만 수정하면 조용히 어긋나는 구조였다. 여기 한 곳으로 모은다.
    """
    RESISTIVE = "resistive"   # 순수 저항 히터: I∝V, P∝V^2
    SMPS = "smps"             # 정전력 스위칭 전원: P=const, I∝1/V
    MOTOR = "motor"           # 유도 전동기/인버터: I∝V^0.7, P∝V^0.7
    PASSIVE = "passive"       # 노이즈 등 부하 아님


# ── 계측 보드 전원 공급 방식별 무부하 바닥 전력 ──────────────────────────────
# 보드가 측정 대상 회로에서 자체 전원을 뽑으면(self-power) 보드 소비가 측정값에 포함된다.
NOISE_FLOOR_EXTERNAL_W = 1.4   # 외부 전원 공급 시 계측계 바닥 노이즈
NOISE_FLOOR_SELFPOWER_W = 2.37  # 자체 전원 공급 시 보드 소비 포함 바닥값


@dataclass(frozen=True)
class DeviceSpec:
    """단일 가전 측정 파일 1개에 대한 명세."""
    appliance_type: str          # 상태 설정(STATE_CONFIGURATIONS) 조회 키
    load_class: LoadClass        # 전압 응답 물리 분류
    noise_floor_w: float = NOISE_FLOOR_EXTERNAL_W
    periodic_duty: bool = False  # 서모스탯/릴레이로 주기적 ON-OFF 를 반복하는가
    low_load: bool = False       # 대기전력과 혼동될 수 있는 저전력 대역(<60W)에서 동작하는가
    # 하루 평균 사용 시간. 합성에서 각 기기가 켜져 있을 확률(= 시간/24)로 쓴다.
    # 이전에는 9종을 균등 추첨해 미니PC와 헤어드라이기가 똑같이 15%씩 나왔는데,
    # 실제로는 미니PC가 42%, 드라이기가 0.4% 로 100배 차이가 난다.
    # 집집마다 다른 값이므로 필요하면 여기서 바로 조정하면 된다.
    daily_usage_hours: float = 2.0


# ── 단일 가전 측정 파일 등록부 ───────────────────────────────────────────────
# 새 가전을 측정하면 반드시 여기에 한 줄 추가해야 파이프라인이 받아들인다.
DEVICE_FILES: Dict[str, DeviceSpec] = {
    "air_conditioner":  DeviceSpec("air_conditioner",  LoadClass.MOTOR,     daily_usage_hours=4.0),
    "beam_projector":   DeviceSpec("beam_projector",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=2.0),
    "beam_projector_2": DeviceSpec("beam_projector",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=2.0),
    "electiric_kettle": DeviceSpec("electiric_kettle", LoadClass.RESISTIVE, daily_usage_hours=0.15),
    "fan_1":            DeviceSpec("fan",              LoadClass.MOTOR,     low_load=True,  daily_usage_hours=5.0),
    "fan_2":            DeviceSpec("fan",              LoadClass.MOTOR,     low_load=True,  daily_usage_hours=5.0),
    "fan_3":            DeviceSpec("fan",              LoadClass.MOTOR,     low_load=True,  daily_usage_hours=5.0),
    "hair_dryer_1":     DeviceSpec("hair_dryer",       LoadClass.RESISTIVE, daily_usage_hours=0.1),
    "hair_dryer_2":     DeviceSpec("hair_dryer",       LoadClass.RESISTIVE, daily_usage_hours=0.1),
    # 핫플레이트는 릴레이가 약 1초 ON / 1초 OFF 로 통전을 끊는 주기 부하다.
    "hotplate_1":       DeviceSpec("hotplate",         LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    "hotplate_2":       DeviceSpec("hotplate",         LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    "laptop_charger_1": DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    "laptop_charger_2": DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    "minipc_1":         DeviceSpec("minipc",           LoadClass.SMPS,      low_load=True,  daily_usage_hours=10.0),
    "minipc_2":         DeviceSpec("minipc",           LoadClass.SMPS,      low_load=True,  daily_usage_hours=10.0),
    # minipc_3: CPU 부하 구간 측정 (2026-08-21). 7.2절 1순위였던 28W 대역 보강용.
    "minipc_3":         DeviceSpec("minipc",           LoadClass.SMPS,      low_load=True,  daily_usage_hours=10.0),
    # 오븐은 히터가 꺼져도 팬/조명 약 17W 가 남아 서모스탯 주기가 전력에 그대로 드러난다.
    "oven":             DeviceSpec("oven",             LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    # oven_2: 다른 목표온도 측정 (2026-08-21). 7.2절이 지적한 "듀티비가 온도의 함수인데
    # 1점만 있다" 와 12.3절의 "오븐 독립 활성화 2개" 를 함께 겨냥한다.
    "oven_2":           DeviceSpec("oven",             LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
}


# ── 무부하 기준 노이즈 파일 등록부 ───────────────────────────────────────────
NOISE_FILES: Dict[str, float] = {
    "noise_noselfpower": NOISE_FLOOR_EXTERNAL_W,
    "noise_selfpower": NOISE_FLOOR_SELFPOWER_W,
}


# ── 복합 부하 실측(검증 전용) 파일 판별 규칙 ─────────────────────────────────
# 이 패턴에 걸린 파일은 절대 단일 가전으로 취급하지 않는다.
#   - test, test2, test.2, test3, test_evening ... : 손으로 만든 복합 부하 검증 녹화
#   - nilm_20260818_215047                         : 수신기 기본 파일명(측정 내용 미상)
COMPOSITE_EVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"^test", re.IGNORECASE),
    re.compile(r"^nilm_\d{8}_\d{6}$", re.IGNORECASE),
    re.compile(r"(composite|mixed|multi|scenario)", re.IGNORECASE),
]


@dataclass(frozen=True)
class FileClassification:
    """파일 1개에 대한 분류 결과."""
    stem: str
    role: FileRole
    appliance_type: Optional[str]   # DEVICE 일 때만 값이 있다
    load_class: LoadClass
    noise_floor_w: float
    periodic_duty: bool
    low_load: bool
    reason: str                     # 사람이 읽을 수 있는 분류 근거

    @property
    def usable_for_synthesis(self) -> bool:
        """이 파일이 합성용 가전 세그먼트 풀에 들어가도 되는가."""
        return self.role == FileRole.DEVICE


def _normalize_stem(file_path: Union[str, Path]) -> str:
    """'data/test.2.csv' -> 'test.2' 처럼 확장자만 떼어낸 이름을 얻는다."""
    name = Path(file_path).name
    return name[:-4] if name.lower().endswith(".csv") else Path(file_path).stem


def classify_file(file_path: Union[str, Path]) -> FileClassification:
    """원본 CSV 1개의 역할을 판정한다. 판정 순서 자체가 안전장치다.

    복합 부하 패턴을 가전 등록부보다 **먼저** 확인한다. 그래야 누군가
    test_fan.csv 같은 이름을 만들어도 선풍기로 오인되지 않는다.
    """
    stem = _normalize_stem(file_path)

    # 1) 복합 부하 실측 패턴 우선 차단 (가전 등록부보다 먼저 본다)
    for pattern in COMPOSITE_EVAL_PATTERNS:
        if pattern.search(stem):
            return FileClassification(
                stem=stem,
                role=FileRole.COMPOSITE_EVAL,
                appliance_type=None,
                load_class=LoadClass.PASSIVE,
                noise_floor_w=NOISE_FLOOR_EXTERNAL_W,
                periodic_duty=False,
                low_load=False,
                reason=f"복합 부하 실측 패턴 '{pattern.pattern}' 일치 - 검증 전용, 세그먼트 풀 진입 금지",
            )

    # 2) 무부하 기준 노이즈
    if stem in NOISE_FILES:
        return FileClassification(
            stem=stem,
            role=FileRole.NOISE,
            appliance_type=stem,
            load_class=LoadClass.PASSIVE,
            noise_floor_w=NOISE_FILES[stem],
            periodic_duty=False,
            low_load=False,
            reason="무부하 기준 노이즈 파일 - 배경 노이즈 풀 전용",
        )

    # 3) 등록된 단일 가전
    if stem in DEVICE_FILES:
        spec = DEVICE_FILES[stem]
        return FileClassification(
            stem=stem,
            role=FileRole.DEVICE,
            appliance_type=spec.appliance_type,
            load_class=spec.load_class,
            noise_floor_w=spec.noise_floor_w,
            periodic_duty=spec.periodic_duty,
            low_load=spec.low_load,
            reason=f"등록된 단일 가전 '{spec.appliance_type}'",
        )

    # 4) 미등록 - 추측하지 않는다
    return FileClassification(
        stem=stem,
        role=FileRole.UNKNOWN,
        appliance_type=None,
        load_class=LoadClass.PASSIVE,
        noise_floor_w=NOISE_FLOOR_EXTERNAL_W,
        periodic_duty=False,
        low_load=False,
        reason="미등록 파일 - file_registry.DEVICE_FILES 에 등록하거나 파일명을 test_* 로 바꾸세요",
    )


class UnregisteredFileError(ValueError):
    """미등록 파일을 엄격 모드에서 처리하려 할 때 발생한다."""


def require_known(classification: FileClassification) -> FileClassification:
    """미등록 파일이면 조용히 넘어가지 않고 즉시 실패시킨다."""
    if classification.role == FileRole.UNKNOWN:
        raise UnregisteredFileError(
            f"'{classification.stem}' 의 역할을 알 수 없어 처리를 중단합니다.\n"
            f"  - 단일 가전이라면 src/preprocessing/file_registry.py 의 DEVICE_FILES 에 등록하세요.\n"
            f"  - 여러 가전이 동시에 돌아간 검증용 녹화라면 파일명을 test_* 로 시작하게 바꾸세요.\n"
            f"  - 무부하 노이즈라면 NOISE_FILES 에 등록하세요."
        )
    return classification


# ── 가전 종류(appliance_type) 단위 조회 ──────────────────────────────────────
# 파일은 여러 개여도 가전 종류는 하나다(fan_1/2/3 -> fan). 합성 단계에서는
# 파일이 아니라 가전 종류로 다루므로 종류 단위 속성이 따로 필요하다.

def _build_appliance_index() -> Dict[str, DeviceSpec]:
    index: Dict[str, DeviceSpec] = {}
    for spec in DEVICE_FILES.values():
        # 같은 가전 종류의 파일들은 동일한 물리 속성을 가진다고 본다.
        index.setdefault(spec.appliance_type, spec)
    return index


APPLIANCE_SPECS: Dict[str, DeviceSpec] = _build_appliance_index()


def get_load_class(appliance_type: str) -> LoadClass:
    """가전 종류의 전압 응답 물리 분류를 반환한다 (미등록 시 저항성으로 가정)."""
    spec = APPLIANCE_SPECS.get(appliance_type)
    return spec.load_class if spec else LoadClass.RESISTIVE


def is_periodic_duty(appliance_type: str) -> bool:
    """서모스탯/릴레이 주기 부하 여부. 시간 워핑 방식을 결정한다."""
    spec = APPLIANCE_SPECS.get(appliance_type)
    return bool(spec and spec.periodic_duty)


def is_low_load(appliance_type: str) -> bool:
    """대기전력과 혼동될 수 있는 저전력 가전 여부. 하드 네거티브 생성에 쓴다."""
    spec = APPLIANCE_SPECS.get(appliance_type)
    return bool(spec and spec.low_load)


def get_low_load_appliances() -> List[str]:
    """대기전력 오탐이 일어나기 쉬운 저전력 가전 목록."""
    return sorted(a for a, s in APPLIANCE_SPECS.items() if s.low_load)


def get_resistive_appliances() -> List[str]:
    """순수 저항 발열 부하 목록.

    이 기기들은 니크롬선만 있어 고조파가 거의 없다. 실측에서 전기포트와 오븐 히터의
    고조파 지문 거리는 0.596%p 로 사실상 같은 신호이며, 서로를 가르는 단서는
    시간 패턴(포트는 한 번에 끝, 오븐은 주기 반복)뿐이다.
    그런데 하필 이 기기들이 사용 빈도가 낮아 학습 표본이 가장 적으므로,
    합성 단계에서 따로 챙겨 줘야 한다.
    """
    return sorted(
        a for a, s in APPLIANCE_SPECS.items() if s.load_class == LoadClass.RESISTIVE
    )


def get_usage_probability(appliance_type: str) -> float:
    """임의의 순간에 이 가전이 켜져 있을 확률 (하루 사용 시간 / 24)."""
    spec = APPLIANCE_SPECS.get(appliance_type)
    hours = spec.daily_usage_hours if spec else 2.0
    return float(min(max(hours / 24.0, 0.0), 1.0))


def get_usage_probabilities() -> Dict[str, float]:
    """모든 가전의 가동 확률."""
    return {a: get_usage_probability(a) for a in APPLIANCE_SPECS}


def get_all_appliance_types() -> List[str]:
    """등록된 모든 가전 종류."""
    return sorted(APPLIANCE_SPECS.keys())
