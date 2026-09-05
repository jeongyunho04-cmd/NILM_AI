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
4. QUARANTINE     : 정체는 알지만 **어느 풀에도 넣지 않는** 파일 (12.184). 파이프라인은
                    건너뛰고, 탐침만 이름을 지정해 읽는다. 풀에 넣으려면 `QUARANTINED_FILES`
                    에서 빼면 된다 — 등록 자체는 DEVICE_FILES/NOISE_FILES 에 그대로 있다.
5. UNKNOWN        : 미등록. 기본적으로 오류를 발생시킨다.

[장소]
녹화가 세 장소에서 이뤄졌고 장소마다 전압 고조파가 다르다 (12.179.4, 12.184). `site_of()`
가 stem -> 'A'/'B'/'C' 를 준다. 판정 근거는 vrms 와 vh3/vh9/vh15 의 지문이다:
    A  218~224V  vh3 3.9~4.5  vh9 1.4~2.1  vh15 1.4~2.1   (격리 녹화 대부분, test_3~13)
    B  225~228V  vh3 4.1~4.6  vh9 3.2      vh15 0.2~0.3   (test_14~18, 9/02 의 포트·핫플 녹화)
    C  234~239V  vh3 9~10     vh9 2.7~3.3  vh15 0.2~0.3   (8/26 의 charger_4/hair_dryer_1, 9/04 의 *_C)

[위상 복원]
`PHASE_FIX_DEG_PER_ORDER` — 펌웨어 위상 교정이 틀린 채 녹화된 파일을 읽을 때 되돌린다 (12.184.3).
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import re  # noqa: E402


class FileRole(str, Enum):
    """원본 CSV 파일의 측정 성격."""
    DEVICE = "device"                   # 단일 가전 단독 측정 -> 합성 세그먼트 풀 사용 가능
    NOISE = "noise"                     # 무부하 기준 노이즈 -> 배경 노이즈 풀로만 사용
    COMPOSITE_EVAL = "composite_eval"   # 복합 부하 실측 -> 검증 전용, 세그먼트 풀 진입 금지
    QUARANTINE = "quarantine"           # 정체는 알지만 풀 진입 보류 -> 파이프라인이 건너뛴다
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
    # 2026-08-27 추가 (3차 펌웨어, 레인지 전환 수정본). 약/강 두 단계를 다 담았다 —
    # **약은 반파 정류다**: P 가 정확히 절반(466 vs 879W)이고 |I2|/|I1| 0.395 로
    # 이론값 0.424 에 붙는다. 옛 녹화(hair_dryer_1/2)도 같다 (0.414). 12.109 참조.
    "hair_dryer_3":     DeviceSpec("hair_dryer",       LoadClass.RESISTIVE, daily_usage_hours=0.1),
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
    # ── 펌웨어 수정 후 재측정 (2026-08-25, 설계 문서 12.72) ──────────────────
    # 레인지 전환 오프셋 단차를 고친 펌웨어로 다시 잡은 녹화다. **짝수차 고조파가
    # 절반이 된다** (충전기 |I2|/|I1| 0.0551 -> 0.0253). 홀수차는 불변이므로
    # (0.9281 -> 0.9261) 기존 녹화와 **같은 풀에 섞어 쓴다** — 어차피 짝수차는
    # 입력·손실 양쪽에서 배제하는 방향이라(12.74, 12.75) 무해하고, 두 펌웨어의
    # 짝수차가 섞이면 그 채널에 대한 강건성이 오히려 는다.
    "beam_projector_3_fixed":  DeviceSpec("beam_projector",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=2.0),
    "electric_kettle_2_fixed": DeviceSpec("electiric_kettle", LoadClass.RESISTIVE, daily_usage_hours=0.15),
    "hotplate_3_fixed":        DeviceSpec("hotplate",         LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    "laptop_charger_3_fixed":  DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    # 2026-08-26 추가. 237V 회선, 9.2분, 52 -> 70W 로 **오르는** 구간이다
    # (기존 녹화는 전부 65 -> 39W 로 내려가는 테이퍼였다). 12.30.6 이 "충전기는
    # 다른 두 기기의 지문 위를 가로지르는 궤적" 이라 했는데, 그 궤적의 반대쪽
    # 끝을 채운다. 체크리스트 B-1 의 "충전 상태를 바꿔 가며" 항목이다.
    "laptop_charger_4_fixed":  DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    "oven_3_fixed":            DeviceSpec("oven",             LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    # ── 2026-09-04 장소 C, 펌웨어 v5 (vhdeg1~15 추가) ────────────────────────────
    # 충전기: 96분, 배터리 30~40% -> 97% 를 끝까지 (72W 정속 60분 뒤 CV 테이퍼 68 -> 27W).
    # ⚠ 녹화 직전에 **부하 없이 USER 버튼 위상 교정**이 눌려 LOW 레인지 전체가 −10.8°×h
    # 돌아 있다 (12.184.3: 같은 부팅의 대기전류 위상이 +10.8 -> +0.6 로 떨어졌고, 옛 녹화
    # laptop_charger_4_fixed 와 크기 1.00~1.07 / 위상 h 선형 잔차 0.5°). 읽을 때
    # PHASE_FIX_DEG_PER_ORDER 로 되돌린다 — 되돌리면 옛 녹화와 모든 차수 1° 안.
    # 격리 이유는 그것이 아니라 **장소 C 혼입**(0‴): 풀 편입은 사용자 결정.
    "laptop_charger_5C":       DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    # 드라이기: 5분, 강(1009W, HIGH)/약(513W 반파, LOW·HIGH 불감대) 교대. 리셋 뒤라 교정은
    # 기본값 — 강 기본파 위상 −0.01° (순저항 확인). 약 모드가 LOW/HIGH 라벨을 오가며
    # 같은 원시 신호에 0.44°/2.62° 를 번갈아 적용해 2.2° 얼룩이 보인다 (12.184.7).
    "hair_dryer_4C":           DeviceSpec("hair_dryer",       LoadClass.RESISTIVE, daily_usage_hours=0.1),
    # 미니PC: 19분, 새 부팅(13:50, seq 0). IDLE 9.2W 8분 -> 부하 12~30W 계단 -> 끝 2분은 어댑터
    # 대기(2.6W). 교정은 기본값 — 기본파가 장소 A 곡선과 +1.4° 안이고 어댑터 대기 위상도
    # 장소 A 와 맞는다 (12.184.8). LOW 레인지만.
    "minipc_4C":               DeviceSpec("minipc",           LoadClass.SMPS,      low_load=True,  daily_usage_hours=10.0),
    # 전기포트: 6분, 1485W 6.4A HIGH, on/off 계단 3회 (7.8 / 229.5 / 373s). 순저항 검정용 (12.184.11):
    # 두 채널이 h3 에서 V1 의 3%, h5~h15 에서 0.3~0.6% 어긋난다. 무부하 대기전류 위상 +10.2° = 기본 교정.
    "electric_kettle_4C":      DeviceSpec("electiric_kettle", LoadClass.RESISTIVE, daily_usage_hours=0.15),
    # 프로젝터: 16분, ON 49W(LOW) 4구간 + 대기 3.1W. 17:15~17:31 녹화 — 펌웨어 LOW 교정을 바꾸기 **전**이다
    # (대기 위상 +65° = 장소 A 대기 +65.5°, 옛 규약). LOW_CAL_OVERRIDE 로 0.44 를 못박는다 (12.184.14).
    "beam_projector_4C":       DeviceSpec("beam_projector",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=2.0),
    # ── 2026-09-02 장소 B 단독 녹화 (3차 펌웨어). 지금까지 **미등록**이라 `run_preprocess_and_label`
    # 이 data/ 전체를 돌면 여기서 멈췄다. 선로 임피던스(12.167, z_site2)에만 썼고 풀에는 안 들어갔다.
    # 풀 편입은 사용자 결정으로 남긴다 (QUARANTINED_FILES) — 장소 B 저항 부하가 풀에 들어가면
    # 저항 서명의 장소 분포가 바뀐다.
    "electric_kettle_3_new":   DeviceSpec("electiric_kettle", LoadClass.RESISTIVE, daily_usage_hours=0.15),
    "hotplate_4_new":          DeviceSpec("hotplate",         LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
}


# ── 무부하 기준 노이즈 파일 등록부 ───────────────────────────────────────────
NOISE_FILES: Dict[str, float] = {
    "noise_noselfpower": NOISE_FLOOR_EXTERNAL_W,
    "noise_selfpower": NOISE_FLOOR_SELFPOWER_W,
    # 2026-09-04 장소 C, 4차 펌웨어, 외부 전원, 5분. 중앙 1.70W / |I1| 7.3mA (k≈1.0).
    # 격리 상태 — 노이즈 풀의 `noise_signature`(1.41W, k 1.37)를 조용히 바꾸지 않도록.
    "noise_noselfpower_C": NOISE_FLOOR_EXTERNAL_W,
}


# ── 격리 목록: 등록은 됐지만 어느 풀에도 넣지 않는다 (stem -> 이유) ──────────────
# 파이프라인(`run_preprocess_and_label`, `process_directory`)은 이 파일을 건너뛴다.
# 탐침이 이름을 지정해 원본 CSV 를 읽는 것은 막지 않는다 (`run_circuit_gate_probe`).
QUARANTINED_FILES: Dict[str, str] = {
    "laptop_charger_5C":   "장소 C 녹화 — 풀(장소 A)에 섞을지는 사용자 결정 (0‴). 위상은 PHASE_FIX 로 복원됨 — 12.184.3",
    "hair_dryer_4C":       "장소 C 녹화 — 풀 혼입은 사용자 결정 (0‴) — 12.184.7",
    "minipc_4C":           "장소 C 녹화 — 풀 혼입은 사용자 결정 (0‴) — 12.184.8",
    "electric_kettle_4C":  "장소 C 녹화 — 풀 혼입은 사용자 결정 (0‴) — 12.184.11",
    "beam_projector_4C":   "장소 C 녹화 — 풀 혼입은 사용자 결정 (0‴) — 12.184.14",
    "noise_noselfpower_C": "펌웨어 v5 + 장소 C 배경. 노이즈 풀 서명을 바꾸지 않도록 보류 — 12.184.4",
    "electric_kettle_3_new": "장소 B 단독 녹화, 지금까지 미등록. 풀 편입은 사용자 결정 (12.184.5)",
    "hotplate_4_new":        "장소 B 단독 녹화, 지금까지 미등록. 풀 편입은 사용자 결정 (12.184.5)",
}


# ── 장소 ─────────────────────────────────────────────────────────────────────
# 전압 고조파 지문으로 판정했다 (모듈 docstring). 등록 안 된 stem 은 "" 를 준다.
SITE_OF_STEM: Dict[str, str] = {
    **{s: "A" for s in ("test.2", "test3", "test_4", "test_5", "test_6", "test_7", "test_8",
                        "test_9", "test_10", "test_11", "test_12", "test_13",
                        "air_conditioner", "beam_projector", "beam_projector_2", "electiric_kettle",
                        "fan_1", "fan_2", "fan_3", "hair_dryer_2", "hair_dryer_3", "hotplate_1",
                        "hotplate_2", "laptop_charger_1", "laptop_charger_2", "minipc_1", "minipc_2",
                        "minipc_3", "oven", "oven_2", "beam_projector_3_fixed", "electric_kettle_2_fixed",
                        "hotplate_3_fixed", "laptop_charger_3_fixed", "oven_3_fixed",
                        "noise_noselfpower", "noise_selfpower")},
    **{s: "B" for s in ("test_14", "test_15", "test_16", "test_17", "test_18",
                        "electric_kettle_3_new", "hotplate_4_new")},
    **{s: "C" for s in ("laptop_charger_4_fixed", "hair_dryer_1",
                        "laptop_charger_5C", "hair_dryer_4C", "minipc_4C", "electric_kettle_4C",
                        "beam_projector_4C", "noise_noselfpower_C")},
}


def site_of(stem: str) -> str:
    """stem -> 'A' / 'B' / 'C'. 모르면 ''."""
    return SITE_OF_STEM.get(_normalize_stem(stem), "")


# ── 전류 위상 복원 [°/차수] ───────────────────────────────────────────────────
# 펌웨어의 위상 교정은 "h차 빈을 −h×delay 만큼 회전" 하는 시간지연 모델이라, 교정값이
# 틀리면 모든 차수가 h 에 비례해 돈다. 그 파일을 읽을 때 `ihdeg_h += fix × h` 로 되돌린다
# (`raw_csv.read_raw_csv`, `pipeline.process_file` 이 적용). 값의 근거는 등록부 주석과 설계 12.184.3.
# 2026-09-05 (12.185.6): 원시 스냅샷으로 **직접 쟀다**. 원시는 v 와 i 가 같은 표본 인덱스에서
# 나오므로 V–I 정렬이 참값이다. 충전기 5개 스냅샷(19~72W, 두 세션) 대 5C 의 같은 전력대:
#   −9.29 −9.29 −9.45 −9.69 −9.70 °/차수 (중앙값 −9.45, 선형 잔차 0.3~0.7° -> 순수 지연)
# 기준 파일 beam_projector_4C 는 원시와 +0.07°/차수 로 사실상 일치하므로, 5C 를 그 관계로
# 맞추는 값이 **+9.5** 다. 옛 값 10.8 (대기전류 위상으로 추정) 은 1.3°/차수 과교정이었다 —
# h15 에서 19°. ⚠ 원인(부하 없는 USER 버튼 교정 대 DFT 창 오정렬)은 둘 다 −k×h 를 내므로
# 이 자료로 못 가른다. 값만 고친다. 불확도 약 ±0.2°/차수.
PHASE_FIX_DEG_PER_ORDER: Dict[str, float] = {
    "laptop_charger_5C": 9.5,     # 원시 스냅샷 실측 (12.185.6). 0.44 ms ≈ 6.8 표본 상당
}
#: minipc_4C 도 원시 대비 −0.7 ~ −2.4 °/차수 로 어긋나 보이지만, **원시 세션끼리도 1표본 다르고**
#: 선형 잔차가 1.6~2.3° 라 순수 지연으로 안 떨어진다 (충전기는 0.3~0.7°). 값을 정할 수 없으므로
#: 보정하지 않는다 — 재녹화 대상 (12.185.6). h13 에서 15~30° 어긋날 수 있음을 감안하고 읽어라.


def phase_fix_of(stem: str) -> float:
    """stem -> 되돌릴 위상 [°/차수]. 없으면 0."""
    return PHASE_FIX_DEG_PER_ORDER.get(_normalize_stem(stem), 0.0)


# ── 원시 파형 스냅샷 (장소 C, 2026-09-04 22~23시) ────────────────────────────
# 40주기 × 256표본 (15360Hz). 열: t_s,cyc,seq,n,high,v_r1,low,v_r2,range,i_a,v_v,
# i_low_a,i_high_a,fs_hz,off_high,off_low,off_volt,gap_before.
# 2Hz 파일과 달리 v 와 i 가 같은 표본 인덱스에서 나오므로 **V–I 정렬이 참값**이다
# (2Hz 쪽 세션 오프셋을 이것으로 잰다). 회로 파라미터 적합의 정본 자료 (`synthesis.fit_raw`).
# ⚠ 파일명 오타(latop, charge_3)는 원본 그대로 둔다.
RAW_SNAPSHOT_FILES: Dict[str, List[str]] = {
    "laptop_charger": ["raw_latop_charger_1", "raw_laptop_charger_2", "raw_laptop_charge_3",
                       "raw_laptop_charger_4", "raw_laptop_charger_5"],
    "beam_projector": ["raw_beam_projector_1", "raw_beam_projector_2"],
    "minipc": ["raw_minipc_1", "raw_minipc_2", "raw_minipc_3", "raw_minipc_4", "raw_minipc_5"],
}
#: 원시 스냅샷 중 LOW/HIGH 가 섞인 것 (펄스 피크가 LOW 포화 2.2A 를 넘음 — 이어 붙인 파형이라
#: 두 경로의 위상 응답이 한 파형에 섞인다, 12.184.15b). 적합에 쓰되 잔차를 따로 본다.
RAW_RANGE_MIXED = {"raw_latop_charger_1", "raw_laptop_charger_2"}


def raw_snapshots_of(device: str) -> List[str]:
    """기기 -> 원시 스냅샷 stem 목록. 없으면 빈 목록."""
    return list(RAW_SNAPSHOT_FILES.get(device, []))


#: 조합 원시 스냅샷 (장소 C, 2026-09-05 00:29~00:33) -> 켜져 있던 기기.
#: 조합 녹화는 **단자 전압 V_term 을 직접 잰다** — 고정점 반복 없이 결합을 검정할 수 있다 (12.185.12).
#: 전력 배분: 프로젝터·미니PC 는 단독 스냅샷 전력으로 고정하고 나머지를 충전기에 준다
#: (충전기만 배터리 상태로 변한다). 충전기는 여기서 28~33W 로 단독 스냅샷 17W 보다 크다.
#: 장소 C 순저항 원시 (⚠ 파일명 오타 `elcetric` 은 원본 그대로). 1485W / 6.4A / HIGH 86%.
#: **계측 판정의 정본** — v–i 정렬·전압 채널 인공물·짝수차를 모델 없이 가른다 (12.185.25).
RAW_RESISTIVE_FILES: List[str] = ["raw_elcetric_kettle_1", "raw_elcetric_kettle_2"]

RAW_COMBO_FILES: Dict[str, List[str]] = {
    "raw_beam_minipc_1": ["beam_projector", "minipc"],
    "raw_beam_minipc_2": ["beam_projector", "minipc"],
    "raw_beam_charger_1": ["beam_projector", "laptop_charger"],
    "raw_beam_charger_2": ["beam_projector", "laptop_charger"],
    "raw_smps3_1": ["beam_projector", "minipc", "laptop_charger"],
    "raw_smps3_2": ["beam_projector", "minipc", "laptop_charger"],
}
#: 조합에서 각 기기와 짝지을 단독 스냅샷 (같은 세션·같은 동작점을 고른다)
RAW_COMBO_SOLO: Dict[str, Dict[str, str]] = {
    "raw_beam_minipc_1": {"beam_projector": "raw_beam_projector_1", "minipc": "raw_minipc_1"},
    "raw_beam_minipc_2": {"beam_projector": "raw_beam_projector_2", "minipc": "raw_minipc_2"},
    "raw_beam_charger_1": {"beam_projector": "raw_beam_projector_1", "laptop_charger": "raw_laptop_charge_3"},
    "raw_beam_charger_2": {"beam_projector": "raw_beam_projector_2", "laptop_charger": "raw_laptop_charge_3"},
    "raw_smps3_1": {"beam_projector": "raw_beam_projector_1", "minipc": "raw_minipc_1",
                    "laptop_charger": "raw_laptop_charge_3"},
    "raw_smps3_2": {"beam_projector": "raw_beam_projector_2", "minipc": "raw_minipc_2",
                    "laptop_charger": "raw_laptop_charge_3"},
}
#: 조합 중 LOW/HIGH 가 섞인 것 (겹친 펄스 피크가 LOW 포화를 넘는다)
RAW_COMBO_RANGE_MIXED = {"raw_beam_charger_2", "raw_smps3_1", "raw_smps3_2"}


# ── 복합 녹화의 사용자 제공 타임라인 ─────────────────────────────────────────
# ⚠ 파일명이 `tesr_19.csv` 다 (오타, 원본 그대로 둔다).
# 사용자가 준 표는 `seq` 와 "분" 두 열인데, 실제로는 **seq 가 2Hz 프레임 번호**이고
# `t_s = (seq − 60) / 2` [초] 다 (주신 "분" 값 × 2 = t_s 초). 파일과 대조해 확인했다:
#   seq 226 -> t 83.0s, seq 334 -> 137.0s, seq 506 -> 223.4s, seq 738 -> 339.0s ✓
# 계단도 맞는다 (프로젝터 −46/+45W, 미니PC −9/+16W).
# ⚠ 충전기 OFF(seq 430) 만 −11W 계단 뒤 223초까지 86 -> 57W 로 **완만히** 내려간다 —
#   깨끗한 계단이 아니다. 그 구간을 델타 서명에 쓰지 마라.
TEST19_TIMELINE = [
    (60, 0.0, "start", None, "기록 시작 — 3종 전부 ON"),
    (226, 83.0, "beam_projector", "off", "ΔP −46W (사용자 표 −40W)"),
    (334, 137.0, "beam_projector", "on", "ΔP +45W (사용자 표 +52W)"),
    (430, 185.0, "laptop_charger", "off", "ΔP −11W 뒤 223초까지 완만한 하강 — 계단 아님"),
    (506, 223.4, "laptop_charger", "on", "ΔP +36W (사용자 표 +25W)"),
    (629, 284.5, "minipc", "off", "ΔP −9W (11W -> 0)"),
    (738, 339.0, "minipc", "on", "ΔP +16W (0 -> 19W)"),
]


def test19_events():
    """`tesr_19.csv` 의 (seq, t_s, 기기, on/off, 비고). 라벨 없는 COMPOSITE_EVAL 의 참값."""
    return list(TEST19_TIMELINE)


# ── LOW 레인지 위상 교정값의 규약 (12.184.12~13, **12.184.16 에서 정정**) ──────────────
# 펌웨어는 레인지별 교정값 delay 로 모든 차수를 −h×delay 회전한다. 원래 기본값 LOW 0.44° / HIGH 2.62° 는
# 펌웨어 작성자가 각 경로에 순저항(납땜인두 / 포트)을 물려 직접 잰 값이다. 2026-09-04 저녁에 "두 경로의
# 지연이 같아야 한다" 고 보고 LOW 를 2.62 로 바꿨는데 (12.184.12), 그 근거였던 드라이기 불감대 자료는
# 표본마다 LOW/HIGH 를 이어 붙인 **같은 파형**에 라벨만 번갈아 붙은 것이라 경로 지연을 재 주지 않는다.
# 2.18° 는 LOW 경로의 안티앨리어싱 RC(1kΩ·100nF, fc 1591Hz) 한 극이 60Hz 에서 내는 2.16° 와 같다 — 즉
# **원래 0.44 가 옳다** (12.184.16). 정본 = 0.44. 2.62 로 플래시된 보드로 찍은 파일은 읽을 때 `range==0`
# 사이클을 +2.18°×h 돌려 정본으로 되돌린다.
# ⚠ 원본 CSV 를 읽는 경로(read_raw_csv, pipeline.process_file)에만 걸린다. npz·캐시·체크포인트는 전부 정본(0.44)이다.
LOW_CAL_DEG_CANONICAL = 0.44      #: 정본 = 펌웨어 원래 기본값 (순저항으로 잰 값)
LOW_CAL_DEG_LEGACY = 0.44         #: 2026-09-04 저녁 이전 모든 녹화의 값 (= 정본)
LOW_CAL_DEG_FLASHED = 2.62        #: 2026-09-04 저녁에 잘못 올린 값. 되돌리기 전까지 찍은 파일에만 해당
#: 이 시각 이후의 첫 host_time 이면 2.62 보드로 찍은 것으로 본다. ⚠ 잠정 — 실제 플래시 시각을 모른다.
#: 보드를 0.44 로 되돌리면 LOW_CAL_FLASHED_ACTIVE 를 False 로 — 그러면 시각 판정을 안 한다.
LOW_CAL_FIXED_AT = "2026-09-04 18:30:00"
LOW_CAL_FLASHED_ACTIVE = False    #: 2026-09-04 저녁 보드를 0.44 로 되돌렸다
#: stem -> 녹화 당시 LOW 교정값 [°]. 자동 판정(host_time)을 덮는다.
#
# ⚠ 12.185.2 (2026-09-05): 대기전류 h1 위상(+65.8°)으로 판정한 옛 근거는 **성립하지 않는다**.
#   대기 부하가 파일마다 다르고(미니PC 대기 h3/h1 0.278 vs 프로젝터 대기 0.348), 같은 파일 안에서도
#   바뀐다(프로젝터 대기0 vs 대기 크기비 0.85). h1 하나로는 h 선형 회전을 못 잰다.
#   같은 기기·같은 동작점의 **계단 델타**로 다시 재니 test_19/20 은 minipc_4C 대비 −2.4~−2.9°/차수
#   (크기비 1.01~1.03, 즉 순수 회전) 였고, 장소 전압 차가 내는 몫은 회로 모델로 +0.33°/차수뿐이었다.
#   -> 두 파일은 다른 교정 판에서 찍혔다. 사용자가 삭제하기로 했다 (`run_cal_epoch_probe`).
#   beam_projector_4C 는 이 방법으로 판정이 안 된다 (프로젝터 서명이 예열로 같은 파일 안에서 +0.7°/차수
#   흐른다). 사용자 진술("교정값 바뀌기 전에 측정")에만 근거한다 — 회로 모델에서 참고 기기로만 쓴다.
LOW_CAL_OVERRIDE: Dict[str, float] = {
    "beam_projector_4C": LOW_CAL_DEG_LEGACY,     # 17:15 녹화 — 사용자 진술로만 "플래시 전" (자료로는 미확인)
}


def low_cal_of(stem: str, host_time_first: Optional[str] = None) -> float:
    """그 파일이 녹화될 때 적용돼 있던 LOW 교정값 [°]. host_time 이 없으면 정본(0.44)으로 본다."""
    s = _normalize_stem(stem)
    if s in LOW_CAL_OVERRIDE:
        return LOW_CAL_OVERRIDE[s]
    if not LOW_CAL_FLASHED_ACTIVE or host_time_first is None:
        return LOW_CAL_DEG_LEGACY
    return LOW_CAL_DEG_LEGACY if str(host_time_first) < LOW_CAL_FIXED_AT else LOW_CAL_DEG_FLASHED


def low_cal_shift_deg(stem: str, host_time_first: Optional[str] = None) -> float:
    """`range==0` 사이클에 걸 회전 [°/차수] = −(정본 − 녹화 당시). 정본 보드로 찍은 파일이면 0, 2.62 보드면 +2.18."""
    return -(LOW_CAL_DEG_CANONICAL - low_cal_of(stem, host_time_first))


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

    # 2) 격리 — 정체는 알지만 풀 진입 보류. 등록부의 명세는 그대로 달아서 돌려준다.
    if stem in QUARANTINED_FILES:
        spec = DEVICE_FILES.get(stem)
        return FileClassification(
            stem=stem,
            role=FileRole.QUARANTINE,
            appliance_type=spec.appliance_type if spec else stem,
            load_class=spec.load_class if spec else LoadClass.PASSIVE,
            noise_floor_w=spec.noise_floor_w if spec else NOISE_FILES.get(stem, NOISE_FLOOR_EXTERNAL_W),
            periodic_duty=bool(spec and spec.periodic_duty),
            low_load=bool(spec and spec.low_load),
            reason=f"격리: {QUARANTINED_FILES[stem]}",
        )

    # 3) 무부하 기준 노이즈
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

    # 4) 등록된 단일 가전
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

    # 5) 미등록 - 추측하지 않는다
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


def get_smps_appliances() -> List[str]:
    """SMPS(정전력 스위칭 전원) 가전 목록.

    프로젝터·충전기·미니PC 가 여기 들어간다. 12.81 이 미니PC 미검출의 조건을
    "경쟁 SMPS 가 함께 켜져 있을 때" 로 좁혔고(경쟁 없으면 재현율 99%,
    있으면 30~67%), 12.88.4 가 남긴 유일한 축이 그 조합을 학습에서 늘리는 것이다.
    합성 레시피 `smps_overlap` 이 이 목록에서 뽑는다.
    """
    return sorted(
        a for a, s in APPLIANCE_SPECS.items() if s.load_class == LoadClass.SMPS
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


# ── 원시 파형의 위상 교정 (12.185.25) ────────────────────────────────────────
# 펌웨어의 위상 교정(`NILM_CAL_DEFAULT_LOW_DEG` 0.44° / HIGH 2.62°, 차수당)은 **2Hz 고조파
# 블록에만** 걸린다 — 주파수영역에서 `ihdeg_h -= cal·h` 로 도는 연산이라 시간영역 원시
# 표본에는 적용될 수 없다. 그래서 **원시 파형에는 전류-전압 채널 어긋남이 그대로 남아 있다.**
#
# 장소 C 포트 원시(`raw_elcetric_kettle_1/2`, 순저항 1485W)로 직접 쟀다. 순저항은
# `i(t) = v(t)/R` 이 모든 표본에서 성립하므로 회로 모델 없이 잰다:
#     ∠I₁ − ∠V₁ = +2.864° / +2.871°   (두 파일, 기본파 SNR ~1000)
#     펌웨어 HIGH 상수                  2.62°     <- 방향 일치, 크기 90%
# **전류가 전압보다 앞선다.** 그러므로 늦은 v 로 계산한 시뮬 전류는 실측보다 늦고,
# 비교 전에 **앞당겨야** 한다 (`fit_raw.sim_current` 의 `i_skew_samp` 가 음수).
#
# 원시 스냅샷 18개는 전부 LOW 레인지다 (피크 0.27~1.91A < 2.22A, `range!=0` ≤0.1%).
# 그래서 LOW 상수를 쓴다. 표본 하나 = 360°·60Hz/15360Hz = 1.406°/차수.
#
# 유보 자료(조합 6개, 적합에 안 들어감)로 확인: 스큐 0 -> 5.92%, −0.313 -> 4.21% (−29%).
# 그리고 L 이 780/536/1014 -> 846/616/1245µH 로 **올라간다** — 자리표들이 L 을 무너뜨린 것과
# 정반대다 (12.185.23·24 의 nvt·계측극·fc·양의 스큐).
# ⚠ 유보 자료의 최소는 −0.500 표본(0.70°/차수)이고 펌웨어 상수는 −0.313 이다. 0.19 표본
#    차이는 미결이다 — 가르려면 **LOW 레인지 순저항**(백열전구·납땜인두 0.3~1A) 원시가 필요하다.
RAW_PHASE_CAL_DEG_PER_ORDER: Dict[str, float] = {"LOW": 0.44, "HIGH": 2.62}
RAW_SAMPLES_PER_CYCLE = 256
#: 원시 비교에 쓸 기본 스큐 [표본]. 음수 = 시뮬 전류를 앞당긴다.
RAW_SKEW_SAMP_LOW = -RAW_PHASE_CAL_DEG_PER_ORDER["LOW"] * RAW_SAMPLES_PER_CYCLE / 360.0
