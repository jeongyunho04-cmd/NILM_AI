"""
NILM 원본 파일 역할 레지스트리 (File Role Registry)
====================================================
data/ 안의 모든 CSV가 "어떤 성격의 측정인지"를 결정하는 단일 출처(Single Source of Truth).

⚠ **2026-09-06 계측기 교체 — 이 등록부는 그날 새로 시작했다.** `READ_ME_FIRST.md` 를 먼저 읽어라.
   옛 계측기(차동 ADC, VINN=1.65V 고정)는 VINP 가 2.65V(≈VDDA−0.65V)를 넘는 위쪽 피크에서
   입력단이 비선형이라 읽기를 부풀렸고, 그것이 짝수차 인공물·전압 채널 h3 인공물·장소 지문의
   정체였다. 그 계측기로 찍은 자료는 **전부 삭제**됐고, 그 자료로 세운 결론은 전부 재검토 대상이다.
   옛 파일명 등록은 `LEGACY_*` 로 옮겨 두었다 — `classify_file` 은 그 이름을 **받지 않는다**
   (같은 이름의 새 녹화가 옛 자료로 오인되는 것을 막는다). 새 녹화는 `DEVICE_FILES`/`NOISE_FILES`
   에 새로 등록한다. 설계 문서 13.1.

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
5. UNKNOWN        : 미등록. 기본적으로 오류를 발생시킨다. **옛 계측기 파일명도 여기로 온다.**

[파일명 규약]
새 수신기는 교정본 CSV 에 `.cal2` 를 붙인다 (`hotplate_1.cal2.csv`). `_normalize_stem` 이 그
꼬리를 떼므로 등록·산출물(npz, 라벨) 이름은 `hotplate_1` 이다. `beam_projector_2.csv` 처럼
꼬리 없는 파일도 같은 계측기 자료다 (열 구성·`cal_applied=1` 동일) — 꼬리는 계측기 세대의
표지가 **아니다**. 세대의 표지는 이 등록부와 `LEGACY_DATA_CUTOFF` 다.

[가전 종류 ↔ 파일]
가전 종류의 물리 속성은 `APPLIANCE_CATALOG` 에 있다 (닫힌 세계 9종, 3.3절). 파일 등록은
그 종류를 가리킨다. 파일이 아직 없는 종류(선풍기·드라이기·에어컨, 2026-09-06 현재)도
카탈로그에는 있다 — 합성기는 풀에 있는 종류만 쓰므로(`SegmentPool.get_appliance_types`)
녹화가 들어오기 전에는 모델 머리 수가 9 가 아니다. READ_ME_FIRST.md 의 "없는 것" 참조.

[장소]
**장소는 글자로만 부른다** (사용자 지시 2026-09-06). 시간대·이름을 붙이지 않는다 — 같은 자리도
시각에 따라 전압이 바뀌므로 "저녁 장소" 같은 이름은 곧 틀린 말이 된다. 지금 자리는 **D 와 E** 다
(옛 계측기 시대의 A/B/C 와 겹치지 않게 새 글자를 썼다, `LEGACY_SITE_OF_STEM_OLD_ADC`).

장소를 가르는 것은 **전압 고조파 스펙트럼**이고, 그중 vh7 이 가장 깨끗하다 (겹침이 없다):
    D   vrms 210.7~219.4   vh3 0.59~0.85%   **vh7 0.89~1.28%**   vh9 0.50~0.64%
    E   vrms 227.5~231.0   vh3 2.80~3.25%   **vh7 0.08~0.19%**   vh9 0.92~1.13%
vh3 은 4배, vh7 은 7배 차이고 사이가 비어 있다. 전압 크기(216 대 229V)만으로는 못 가른다 —
같은 자리도 세션마다 몇 V 씩 움직인다 (D 의 두 세션이 215.2~219.4 와 210.7~216.7).

**전압·임피던스는 자리 × 세션의 성질이다.** 자리는 안 바뀌지만 그 값은 바뀐다. `SITE_PROFILE` 의
숫자는 대푯값이고, 세션별 값은 `SITE_SESSIONS` 에 있다.

[위상 복원]
`PHASE_FIX_DEG_PER_ORDER` — 펌웨어 위상 교정이 틀린 채 녹화된 파일을 읽을 때 되돌린다 (12.184.3).
새 계측기 자료에서는 저항 부하(포트 HIGH / 핫플 LOW / 오븐 HIGH)의 `ihdeg1` 이 −0.05~+0.06° 라
(규칙 74) 되돌릴 것이 없다. 비어 있다.
"""
from dataclasses import dataclass, replace as _dc_replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import re  # noqa: E402


# ── 계측기 세대 ───────────────────────────────────────────────────────────────
#: 지금 자료를 만든 계측기. 단일 입력 ADC(LSB 201.4µV) + 바이어스 채널 소프트웨어 감산.
MEASUREMENT_ERA = "single-ended-adc-2026-09-06"
#: 이 시각보다 앞선 `host_time` 으로 시작하는 CSV 는 옛 계측기(차동 ADC) 자료다.
#: 새 계측기의 첫 녹화가 2026-09-05 21:26 이다. 옛 자료는 삭제됐으므로 이 검사에 걸리는
#: 파일이 있다면 어딘가에서 다시 복사된 것이다 — 풀에 넣으면 안 된다.
LEGACY_DATA_CUTOFF = "2026-09-05 20:00:00"


class LegacyDataError(ValueError):
    """옛 계측기(2026-09-06 이전) 자료를 새 파이프라인에 넣으려 할 때."""


def check_measurement_era(host_time_first: Optional[str], name: str = "") -> None:
    """첫 `host_time` 이 `LEGACY_DATA_CUTOFF` 보다 앞서면 실패시킨다. host_time 이 없으면 통과."""
    if host_time_first is None:
        return
    if str(host_time_first) < LEGACY_DATA_CUTOFF:
        raise LegacyDataError(
            f"{name or '이 파일'} 의 첫 host_time({host_time_first})이 {LEGACY_DATA_CUTOFF} 보다 앞선다 — "
            "옛 계측기(차동 ADC) 자료다. 2026-09-06 에 전부 폐기했다 (READ_ME_FIRST.md). "
            "새 계측기로 다시 재라.")


class FileRole(str, Enum):
    """원본 CSV 파일의 측정 성격."""
    DEVICE = "device"                   # 단일 가전 단독 측정 -> 합성 세그먼트 풀 사용 가능
    NOISE = "noise"                     # 무부하 기준 노이즈 -> 배경 노이즈 풀로만 사용
    COMPOSITE_EVAL = "composite_eval"   # 복합 부하 실측 -> 검증 전용, 세그먼트 풀 진입 금지
    QUARANTINE = "quarantine"           # 정체는 알지만 풀 진입 보류 -> 파이프라인이 건너뛴다
    RAW = "raw"                         # 원시 파형 스냅샷 (`raw_*.csv`, 256표본/주기) -> 2Hz 파이프라인은 건너뛴다
    UNKNOWN = "unknown"                 # 미등록 -> 처리 거부


class LoadClass(str, Enum):
    """전압 변화에 대한 물리적 응답 특성 분류."""
    RESISTIVE = "resistive"   # 니크롬 히터: I∝V, P∝V²
    SMPS = "smps"             # 스위칭 전원: P 일정, I∝1/V
    MOTOR = "motor"           # 유도 전동기/인버터: I∝V^0.7, P∝V^0.7
    PASSIVE = "passive"       # 노이즈 등 부하 아님


# ── 계측 보드 전원 공급 방식별 무부하 바닥 전력 ──────────────────────────────
# 보드가 측정 대상 회로에서 자체 전원을 뽑으면(self-power) 보드 소비가 측정값에 포함된다.
# 2026-09-06 새 계측기로 다시 쟀다 (`noise_*` 3파일, pll_locked·클립 제외, 중앙값):
#     noise_noselfpower   1.59W (p5~p95 1.56~1.63)   |I1| 7.46mA ∠+12.2°   |I3| 1.76mA
#     noise_selfpower_1   2.41W (2.37~2.46)           |I1| 11.08mA ∠+4.9°   G 0.0506mS
#     noise_selfpower_2   2.21W (2.14~2.28)           |I1| 10.09mA ∠−1.4°   G 0.0460mS
# 옛 값 1.4 / 2.37 은 옛 계측기(장소 A)의 것이었다. 새 값으로 바꾼다 — 라벨 `p_target_w` 와
# 2단계의 `p_noise` 가 이 상수를 쓴다. (`circuit_model/fcm12.G_BG_DEFAULT` 0.051mS 와 맞는다.)
NOISE_FLOOR_EXTERNAL_W = 1.6    # 외부 전원 공급 시 계측계 바닥 노이즈
NOISE_FLOOR_SELFPOWER_W = 2.3   # 자체 전원 공급 시 보드 소비 포함 바닥값


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


# ── 가전 종류 카탈로그 (닫힌 세계 9종) ───────────────────────────────────────
# 파일이 아니라 **종류**의 물리 속성이다. 파일 등록(DEVICE_FILES)은 여기서 복사한다.
# 계측기가 바뀌어도 기기는 그대로이므로 이 표는 2026-09-06 이전 것을 그대로 쓴다.
APPLIANCE_CATALOG: Dict[str, DeviceSpec] = {
    "air_conditioner":  DeviceSpec("air_conditioner",  LoadClass.MOTOR,     daily_usage_hours=4.0),
    "beam_projector":   DeviceSpec("beam_projector",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=2.0),
    # ⚠ 철자 `electiric` 은 상태 정의·후처리·손실·라벨 파서가 전부 쓰는 키다. 파일명은 `electric_kettle_*` 로
    #   바르게 적되 종류 키는 바꾸지 않는다 (바꾸면 체크포인트의 머리 이름까지 전부 갈린다).
    "electiric_kettle": DeviceSpec("electiric_kettle", LoadClass.RESISTIVE, daily_usage_hours=0.15),
    "fan":              DeviceSpec("fan",              LoadClass.MOTOR,     low_load=True,  daily_usage_hours=5.0),
    "hair_dryer":       DeviceSpec("hair_dryer",       LoadClass.RESISTIVE, daily_usage_hours=0.1),
    # 핫플레이트는 릴레이가 약 1초 ON / 1초 OFF 로 통전을 끊는 주기 부하다.
    "hotplate":         DeviceSpec("hotplate",         LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
    "laptop_charger":   DeviceSpec("laptop_charger",   LoadClass.SMPS,      low_load=True,  daily_usage_hours=8.0),
    "minipc":           DeviceSpec("minipc",           LoadClass.SMPS,      low_load=True,  daily_usage_hours=10.0),
    # 오븐은 히터가 꺼져도 팬/조명 약 16W 가 남아 서모스탯 주기가 전력에 그대로 드러난다.
    "oven":             DeviceSpec("oven",             LoadClass.RESISTIVE, periodic_duty=True, daily_usage_hours=0.5),
}


def _dev(appliance_type: str, **override) -> DeviceSpec:
    """카탈로그의 종류 명세를 파일 등록용으로 복사한다 (파일별로 다른 값만 덮는다)."""
    return _dc_replace(APPLIANCE_CATALOG[appliance_type], **override)


# ── 단일 가전 측정 파일 등록부 (새 계측기, 2026-09-06 ~) ───────────────────────
# 새 가전을 측정하면 반드시 여기에 한 줄 추가해야 파이프라인이 받아들인다.
# 각 줄의 수치는 2026-09-06 에 원본 CSV 에서 직접 잰 것이다 (pll_locked·전압 클립 창 제외).
DEVICE_FILES: Dict[str, DeviceSpec] = {
    # 프로젝터 ×2. 09-06 00:40 / 01:10 (E1, 231V / 229V). ON 46.2 / 46.1W (사이클 최대 50.9 / 48.0),
    # 대기 3.1 / 2.9W, ∠I1 +7.5°, LOW 만. 두 파일 다 같은 계측기·같은 열 구성이다 (2 는 `.cal2` 꼬리가 없을 뿐).
    "beam_projector_1":  _dev("beam_projector"),
    "beam_projector_2":  _dev("beam_projector"),
    # 전기포트. 09-06 00:30 (E1, 228V). 1458W, HIGH 61%, on/off 여러 번. R = V²/P 35.61Ω (p5~p95 35.18~35.69).
    # 순저항 검정(규칙 74): ihdeg1 −0.05° (HIGH), |I3/I1| 3.23% = vh3/V1 3.25% (옛 계측기의 h3 바닥이 없다).
    "electric_kettle_1": _dev("electiric_kettle"),
    # 선풍기. 09-06 01:46 (E1, 229V). 9.7분, 1/2/3단 22.9 / 30.8 / 39.0W (옛 상태 문턱 10/27/36 그대로 맞는다),
    # ∠I1 +30 / +21 / +4°, PF 0.86 / 0.93 / 1.00, |I3|/|I1| 0.09 / 0.07 / 0.03, 짝수차 0.002. LOW 만. `.cal2` 꼬리 없음.
    "fan_1":             _dev("fan"),
    # 드라이기. 09-06 01:57 (E1, 227V). 8.2분, 강 965.8W(HIGH 100%, R 53.22Ω, |I2| 0.000, ∠I1 −0.11° — 순저항) /
    # 약 488.4W(반파: |I2|/|I1| 0.431 이론 0.424, |I4|/|I1| 0.102 이론 0.085, R 106.0Ω, LOW·HIGH 불감대 30%) 교대.
    # 옛 결론 12.109.2(약풍 = 다이오드 반파)가 새 계측기에서 그대로 선다. `.cal2` 꼬리 없음.
    "hair_dryer_1":      _dev("hair_dryer"),
    # 핫플레이트. 09-05 23:00 (D1, 215V). 456.5W 통전 41%, 릴레이 주기 부하, LOW 만. R 100.83Ω. ihdeg1 +0.05°.
    "hotplate_1":        _dev("hotplate"),
    # 충전기. 09-05 21:26 (D1, 216V). 16분, 65W 정속이 대부분(60~70W 87%), 20~60W 테이퍼 일부. 사이클 최대 68.2W.
    # ⚠ 72W 정전류 구간·돌입이 없다 — `postproc.ABSORB_CAP_W` 84.6 은 옛 녹화의 사이클 최대라 이 파일로 못 낮춘다.
    "laptop_charger_1":  _dev("laptop_charger"),
    # 미니PC. 09-05 22:17 (D1, 218V). 27분, IDLE 10~13W 28% / 부하 13~30W 64% / 대기 3.4W. 사이클 최대 29.5W.
    "minipc_1":          _dev("minipc"),
    # ── 2026-09-06 21:16~21:38 추가 (세션 E2, 230~231V). **E 의 미니PC** — 13.15.7 ①이 요청한 자료.
    # minipc_2: 14.7분. 대기 2.67W(10.5%) / 유휴 11.7·13.9W(29.4%) / 부하 21.0W(60.1%).
    # minipc_3:  6.1분. 같은 상태 구성. 사이클 최대 248W 는 돌입이고 1초 중앙값은 35W 를 안 넘는다.
    # 상태가 minipc_1(D1) 과 자리마다 맞는다 (2.7~3.4 / 10.7~11.9 / 13.5~14.3 / 21.0W) — 같은 기기다.
    # ⚠ |I3|/|I1| 이 D 0.681·0.751·0.813 대 E 0.833·0.866·0.901 로 **자리마다 다르다** (설계 13.18).
    "minipc_2":          _dev("minipc"),
    "minipc_3":          _dev("minipc"),
    # 에어컨. 09-06 02:11 (E1, 228V). 27.8분, 대기 6.8W/117mA∠+76°(꽂힌 대기, 용량성) / 송풍 10~30W / 냉방 80~640W
    # (인버터: |I3|/|I1| 0.58, PF 0.81, ∠I1 −15°), 사이클 최대 645W, HIGH 35%. 옛 상태 문턱(10/80/350/600W) 안에 다 든다.
    # 꼬리 20초 1.41W/6.2mA. `.cal2` 꼬리 없음.
    "air_conditioner_1": _dev("air_conditioner"),
    # 충전기 2. 09-06 02:42 (E1, 230V). 4.2분, 10~70W 테이퍼·정속 골고루, 꼬리 20초 1.47W/6.4mA. 규칙 2 를 지킨 첫 충전기 녹화.
    "laptop_charger_2":  _dev("laptop_charger"),
    # 오븐. 09-05 21:47 (D1, 216V). 18분, 히터 1101W(30.8%) / 팬·조명 16.2W(61.6%) / 대기 2.4W. R(히터) 40.09Ω.
    # 팬·조명 상태 |I1| 75mA, |I2|/|I1| 0.058 — 짝수차 신원(12.164.16)이 새 계측기에서도 그대로다.
    "oven_1":            _dev("oven"),
    # ── 2026-09-06 18:55~19:41 추가 (세션 D2, 215~216V). **D 에 프로젝터·핫플·오븐이 생겼다.**
    # 프로젝터 3. 18:55, 7.4분. ON 46.5W(71.8%) / 대기 2.9W(21.4%) / 예열 5.7W(6.8%).
    # ⚠ **∠I1 이 E 의 +7.5° 대신 −13.0° 다** — 같은 기기인데 자리가 바뀌면 기본파 위상이 20° 돈다
    # (|I3|/|I1| 도 0.86). 장소 반응의 직접 증거다 (설계 13.15). 원시 스냅샷 raw_beam_projector_3·4 (215.5V, 46.1W).
    "beam_projector_3":  _dev("beam_projector"),
    # 핫플 2. 19:09, 16.7분. 통전 454.8W(23.3%) / 대기 1.9W(71.1%). R 101.18Ω (hotplate_1 100.83Ω 과 같다).
    # 릴레이 주기 부하라 1초 중앙값에는 55W·172W 같은 부분 통전 칸이 생긴다 (상태가 아니다).
    "hotplate_2":        _dev("hotplate"),
    # 오븐 2. 19:31, 10.3분. 히터 1107.9W(31.3%) / 팬·조명 16.2W(40.7%) / 대기 2.5W(27.9%). R(히터) 40.12Ω.
    # 팬·조명 |I1| 75.4mA, |I2|/|I1| 0.070, ∠ −5.4° — oven_1 과 같은 상태다.
    # **오븐은 스위치를 켜면 반드시 이 16W 상태를 거쳐 히터로 간다** (사용자 진술, 라벨 정밀화가 이것을 쓴다).
    "oven_2":            _dev("oven"),
}


# ── 무부하 기준 노이즈 파일 등록부 (stem -> 바닥 전력) ───────────────────────
NOISE_FILES: Dict[str, float] = {
    "noise_noselfpower": NOISE_FLOOR_EXTERNAL_W,    # 09-05 22:45, 3분, 1.59W
    "noise_selfpower_1": NOISE_FLOOR_SELFPOWER_W,   # 09-05 22:48, 4분, 2.41W
    "noise_selfpower_2": NOISE_FLOOR_SELFPOWER_W,   # 09-05 22:53, 4.6분, 2.21W (앞 20초에 23W 과도 — 정제기가 거른다)
}
#: 노이즈 파일 stem -> 상태 정의(`STATE_CONFIGURATIONS`) 키. 파일이 여러 개라도 종류는 둘이다.
NOISE_TYPE_OF_STEM: Dict[str, str] = {
    "noise_noselfpower": "noise_noselfpower",
    "noise_selfpower_1": "noise_selfpower",
    "noise_selfpower_2": "noise_selfpower",
}


# ── 격리 목록: 등록은 됐지만 어느 풀에도 넣지 않는다 (stem -> 이유) ──────────────
# 파이프라인(`run_preprocess_and_label`, `process_directory`)은 이 파일을 건너뛴다.
# 탐침이 이름을 지정해 원본 CSV 를 읽는 것은 막지 않는다.
# 2026-09-06: 비어 있다. 옛 격리 목록(장소 C 5파일·장소 B 2파일·배경 C)은 전부 옛 계측기 자료라 삭제됐다.
QUARANTINED_FILES: Dict[str, str] = {}


# ── 옛 계측기(차동 ADC, ~2026-09-05) 등록 — 참고용. `classify_file` 은 이 이름을 받지 않는다 ──
# 자료는 전부 삭제됐다. 여기 남기는 이유는 둘이다: (1) 설계 문서·인수인계의 파일명을 읽을 때
# 무엇이었는지 찾을 수 있게, (2) 같은 이름의 파일이 다시 나타나면 옛 자료 혼입을 의심할 수 있게.
# 파일별 상세(녹화 조건·위상 복원 값·격리 사유)는 git 이력의 이 파일(커밋 88ec645 이전)에 있다.
LEGACY_DEVICE_FILES_OLD_ADC: Dict[str, str] = {
    "air_conditioner": "air_conditioner", "beam_projector": "beam_projector", "beam_projector_2": "beam_projector",
    "electiric_kettle": "electiric_kettle", "fan_1": "fan", "fan_2": "fan", "fan_3": "fan",
    "hair_dryer_1": "hair_dryer", "hair_dryer_2": "hair_dryer", "hair_dryer_3": "hair_dryer",
    "hotplate_1": "hotplate", "hotplate_2": "hotplate", "laptop_charger_1": "laptop_charger",
    "laptop_charger_2": "laptop_charger", "minipc_1": "minipc", "minipc_2": "minipc", "minipc_3": "minipc",
    "oven": "oven", "oven_2": "oven",
    "beam_projector_3_fixed": "beam_projector", "electric_kettle_2_fixed": "electiric_kettle",
    "hotplate_3_fixed": "hotplate", "laptop_charger_3_fixed": "laptop_charger",
    "laptop_charger_4_fixed": "laptop_charger", "oven_3_fixed": "oven",
    "laptop_charger_5C": "laptop_charger", "hair_dryer_4C": "hair_dryer", "minipc_4C": "minipc",
    "electric_kettle_4C": "electiric_kettle", "beam_projector_4C": "beam_projector",
    "electric_kettle_3_new": "electiric_kettle", "hotplate_4_new": "hotplate",
}
LEGACY_NOISE_FILES_OLD_ADC: Dict[str, float] = {
    "noise_noselfpower": 1.4, "noise_selfpower": 2.37, "noise_noselfpower_C": 1.4,
}
#: 옛 복합 녹화. `real_events.json`(삭제)의 키였다. test.csv 는 봉인 파일이었다 (`evaluation.sealing`).
LEGACY_COMPOSITE_FILES_OLD_ADC: List[str] = [
    "test", "test.2", "test3", "test_4", "test_5", "test_6", "test_7", "test_8", "test_9", "test_10",
    "test_11", "test_12", "test_13", "test_14", "test_15", "test_16", "test_17", "test_18", "tesr_19", "test_20",
]
#: 옛 장소 표. 지문(vrms/vh3/vh9/vh15)이 옛 계측기 전압 채널 인공물을 담고 있어 새 자료에는 못 쓴다.
LEGACY_SITE_OF_STEM_OLD_ADC: Dict[str, str] = {
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


def is_legacy_stem(stem: str) -> bool:
    """옛 계측기 시절의 파일명인가 (새 등록부에 **없는** 이름만). 같은 이름을 새로 등록했으면 False."""
    s = _normalize_stem(stem)
    if s in DEVICE_FILES or s in NOISE_FILES:
        return False
    return (s in LEGACY_DEVICE_FILES_OLD_ADC or s in LEGACY_NOISE_FILES_OLD_ADC
            or s in LEGACY_COMPOSITE_FILES_OLD_ADC)


# ── 장소 ─────────────────────────────────────────────────────────────────────
# **글자로만 부른다** (사용자 지시 2026-09-06). 시간대 이름("저녁"/"심야")은 쓰지 않는다 —
# 같은 자리도 세션마다 전압이 움직여서 곧 틀린 말이 된다. 옛 계측기 시대의 A/B/C 와 겹치지
# 않게 D/E 를 새로 붙였다 (`LEGACY_SITE_OF_STEM_OLD_ADC`).
#
# ⚠ **2026-09-06 정정 (13.11, 사용자 확인).** 한동안 "새 자료는 전부 한 자리이고 축은 시간대" 로
#   적어 뒀는데 틀렸다. **두 자리다.** 가르는 것은 전압 고조파이고 vh7 이 가장 깨끗하다:
#
#       자리  vrms          vh3          vh7           Z
#       D     210.7~219.4V  0.59~0.85%   0.89~1.28%    1.15Ω
#       E     227.5~231.0V  2.80~3.25%   0.08~0.19%    0.42Ω
#
#   vh3 4배 · vh7 7배 차이고 사이가 비어 있다. 세션이 셋(D 둘·E 하나)인데도 갈림이 유지된다.
#
# ⚠ **기기와 장소가 엉켜 있다 — 그러나 풀리고 있다.** 세션 D2 로 프로젝터가, E2 로 미니PC 가
#   양쪽에 생겨 **양쪽에 있는 기기가 셋(충전기·프로젝터·미니PC)** 이 됐다. 남은 엉킴은
#   포트·드라이기·에어컨·선풍기(E 만)와 핫플·오븐(D 만)이다 (설계 13.14~13.18).
SITE_OF_STEM: Dict[str, str] = {
    # D
    **{s: "D" for s in ("hotplate_1", "hotplate_2", "laptop_charger_1", "oven_1", "oven_2",
                        "minipc_1", "beam_projector_3",
                        "noise_noselfpower", "noise_selfpower_1", "noise_selfpower_2")},
    **{s: "D" for s in ("test_1", "test_3", "test_4", "test_5")},
    # E
    **{s: "E" for s in ("hair_dryer_1", "electric_kettle_1", "air_conditioner_1",
                        "beam_projector_1", "beam_projector_2", "fan_1", "laptop_charger_2",
                        "minipc_2", "minipc_3")},
    "test_2": "E",
}

#: 자리별 **대푯값**. 세션마다 움직이므로 정본은 `SITE_SESSIONS` 다.
#: `grid_simulator.MEASURED_SITE_Z_OHM` 과 짝이다.
SITE_PROFILE: Dict[str, dict] = {
    "D": {"v_rms": 216.0, "vh3_pct": 0.72, "vh7_pct": 1.05, "z_ohm": 1.15},
    "E": {"v_rms": 229.5, "vh3_pct": 3.01, "vh7_pct": 0.13, "z_ohm": 0.42},
}

#: 세션 = (자리, 녹화 묶음). **전압·임피던스는 자리가 아니라 세션의 성질이다.**
#: 값은 원본 CSV 에서 직접 잰 중앙값 (vh 는 /V1 %).
SITE_SESSIONS: Dict[str, dict] = {
    "D1": {"site": "D", "date": "2026-09-05", "v_rms": 216.5, "vh3_pct": 0.70, "vh5_pct": 1.70,
           "vh7_pct": 0.99, "z_ohm": 1.15,
           "stems": ("laptop_charger_1", "oven_1", "minipc_1", "hotplate_1", "test_1",
                     "noise_noselfpower", "noise_selfpower_1", "noise_selfpower_2")},
    "E1": {"site": "E", "date": "2026-09-06", "v_rms": 229.0, "vh3_pct": 3.01, "vh5_pct": 1.84,
           "vh7_pct": 0.13, "z_ohm": 0.42,
           "stems": ("electric_kettle_1", "beam_projector_1", "beam_projector_2", "fan_1",
                     "hair_dryer_1", "air_conditioner_1", "laptop_charger_2", "test_2")},
    # 2026-09-06 18:55~20:20. 같은 자리(D)인데 vh5 가 1.70 -> 1.43% 로 내려가고 vh15 가
    # 0.42 -> 0.62% 로 올라갔다. vh7 0.94~1.28% 는 그대로라 자리 판별은 흔들리지 않는다.
    "D2": {"site": "D", "date": "2026-09-06", "v_rms": 215.8, "vh3_pct": 0.73, "vh5_pct": 1.43,
           "vh7_pct": 1.13, "z_ohm": 1.19,   # test_3 1.35±0.42(9계단) / test_5 1.04±0.15(12계단)
           "stems": ("beam_projector_3", "hotplate_2", "oven_2", "test_3", "test_4", "test_5")},
    # 2026-09-06 21:16~21:38. E 인데 E1 보다 vh3 이 3.0 -> 4.0% 로 더 크고 vh7 도 0.13 -> 0.29% 다.
    # 그래도 D(0.86~1.28%)와는 3배 이상 떨어져 있어 자리 판별은 흔들리지 않는다.
    # Z 는 아직 안 쟀다 — 이 세션에는 1kW 급 계단이 없다 (미니PC 뿐).
    "E2": {"site": "E", "date": "2026-09-06", "v_rms": 230.7, "vh3_pct": 4.01, "vh5_pct": 1.26,
           "vh7_pct": 0.30, "z_ohm": None,
           "stems": ("minipc_2", "minipc_3")},
}


def site_of(stem: str) -> str:
    """stem -> 'D' / 'E' (자리. **글자로만 부른다** — 위 [장소] 참조). 모르면 ''.

    옛 계측기 자료의 'A'/'B'/'C' 는 `LEGACY_SITE_OF_STEM_OLD_ADC` 에 따로 있다 — 섞지 않는다.
    """
    return SITE_OF_STEM.get(_normalize_stem(stem), "")


# ── 전류 위상 복원 [°/차수] ───────────────────────────────────────────────────
# 펌웨어의 위상 교정은 "h차 빈을 −h×delay 만큼 회전" 하는 시간지연 모델이라, 교정값이
# 틀리면 모든 차수가 h 에 비례해 돈다. 그 파일을 읽을 때 `ihdeg_h += fix × h` 로 되돌린다
# (`raw_csv.read_raw_csv`, `pipeline.process_file` 이 적용).
# 2026-09-06: 비어 있다. 옛 항목(`laptop_charger_5C: 9.5`, 부하 없이 눌린 USER 버튼 교정, 12.184.3/12.185.6)
# 은 옛 계측기 파일의 것이라 뺐다. 새 자료의 순저항 `ihdeg1` 은 −0.05~+0.06° (규칙 74 통과).
PHASE_FIX_DEG_PER_ORDER: Dict[str, float] = {}


def phase_fix_of(stem: str) -> float:
    """stem -> 되돌릴 위상 [°/차수]. 없으면 0."""
    return PHASE_FIX_DEG_PER_ORDER.get(_normalize_stem(stem), 0.0)


# ── 원시 파형 스냅샷 ───────────────────────────────────────────────────────────
# 옛 계측기 원시(장소 C, 2026-09-04~05, `raw_*.csv` 18개 + 순저항 2개)는 전부 삭제됐다. 그 목록과
# LOW/HIGH 혼합 표시·조합 짝은 git 이력에 있다. 옛 원시 포맷(`high,v_r1,low,v_r2`, LSB 402.8µV,
# 차동)은 `synthesis.fit_raw.load_raw` 가 읽던 것이고, 새 포맷은 그 함수가 **거부**한다.
#
# 새 계측기 원시(단일 입력 포맷 `low, v, high, bias` + 수신기 파생 `i_a, v_v, range`, LSB 201.4µV)로
# 만든 회로 모델은 `circuit_model/circ12_<dev>.pkl` 이고, 적합에 쓴 파일명은 아래다 (pkl 의 `files`).
# 파일 자체는 저장소 밖에 있다 — 필요하면 사용자에게 요청. 적합 코드는 `circuit_model/fit12.py`.
RAW12_FIT_FILES: Dict[str, List[str]] = {
    "laptop_charger": ["1788611281257_raw_laptop_charger_2.csv", "1788611842569_raw_laptop_charger_3.csv",
                       "1788611842569_raw_laptop_charger_4.csv"],                     # 65W ×3, 켠 직후 -> 10분
    "minipc": ["1788614740094_raw_minipc_1.csv", "1788614740094_raw_minipc_3.csv", "1788614740094_raw_minipc_2.csv",
               "1788614740094_raw_minipc_4.csv", "1788614740093_raw_minipc_5.csv"],   # 10~27W
    "beam_projector": ["1788622997330_raw_20260906_004123.csv", "1788622997329_raw_20260906_004129.csv"],  # 47W ×2
}
#: 새 계측기 원시 스냅샷이 `data/` 에 있을 때 (2026-09-06 02:43~02:45, 충전기 65W ×4, 40주기, 229V, LOW).
#: `classify_file` 은 `raw_` 접두를 RAW 역할로 돌려 2Hz 파이프라인이 건너뛴다. 읽는 쪽: `circuit_model/fit12.py`,
#: `fcm.source_from_raw12`. 열: t_s,cyc,seq,n,low,v,high,bias,range,i_a,v_v,i_low_a,i_high_a,fs_hz,off_*,gap_before.
RAW12_SNAPSHOT_FILES: Dict[str, List[str]] = {
    "laptop_charger": ["raw_laptop_charger_5", "raw_laptop_charger_6", "raw_laptop_charger_7", "raw_laptop_charger_8"],
}
RAW_SNAPSHOT_PATTERN = re.compile(r"^raw_", re.IGNORECASE)
RAW_SNAPSHOT_FILES: Dict[str, List[str]] = {}   #: 옛 원시 등록부 — 비어 있다 (위 설명)
RAW_RANGE_MIXED = set()


def raw_snapshots_of(device: str) -> List[str]:
    """기기 -> 원시 스냅샷 stem 목록. 없으면 빈 목록. (옛 원시는 삭제됐다 — `RAW12_FIT_FILES` 참조)"""
    return list(RAW_SNAPSHOT_FILES.get(device, []))


RAW_RESISTIVE_FILES: List[str] = []            #: 옛 순저항 원시(`raw_elcetric_kettle_1/2`) — 삭제
RAW_COMBO_FILES: Dict[str, List[str]] = {}     #: 옛 조합 원시 6개 — 삭제
RAW_COMBO_SOLO: Dict[str, Dict[str, str]] = {}
RAW_COMBO_RANGE_MIXED = set()


# ── 복합 녹화의 사용자 제공 타임라인 (옛 계측기 `tesr_19.csv`, 삭제) ─────────────
# 형식의 본보기로만 남긴다: (seq, t_s, 기기, on/off, 비고). seq 는 2Hz 프레임 번호, t_s=(seq−60)/2.
# 새 복합 녹화 `test_1` 의 스위치 로그는 아직 없다 — READ_ME_FIRST.md "사용자에게 필요한 것".
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
    """옛 `tesr_19.csv` 의 (seq, t_s, 기기, on/off, 비고). 파일은 삭제됐다 — 형식 본보기."""
    return list(TEST19_TIMELINE)


# ── LOW 레인지 위상 교정값의 규약 — **옛 보드(차동 ADC) 전용, 지금은 전부 no-op** ────────────
# 옛 펌웨어는 레인지별 교정값 delay 로 모든 차수를 −h×delay 회전했고, 기본값 LOW 0.44° / HIGH 2.62° 는
# 펌웨어 작성자가 각 경로에 순저항을 물려 잰 값이었다 (12.184.16, 규칙 76). 2026-09-04 저녁 잠깐 2.62 로
# 플래시됐다가 되돌린 이력 때문에 `host_time` 으로 판정하는 코드가 있었다.
# 2026-09-06 계측기 교체 뒤에는 그 보드도, 그 보드로 찍은 파일도 없다. 새 보드의 교정 상수는 이 저장소에
# 기록돼 있지 않다 (펌웨어 소스 확인 필요). 새 자료의 저항 부하 ihdeg1 이 LOW/HIGH 양쪽에서 0.06° 안이라
# 결과적으로는 맞다. `LOW_CAL_FLASHED_ACTIVE=False` 이고 `LOW_CAL_OVERRIDE` 가 비어 회전은 항상 0 이다.
LOW_CAL_DEG_CANONICAL = 0.44      #: (옛 보드) 정본 = 펌웨어 원래 기본값 (순저항으로 잰 값)
LOW_CAL_DEG_LEGACY = 0.44         #: (옛 보드) 2026-09-04 저녁 이전 모든 녹화의 값 (= 정본)
LOW_CAL_DEG_FLASHED = 2.62        #: (옛 보드) 2026-09-04 저녁에 잘못 올린 값
LOW_CAL_FIXED_AT = "2026-09-04 18:30:00"
LOW_CAL_FLASHED_ACTIVE = False    #: 옛 보드를 0.44 로 되돌렸고, 그 뒤 보드 자체가 교체됐다
#: stem -> 녹화 당시 LOW 교정값 [°]. 자동 판정(host_time)을 덮는다. 옛 항목(beam_projector_4C)은 뺐다.
LOW_CAL_OVERRIDE: Dict[str, float] = {}


def low_cal_of(stem: str, host_time_first: Optional[str] = None) -> float:
    """그 파일이 녹화될 때 적용돼 있던 LOW 교정값 [°]. host_time 이 없으면 정본(0.44)으로 본다."""
    s = _normalize_stem(stem)
    if s in LOW_CAL_OVERRIDE:
        return LOW_CAL_OVERRIDE[s]
    if not LOW_CAL_FLASHED_ACTIVE or host_time_first is None:
        return LOW_CAL_DEG_LEGACY
    return LOW_CAL_DEG_LEGACY if str(host_time_first) < LOW_CAL_FIXED_AT else LOW_CAL_DEG_FLASHED


def low_cal_shift_deg(stem: str, host_time_first: Optional[str] = None) -> float:
    """`range==0` 사이클에 걸 회전 [°/차수] = −(정본 − 녹화 당시). 지금은 항상 0 (위 설명)."""
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


#: 수신기가 교정본 CSV 에 붙이는 꼬리. stem 에서 뗀다 (`hotplate_1.cal2.csv` -> `hotplate_1`).
CALIBRATION_SUFFIXES = (".cal2",)


def resolve_csv(stem: str, data_dir: Union[str, Path] = "data") -> Optional[Path]:
    """stem -> 원본 CSV 경로. `data/<stem>.csv` 와 `data/<stem>.cal2.csv` 둘 다 찾는다. 없으면 None."""
    d = Path(data_dir)
    s = _normalize_stem(stem)
    for name in (f"{s}.csv", *(f"{s}{suf}.csv" for suf in CALIBRATION_SUFFIXES)):
        if (d / name).exists():
            return d / name
    return None


def _normalize_stem(file_path: Union[str, Path]) -> str:
    """'data/test.2.csv' -> 'test.2', 'data/hotplate_1.cal2.csv' -> 'hotplate_1'. 확장자·교정 꼬리만 뗀다."""
    name = Path(file_path).name
    stem = name[:-4] if name.lower().endswith(".csv") else Path(file_path).stem
    for suf in CALIBRATION_SUFFIXES:
        if stem.lower().endswith(suf):
            stem = stem[:-len(suf)]
    return stem


def classify_file(file_path: Union[str, Path]) -> FileClassification:
    """원본 CSV 1개의 역할을 판정한다. 판정 순서 자체가 안전장치다.

    복합 부하 패턴을 가전 등록부보다 **먼저** 확인한다. 그래야 누군가
    test_fan.csv 같은 이름을 만들어도 선풍기로 오인되지 않는다.
    옛 계측기 파일명은 등록부에 **없으므로** UNKNOWN 이다 — 새로 잰 파일이면 등록해서 쓴다.
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

    # 1b) 원시 파형 스냅샷 — 2Hz 자료가 아니다. 회로 모델 쪽(fit12)만 읽는다
    if RAW_SNAPSHOT_PATTERN.search(stem):
        return FileClassification(
            stem=stem, role=FileRole.RAW, appliance_type=None, load_class=LoadClass.PASSIVE,
            noise_floor_w=NOISE_FLOOR_EXTERNAL_W, periodic_duty=False, low_load=False,
            reason="원시 파형 스냅샷 (raw_*) - 2Hz 전처리 대상이 아님. circuit_model/fit12.py 가 읽는다",
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
            appliance_type=NOISE_TYPE_OF_STEM.get(stem, stem),
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

    # 5) 옛 계측기 파일명 - 자료는 폐기됐다. 새로 잰 것이면 등록해서 쓴다
    if is_legacy_stem(stem):
        return FileClassification(
            stem=stem,
            role=FileRole.UNKNOWN,
            appliance_type=None,
            load_class=LoadClass.PASSIVE,
            noise_floor_w=NOISE_FLOOR_EXTERNAL_W,
            periodic_duty=False,
            low_load=False,
            reason=("옛 계측기(차동 ADC, ~2026-09-05) 파일명 - 그 자료는 폐기됐다 (READ_ME_FIRST.md). "
                    "새 계측기로 다시 잰 파일이면 file_registry.DEVICE_FILES 에 새로 등록하세요"),
        )

    # 6) 미등록 - 추측하지 않는다
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
            f"  - {classification.reason}\n"
            f"  - 단일 가전이라면 src/preprocessing/file_registry.py 의 DEVICE_FILES 에 등록하세요.\n"
            f"  - 여러 가전이 동시에 돌아간 검증용 녹화라면 파일명을 test_* 로 시작하게 바꾸세요.\n"
            f"  - 무부하 노이즈라면 NOISE_FILES 에 등록하세요."
        )
    return classification


# ── 가전 종류(appliance_type) 단위 조회 ──────────────────────────────────────
# 파일은 여러 개여도 가전 종류는 하나다(fan_1/2/3 -> fan). 합성 단계에서는
# 파일이 아니라 가전 종류로 다루므로 종류 단위 속성이 따로 필요하다.

def _build_appliance_index() -> Dict[str, DeviceSpec]:
    """카탈로그 9종 + (혹시) 카탈로그 밖 종류로 등록된 파일. 파일이 없는 종류도 카탈로그에 남는다."""
    index: Dict[str, DeviceSpec] = dict(APPLIANCE_CATALOG)
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
    """카탈로그의 모든 가전 종류 (파일이 아직 없는 종류 포함)."""
    return sorted(APPLIANCE_SPECS.keys())


def get_registered_appliance_types() -> List[str]:
    """지금 등록된 파일이 있는 가전 종류만."""
    return sorted({s.appliance_type for s in DEVICE_FILES.values()})


# ── 원시 파형의 위상 교정 — **옛 계측기(차동 ADC) 원시 전용** ─────────────────────
# 옛 펌웨어의 위상 교정(LOW 0.44° / HIGH 2.62°, 차수당)은 2Hz 블록에만 걸리고 원시 표본에는
# 남아 있었다 (12.185.25: 장소 C 포트 원시에서 ∠I₁−∠V₁ = +2.87°). 그래서 옛 원시를 맞출 때
# 시뮬 전류를 0.313 표본 앞당겼다 (`RAW_SKEW_SAMP_LOW`). 옛 원시는 삭제됐고 `fit_raw.load_raw`
# 는 새 포맷을 거부하므로 이 상수는 더 이상 쓰이지 않는다. 값은 기록으로 남긴다.
#
# 새 계측기 원시(`circuit_model/fit12.py`)는 **스큐 보정을 하지 않는다** — README_v12 가
# "LOW 지연 보정" 을 기각 유지로 적었고, 규약은 `역RC(V) → 시뮬 → RC(i)`, τ=60µs 다.
RAW_PHASE_CAL_DEG_PER_ORDER: Dict[str, float] = {"LOW": 0.44, "HIGH": 2.62}   # (옛 보드)
RAW_SAMPLES_PER_CYCLE = 256
#: (옛 보드) 원시 비교에 쓰던 기본 스큐 [표본]. 음수 = 시뮬 전류를 앞당긴다.
RAW_SKEW_SAMP_LOW = -RAW_PHASE_CAL_DEG_PER_ORDER["LOW"] * RAW_SAMPLES_PER_CYCLE / 360.0
#: 새 계측기 원시의 스큐. 보정 없음.
RAW_SKEW_SAMP_LOW_V12 = 0.0
#: 새 계측기 원시의 계측 RC 시정수 [s] (`circuit_model` 규약, LOW/HIGH 비 실측 + 스캔 최적).
RAW_RC_TAU_V12 = 60e-6


# ── 새 보드의 계측 상수 — **펌웨어 소스로 확인** (13.5, 2026-09-06) ────────────────
# 출처: `NILM_ECE_IF-fix-even-harmonics-offset-bug/Core/{Inc/nilm_dsp.h, Src/nilm_dsp.c}`.
# 세 ADC 입력이 같은 1kΩ·100nF 저역통과를 지난다 (`NILM_RC_FC_HZ 1591.55f`).
#: 하드웨어 안티앨리어싱 RC [s] = 1/(2π·1591.55). 위 `RAW_RC_TAU_V12`(60µs)는 이것의 **근사**다 —
#: 근사인데도 h1~h9 에서 정확한 연산자와 0.4° 안에서 같아 바꾸지 않는다 (13.5.3).
METER_RC_TAU_S = 100.0e-6
#: 펌웨어가 스펙트럼에 곱하는 RC **크기** 역보정 `s_rc_gain[h] = |1 + j·2π·f_h·τ|`. 위상은 안 건드린다.
#: 그래서 2Hz 의 `ih` 는 **참 전류 크기**이고(RC 감쇠가 되돌려져 있다), `ihdeg` 에는 RC 위상이 남아 있다.
#: 원시 `i_a` 는 보정이 전혀 없다 (아날로그 RC 가 크기·위상 모두 들어 있다).
#: 펌웨어 `NILM_CAL_DEFAULT_{LOW,HIGH}_DEG`. 두 레인지가 같은 값이다 — 전환되는 이득단이 없는 보드라
#: 옛 보드의 LOW 0.44 / HIGH 2.62 (2.18° 차)는 실재하지 않았다고 펌웨어 주석이 적고 있다
#: (같은 표본에서 두 경로를 재구성하니 차이가 0.00±0.03°). 기준 부하는 전기포트 6.45A/1468W 에서 +1.377/+1.402°.
METER_PHASE_CAL_DEG_V12: Dict[str, float] = {"LOW": 1.40, "HIGH": 1.40}
#: `ihdeg[h] -= h · METER_PHASE_CAL_DEG_V12[레인지]` 로 들어간다 (nilm_dsp.c 의 `a = -(float)h * cal->delay_rad`).
#: 교정 상수가 바뀌기 전에 모은 CSV 를 새 눈금으로 옮긴 값 (`migrate_cal.py --phase-low 0.96 --phase-high -1.22`).
#: 새 값 − 옛 값 = 1.40 − 0.44 = +0.96 (LOW) / 1.40 − 2.62 = −1.22 (HIGH). `.cal2` 꼬리가 그 결과다.
#: 검산: 이주 뒤 순저항 `ihdeg1` 이 LOW +0.05° / HIGH +0.06° (이주 전에는 +1.01 / −1.16 으로 2.17° 어긋났다).
MIGRATE_CAL_DELTA_DEG: Dict[str, float] = {"LOW": +0.96, "HIGH": -1.22}


def raw_to_2hz_transfer(h_max: int = 15):
    """2Hz 고조파 / 원시 FFT 고조파 = T(h). 여덟 스냅샷(두 세션, 32~65W)에서 4자리까지 같았다 (13.5.2).

    `T(h) = |1 + j·2π·60h·METER_RC_TAU_S| · exp(−j·h·METER_PHASE_CAL_DEG_V12)`
    크기는 펌웨어의 RC 역보정, 위상은 위상 교정 상수다. RC **위상**은 두 경로에 똑같이 들어 있어 비에서 상쇄된다.
    원시로 맞춘 모델을 2Hz 영역에 놓거나 그 반대로 옮길 때 쓴다.
    """
    import numpy as np
    h = np.arange(1, h_max + 1)
    return (np.abs(1 + 1j * 2 * np.pi * 60.0 * h * METER_RC_TAU_S)
            * np.exp(-1j * np.radians(h * METER_PHASE_CAL_DEG_V12["LOW"])))


#: 원시 스냅샷의 `seq` 는 그 순간 돌고 있던 2Hz 녹화의 **국소 사이클 번호**와 같다:
#: `raw.seq == 2Hz.seq * 30 + 2Hz.cycle` (오프셋 0). 2Hz CSV 의 `seq` 는 0.5초 블록(30주기)이고
#: `cycle` 은 블록 안 0..29 다. 원시 여덟 개 전부 전력 무늬 상관 1.0000 으로 확인됐다 (13.5.1).
#: **npz 의 `seq` 는 블록 번호다** — 원시의 `seq`(사이클)와 직접 비교하면 안 된다.
RAW_SEQ_IS_2HZ_LOCAL_CYCLE = True
RAW_2HZ_CYCLES_PER_BLOCK = 30

#: `data/` 에 있는 원시 스냅샷 11개와 그것이 찍힌 2Hz 녹화 (13.7.1). 세션은 Vrms 로 갈린다.
#: 2026-09-06 D2 로 D 프로젝터 원시(raw_beam_projector_3·4)가 들어왔다. 미니PC 는 아직 D 만이라
#: E 미니PC 원시가 남은 축퇴 자료다 (설계 13.6.2 · 13.15.7).
RAW12_IN_RECORDING = {
    "raw_laptop_charger_1": ("laptop_charger_1", "evening", 215.6),
    "raw_laptop_charger_2": ("laptop_charger_1", "evening", 214.5),
    "raw_laptop_charger_3": ("laptop_charger_1", "evening", 215.4),
    "raw_laptop_charger_4": ("laptop_charger_1", "evening", 215.7),
    "raw_laptop_charger_5": ("laptop_charger_2", "night", 229.3),
    "raw_laptop_charger_6": ("laptop_charger_2", "night", 229.5),
    "raw_laptop_charger_7": ("laptop_charger_2", "night", 229.4),
    "raw_laptop_charger_8": ("laptop_charger_2", "night", 229.5),
    "raw_minipc_1": ("minipc_1", "evening", 217.1),
    "raw_minipc_2": ("minipc_1", "evening", 216.9),
    "raw_minipc_3": ("minipc_1", "evening", 216.4),
    "raw_minipc_4": ("minipc_1", "evening", 216.8),
    "raw_minipc_5": ("minipc_1", "evening", 216.7),
    "raw_beam_projector_1": ("beam_projector_1", "night", 231.1),
    "raw_beam_projector_2": ("beam_projector_1", "night", 231.0),
}
