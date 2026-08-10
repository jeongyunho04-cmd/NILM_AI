# NILM AI Raw Data Preprocessing Project

본 프로젝트는 STM32/ESP 수신기로 수집된 NILM(전기하중 모니터링) Raw 센서 데이터(CSV)를 인공지능(AI) 모델 학습에 적합한 시계열 데이터셋으로 전처리하는 파이썬 전용 패키지입니다.

---

## 주요 기능

1. **시퀀스 정렬 및 누락/중복 처리 (`SequenceAligner`)**:
   - 패킷 시퀀스 번호(`seq`)와 주기 인덱스(`cycle`, 0~29)를 기반으로 시간순 정렬.
   - 중복 프레임 제거 및 연속 글로벌 cycle 인덱스 (`global_cycle`) 생성.
   - 시퀀스 번호 비연속(패킷 누락/Gap) 검출 및 요약 보고서 출력.

2. **전자기기 ON/OFF 및 과도상태(Transient) 라벨링 (`ApplianceStateLabeler`)**:
   - 능동 전력(`p_w`) 및 RMS 전류(`irms`) 기반 이중 문턱값(Hysteresis Thresholding) 라벨링.
   - 상태 코드 정의:
     - `0` (`STEADY_OFF`): 기기 끄짐 / 대기 전력 상태
     - `1` (`ON_TRANSIENT`): 기기 켜짐 시점의 전류/전력 급증 과도상태 (기본 30 cycle / 약 0.5초)
     - `2` (`STEADY_ON`): 정상 동작 상태
     - `3` (`OFF_TRANSIENT`): 기기 꺼짐 시점의 감쇄 과도상태
   - 전력 변화율($dP/dt$)을 감지하여 과도상태 범위를 동적으로 미세 조율 가능.

3. **특징 공학 (Feature Engineering)**:
   - 전력/전류 변동량 ($\Delta P, \Delta I_{rms}$) 산출.
   - 이동 평균(Moving Average) 기반 노이즈 평활화 (`p_w_smooth`, `irms_smooth`).
   - 기본파(`ih1`) 대비 차수별 전류 고조파 비율 (`ih2_ratio` ~ `ih15_ratio`) 정규화.

---

## 프로젝트 구조

```
preprocessing/
├── config.yaml                # 문턱값, 정렬 기준, 컬럼명 설정 파일
├── requirements.txt           # 의존성 패키지 목록
├── run_preprocessing.py       # CLI 메인 실행 스크립트
├── README.md                  # 사용 설명서
├── nilm_preprocessing/        # 핵심 모듈 패키지
│   ├── __init__.py
│   ├── data_loader.py         # CSV 파일 로드 및 타입 처리
│   ├── sequence_aligner.py    # 시퀀스 정렬, 중복제거, Gap 검출
│   ├── labeler.py             # ON/OFF 및 Transient 라벨링 알고리즘
│   ├── feature_engineering.py # 이동평균, dP/dI, 고조파 정규화
│   └── pipeline.py            # 전과정 파이프라인
└── tests/                     # 테스트 스위트
    ├── test_sequence_aligner.py
    ├── test_labeler.py
    └── test_pipeline.py
```

---

## 사용법

### 1. 환경 설정 및 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 기본 실행 (CLI)
`D:\stm project\NILM_AI\data\` 디렉터리의 전체 CSV를 전처리하여 `./output/processed_nilm_data.csv`에 저장합니다.
```bash
python run_preprocessing.py
```

### 3. 사용자 지정 경로 및 파라미터 실행
```bash
python run_preprocessing.py \
    --input "../data/laptop_charger_tran.csv" \
    --output "./output/laptop_processed.csv" \
    --on-power 5.0 \
    --off-power 2.0 \
    --transient-window 30
```

### 4. 파이썬 코드 내 모듈 직접 사용 예시
```python
from nilm_preprocessing.pipeline import PreprocessingPipeline

# 파이프라인 초기화 및 실행
pipeline = PreprocessingPipeline("config.yaml")
df, summary = pipeline.run("../data/*.csv", output_path="./output/processed.csv")

print("라벨링 분포:", summary["state_distribution"])
```

---

## 단위 테스트 실행
```bash
pytest tests/
```
