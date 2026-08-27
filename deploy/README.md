# NILM 실시간 추론 — 배포 묶음

학습 워크스페이스에서 **운영에 필요한 것만** 떼어 낸 것이다. 이 폴더만 복사하면
다른 PC에서 돈다. 학습·합성·평가 코드는 들어 있지 않다.

```
deploy/
├─ README.md                 이 문서
├─ run_predict.py            CLI (수신기 CSV -> 기기별 전력)
├─ requirements.txt          numpy / torch (그게 전부다)
├─ models/
│   └─ adapt_ovh.pt          운영점 체크포인트 (2.4MB)
└─ nilm_runtime/
    ├─ __init__.py           `from nilm_runtime import NILMPredictor`
    ├─ predictor.py          ★ UI 가 부르는 API — 링버퍼 + 모델 + 후처리
    ├─ receiver.py           수신기 (보드 -> CSV). 원본 `nilm_receiver.py`
    ├─ inputs.py             33채널 -> 모델 입력(세밀 50ch + 광역 12ch) 변환
    ├─ net.py                모델 구조 (2갈래 CNN, 0.61M 파라미터)
    ├─ postproc.py           물리 전력 상한 후처리
    ├─ file_registry.py      기기 명세 (정격·부하 분류)
    └─ state_definitions.py  기기별 상태 정의
```

---

## 1. 빠른 시작

```bash
pip install -r requirements.txt

# ① 수신기: 보드(ESP-01S)가 TCP 로 접속해 오는 것을 받아 CSV 로 쓴다
#    별도 터미널에서 계속 돌린다. 기본 포트 5000, 파일은 옆의 data/ 안에 쌓인다
python -m nilm_runtime.receiver --csv live.csv

# ② 예측기: CSV -> 기기별 전력
python run_predict.py --csv data/live.csv
```

> 수신기는 **TCP 서버**다 (시리얼이 아니다). 보드의 ESP-01S 가 이 PC 로 접속해
> 온다 — 방화벽에서 5000 포트를 열어야 한다. 프레임 형식은 펌웨어
> `NILM_ECE_IF/Core/Inc/nilm_link.h` 와 1:1 이고, 프로토콜 v4 다.

동작 확인만 하려면 기존 녹화를 재생한다 (보드 없이 된다).

```bash
python run_predict.py --replay ../data/test_8.csv --speed 0
```

---

## 2. 데이터 흐름

```
   계측 보드 (STM32, 60Hz)
        │  프레임 30사이클(0.5초) 단위, 재전송 때문에 순서가 뒤바뀐다
        ▼
   receiver.py (TCP 서버 :5000) ▶  CSV  (한 행 = 한 사이클, 초당 60행)
        │                        t_s, vrms, irms, p_w, phase_deg, ih1..ih15, ihdeg1..15
        ▼
   NILMPredictor.push_row()
        │  CycleRing: t_s 로 자리를 정해 꽂는다 (지연 0)
        │    · 순서 뒤바뀜 보정 — 실측에서 2~3% 행이 역전돼 온다
        │    · 세션 이어붙임 — 보드 리셋으로 t_s 가 0 으로 돌아가면 버퍼를 비운다
        │    · 유실 사이클은 직전 값으로 채운다
        ▼
   창 60초(3,600 사이클) 가 차면
        │
        ├─ inputs.build_inputs()   세밀 갈래 (50ch × 600) = 뒤 10초 @60Hz
        │                          광역 갈래 (12ch × 120) = 60초 @2Hz
        ▼
   net.NILMNet                     기기 9종 × {전력, ON확률, 대기, 상태}
        │                          타깃은 창 끝에서 6초 안쪽 시점
        ▼
   postproc.apply_postproc()       프로젝터 55W 상한 + 초과분을 다른 SMPS 로 재배분
        ▼
   PredictionResult                UI 로
```

**왜 창이 60초인가.** 모델은 한 시점을 판단하려고 그 앞 60초를 본다. 그래서
**시작 후 60초 동안은 결과가 안 나온다**(`predict()` 가 `None`). UI 는 그동안
"창 채우는 중 (`stats()['fill_ratio']`)" 을 보여 주면 된다.

---

## 3. UI 붙이기

### 방법 A — 같은 프로세스에서 직접 호출 (권장)

```python
import csv, threading, queue
from nilm_runtime import NILMPredictor, APPLIANCE_KO

results = queue.Queue()

def worker(csv_path: str):
    pred = NILMPredictor("models/adapt_ovh.pt")      # postproc="on" 이 기본
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        pred.set_header(next(r))
        for i, row in enumerate(r):
            pred.push_row(row)
            if i % 30:                                  # 0.5초마다 추론
                continue
            out = pred.predict()
            if out is not None:
                results.put(out)                        # UI 스레드로 넘긴다

threading.Thread(target=worker, args=("data/live.csv",), daemon=True).start()

# UI 스레드 (Tk/Qt 무엇이든)
def tick():
    while not results.empty():
        r = results.get()
        draw(r.t_s, r.observed_w, r.total_w,
             {APPLIANCE_KO[k]: v for k, v in r.power_w.items()},
             on=r.on())                                  # 켜졌다고 본 기기 목록
    root.after(200, tick)
```

> **`NILMPredictor` 는 스레드 안전하지 않다.** `push_row`/`predict` 는 한
> 스레드에서만 부르고, 결과는 큐로 넘겨라. 위 예시가 그 형태다.

### 방법 B — 별도 프로세스 + JSONL 구독

UI 를 다른 언어(Electron, C#, 웹)로 짤 때 쓴다.

```bash
python run_predict.py --csv data/live.csv --jsonl data/pred.jsonl --quiet
```

`pred.jsonl` 에 한 줄씩 쌓인다. UI 는 그 파일을 tail 하면 된다.

```json
{"t_s": 59.983, "observed_w": 112.38, "total_w": 112.40, "residual_w": -0.02,
 "power_w": {"beam_projector": 55.0, "laptop_charger": 30.1, "minipc": 26.2, ...},
 "gate":    {"beam_projector": 0.99, "laptop_charger": 0.87, "minipc": 0.71, ...}}
```

---

## 4. API

### `NILMPredictor(ckpt_path, device=None, postproc="on", resmatch=0.02, reorder=True)`

| 인자 | 뜻 |
|---|---|
| `ckpt_path` | `models/adapt_ovh.pt` |
| `device` | `"cuda"` / `"cpu"`. 생략하면 있는 쪽 |
| `postproc` | `"off"` / `"on"` / `"sync"` — 아래 6절 |
| `resmatch` | 저항 부하 정합 허용오차. 운영 기본 0.02, 0 이면 끔 — 아래 6절 |
| `reorder` | 순서 뒤바뀜 보정. **끄지 말 것** (2~3% 행이 역전돼 온다) |

| 메서드 | 하는 일 |
|---|---|
| `set_header(header)` | CSV 헤더를 한 번 알려 준다 |
| `push_row(row)` | 한 사이클 넣기. `"new"/"backfill"/"stale"/"seam"` 반환 |
| `ready()` | 창(60초)이 찼는가 |
| `predict()` | 추론 1회 → `PredictionResult` 또는 `None` |
| `stats()` | 링버퍼 통계 (진단 패널용) |

### `PredictionResult`

| 필드 | 뜻 |
|---|---|
| `t_s` | 창 머리 시각 (초, 보드 기준) |
| `observed_w` | 관측 총전력 |
| `total_w` | 예측 합계 (활성 + 대기) |
| `power_w` | 기기별 예측 전력 `{"minipc": 12.3, ...}` |
| `gate` | 기기별 ON 확률 0~1 |
| `standby_w` | 대기전력 합 |
| `residual_w` | `observed_w − total_w`. **UI 에 띄울 값이다** — 이것이 크면 못 가른 부하가 있다는 뜻 |
| `.on(threshold=0.5)` | 켜졌다고 본 기기 목록 |

### `stats()`

```python
{"n_pushed": 38730, "reorder_rate": 0.0248, "max_backfill_s": 0.98,
 "n_seam": 0, "fill_ratio": 1.0, "device": "cuda", "postproc": "on"}
```

`fill_ratio` 가 0.98 미만이면 아직 창이 안 찼거나 데이터가 끊긴 것이다.
`reorder_rate` 가 10% 를 넘으면 통신 상태를 의심할 것.

---

## 5. 성능 — UI 가 무엇을 믿어도 되는가

실측 11파일(총 62분) 기준. **기기마다 신뢰도가 다르다.**

| 기기 | on/off F1 | UI 표시 |
|---|---|---|
| 충전기 | 0.935 | 믿을 만하다 |
| 핫플레이트 | 0.885 | 믿을 만하다 |
| 프로젝터 | 0.850 | 믿을 만하다 |
| 미니PC | 0.821 | 보통 |
| 전기포트 | 0.78~0.92 | 보통 (파일에 따라 갈린다) |
| 드라이기 | 0.84~0.92 | 보통 |
| 오븐 | 0.627 | **주의 — 아래 설명** |

> **오븐 숫자는 낮게 나온다.** 모델은 오븐의 **히터 통전**(전체의 25~43%)을
> 예측하는데 정답 라벨은 사람이 스위치를 켠 **세션 전체**다. 정의가 다르므로
> 0.627 은 실제 성능보다 낮게 찍힌 값이다. 정밀도는 0.72~0.76 이다.
> UI 에서 "오븐 켜짐" 을 보이려면 **히터 펄스를 1~2분 창으로 묶어** 세션으로
> 바꿔 표시하는 편이 사용자 기대에 맞는다.

| 총량 지표 | 값 |
|---|---|
| 총전력 잔차 (절대 평균) | 10.0W |
| 없는 기기에 붙은 전력 | 5.0W |
| 추론 속도 | 120~200회/초 (RTX 2050) — 60Hz 요건의 2배 이상 |

### 알려진 한계 (UI 설계에 반영할 것)

1. **SMPS 3종(프로젝터·충전기·미니PC)은 서로 헷갈린다.** 전이 시점 귀속이
   59건 중 38건 정확하고, 틀린 것은 **전부 이 세 기기 사이의 맞바꿈**이다.
   합계는 맞으므로 셋을 묶어 **"SMPS 합계"** 로 보여 주면 훨씬 정확하다.
2. **에어컨 + 드라이기가 함께 켜지면 드라이기를 놓친다** (학습 데이터에 그 조합이
   0.2% 뿐). 그 구간에서 잔차가 500W 이상 뜬다 — `residual_w` 로 감지 가능하다.
3. **234V 회선은 검증이 얕다.** 검증 파일 대부분이 209~223V 에서 측정됐다.
4. 시작 후 **60초는 결과가 없다.**

> **UI 권장 표시.** 기기별 막대 + `residual_w` 게이지를 함께 두면, 모델이 못
> 가른 부하가 있을 때 사용자가 바로 안다. 잔차가 100W 를 넘으면 경고를 띄우는
> 정도가 적당하다 (정상 구간은 10W 안쪽이다).

## 6. 후처리 (`postproc`, `resmatch`)

**① 물리 전력 상한 (`postproc`).** 프로젝터는 실제로 48.5~49.3W 만 먹는데
모델은 복합에서 창의 74%를 그보다 크게(중앙 73.5W, 최대 137W) 예측한다.
그 초과분은 실은 다른 SMPS 의 몫이다. 55W 를 넘는 만큼을 잘라 넘긴다.

| 값 | 효과 |
|---|---|
| `"off"` | 모델 출력 그대로 |
| `"on"` (기본) | 전이 귀속 27→44/59, 잔차 유지 |
| `"sync"` | 위 + 넘겨받은 기기의 ON 게이트도 켠다 (미니PC F1 +0.07) |

**② 저항 부하 정합 (`resmatch`).** 저항은 니크롬선이라 `P = V²/R` 이고
**R 이 기기 고유값**이다 (녹화 간 재현성 0.1~1.3%):

```
포트 35.8Ω   오븐 40.6Ω   드라이기 54.3Ω   핫플 101.8Ω
```

고조파로는 0.596%p 밖에 안 갈리는 세 기기가 저항값으로는 13~180% 갈린다.
관측 전력·전압에서 컨덕턴스를 역산해 저항 조합을 **맞바꾼다** (기기 수는 안 바꾼다 —
바꾸게 두면 없는 기기를 발명한다).

| 값 | 효과 |
|---|---|
| `0` | 끔 |
| `0.02` (기본) | 없는 기기 전력 7.6 → 5.0W, 저항 전용 파일 F1 0.76/0.79 → 0.78/0.84 |

## 7. 모델 교체

체크포인트만 바꾸면 된다. 기기 목록·채널 수·창 길이는 파일 안에 들어 있어
`NILMPredictor` 가 알아서 맞춘다.

```bash
cp <학습PC>/results/<새모델>.pt models/
python run_predict.py --ckpt models/<새모델>.pt --replay ../data/test_11.csv --speed 0
```

**호환 조건**: 창 3,600 사이클 / 타깃 끝-6초 / 세밀 채널 ≤ 58. 학습 저장소에서
`zero_even_harmonics=True` 로 학습한 모델이어야 입력 규약이 맞는다.

---

## 8. 학습 저장소와의 관계

이 폴더의 `.py` 는 학습 저장소의 **사본**이다. 모델 구조나 입력 규약을 고치면
양쪽이 어긋난다.

| 배포 | 원본 |
|---|---|
| `nilm_runtime/inputs.py` | `src/model/inputs.py` |
| `nilm_runtime/net.py` | `src/model/net.py` |
| `nilm_runtime/postproc.py` | `src/model/postproc.py` |
| `nilm_runtime/receiver.py` | `nilm_receiver.py` |
| `nilm_runtime/predictor.py` | `src/run_live.py` 에서 런타임 부분만 발췌 |

설계 근거와 측정 기록은 학습 저장소의 `MODEL_TRAINING_DESIGN.md` 에 있다
(후처리는 12.100~12.104, 운영점 교체는 12.103, 봉인 평가는 12.105).
