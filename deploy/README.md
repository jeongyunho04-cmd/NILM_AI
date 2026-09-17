# NILM 실시간 추론 — 배포 묶음

학습 워크스페이스에서 **운영에 필요한 것만** 떼어 낸 것이다. 이 폴더만 복사하면
다른 PC에서 돈다. 학습·합성·평가 코드는 들어 있지 않다.

```
deploy/
├─ README.md                 이 문서
├─ run_predict.py            CLI (수신기 CSV -> 기기별 전력)
├─ sync_runtime.py           ★ 학습 저장소와 **다시 맞춘다** (+ 표를 굽는다)
├─ gate_deploy.py            ★ 관문 — 묶음과 연구 갈래가 같은 답을 내나 (6/6)
├─ requirements.txt          numpy / torch (그게 전부다)
├─ models/
│   ├─ cnn_comb_s0.pt        **운영점** (3.4MB, 조합 머리 + 추론 꺾기. 2026-09-17)
│   ├─ adapt_zi_s0.pt        옛 운영점 (2026-09-02) — ⚠ **지금 런타임에서 안 돈다**
│   └─ adapt_ovh.pt          더 옛것 (2026-08-27) — ⚠ 세밀 채널 50 이라 규약이 다르다
├─ nilm_runtime/             ★ **묶음 고유 코드만** 남는다
│   ├─ predictor.py          UI 가 부르는 API — 링버퍼 + 모델 + Ĝ + 후처리
│   ├─ receiver.py           수신기 (보드 -> CSV). 원본 `nilm_receiver.py`
│   ├─ signatures.npz        와트당 고조파·무효 지문 (잔차 흡수가 쓴다)
│   └─ runtime_tables.npz    ★ 상태별 지문 (Ĝ 추정기의 에어컨 기둥)
├─ src/                      ★ **연구 코드를 그대로 비춘 것** (손대지 마라)
│   ├─ model/                inputs · net · build · postproc · losses
│   │                        **gbudget · physdecomp · fcmtab** (Ĝ 추정기)
│   ├─ labeling/             state_definitions
│   └─ synthesis/            fcm · circuit_sim
└─ circuit_model/            circ12_*.pkl + fcm12 (FCM 회로 모델 3종)
```

> ⚠⚠ **`src/` 와 `circuit_model/` 은 고치지 마라.** 학습 저장소와 **바이트가 같아야**
> 하고 `gate_deploy.py [1]` 이 그것을 지킨다. 고칠 일이 있으면 저장소 쪽을 고치고
> `python sync_runtime.py` 를 돌려라.


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
> `NILM_ECE_IF/Core/Inc/nilm_link.h` 와 1:1 이고, 프로토콜 v5 다 (2026-09-04: 공통부에 vh_cdeg[15] 추가, 116B / 프레임 3243B).

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
   postproc.snap_power()           프로젝터를 참값 46.9W 로 스냅 (12.129)
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
    pred = NILMPredictor("models/adapt_zi_s0.pt")      # postproc="on" 이 기본
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

### `NILMPredictor(ckpt_path, device=None, postproc="on", resmatch=0.02, snap=0.0, squelch=0.1, absorb=1.0, reorder=True)`

| 인자 | 뜻 |
|---|---|
| `ckpt_path` | `models/adapt_zi_s0.pt` |
| `device` | `"cuda"` / `"cpu"`. 생략하면 있는 쪽 |
| `postproc` | `"off"` / `"on"` / `"sync"` — 아래 6절 |
| `resmatch` | 저항 부하 정합 허용오차. 운영 기본 0.02, 0 이면 끔 — 아래 6절 |
| `rm_snap` | 저항 정합이 조합을 확인한 창에서 전력도 `V²/R` 로 맞춘다. 운영 기본 `True` (12.117) |
| `snap` | 프로젝터를 격리 참값 W 로 맞추고 차액을 다른 SMPS 로 넘긴다. **2026-09-02 기본 `0`(끔)** — 복소 Z·I 가 프로젝터를 −0.37W 로 닫아서 못 박을 것이 없고 `absorb` 와 충돌한다 (12.149.2) |
| `squelch` | 게이트가 이 값 아래인 기기의 전력을 `0` 으로 (12.149). `P = σ(on)·p_raw` 라 **꺼졌다고 보고한 기기가 와트를 낸다** — 에어컨 σ 0.008 × 592W = 4.9W. 운영 기본 `0.1` |
| `absorb` | 그렇게 빠진 와트를 고조파가 닮은 SMPS 로 되돌린다 (12.104). **`squelch` 와 짝이다** — 하나만 켜면 반쪽이다. 운영 기본 `1.0` |
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

**①-b 프로젝터 참값 스냅 (`snap`, 기본 46.9W).** 상한이 55W 로 자른 뒤,
프로젝터를 격리 참값으로 맞추고 차액을 다른 SMPS 로 넘긴다.

**왜 상한만으로는 부족한가.** 상한은 "55W 를 넘지 마라" 이고 참값은 46.9W 다 —
남는 8.1W 가 계속 프로젝터에 붙어 있었다. 그 8.1W 의 정체가 밝혀졌다: 학습된
모델은 프로젝터에 +17.00W 를 얹고 **충전기에서 정확히 −17.01W 를 뺀다**
(합 −0.01W, 제로섬). 총전력 부족분과는 무관하다. 맞바꿈이 보존적이므로
프로젝터를 참값에 못 박으면 그 몫이 충전기로 돌아간다.

| | 스냅 없음 | 스냅 46.9 |
|---|---|---|
| 프로젝터 중앙\|오차\| | 8.09W | **0.00W** |
| 격리 폭(44.2~47.8W) 안에 드는 비율 | 2.5% | **99.9%** |
| 충전기 (격리 33.3~68.4W) | 8.9W | 30.6W |
| 없는 기기 전력 | — | +0.4W |
| 잔차 | — | +0.04W |

⚠ 46.9 로 맞추고 46.9 로 채점하면 오차 0 은 당연하다. 근거는 위 표의
**격리 폭 안에 드는 비율**과 **충전기가 독립 기준(1단계 모델 31.1W)으로
돌아온 것**이다. 충전기·미니PC 에는 못 건다 — 격리 통전 폭이 넓어(33~68W,
8~23W) 단일 참값이 없다.

**②-b 전력 스냅 (`rm_snap`, 기본 켜짐).** 정합이 조합을 옳다고 판정한 창에서는
예전에 아무것도 안 했다. 그런데 **조합은 맞고 전력이 틀린 창**이 있다 — 오븐이
옳게 켜져 있는데 780W(`V²/R`=1130W)로 모자라고 그 차이가 문턱 아래 게이트를 타고
전기포트로 샜다. 이제 그 창에서도 `V²/R` 로 맞춘다. 시드 4개에서 없는 기기 전력
8.98 -> 5.23W, 실행 간 폭 6.05 -> 4.15W, 잔차 10.15 -> 7.30W, F1 과 전이 귀속은
불변이다. 끄려면 `rm_snap=False`.

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

## 7. ★ Ĝ 추정기 — 이제 모델 혼자 못 돈다 (14.378)

운영점 체크포인트는 **조합 머리**(`comb_tau > 0`)를 쓴다. 저항 4종(포트·드라이기·
핫플·오븐)의 전력을 자유 크기가 아니라 **컨덕턴스 표**에서 내고, 그 표의 어느 칸을
쓸지 고르는 것이 `Ĝ`(관측 총 컨덕턴스, mS)다.

```
  score_c = Σ_k β_k·z_{k,c_k}  −  |Ĝ − ΣG(c)| / τ  −  relu(ΣG(c) − Ĝ − 2.0) / 0.05
                                  └─ 물리 잔차 ─┘    └─ **초과 주장 꺾기** ─┘
  ⇒ `g_hat` 을 안 넘기면 `net.forward` 가 **ValueError 로 멈춘다.** 조용히 반쪽으로
    돌지 않는다 — 그러라고 그렇게 지었다
```
`NILMPredictor` 가 알아서 세운다. 비용은 잰 값이다:
```
  시작   **1.18s**  (FCM 기둥 표 한 번. `v_ref` = 그 창의 전압 중앙값)
  창마다 **0.053 ms**  (정규화 NNLS, 미지수 9개)
  ⚠ 기준전압이 *창 중앙값* 이라 연구 갈래(*녹화 전체* 중앙값)와 **0 은 아니다** —
    잰 차가 중앙 **0.0029 mS** · 최대 0.0079 다. Ĝ 잡음 σ 0.265 · 포트 식별 여유
    0.635 이니 여유의 **1.2%** 다. 그래서 표는 한 번만 짓는다
  ⚠ 자리가 바뀌면(다른 건물) |V₁| 이 5% 넘게 움직여 **자동으로 다시 짓는다**
```
★★ **꺾기(`comb_over`)는 추론 설정이지 학습값이 아니다.** 체크포인트에 적힌 값을
따라가면 안 된다 — 그래서 한 번 판정을 틀렸다(§45.1). 묶음은 **항상 운영점 0.05** 를
쓰고 한 줄 찍는다. 일부러 끄려면 `NILM_COMB_OVER=0` 으로 **명시**해야 한다.

---

## 8. 순서 뒤바뀜 — 대비돼 있다

수신기는 프레임을 **도착 순서 그대로** CSV 에 쓰는데, 펌웨어가 확인 못 받은 옛 장을
재전송하므로 `37 -> 28 -> 38 -> 29` 처럼 온다. `CycleRing` 이 **도착 순서가 아니라
`t_s`(보드 seq 로 계산한 값)** 로 자리를 정해 되꽂는다.

실측 녹화에서 잰 것:
```
  test_1 · test_3 · test_5   역전 **0건**
  test_2                     3건 (0.01%) · 최대 **59사이클 = 0.98초**
  test_4                     1건          · 최대 59사이클
```
관문 `gate_deploy.py [3]` 이 실제 창을 일부러 섞어 넣고 **바이트까지 같은지** 본다:
```
  순서대로 · 프레임 2장 뒤바꿈(1초) · 프레임 9장 역전(4.5초) · 실측 최악(59사이클)
  -> 네 경우 모두 최대차 **0.0e+00** (되꽂음 210~990건)
```
⚠ `reorder=False` 로 끄지 마라. 프레임 하나가 30사이클(0.5초)이고 세밀 갈래의 타깃이
창 끝에서 6초 안쪽이라, 오염되는 자리가 **정확히 타깃 근방**이다.
⚠ 창보다 더 뒤인 행이 연달아 오면 **세션 리셋**으로 보고 버퍼를 비운다 (보드 재부팅).

---

## 9. 모델 교체

```bash
cp <학습PC>/results/<새모델>.pt models/
python sync_runtime.py                     # 연구 코드와 다시 맞추고 표를 굽는다
python gate_deploy.py --ckpt models/<새모델>.pt
python run_predict.py --ckpt models/<새모델>.pt --replay ../data/test_1.csv --speed 0
```
⚠ **체크포인트만 복사하면 안 된다.** 2026-09-02 판 묶음이 그렇게 15일을 있다가
```
  세밀 채널 50 대 **61** · 링버퍼 33 대 **49** · 조합 머리 **없음** · Ĝ 추정기 **없음**
```
이 되어 새 모델을 **원리상 못 실었다.** 구조가 바뀌면 `sync_runtime.py` 를 돌려야 한다.
`gate_deploy.py` 가 그것을 지킨다 — 관문이 통과해야 배포다.

⚠ 입력 규약(`even_median` · `volt_orders`)은 체크포인트가 들고 있고 묶음이 **맞춘다**.
  어긋나면 죽지 않고 *그럴듯한 틀린 수*를 내므로 `check_input_convention` 이 **멈춘다**.

---

## 10. 학습 저장소와의 관계

**사본은 없다.** `deploy/src/` 와 `deploy/circuit_model/` 은 저장소 파일을 **바이트
그대로** 비춘 것이고 `sync_runtime.py --check` 가 그것을 확인한다.

| 배포 | 원본 | 관계 |
|---|---|---|
| `src/**`, `circuit_model/**` | 같은 경로 | **바이트 동일** (17개) |
| `src/*/__init__.py` | — | **빈 파일** (원본은 학습 스택을 끌어온다) |
| `nilm_runtime/predictor.py` | `src/run_live.py` | 묶음 고유. 링버퍼·CSV 는 이제 **공용** |
| `nilm_runtime/receiver.py` | `nilm_receiver.py` | 묶음 고유 |
| `nilm_runtime/*.npz` | `SegmentPool` | `sync_runtime.py bake()` 가 굽는다 |

설계 근거와 측정 기록은 학습 저장소의 `MODEL_TRAINING_DESIGN.md` 에 있다
(후처리 12.100~12.104 · 조합 머리 14.347 · 꺾기 14.352 · 배포 재동기화 14.378).
