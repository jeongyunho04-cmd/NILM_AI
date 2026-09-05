# READ ME FIRST — 2026-09-06 계측기가 바뀌었다. **그 전 자료로 내린 결론은 전부 재검토 대상이다**

> 이 저장소의 다른 문서(설계 문서 12.185 까지, 인수인계 09-05b 까지, `CIRCUIT_FCM_GUIDE.md`,
> `TEST_DATASET_TIMELINE_ANALYSIS.txt`, 옛 메모리)를 읽을 때 **먼저 이 문서의 §4 표를 본다.**
> 거기 없는 결론을 인용하려면 "옛 계측기 자료다" 를 붙이고, 채택하려면 새 자료로 다시 잰다.
> 새 규칙은 `MEASUREMENT_RULES.md`(v2, 2026-09-06 부터 새로 쌓는다)에 있다.

## 0. 무슨 일이 있었나 (사용자 진술, 2026-09-06)

옛 계측기(STM32 보드, 차동 ADC)의 결함:

> ADC 차동 입력의 공통모드 한계. 이 회로는 VINN 을 VREF/2 에 박아 놓고 VINP 만 한쪽으로 휘두른다.
> 그러면 공통모드 (VINP+VINN)/2 가 위쪽 피크에서 2.4V 까지 올라간다. 차동 SAR 은 보통 두 입력이
> 대칭으로 움직이는 걸 전제로 특성이 잡혀 있고, 공통모드가 VDDA 쪽으로 치우치면 입력단(부트스트랩
> 스위치·샘플링 캡 리셋 상태)이 선형을 벗어난다. 문턱이 2.65V ≈ VDDA − 0.65V 라는 것, 위쪽만
> 그렇다는 것이 이 그림과 맞는다. 샘플링 캡이 변환 사이에 VINP 쪽은 VDDA, VINN 쪽은 VSS 로
> 리셋되는 구조라면, 정착이 덜 됐을 때 ΔV 가 실제보다 커지고 최대 VDDA 까지 부풀 수 있다 —
> 관측된 14104 가 딱 그 범위 안이다.

즉 **위쪽 피크만 부풀었다.** 그것이 이 저장소가 1년치 문서에서 "짝수차 인공물", "전압 채널 h3 덧셈
인공물", "계측 h3 바닥", "+352/−323V 전압 비대칭" 이라 부르던 것의 뿌리다. 옛 계측기는 폐기됐고,
**단일 입력 ADC + 바이어스 채널 소프트웨어 감산** 으로 다시 만들었다. 옛 자료는 혼동을 막기 위해
**전부 지웠다.** 2026-09-06 이전에 만든 자료로 내린 결론은 모두 재검토가 필요하다. 옛부터 문제였던
짝수차 인공물은 사라졌다. 새로 잰 자료를 `data/` 에 넣었고, 없는 기기는 재서 추가한다.

사용자가 새 녹화에서 세운 규약 (MEASUREMENT_RULES.md 규칙 2·3): 단독 녹화는 **플러그 꽂힌 대기로
시작**하고, 끝내기 전에 **기기 플러그를 뽑고 ~10초** 를 더 찍어 그 파일의 계측계 노이즈를 잰다.
빼먹은 녹화는 전역 노이즈 파일로 대체한다. **seq 0** 프레임은 계측기 시작 단계라 버린다.

## 1. 계측기 세대의 경계

| | 옛 계측기 (~2026-09-05) | 새 계측기 (2026-09-05 21:26 ~) |
|---|---|---|
| ADC | 차동, VINN=1.65V 고정, VINP>2.65V 에서 부풀림 | **단일 입력** + 바이어스 채널 감산 |
| 전류 짝수차 (순저항 포트) | 0.07~0.09% + 레인지 전환 단차 | **0.03%** |
| 전압 짝수차 vh2/V1 | **2.6%** (12.185.25) | **0.03%** |
| 전압 h3 (저항 부하가 보는 것) | 덧셈 33.5mA∠164° + 장소 | 없음 — 포트 |I3/I1| 3.23% = vh3/V1 3.25% |
| 원시 CSV | `high,v_r1,low,v_r2`, LSB 402.8µV, 반파 대칭화 필수, 스큐 −0.313 표본 | `low,v,high,bias` + `i_a,v_v,range`, LSB 201.4µV, 대칭화 불필요, 스큐 없음, RC τ=60µs |
| 2Hz CSV | 프로토콜 v5 (vhdeg 포함) | **같다.** 파일명에 `.cal2` 꼬리 (`cal_applied=1`), 뒤 파일들은 꼬리 없음 |
| 회로 모델 | `results/_circuit_raw_C.json` (v3, rd 고정) | `circuit_model/circ12_*.pkl` (v12g, 잔차 2.1~4.5%) |
| 녹화 규약 | 없음 | 꽂힌 대기로 시작, 플러그 뽑은 10초로 끝 (규칙 2), seq 0 폐기 (규칙 3) |
| 코드 경계 | `file_registry.LEGACY_*`, `LEGACY_DATA_CUTOFF` | `DEVICE_FILES`, `MEASUREMENT_ERA` |

`file_registry.check_measurement_era()` 가 전처리에서 첫 `host_time < 2026-09-05 20:00` 인 파일을 **거부**한다.
옛 파일명(`fan_2`, `test_5`, …)은 등록부에 없어 `UNKNOWN` 이다 — 같은 이름의 새 녹화를 옛 자료로 오인하지 않게
(새로 잰 파일은 같은 이름이라도 `DEVICE_FILES` 에 다시 등록하면 된다; `fan_1`·`hair_dryer_1` 이 그렇게 들어갔다).

## 2. 지금 있는 자료 (전부 사용자 자리, 새 계측기)

| 파일 | 시각 | 길이 | vrms | 꼬리(플러그 뽑음) | 요지 (원본 CSV 에서 직접 잰 것) |
|---|---|---|---|---|---|
| laptop_charger_1 | 09-05 21:26 | 16.4분 | 215.7 | **없음** (충전 중 종료) | 65W 정속 87%, 20~60W 테이퍼 일부, 사이클 최대 68.2W, ∠I1 +6.0°. 72W 정전류·돌입 없음 |
| oven_1 | 21:47 | 18.3분 | 216.1 | **없음** (대기 2.5W/14mA 로 종료) | 히터 1101W 31% / 팬·조명 16.2W 62% / 대기 2.4W. R(히터) 40.09Ω. 팬·조명 |I2|/|I1| 0.058 |
| minipc_1 | 22:17 | 26.7분 | 217.6 | 34초 1.58W/7.4mA | IDLE 10~13W 28%, 부하 13~30W 64%, 대기 3.4W. 사이클 최대 29.5W |
| noise_noselfpower | 22:45 | 3.0분 | 218.3 | (전체가 계측계) | 1.59W, 7.46mA∠+12.2°, |I3| 1.76mA |
| noise_selfpower_1 / _2 | 22:48 / 22:53 | 4.0 / 4.6분 | 218~219 | (전체) | 2.41 / 2.21W, G 0.051 / 0.046mS (fcm12 `G_BG_DEFAULT` 0.051 과 맞음) |
| hotplate_1 | 23:00 | 13.6분 | 215.6 | 18초 1.39W/6.7mA | 456.5W 통전 41%, LOW 만. R 100.83Ω, ihdeg1 +0.05°. 두 세션(17~665 / 695~795초, 사이 29초 스위치 내림). 뒤 세션은 2~5사이클 펄스 |
| test_1 | 23:50 | 13.3분 | 215.2 | (복합) | 포트·오븐·핫플·충전기·미니PC. **스위치 로그 → 라벨 완료 (§3b)** |
| electric_kettle_1 | 09-06 00:30 | 7.7분 | 228.0 | 31초 1.48W/6.4mA | 1458W, HIGH 61%, on/off 여러 번. R 35.61Ω, ihdeg1 −0.05° |
| beam_projector_1 | 00:40 | 13.5분 | 231.0 | 23초 1.46W/6.4mA | ON 46.2W (60초창 중앙 46.2), 대기 3.1W, 사이클 최대 50.9W |
| beam_projector_2 | 01:10 | 9.9분 | 228.6 | 16초 1.33W/6.2mA | ON 46.1W (60초창 45.9), 대기 2.9W. `.cal2` 꼬리 없음 — 열 구성·`cal_applied` 같음 |
| fan_1 | 01:46 | 9.7분 | 228.8 | 33초 1.44W/6.5mA | 1/2/3단 22.9 / 30.8 / 39.0W, ∠I1 +30/+21/+4°, PF 0.86/0.93/1.00. 옛 상태 문턱(10/27/36W) 그대로 |
| hair_dryer_1 | 01:57 | 8.2분 | 227.5 | 22초 1.43W/6.4mA | 강 965.8W R 53.22Ω 순저항(|I2| 0.000) / 약 488.4W R 106.0Ω **반파** |I2|/|I1| 0.431 (이론 0.424) |
| air_conditioner_1 | 02:11 | 27.8분 | 228.2 | 22초 1.42W/6.4mA | 꽂힌 대기 6.8W/117mA∠+76° / 송풍 10~30W / 냉방 80~640W (인버터 |I3|/|I1| 0.58, PF 0.81). 옛 상태 문턱(10/80/350/600) 안 |
| laptop_charger_2 | 02:42 | 4.2분 | 229.6 | 21초 1.47W/6.4mA | 10~70W 테이퍼·정속. **꼬리를 지킨 첫 충전기 녹화** |
| raw_laptop_charger_5~8 | 02:43~45 | 40주기 ×4 | 229 | (원시) | 새 계측기 원시 스냅샷 (충전기 65W, LOW). 2Hz 파이프라인은 건너뛴다(RAW 역할). `fit12`·`fcm.source_from_raw12` 용 |

꼬리의 계측계 값이 파일마다 1.33~1.60W / 6.2~7.5mA / 위상 −21~+13° 로 다르다 — 규칙 2 의 근거. 9/11 기기 파일이
자기 꼬리를 쓰고 충전기_1·오븐_1 은 전역 `noise_noselfpower` 로 대체됐다 (`SegmentPool.noise_source`).

**같은 콘센트가 시간대에 따라 다르다** — 옛 문서의 "장소 A/B/C" 축은 이제 **세션(시간대)** 축이다:

| 시간대 | vrms | vh3/V1 (위상, V1 기준) | vh5 | vh9 (위상) | vh15 |
|---|---|---|---|---|---|
| 저녁 21:26~23:50 | 214.7~218.4 | 0.58~0.85% (−98~−105°) | 1.6~1.9% | 0.53~0.63% (−120~−136°) | 0.35~0.51% |
| 심야 00:30~02:05 | 227.5~231.1 | 2.9~3.25% (−116~−117°) | 1.7~1.9% | 0.92~1.08% (+87~+90°) | 0.15~0.17% |

**없는 것 (재측정 대상):** 충전기 72W 정전류 구간, 오븐 **꼬리**, 프로젝터 두 번째 동작점, **긴 무부하 배경**,
회로 모델(pkl)을 만든 원시 파일(`RAW12_FIT_FILES`, 저장소 밖 — `raw_laptop_charger_5~8` 은 새로 들어온 것),
LOW 에 머무는 순저항 원시, 그리고 **활성화 수**: 핫플 1 · 에어컨 3 · 미니PC 4 (합성기 경고, §8 ①. 오븐은 3 — 아래).

## 3. 새 자료에서 이미 확인한 것 (2026-09-06, 원본 CSV)

**규칙 74(옛) 통과** — 순저항의 `ihdeg1` 이 LOW·HIGH 양쪽에서 0 이다. 크기비도 1 이다.

| 파일 | 레인지 | ihdeg1 | h3: ∠I−∠V / 크기비 | h5 | h7 | h9 |
|---|---|---|---|---|---|---|
| electric_kettle_1 | HIGH | −0.05° | −8.8° / 1.01 | −4.7° / 1.00 | −17.3° / 1.21 | −12.8° / 0.95 |
| hotplate_1 | LOW | +0.05° | −33.8° / 0.86 | −5.1° / 0.99 | −9.8° / 0.90 | −11.1° / 1.16 |
| oven_1 | HIGH | +0.06° | −19.1° / 0.78 | −3.4° / 0.93 | −16.9° / 0.96 | −7.8° / 0.92 |
| hair_dryer_1 강 | HIGH | −0.11° | (|I2| 0.000) | | | |

크기비 `|I_h/I_1| / |V_h/V_1|` 가 h3·h5 에서 0.8~1.0 — 옛 계측기의 h3 덧셈 바닥(33.5mA)이 없다.
h3 위상차 −9~−34° 는 남아 있다 (vh3 가 1.4~7V 로 작아 위상 잡음일 수 있다; **미상**, 새 보드의 LOW 순저항 원시로 가를 것).

**짝수차**: 포트 |I2|/|I1| 0.03%, 핫플 0.08%, 오븐 히터 0.18%, 전압 0.03%. 남은 것은 실신호 — SMPS 1.3~1.9%
(도통 비대칭), 오븐 팬·조명 5.8% (12.164.16 의 신원), 드라이기 약풍 반파 43.1% (강풍은 0.000).

**기기 상수**: 등가저항 포트 35.61 / 오븐 40.09 / 핫플 100.83 / 드라이기 강 53.22 · 약 106.0Ω — 옛 35.8 / 40.6 / 101.8 /
54.3 · 108.6 의 2.4% 안. 프로젝터 46.2/45.9W (옛 46.9). 미니PC 사이클 최대 29.5 (옛 29.8). 계측 배경 1.59 / 2.41·2.21W
(옛 1.4 / 2.37). 선풍기 3단 22.9/30.8/39.0W (옛 문턱 10/27/36 안). **드라이기 약풍 반파** |I2|/|I1| 0.431 (12.109.2 유지).

**seq 0** (13파일 전부): pll_locked=0 이 30사이클, 전압 클립 창 1개, vrms 가 다음 프레임과 2~3V 다르다 → 규칙 3.

**핫플의 세 바닥** (규칙 5): 플러그만 1.54W/7.3mA/|I3| 1.8mA · 스위치 켜짐·릴레이 열림 1.97W/9.2mA/|I3| 1.0mA ·
플러그 뽑음 1.39W/6.7mA. 0.4W·1.9mA 차이. 라벨에 `ARMED_IDLE`(state 1)이 생겼고 통전(state 2)만 ON 이다.

### 3b. test_1 라벨 — `user_timeline.txt` 를 신호로 정밀화 (규칙 4)

`run_switch_sig`(단독 9녹화 → 전이 240개) → `run_refine_labels --all` → `run_write_labels` → `processed_data/real_events.json`.
18항목 중 **17 맞춤 + 1 시작부터 켜짐**, 사람 시각과의 차 |Δt| 중앙 1.0초 · 최대 3.6초 (사용자 진술 ±3초와 맞는다).

| 사람 seq | 기기 | 동작 | 신호 t_s | Δt | ΔP(기기) | 비고 |
|---|---|---|---|---|---|---|
| 0 | laptop_charger | 켬 | 0.0 | — | — | 시작부터 켜짐 (17W) |
| 70 | hotplate | 켬 | 37.4 | +2.9 | +464 | |
| 125 | oven | 켬 | 61.0 | −1.0 | +12.7 | 제어보드·팬·조명 |
| 210 | minipc | 켬 | 104.4 | −0.1 | +9.1 (h3) | |
| 275 | laptop_charger | 끔 | 137.1 | +0.1 | −13.3 (h3) | 총전력 −397 은 핫플 펄스 |
| 329 | laptop_charger | 켬 | 165.1 | +1.1 | +52.4 (h3) | |
| 390 | oven | 끔 | 195.8 | +1.3 | −13.5 | **구제**: 191.3초의 −1467W 는 히터 서모스탯. 스위치는 핫플 펄스 사이 바닥에서 |
| 449 | minipc | 끔 | 224.5 | +0.5 | −7.6 (h3) | |
| 537 | oven | 켬 | 266.7 | −1.3 | +12.7 | |
| 614 | hotplate | 끔 | 305.4 | −1.1 | −450 | |
| 635 | minipc | 켬 | 317.9 | +0.9 | +9.7 (h3) | |
| 842 | hotplate | 켬 | 424.1 | +3.6 | +453 | **구제**: 0.5초짜리 약한 단 펄스, 릴레이가 +3.5초 뒤에 붙음 |
| 895 | minipc | 끔 | 447.3 | +0.3 | −11.4 (h3) | |
| 1064 | laptop_charger | 끔 | 531.7 | +0.2 | −13.6 (h3) | |
| 1083 | minipc | 켬 | 542.3 | +1.3 | +9.6 (h3) | |
| 1330 | oven | 끔 | 666.2 | +1.7 | −13.6 | |
| 1440 | hotplate | 끔 | 719.9 | +0.4 | −455 | **구제**: 마지막 0.5초 펄스의 끝 |
| 1556 | minipc | 끔 | 778.1 | +0.6 | −8.9 (h3) | |

시각은 npz 기준(seq 0 을 버려 `t_rel = (seq−1)·0.5`). ΔP 는 기기 몫(SMPS 는 h3 에서 되돌린 값). `uncertain` 0개.
이 라벨의 출처 표지는 `human_switching_log_signal_refined` 이고 2단계 사람 라벨 지도가 받는다.

## 4. 옛 자료로 세운 결론 — 무엇이 어떻게 됐나

상태: **폐기** = 새 계측기에서 뿌리가 사라졌다 · **대체** = 새 자료로 다시 만들었다 · **유지** = 새 자료로 재확인됐다 ·
**미검증** = 아직 못 쟀다, 채택 금지.

| 결론 (근거 절) | 상태 | 새 계측기에서 | 코드에 한 일 |
|---|---|---|---|
| 짝수차는 레인지 전환 단차 인공물 (12.72) | **폐기** | 뿌리는 ADC 공통모드 한계. 인공물 0.03% | `inputs.ZERO_EVEN_HARMONICS` 는 **True 그대로** — 되살리는 것은 재학습 실험 (§6) |
| 짝수차 지터 `--dither-even-amp` / `cnn_even` 계열 (12.76), 프로젝터↔충전기 |I2|/|I1| 분리 d'>3 | **폐기** | 그 분리는 인공물이었다: 새 계측기 1.5% vs 1.8%, d' 1.65 | 테스트 `test_even_dither_merges_the_pair_ratio` 를 전제 없으면 skip 으로 |
| 전압 채널 h3 2.6%·짝수차 2.6% 인공물, 반파 대칭화 필수 (12.185.25, 옛 규칙 77) | **폐기** | vh2 0.03%. 대칭화 불필요 (README_v12) | `fit_raw.load_raw` 가 새 포맷을 거부. `fcm.to_spectrum` 짝수차 소거는 무해라 둠 |
| 계측 h3 바닥 33.5mA∠164°, ①a 장소 h3 왜곡 d3 (12.185.21) | **폐기** | 바닥 없음. 저항 녹화가 세션의 진짜 vh3 을 담는다 | `grid_simulator.METER_H3_FLOOR_A=0`, 무리 d3=0, **①a 끔** (재설계는 차분으로) |
| 장소 A/B/C 지문 vrms·vh3·vh9·vh15 (12.179.4, 12.184) | **폐기** | 장소 아니라 시간대. 저녁/심야 두 무리 | `OBSERVED_VOLTAGE_CLUSTERS` 216.5/229.5V, `SITE_OF_STEM` 전부 'C' |
| ①b 전압 텍스처 지배, 결합 1/10 (12.185.22) | **대체** | 텍스처는 2Hz 녹화의 vh·vhdeg 에서(173개). test_2 로 확인: 녹화 중첩 0.248 → 텍스처 델타 0.111 (12.187) | 합성기에 텍스처 델타 + 결합 델타 (v12g). `run_mixval12` 가 자 |
| 원시 위상 스큐 −0.313 표본, LOW 0.44 / HIGH 2.62 (12.185.25, 옛 규칙 76, 메모리) | **폐기(옛 보드)** | 새 보드는 스큐 보정 없음(기각 유지). 교정 상수는 미기록 | `RAW_SKEW_SAMP_LOW` 옛 보드 표기, `RAW_SKEW_SAMP_LOW_V12=0`, LOW_CAL 블록 no-op |
| 회로 파라미터 C/R/L/Cx, rd 0.3 고정 (12.185.19/25) | **대체** | v12g: 포화 L + 선로측 Cx + 덧셈 G, 잔차 2.1~4.5% | `fcm.load_models()` 기본 = `circuit_model/circ12_*.pkl` (`DeviceModel12`) |
| nvt·Gp·alpha·Shockley·계측 2극·fc·CT 고역 기각 (12.185.20~24) | **부분 유지** | README_v12 가 기각 유지로 적음 (새 자료로 확인) | — |
| 등가저항 35.8/40.6/54.3/101.8Ω, 반파 108.6 (12.112, 12.109.2) | **유지** | 35.61/40.09/53.22/100.83, 반파 106.0 (|I2|/|I1| 0.431) | 값 그대로, 재검증 주석 |
| 계측 배경 1.4/2.37W, 7.3mA∠+11° (옛 규칙 78) | **대체** | 1.59/2.3W, 7.46mA∠+12° — **그리고 파일마다 다르다** → 규칙 2 | `NOISE_FLOOR_EXTERNAL_W 1.6`, `SELFPOWER 2.3`, 파일별 꼬리 우선 |
| 오븐 팬·조명 14.2W/64mA, 신원은 h2 (12.156, 12.164.16~18) | **유지** | 16.2W 총, 75mA, |I2|/|I1| 0.058. test_1 의 오븐 켬/끔 ΔP +12.7/−13.5 | — |
| 프로젝터 참값 46.9 (`power_ref`, `SNAP_TARGET_W`) | **유지, 재계산 필요** | 46.2/45.9 (폭 44.8~47.3) | 값 그대로 + 경고. `run_power_check --recompute-ref` 로 갱신 |
| `REFERENCE_W` 저항 3종 (1277/460/1143W) | **미검증** | 포트 1458W@228V — 전압에 걸린 값이다 | 경고. 저항은 `V²/R` 로 내도록 재계산 때 고칠 것 |
| `ABSORB_CAP_W` 55/84.6/29.8 | **유지** | 50.9·48.0 / 68.2 / 29.5 (충전기는 이번 녹화가 낮은 것) | 그대로 |
| 상태 정의 문턱 (`state_definitions`) | **유지+보강** | 미니PC IDLE 10~13W, 프로젝터 46W, 오븐 팬·조명 16.2W, 선풍기 3단. 핫플은 플러그만/ARMED/통전 세 상태 (규칙 5) | 핫플 `ARMED_IDLE`, 평활 0.05초, 체류 1사이클 |
| 사람 타임라인 정밀화 방법 (12.155: h3 로 SMPS, P 로 저항, ±7초) | **유지+보강** | 17/17. 창 ±3.5초. 핫플 짧은 펄스·오븐 바닥 계단은 **구제** 후처리 | `run_refine_labels`: `SWITCH_DP_MAX_W`, `rescue_pulses`, `rescue_envelope` |
| SMPS 오배분 뿌리 = 계통 임피던스, `--harm-offset` (12.148), 장소 전달비 `site_transfer` (12.179~181), 위상 지터 (12.182~183), 스켈치+흡수 (12.149~153) | **미검증** | 전부 옛 복합 녹화 위의 결론. 짝수차·h3 인공물이 손실에 들어가 있던 상태에서 잰 것 | 코드는 그대로. **새 자료로 재실험 전 채택 금지** |
| 운영점 `results/adapt_ph5_s0.pt` + 후처리 (인수인계 09-05b §1) | **폐기(모델)** | 옛 지문·대기·harm_offset 이 내장돼 있다 | 체크포인트는 남겨 둠. `run_live` 기본값은 옛 파일을 가리킨다 — 새 학습 전에는 배포 불가 |
| 사람 라벨 `real_events.json` 5파일, `minipc-idle-only` 메모리 (test_14~18) | **폐기(자료 삭제)** | 새 라벨은 test_1 하나 (§3b) | `realdata.HUMAN_ON_*` 옛 stem 은 무해(건너뜀) — test_1 을 넣으려면 그 목록 갱신 |
| 12.164.18 `ch51=|I2|` 채널이 장소 B 붕괴를 고침 | **미검증** | 신호(팬·조명 h2)는 실재 | 채널은 그대로. 효과는 재학습으로 |
| 옛 규칙 82 (순저항 원시가 계측기의 자) | **유지** | 새 보드에도 그대로 적용 | 규칙 1 에 흡수 |

## 5. 코드에서 바꾼 것 (2026-09-06)

```
src/preprocessing/file_registry.py   전면 개정. APPLIANCE_CATALOG(9종) / DEVICE_FILES(새 11파일) / NOISE_FILES(3) / RAW 역할(raw_*)
                                     / LEGACY_*_OLD_ADC / MEASUREMENT_ERA / check_measurement_era / .cal2 꼬리 제거
                                     / resolve_csv / SITE_OF_STEM 전부 C / PHASE_FIX·QUARANTINE·RAW_* 비움 / RAW12_FIT_FILES
src/preprocessing/pipeline.py        옛 계측기 자료(host_time < 2026-09-05 20:00) 거부. 꼬리 검출은 단독 기기 파일만
src/preprocessing/cleaner.py         seq 0 프레임 폐기, 앞머리 무효 구간 폐기(보간 아님), 플러그 뽑은 꼬리 검출
                                     (`is_unplugged`), 파일별 바닥(`noise_floor_source`)          <- 규칙 2·3
src/preprocessing/raw_csv.py         read_raw_csv(drop_startup=True) — 탐침·정밀화도 같은 시간축
src/preprocessing/numpy_exporter.py  npz 에 seq / cycle / is_unplugged, 메타에 trailing_noise·noise_floor_source·seq_first
src/synthesis/segment_pool.py        파일별 꼬리를 그 파일의 노이즈 기준으로 (없으면 전역 파일), 대기 지문에서 꼬리 제외
src/synthesis/grid_simulator.py      전압 무리 216.5/229.5V, METER_H3_FLOOR_A=0, d3=0 (①a 끔). 12.187: 환경에 텍스처·R 상태,
                                     apply_voltage_texture(녹화 파일 기준 델타), apply_smps_coupling(Z 결합 델타)
src/synthesis/synthesizer.py         사이클별 출처 파일 id(rec_tex) 추적, 되먹임 루프에 텍스처 델타 → 결합 델타
src/synthesis/coupling.py            (재작성) SmpsCircuit: v12g 위 텍스처 델타·결합 델타 + 캐시 (12.187)
src/synthesis/vtexture.py            (신규) 2Hz 녹화의 vh·vhdeg 로 전압 텍스처 라이브러리 (12.187)
src/synthesis/fcm.py                 DeviceModel12(+simulate_true) / load_models_v12 / load_models() 기본 v12 / source_from_raw12
src/run_mixval12.py                  (신규) 혼합검증: 실측 총전류 대 녹화중첩/텍스처델타/모델단독 (규칙 6)
src/synthesis/fit_raw.py             새 원시 포맷 거부 (옛 차동 ADC 포맷 전용임을 명시)
src/labeling/timeline_parser.py      `seq0 …` 표기, '하플' 오타
src/labeling/state_definitions.py    핫플 ARMED_IDLE 상태(바닥 준위, 규칙 5), 기기별 평활 창(smooth_window_s)
src/labeling/state_classifier.py     평활 창을 설정에서 (핫플 0.05초 — 2~5사이클 펄스가 통전으로 잡힌다)
src/labeling/annotator.py            `_apply_armed_state`: p_target 4초 10백분위 >= 0.3W 면 ARMED, 아니면 플러그만
src/synthesis/segment_pool.py        펄스 사이에 state 0 이 1초 이상 끼면 세션을 가른다 (오븐 3, 핫플 2)
src/run_switch_sig.py                단독 녹화 목록을 등록부에서, resolve_csv, to_blocks(return_lo)
src/run_refine_labels.py             --timeline user_timeline.txt, --max-dt 3.5, seq_lo 자료에서, 시작 이전 켬=already_on,
                                     오븐 켬/끔은 제어보드 갈래만(SWITCH_DP_MAX_W), 구제(rescue_pulses/rescue_envelope)
src/run_write_labels.py              seq_time_map 없이, uncertain ±4초, 기본 출력 processed_data/real_events.json
src/model/realdata.py                정밀화 출처(HUMAN_PROVENANCES)도 사람 라벨로 받음
src/model/inputs.py                  ZERO_EVEN_HARMONICS 근거 상실 주석 (동작 불변), V_CENTER 주석
src/model/postproc.py                RESISTIVE_OHM·ABSORB_CAP_W·SNAP_TARGET_W 재검증 주석 (값 불변)
src/evaluation/power_ref.py          REFERENCE_W 옛 값 경고
src/visualization/plot_labeled_data.py, run_synthesis.py   PNG 저장 재시도 (갓 쓴 파일 잠금 Errno 22 두 번)
tests/                               라벨 테스트 3개를 옛 test3 고정에서 '지금 있는 라벨'로, 짝수차 지터 테스트는 전제 없으면 skip
circuit_model/__init__.py, fcm12.py  패키지로 임포트되게 (상대 임포트 폴백). pkl 3개를 zip 에서 풀어 둠
MEASUREMENT_RULES.md                 v2 새로 시작 (규칙 1~4). 옛 1~82 는 git show 88ec645:MEASUREMENT_RULES.md
```

**돌려 본 것 (2026-09-06 02:10~03:00):**
```
python -m src.run_preprocess_and_label   15파일+원시 4 건너뜀, 51초. seq 0 폐기, 꼬리 9/11 검출 (충전기_1·오븐 없음 -> 등록부 바닥)
python -m src.run_switch_sig             단독 9녹화 -> 전이 240개 (항등식 Re(ΔI₁)·V/ΔP 0.999~1.025)
python -m src.run_refine_labels --all    test_1: 17/17 맞춤 (구제 3), |Δt| 중앙 1.0초 최대 3.6초
python -m src.run_write_labels           -> processed_data/real_events.json (사건 17, 구간 기기 4, 불확실 0)
python -m src.run_synthesis              **9종**, 텍스처 델타 + 결합 델타 켬 (12.187), 파일별 꼬리 노이즈 9/11. 활성화: 오븐 3 · 핫플 2 · 에어컨 3 · 미니PC 4
python -m src.run_mixval12 --stems test_2 test_1   녹화중첩 0.248/0.105 → 텍스처델타 0.111/0.115, 모델단독 0.116/0.287 (12.187.4)
   (합성 벤치 135 -> 37 창/초: 시뮬 캐시가 데워지기 전 값. 캐시 빌드 30만 창 ≈ 12분/11워커)
python -m pytest tests -q                **137 passed / 0 failed / 5 skipped**
```

## 6. 남아 있는 옛 산출물 — 쓰지 말 것

| 자리 | 무엇 | 처리 |
|---|---|---|
| `results/*.pt` (353개), `deploy/models/*.pt` | 옛 자료로 학습한 체크포인트 (지문·대기·harm_offset·site_transfer 내장) | 배포 금지. 새 학습 뒤 지울지 결정 |
| `results/_circuit_*.json`, `_meas_rc.json`, `_residual.json`, `_coupling.json`, `_resistive_site.json`, `_site_voltage.npz`, `_vh_check_*.npz` | 옛 원시·2Hz 로 낸 회로·계측 판정 | 참고용. `fcm.load_models("results/_circuit_raw_C.json")` 은 경고를 낸다 |
| `results/*harm_offset*.npz`, `*site_transfer*.npy`, `sig_*.npz` 류 | 옛 복합 녹화의 보정 계수 | 새 자료에 걸면 **조용히 틀린다** |
| `results/switch_sig.json`, `results/refined_labels.json` | **새 자료** (2026-09-06) | 라벨 정밀화의 중간 산출물 — 새 녹화가 오면 다시 만든다 |
| `deploy/nilm_runtime/signatures.npz`, `deploy/nilm_runtime/*.py`(8/26 사본) | 옛 지문·옛 채널 규약(58ch) | 새 운영점이 서면 다시 복사 (메모리 power-monitor-deploy-bundle) |
| `docs/external/code/vtexture_C.npz`, `vtemplate_C*.npz`, `curveC.npy`, `circ_rc_laptop_charger.pkl` | 옛 계측기 텍스처·템플릿 | 참고용 |
| `MODEL_TRAINING_DESIGN.md` 12.1~12.185, `HANDOFF_*` ≤ 09-05b, `CIRCUIT_FCM_GUIDE.md`, `TEST_DATASET_TIMELINE_ANALYSIS.txt`, `REPLY_vtemplate_rc_2026-09-05.md`, `PROMPT_open_problems_2026-09-05.md` | 옛 계측기 시절 기록 | 읽을 때 §4 표를 옆에 둔다. 12.186 이 경계다 |
| `NILM_EXECUTION_GUIDE.txt` | 명령은 유효, 예시 경로·등록 예시는 옛것 | [2-1] 등록 예시를 `_dev("type")` 로 고쳤다 |
| 삭제된 문서 (git 이력에만): `READ_ME_FIRST.txt`, `SMPS_PLAN_2026-08-31.md`, `MEASUREMENT_CHECKLIST.md`, `OPERATING_POINT.txt`, 옛 `MEASUREMENT_RULES.md` | 사용자가 의도적으로 지움 — 새 자료에서 새 규칙을 쌓는다 | 옛 규칙 1~82 는 `git show 88ec645:MEASUREMENT_RULES.md`. 새 v2 는 규칙 1~4 |

## 7. 규칙 1 — 계측기가 바뀌면 그 전 자료의 결론은 전부 "미검증" 으로 내린다

(`MEASUREMENT_RULES.md` 규칙 1 과 같다.) 계측기의 결함은 신호 통계로 위장한다. 이 저장소는 짝수차 인공물을
**레인지 전환 단차**로(12.72), 전압 h3 인공물을 **ZMPT 자기 왜곡**으로(12.184.11), 장소 지문을 **콘센트의
성질**로(12.179) 읽고 각각에 처방(지터·마스크·대칭화·h3 바닥·장소 무리)을 얹었다. 뿌리는 하나(ADC 공통모드)였고
처방은 전부 그 뿌리를 가리는 자리표였다. 계측기를 바꾸면:

1. **자료를 지운다.** 섞이면 어느 쪽이 인공물인지 다시 못 가른다 (이번에 그렇게 했다).
2. **결론을 표로 옮긴다** (§4). 상태를 폐기/대체/유지/미검증 중 하나로 적고, "유지" 는 새 자료로 잰 숫자를 옆에 둔다.
3. **첫 녹화는 순저항**이다. `ihdeg1 = 0`, `ihdeg_h = vhdeg_h`, `|I_h/I_1| = |V_h/V_1|` 셋을 LOW·HIGH 에서 본다.
4. **코드에 경계를 박는다** — 옛 파일명은 등록부에서 빼고, 옛 시각의 파일은 파이프라인이 거부한다.
5. 옛 문서는 지우지 않되, 읽는 쪽이 **경계 절 번호**(12.186)를 알게 한다.

## 8. 다음 순서

```
① (사용자) 세션 더: 활성화 = 켬~끔 한 번 (릴레이 펄스는 한 세션으로 묶는다). 지금 오븐 3(333/288/329초) · 핫플 2
   (649/68초, 규칙 5 의 바닥 준위로 갈림) · 에어컨 3 · 미니PC 4. 듀티(통전·휴지 길이)는 증강기가 이미 0.5~2배로 흔들고
   위상도 무작위로 자르므로 **듀티 다양성은 녹화로 채울 필요가 없다.** 녹화가 채워야 하는 것은 **켜는 순간·끄는 순간의 수와
   설정(단·온도·부하)의 가짓수**다. 5회 이상, 설정을 바꿔 가며, 파일 끝에 꼬리 10초. 새 파일은 `DEVICE_FILES` 에 한 줄
② python -m src.run_power_check --recompute-ref      REFERENCE_W / SNAP_TARGET_W 재계산 (저항은 V²/R 로)
③ 새 복합 녹화가 오면: user_timeline.txt 에 머리(`test_3.csv`)와 줄을 더하고 규칙 4 의 세 명령 (test_2 는 31/31 로 끝났다)
   생성기를 바꿨으면 `run_mixval12` 로 먼저 잰다 (규칙 6). 새 원시 raw_laptop_charger_5~8 는 fit12 재적합 후보
④ 캐시·1단계·2단계 — GPU. 돌리기 전에 묻는다 (메모리 ask-before-long-gpu-runs). 머리 수 9 (전부 있다)
   2단계 사람 라벨 지도를 쓰려면 `realdata.HUMAN_ON_DEFAULT_STEMS` 에 test_1 을 넣는다
⑤ README_v12 의 남은 일: crb/mixval/vtexture 를 fcm12 로, 충전기 20~40W 스냅샷, CT 위상 +1.4° 상수, --rc-high
```

## 9. 사용자에게 필요한 것 / 확인할 것

- 01:10 이후 파일(`beam_projector_2`, `fan_1`, `hair_dryer_1`, `air_conditioner_1`, `laptop_charger_2`)은 `.cal2` 꼬리가 없다 — 같은 수신기 판인지. (열 구성·`cal_applied=1` 은 같아서 같은 자료로 등록했다.)
- 회로 모델 pkl 을 만든 원시 파일(`1788…_raw_*.csv`)의 위치. `data/raw_laptop_charger_5~8` 은 새로 들어왔고 등록부 `RAW12_SNAPSHOT_FILES` 에 적었다.
- 새 보드 펌웨어의 **위상 교정 상수·RC 규약** 한 줄 (등록부 LOW_CAL 블록에 적을 값). 결과는 맞다(§3) — 값만 모른다.
- 짝수차 채널을 되살리는 실험(§4 첫 줄)을 할지 — 캐시+1단계+2단계 ≈ 1시간 GPU, 단일 변수, 대조 파일 필요.
- test_1 라벨(§3b)에서 오븐 끔 195.8초는 ±2초 안에서 평평한 자리다(핫플 펄스 사이 바닥 계단). 채점 허용폭 3초 안이다.
