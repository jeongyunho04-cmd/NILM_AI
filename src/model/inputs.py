"""
2갈래 모델 입력 구성 (Multi-scale Input Builder)
=================================================
합성기가 주는 33채널 60초 창 `(33, 3600)` 을 설계 문서 1.1절의 두 갈래로 나눈다.

    세밀  (36, 600)   뒤 10초 @ 60Hz    고조파 지문, 돌입, 릴레이 위상
    광역  (12, 120)   전체 60초 @ 2Hz   오븐 듀티 주기

12.8.1절에서 이 구성이 60초 전체를 60Hz 로 넣은 것과 **구분되지 않으면서**
입력 샘플은 1/6 이라는 것을 측정했다 (F1 이득의 89% 유지).

[타깃 시점]
60초 창의 끝에서 1초 안쪽 = 인덱스 3539. 세밀 갈래 안에서는 539/600 이다.
세밀 갈래가 뒤 10초이므로 타깃이 그 안에 들어온다 - 이것이 중요하다.
12.7·12.8절에서 두 번 확인했듯, **타깃 근방의 순시 정보가 헤드까지 닿아야 한다.**

[스케일 상수를 고정하는 이유]
1.2절: 배치마다 다시 계산하면 배치 구성에 따라 입력 분포가 흔들린다.
여기서는 물리적으로 의미 있는 고정 상수를 쓴다 (데이터에서 추정한 값이 아니라
설계 문서에 적힌 값). 그래야 다른 장비에서도 같은 전처리가 된다.
"""
from typing import Tuple
import numpy as np

FINE_CYCLES = 600            # 세밀 갈래 길이 (10초 @ 60Hz)
WIDE_HZ = 2.0                # 광역 갈래 해상도
WIDE_BLOCK = int(60 / WIDE_HZ)   # 30 사이클 = 0.5초
TARGET_LOOKAHEAD = 360       # 창 끝에서 6초 안쪽 (12.9.13절)
# 12.45 의 9초 구성으로 되돌리려면 위 둘을 540 / 780 으로 바꾸고
#   캐시 cache/train60_la9, 홀드아웃 processed_data/holdout60_la9 를 쓴다.
#   체크포인트 cnn_la9 는 그 값이 아니면 load_model 이 거부한다.

N_HARM = 15
# 50 이다. 48~49 는 다단 강하 2채널 (12.62절).
#
# 과도 3채널은 12.37 에서 만들고 **반증됐다** - 모델이 실제로 가장 많이
# 쓰는데도(절제 민감도 2위) 실측 유령이 34.9 -> 83.5W 로 악화됐다.
# 그 블록을 50~52 로 옮겨 `>= 53` 가드 뒤에 남겨 두었다 (다시 켜려면 53 으로).
# **번호를 옮긴 이유**: `out` 이 `np.empty` 라 FINE_CHANNELS 아래의 모든 칸은
# ── 세밀 채널 배치 v2 (2026-09-06, 13.12) ─────────────────────────────────
#      0~ 7   Re(I_h)  h=1,3,…,15          홀수 8
#      8~15   Im(I_h)  h=1,3,…,15          홀수 8
#     16~22   |I_h|    h=2,4,…,14          짝수 7 — **위상 없음**
#     23~25   P · Q · V
#     26~28   |I3|/|I1| · |I5|/|I1| · |I2|/|I1|
#     29·30   P추세 ±0.5초 · ±2.5초
#     31~38   cos/sin φ3 · φ5 · φ7 · φ9
#     39·40   PF · |I9|/|I3|
#     41·42   다단강하 3초 · 5.5초
#     43·44   |I2|−|I4| · |I2|  (61주기 평활)
#
# v1(52채널)에서 바뀐 것: 짝수차 Re/Im 14개를 크기 7개로 (위상은 플러그 방향이라 기기 속성이
# 아니다 — `build_fine` 독스트링), 은퇴 블록 둘(12.37 |I3| 과도·12.80 홀수차 크기) 삭제,
# `if FINE_CHANNELS >= N` 가드 삭제(배치가 바뀌면 옛 채널 수를 재현할 수 없다).
#
# ⚠ **앞부분의 뜻이 바뀌어 `fine[:, :N]` 슬라이스가 무의미하다.** `FINE_LAYOUT` 을 캐시 meta 와
#   체크포인트에 적고 로더가 대조해 거부한다.
ODD_ORDERS = (1, 3, 5, 7, 9, 11, 13, 15)
EVEN_ORDERS = (2, 4, 6, 8, 10, 12, 14)
EVEN_MAG0 = 16                #: 짝수차 크기 블록의 시작
PHI_ORDERS = (3, 5, 7, 9)
PHI0 = 31                     #: φ cos/sin 블록의 시작
DROP0 = 41                    #: 다단강하 전방탭 블록의 시작 (12.62)

#: **전압 고조파** (13.26, 2026-09-07). 원시 채널 33~44 = Re(V_h) 6개 + Im(V_h) 6개.
#:
#: 왜 넣나: 선로 임피던스 Z 가 창마다 달라지는데(0.30~2.00Ω) 모델이 그것을 **보정하지 못하고
#: 흔들리기만** 했다 — 참 전력이 Z 에 대해 정확히 0.00W 불변인데 예측이 2.3~7.8W 움직였다.
#: 원인은 관측량 부재였다: `V_h = V_개방,h − Z_h·I_h` 인데 모델은 기본파 실효값 하나만 받아
#: 차수별 강하를 볼 수 없었다. 계측기는 `voltage_harmonics_complex` 로 이미 재고 있다.
#: 능선 탐침의 Z 식별이 **R² 0.192 -> 0.879** (MAE 0.417 -> 0.158Ω) 가 된다.
#:
#: ⚠ **위상이 핵심이다.** Z = R + jωhL 이 복소수라 크기만으로는 강하의 방향을 잃는다
#: (크기만 0.47 대 Re/Im 0.88). 홀수차만 쓰므로 플러그 방향 문제(13.11.7)는 안 걸린다 —
#: 반대로 꽂으면 짝수차만 180° 돈다.
#: ⚠ 차수를 더 넣는다고 좋아지지 않는다: h1~h11 홀수 12채널 0.879 > h1~h15 홀수 16채널 0.872.
#: 낮은 차수가 더 중요하다 (전류가 크니 강하도 크다).
VOLT_ORDERS = (1, 3, 5, 7, 9, 11)
VOLT_RE0 = 33                 #: 원시에서 Re(V_h) 블록의 시작
VOLT_IM0 = VOLT_RE0 + len(VOLT_ORDERS)
#: 합성기·실측 로더가 모델에 넘기는 **원시** 채널 수.
RAW_CHANNELS = VOLT_IM0 + len(VOLT_ORDERS)      # 33 + 12 = 45

FINE_CHANNELS = 45 + 2 * len(VOLT_ORDERS)       # 45 + 12 = 57
FINE_VOLT0 = 45               #: 세밀에서 전압 고조파 블록의 시작
FINE_LAYOUT = "v3"

#: 반파 정류 대비 `|I2|−|I4|` (61주기 평활, 12.114).
HALFWAVE_CH = 43
#: 짝수 2차의 절대량 (61주기 평활, 12.164.16). 오븐 FAN_LIGHT 전용 판별자다.
EVEN2_CH = 44

#: 짝수차에서 오는 채널 — 크기 블록과 `|I2|/|I1|`. `ZERO_EVEN_HARMONICS` 가 이것을 0 으로 만든다.
#: `HALFWAVE_CH`·`EVEN2_CH` 는 **일부러 뺐다** (12.114 와 같은 이유 — 평활·차분이라 다른 통계다).
EVEN_FINE_CHANNELS = list(range(EVEN_MAG0, EVEN_MAG0 + len(EVEN_ORDERS))) + [28]

# ── 광역 갈래 배치 (2026-09-06, 13.12) ────────────────────────────────────
# 12 -> 35. 12.21 이 잰 것: 합성에서 학습한 선형 probe 가 실측에서 **세밀은 AUC 0.32 로
# 뒤집히고 광역은 0.69 를 유지한다.** 전이에서 살아남는 갈래가 광역인데 채널이 12개뿐이었고
# **위상이 하나도 없었다** — 전부 크기·전력이다. 12.34 가 잰 단일 최고 판별자 φ3(d′ 2.15)이
# 전이에 강한 갈래에 빠져 있었다는 뜻이다.
#
#      0~11   옛 12채널 (P·Q·V·|I1|·|I3|·|I5|·|I2|·블록내σ·|I3|/|I1|·|I2|/|I1|·P추세·계단전이율)
#     12~26   |I_h| h=1..15  절대 크기      <- ch3~6 과 4개 겹치지만 앞을 밀지 않는다
#     27~34   φ3·φ5·φ7·φ9 의 cos/sin       <- 세밀과 같은 식, 블록 중앙값
#
# **짝수차 위상은 여기에도 안 넣는다** — 플러그 방향이라 기기 속성이 아니다 (13.11/13.12).
# 크기 블록에는 짝수차도 들어간다 (크기는 면역).
#
# ⚠ **세밀의 `fine_channels` 같은 슬라이스 장치가 없다.** `net.py` 가 이 값을 conv 입력 채널로
#   직결하므로 바꾸면 옛 체크포인트를 아예 못 싣는다. 체크포인트에 `wide_channels` 를 적고
#   로더가 검사한다. 절제는 `--zero-wide-channels` 로 같은 캐시에서 한다.
WIDE_CHANNELS = 35 + 2 * len(VOLT_ORDERS)       # 35 + 12 = 47
WIDE_VOLT0 = 35               #: 광역에서 전압 고조파 블록의 시작 (블록 중앙값, 13.26)

#: 광역의 고조파 크기 블록(12~26)과 위상 블록(27~34). `--zero-wide-channels` 대조용.
WIDE_MAG0 = 12
WIDE_PHI0 = 27
WIDE_MAG_CHANNELS = list(range(WIDE_MAG0, WIDE_MAG0 + N_HARM))
WIDE_PHASE_CHANNELS = list(range(WIDE_PHI0, WIDE_PHI0 + 8))

# 고조파 위상 불변량의 크기 게이트 (A). |I_h| 가 이 값 근처 아래로 내려가면
# 위상이 잡음이라 채널을 0 으로 죽인다. 계측 바닥의 |I3| 가 0.0043A 이므로
# 그보다 위, 미니PC 의 |I3| ~0.1A 보다는 한참 아래로 잡는다.
PHASE_GATE_A = 0.01

# 12.34 이전 체크포인트가 학습된 세밀 채널 수. 그때는 FINE_CHANNELS 가 38 이었다.
# 체크포인트에 `fine_channels` 키가 없으면 이 값으로 본다.
LEGACY_FINE_CHANNELS = 38

# ── 짝수차 고조파 배제 (2026-08-25, 12.77절) ────────────────────────────────
# 12.72 가 짝수차를 **계측 인공물**로 확정했다 — 두 증폭 경로의 DC 오프셋이 개별
# 보정되지 않아 레인지 전환마다 단차가 생기고, 그 1/h 스펙트럼에 펌웨어의
# `s_rc_gain[h] ∝ h` 보상이 곱해져 모든 차수에서 평평한 바닥이 된다. 부하의
# 물리량이 아니고 **같은 기기의 녹화 사이에서 1.3~1.8배씩 흔들린다** (12.72.4).
#
# 12.74 가 추론에서 0 으로 만들어 전이 귀속을 21 -> 24/41 로 올렸고, 12.76 이
# 홀수차 크기 지터와 합쳐 **28/41 (맞바꿈 15->9), 유령 7.90 -> 2.22W** 를 냈다.
#
# **채널을 지우지 않고 0 으로 만든다.** 인덱스를 밀면 "새 채널은 뒤에 붙인다" 규약이
# 깨져 기존 체크포인트(adapt_ph1 44ch, cnn_ov1 48ch)가 전부 무효가 된다. 0 은
# 수학적으로 부재와 같다 — conv 항이 정확히 0 이고 그 가중치의 경사도 0 이다.
# 원시 타깃 슬라이스도 트렁크의 Linear 에 0 으로 들어가 기여가 없다
# (`fp = fine[:, 30]` 은 전력 채널이라 무관하다).
#
# ⚠ **학습과 추론이 반드시 같아야 한다.** 12.74 에서 지터 없이 학습한 모델의
# 짝수차를 추론에서만 껐더니 충전기가 0.937 -> 0.868 로 무너졌다. 그래서 캐시와
# 추론이 함께 지나는 `build_fine` 한 곳에 넣고 체크포인트에 기록해 로더가 검사한다.
#
# ── **2026-09-06 되살렸다 (설계 13.10). 위 12.72~12.77 은 옛 계측기의 기록이다.** ──────────
# 짝수차 인공물의 뿌리는 차동 ADC 의 공통모드 한계였고(VINP>2.65V 에서 위쪽 피크 부풀림),
# 단일 입력 ADC 로 바꾼 새 계측기에는 **없다**:
#
#     순저항 |I2|/|I1|   포트 0.03% · 핫플 0.08% · 오븐 0.18%   (옛 계측기 2.9%, 평평한 바닥)
#     전압   vh2/vh1     0.03%                                   (옛 2.6%)
#
# 인공물이 사라졌으므로 남은 짝수차는 **실신호**다. 그리고 그 신호에는 홀수차가 못 나르는 신원이 있다:
#
#     오븐 팬·조명   |I2|/|I1| 5.8%   h3 에서는 프로젝터에 28.6배, 에어컨에 212배 묻힌다 (12.164.16~18)
#     SMPS 3종       1.3~1.9%         도통 비대칭. 프로젝터 1.5% vs 충전기 1.8% (d′ 1.65 — 가르지는 못한다)
#     드라이기 약풍   43%              반파 정류 (|I2|/|I1| 0.431, 등가저항 106.0Ω)
#
# **사용자 지시(2026-09-06)로 켠다.** 이 값은 `build_fine` 한 곳에만 걸리고 그 출력이 캐시에 구워지므로
# 캐시·1단계·2단계를 다시 돌아야 한다. 체크포인트에 `zero_even_harmonics` 로 기록되고
# `run_gate_check` 가 현재 코드와 다르면 거부한다 (학습과 추론이 어긋나면 조용히 틀린다 — 12.74 에서
# 추론에서만 껐다가 충전기가 0.937 -> 0.868 로 무너졌다).
#
# ⚠ **대조는 같은 캐시에서 낸다.** 이 값을 False 로 두고 구운 캐시에 `--zero-channels`
#   (`EVEN_FINE_CHANNELS` 목록)를 걸면 옛 규약이 그대로 재현된다 — 캐시를 두 번 굽지 않고 단일 변수
#   A/B 가 된다. 반대 방향(True 로 굽고 되살리기)은 불가능하다: 0 으로 구워진 것은 못 되돌린다.
#
# ⚠ ch50(|I2|−|I4|)·ch51(|I2|) 은 **그대로 둔다.** 짝수차 Re/Im 이 살아났어도 그 둘은 평활(61주기)과
#   차분이 들어간 다른 통계다. 지우면 그것 자체가 또 하나의 변수가 된다.
#
# ⚠ 되돌리려면 여기만 True 로 바꾸고 캐시부터 다시 굽는다. `deploy/nilm_runtime/inputs.py` 사본은
#   **새 운영점을 고를 때 같이 복사한다** — 지금 사본을 고치면 옛 체크포인트와 규약이 어긋난다.
ZERO_EVEN_HARMONICS = False


# 12.45 이전 체크포인트가 학습된 타깃 시점 구성. 그 키가 없는 체크포인트는
# 이 값으로 학습된 것으로 본다. **채널 수와 달리 이것은 슬라이스로 못 맞춘다** —
# 타깃 시점이 어긋나면 입력과 라벨이 다른 순간을 가리키므로 조용히 틀린다.
LEGACY_TARGET_LOOKAHEAD = 360
LEGACY_FINE_CYCLES = 600

# 추세 제거 전력 채널의 스케일. P 자체(POWER_SCALE=100)보다 작게 잡아야
# 수백 W 리플이 해상된다.
RIPPLE_SCALE = 20.0
RIPPLE_HALF_SHORT = 30       # ±0.5초 - 핫플 릴레이(주기 약 2초) 대역
RIPPLE_HALF_LONG = 150       # ±2.5초 - 더 느린 주기

# 과도 서술자가 되돌아보는 길이 (12.37). 충전기 돌입이 2.76초에 걸쳐 정착하므로
# 그보다 길어야 첨두와 정착을 한 창에서 본다.
TRANSIENT_LOOKBACK = 180     # 3초
TRANSIENT_BLOCK = 30         # 0.5초. 되돌아보기 최대를 블록 단위로 계산한다

# 다단 강하 전방 탭 (12.62절). 프로젝터 팬 기착의 **두 번째 계단**을 겨냥한다.
#   실측  48.7W -> 5.4~5.6W (3초 유지) -> 2.3W
#   합성  45~49W -> 4.0W (3~5초 유지) -> 0      (창 8/8 에서 재현, 12.60.1)
# 12.59 가 4.5초 단일 탭을 제안했으나 **기착 길이가 3~5초라 단일 탭은 짧은 쪽을
# 놓친다** - 기착이 3초면 t+4.5초는 이미 두 번째 계단 뒤이지만 5초면 아직 앞이라
# 차이가 0 으로 잡힌다. 그래서 3초와 5초로 구간을 물린다.
# **전방 창의 최소/최대 대신 단일 탭 두 개**인 이유: 복합 널 |ΔP| p95 가 13.02W
# (12.53.7)인데 min 은 극값 통계라 3~6초 안에 남이 하나만 꺼져도 그 값을 집는다.
# 단일 탭은 오염이 표본 하나에 그친다.
# 긴 탭은 기착 **밖**이어야 한다. 5.0초로 두면 5초 기착에서 탭이 두 번째 계단
# 경계에 정확히 앉아 차이가 0 이 된다 (구현 뒤 확인했다). 기착 상한이 5초이므로
# 5.5초로 뺀다. 짧은 탭 3.0초는 짧은 기착에서 발화해 **기착 길이** 자체를 준다.
DROP_TAPS = (180, 330)       # 3.0초 / 5.5초. 타깃(239) 에서 419 / 569 - 창(600) 안이다

# 1.2절의 asinh 스케일 상수. 배치마다 재계산하지 않는다.
RATIO_SCALE = 50.0           # 고조파비
CURRENT_SCALE = 20.0         # 절대 고조파 전류

# 반파 대비 채널의 평활 반폭 (12.114). ±0.5초 = 61주기.
# **중앙값이 아니라 평균이다.** `_movavg` 독스트링대로 (B,600,61) 정렬은 캐시
# 생성을 25분 넘게 늘린다. 재 보니 평균으로도 전이 오탐이 다 죽는다 —
# 핫플 릴레이(0.9초 펄스)의 원시 오탐 2.58/3.32/4.33% 가 셋 다 0.00% 가 되고,
# 포트·오븐·선풍기·에어컨·프로젝터·충전기·미니PC 도 전부 0.00% 다.
# 드라이기 약풍은 평균에서 오히려 더 잘 산다 (99.68 -> 100.00%).
HALFWAVE_HALF = 30
POWER_SCALE = 100.0          # P, Q
# ⚠ 2026-09-06: 새 계측기 자료의 vrms 는 D 210.7~219.4 / E 227.5~231.0V 다. 정규화 중심 222 는
#   그 사이에 있어 그대로 둔다 (값을 바꾸면 모든 체크포인트·캐시의 입력 프레임이 갈린다).
V_CENTER, V_SPAN = 222.0, 10.0
#: 전압 **고조파** 크기 (13.26). 실측 |V_h| 는 h3 1.3V · h5 3.7V · h7 1.9V · h9 1.3V · h11 0.5V 라
#: 이 배율에서 asinh 가 거의 선형이다. 기본파(h1, ~215V)는 V_CENTER/V_SPAN 로 따로 정규화한다.
VOLT_HARM_SCALE = 2.0


def target_index(window_cycles: int) -> int:
    return window_cycles - 1 - TARGET_LOOKAHEAD


def wide_target_index(n_blocks: int) -> int:
    """광역 갈래 안에서 **타깃 사이클이 든 블록** (2026-09-08, 13.44).

    사용자: *"우리 모델이 seq2point잖아 그러면 예측하는 그 포인트에 대한 정보를
    주어야하는거 아니야?"* — 맞다. 세밀은 `fine[:, :, fine_target_index()]` 로
    타깃을 정확히 짚는데 **광역에는 타깃 포인터가 없었다.** 헤드로 가는 광역 특징이
    `hw.mean(-1)` 하나뿐이라 순서에 불변이고, 60초 안 어디서 일어난 일인지 모른다.

    `--wide-summary` 가 주는 `hw[:, :, -1]` 은 **창 끝**이라 타깃에서
    `TARGET_LOOKAHEAD`(6초 = 12블록)만큼 어긋난다. 탐침(13.44)에서 창끝 슬라이스가
    타깃 슬라이스보다 실측 평균 0.902 대 0.912 로 나빴다.

    ⚠ **평균을 대체하지 말 것.** 미니PC 는 60초 문맥이 순시값보다 낫다
    (평균 0.795 대 타깃블록만 0.680). 둘 다 줘야 0.937 로 최고가 된다.
    """
    return max(0, min(int(n_blocks) - 1,
                      (int(n_blocks) * WIDE_BLOCK - 1 - TARGET_LOOKAHEAD) // WIDE_BLOCK))


def fine_target_index() -> int:
    """세밀 갈래 안에서의 타깃 위치. 창 길이와 무관하게 539 다."""
    return FINE_CYCLES - 1 - TARGET_LOOKAHEAD


def _movavg(a: np.ndarray, half: int) -> np.ndarray:
    """(B, T) 가장자리 반사 없는 이동평균. cumsum 이라 O(T) 다.

    중앙값이 이상치에 강하지만 (B,600,61) 짜리 정렬이 필요해 캐시 생성이
    25분 넘게 늘어난다. 리플 검출에는 평균으로 충분하다.
    """
    k = 2 * half + 1
    pad = np.pad(a, ((0, 0), (half, half)), mode="edge")
    c = np.zeros((a.shape[0], pad.shape[1] + 1), np.float64)
    np.cumsum(pad, axis=1, out=c[:, 1:])
    return ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)


def _trailing_sustained(a: np.ndarray, lookback: int, block: int) -> np.ndarray:
    """(B, T) 각 시점에서 **직전 lookback 의 절반 이상 동안 넘었던** 값.

    0.5초 블록의 최대를 먼저 내고, 블록 여섯 개(3초)의 **중앙값**을 취한다.
    시점마다 sliding window 를 돌리면 300,000창 x 600 x 180 이라 못 쓴다.

    **최대값이 아니라 중앙값이어야 한다** (12.37.3). 최대값을 쓰면 한두 사이클짜리
    스파이크가 3초간 유지돼 지속형 돌입과 구분이 안 된다. 실측이 그것을 보여준다 -
    통전 1.5초 뒤 서술자 값이

        최대값:  포트 0.979  오븐 0.881  드라이기 0.617  |  충전기 0.107  프로젝터 0.009
        중앙값:  포트 -0.010 오븐 0.000  드라이기 0.001  |  충전기 0.034  프로젝터 -0.085

    저항 부하의 |I3| 스파이크는 크지만 **0.02초**뿐이고(포트 +0.174A, 오븐 +0.023A),
    충전기 돌입은 **2.68초** 지속된다. 최대값으로는 포트가 충전기의 9.2배가 되어
    'SMPS 돌입' 이 아니라 '큰 저항 부하가 방금 켜졌다' 를 재게 된다.
    중앙값으로 바꾸면 저항 오염이 사라지고 충전기(+)와 프로젝터(-)가 부호로 갈린다.
    """
    b, t = a.shape
    nb = t // block
    blk = a[:, :nb * block].reshape(b, nb, block).max(axis=2)      # (B, nb)
    k = max(1, lookback // block)
    lag = np.stack([np.concatenate(
        [np.repeat(blk[:, :1], i, axis=1), blk[:, :nb - i]], axis=1) if i else blk
        for i in range(k)], axis=0)                                # (k, B, nb)
    agg = np.median(lag, axis=0)
    out = np.repeat(agg, block, axis=1)
    if out.shape[1] < t:
        out = np.pad(out, ((0, 0), (0, t - out.shape[1])), mode="edge")
    return out[:, :t]


def _shift_back(a: np.ndarray, n: int) -> np.ndarray:
    """(B, T) 를 n 샘플 과거로 민다. 창 앞은 가장자리 값으로 채운다."""
    if n <= 0:
        return a
    return np.concatenate([np.repeat(a[:, :1], n, axis=1), a[:, :-n]], axis=1)


def _shift_fwd(a: np.ndarray, n: int) -> np.ndarray:
    """(B, T) 를 n 샘플 미래로 민다 - `a[:, t]` 자리에 `a[:, t+n]` 이 온다.

    `_shift_back` 의 반대다. 창 뒤는 가장자리 값으로 채운다. 세밀 갈래의 타깃은
    239 번이고 뒤로 360 사이클(6초)이 남으므로, 최대 탭 300 은 창 안에서 해소된다.
    """
    if n <= 0:
        return a
    return np.concatenate([a[:, n:], np.repeat(a[:, -1:], n, axis=1)], axis=1)


def build_fine(x: np.ndarray) -> np.ndarray:
    """(B, 33, W) -> (B, 45, 600). 창의 **뒤 10초**만 쓴다. 배치 v2 (13.12).

    ```
     0~ 7   Re(I_h)  h=1,3,…,15          홀수 8
     8~15   Im(I_h)  h=1,3,…,15          홀수 8
    16~22   |I_h|    h=2,4,…,14          짝수 7 — **위상 없음** (아래)
    23~25   P · Q · V
    26~28   |I3|/|I1| · |I5|/|I1| · |I2|/|I1|
    29·30   P − 이동평균(±0.5초) · (±2.5초)
    31~38   cos/sin φ3 · φ5 · φ7 · φ9
    39·40   역률 PF · |I9|/|I3|
    41·42   다단강하 전방탭 3초 · 5.5초
    43·44   |I2|−|I4| · |I2|  (61주기 평활)
    ```

    **짝수차에는 위상을 주지 않는다 (13.11/13.12).** 플러그를 반대로 꽂으면 측정 전압·전류의
    부호가 같이 바뀌고 위상 기준(전압 영교차)이 반주기 민다 — `I_h -> −(−1)^h I_h` 라
    **짝수차만 180° 돈다.** 드라이기 약풍의 격리 녹화 대 복합 녹화에서 h2 가 −6.63° 대
    +172.60°, h4 가 −177.52° 대 −0.74° 였고 홀수차는 10.7° 안이었다. 크기는 926 대 873mA 로
    같다. 즉 짝수차 위상은 기기 속성이 아니라 그날 꽂은 방향이다.

    ⚠ **배치가 v1(52채널)과 다르다.** 앞부분의 뜻이 바뀌었으므로 `fine[:, :N]` 슬라이스로
      옛 체크포인트를 돌릴 수 없다. `FINE_LAYOUT` 을 캐시·체크포인트에 적고 로더가 거부한다.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.shape[1] < RAW_CHANNELS:
        raise ValueError(
            f"원시 {RAW_CHANNELS}채널이 필요합니다: {x.shape[1]}개를 받았습니다. "
            "33채널은 전압 고조파가 없던 옛 배치(FINE_LAYOUT v2)입니다 — 캐시를 다시 구우십시오 (13.26)")
    seg = x[:, :, -FINE_CYCLES:]
    if seg.shape[2] < FINE_CYCLES:
        pad = FINE_CYCLES - seg.shape[2]
        seg = np.pad(seg, ((0, 0), (0, 0), (pad, 0)), mode="edge")

    re, im = seg[:, 0:N_HARM], seg[:, N_HARM:2 * N_HARM]
    p, q, v = seg[:, 30], seg[:, 31], seg[:, 32]
    mag = np.hypot(re, im)
    i1 = mag[:, 0] + 1e-9

    b = seg.shape[0]
    out = np.empty((b, FINE_CHANNELS, FINE_CYCLES), np.float32)

    # ── 0~15: 홀수차 Re/Im ─────────────────────────────────────────────
    for s, h in enumerate(ODD_ORDERS):
        out[:, s] = np.arcsinh(re[:, h - 1] * CURRENT_SCALE)
        out[:, len(ODD_ORDERS) + s] = np.arcsinh(im[:, h - 1] * CURRENT_SCALE)

    # ── 16~22: 짝수차 **크기** (위상 없음, 13.12) ──────────────────────
    for s, h in enumerate(EVEN_ORDERS):
        out[:, EVEN_MAG0 + s] = np.arcsinh(mag[:, h - 1] * CURRENT_SCALE)

    out[:, 23] = np.arcsinh(p / POWER_SCALE)
    out[:, 24] = np.arcsinh(q / POWER_SCALE)
    out[:, 25] = (v - V_CENTER) / V_SPAN
    out[:, 26] = np.arcsinh(mag[:, 2] / i1 * RATIO_SCALE)   # |I3|/|I1|
    out[:, 27] = np.arcsinh(mag[:, 4] / i1 * RATIO_SCALE)   # |I5|/|I1|
    out[:, 28] = np.arcsinh(mag[:, 1] / i1 * RATIO_SCALE)   # |I2|/|I1|

    # ── 29·30 추세 제거 전력 (12.9.12) ─────────────────────────────────
    # asinh(P/100) 은 고부하 위에 얹힌 리플의 대비를 죽인다. 474W 핫플 리플이 바닥에서는
    # 채널을 2.260 움직이는데 오븐(1140W) 위에서는 0.347 로 1/6.5 다. 국소 추세를 빼면
    # 기준선이 0 이든 1600W 든 같은 크기로 들어온다. 광역의 `p_dev` 는 0.5초 블록이라
    # 0.4초 휴지를 지운다 — 60Hz 에서 해야 한다.
    out[:, 29] = np.arcsinh((p - _movavg(p, RIPPLE_HALF_SHORT)) / RIPPLE_SCALE)
    out[:, 30] = np.arcsinh((p - _movavg(p, RIPPLE_HALF_LONG)) / RIPPLE_SCALE)

    # ── 31~38 고조파 위상 불변량 φ_h (12.34) ───────────────────────────
    # φ_h = arg(I_h) − h·arg(I_1). 계통 전압 위상과 부하 크기에 따라 도는 절대 위상을
    # 제거하고 **정류기 도통각만 남긴** 값이다. 크기 비율은 SMPS 3종이 전부 겹치는데
    # (기기간 최소 간격이 넷 다 음수) φ3 만 +14.1° 로 양수다. Re/Im 에 위상이 들어 있긴
    # 하지만 **절대 프레임**이라 부하와 함께 돌고, φ_h 는 arctan 두 번 + h배 곱 + 차의
    # 비선형 조합이라 conv 가 합성하기를 기대할 근거가 없다. 그래서 직접 준다.
    # 감김 불연속을 피해 cos/sin 쌍으로 넣고, |I_h| 가 작으면 위상이 잡음이라 게이트로 죽인다.
    #
    # **홀수차만 준다** — 짝수차 위상은 플러그 방향이다 (13.12, 위 독스트링).
    z1 = (re[:, 0] + 1j * im[:, 0]) / i1
    for s, h in enumerate(PHI_ORDERS):
        j = h - 1
        u = (re[:, j] + 1j * im[:, j]) * np.conj(z1) ** h
        a = np.abs(u) + 1e-12
        w = mag[:, j] / (mag[:, j] + PHASE_GATE_A)
        out[:, PHI0 + 2 * s] = w * (u.real / a)
        out[:, PHI0 + 2 * s + 1] = w * (u.imag / a)

    # ── 39·40 역률과 고차 형상 (12.36) ─────────────────────────────────
    # 탐욕 전진 선택으로 고른 조합이다 (최악 쌍 = 프로젝터↔충전기):
    #   |I9|/|I3| 만 1.73 -> +φ3 3.25 -> +PF 3.77. 원시 와트당 벡터가 1.12 였으니 3.4배다.
    # PF 는 33채널에 없지만 있는 것만으로 정확히 복원된다 (상관 0.9975~1.0000).
    i_rms = np.sqrt((mag ** 2).sum(axis=1))
    out[:, 39] = np.clip(p / np.maximum(v * i_rms, 1e-6), 0.0, 1.2)
    # 분모 바닥 0.1mA — 계측 바닥의 |I3| 가 4.3mA 라 정상 구간은 안 건드리고 폭주만 막는다.
    out[:, 40] = np.arcsinh(mag[:, 8] / (mag[:, 2] + 1e-4) * RATIO_SCALE)

    # ── 41·42 다단 강하 전방 탭 (12.62) ────────────────────────────────
    # 프로젝터와 충전기를 실제로 가르는 첫 특징이다. 소등 뒤 팬이 몇 초 더 도는 기착이
    # 프로젝터에만 있다: 1단은 41~44W 로 구별 불가인데 **2단은 30배 차이에 겹침이 0** 이다
    # (충전기 0.01~0.10W, 프로젝터 3.0~14.4W). 깊은 층 타깃 슬라이스의 수용영역이 ±1.56초라
    # +3~5초가 위치 없이만 닿고, 36~37 의 대칭 이동평균은 전방 계단을 뒤쪽과 섞어 지운다.
    # 그래서 전방 차분을 미리 계산해 준다. 기착 길이가 3~5초라 탭을 둘 물린다.
    for s, tap in enumerate(DROP_TAPS):
        out[:, DROP0 + s] = np.arcsinh((p - _shift_fwd(p, tap)) / RIPPLE_SCALE)

    # ── 43 반파 정류 대비 |I2| − |I4| (12.114) ─────────────────────────
    # 드라이기 약풍은 **다이오드 반파 정류**다 (P 가 정확히 절반, |I2|/|I1| 0.431).
    # 반파는 차수 프로파일이 가파르고(0.424 : 0.085 : 0.036 : 0.020) 계측 잡음은 평평해서
    # 차분이 반파만 남긴다. 격리에서 (61주기 평활 후) 약풍 0.670A vs 나머지 전 기기·전 상태
    # ≤0.014A, 겹침 0 이다. 핫플(101.8Ω)과 드라이기 약(겉보기 102.6Ω)은 저항 정합의 유일한
    # 모호쌍인데 여기서 0.0001 vs 0.670 으로 갈린다.
    #
    # **16~22 의 |I2|·|I4| 와 중복이 아니다** — 이쪽은 61주기 이동평균이라 √61 ≈ 7.8배
    # 잡음이 낮고, 차분은 conv 가 합성해야 하는 비선형 조합이다 (12.34 와 같은 논리).
    out[:, HALFWAVE_CH] = np.arcsinh(
        _movavg(mag[:, 1] - mag[:, 3], HALFWAVE_HALF) * CURRENT_SCALE)

    # ── 44 |I2| 절대량 (12.164.16) ─────────────────────────────────────
    # 오븐 팬·조명(14.2W / 67mA)의 신원은 **h2 에만** 남아 있다. 15W 로 맞춰 비교하면 h1 은
    # 9종이 모두 64~71mA 라 크기만 말하고 신원은 말하지 않는데, h2/h1 은 FAN_LIGHT 9.33% vs
    # 저항 4종 0.06~0.35% / SMPS 3종 0.28~1.42% 다. 차수별 전류 예산으로도 **h2 에서만
    # 안 묻힌다** — h3 에서는 프로젝터 28.6배, 에어컨 212배다.
    # 차분(ch43)이 아니라 절대량인 이유: FAN_LIGHT 의 짝수차는 오히려 평평해서
    # (6.28/4.31/4.19) 차분이 1.97mA 로 신호를 지운다.
    out[:, EVEN2_CH] = np.arcsinh(_movavg(mag[:, 1], HALFWAVE_HALF) * CURRENT_SCALE)

    # ── 45~56 전압 고조파 Re/Im, 홀수 h=1,3,5,7,9,11 (13.26) ───────────
    # 선로 임피던스 Z 를 드러내는 유일한 관측량이다: V_h = V_개방,h − Z_h·I_h.
    # 기본파는 ~215V 라 V_CENTER/V_SPAN 로, 고차는 0.5~4V 라 asinh 로 정규화한다.
    for s, h in enumerate(VOLT_ORDERS):
        vr, vi = seg[:, VOLT_RE0 + s], seg[:, VOLT_IM0 + s]
        if h == 1:
            out[:, FINE_VOLT0 + s] = (vr - V_CENTER) / V_SPAN
            out[:, FINE_VOLT0 + len(VOLT_ORDERS) + s] = vi / V_SPAN
        else:
            out[:, FINE_VOLT0 + s] = np.arcsinh(vr * VOLT_HARM_SCALE)
            out[:, FINE_VOLT0 + len(VOLT_ORDERS) + s] = np.arcsinh(vi * VOLT_HARM_SCALE)

    if ZERO_EVEN_HARMONICS:
        out[:, EVEN_FINE_CHANNELS] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_wide(x: np.ndarray) -> np.ndarray:
    """(B, 33, W) -> (B, 12, W/30). 창 **전체**를 2Hz 로 요약한다.

    1.3절의 12채널을 따르되, `EMA(P,1분) − EMA(P,20분)` 은 뺐다. 창이 60초라
    20분 문맥이 물리적으로 없다 (1.3절 경고 참조). 창 안에서 계산 가능한
    `P − 10초 이동중앙값` 으로 대체했다 — 같은 "국소 편차" 역할이다.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.shape[1] < RAW_CHANNELS:
        raise ValueError(
            f"원시 {RAW_CHANNELS}채널이 필요합니다: {x.shape[1]}개를 받았습니다 (13.26)")
    b, _, w = x.shape
    nb = w // WIDE_BLOCK
    if nb < 2:
        raise ValueError(f"창이 너무 짧습니다: {w} 사이클")
    cut = nb * WIDE_BLOCK

    re, im = x[:, 0:N_HARM, :cut], x[:, N_HARM:2 * N_HARM, :cut]
    mag = np.hypot(re, im)
    p, q, v = x[:, 30, :cut], x[:, 31, :cut], x[:, 32, :cut]

    blk = lambda a: a.reshape(a.shape[0], nb, WIDE_BLOCK)
    med = lambda a: np.median(blk(a), axis=2)

    p_b, q_b, v_b = med(p), med(q), med(v)
    i_b = {k: med(mag[:, k - 1]) for k in (1, 2, 3, 5)}
    p_std = blk(p).std(axis=2)                      # 블록 내 변동 - 듀티 부하 판별

    # P - 10초 이동중앙값. 1.3절의 EMA 차 대체 (창 안에서 계산 가능한 국소 편차)
    win = max(1, int(10 * WIDE_HZ))
    pad = np.pad(p_b, ((0, 0), (win // 2, win - win // 2 - 1)), mode="edge")
    roll = np.median(np.lib.stride_tricks.sliding_window_view(pad, win, axis=1), axis=2)
    p_dev = p_b - roll[:, :nb]

    # 계단 전이율: 블록 간 |ΔP| 가 창 진폭의 20% 를 넘는 비율 (10블록 이동)
    dp = np.abs(np.diff(p_b, axis=1, prepend=p_b[:, :1]))
    thr = 0.2 * np.maximum(p_b.max(1) - p_b.min(1), 1.0)[:, None]
    step = np.pad((dp > thr).astype(np.float32), ((0, 0), (win - 1, 0)), mode="edge")
    step_rate = np.lib.stride_tricks.sliding_window_view(step, win, axis=1)[:, :nb].mean(2)

    i1s = i_b[1] + 1e-9
    chans = [
        np.arcsinh(p_b / POWER_SCALE),
        np.arcsinh(q_b / POWER_SCALE),
        (v_b - V_CENTER) / V_SPAN,
        np.arcsinh(i_b[1] * CURRENT_SCALE),
        np.arcsinh(i_b[3] * CURRENT_SCALE),
        np.arcsinh(i_b[5] * CURRENT_SCALE),
        np.arcsinh(i_b[2] * CURRENT_SCALE),
        np.arcsinh(p_std / 10.0),
        np.arcsinh(i_b[3] / i1s * RATIO_SCALE),
        np.arcsinh(i_b[2] / i1s * RATIO_SCALE),
        np.arcsinh(p_dev / POWER_SCALE),
        step_rate,
    ]

    # ── 12~26 전 차수 크기 (2026-09-06, 13.12) ─────────────────────────
    # 광역은 h5 에서 끊겨 있었다 (+h2). 세밀은 15차까지 다 본다 — 전이에서 살아남는 갈래에
    # 고차가 없으면 SMPS 신원의 절반을 광역이 못 나른다. 짝수차도 **크기라서** 넣는다.
    if WIDE_CHANNELS >= WIDE_MAG0 + N_HARM:
        chans += [np.arcsinh(med(mag[:, h - 1]) * CURRENT_SCALE)
                  for h in range(1, N_HARM + 1)]

    # ── 27~34 위상 불변량 φ3·φ5·φ7·φ9 (2026-09-06, 13.12) ─────────────
    # **광역에는 위상이 하나도 없었다.** 세밀 PHI0 블록과 같은 식을 쓰고 블록 중앙값만 다르다:
    # 주기마다 불변량을 구해 크기 게이트를 곱한 뒤 cos/sin 을 **따로** 중앙값 낸다.
    # 각도를 직접 중앙값 내면 감김 불연속에서 틀린다.
    # 홀수차만 준다 — 짝수차 위상은 플러그 방향이다 (13.11).
    if WIDE_CHANNELS >= WIDE_PHI0 + 2 * len(PHI_ORDERS):
        z1 = (re[:, 0] + 1j * im[:, 0]) / (mag[:, 0] + 1e-9)
        for h in PHI_ORDERS:
            j = h - 1
            u = (re[:, j] + 1j * im[:, j]) * np.conj(z1) ** h
            a = np.abs(u) + 1e-12
            g = mag[:, j] / (mag[:, j] + PHASE_GATE_A)
            chans += [med(g * (u.real / a)), med(g * (u.imag / a))]

    # ── 35~46 전압 고조파 Re/Im, 홀수 h=1,3,5,7,9,11 (13.26) ───────────
    # Z 는 창 안에서 상수라 **광역이 자연스러운 자리다** — 2Hz 로 60초를 다 본다.
    # 세밀(45~56)과 같은 양이고 배치도 같다 (Re 6개 -> Im 6개), 블록 중앙값이라 잡음만 낮다.
    def _vnorm(a, h):
        return (a - V_CENTER) / V_SPAN if h == 1 else np.arcsinh(a * VOLT_HARM_SCALE)

    chans += [_vnorm(med(x[:, VOLT_RE0 + s, :cut]), h) for s, h in enumerate(VOLT_ORDERS)]
    chans += [(med(x[:, VOLT_IM0 + s, :cut]) / V_SPAN if h == 1
               else np.arcsinh(med(x[:, VOLT_IM0 + s, :cut]) * VOLT_HARM_SCALE))
              for s, h in enumerate(VOLT_ORDERS)]

    out = np.stack(chans, axis=1).astype(np.float32)
    assert out.shape[1] == WIDE_CHANNELS, (out.shape[1], WIDE_CHANNELS)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_inputs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(B, 33, W) -> ((B, 36, 600), (B, 12, W/30))."""
    return build_fine(x), build_wide(x)
