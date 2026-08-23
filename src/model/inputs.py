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
TARGET_LOOKAHEAD = 360       # 창 끝에서 6초 안쪽 (12.9.12절)

N_HARM = 15
FINE_CHANNELS = 44           # 36 + 추세 제거 전력 2 + 고조파 위상 6 (아래 참조)
WIDE_CHANNELS = 12

# 고조파 위상 불변량의 크기 게이트 (A). |I_h| 가 이 값 근처 아래로 내려가면
# 위상이 잡음이라 채널을 0 으로 죽인다. 계측 바닥의 |I3| 가 0.0043A 이므로
# 그보다 위, 미니PC 의 |I3| ~0.1A 보다는 한참 아래로 잡는다.
PHASE_GATE_A = 0.01

# 12.34 이전 체크포인트가 학습된 세밀 채널 수. 그때는 FINE_CHANNELS 가 38 이었다.
# 체크포인트에 `fine_channels` 키가 없으면 이 값으로 본다.
LEGACY_FINE_CHANNELS = 38

# 추세 제거 전력 채널의 스케일. P 자체(POWER_SCALE=100)보다 작게 잡아야
# 수백 W 리플이 해상된다.
RIPPLE_SCALE = 20.0
RIPPLE_HALF_SHORT = 30       # ±0.5초 - 핫플 릴레이(주기 약 2초) 대역
RIPPLE_HALF_LONG = 150       # ±2.5초 - 더 느린 주기

# 1.2절의 asinh 스케일 상수. 배치마다 재계산하지 않는다.
RATIO_SCALE = 50.0           # 고조파비
CURRENT_SCALE = 20.0         # 절대 고조파 전류
POWER_SCALE = 100.0          # P, Q
V_CENTER, V_SPAN = 222.0, 10.0


def target_index(window_cycles: int) -> int:
    return window_cycles - 1 - TARGET_LOOKAHEAD


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


def build_fine(x: np.ndarray) -> np.ndarray:
    """(B, 33, W) -> (B, FINE_CHANNELS, 600). 창의 **뒤 10초**만 쓴다.

    33채널(15 Re + 15 Im + P + Q + V)에 1.2절의 고조파비 3개를 더해 36채널,
    추세 제거 전력 2채널로 38채널, 고조파 위상 6채널로 44채널이 된다.
    비율은 크기와 무관한 지문이라 여러 기기가 겹쳐도 형태 정보가 남는다.

    **새 채널은 반드시 뒤에 붙인다.** 앞 38채널이 그대로 남아야 옛 체크포인트가
    `fine[:, :38]` 슬라이스만으로 계속 돈다 (net.NILMNet.fine_channels).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.shape[1] < 33:
        raise ValueError(f"33채널이 필요합니다: {x.shape[1]}")
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
    out[:, 0:N_HARM] = np.arcsinh(re * CURRENT_SCALE)
    out[:, N_HARM:2 * N_HARM] = np.arcsinh(im * CURRENT_SCALE)
    out[:, 30] = np.arcsinh(p / POWER_SCALE)
    out[:, 31] = np.arcsinh(q / POWER_SCALE)
    out[:, 32] = (v - V_CENTER) / V_SPAN
    out[:, 33] = np.arcsinh(mag[:, 2] / i1 * RATIO_SCALE)   # |I3|/|I1|
    out[:, 34] = np.arcsinh(mag[:, 4] / i1 * RATIO_SCALE)   # |I5|/|I1|
    out[:, 35] = np.arcsinh(mag[:, 1] / i1 * RATIO_SCALE)   # |I2|/|I1|

    # ── 추세 제거 전력 (12.9.12절) ────────────────────────────────────
    # asinh(P/100) 은 **고부하 위에 얹힌 리플의 대비를 죽인다.** 474W 핫플
    # 리플이 바닥에서는 채널을 2.260 움직이는데, 오븐(1140W) 위에서는 0.347 로
    # 1/6.5 다. asinh 는 저부하 판별을 살리려고 넣은 것인데(0.3/1.2절) 여기서는
    # 정반대로 작용한다. 실측 test_4 의 핫플 미탐이 정확히 그 영역에서 난다.
    #
    # 국소 추세를 빼면 기준선이 0 이든 1600W 든 같은 크기로 들어온다.
    # 광역 갈래에도 `p_dev` 가 있지만 2Hz/0.5초 블록이라 0.4초 휴지가
    # 평균에 지워진다 - 60Hz 에서 해야 한다.
    #
    # **36채널 구성도 만들 수 있어야 한다.** v11 이전 체크포인트를 정정된 정답으로
    # 다시 채점하려면 그때의 입력을 재현해야 하는데, 모듈 상수만 되돌려 놓고
    # 이 두 줄이 무조건 실행되면 IndexError 가 난다.
    if FINE_CHANNELS >= 38:
        out[:, 36] = np.arcsinh((p - _movavg(p, RIPPLE_HALF_SHORT)) / RIPPLE_SCALE)
        out[:, 37] = np.arcsinh((p - _movavg(p, RIPPLE_HALF_LONG)) / RIPPLE_SCALE)

    # ── 고조파 위상 불변량 (2026-08-23, 12.34절) ────────────────────────
    # φ_h = arg(I_h) − h·arg(I_1). 계통 전압 위상과 부하 크기에 따라 도는 절대
    # 위상을 제거하고 **정류기 도통각만 남긴** 값이다.
    #
    # 왜 필요한가: 크기 비율(|I3|/|I1| 등)은 SMPS 3종이 전부 겹친다. 부하 3분위로
    # 재면 기기간 최소 간격이 I3/I1 −0.026, I5/I3 −0.021, I7/I3 −0.029, I9/I3
    # −0.017 로 **넷 다 음수(겹침)** 다. φ3 만 +14.1° 로 양수다:
    #     미니PC −119.2 ~ −58.4  |  프로젝터 −27.1 ~ −26.9  |  충전기 −12.8 ~ −9.2
    # 미니PC 는 부하 따라 60.8° 움직이면서도 남의 영역을 넘지 않는다. 같은 |I1|
    # 에서 비교해도 미니PC 와 충전기가 27~44° 벌어지므로 크기 아티팩트가 아니다.
    #
    # Re/Im 채널(0~29)에 위상이 들어 있긴 하지만 **절대 프레임**이라 부하와 함께
    # 돈다. φ_h 는 arctan 두 번 + h배 곱 + 차의 비선형 조합이라 conv 가 그것을
    # 합성하기를 기대할 근거가 없다. 그래서 직접 준다.
    #
    # 감김 불연속을 피하려고 cos/sin 쌍으로 넣고, |I_h| 가 작을 때는 위상이
    # 잡음이므로 크기 게이트를 곱해 0 으로 죽인다.
    if FINE_CHANNELS >= 44:
        z1 = (re[:, 0] + 1j * im[:, 0]) / i1        # I1 의 단위 페이저
        for slot, h in ((38, 3), (40, 5), (42, 7)):
            j = h - 1
            u = (re[:, j] + 1j * im[:, j]) * np.conj(z1) ** h
            a = np.abs(u) + 1e-12
            w = mag[:, j] / (mag[:, j] + PHASE_GATE_A)
            out[:, slot] = w * (u.real / a)
            out[:, slot + 1] = w * (u.imag / a)
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
    out = np.stack([
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
    ], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_inputs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(B, 33, W) -> ((B, 36, 600), (B, 12, W/30))."""
    return build_fine(x), build_wide(x)
