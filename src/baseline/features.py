"""
창 단위 수작업 특징 (Hand-crafted Window Features)
===================================================
CNN 과 **똑같은 33채널 입력** `(B, 33, W)` 을 받아 특징 벡터로 줄인다.
입력이 같아야 "CNN 이 수작업 특징을 이기는가" 라는 비교가 성립한다.

    채널 0:15  고조파 실수부 I_k Re   (k=1..15)
    채널 15:30 고조파 허수부 I_k Im
    채널 30    P (W)   31 Q (VAR)   32 V (Vrms)

[특징을 고른 근거]
1. **크기 무관 고조파비** `asinh(|I_k|/|I_1| * 50)` — 0.2절. 저항성 무리를 가르는
   주 축이다. 오븐 0.0025 / 포트 0.0047 / 핫플 0.014 / 드라이기 0.031 로 계단처럼
   갈리는데, asinh 없이 z-score 하면 SMPS(0.9)와 함께 뭉개진다 (0.3절 b).
2. **절대 고조파 실수/허수부** — 9.1절. 저항 부하는 3·5·7차를 거의 안 올리므로
   1.2kW 오븐이 켜져 있어도 ih3 에 프로젝터 신호가 남는다. 즉 절대 고조파는
   중첩되어도 SMPS 기기 몫이 보존된다. 위상까지 살리려고 Re/Im 을 그대로 쓴다
   (크기만 쓰면 페이저 덧셈 구조가 깨진다).
3. **변동 질감** — 0.4절. 오븐(10~25초 통전)과 핫플레이트(0.9초 통전)는 시간
   패턴으로만 갈린다. 0.5초 블록 20개로 요약해 전이 횟수와 통전율을 잡는다.
4. **전압 강하** — 자기 부하가 클수록 단자 전압이 내려간다. 실측 오븐 펄스마다
   7.7~9.9V. 고부하 존재의 독립 증거다.

[시간 범위를 둘로 나눈 이유]
타깃은 창 전체가 아니라 **한 시점**(기본 539/600)의 전력이다. 창 전체 중앙값만
쓰면 창 중간에 켜지고 꺼진 기기 때문에 타깃 시점과 어긋난다.
그래서 `recent`(타깃 ±1초)와 `full`(창 전체)을 따로 낸다.
"""
from typing import List, Sequence, Tuple
import numpy as np

# 고조파비를 asinh 로 누를 때의 스케일 (설계 문서 1.2절 채널 33~35 과 동일)
RATIO_SCALE = 50.0
# 절대 고조파 전류 asinh 스케일
ABS_SCALE = 20.0
# 변동 질감을 볼 블록 크기 (0.5초). 핫플레이트 주기(약 2초)를 담으려면
# 1초 블록으로는 앨리어싱이 난다.
BLOCK_CYCLES = 30
# recent 범위 반폭 (타깃 ±1초)
RECENT_HALF = 60
# near 범위 반폭 (타깃 ±약 0.17초).
#
# **RECENT_HALF 만으로는 핫플레이트를 놓친다.** ±60 사이클 = 120 사이클인데
# 핫플레이트 릴레이 주기가 정확히 120 사이클이라, 중앙값이 한 주기를 통째로
# 평균해 타깃 시점의 통전 여부를 지워 버린다. 실측:
#     타깃 샘플 P   통전 562.7W / 휴지 12.0W  -> 차이 550.7W
#     ±60 중앙값    통전  80.5W / 휴지 13.2W  -> 차이  67.3W  (정보 소멸)
# 주기보다 훨씬 짧은 범위를 따로 둬야 순시 상태가 남는다.
NEAR_HALF = 10
# 절대 고조파를 따로 낼 차수 (짝수 2차는 드라이기 모터 지문, 3/5/7 은 SMPS)
ABS_ORDERS = (1, 2, 3, 5, 7, 9)


def feature_names(n_harm: int = 15) -> List[str]:
    """`extract()` 가 내는 열 이름. 중요도 해석에 쓴다."""
    names: List[str] = []
    for scope in ("recent", "full"):
        names += [f"ratio_i{k}_{scope}" for k in range(2, n_harm + 1)]
    for k in ABS_ORDERS:
        names += [f"re_i{k}_recent", f"im_i{k}_recent"]
    names += [f"absi{k}_full" for k in ABS_ORDERS]
    names += [
        "p_target", "p_near", "q_near", "irms_near",
        "ratio_i2_near", "ratio_i3_near", "ratio_i5_near", "ratio_i7_near",
        "re_i1_near", "im_i1_near", "re_i3_near", "im_i3_near",
    ]
    names += [
        "p_recent", "p_full_med", "p_p10", "p_p90", "p_std", "p_range",
        "q_recent", "q_full_med", "pf_recent", "thd_recent", "irms_recent", "irms_full",
        "v_recent", "v_min", "v_drop", "v_std",
        "blk_std", "blk_transitions", "blk_duty", "blk_hi_med", "blk_lo_med", "blk_ratio",
        "dp_max", "dp_min", "step_recent", "p_slope",
    ]
    return names


def _med(a: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.median(a, axis=axis)


def extract(x: np.ndarray, target_index: int = 539) -> np.ndarray:
    """(B, 33, W) -> (B, F) float32. 배치 전체를 벡터 연산으로 처리한다.

    Args:
        x: 배치 입력. 단일 창이면 (33, W) 도 받는다.
        target_index: seq2point 타깃 시점. `recent` 범위의 중심이 된다.
    """
    single = x.ndim == 2
    if single:
        x = x[None]
    x = np.asarray(x, dtype=np.float32)
    b, c, w = x.shape
    if c < 33:
        raise ValueError(f"33채널 입력이 필요합니다: {c}")
    n_harm = 15

    re, im = x[:, 0:n_harm], x[:, n_harm:2 * n_harm]
    p, q, v = x[:, 30], x[:, 31], x[:, 32]

    lo = max(0, min(target_index - RECENT_HALF, w - 1))
    hi = min(w, max(target_index + RECENT_HALF, lo + 1))
    sl = slice(lo, hi)

    mag = np.hypot(re, im)                                  # (B, 15, W)
    i1 = mag[:, 0] + 1e-9                                   # (B, W)

    feats: List[np.ndarray] = []

    # ── 1. 크기 무관 고조파비 (recent / full) ────────────────────────────
    ratio = mag[:, 1:] / i1[:, None, :]                     # (B, 14, W)
    for rng in (sl, slice(0, w)):
        feats.append(np.arcsinh(_med(ratio[:, :, rng]) * RATIO_SCALE))

    # ── 2. 절대 고조파 실수/허수부 (위상 보존) ───────────────────────────
    for k in ABS_ORDERS:
        j = k - 1
        feats.append(np.arcsinh(_med(re[:, j, sl])[:, None] * ABS_SCALE))
        feats.append(np.arcsinh(_med(im[:, j, sl])[:, None] * ABS_SCALE))
    for k in ABS_ORDERS:
        j = k - 1
        feats.append(np.arcsinh(_med(mag[:, j, :])[:, None] * ABS_SCALE))

    # ── 2b. 순시 특징 (릴레이 주기보다 짧은 범위) ────────────────────────
    nlo = max(0, min(target_index - NEAR_HALF, w - 1))
    nhi = min(w, max(target_index + NEAR_HALF, nlo + 1))
    nsl = slice(nlo, nhi)
    ti = min(max(target_index, 0), w - 1)
    mag_n = mag[:, :, nsl]
    i1_n = mag_n[:, 0] + 1e-9
    feats += [
        np.arcsinh(p[:, ti] / 100.0)[:, None],
        np.arcsinh(_med(p[:, nsl]) / 100.0)[:, None],
        np.arcsinh(_med(q[:, nsl]) / 100.0)[:, None],
        np.arcsinh(_med(np.sqrt(np.sum(mag_n ** 2, axis=1))) * ABS_SCALE)[:, None],
    ]
    for k in (2, 3, 5, 7):
        feats.append(np.arcsinh(_med(mag_n[:, k - 1] / i1_n) * RATIO_SCALE)[:, None])
    for k in (1, 3):
        feats.append(np.arcsinh(_med(re[:, k - 1, nsl]) * ABS_SCALE)[:, None])
        feats.append(np.arcsinh(_med(im[:, k - 1, nsl]) * ABS_SCALE)[:, None])

    # ── 3. 전력 통계 ─────────────────────────────────────────────────────
    p_rec, p_med = _med(p[:, sl]), _med(p)
    p10, p90 = np.percentile(p, 10, axis=-1), np.percentile(p, 90, axis=-1)
    irms = np.sqrt(np.sum(mag ** 2, axis=1))                # (B, W)
    irms_rec = _med(irms[:, sl])
    v_rec = _med(v[:, sl])
    higher = np.sqrt(np.sum(mag[:, 1:] ** 2, axis=1))
    scal = [
        np.arcsinh(p_rec / 100.0), np.arcsinh(p_med / 100.0),
        np.arcsinh(p10 / 100.0), np.arcsinh(p90 / 100.0),
        np.arcsinh(np.std(p, axis=-1) / 10.0), np.arcsinh((p90 - p10) / 100.0),
        np.arcsinh(_med(q[:, sl]) / 100.0), np.arcsinh(_med(q) / 100.0),
        np.clip(p_rec / (v_rec * irms_rec + 1e-6), -1.5, 1.5),      # PF 근사
        np.arcsinh(_med(higher[:, sl]) / (_med(i1[:, sl]) + 1e-9)),  # THD_i
        np.arcsinh(irms_rec * ABS_SCALE), np.arcsinh(_med(irms) * ABS_SCALE),
        (v_rec - 222.0) / 10.0, (np.min(v, axis=-1) - 222.0) / 10.0,
        (np.max(v, axis=-1) - np.min(v, axis=-1)) / 10.0, np.std(v, axis=-1),
    ]
    feats += [s[:, None] for s in scal]

    # ── 4. 변동 질감 (듀티 부하 판별) ────────────────────────────────────
    nb = w // BLOCK_CYCLES
    blk = _med(p[:, :nb * BLOCK_CYCLES].reshape(b, nb, BLOCK_CYCLES))    # (B, nb)
    b_lo = np.percentile(blk, 10, axis=-1)
    b_hi = np.percentile(blk, 90, axis=-1)
    mid = (b_lo + b_hi) / 2.0
    above = blk > mid[:, None]
    trans = np.abs(np.diff(above.astype(np.int8), axis=-1)).sum(axis=-1)
    span = np.maximum(b_hi - b_lo, 1e-6)
    texture = [
        np.arcsinh(np.std(blk, axis=-1) / 10.0),
        trans.astype(np.float32),                          # 전이 횟수 - 핫플/오븐 지문
        above.mean(axis=-1),                               # 통전율
        np.arcsinh(b_hi / 100.0), np.arcsinh(b_lo / 100.0),
        np.arcsinh(span / 100.0),
    ]
    feats += [t[:, None].astype(np.float32) for t in texture]

    # ── 5. 과도 (돌입/스위칭) ────────────────────────────────────────────
    dp = np.diff(p, axis=-1)
    step = p[:, min(hi - 1, w - 1)] - p[:, lo]
    tt = np.arange(w, dtype=np.float32)
    tt = (tt - tt.mean()) / (tt.std() + 1e-9)
    slope = (p * tt).mean(axis=-1) / 100.0
    feats += [np.arcsinh(np.max(dp, axis=-1) / 100.0)[:, None],
              np.arcsinh(np.min(dp, axis=-1) / 100.0)[:, None],
              np.arcsinh(step / 100.0)[:, None],
              np.arcsinh(slope)[:, None]]

    out = np.concatenate(feats, axis=1).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out[0] if single else out


def sanity_check() -> Tuple[int, List[str]]:
    """열 개수와 이름이 맞는지. 테스트와 CLI 양쪽에서 쓴다."""
    names = feature_names()
    f = extract(np.zeros((2, 33, 600), dtype=np.float32))
    if f.shape[1] != len(names):
        raise AssertionError(f"특징 수 불일치: 배열 {f.shape[1]} vs 이름 {len(names)}")
    return f.shape[1], names
