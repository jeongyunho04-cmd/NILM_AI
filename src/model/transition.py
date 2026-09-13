# -*- coding: utf-8 -*-
"""전이 차분 표현 — 형제가 상수로 빠지는 축 (13.84.21~24).

정상상태 합에서 미니PC 14W 를 읽으려면 형제 75W 를 빼야 하는데, 형제 틀의 오차가
14W 보다 크다(13.84.16: 미니PC/형제 크기비가 전 차수 0.20~0.48, 형제 제자리 편차 h1 112mA
대 미니PC 벡터 전체 47mA). 차분에서는 형제가 **양쪽에 다 있어** 지워진다 — 그래서 제자리
미니PC 지문이 풀 템플릿과 5%·12° 안에서 맞고(13.84.10), 합성으로 배운 분류기가 실측 사건
30개에서 미니PC 를 9/11 맞힌다(13.84.21).

`feat` 은 규모 불변 모양 + 크기 두 개다. 위상은 Δ1 을 기준으로 돌려 **세션 위상 기준**을 지운다.
"""
from typing import Tuple

import numpy as np
from scipy.stats import trim_mean

FS = 60
PRE, POST = 13, 3          # 초. 전이 앞뒤로 이만큼 떨어진 구간을 평균한다
N_FEAT = 47        # 15차 x (크기비·cos·sin) + 크기 둘


def feat(dh: np.ndarray, dp: float) -> np.ndarray:
    """(15,) 복소 Δ 와 ΔP -> (32,) 특징. 규모 불변 모양 + 크기 둘."""
    dh = np.asarray(dh)
    a = np.abs(dh)
    a1 = a[0] + 1e-9
    z1 = dh[0] / a1
    rel = dh * np.conj(z1) ** np.arange(1, len(dh) + 1)
    ang = np.angle(rel)
    return np.concatenate([a / a1, np.cos(ang), np.sin(ang),
                           [np.log10(max(abs(dp), 0.5)), np.log10(a1 * 1e3 + 1e-9)]])


def delta_at(H: np.ndarray, P: np.ndarray, c: int) -> Tuple[np.ndarray, float]:
    """시각 `c`(사이클) 앞뒤 절사평균의 차. 반환은 (복소 15차 Δ, ΔP)."""
    lo, hi = c - PRE * FS, c - POST * FS
    lo2, hi2 = c + POST * FS, c + PRE * FS
    dh = trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0)
    dp = float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2))
    return dh, dp


def feat_at(H: np.ndarray, P: np.ndarray, c: int) -> Tuple[np.ndarray, float]:
    """`delta_at` + 켜짐 방향으로 부호를 맞춰 `feat` 까지. 반환은 (특징, ΔP)."""
    dh, dp = delta_at(H, P, c)
    up = dp > 0
    return feat(dh if up else -dh, dp if up else -dp), dp
