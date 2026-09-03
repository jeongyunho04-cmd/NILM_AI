"""부하 의존 고조파 서명 `s(p)` — 고정 전류 주입 모형의 교체 (12.166).

**무엇이 틀렸나.** 우리는 기기마다 **와트당 페이저 하나**를 쓴다
(`net.harmonic_signatures`). 그것은 "고조파 모양은 부하와 무관하다" 는 가정인데,
PFC 없는 capacitor-input rectifier 에서는 성립하지 않는다. 벌크 캡은 순시 입력
전압이 캡 전압보다 높은 구간에서만 전류를 끌어가므로, **경부하일수록 도통각이
좁아 파형이 뾰족해지고 고차 비율이 커진다.**

우리 실측(train 분할, 충전기 198k / 미니PC 213k / 프로젝터 81k 사이클)에서
직접 확인했다 — h1 정규화 복소 서명 h2~h15 를 28차원 실수로 펴서 SVD:

    PC1 79.3%   PC2 12.2%   PC3 3.6%      <- 사실상 1~2차원 다양체

그리고 **그 축은 기기 정체성이 아니라 부하율이다**:

    충전기 29W ↔ 미니PC 19W  (다른 기기)      9.0°
    충전기 47W ↔ 미니PC 27W  (다른 기기)     12.8°
    충전기 17W ↔ 충전기 69W  (**같은 기기**)  45.7°
    미니PC  10W ↔ 미니PC  27W (**같은 기기**)  45.9°

같은 기기의 부하 차이가 다른 기기와의 차이보다 **3.6~7.3배** 크다. 고정 서명은
각 기기를 이 곡선 위 한 점으로 붕괴시키므로, 평균을 취하면 충전기와 미니PC 가
같은 자리에 놓여 **판별 정보가 사라진다.** 실제 판별자는 "같은 모양이 나오는
전력이 1.5배 다르다" 는 **전력 눈금**이다.

THD 도 캡 입력의 예측대로 단조 감소한다 (충전기 1.68->1.56, 미니PC 1.89->1.76).

**배경 성분.** 곡선 묶음에는 기기가 아닌 `background` 가 하나 더 있다. 실측
"모든 기기 OFF" 창에서 재면 실재한다:

    파일      P       |I1|      k     THD   이 곡선과의 각도
    test_5   4.75W   0.0655   3.05   0.53      20.7°
    test_7   4.76W   0.0654   3.06   0.55      10.7°
    test_8   5.33W   0.0733   3.08   0.52       4.4°

`k = |I1|·V/P ≈ 3.1` 은 변위역률 0.33 을 뜻한다 — 강한 용량성이고 SMPS(k≈1.0)와
물리적으로 다른 성분이다. 우리 `noise_signature`(1.41W, k=1.37)와는 **서명 각도가
60~79°** 로 완전히 다르며 전력도 3.5배 작다. 즉 **우리는 이것을 안 갖고 있었다.**

크기가 중요하다 — 배경 5W 의 |I1| 0.074A 가 **미니PC 9.5W 의 0.050A 보다 크다.**
모델이 이 상시 전류를 설명할 곳이 없으면 가장 싼 기기로 흘리고, 그것이
12.159 가 잰 장소 B 미니PC **−77% 과소평가**의 유력한 정체다.

**한계 (그대로 안고 간다).**
  - 곡선은 측정 콘센트의 `Z_line` 이 박제된 상태다. 기존 고정 서명도 똑같으므로
    새로 생긴 결함은 아니다. 우리 `r_grid` 폭은 0.35~1.8Ω 로 좁은 편이다.
  - 프로젝터는 ON 전력이 48.7±0.66W 로 사실상 고정이라 `discrete` 3점이다.
  - 미니PC 유효 하한이 9.5W 인데 장소 B 의 IDLE 이 9.90W 로 겨우 안쪽이다.
  - 전압 고조파 응답(Norton/FCM)은 없다. 우리 `harm_offset`(12.148)이 그 자리다.
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:                                                    # 단조 보존 보간이 있으면 쓴다
    from scipy.interpolate import PchipInterpolator as _Pchip
except ImportError:                                     # 없으면 선형으로 떨어진다
    _Pchip = None

#: 곡선 묶음의 기본 경로. `files (1)/sp_curves_v2.npz` 를 여기로 들여왔다.
DEFAULT_CURVES = "processed_data/sp_curves.npz"

#: 기기가 아닌 상시 성분의 이름.
BACKGROUND = "background"


def _interp(x: np.ndarray, y: np.ndarray):
    """범위 밖은 끝값으로 고정. 점이 3개 미만이면 선형."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return lambda q: np.full(np.shape(np.atleast_1d(q)), y[0], float)
    if _Pchip is not None and len(x) >= 3:
        f = _Pchip(x, y, extrapolate=False)
        return lambda q: f(np.clip(np.atleast_1d(np.asarray(q, float)), x[0], x[-1]))
    return lambda q: np.interp(np.clip(np.atleast_1d(q), x[0], x[-1]), x, y)


class SPCurve:
    """기기 하나(또는 배경)의 `s(p)` 곡선."""

    def __init__(self, name, P, S, i1, ph1_rad, k, v_ref,
                 discrete=False, standby=None):
        o = np.argsort(np.asarray(P, float))
        self.name = str(name)
        self.P = np.asarray(P, float)[o]
        self.S = np.asarray(S)[o]                 # (K,15) complex, S[:,0] == 1
        self.i1 = np.asarray(i1, float)[o]
        self.ph1 = np.asarray(ph1_rad, float)[o]
        self.k = np.asarray(k, float)[o]
        self.v_ref = float(v_ref)
        self.discrete = bool(discrete)
        self.standby = standby
        self.p_min, self.p_max = float(self.P[0]), float(self.P[-1])
        self._re = [_interp(self.P, self.S[:, h].real) for h in range(self.S.shape[1])]
        self._im = [_interp(self.P, self.S[:, h].imag) for h in range(self.S.shape[1])]
        self._k = _interp(self.P, self.k)
        self._ph = _interp(self.P, self.ph1)

    def signature(self, p: float) -> np.ndarray:
        """h1 정규화 복소 서명 (H,). `s[0] == 1+0j`."""
        if self.standby is not None and p < self.p_min * 0.6:
            return self.standby["S"].copy()
        if self.discrete:
            return self.S[int(np.argmin(np.abs(self.P - float(p))))].copy()
        q = float(np.clip(p, self.p_min, self.p_max))
        s = np.array([self._re[h](q)[0] + 1j * self._im[h](q)[0]
                      for h in range(self.S.shape[1])])
        s[0] = 1.0 + 0j
        return s

    def current(self, p: float, vrms: Optional[float] = None) -> np.ndarray:
        """실제 전류 고조파 (H,) complex [A]. 위상은 전압 기본파 기준.

        `|I1| = k(p)·p / V` — `k` 가 변위역률의 역수라 전력에서 전류를 복원한다.
        """
        if p <= 0:
            return np.zeros(self.S.shape[1], complex)
        v = self.v_ref if vrms is None else float(vrms)
        if self.standby is not None and p < self.p_min * 0.6:
            sb = self.standby
            return sb["S"] * (sb["k"] * p / v) * np.exp(1j * sb["ph1_rad"])
        q = float(np.clip(p, self.p_min, self.p_max))
        return (self.signature(q) * (float(self._k(q)[0]) * p / v)
                * np.exp(1j * float(self._ph(q)[0])))

    def thd(self, p: float) -> float:
        s = self.signature(p)
        return float(np.sqrt(np.sum(np.abs(s[1:]) ** 2)))

    def __repr__(self):
        return (f"<SPCurve {self.name} K={len(self.P)} "
                f"{self.p_min:.1f}~{self.p_max:.1f}W"
                f"{' discrete' if self.discrete else ''}>")


def load_curves(path: str = DEFAULT_CURVES) -> Dict[str, SPCurve]:
    """`{이름: SPCurve}`. 파일이 없으면 빈 딕셔너리 — 호출부가 기존 경로로 간다."""
    p = Path(path)
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    out: Dict[str, SPCurve] = {}
    for n in (str(x) for x in z["device_names"]):
        g = lambda key: z[f"{n}__{key}"]                              # noqa: E731
        sb = None
        if f"{n}__sb_S" in z:
            sb = dict(P=float(g("sb_P")), S=g("sb_S"),
                      k=float(g("sb_k")), ph1_rad=float(g("sb_ph1_rad")))
        out[n] = SPCurve(n, g("P"), g("S"), g("i1"), g("ph1_rad"), g("k"),
                         float(g("V_ref")), bool(g("discrete")), sb)
    return out


def rescale_to_power(c: np.ndarray, p_from: np.ndarray, p_scale: float,
                     curve: "SPCurve") -> np.ndarray:
    """전력을 `p_scale` 배 할 때의 전류 페이저를 `s(p)` 로 옮긴다 (12.166.2).

    기존 증강은 `I <- I · p_scale` 로 **선형** 스케일했다. 그것은 "모양은 부하와
    무관" 이라는 고정 서명 가정이고, 캡 입력 SMPS 에서는 틀리다. 올바른 이동은

        I(p·a) = I(p) · a · [k(p·a)/k(p)] · [s(p·a)/s(p)]

    로, h1 은 `k` 비만, h2 이상은 거기에 모양비가 더 곱해진다.

    Args:
        c: (T,H) complex 원본 페이저
        p_from: (T,) 원본 전력 W
        p_scale: 배율
        curve: 그 기기의 곡선
    Returns:
        (T,H) complex
    """
    c = np.asarray(c)
    p0 = np.asarray(p_from, float)
    # **유효 범위 안에서만 곡선을 쓴다.** 밖에서는 `current()` 가 끝값 클램프나
    # 대기 분기로 빠져 물리적으로 틀린 값을 준다 — 미니PC(하한 9.5W)를 8.8W 에서
    # 0.5배 하면 4.4W 가 되어 대기 분기에 걸리고 |I1| 이 3.0배로 튀었다.
    # 범위 밖 사이클은 기존 선형 스케일 그대로 둔다 (모르는 것은 안 건드린다).
    lo_ok = curve.p_min * (0.6 if curve.standby is not None else 1.0)
    ok = (p0 > 0.0) & (p0 >= lo_ok) & (p0 <= curve.p_max)         & (p0 * p_scale >= lo_ok) & (p0 * p_scale <= curve.p_max)
    if not ok.any() or abs(p_scale - 1.0) < 1e-9:
        return c * p_scale
    H = c.shape[1]
    out = (c * p_scale).astype(np.complex128)
    # 전력이 여러 값을 오가므로 대표 몇 점만 계산해 보간한다 (곡선 자체가
    # 부드러우므로 사이클마다 다시 풀 이유가 없다).
    lo, hi = float(np.min(p0[ok])), float(np.max(p0[ok]))
    grid = np.unique(np.linspace(lo, hi, 16))
    ratio = np.empty((len(grid), H), complex)
    for i, q in enumerate(grid):
        a = curve.current(q, curve.v_ref)[:H]
        b = curve.current(q * p_scale, curve.v_ref)[:H]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.abs(a) > 1e-12, b / np.where(np.abs(a) > 1e-12, a, 1.0),
                         p_scale)
        ratio[i] = r / p_scale          # 선형분은 이미 곱해 뒀다
    idx = np.clip(np.searchsorted(grid, p0[ok]) - 1, 0, max(len(grid) - 2, 0))
    if len(grid) >= 2:
        w = ((p0[ok] - grid[idx]) / np.maximum(grid[idx + 1] - grid[idx], 1e-12))
        w = np.clip(w, 0.0, 1.0)[:, None]
        r = ratio[idx] * (1 - w) + ratio[idx + 1] * w
    else:
        r = np.repeat(ratio[:1], ok.sum(), 0)
    out[ok] = out[ok] * r
    return out.astype(c.dtype)


def background_signature(w: Optional[float] = None, vrms: float = 224.0,
                         n_harm: int = 15, path: str = DEFAULT_CURVES) -> np.ndarray:
    """상시 배경의 페이저 (H,2) Re/Im [A]. 손실의 `noise_sig` 에 더한다 (12.166.3).

    합성이 창마다 `BACKGROUND_W_RANGE` 에서 뽑으므로, 순방향 모형은 그 **중앙**을
    쓴다. `noise_sig` 가 상수 하나라 변동까지는 못 담지만, 지금처럼 **아예 빼놓는
    것보다는 낫다** — 안 넣으면 매 창 |I1| 0.074A 가 설명되지 않은 채 남고 그것이
    가장 싼 SMPS 로 흘러간다.

    Args:
        w: 대표 전력 W. `None` 이면 곡선 범위의 중앙.
    """
    m = load_curves(path).get(BACKGROUND)
    out = np.zeros((n_harm, 2), np.float32)
    if m is None:
        return out
    p = float(0.5 * (m.p_min + m.p_max)) if w is None else float(w)
    c = m.current(p, vrms)[:n_harm]
    out[:len(c), 0] = c.real
    out[:len(c), 1] = c.imag
    return out


def background_power(w: Optional[float] = None, path: str = DEFAULT_CURVES) -> float:
    """`background_signature` 와 짝이 되는 전력 W."""
    m = load_curves(path).get(BACKGROUND)
    if m is None:
        return 0.0
    return float(0.5 * (m.p_min + m.p_max)) if w is None else float(w)
