# -*- coding: utf-8 -*-
"""회로 모델 적합 (12.185) — 잔차 최소제곱 · 겹쳐 시작 · 교차검증 · 식별성 · 계측 전개.

왜 다시 쓰는가
--------------
1. `run_site_voltage_probe.refit` / `run_circuit_element_probe.refit` 는 Nelder-Mead 로
   **스칼라** 손실을 최소화한다. 5~7 차원에서 700~900 반복은 모자라서, 확장 모형이
   **중립값보다 나쁜** 점에서 멈췄다 (충전기 E3 `+alpha`: 기록된 "최적" 0.008759 >
   중립 α=1 에서의 0.008725). 겹친 모형은 기본 모형보다 나쁠 수 없으므로 그 표는
   요소의 증거가 아니라 최적화 실패의 증거다. 12.184.10 의 기각은 다시 세워야 한다.
2. 손실은 잔차의 제곱합이다. 잔차 **벡터**를 그대로 `least_squares`(TRF)에 주면
   야코비를 쓰므로 같은 시간에 훨씬 깊이 내려간다. 손실이 파라미터에 매끄러운 것은
   C_dc 미세 훑기로 확인했다.
3. `to_wave` 는 적합 중에 변하지 않는데 매 평가마다 다시 만들고 있었다 (73728 표본 × 15 차).
   미리 만들어 두면 평가가 4배 빨라진다. 과도는 8주기면 1e-13 으로 잠긴다 (24주기는 낭비).

계측 전개 (`rc_phase_err_deg`)
------------------------------
펌웨어(`nilm_dsp.c`)는 안티앨리어싱 RC(1k·100n, fc 1591.55Hz)의 **크기**만 되돌리고
위상은 "세 채널이 같은 RC 라 V-I 위상에서 상쇄된다" 고 둔다. 같은 차수끼리는 맞는 말이지만
발행 관례가 `arg(I_h) − h·arg(V_1)` 이라 h 차에는 `h·δ1 − δh` 가 남는다
(`δh = atan(60h/fc)`, atan 이 오목하므로 양수). h9 +0.69°, h11 +1.22°, h13 +1.95°, h15 +2.90°.
전류·전압 페이저 **양쪽**에 같은 양이 실려 있고, 순수 지연이 아니라서(h 에 비선형) 시뮬과
비교할 때 상쇄되지 않는다. 자유 파라미터 없이 뺄 수 있다.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.synthesis.circuit_sim import F, NPC, simulate, to_wave

H = 15
#: 가이드 §4.1 의 차수 가중 (h2..h15)
W_H = np.r_[1, 1, 1, 1, 1, .8, .8, .6, .6, .45, .45, .35, .35, .3]
#: 적합에 쓰는 시뮬 주기 수. 8주기면 40주기 값과 1e-13 안에서 같다 (원본 기본값 24는 낭비)
NCYC_FIT = 10
#: 안티앨리어싱 RC 차단 주파수 [Hz] — 펌웨어 `NILM_RC_FC_HZ`
RC_FC_HZ = 1591.55

PAR5_NAMES = ("C_dc", "R", "L", "Cx", "rd")
#: par5 로그 하한/상한 (원본 `LO5`/`HI5` 와 같다)
LO5 = np.log(np.array([10e-6, 0.3, 5e-6, 0.01e-6, 0.1]))
HI5 = np.log(np.array([500e-6, 10.0, 5000e-6, 2e-6, 10.0]))

#: 확장 파라미터: (변환, 하한, 상한, **중립값**, 두 번째 시작값). 중립값에서 모형은 기본 5파라미터와 같다.
#: `Gp` 는 초크 병렬 감쇠의 **컨덕턴스** 1/Rp — 0 이 정확히 "없음" 이라 겹쳐 시작이 성립한다
#: (원본은 Rp 를 로그로 잡아 하한 1Ω 이 이미 강한 감쇠였고, 중립점을 표현할 수 없었다).
EXTRA_SPEC: Dict[str, Tuple[str, float, float, float, float]] = {
    "nvt":   ("log", 1e-4, 5.0, 0.0, 0.05),    # 지수 다이오드 무릎 [V] (0 = 없음).
                                               # 물리 다이오드는 0.03~0.15V — 그보다 크면 "무엇이
                                               # 가장자리를 뭉개는지 모른 채 뭉개는 자리표"로 읽어라 (12.185.6)
    "Gp":    ("lin", 0.0, 0.30, 0.0, 0.03),    # 초크 병렬 감쇠 [S] (0 = 없음, 0.03S = 33Ω)
    "alpha": ("lin", -1.0, 1.0, 1.0, 0.0),     # 직류 부하 지수 (1 = 정전력, 0 정전류, −1 저항)
    "k_hi":  ("lin", 0.3, 1.5, 1.0, 0.75),     # h9 이상 전압 크기 배율 (시뮬 인수가 아니라 입력 정형)
}


# ── 계측 전개 ────────────────────────────────────────────────────────────────
def rc_phase_err_deg(h_max: int = H, fc: float = RC_FC_HZ) -> np.ndarray:
    """관례 `arg(X_h) − h·arg(V_1)` 에 남는 RC 위상 오차 `h·δ1 − δh` [°] (측정 = 참 + 이 값)."""
    h = np.arange(1, h_max + 1, dtype=float)
    d1 = np.arctan(F / fc)
    dh = np.arctan(F * h / fc)
    return np.degrees(h * d1 - dh)


def deembed(X: np.ndarray, fc: float = RC_FC_HZ) -> np.ndarray:
    """페이저에서 RC 위상 오차를 뺀다 (전류·전압 모두 같은 양). 크기는 펌웨어가 이미 되돌렸다."""
    return np.asarray(X, complex) * np.exp(-1j * np.deg2rad(rc_phase_err_deg(len(X), fc)))


def odd_only(V: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(V, complex))
    out[0::2] = np.asarray(V, complex)[0::2]
    return out


# ── 자료점과 손실 설정 ───────────────────────────────────────────────────────
@dataclass
class Point:
    """동작점 하나: 전력, 측정 서명(정규화), 측정 절대 전류, 전압 페이저."""
    p_w: float
    s: np.ndarray          # (15,) complex, h1 정규화
    I: np.ndarray          # (15,) complex [A rms]
    V: np.ndarray          # (15,) complex [V rms]
    vsrc: Optional[np.ndarray] = None    # 미리 만든 파형 (설정마다 다시 만든다)


@dataclass
class LossCfg:
    """손실 설정.

    mode      "norm" h1 정규화 서명 (옛 수치와 비교 가능) · "abs" 절대 전류 (|I1|·∠I1 도 정보)
    odd_only_loss  짝수차를 손실에서 뺀다 (시뮬은 짝수차를 안 내므로 상수항이다 — 전체의 약 1%)
    vmax      전압 입력을 h1..h_vmax 로 제한 (E0 계열)
    k_hi      h9 이상 전압 크기 배율 (extras 로도 줄 수 있다)
    deembed_rc  I·V 에서 RC 위상 오차를 뺀다 (H4)
    """
    mode: str = "norm"
    odd_only_loss: bool = False
    vmax: int = H
    deembed_rc: bool = False


def prepare(points: Sequence[Point], cfg: LossCfg, extras: Optional[Dict[str, float]] = None,
            ncyc: int = NCYC_FIT) -> List[Point]:
    """설정에 맞춰 전압 파형을 미리 만든 자료점 사본. (적합 중에 변하지 않는다.)"""
    extras = extras or {}
    out = []
    for pt in points:
        V = odd_only(pt.V)
        V[cfg.vmax:] = 0
        if cfg.deembed_rc:
            V = deembed(V)
        k_hi = extras.get("k_hi", 1.0)
        if k_hi != 1.0:
            V = V.copy()
            V[8:] = V[8:] * k_hi
        s = deembed(pt.s) if cfg.deembed_rc else pt.s
        I = deembed(pt.I) if cfg.deembed_rc else pt.I
        out.append(Point(pt.p_w, s, I, pt.V, to_wave(V, n=ncyc * NPC)))
    return out


def _weights(cfg: LossCfg) -> np.ndarray:
    """잔차 한 점의 가중 (Re·Im 을 이어 붙인 벡터에 곱한다)."""
    if cfg.mode == "abs":
        w = np.r_[1.0, W_H]                      # h1 포함
    else:
        w = W_H.copy()                           # h2..h15
    if cfg.odd_only_loss:
        h0 = 1 if cfg.mode == "abs" else 2       # w[0] 가 가리키는 차수
        for i in range(len(w)):
            if (h0 + i) % 2 == 0:
                w[i] = 0.0
    return np.r_[w, w]


def residual(par5: Sequence[float], extras: Dict[str, float], pts: Sequence[Point],
             cfg: LossCfg) -> np.ndarray:
    """모든 동작점의 가중 잔차를 이어 붙인 벡터. 발산하면 큰 값으로 채운다."""
    ww = _weights(cfg)
    ex = {k: v for k, v in extras.items() if k in ("nvt", "alpha")}
    if extras.get("Gp", 0.0) > 0.0:
        ex["Rp"] = 1.0 / extras["Gp"]
    out = []
    for pt in pts:
        r = simulate(pt.p_w, *par5, vsrc=pt.vsrc, **ex)
        if not r["ok"] or not np.all(np.isfinite(r["s"])):
            out.append(np.full(len(ww), 1e2))
            continue
        if cfg.mode == "abs":
            scale = max(abs(pt.I[0]), 1e-9)
            d = (r["I"] - pt.I) / scale
        else:
            d = r["s"][1:] - pt.s[1:]
        out.append(np.r_[d.real, d.imag] * ww)
    return np.concatenate(out)


def loss(par5, extras, pts, cfg: LossCfg) -> float:
    """가이드 §4.1 과 같은 자 — 잔차 제곱합의 동작점 평균."""
    d = residual(par5, extras, pts, cfg)
    return float(d @ d) / len(pts)


# ── 파라미터 <-> 최적화 변수 ─────────────────────────────────────────────────
@dataclass
class Model:
    name: str
    extras: Tuple[str, ...] = ()
    fixed: Dict[str, float] = field(default_factory=dict)   # par5 이름 -> 고정값 (H6)

    @property
    def free5(self) -> List[int]:
        return [i for i, n in enumerate(PAR5_NAMES) if n not in self.fixed]


def _bounds(model: Model) -> Tuple[np.ndarray, np.ndarray]:
    lo = [LO5[i] for i in model.free5]
    hi = [HI5[i] for i in model.free5]
    for n in model.extras:
        kind, a, b = EXTRA_SPEC[n][:3]
        lo.append(np.log(a) if kind == "log" else a)
        hi.append(np.log(b) if kind == "log" else b)
    return np.array(lo), np.array(hi)


def _pack(model: Model, par5: Sequence[float], extras: Dict[str, float]) -> np.ndarray:
    x = [np.log(par5[i]) for i in model.free5]
    for n in model.extras:
        kind, a, b = EXTRA_SPEC[n][:3]
        v = extras.get(n, EXTRA_SPEC[n][3])
        x.append(np.log(max(v, a)) if kind == "log" else float(np.clip(v, a, b)))
    return np.array(x, float)


def _unpack(model: Model, x: np.ndarray) -> Tuple[Tuple[float, ...], Dict[str, float]]:
    lo, hi = _bounds(model)
    x = np.clip(x, lo, hi)
    par5 = [model.fixed.get(n, np.nan) for n in PAR5_NAMES]
    for j, i in enumerate(model.free5):
        par5[i] = float(np.exp(x[j]))
    k = len(model.free5)
    extras: Dict[str, float] = {}
    for j, n in enumerate(model.extras):
        kind = EXTRA_SPEC[n][0]
        v = float(np.exp(x[k + j]) if kind == "log" else x[k + j])
        if kind == "log" and v <= EXTRA_SPEC[n][1] * 1.0000001:
            v = EXTRA_SPEC[n][3]                           # 하한 = 중립 (nvt -> 0)
        extras[n] = v
    return tuple(par5), extras


def neutral_extras(model: Model) -> Dict[str, float]:
    return {n: EXTRA_SPEC[n][3] for n in model.extras}


# ── 적합 ─────────────────────────────────────────────────────────────────────
@dataclass
class FitResult:
    par5: Tuple[float, ...]
    extras: Dict[str, float]
    loss: float
    nfev: int
    at_bound: List[str]
    x: np.ndarray

    def as_dict(self) -> Dict:
        d = {n: float(v) for n, v in zip(PAR5_NAMES, self.par5)}
        d.update({f"x_{k}": float(v) for k, v in self.extras.items()})
        d["loss"] = float(self.loss)
        d["at_bound"] = list(self.at_bound)
        return d


def _jac_fd(fun: Callable, x: np.ndarray, f0: np.ndarray, step: np.ndarray,
            lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """전진 차분 야코비 (경계에 붙으면 안쪽으로 뒤로 차분)."""
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        dx = step[i]
        if x[i] + dx > hi[i]:
            dx = -dx
        xp = x.copy()
        xp[i] += dx
        J[:, i] = (fun(xp) - f0) / dx
    return J


def fit(pts: Sequence[Point], model: Model, cfg: LossCfg,
        starts: Sequence[Tuple[Sequence[float], Dict[str, float]]],
        max_nfev: int = 400, seed: int = 0) -> FitResult:
    """다중 시작 TRF 최소제곱. `starts` 의 각 (par5, extras) 에서 내려가 가장 좋은 것을 준다.

    **중립점 보장**: `starts` 에 중립 extras 를 넣어 두면 확장 모형의 결과가 기본 모형보다
    나쁠 수 없다 (`fit_nested` 가 그렇게 부른다).
    """
    from scipy.optimize import least_squares
    lo, hi = _bounds(model)
    step = np.full(len(lo), 2e-3)
    for j, n in enumerate(model.extras):
        if EXTRA_SPEC[n][0] == "lin":
            step[len(model.free5) + j] = 5e-3

    cache: Dict[bytes, np.ndarray] = {}

    def fun(x):
        key = np.asarray(x, float).tobytes()
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            par5, ex = _unpack(model, x)
            cache[key] = residual(par5, ex, pts, cfg)
        return cache[key]

    best: Optional[FitResult] = None
    nfev_tot = 0
    for par5_0, ex0 in starts:
        x0 = np.clip(_pack(model, par5_0, ex0), lo, hi)
        try:
            r = least_squares(fun, x0, bounds=(lo, hi), method="trf",
                              jac=lambda x: _jac_fd(fun, x, fun(x), step, lo, hi),
                              xtol=1e-8, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev)
            x, cost, nfev = r.x, float(r.cost), int(r.nfev)
        except Exception:                                  # 발산·수치 실패는 시작점 하나를 버린다
            x, cost, nfev = x0, float(fun(x0) @ fun(x0)) / 2, 0
        nfev_tot += nfev
        L = 2.0 * cost / len(pts)
        if best is None or L < best.loss:
            par5, ex = _unpack(model, x)
            ab = [n for j, n in enumerate([PAR5_NAMES[i] for i in model.free5] + list(model.extras))
                  if abs(x[j] - lo[j]) < 1e-6 or abs(x[j] - hi[j]) < 1e-6]
            best = FitResult(par5, ex, L, nfev, ab, x)
    assert best is not None
    best.nfev = nfev_tot
    return best


def fit_nested(pts: Sequence[Point], model: Model, cfg: LossCfg, base: FitResult,
               extra_starts: Sequence[Tuple[Sequence[float], Dict[str, float]]] = (),
               max_nfev: int = 400) -> FitResult:
    """기본 모형의 최적점 + 중립 extras 에서 시작한다 — 결과가 기본보다 나쁠 수 없다."""
    starts = [(base.par5, neutral_extras(model))]
    # 중립에서 야코비가 0 인 방향(예: Gp=0 에서 하한, nvt 로그 하한)에 대비해 켠 시작점도 넣는다
    warm = {n: EXTRA_SPEC[n][4] for n in model.extras}
    if warm != neutral_extras(model):
        starts.append((base.par5, warm))
    starts.extend(extra_starts)
    r = fit(pts, model, cfg, starts, max_nfev=max_nfev)
    L0 = loss(base.par5, neutral_extras(model), pts, cfg)
    if L0 < r.loss:                                        # 단조 보장 (여기 걸리면 최적화 결함)
        return FitResult(base.par5, neutral_extras(model), L0, r.nfev, [], _pack(model, base.par5, neutral_extras(model)))
    return r


# ── 교차검증과 식별성 ────────────────────────────────────────────────────────
def loo(pts: Sequence[Point], model: Model, cfg: LossCfg, base: FitResult,
        max_nfev: int = 300) -> Tuple[float, np.ndarray]:
    """Leave-one-out: 한 점을 빼고 맞춘 뒤 그 점의 손실. (평균, 점별)"""
    out = np.empty(len(pts))
    for i in range(len(pts)):
        tr = [p for j, p in enumerate(pts) if j != i]
        r = fit_nested(tr, model, cfg, base, max_nfev=max_nfev)
        out[i] = loss(r.par5, r.extras, [pts[i]], cfg)
    return float(out.mean()), out


def sensitivity(par5, extras, pts, cfg: LossCfg, model: Model) -> Dict:
    """잔차 야코비의 특이값 분해 — 자료가 어느 방향을 정하는가 (H6)."""
    x = _pack(model, par5, extras)
    lo, hi = _bounds(model)
    step = np.full(len(x), 2e-3)

    def fun(z):
        p, e = _unpack(model, z)
        return residual(p, e, pts, cfg)

    f0 = fun(x)
    J = _jac_fd(fun, x, f0, step, lo, hi)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    names = [PAR5_NAMES[i] for i in model.free5] + list(model.extras)
    return {"S": S, "Vt": Vt, "names": names, "cond": float(S[0] / max(S[-1], 1e-30)),
            "J": J, "f0": f0}


def power_check(par5, extras, pts, cfg: LossCfg) -> np.ndarray:
    """시뮬 입력 전력 / 측정 전력. 모델의 P 는 직류 부하 전력, 측정은 교류 입력 전력이다."""
    ex = {k: v for k, v in extras.items() if k in ("nvt", "alpha")}
    if extras.get("Gp", 0.0) > 0.0:
        ex["Rp"] = 1.0 / extras["Gp"]
    return np.array([simulate(p.p_w, *par5, vsrc=p.vsrc, **ex)["p_w"] / p.p_w for p in pts])
