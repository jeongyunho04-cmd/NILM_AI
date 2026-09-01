"""
전력·전압 의존 지문 `sig(P, V)` (12.122.9, 2026-09-01)
========================================================
`net.harmonic_signatures` 는 기기당 **와트당 페이저 하나**를 낸다 — 통전 구간의
중앙값이다. 그런데 그 형상이 전력에 따라 크게 변한다:

    충전기  h=11 에서 −1.68 %/W. 작동 범위 57W 를 곱하면 **95%**
    미니PC  h=11 에서 −3.65 %/W

그리고 12.122.3 이 그 지문을 **고전력 상태**에서 만든다 — 충전기가 프로젝터와
형상이 같아지는 바로 그 상태다 (h5 차이 0.3%. 저전력에서는 15.5%).

[왜 층화가 아니라 회귀인가]
격리 녹화에서 **corr(P, V) = +0.40** 이다 (충전기). 전력 분위로만 자르면
전압 효과가 섞여 든다 — 12.122.3 의 `sig(P)` 가 그래서 오염됐고, 그것을
고정점 반복에 넣었다가 더 나빠졌다.

[모형]
차수마다 둘을 따로 적합한다. 크기는 곱셈으로 변하므로 로그 선형이다.

    log|c_h|  = a + b·(P−P0) + c·(V−V0)
    u_h       = a'+ b'·(P−P0) + c'·(V−V0)      u_h = arg(c_h · conj(z1)^h)

`u_h` 는 **위상 불변량**이다 (12.34 의 그것). 절대 위상은 계통 프레임과 함께
도는데 불변량은 정류기 도통각만 남긴다. `h=1` 은 정의상 0 이므로 `arg(c_1)` 을
따로 적합해 프레임을 복원한다.

[적합 품질 — 2026-09-01]

```
                크기 R^2        위상 SD (상수 -> 회귀)
laptop_charger  0.39~0.96      h3  2.3° -> 1.0°   h15 41.7° -> 15.6°
minipc          0.71~0.89      h3 20.7° -> 5.2°   h15 110°  -> 22.8°
beam_projector  0.05~0.21      이미 상수로 충분 (P 가 46~49W 로 안 변한다)
저항 3종         약하다          고조파 자체가 작다
```

**설명력이 없는 곳에서는 상수로 물러선다** (`MIN_R2`, `MIN_SAMPLES`). 회귀가
상수보다 나쁠 수 있는 자리를 만들지 않는다.

[규칙 1 — 이것은 격리 통계다]
전부 격리 녹화에서 적합한다. 복합에서 같은 관계가 성립한다는 가정이 들어간다.
**판정은 실측 재채점으로 해야 한다.**

[규칙 14 — 안 잡는 것]
RMS 전압만 쓴다. **공급 파형 왜곡(평탄화)은 못 잡는다** — npz 에 전압 고조파가
없다. 저항 부하가 만드는 −5V 강하의 RMS 성분은 잡고 파형 성분은 못 잡는다.
"""
from typing import Dict, Optional, Sequence
import numpy as np

#: 회귀를 쓸 최소 표본. 이보다 적으면 상수 중앙값으로 물러선다.
MIN_SAMPLES = 200

#: 회귀를 쓸 최소 설명력. 크기·위상 각각에 따로 건다.
MIN_R2 = 0.15

#: 적합에 쓰는 창 (사이클). 1초 — 짧을수록 (P,V) 가 창 안에서 안 섞인다.
FIT_WINDOW = 60


class SigModel:
    """기기별 `c_h(P, V)` (와트당 복소 페이저). `predict` 가 (H,) 복소를 낸다.

        m = SigModel.fit(pool, appliances)
        c = m.predict("laptop_charger", P=25.0, V=216.0)     # (15,) complex
        A = m.matrix(appliances, P_vec, V)                   # (K,H) complex
    """

    def __init__(self, appliances: Sequence[str], n_harm: int = 15):
        self.appliances = list(appliances)
        self.n_harm = int(n_harm)
        self.const: Dict[str, np.ndarray] = {}          # (H,) 복소 — 물러설 자리
        self.coef: Dict[str, dict] = {}                 # 차수별 회귀 계수
        self.center: Dict[str, tuple] = {}              # (P0, V0)
        self.span: Dict[str, tuple] = {}                # (Pmin, Pmax, Vmin, Vmax)

    # ── 적합 ────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, pool, appliances: Sequence[str], n_harm: int = 15,
            window: int = FIT_WINDOW, use_voltage: bool = True,
            only: Optional[Sequence[str]] = None) -> "SigModel":
        """세그먼트 풀의 격리 활성 구간에서 적합한다.

        Args:
            use_voltage: False 면 전압항을 빼고 `sig(P)` 만 만든다.
                **격리 적합 V 범위가 복합과 안 겹치는 기기가 있다** —
                에어컨·선풍기 0%, 프로젝터 21%, 포트 31%, 핫플 26%
                (2026-09-01). 그런 기기에는 전압항이 외삽으로 들어간다.
            only: 회귀를 걸 기기. 나머지는 상수 그대로 둔다. 저항 3종은
                고조파 자체가 작고 R^2 가 낮아 회귀가 잡음을 태운다.
        """
        m = cls(appliances, n_harm)
        keep = None if only is None else set(only)
        for app in appliances:
            C, P, V = _collect(pool, app, window)
            if len(C) == 0:
                m.const[app] = np.zeros(n_harm, complex)
                continue
            m.const[app] = np.median(C.real, 0) + 1j * np.median(C.imag, 0)
            m.span[app] = (float(P.min()), float(P.max()), float(V.min()), float(V.max()))
            if len(C) < MIN_SAMPLES or (keep is not None and app not in keep):
                continue
            m.center[app] = (float(np.median(P)), float(np.median(V)))
            m.coef[app] = _fit_one(C, P, V, m.center[app], n_harm,
                                   use_voltage=use_voltage)
        return m

    # ── 예측 ────────────────────────────────────────────────────────────
    def predict(self, app: str, P: float, V: float) -> np.ndarray:
        """(H,) 복소 와트당 페이저. 회귀가 없으면 상수를 낸다."""
        c0 = self.const.get(app)
        if c0 is None:
            return np.zeros(self.n_harm, complex)
        co = self.coef.get(app)
        if not co:
            return c0.copy()
        # **적합 범위 밖으로는 외삽하지 않는다.** 규칙 14 — 안 잰 곳이다.
        lo, hi, vlo, vhi = self.span[app]
        P0, V0 = self.center[app]
        dp = float(np.clip(P, lo, hi)) - P0
        dv = float(np.clip(V, vlo, vhi)) - V0

        out = np.array(c0, dtype=complex)
        a1 = co.get("arg1")
        phi1 = (a1[0] + a1[1] * dp + a1[2] * dv) if a1 is not None else np.angle(c0[0])
        for h in range(1, self.n_harm + 1):
            j = h - 1
            mg, ph = co["mag"].get(h), co["phase"].get(h)
            mag = np.exp(mg[0] + mg[1] * dp + mg[2] * dv) if mg is not None else abs(c0[j])
            if ph is not None:
                u = ph[0] + ph[1] * dp + ph[2] * dv
                ang = u + h * phi1
            elif h == 1:
                ang = phi1
            else:
                ang = np.angle(c0[j])
            out[j] = mag * np.exp(1j * ang)
        return out

    def matrix(self, appliances: Sequence[str], powers: Sequence[float],
               V: float) -> np.ndarray:
        """(K, H) 복소. `powers[i]` 에서 평가한 각 기기의 와트당 지문."""
        return np.array([self.predict(a, p, V) for a, p in zip(appliances, powers)])

    def as_constant(self) -> np.ndarray:
        """(K, H, 2) — `harmonic_signatures` 와 같은 모양. 회귀 없이 상수만."""
        out = np.zeros((len(self.appliances), self.n_harm, 2), np.float32)
        for i, a in enumerate(self.appliances):
            c = self.const.get(a, np.zeros(self.n_harm, complex))
            out[i, :, 0], out[i, :, 1] = c.real, c.imag
        return out

    def report(self) -> str:
        rows = [f"  {'기기':16s}{'n':>7s}{'P 범위':>16s}{'V 범위':>16s}  회귀 적용 차수"]
        for a in self.appliances:
            co = self.coef.get(a)
            sp = self.span.get(a)
            nm = len(co["mag"]) if co else 0
            npz = len(co["phase"]) if co else 0
            rows.append(f"  {a:16s}{'-' if not sp else '':>7s}"
                        f"{'' if not sp else f'{sp[0]:.0f}~{sp[1]:.0f}W':>16s}"
                        f"{'' if not sp else f'{sp[2]:.1f}~{sp[3]:.1f}V':>16s}"
                        f"  크기 {nm}/{self.n_harm}, 위상 {npz}/{self.n_harm}")
        return "\n".join(rows)


# ── 내부 ────────────────────────────────────────────────────────────────
def _collect(pool, app: str, window: int):
    """격리 활성 구간에서 (와트당 복소, P, V) 를 창 단위로 모은다."""
    C, P, V = [], [], []
    for a in pool.appliance_activations.get(app, []):
        c = np.asarray(a.net_harmonics_complex)
        p = np.asarray(a.target_power_w)
        # 전압은 `net_power_features` 5열이다 (p,q,s,pf,v,thd).
        # `v_ref_v` 는 녹화당 스칼라라 창 안의 변동을 못 준다.
        nf = np.asarray(getattr(a, "net_power_features", None))
        v = nf[:, 4] if nf.ndim == 2 and nf.shape[1] > 4 else np.full(len(p), np.nan)
        m = p > 5.0
        i = np.flatnonzero(m)
        for k in range(0, len(i) - window, window):
            s = i[k:k + window]
            if s[-1] - s[0] > window * 1.5:
                continue
            pm = p[s].mean()
            if pm <= 0:
                continue
            C.append(c[s].mean(0) / pm)
            P.append(pm)
            V.append(np.nanmean(v[s]))
    if not C:
        return np.zeros((0, 15), complex), np.zeros(0), np.zeros(0)
    return np.array(C), np.array(P), np.array(V)


def _fit_one(C, P, V, center, n_harm, use_voltage: bool = True) -> dict:
    """차수별 (크기 로그선형, 위상 불변량 선형) 적합. R^2 가 낮으면 안 넣는다."""
    P0, V0 = center
    novolt = (not use_voltage) or (not np.isfinite(V).all())
    X = np.column_stack([np.ones(len(P)), P - P0,
                         np.zeros(len(P)) if novolt else V - V0])
    out = {"mag": {}, "phase": {}, "arg1": None}

    def ls(y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1.0 - np.var(y - X @ b) / max(np.var(y), 1e-15)
        return b, r2

    z1 = C[:, 0] / np.maximum(np.abs(C[:, 0]), 1e-15)
    b, r2 = ls(np.unwrap(np.angle(z1)))
    if r2 >= MIN_R2:
        out["arg1"] = b
    for h in range(1, n_harm + 1):
        j = h - 1
        c = C[:, j]
        if not np.isfinite(c).all() or (np.abs(c) <= 0).any():
            continue
        b, r2 = ls(np.log(np.abs(c)))
        if r2 >= MIN_R2:
            out["mag"][h] = b
        if h == 1:
            continue
        # 위상 불변량. 감김을 P 순서로 펴서 선형성을 살린다
        u = np.angle(c * np.conj(z1) ** h)
        o = np.argsort(P)
        uu = np.empty_like(u)
        uu[o] = np.unwrap(u[o])
        b, r2 = ls(uu)
        if r2 >= MIN_R2:
            out["phase"][h] = b
    return out
