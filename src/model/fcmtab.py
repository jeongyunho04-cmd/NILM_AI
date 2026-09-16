# -*- coding: utf-8 -*-
"""FCM 회로모델의 **표 대리모델** — 프레임마다 회로를 못 도니까 (14.319, 사용자 제안).

`fcm.DeviceModel12.current(p, V)` 는 회로 시뮬이라 호출당 ~30ms 다. 90,000 프레임 ×
기기 셋이면 불가능하다. 그래서 **전력 격자 × 전압 두 수준**에서 미리 돌려 표로 만들고,
프레임마다는 **보간 + 1차 전압 보정**만 한다:
```
  T_d(p, v) ≈ interp_p( T0_d[p_grid] ) · (1 + D_d[p] · (v/v_ref − 1))
  T0 = I(p, V_ref)/p            (15,) 복소 **와트당** 페이저
  D  = dlnT/dlnV                (15,) 복소 — 두 전압 수준의 차분
```
격자 한 번 짓는 데 기기당 `|p_grid| × 2` 회 시뮬 (기본 10×2 = 20회, 셋이면 60회 ≈ 2초).

⚠ **이것이 `sig_model.SigModel` 과 다른 점** — SigModel 은 **실측 세그먼트 회귀**이고
  이것은 **회로 물리**다. 13.84.23 이 상수 지문의 SMPS 잔차를 17~27% 로 쟀고 원인을
  "도통각이 전력에 따라 변한다" 로 지목했는데, 도통각을 실제로 푸는 것이 FCM 이다.

⚠ 회로모델이 있는 기기는 **셋뿐**이다 (`circuit_model/circ12_*.pkl`):
  `beam_projector` · `laptop_charger` · `minipc`. 선풍기·에어컨은 없다 — 그 둘은
  여전히 실측 지문을 쓴다. 저항 넷은 애초에 `G` 열이 맡는다.

⚠ `DeviceModel12.current` 는 소스로 **h1..h31** 을 받는 것이 규약이다 (13.73). 우리 측정은
  h15 까지라 꼬리가 0 이고, 그러면 h11~h15 전류가 덜 정확하다 (충전기 |I13| 오차 0.30 대
  꼬리 있을 때 0.03). **이 한계를 표에 적어 둔다** — 고차를 믿을 때 이것을 기억하라.

⚠ 이 노트북에서는 `circ12_*.pkl` 이 numpy 2.x 절임이라 1.24 에서 못 푼다. 스크래치패드의
  `sitecustomize.py` 별칭을 `PYTHONPATH` 로 태워야 로드된다 (프로젝트는 안 건드렸다).
"""
from typing import Dict, Sequence, Tuple

import numpy as np

N_HARM = 15
#: 기본 전력 격자 배수 — 기기별 관측 범위를 로그로 훑는다.
DV = 0.02          #: 전압 미분을 낼 때 흔드는 폭 (±2%)


def fcm_devices() -> Dict[str, object]:
    """회로모델이 있는 기기. 없으면 무엇이 문제인지 말하고 죽는다."""
    from src.synthesis.fcm import load_models
    return load_models()


def build_table(models: Dict[str, object], p_ranges: Dict[str, Tuple[float, float]],
                v_ref: np.ndarray, n_p: int = 10, dv: float = DV
                ) -> Dict[str, Dict[str, np.ndarray]]:
    """기기별 표를 짓는다.

    `v_ref` 는 (>=15,) 복소 **절대** 전압 스펙트럼 (h1..). `p_ranges[d] = (lo, hi)` 와트.
    돌려주는 것: `{d: {"p": (P,), "T0": (P,15)복소, "D": (P,15)복소}}`
    """
    out = {}
    for d, (lo, hi) in p_ranges.items():
        if d not in models:
            continue
        pg = np.geomspace(max(lo, 1.0), max(hi, lo * 1.2 + 1.0), n_p)
        T0 = np.zeros((n_p, N_HARM), complex)
        D = np.zeros((n_p, N_HARM), complex)
        for i, p in enumerate(pg):
            a = models[d].current(float(p), v_ref * (1.0 - dv))
            b = models[d].current(float(p), v_ref * (1.0 + dv))
            c = models[d].current(float(p), v_ref)
            if a is None or b is None or c is None:
                T0[i] = D[i] = np.nan
                continue
            T0[i] = np.asarray(c)[:N_HARM] / p
            #: dlnT/dlnV — 두 수준의 **상대** 차분. 0 나눗셈은 0 으로 둔다.
            ca, cb = np.asarray(a)[:N_HARM], np.asarray(b)[:N_HARM]
            den = np.where(np.abs(c[:N_HARM]) > 1e-15, c[:N_HARM], np.inf)
            D[i] = (cb - ca) / den / (2.0 * dv)
        ok = np.isfinite(T0).all(1)
        out[d] = {"p": pg[ok], "T0": T0[ok], "D": D[ok]}
    return out


def template(tab: Dict[str, np.ndarray], p: float, vrel: float = 1.0) -> np.ndarray:
    """(15,) 복소 **와트당** 페이저. `vrel = |V₁|/|V₁_ref|`.

    격자 밖으로는 **외삽하지 않는다** — 끝값을 쓴다 (규칙 14: 안 잰 곳이다).
    """
    pg, T0, D = tab["p"], tab["T0"], tab["D"]
    if len(pg) == 0:
        return np.zeros(N_HARM, complex)
    q = float(np.clip(p, pg[0], pg[-1]))
    j = int(np.searchsorted(pg, q).clip(1, len(pg) - 1))
    w = (q - pg[j - 1]) / max(pg[j] - pg[j - 1], 1e-12)
    t0 = (1 - w) * T0[j - 1] + w * T0[j]
    dd = (1 - w) * D[j - 1] + w * D[j]
    return t0 * (1.0 + dd * (vrel - 1.0))


def v_ref_from_raw(raw: np.ndarray, volt_re0: int, n_v: int) -> np.ndarray:
    """원시에서 **중앙 전압 스펙트럼** (n_v,) 복소를 뽑는다 (창·프레임 중앙값)."""
    re = np.median(raw[:, volt_re0:volt_re0 + n_v, :], axis=(0, 2))
    im = np.median(raw[:, volt_re0 + n_v:volt_re0 + 2 * n_v, :], axis=(0, 2))
    return re + 1j * im


def spread_report(tab: Dict[str, Dict[str, np.ndarray]]) -> str:
    """★ **쓸 가치가 있나** — 와트당 지문이 전력에 따라 얼마나 변하나.

    상수 지문(`harmonic_signatures`)은 이 변동을 **0 으로 본다.** 변동이 작으면 FCM 을
    붙여도 얻을 것이 없다. 재고 나서 쓴다.
    """
    L = ["  %-18s %8s %10s %10s %10s" % ("기기", "격자", "|T|변동 h1", "h3", "h5")]
    for d, t in sorted(tab.items()):
        if len(t["p"]) < 2:
            L.append("  %-18s 격자 부족" % d)
            continue
        m = np.abs(t["T0"])
        r = [(m[:, h].max() / max(m[:, h].min(), 1e-30) - 1.0) * 100 for h in (0, 2, 4)]
        L.append("  %-18s %4.0f~%4.0fW %9.1f%% %9.1f%% %9.1f%%"
                 % (d, t["p"][0], t["p"][-1], r[0], r[1], r[2]))
    return "\n".join(L)
