# -*- coding: utf-8 -*-
"""SMPS 전류를 회로 모델(v12g)로 **환경에 맞게 보정**한다 — 텍스처 델타 + 공유 임피던스 결합 (13.2, 2026-09-06).

왜 '교체' 가 아니라 '델타' 인가
-----------------------------
회로 모델(`circuit_model/circ12_*.pkl`, 잔차 2.1~4.5%, h13+ ±30%)로 전류를 통째로 만들면 그 오차가 훈련 신호에
그대로 들어간다. 녹화 페이저는 그 기기·그 순간의 진짜 전류다. 그래서 **녹화를 재생하고, 환경이 다를 때
달라지는 몫만 모델의 차분으로 더한다** (옛 12.185.12 의 결론을 새 계측기·새 모델로 옮긴 것):

    I_i(생성) = I_i(녹화 재생, kappa 보정)
              + [ I_sim(p_i, 합성 텍스처·v1) − I_sim(p_i, 녹화 텍스처·v1) ]        (텍스처 델타)
              + [ I_sim(p_i, V_term) − I_sim(p_i, V_src) ]                      (결합 델타, V_term = V_src − Z·ΣI)

차분이라 모델의 공통 편향은 상쇄되고, 전압 **파형**(텍스처)과 **공유 임피던스**(Z·I)에 대한 응답만 남는다.
두 항 모두 같은 v1(합성 기저 전압)에서 계산한다 — V1 크기의 효과는 `apply_cross_appliance_coupling` 의 kappa 가
이미 맡고 있어 여기서 다시 넣으면 이중 계상이다.

새 계측기에서 달라진 것
---------------------
· 텍스처 소스가 원시 스냅샷(삭제)이 아니라 **2Hz 녹화의 vh·vhdeg** (`vtexture.VoltageTextureLibrary`). 모든 녹화가
  텍스처이고, 델타의 기준은 임의의 대표 소스가 아니라 **그 활성화가 녹화된 파일 자체의 텍스처** 다.
· 모델은 `fcm.load_models()` 기본(v12g, `DeviceModel12`). 15차 소스 규약이라 h17+ 가 필요 없다.
· R(NTC 온도 상태)은 pkl 의 실측 범위에서 창마다 뽑아 두 항에 같이 쓴다 (README_v12 "R 은 상태다").
· 옛 ①b 의 "결합은 텍스처의 1/10" 은 옛 계측기 자료의 결론이라 **미검증**이다 — 결합도 같이 넣고 `run_mixval12`
  로 새 자료(test_1/test_2)에서 크기를 다시 잰다.

비용
----
시뮬 1회 ≈ 1ms (numba). (기기, 전력 5W 구간, 텍스처 id, 녹화 파일 id, v1 2V 구간, R 0.1Ω 구간) 을 키로 캐시한다.
텍스처는 파일당 60초에 하나라 수십~수백 개뿐이고, 몇천 창 뒤에는 거의 전부 적중한다.
"""
from typing import Dict, Optional, Tuple
import numpy as np

from src.synthesis import fcm

H = 15
F = 60.0
P_BIN_W = 5.0          #: 전력 구간 — 동작점 간격(7~25W, 17~70W)의 1/3
V_BIN_V = 2.0          #: 기저 전압 구간
R_BIN_OHM = 0.10       #: NTC 상태 R 구간
Z_BIN_OHM = 0.25       #: 선로 R 구간
L_BIN_H = 100e-6       #: 선로 L 구간
#: 회로 모델이 있는 SMPS (`circuit_model/circ12_<dev>.pkl`)
SMPS_DEVICES = ("laptop_charger", "beam_projector", "minipc")
#: 선로 인덕턴스 범위 (미측정 — ∠Z₁ 은 vhdeg1≡0 규약상 못 잰다). `GridSimulator.x_grid_range` 0.02~0.15Ω ↔ 53~400µH.
L_LINE_RANGE = (0.0, 400e-6)
#: 옛 API 호환용 — 옛 계측기 원시 stem 목록은 비었다 (파일 삭제). 텍스처는 `vtexture` 가 준다.
TEXTURE_STEMS: Tuple[str, ...] = ()


def _pbin(p: float) -> int:
    return int(round(float(p) / P_BIN_W))


class SmpsCircuit:
    """v12g 모델 위의 델타 계산기 + 캐시. 생성기(`GridSimulator`)가 하나 들고 쓴다."""

    def __init__(self, models: Optional[Dict[str, object]] = None, max_cache: int = 300_000,
                 n_iter: int = 3):
        self._models = models
        self.max_cache = int(max_cache)
        self.n_iter = int(n_iter)
        self._tex_cache: Dict[tuple, np.ndarray] = {}
        self._cpl_cache: Dict[tuple, Dict[str, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0
        self.failures = 0

    @property
    def models(self) -> Dict[str, object]:
        if self._models is None:
            self._models = fcm.load_models()          # circuit_model/circ12_*.pkl
        return self._models

    def has(self, device: str) -> bool:
        return device in SMPS_DEVICES and device in self.models

    def sample_r(self, device: str, rng: np.random.Generator) -> Optional[float]:
        m = self.models.get(device)
        return None if m is None or not hasattr(m, "sample_R") else float(m.sample_R(rng))

    # ── 원시 호출 ─────────────────────────────────────────────────────────
    def current(self, device: str, p: float, rel: np.ndarray, v1: float,
                R: Optional[float] = None) -> Optional[np.ndarray]:
        """(15,) complex 계측 영역 전류. 실패하면 None."""
        m = self.models.get(device)
        if m is None or p <= 0.5:
            return None
        try:
            I = m.current(float(p), np.asarray(rel, complex) * float(v1), R=R)
        except Exception:
            I = None
        if I is None or not np.all(np.isfinite(I)):
            self.failures += 1
            return None
        return np.asarray(I, complex)

    # ── 텍스처 델타 ───────────────────────────────────────────────────────
    def texture_delta(self, device: str, p: float, rel_env: np.ndarray, env_id: int,
                      rel_rec: np.ndarray, rec_id: int, v1: float,
                      R: Optional[float] = None) -> Optional[np.ndarray]:
        """I_sim(p, rel_env·v1) − I_sim(p, rel_rec·v1). 구간 대표값으로 계산해 같은 키가 같은 답을 준다."""
        if not self.has(device) or p <= 0.5 or rel_env is None or rel_rec is None:
            return None
        pb = _pbin(p); vb = int(round(v1 / V_BIN_V)); rb = -1 if R is None else int(round(R / R_BIN_OHM))
        key = ("tex", device, pb, int(env_id), int(rec_id), vb, rb)
        got = self._tex_cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        pq = max(pb * P_BIN_W, P_BIN_W); vq = vb * V_BIN_V; rq = None if R is None else rb * R_BIN_OHM
        a = self.current(device, pq, rel_env, vq, rq)
        b = self.current(device, pq, rel_rec, vq, rq)
        out = None if (a is None or b is None) else (a - b).astype(np.complex64)
        if len(self._tex_cache) < self.max_cache:
            self._tex_cache[key] = out
        return out

    # ── 결합 델타 ─────────────────────────────────────────────────────────
    def coupling_delta(self, powers: Dict[str, float], rel_env: np.ndarray, env_id: int, v1: float,
                       r_line: float, l_line: float, R: Optional[Dict[str, float]] = None
                       ) -> Dict[str, np.ndarray]:
        """{기기: I_i(V_term) − I_i(V_src)}. SMPS 가 **하나여도** 돈다 (13.22).

        V_term = V_src − Z(h)·Σ_i I_i, Z(h) = r_line + j·2π·60·h·l_line, 고정점 `n_iter` 회 (2회면 잠긴다, 가이드 §6.1).
        내부는 fcm12 규약대로 **참전류**(measured=False)로 돌고, 델타는 계측 영역(measured=True)으로 낸다.

        ⚠ `rel_env` 로는 **개방 전압**(`Texture.source_rel()`)을 받아야 한다. 단자 전압을 주면 그 녹화
        부하의 강하 위에 창의 강하를 또 얹어 두 번 걸린다 — 13.21 에서 실제로 그렇게 해 자리 비가
        나빠졌다(미니PC 0.111 -> 0.153, 충전기 0.201 -> 0.320). 13.22 에서 텍스처를 개방으로 되돌리고
        나서 문턱을 1 로 내렸다.
        """
        p = {d: float(v) for d, v in powers.items() if self.has(d) and v is not None and v > 0.5}
        if not p:
            return {}
        pb = tuple(sorted((d, _pbin(v)) for d, v in p.items()))
        rb = tuple(sorted((d, -1 if (R is None or R.get(d) is None) else int(round(R[d] / R_BIN_OHM))) for d in p))
        key = ("cpl", pb, rb, int(env_id), int(round(v1 / V_BIN_V)),
               int(round(r_line / Z_BIN_OHM)), int(round(l_line / L_BIN_H)))
        got = self._cpl_cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        pq = {d: max(b * P_BIN_W, P_BIN_W) for d, b in pb}
        rq = {d: (None if b < 0 else b * R_BIN_OHM) for d, b in rb}
        vq = int(round(v1 / V_BIN_V)) * V_BIN_V
        zr = int(round(r_line / Z_BIN_OHM)) * Z_BIN_OHM
        zl = int(round(l_line / L_BIN_H)) * L_BIN_H
        out = self._compute_coupling(pq, rel_env, vq, zr, zl, rq)
        if len(self._cpl_cache) < self.max_cache:
            self._cpl_cache[key] = out
        return out

    def _compute_coupling(self, powers: Dict[str, float], rel_env: np.ndarray, v1: float,
                          r_line: float, l_line: float, R: Dict[str, Optional[float]]) -> Dict[str, np.ndarray]:
        h = np.arange(1, H + 1)
        Z = r_line + 1j * 2 * np.pi * F * h * l_line
        V_src = np.asarray(rel_env, complex) * float(v1)
        ms = self.models
        try:
            def total_true(V):
                s = np.zeros(H, complex)
                for d, p in powers.items():
                    I = ms[d].simulate_true(p, V, R=R.get(d)) if hasattr(ms[d], "simulate_true") else None
                    if I is None:
                        raise RuntimeError(d)
                    s += I
                return s
            I_tot = total_true(V_src)
            V_term = V_src.copy()
            for _ in range(self.n_iter):
                V_term = V_src - Z * I_tot
                I_tot = total_true(V_term)
            out = {}
            for d, p in powers.items():
                a = self.current(d, p, V_term / float(v1), v1, R.get(d))
                b = self.current(d, p, rel_env, v1, R.get(d))
                if a is None or b is None:
                    raise RuntimeError(d)
                out[d] = (a - b).astype(np.complex64)
            return out
        except Exception:
            self.failures += 1
            return {}

    def stats(self) -> Dict[str, float]:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "failures": self.failures,
                "hit_rate": (self.hits / n) if n else 0.0,
                "size": len(self._tex_cache) + len(self._cpl_cache)}


# ── 옛 이름 호환 (탐침용). 새 코드는 SmpsCircuit 을 쓴다 ─────────────────────
TextureModel = SmpsCircuit
CouplingModel = SmpsCircuit
