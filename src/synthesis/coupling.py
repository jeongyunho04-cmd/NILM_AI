# -*- coding: utf-8 -*-
"""SMPS 사이의 공유 임피던스 결합 (①b, 12.185.12 / 12.185.22).

**무엇이 빠져 있었나.** 지금 생성기는 기기별 녹화 페이저를 그냥 더한다 = 단순 중첩.
12.185.12 가 원시 조합 스냅샷 6개로 그 대가를 쟀다 (조합에 맞춘 자유 파라미터 0개):

    [A] 단순 중첩 (지금 생성기)          12.7 ~ 15.9%
    [B] 모델 + 조합 V_term                3.5 ~  8.3%     <- 4배
    [C] 모델 + 단독 V_term               12.8 ~ 15.7%  = [A]

[B]와 [C]는 회로도 파라미터도 같고 **단자 전압만** 다르다. 개선이 전적으로 거기서 온다.

**⚠ 그런데 그 단자 전압 차의 정체는 공유 임피던스가 아니었다** (12.185.22). 조합 녹화의
실측 전압과 단독 녹화의 전압은 V1 의 0.87~1.03% 다른데 `Z·I_total` 이 설명하는 몫은
0.05~0.08% 뿐이다 (1/12). 나머지는 **세션이 다르다** 는 사실이다 — 단독은 22~23시,
조합은 00시 녹화다. 실측으로 갈랐다 (`raw_beam_minipc_1/2`, 전력 재배분 없는 두 조합):

    단순 중첩            13.5%
    + 결합(Z·I) 만       12.7%      <- 거의 안 내려간다
    + 텍스처만            2.7%      <- 이것이 [B] 의 정체다
    + 둘 다               2.8%      <- 소스가 이미 측정된 단자 전압이라 결합이 이중 계상된다

그래서 이 모듈은 **둘을 나눠 준다**: `CouplingModel`(공유 임피던스, 작지만 실체) 과
`TextureModel`(세션·장소 전압 파형, 지배적). 생성기 기본은 텍스처다.

**전면 교체가 아니라 증강.** 시뮬 절대 전류로 갈아 끼우면 우리 모델의 단독 오차(1.4~4.6%)가
훈련 신호에 통째로 들어간다. 대신 **빠진 몫만** 더한다:

    ΔI_i = I_i(조합 V_term) − I_i(단독 V_term)
    I_i(생성) = I_i(녹화 재생) + ΔI_i

ΔI 는 차분이라 모델의 공통 편향이 상쇄된다.

**비용.** `fcm.forward` 는 3기기 3반복에 0.1초라 표본(60Hz)마다 못 부른다. 창 안에서 켜진
조합·전력·Z 가 같은 구간은 같은 ΔI 를 쓰고, (조합, 전력구간, Z구간) 을 키로 캐시한다.
전력 구간 5W · Z 구간 0.25Ω 이면 몇천 창 뒤 거의 전부 적중한다.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.synthesis import fcm

H = 15
#: 캐시 구간. 전력 5W 는 우리 동작점 간격(7~25W, 17~70W)의 1/3 이라 결합 곡선을 뭉개지 않는다.
P_BIN_W = 5.0
Z_BIN_OHM = 0.25
#: `fcm.load_models()` 가 아는 기기 (원시 파형으로 맞춘 셋)
SMPS_DEVICES = ("laptop_charger", "beam_projector", "minipc")
#: ΔI 를 계산할 때 쓰는 대표 소스. [B2] 가 "소스에 둔감" 이라고 하면 하나로 충분하다.
DEFAULT_SOURCE = "raw_beam_projector_1"
#: 선로 인덕턴스는 미측정이다 (∠Z₁ 을 vhdeg1≡0 규약상 못 잰다, 12.185.14).
#: 부록A §5 의 R 범위 0.7~2.0Ω 에 맞춰 L 은 0~400µH 로 흔든다.
L_LINE_RANGE = (0.0, 400e-6)


def forward_per_device(powers: Dict[str, float], models: Dict[str, fcm.DeviceModel],
                       V_src, Z: np.ndarray, n_iter: int = 3
                       ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """`fcm.forward` 의 기기별 판. 반환 ({기기: (15,) 전류}, V_term 스펙트럼).

    `fcm.forward` 는 총합만 돌려주는데 우리는 기기별 정답(`gt_harmonics_ri`)에 나눠 실어야
    하므로 고정점 마지막 반복의 개별 전류가 필요하다.
    """
    Vs = fcm.to_spectrum(V_src)
    Zf = np.zeros(len(Vs), complex)
    Z = np.asarray(Z, complex)
    Zf[1:len(Z) + 1] = Z[:len(Vs) - 1]

    def each(V) -> Optional[Dict[str, np.ndarray]]:
        out = {}
        for d, p in powers.items():
            I = models[d].current(p, V)
            if I is None:
                return None
            out[d] = I
        return out

    cur = each(Vs)
    if cur is None:
        return {}, Vs
    V_term = Vs
    for _ in range(n_iter):
        If = np.zeros(len(Vs), complex)
        If[1:H + 1] = sum(cur.values())
        V_new = Vs - Zf * If
        V_new[2::2] = 0.0                      # 짝수차는 아날로그 전단의 인공물 (규칙 77)
        nxt = each(V_new)
        if nxt is None:
            break
        V_term, cur = V_new, nxt
    return cur, V_term


class CouplingModel:
    """결합 보정 ΔI 를 계산하고 캐시한다."""

    def __init__(self, models: Optional[Dict[str, fcm.DeviceModel]] = None,
                 source: Optional[np.ndarray] = None, p_bin_w: float = P_BIN_W,
                 z_bin_ohm: float = Z_BIN_OHM, max_cache: int = 200_000):
        self._models = models
        self._source = source
        self.p_bin_w = float(p_bin_w)
        self.z_bin_ohm = float(z_bin_ohm)
        self.max_cache = int(max_cache)
        self._cache: Dict[tuple, Dict[str, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0
        self.failures = 0

    # ── 지연 적재 (모델·소스는 첫 호출에서 읽는다. 생성기 임포트를 무겁게 하지 않는다) ──
    @property
    def models(self) -> Dict[str, fcm.DeviceModel]:
        if self._models is None:
            self._models = fcm.load_models()
        return self._models

    @property
    def source(self) -> np.ndarray:
        if self._source is None:
            self._source = fcm.source_from_raw(DEFAULT_SOURCE)
        return self._source

    def _key(self, powers: Dict[str, float], r_line: float, l_line: float) -> tuple:
        return (tuple(sorted((d, int(round(p / self.p_bin_w)))
                             for d, p in powers.items() if p > 0.5)),
                int(round(r_line / self.z_bin_ohm)),
                int(round(l_line / 100e-6)))

    def delta(self, powers: Dict[str, float], r_line: float,
              l_line: float = 0.0) -> Dict[str, np.ndarray]:
        """{기기: (15,) complex ΔI}. 결합 상대가 없으면 빈 dict.

        `powers` 는 **교류 입력 전력** [W]. 모르는 기기는 무시한다.
        """
        p = {d: float(v) for d, v in powers.items()
             if d in SMPS_DEVICES and v is not None and v > 0.5}
        if len(p) < 2:                                  # 결합할 상대가 없다 ([B5a])
            return {}
        key = self._key(p, r_line, l_line)
        got = self._cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        # 키의 구간 대표값으로 계산한다 (같은 키가 항상 같은 답을 주도록)
        pq = {d: max(round(v / self.p_bin_w) * self.p_bin_w, self.p_bin_w) for d, v in p.items()}
        rq = max(round(r_line / self.z_bin_ohm) * self.z_bin_ohm, self.z_bin_ohm)
        lq = round(l_line / 100e-6) * 100e-6
        out = self._compute(pq, rq, lq)
        if len(self._cache) < self.max_cache:
            self._cache[key] = out
        return out

    def _compute(self, powers: Dict[str, float], r_line: float,
                 l_line: float) -> Dict[str, np.ndarray]:
        Z = fcm.line_impedance(r_line, l_line, H)
        src = self.source
        try:
            combo, _ = forward_per_device(powers, self.models, src, Z)
            if not combo:
                raise RuntimeError("고정점 발산")
            # 단독: 같은 소스, Z 없음 (다른 기기가 없을 때 이 기기가 보는 단자 전압)
            solo = {}
            for d, p in powers.items():
                one, _ = forward_per_device({d: p}, self.models, src, Z)
                if not one:
                    raise RuntimeError(f"단독 발산 {d}")
                solo[d] = one[d]
        except Exception:
            self.failures += 1
            return {}
        return {d: (combo[d] - solo[d]).astype(np.complex64) for d in powers}

    def stats(self) -> Dict[str, float]:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "failures": self.failures,
                "hit_rate": (self.hits / n) if n else 0.0, "size": len(self._cache)}


# ── 전압 텍스처 ──────────────────────────────────────────────────────────────
#: **①b 의 지배 축은 결합이 아니라 텍스처다** (12.185.22).
#:
#: 조합 녹화의 실측 단자 전압과 단독 녹화의 전압은 V1 의 0.87~1.03% 만큼 다른데,
#: 그중 `Z·I_total` 이 설명하는 몫은 **0.05~0.08% 뿐이다 (1/12)**. Z·I 를 빼도 차가
#: 0.89~1.02% 로 거의 그대로 남는다. 남는 것은 **세션이 다르다** 는 사실이다
#: (단독 22:17~23:21, 조합 00:29~00:33; `REPLY_vtemplate_rc_2026-09-05.md` §3 이 같은 것을
#: 텍스처가 녹화 시각으로 묶인다고 적었다).
#:
#: 그런데 그 0.9% 가 전류를 크게 바꾼다 — 도통각이 전압 왜곡에 초선형이라(12.185.16)
#: h17+ 의 0.27% 가 h9~h15 에서 크기 14%·위상 20~37° 를 만든다. 그래서 12.185.12 의
#: "[B] 3.5~8.3% 대 [C] 12.8~15.7%" 는 결합이 아니라 **텍스처가 낸 4배**였다.
#:
#: 결합과 같은 방식으로 더한다:  ΔI_i = I_i(다른 텍스처) − I_i(기준 텍스처)
TEXTURE_STEMS: Tuple[str, ...] = (
    # 22:19~22:22 무리
    "raw_beam_projector_1", "raw_beam_projector_2", "raw_minipc_1", "raw_minipc_2",
    # 23:19~23:21 무리
    "raw_minipc_3", "raw_minipc_4", "raw_minipc_5", "raw_laptop_charger_4", "raw_laptop_charger_5",
    # 00:29~00:33 무리 (조합 녹화)
    "raw_beam_minipc_1", "raw_beam_charger_1", "raw_smps3_1",
)
#: 텍스처 델타의 기준. 이 소스에서의 전류가 "녹화 재생" 에 해당한다고 본다.
TEXTURE_REF = DEFAULT_SOURCE


class TextureModel:
    """전압 텍스처가 바꾸는 전류 ΔI_i = I_i(텍스처 k) − I_i(기준 텍스처).

    캐시 키는 (기기, 전력구간, 텍스처). 기기별로 독립이라 조합 폭발이 없다 —
    `CouplingModel` 과 달리 기기 하나씩 계산한다.
    """

    def __init__(self, models: Optional[Dict[str, fcm.DeviceModel]] = None,
                 stems: Sequence[str] = TEXTURE_STEMS, ref: str = TEXTURE_REF,
                 p_bin_w: float = P_BIN_W):
        self._models = models
        self.stems = tuple(stems)
        self.ref = ref
        self.p_bin_w = float(p_bin_w)
        self._src: Dict[str, np.ndarray] = {}
        self._cache: Dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @property
    def models(self) -> Dict[str, fcm.DeviceModel]:
        if self._models is None:
            self._models = fcm.load_models()
        return self._models

    def source(self, stem: str) -> np.ndarray:
        if stem not in self._src:
            self._src[stem] = fcm.source_from_raw(stem)
        return self._src[stem]

    def delta(self, device: str, p_ac: float, stem: str) -> Optional[np.ndarray]:
        """(15,) complex ΔI. 기준 텍스처면 0, 모르는 기기면 None."""
        if device not in SMPS_DEVICES or p_ac <= 0.5:
            return None
        if stem == self.ref:
            return np.zeros(H, np.complex64)
        key = (device, int(round(p_ac / self.p_bin_w)), stem)
        got = self._cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        p = max(round(p_ac / self.p_bin_w) * self.p_bin_w, self.p_bin_w)
        m = self.models[device]
        a = m.current(p, self.source(stem))
        b = m.current(p, self.source(self.ref))
        out = (np.zeros(H, np.complex64) if a is None or b is None
               else (a - b).astype(np.complex64))
        self._cache[key] = out
        return out

    def stats(self) -> Dict[str, float]:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / n) if n else 0.0, "size": len(self._cache)}
