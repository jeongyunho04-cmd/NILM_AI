# -*- coding: utf-8 -*-
"""HNE / FCM 추출과 결합 생성기 (12.185.13, 가이드 §5·§6). 사용자 `fcm.py` 를 우리 시뮬 위로 옮긴 판.

    I(h) = J(h) − Σₖ Y(h,k)·ΔV(k)          (HNE; Y = FCM)

원본과 다른 점
--------------
1. `circuit3`/`circuit5` 대신 검증된 `synthesis.circuit_sim.simulate` 를 쓴다. `circuit_sim` 은
   가이드 §3.2 와 같은 토폴로지(v3)이고, 원시 파형으로 세 기기를 4.6·1.4·4.2% 로 재현한다
   (12.185.10). v5(선로측 X-cap+NTC)는 넣지 않았다 — 자료로 구분되지 않는다 (12.185.5).
2. 파라미터는 `circ_<dev>.pkl` 대신 `results/_circuit_raw_C.json` (원시 적합) 에서 읽는다.
3. **전력 규약**: 인수 `p` 는 **교류 입력 전력**이다 (NILM 이 다루는 양). 시뮬의 `P` 는 직류 부하
   전력이라 3~5% 다르므로 안쪽에서 되풀이해 맞춘다 (`fit_raw` 와 같은 규약).

Y 를 생성에 쓰지 마라
---------------------
사용자 `fcm_check` 와 가이드 §5.2: V3 를 1% 만 흔들어도 `I ≈ J − Y·ΔV` 가 h9 에서 27%, h13 에서
70% 틀린다. 도통각이 전압 왜곡에 초선형이기 때문이다. 그래서 `forward()` 는 고정점 안에서
시뮬레이터를 직접 부른다. `circuit_sim.norton` 의 절대판도 같은 이유로 생성용이 아니다.
Y 의 용도는 감도 구조 분석과 판별력 진단뿐이다.

결합은 실측으로 섰다 (12.185.12): 조합 원시 스냅샷에서 각 기기를 **조합의 단자 전압**으로 돌려
더하면 3.6~8.8% 로 맞는데, 각자 단독 녹화의 단자 전압으로 돌려 더하면 12.8~15.7% 로 단순 중첩과
같아진다. 개선은 전적으로 단자 전압에서 온다 — `forward()` 가 하는 일이 바로 그것이다.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.synthesis.circuit_sim import F, NPC, simulate, to_wave

H = 15
HFULL = 128                     #: 소스 전압 스펙트럼의 최고 차수 (원시 256표본/주기의 나이퀴스트)
NCYC = 10                       #: 시뮬 주기 수 (8주기면 잠긴다)
RAW_FIT_JSON = "results/_circuit_raw_C.json"


def odd_only(V: np.ndarray) -> np.ndarray:
    """짝수 차수를 지운다 — 전압 채널의 짝수차는 계측 인공물이다 (12.185.3, 가이드 §10.3)."""
    out = np.zeros_like(np.asarray(V, complex))
    out[0::2] = np.asarray(V, complex)[0::2]
    return out


# ── 소스 전압 ────────────────────────────────────────────────────────────────
def to_spectrum(V, h_full: int = HFULL) -> np.ndarray:
    """소스 전압을 `(h_full+1,)` complex 스펙트럼으로 (h0..h_full, 홀수차만, V1 기준 위상).

    받는 것 (세 가지, 길이로 구분한다):
      · 길이 `h_full+1` 복소 배열  -> **차수 색인** `X[h]` (이 함수의 출력 형식, 되먹임용)
      · 그 밖의 복소 배열          -> `V[k-1] = h(k)` (펌웨어·`simulate` 의 (15,) 관례)
      · 실수 배열                  -> **한 주기** 시간영역 파형

    **왜 h15 로 두면 안 되는가 (12.185.16).** 실시간 입력은 고조파 15개지만 **합성은
    오프라인이라 소스 전압을 우리가 고른다** — 원시 스냅샷의 전대역 파형을 그대로 넣으면 된다.
    h15 로 자르면 파형 피크가 +0.94V 오르고 도통각이 38.8° -> 36.9° 로 2° 밀리는데, 도통각이
    전압 왜곡에 초선형이라 그 2° 가 **우리가 내놓아야 하는 h9~h15 전류**를 크기 14%·위상
    20~37° 바꾼다. 실측: 조합 재현이 전대역 3.5~8.7% 대 h15 절단 6.3~13.9%.
    (h17+ 를 예측하자는 게 아니다. 시뮬레이터 입력의 세부일 뿐이고, 그 세부가 h≤15 를 흔든다.)
    """
    V = np.asarray(V)
    X = np.zeros(h_full + 1, complex)
    if np.iscomplexobj(V) and len(V) == h_full + 1:
        X[:] = V                                     # 이미 차수 색인
    elif np.iscomplexobj(V):
        n = min(len(V), h_full)
        X[1:n + 1] = V[:n]                          # V[k-1] = h(k) 관례
    else:                                            # 한 주기 실수 파형
        B = np.fft.rfft(np.asarray(V, float)) / (len(V) / 2) / np.sqrt(2.0)
        n = min(len(B) - 1, h_full)
        X[1:n + 1] = B[1:n + 1]
    X[0] = 0.0
    X[2::2] = 0.0                                    # 짝수차는 계측 인공물
    if X[1] != 0:
        X = X * np.exp(-1j * np.arange(h_full + 1) * np.angle(X[1]))
    return X


def spec_to_wave(X: np.ndarray, npc: int = NPC, ncyc: int = NCYC) -> np.ndarray:
    """스펙트럼(rms 페이저, h0..) -> 시간영역 `ncyc` 주기. `to_wave` 와 같은 규약이지만 irfft 라 빠르다."""
    Y = np.zeros(npc // 2 + 1, complex)
    n = min(len(X), npc // 2 + 1)
    Y[:n] = np.asarray(X, complex)[:n] * np.sqrt(2.0) * (npc / 2)
    return np.tile(np.fft.irfft(Y, npc), ncyc)


def source_from_raw(stem: str, data_dir: str = "data", h_full: int = HFULL) -> np.ndarray:
    """원시 스냅샷 -> 생성기용 **전대역** 소스 전압 스펙트럼.

    역RC(참 계통 전압 복원) + 반파 대칭화 + V1 기준 위상까지 끝난 것. 장소 C 는
    `file_registry.RAW_SNAPSHOT_FILES` / `RAW_COMBO_FILES` 의 18개가 전압 파형 라이브러리다.
    """
    from src.synthesis.fit_raw import load_raw, rc_filter
    pt = load_raw(stem, data_dir=data_dir)
    return to_spectrum(rc_filter(pt.v, inverse=True), h_full)


def line_impedance(r_line: float, l_line: float, h_max: int = H) -> np.ndarray:
    """`Z(h) = R + j·2π·60·h·L` (h_max,) complex."""
    h = np.arange(1, h_max + 1)
    return r_line + 1j * 2 * np.pi * F * h * l_line


@dataclass
class DeviceModel:
    """한 기기의 회로 모델. `par5 = (C_dc, R, L, Cx, rd)`."""
    name: str
    par5: Tuple[float, ...]

    def current(self, p_ac: float, V, n_match: int = 3,
                ncyc: int = NCYC) -> Optional[np.ndarray]:
        """(15,) complex 절대 전류 [A rms], 펌웨어 관례. `p_ac` 는 **교류 입력 전력** [W].

        `V` 는 `to_spectrum` 이 받는 것 아무거나 — (15,) 페이저, 전대역 스펙트럼, 한 주기 파형.
        **전대역을 넣어라** (12.185.16, `to_spectrum` 주석). 시뮬의 P 는 직류 부하 전력이라
        브리지·직렬저항 손실만큼 다르므로 되풀이해 맞춘다.
        """
        vsrc = spec_to_wave(to_spectrum(V), NPC, ncyc)
        P = p_ac
        for _ in range(n_match):
            r = simulate(P, *self.par5, vsrc=vsrc)
            if not r["ok"] or not np.isfinite(r["p_w"]) or r["p_w"] <= 0:
                return None
            P *= p_ac / r["p_w"]
        r = simulate(P, *self.par5, vsrc=vsrc)
        return r["I"] if r["ok"] else None

    def norton(self, p_ac: float, V_base: Sequence[complex], rel: float = 0.01
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(J, Y). `Y(h,k) = −∂I_h/∂V_k` [S]. 홀수 k 만 흔든다 (반파 대칭 입력).

        ⚠ 진단 전용. 생성에는 `forward()` 를 써라 (선형화는 h9 이상에서 25~70% 틀린다).
        """
        V_base = np.asarray(V_base, complex)
        J = self.current(p_ac, V_base)
        if J is None:
            raise RuntimeError(f"시뮬 발산: {self.name} @ {p_ac:.1f}W")
        Y = np.zeros((H, H), complex)
        dV = rel * abs(V_base[0])
        for k in range(1, H + 1, 2):
            acc = np.zeros(H, complex)
            for ph in (0.0, np.pi / 2):
                Vp = V_base.copy()
                Vp[k - 1] += dV * np.exp(1j * ph)
                Ip = self.current(p_ac, Vp)
                if Ip is None:
                    raise RuntimeError(f"섭동 발산: {self.name} k={k}")
                acc += -(Ip - J) / (dV * np.exp(1j * ph))
            Y[:, k - 1] = acc / 2
        return J, Y


def load_models(path: str = RAW_FIT_JSON, key: str = "fit_fixrd") -> Dict[str, DeviceModel]:
    """원시 적합 결과에서 기기 모델을 읽는다.

    기본은 `fit_fixrd` — **rd 를 물리값 0.3Ω 에 고정한 정본**이다 (12.185.19). 자유 적합은
    R 과 rd 가 도통 구간에서 직렬로만 보여 맞바뀌고(미니PC R 2.71 / rd 6.67Ω), rd 6.67Ω 은
    실리콘 정류 다이오드의 물리 범위(0.1~0.5Ω) 밖이다. 고정해도 훈련·LOO 가 그대로다.
    `key`: fit_fixrd(정본) / fit_free / fit_fixcx / fit_fixboth.
    """
    import json
    d = json.load(open(path, encoding="utf-8"))["devices"]
    return {k: DeviceModel(k, tuple(v[key]["par5"])) for k, v in d.items()}


# ── 결합 생성기 ──────────────────────────────────────────────────────────────
def forward(powers: Dict[str, float], models: Dict[str, DeviceModel], V_src: Sequence[complex],
            Z: np.ndarray, n_iter: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    """공유 임피던스로 결합된 총 전류 (가이드 §6.1).

        V_term = V_src − Z(h)·I_total,   I_total = Σᵢ Iᵢ(pᵢ, V_term)

    `V_src` 는 `to_spectrum` 이 받는 것 아무거나. **전대역을 넣어라** — 우리 기기의 전류는
    h15 까지만 모델링하므로 `Z·I` 보정도 h1..h15 에만 걸리고, h17+ 는 V_src 의 값이 그대로
    단자 전압에 남는다 (우리 부하의 h17+ 전류는 무시할 만하다).

    반환 (I_total (15,), V_term (HFULL+1,) 스펙트럼, 마지막 반복의 상대 변화).
    분해가 개입하지 않으므로 §1.2 의 인수분해 문제가 원천적으로 없다. 3기기 × 3반복 0.1초.
    """
    Vs = to_spectrum(V_src)
    Zf = np.zeros(len(Vs), complex)
    Z = np.asarray(Z, complex)
    Zf[1:len(Z) + 1] = Z[:len(Vs) - 1]

    def total(V):
        s = np.zeros(H, complex)
        for d, p in powers.items():
            I = models[d].current(p, V)
            if I is None:
                raise RuntimeError(f"시뮬 발산: {d} @ {p:.1f}W")
            s = s + I
        return s

    I_tot = total(Vs)
    V_term = Vs.copy()
    delta = np.nan
    for _ in range(n_iter):
        If = np.zeros(len(Vs), complex)
        If[1:H + 1] = I_tot
        V_new = Vs - Zf * If
        V_new[2::2] = 0.0
        I_new = total(V_new)
        delta = float(np.linalg.norm(I_new - I_tot) / max(np.linalg.norm(I_new), 1e-12))
        V_term, I_tot = V_new, I_new
    return I_tot, V_term, delta


def z_from_steps(stem: str = "electric_kettle_4C", p_on: float = 700.0, guard: float = 1.0,
                 win: float = 5.0, h_max: int = H) -> Tuple[np.ndarray, np.ndarray, int]:
    """대전력 부하 on/off 계단에서 `Z(h) = −ΔV/ΔI` (부록A §3 의 방법을 차수별로). 12.185.14.

    반환 (Z 중앙값 (15,) complex, 계단 간 상대 산포 (15,), 계단 수).
    ON 창은 `range==1`, OFF 창은 `range==0` 만 쓴다 (레인지별 위상 교정이 두 창을 같은 자로
    만든다는 전제). 계단 앞뒤 `guard` 초를 버리고 `win` 초를 평균한다.

    ⚠ **h1 만 믿어라.** h3~h7 은 78개 계단에서 산포 22~37% 로 재현되지만 ∠Z 가 141°/62°/11°
    로 수동 R+jωL 이 낼 수 없는 값이다 (h3 은 실수부가 음수). 두 가지가 섞여 있다:
    (a) 분모 — 포트가 HIGH 레인지라 저항 고조파 전류에 0.5~3% 의 인공물이 실린다 (12.184.11,
        규칙 75). ΔI3 0.23A 자체가 그 크기다. (b) 분자 — 우리 부하가 6.4A 를 끌면 계통이
        3.4V 내려가고 **상류의 다른 정류 부하들이 그만큼 도통을 바꿔** V3~V7 을 되민다.
    즉 이 비는 수동 임피던스가 아니라 "그 부하에 대한 그 장소의 응답" 이다.
    """
    from src.preprocessing.raw_csv import read_raw_csv
    from src.preprocessing.raw_phasors import current_phasors, voltage_phasors

    cols = ["t_s", "p_w", "vrms", "over_range", "range"] + [f"ih{k}" for k in range(1, h_max + 1)]         + [f"ihdeg{k}" for k in range(1, h_max + 1)] + [f"vh{k}" for k in range(1, h_max + 1)]         + [f"vhdeg{k}" for k in range(1, h_max + 1)]
    df, _ = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    t = df["t_s"].to_numpy(float)
    p = df["p_w"].to_numpy(float)
    rg = df["range"].to_numpy()
    over = df["over_range"].to_numpy()
    on = p > p_on

    def w(lo, hi, want_on):
        m = (t >= lo) & (t < hi) & (on == want_on) & (rg == (1 if want_on else 0)) & (over == 0)
        if m.sum() < 15:
            return None
        C = current_phasors(df, h_max)[m]
        return (np.median(C.real, 0) + 1j * np.median(C.imag, 0)), voltage_phasors(df, m)[0]

    Zs = []
    for i in np.where(np.diff(on.astype(int)) != 0)[0]:
        a = w(t[i] - guard - win, t[i] - guard, on[i])
        b = w(t[i] + guard, t[i] + guard + win, on[i + 1])
        if a is None or b is None:
            continue
        Zs.append(-(b[1] - a[1]) / (b[0] - a[0]))
    if not Zs:
        return np.full(h_max, np.nan, complex), np.full(h_max, np.nan), 0
    A = np.array(Zs)
    Zm = np.median(A.real, 0) + 1j * np.median(A.imag, 0)
    return Zm, np.std(np.abs(A), 0) / np.maximum(np.abs(Zm), 1e-9), len(Zs)


def measure_z(V_a: np.ndarray, I_a: np.ndarray, V_b: np.ndarray, I_b: np.ndarray) -> np.ndarray:
    """두 녹화(부하만 다르고 계통은 같다고 보는)의 차분에서 `Z(h) = −ΔV/ΔI` (12.185.14).

    `V = V_src − Z·I` 에서 V_src 가 같으면 `V_a − V_b = −Z·(I_a − I_b)`.
    ⚠ 계통 자체의 표류가 ΔV 에 섞인다 — 녹화 간격이 짧아야 하고, ΔI 가 클수록 낫다.
    """
    dI = np.asarray(I_a, complex) - np.asarray(I_b, complex)
    dV = np.asarray(V_a, complex) - np.asarray(V_b, complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -dV / dI
