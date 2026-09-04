"""
fcm.py — Harmonic Norton Equivalent / Frequency Coupling Matrix 추출 및 생성기
=================================================================================

    I(h) = J(h) − Σₖ Y(h,k)·V(k)         (HNE;  Y = FCM)

`fit_raw.py` 가 만든 `circ_<device>.pkl` 을 읽어 유한 차분으로 J, Y 를 뽑는다.
같은 시뮬레이터(`_core3` / `_core5`)와 같은 전압 관례(cos, 홀수만)를 쓴다.

    from fcm import FCM, forward
    f = FCM.from_pickle('circ_laptop_charger.pkl')
    I  = f.simulate(p=45.0, V15)               # 전류 (15,) complex, 전압 기준 위상  <- 생성에는 이것
    J, Y = f.extract(p=45.0, V_base=V15)      # HNE 계수 — 분석·진단용
    I_total, V_term = forward({'laptop_charger': 45, 'minipc': 18},
                              R_line=0.48, L_line=100e-6, V_src=V15,
                              models={'laptop_charger': f, 'minipc': g})   # 결합 생성 (exact)

중요 (fcm_check.py 결과, 2026-09):
  선형화 I ≈ J − Y·ΔV 는 ΔV < 1% 에서도 h9 이상에서 25~70% 틀린다.
  도통각이 전압 왜곡에 초선형으로 반응하기 때문. 따라서
    - 생성기: forward(..., models=...) 로 시뮬레이터를 고정점 안에서 직접 호출 (3기기 0.1초)
    - Y 행렬: 감도 구조 분석, 판별력 진단(gain projection)에만 사용
  Xie 2024 §I 의 "전압 왜곡이 크면 Norton 모델 오차가 커진다" 가 우리 조건에 해당.

설계 근거: CIRCUIT_FCM_GUIDE.md §2, §5, §6.  검증: fcm_check.py
"""
import pickle
import numpy as np
from circuit3 import _core3
from circuit5 import _core5

F = 60.0
NPC = 3072          # 시뮬 해상도 (fit_raw 는 256×12 업샘플 = 3072 로 동일)
NCYC = 24
NDFT = 4
N = NCYC * NPC
DT = 1.0 / (F * NPC)
T = np.arange(N) * DT
H = np.arange(1, 16)


# ---------------------------------------------------------------- 전압 관례
def wave_from_harmonics(V15, odd_only=True):
    """
    (15,) complex 전압 고조파 -> 시간영역 파형 (N 샘플, NCYC 주기)
    관례: V(h) = |V_h| exp(j·(arg V_h − h·arg V_1)),  v(t) = Σ √2|V_h| cos(hωt + φ_h)
    V_h 는 RMS. 짝수 차수는 계측 아티팩트이므로 기본 제외 (가이드 §10.3).
    """
    v = np.zeros(N)
    for h in range(1, 16):
        if odd_only and h % 2 == 0:
            continue
        a = abs(V15[h - 1]); ph = np.angle(V15[h - 1])
        v += np.sqrt(2) * a * np.cos(2 * np.pi * F * h * T + ph)
    return v


def harmonics_from_wave(x, vsrc, normalize=False):
    """마지막 NDFT 주기 DFT.  전류/전압 모두 전압 기본파 기준 위상.  반환 (15,) complex (RMS)."""
    sl = slice(N - NDFT * NPC, N)
    Xh = np.fft.rfft(x[sl])[NDFT:NDFT * 16:NDFT] / (NDFT * NPC) * 2 / np.sqrt(2)
    Vh = np.fft.rfft(vsrc[sl])[NDFT:NDFT * 16:NDFT]
    s = np.abs(Xh) * np.exp(1j * (np.angle(Xh) - H * np.angle(Vh[0])))
    return s / s[0] if normalize else s


# ---------------------------------------------------------------- 모델
class FCM:
    def __init__(self, device, topo, params, Vf=1.4):
        self.device, self.topo, self.params, self.Vf = device, topo, tuple(params), Vf

    @classmethod
    def from_pickle(cls, path):
        d = pickle.load(open(path, 'rb'))
        return cls(d['device'], d['topo'], d['params'])

    # --- 시뮬레이터 (전류 A, 전압 기준 위상)
    def simulate(self, p, V15):
        vsrc = wave_from_harmonics(V15)
        if self.topo == 'v3':
            C, R, L, Cx, rd = self.params
            I = _core3(vsrc, DT, p, C, max(R, 1e-3), L, Cx, self.Vf, max(rd, 1e-3), vsrc.max() - self.Vf)
        else:
            C, L2, R2, Cx, rd, Rc, I0 = self.params; Rs = R2 + Rc
            for _ in range(4):
                I, ir = _core5(vsrc, DT, p, C, L2, max(Rs, 1e-3), Cx, self.Vf, max(rd, 1e-3), vsrc.max() - self.Vf)
                if not np.all(np.isfinite(I)):
                    return None
                irms = np.sqrt(np.mean(ir[-NDFT * NPC:] ** 2))
                Rs = 0.5 * Rs + 0.5 * (R2 + Rc * np.exp(-irms / max(I0, 1e-4)))
            I, _ = _core5(vsrc, DT, p, C, L2, max(Rs, 1e-3), Cx, self.Vf, max(rd, 1e-3), vsrc.max() - self.Vf)
        if not np.all(np.isfinite(I)):
            return None
        return harmonics_from_wave(I, vsrc)

    # --- HNE 추출
    def extract(self, p, V_base, rel=0.01, odd_only=True):
        """
        J(h)   : V_base 에서의 전류 (15,) complex
        Y(h,k) : −∂I_h/∂V_k  (15,15) complex.  실수·허수 섭동을 평균하면 Y⁺ 가 나온다 (Y⁻ 무시, Xie Fig.4).
        odd_only: 짝수 k 열은 섭동하지 않고 0 으로 둔다 (반파 대칭 입력 가정)
        """
        V_base = np.asarray(V_base, complex)
        J = self.simulate(p, V_base)
        if J is None:
            raise RuntimeError('시뮬레이션 발산 (p=%.1f)' % p)
        Y = np.zeros((15, 15), complex)
        dV = rel * abs(V_base[0])
        for k in range(1, 16):
            if odd_only and k % 2 == 0:
                continue
            acc = np.zeros(15, complex)
            for ph in (0.0, np.pi / 2):
                Vp = V_base.copy(); Vp[k - 1] += dV * np.exp(1j * ph)
                Ip = self.simulate(p, Vp)
                if Ip is None:
                    raise RuntimeError('섭동 시뮬레이션 발산 (k=%d)' % k)
                acc += -(Ip - J) / (dV * np.exp(1j * ph))
            Y[:, k - 1] = acc / 2
        return J, Y

    def predict(self, p, V15, J=None, Y=None, V_base=None):
        """선형화 예측  I ≈ J − Y·(V − V_base).   J,Y,V_base 를 주면 재추출 안 함."""
        if J is None:
            J, Y = self.extract(p, V15); V_base = V15
        return J - Y @ (np.asarray(V15, complex) - np.asarray(V_base, complex))


# ---------------------------------------------------------------- 표 (전력별)
def build_table(fcm, powers, V_base, **kw):
    """전력 격자에서 J, Y 를 미리 뽑아 보간용 표를 만든다."""
    tab = []
    for p in powers:
        J, Y = fcm.extract(p, V_base, **kw)
        tab.append(dict(p=float(p), J=J, Y=Y))
    return dict(device=fcm.device, V_base=np.asarray(V_base, complex), table=tab)


def lookup(tab, p):
    """가장 가까운 두 전력 사이 선형 보간."""
    ps = np.array([t['p'] for t in tab['table']])
    if p <= ps[0]:
        return tab['table'][0]['J'], tab['table'][0]['Y']
    if p >= ps[-1]:
        return tab['table'][-1]['J'], tab['table'][-1]['Y']
    i = np.searchsorted(ps, p); a, b = tab['table'][i - 1], tab['table'][i]
    w = (p - a['p']) / (b['p'] - a['p'])
    return (1 - w) * a['J'] + w * b['J'], (1 - w) * a['Y'] + w * b['Y']


# ---------------------------------------------------------------- 생성기
def forward(powers, R_line, L_line, V_src, models=None, tables=None, n_iter=3):
    """
    여러 기기가 공유 임피던스를 통해 결합된 총 전류.
        V_term = V_src − Z(h)·I_total,   Z(h) = R + jωhL
        I_total = Σᵢ Iᵢ(p_i, V_term)
    powers : {device: W}
    models : {device: FCM}   -> 시뮬레이터 직접 호출 (기본, 정확, 3기기 약 0.1초)
    tables : {device: build_table 결과} -> 선형화 J − Y·ΔV (빠르지만 h9 이상 부정확. 진단용)
    둘 다 주면 models 우선.
    """
    if models is None and tables is None:
        raise ValueError('models 또는 tables 필요')
    V_src = np.asarray(V_src, complex)
    Z = R_line + 1j * 2 * np.pi * F * H * L_line
    def dev_current(d, p, V):
        if models and d in models:
            I = models[d].simulate(p, V)
            if I is None:
                raise RuntimeError('시뮬레이션 발산: %s @ %.1fW' % (d, p))
            return I
        J, Y = lookup(tables[d], p)
        return J - Y @ (V - tables[d]['V_base'])
    I_tot = sum(dev_current(d, p, V_src) for d, p in powers.items())
    V_term = V_src
    for _ in range(n_iter):
        V_term = V_src - Z * I_tot
        I_tot = sum(dev_current(d, p, V_term) for d, p in powers.items())
    return I_tot, V_term
