"""
fcm12.py — v12g 회로 모델의 시뮬레이터/생성기 (fcm.py 대체, 2026-09-06)

  M = FCM.from_pickle('circ12_laptop_charger.pkl')
  I15 = M.simulate(45.0, V15)                 # 계측 영역 (RC τ 적용) h1..h15, h배 위상 관례
  I15 = M.simulate(45.0, V15, measured=False) # 회로 참전류 (결합 고정점 내부용)
  I_tot, V_term = forward({'laptop_charger':45,'minipc':18,'beam_projector':47}, R_line, L_line, V_src, models=Ms)

규약
- V15 는 참 전압 (계측 RC 벗긴 것). 단일 입력 ADC 데이터라면 전압에 짝수 아티팩트가 없으니 대칭화 불필요.
- 출력은 기본 measured=True: 펌웨어 ih/ihdeg 와 같은 영역. 학습 데이터 생성은 이걸로.
- G 처리: pkl 의 G 에는 계측 배경(자체전원 충전기, ≈0.051mS ≈ 2.4W@217V)이 섞여 있다.
  기기 단독 서명은 G_dev = G − G_bg 로 (include_bg=False 기본). 배경까지 재현하려면 include_bg=True.
- R 랜덤화: R_files 에 세션 안에서 본 R 의 범위가 있다 (충전기 2.4~3.3Ω = NTC 온도 상태). 생성기는 이 범위에서 뽑는다.
- 결합: V_term = V_src − (R_line + jωL_line)·I_tot 을 참전류로 2~3회 고정점. 선형화(Y 행렬)는 쓰지 않는다 (h9+ 발산, §11.5).
"""
import numpy as np, pickle
try:                                   # 패키지(circuit_model.fcm12)로도, 단독 실행으로도 되게
    from .circuit12 import sim_harmonics, F
except ImportError:
    from circuit12 import sim_harmonics, F

H = np.arange(1, 16)
G_BG_DEFAULT = 0.051e-3        # 자체전원 충전기 배경의 동상 컨덕턴스 (11mA/217V)


class FCM:
    def __init__(self, device, params, tau=60e-6, R_range=None, G_bg=G_BG_DEFAULT):
        self.device = device; self.params = tuple(params); self.tau = tau
        self.R_range = R_range or (self.params[1], self.params[1]); self.G_bg = G_bg

    @classmethod
    def from_pickle(cls, path):
        d = pickle.load(open(path, 'rb'))
        Rs = list(d.get('R_files', {}).values()) or [d['params'][1]]
        return cls(d['device'], d['params'], d.get('tau', 60e-6), (min(Rs), max(Rs)))

    def _params(self, R=None, include_bg=False):
        C, R0, L0, Isat, Cx, rd, G = self.params
        if R is None:
            R = R0
        if not include_bg:
            G = max(G - self.G_bg, 0.0)
        return (C, R, L0, Isat, Cx, rd, G)

    def simulate(self, p, V15, measured=True, R=None, include_bg=False):
        """전력 p [W] (기기 총전력, 배경 제외) → I h1..h15 (RMS, h배 관례).  실패 시 None."""
        return sim_harmonics(p, V15, self._params(R, include_bg), tau=self.tau, measured=measured)

    def sample_R(self, rng):
        lo, hi = self.R_range
        return rng.uniform(lo, hi)


def forward(powers, R_line, L_line, V_src, models, n_iter=3, measured=True, Rs=None, include_bg=False):
    """
    다기기 결합 생성기. powers: {device: P}, V_src: h1..h15 (참 전압, 소스), models: {device: FCM}.
    Rs: {device: R} 세션 상태 (없으면 명목값). 반환 (I_tot 계측영역, V_term 참전압).
    """
    Z = R_line + 1j * 2 * np.pi * F * H * L_line
    I_tot = np.zeros(15, complex); V_term = V_src.copy()
    for _ in range(n_iter):
        V_term = V_src - Z * I_tot
        I_tot = sum(models[d].simulate(p, V_term, measured=False, R=(Rs or {}).get(d), include_bg=include_bg)
                    for d, p in powers.items())
    if measured:
        tau = next(iter(models.values())).tau
        I_tot = I_tot / (1.0 + 1j * 2 * np.pi * F * H * tau)
    return I_tot, V_term


if __name__ == '__main__':
    import time
    Ms = {d: FCM.from_pickle('circ12_%s.pkl' % d) for d in ('laptop_charger', 'minipc', 'beam_projector')}
    V15 = np.zeros(15, complex); V15[0] = 217.0; V15[2] = 217 * 0.02 * np.exp(-1j * np.deg2rad(110)); V15[4] = 217 * 0.017 * np.exp(-1j * np.deg2rad(160))
    for d, M in Ms.items():
        t0 = time.time(); I = M.simulate(40.0 if d != 'minipc' else 20.0, V15)
        print('%-15s R범위 %.2f~%.2f  |I| h1 %.3f h3 %.3f h5 %.3f h15 %.4f  ih3/ih1 %.2f  ∠h3 %+.0f°  (%.2fs)' % (
            d, *M.R_range, abs(I[0]), abs(I[2]), abs(I[4]), abs(I[14]), abs(I[2]) / abs(I[0]), np.degrees(np.angle(I[2])), time.time() - t0))
    t0 = time.time(); It, Vt = forward({'laptop_charger': 45, 'minipc': 18, 'beam_projector': 47}, 0.48, 100e-6, V15, Ms)
    print('forward 3기기: |I_tot| 홀수 %s  (%.2fs)' % (np.round(np.abs(It[::2]), 3), time.time() - t0))
