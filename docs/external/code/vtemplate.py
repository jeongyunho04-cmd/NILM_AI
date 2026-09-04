"""
vtemplate.py — 전압 고차(h17+) 템플릿: V15 에서 전대역 전압 파형 복원
=======================================================================

문제 (가이드 §10.11, Claude Code P1):
  실시간 입력은 전압 고조파 15개. h15 로 자르면 피크가 +0.6~0.9V 오르고 도통각이
  2° 바뀌어 혼합 재현이 5.5% → 9.0% 로 나빠진다. h17~h31 은 V1 의 0.27~0.31% 뿐이지만
  도통각이 전압 왜곡에 초선형이라 무시할 수 없다.

왜 함수형 사전분포는 안 되는가:
  장소 C 의 h17~h31 은 단조 감쇠가 아니다 (h23 0.14%, h29 0.12% 봉우리, h17 0.01%).
  지수 외삽·평탄화 모델·Lanczos 전부 실패 (p1_vprior.py).
  선로 결합 Z·I_pulse 도 아니다 — R=0.52Ω 시간영역 결합으로 0.31% 중 0.02% 만 복원.
  단독 파일에도 0.27% 가 있으니 계통 자체의 고차 텍스처다.

왜 경험적 템플릿은 되는가:
  같은 장소 18개 파일(2일, 여러 시간대)에서 h21~h31 의 파일 간 CV 가 0.33~0.42 —
  h11~h13 과 같은 수준. 위상도 일관 (h23 +19°, h29 +94°). 집합 부하와 아날로그
  체인이 만드는 그 장소의 고정 텍스처.

검증 (장소 C 혼합, 전류 재현 RMS h≤15):
                    V전체    V15     V15+템플릿   h21절단(펌웨어)
  smps3_1           5.5%    9.0%      6.3%         6.6%
  beam_charger_1    4.9%    9.2%      6.7%         5.3%
  beam_minipc_1     6.2%    4.6%      6.2%         5.6%
  -> 펌웨어 h21 발행과 같은 수준을 펌웨어 없이 얻는다.

사용:
  # 1회 (장소당): 세션 정렬이 검증된(n_shift=0) 원시 스냅샷에서
  T = VTemplate.from_raw('raw_beam_projector_1.csv'); T.save('vtemplate_C.npz')
  # 매 프레임
  T = VTemplate.load('vtemplate_C.npz')
  X_full = T.extend(V15)            # (129,) complex, h0..h128, V1 기준 위상 — fcm.wave_from_harmonics 등에 투입
  v_t    = T.wave(V15, n=3072)      # 시간영역 한 주기

조건 (2026-09-05 개정 — Claude Code 회신 §3 반영):
  (a) 템플릿 스냅샷의 V–I 정렬이 0 이어야 한다 (저항 부하 ihdeg1≈0 자가진단, 가이드 §10.9).
      오염된 파일이 섞인 평균 템플릿은 단일 깨끗한 스냅샷보다 나쁘다 (7.5% vs 6.3%).
  (b) **템플릿은 장소가 아니라 세션에 고정된다.** 같은 시각에 녹화된 파일은 켜진 기기와
      무관하게 같은 텍스처(00:36 세션 6개: h17 전부 ∠−64°)이고, 1시간 떨어진 세션은
      거리 0.7~1.1 (참조 크기와 같다). 세션 내 거리는 0.33~0.41.
        같은 세션 템플릿   5.8%   (전체 파형 5.5%)
        1시간 전 템플릿    5.9%   (이 파이프라인. 저쪽 파이프라인에서는 실패 — 규약 차이)
      → 운용: 10~15분마다 원시 스냅샷(20주기, 약 2초)을 받아 템플릿 갱신. 펌웨어 h21
        상시 발행(요청 A)보다 가벼운 요청 A′.
  (c) 차수를 줄이지 말 것. h23·h29 만 남기면 6.5% 로 나빠진다 — h17~h21 이 흔들려도
      피크 형상 정보를 갖고 있다.
  (d) 저쪽 규약(전압 역RC)에 맞추려면 from_raw(deembed=True). 효과 0.3%p.
"""
import numpy as np, pandas as pd

HMAX = 128


class VTemplate:
    def __init__(self, T, meta=None):
        """T: (HMAX+1,) complex, V1 으로 정규화된 스펙트럼 (T[1]=1). h0, 짝수는 0."""
        self.T = np.asarray(T, complex); self.meta = meta or {}

    # ------------------------------------------------------------ 생성
    @classmethod
    def from_raw(cls, csv_path, hmin=17, symmetrize=True, deembed=False, fc=1591.55):
        """
        deembed=True: 계측 RC(1kΩ·100nF, 1극 저역통과 fc) 를 전압에서 벗긴 규약으로 변환.
          T_true(h) = T_filt(h) · |H(1)| · (1 + j·60h/fc) · exp(−j·h·atan(60/fc))
          (h배 위상 관례에서 V1 의 RC 위상 이동 h·atan(60/fc) 이 함께 빠진다.)
          크기 h17 ×1.19 → h31 ×1.54, 위상 −4° → −17°. Claude Code 회신 §2 와 소수점까지 일치.
        """
        d = pd.read_csv(csv_path)
        v = d.v_v.values.astype(float); rg = d['range'].values
        fs = float(d.fs_hz.iloc[0]); npc = int(round(fs / 60.0))
        k0 = np.argmax(v[:npc]); st = (k0 - npc // 4) % npc
        cyc = [v[c:c + npc] for c in range(st, len(v) - npc, npc) if rg[c:c + npc].std() == 0]
        V = np.median(cyc, 0)
        if symmetrize:
            V = (V - np.roll(V, npc // 2)) / 2                    # 짝수차(아날로그 아티팩트) 제거
        X = np.fft.rfft(V) / npc * 2
        h = np.arange(len(X)); Xn = X * np.exp(-1j * h * np.angle(X[1])) / abs(X[1])
        T = np.zeros(HMAX + 1, complex); n = min(len(Xn), HMAX + 1)
        T[:n] = Xn[:n]; T[:hmin] = 0; T[::2] = 0; T[1] = 1.0
        if deembed:
            hh = np.arange(HMAX + 1); H1 = 1 / np.sqrt(1 + (60.0 / fc) ** 2)
            T *= H1 * (1 + 1j * 60.0 * hh / fc) * np.exp(-1j * hh * np.arctan(60.0 / fc)); T[1] = 1.0
        meta = dict(source=csv_path.split('/')[-1], n_cycles=len(cyc), npc=npc, deembed=deembed,
                    V1_rms=float(abs(X[1]) / np.sqrt(2)),
                    hi_content_pct=float(100 * np.sqrt(np.sum(abs(T[hmin:32:2]) ** 2))))
        return cls(T, meta)

    @classmethod
    def from_raw_files(cls, paths, hmin=17):
        """여러 스냅샷의 복소 평균. 전부 정렬 0 이 확인된 파일만 넣을 것."""
        Ts = [cls.from_raw(p, hmin).T for p in paths]
        T = np.mean(Ts, 0); T[1] = 1.0
        return cls(T, dict(source='mean of %d' % len(paths), files=[p.split('/')[-1] for p in paths]))

    # ------------------------------------------------------------ 저장
    def save(self, path): np.savez(path, T=self.T, meta=np.array([str(self.meta)]))
    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True); return cls(z['T'], eval(str(z['meta'][0])))

    # ------------------------------------------------------------ 적용
    def extend(self, V15):
        """
        V15: (15,) complex, h1..h15, V1 기준 위상 (fcm.py 관례), RMS 또는 피크 어느 쪽이든 그대로 유지.
        반환 (HMAX+1,) complex, h0..h128. h1..h15 는 입력 그대로, h17+ 는 템플릿 × V15[0].
        """
        V15 = np.asarray(V15, complex); X = np.zeros(HMAX + 1, complex)
        X[1:16] = V15; X[17:] = self.T[17:] * V15[0]
        return X

    def wave(self, V15, n=3072, f=60.0, ncyc=1, is_rms=True):
        """시간영역 파형 (cos 관례). is_rms=True 면 V15 를 RMS 로 보고 √2 배."""
        X = self.extend(V15); t = np.arange(n * ncyc) / n
        s = np.sqrt(2) if is_rms else 1.0; v = np.zeros(n * ncyc)
        for h in range(1, HMAX + 1, 2):
            if X[h] == 0: continue
            v += s * abs(X[h]) * np.cos(2 * np.pi * h * t + np.angle(X[h]))
        return v

    def __repr__(self):
        return 'VTemplate(%s, h17+ = %.3f%% of V1)' % (self.meta.get('source', '?'), self.meta.get('hi_content_pct', float('nan')))


# ---------------------------------------------------------------- 검증 도구
def stability_report(paths, hs=(17, 19, 21, 23, 25, 27, 29, 31)):
    """여러 스냅샷에서 h17+ 의 파일 간 산포. CV 가 0.5 를 넘는 차수는 템플릿 신뢰도 낮음."""
    Ts = np.array([VTemplate.from_raw(p).T for p in paths]); m = Ts.mean(0)
    print('%-22s' % 'file' + ''.join('%9s' % ('h%d' % h) for h in hs))
    for p, T in zip(paths, Ts):
        print('%-22s' % p.split('/')[-1][-22:] + ''.join('%5.2f∠%+4.0f' % (100 * abs(T[h]), np.degrees(np.angle(T[h]))) for h in hs))
    cv = [np.abs(Ts[:, h] - m[h]).std() / max(abs(m[h]), 1e-12) for h in hs]
    print('%-22s' % '복소평균' + ''.join('%5.2f∠%+4.0f' % (100 * abs(m[h]), np.degrees(np.angle(m[h]))) for h in hs))
    print('%-22s' % 'CV' + ''.join('%9.2f' % c for c in cv))
    return m, cv


if __name__ == '__main__':
    import sys, glob
    if len(sys.argv) < 2:
        print(__doc__); sys.exit()
    if sys.argv[1] == 'stability':
        stability_report(sorted(glob.glob(sys.argv[2])))
    else:
        T = VTemplate.from_raw(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else 'vtemplate.npz'
        T.save(out); print(T, '->', out)
