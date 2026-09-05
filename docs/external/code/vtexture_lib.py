"""
vtexture_lib.py — 세션 전압 텍스처 라이브러리 (생성기 랜덤화용)

근거 (Claude Code ①b + 가이드 §11.1 정정):
  조합 V 와 단독 V 의 차이 0.9% 중 Z·I 는 1/12. 나머지는 세션 텍스처.
  단순 중첩 13.5% → +결합 12.7% → +텍스처 2.7%.  지배 축은 텍스처다.
  텍스처 차이가 서명을 12~27° 돌리고 Z(0~400µH) 는 7° 이하.  랜덤화 예산을 큰 축에.

사용:
  lib = TextureLib.from_raw_files(paths, session_key=lambda p: ...)   # 세션별 h3~h127 스펙트럼
  V   = lib.sample(rng, V1_rms=237.0)        # (15,) 또는 (129,) complex — 한 세션 텍스처를 뽑아 V1 에 얹음
  V   = lib.sample(rng, V1_rms, mix=True)    # 두 세션의 볼록 결합 (보간)
  lib.save('vtexture_C.npz')

각 세션 텍스처는 그 세션 파일들의 복소 평균 (V1 정규화, h배 위상 관례, 홀수만).
기기 부하가 만드는 성분은 세션 안에서 상쇄되지 않으므로 ── 같은 세션에 부하가 다른 파일이
여럿 있으면 평균이 부하 성분을 약화시킨다. 저항 부하(포트) 스냅샷이 있으면 그것이 최선.
"""
import numpy as np, pandas as pd, datetime
from vtemplate import VTemplate, HMAX

def kst(path):
    ms=int(path.split('/')[-1].split('_')[0]); return datetime.datetime.utcfromtimestamp(ms/1000+9*3600)

class TextureLib:
    def __init__(self, sessions):  # {key: (T (HMAX+1,) complex, meta)}
        self.sessions=sessions; self.keys=sorted(sessions)
    @classmethod
    def from_raw_files(cls, paths, session_key=None, hmin=3, deembed=False):
        if session_key is None: session_key=lambda p: kst(p).strftime('%m-%d %H')
        grp={}
        for p in paths:
            T=VTemplate.from_raw(p,hmin=hmin,deembed=deembed)
            grp.setdefault(session_key(p),[]).append((T.T,T.meta))
        sess={}
        for k,items in grp.items():
            Tm=np.mean([t for t,_ in items],0); Tm[1]=1.0
            sess[k]=(Tm,dict(n_files=len(items),files=[m['source'] for _,m in items],
                             hi_pct=float(100*np.sqrt(np.sum(abs(Tm[17:32:2])**2))),
                             h3=complex(Tm[3]),h5=complex(Tm[5])))
        return cls(sess)
    def sample(self, rng, V1_rms, mix=False, n_out=15):
        ks=self.keys
        if mix and len(ks)>=2:
            a,b=rng.choice(len(ks),2,replace=False); w=rng.uniform()
            T=w*self.sessions[ks[a]][0]+(1-w)*self.sessions[ks[b]][0]
        else:
            T=self.sessions[ks[rng.integers(len(ks))]][0]
        X=T[:n_out+1]*V1_rms
        return X[1:] if n_out==15 else X
    def save(self,path):
        np.savez(path,keys=np.array(self.keys),T=np.array([self.sessions[k][0] for k in self.keys]),
                 meta=np.array([str(self.sessions[k][1]) for k in self.keys]))
    @classmethod
    def load(cls,path):
        z=np.load(path,allow_pickle=True)
        return cls({k:(T,eval(m)) for k,T,m in zip(z['keys'],z['T'],z['meta'])})
    def report(self):
        print('%-12s %3s %8s %14s %14s %8s'%('세션','n','vh3%','∠3','∠5','h17+%'))
        for k in self.keys:
            T,m=self.sessions[k]
            print('%-12s %3d %7.2f%% %13.0f° %13.0f° %7.3f%%'%(k,m['n_files'],100*abs(T[3]),np.degrees(np.angle(T[3])),np.degrees(np.angle(T[5])),m['hi_pct']))

if __name__=='__main__':
    import glob, sys
    paths=sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else '/mnt/user-data/uploads/*raw_*.csv'))
    lib=TextureLib.from_raw_files(paths); lib.report(); lib.save('vtexture_C.npz'); print('-> vtexture_C.npz')
