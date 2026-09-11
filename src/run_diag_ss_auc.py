import sys, json
sys.path.insert(0,'.')
from src import env_guard
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
C=Path('cache/seqraw_v1'); meta=json.loads((C/'meta.json').read_text(encoding='utf-8'))
apps=meta['appliances']; N=1500; K=len(apps); T=117
obs=np.asarray(np.load(C/'obs_harm.npy',mmap_mode='r')[:N])
yon=np.asarray(np.load(C/'y_on.npy',mmap_mode='r')[:N]).astype(bool)
ypl=np.asarray(np.load(C/'y_plugged.npy',mmap_mode='r')[:N]).astype(bool)
ypw=np.asarray(np.load(C/'y_power.npy',mmap_mode='r')[:N]).astype(np.float64)
km=apps.index('minipc'); sib=[apps.index(x) for x in ('laptop_charger','beam_projector')]
O=(obs[...,0]+1j*obs[...,1]).reshape(-1,15); W=ypw.reshape(-1,K)
I=(ypl&~yon).astype(np.float64).reshape(-1,K); ON=yon.reshape(-1,K)
rec=np.repeat(np.arange(N),T)
NB=3; bands=np.zeros_like(W,dtype=np.int8)
for j in range(K):
    v=W[:,j]; m=v>1
    if m.sum()>100: bands[:,j]=np.digitize(v,np.quantile(v[m],[1/3,2/3]))
# 미니PC 를 뺀 사전 (그 기여가 남아 있어야 판별이 된다)
oth=[j for j in range(K) if j!=km]
Pb=np.stack([W[:,j]*(bands[:,j]==q) for j in oth for q in range(NB)],1)
A=np.concatenate([Pb,I[:,oth],np.ones((len(W),1))],1)
rng=np.random.RandomState(0); rr=np.arange(N); rng.shuffle(rr)
tr_r=set(rr[:N*3//4].tolist()); trm=np.array([r in tr_r for r in rec])
sel=(ON[:,sib].sum(1)>0)            # 형제가 켜진 단계 (어려운 자리)
ORD=[1,3,5,7,9,11,13,15]; oi=[o-1 for o in ORD]
Rm=np.zeros((len(O),len(ORD)),complex)
for jj,j in enumerate(oi):
    y=O[:,j]
    # 미니PC OFF **학습 기록**에서만 전역 사전을 맞춘다 (미니PC 기여가 사전에 안 들어가게)
    f=trm&(~ON[:,km])
    wr,*_=np.linalg.lstsq(A[f],np.real(y)[f],rcond=None)
    wi,*_=np.linalg.lstsq(A[f],np.imag(y)[f],rcond=None)
    Rm[:,jj]=(np.real(y)-A@wr)+1j*(np.imag(y)-A@wi)
def auc(R,tag):
    X=np.concatenate([np.abs(R),np.angle(R*np.conj(R[:,[0]]+1e-12))],1)
    X=(X-X.mean(0))/(X.std(0)+1e-9)
    m=sel; Y=ON[:,km].astype(int)
    tr=m&trm; te=m&(~trm)
    lr=LogisticRegression(max_iter=6000,C=0.5).fit(X[tr],Y[tr])
    print('   %-44s AUC %.3f'%(tag,roc_auc_score(Y[te],lr.predict_proba(X[te])[:,1])))
print('미니PC 판별 — 잔차 벡터로 (형제 켜진 단계, 환경 밖 채점)')
auc(Rm,'① 전역 사전 (전력구간별 포함)')
# 기록별 지문 보정: 그 기록의 **미니PC 참 OFF** 단계로만 보정을 맞춘다 (신탁 = 귀속용)
Rc=Rm.copy()
for i in range(N):
    sl=slice(i*T,(i+1)*T)
    f=~ON[sl,km]
    if f.sum()<8: continue
    Wi=W[sl]; cols=[c for c in oth if (Wi[f,c]>1).sum()>=6]
    if not cols: continue
    Xa=Wi[:,cols]/100.0; Xf=Xa[f]
    G=Xf.T@Xf; G=G+1e-3*np.trace(G)/len(cols)*np.eye(len(cols))
    for jj in range(len(ORD)):
        br=np.linalg.solve(G,Xf.T@np.real(Rm[sl,jj][f]))
        bi=np.linalg.solve(G,Xf.T@np.imag(Rm[sl,jj][f]))
        Rc[sl,jj]=Rm[sl,jj]-(Xa@br+1j*(Xa@bi))
auc(Rc,'② + **기록별 지문 보정** (신탁 · 귀속용 상한)')
print('\n   참고 13.84.33: 기본특징 0.648 · 기록 안 채점 0.767 · 기록마다 경계 재적합 0.879')
