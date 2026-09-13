import sys, json
sys.path.insert(0,'.')
from src import env_guard
import numpy as np
from pathlib import Path
C=Path('cache/seqraw_v1'); meta=json.loads((C/'meta.json').read_text(encoding='utf-8'))
apps=meta['appliances']; N=1500
obs=np.asarray(np.load(C/'obs_harm.npy',mmap_mode='r')[:N])
yon=np.asarray(np.load(C/'y_on.npy',mmap_mode='r')[:N]).astype(bool)
ypl=np.asarray(np.load(C/'y_plugged.npy',mmap_mode='r')[:N]).astype(bool)
ypw=np.asarray(np.load(C/'y_power.npy',mmap_mode='r')[:N]).astype(np.float64)
T=yon.shape[1]; K=len(apps)
O=(obs[...,0]+1j*obs[...,1]).reshape(-1,15)
W=ypw.reshape(-1,K); I=(ypl&~yon).astype(np.float64).reshape(-1,K)
NB=3; bands=np.zeros_like(W,dtype=np.int8)
for j in range(K):
    v=W[:,j]; m=v>1
    if m.sum()>100: bands[:,j]=np.digitize(v,np.quantile(v[m],[1/3,2/3]))
Pb=np.stack([W[:,j]*(bands[:,j]==q) for j in range(K) for q in range(NB)],1)
A=np.concatenate([Pb,I,np.ones((len(W),1))],1)
rng=np.random.RandomState(0); rr=np.arange(N); rng.shuffle(rr)
tr_r=set(rr[:N*3//4].tolist())
rec=np.repeat(np.arange(N),T); tr=np.array([r in tr_r for r in rec])
from src.model.net import harmonic_signatures
from src.synthesis.segment_pool import SegmentPool
pool=SegmentPool(npz_dir='processed_data/npz',time_split='train'); sg=harmonic_signatures(pool,apps); del pool
km=apps.index('minipc')
te_recs=rr[N*3//4:]
print('기록마다 **그 기록에 켜진 기기의 지문 보정**만 (능형, 앞절반 학습 뒷절반 채점)')
print('%-5s %11s %14s %8s | %10s %7s'%('차수','② 전력구간','+기록별 지문','줄임','미니PC12W','바닥'))
LAM=1e-3
for o in [1,3,5,7,9,11,13,15]:
    j=o-1; y=O[:,j]
    wr,*_=np.linalg.lstsq(A[tr],np.real(y)[tr],rcond=None)
    wi,*_=np.linalg.lstsq(A[tr],np.imag(y)[tr],rcond=None)
    r=(np.real(y)-A@wr)+1j*(np.imag(y)-A@wi)
    base,corr=[],[]
    for i in te_recs:
        sl=slice(i*T,(i+1)*T); ri=r[sl]; Wi=W[sl]; h=T//2
        cols=[c for c in range(K) if (Wi[:h,c]>1).sum()>=8 and (Wi[h:,c]>1).sum()>=4]
        base.append(np.abs(ri[h:]))
        if not cols: corr.append(np.abs(ri[h:])); continue
        Xa=Wi[:,cols]/100.0
        sc=Xa[:h].T@Xa[:h]+LAM*np.trace(Xa[:h].T@Xa[:h])/len(cols)*np.eye(len(cols))
        br=np.linalg.solve(sc,Xa[:h].T@np.real(ri[:h]))
        bi=np.linalg.solve(sc,Xa[:h].T@np.imag(ri[:h]))
        corr.append(np.abs(ri[h:]-(Xa[h:]@br+1j*(Xa[h:]@bi))))
    b=1000*np.median(np.concatenate(base)); cc=1000*np.median(np.concatenate(corr))
    mp=1000*abs(sg[km,j,0]+1j*sg[km,j,1])*12.0
    print('h%-4d %10.1f %13.1f %7.0f%% | %9.1f %7s'%(o,b,cc,100*(cc-b)/b,mp,'3~4'))
