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
rec=np.repeat(np.arange(N),T)
NB=3; bands=np.zeros_like(W,dtype=np.int8)
for j in range(K):
    v=W[:,j]; m=v>1
    if m.sum()>100: bands[:,j]=np.digitize(v,np.quantile(v[m],[1/3,2/3]))
Pb=np.stack([W[:,j]*(bands[:,j]==q) for j in range(K) for q in range(NB)],1)
A=np.concatenate([Pb,I,np.ones((len(W),1))],1)
rng=np.random.RandomState(0); rr=np.arange(N); rng.shuffle(rr)
tr_r=set(rr[:N*3//4].tolist()); tr=np.array([r in tr_r for r in rec]); te=~tr
from src.model.net import harmonic_signatures
from src.synthesis.segment_pool import SegmentPool
pool=SegmentPool(npz_dir='processed_data/npz',time_split='train'); sg=harmonic_signatures(pool,apps); del pool
km=apps.index('minipc')
print('기록마다 **상수 하나**(복소)를 더 주면 — 그 기록 앞절반으로 정하고 뒷절반 채점')
print('%-5s %11s %13s %9s | %11s %9s'%('차수','② 전력구간','+기록상수','줄임','미니PC12W','바닥'))
te_recs=rr[N*3//4:]
for o in [1,3,5,7,9,11,13,15]:
    j=o-1; y=O[:,j]
    w_r,*_=np.linalg.lstsq(A[tr],np.real(y)[tr],rcond=None)
    w_i,*_=np.linalg.lstsq(A[tr],np.imag(y)[tr],rcond=None)
    r=(np.real(y)-A@w_r)+1j*(np.imag(y)-A@w_i)
    base,corr=[],[]
    for i in te_recs:
        sl=slice(i*T,(i+1)*T); ri=r[sl]; h=T//2
        c=np.median(ri[:h].real)+1j*np.median(ri[:h].imag)
        base.append(np.abs(ri[h:])); corr.append(np.abs(ri[h:]-c))
    b=1000*np.median(np.concatenate(base)); cc=1000*np.median(np.concatenate(corr))
    mp=1000*abs(sg[km,j,0]+1j*sg[km,j,1])*12.0
    print('h%-4d %10.1f %12.1f %8.0f%% | %10.1f %8s'%(o,b,cc,100*(cc-b)/b,mp,'3~4'))
