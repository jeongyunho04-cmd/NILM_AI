import sys, json
sys.path.insert(0,'.')
from src import env_guard
import numpy as np
from pathlib import Path
from itertools import combinations
C=Path('cache/seqraw_v1'); meta=json.loads((C/'meta.json').read_text(encoding='utf-8'))
apps=meta['appliances']; N=1500
obs=np.asarray(np.load(C/'obs_harm.npy',mmap_mode='r')[:N])
yon=np.asarray(np.load(C/'y_on.npy',mmap_mode='r')[:N]).astype(bool)
ypl=np.asarray(np.load(C/'y_plugged.npy',mmap_mode='r')[:N]).astype(bool)
ypw=np.asarray(np.load(C/'y_power.npy',mmap_mode='r')[:N]).astype(np.float64)
T=yon.shape[1]; O=obs[...,0]+1j*obs[...,1]
Y=O.reshape(-1,O.shape[-1]); W=ypw.reshape(-1,ypw.shape[-1])
I=(ypl&~yon).astype(np.float64).reshape(-1,ypw.shape[-1])
rec=np.repeat(np.arange(N),T); K=W.shape[1]
NB=3; bands=np.zeros_like(W,dtype=np.int8)
for j in range(K):
    v=W[:,j]; m=v>1
    if m.sum()>100:
        bands[:,j]=np.digitize(v,np.quantile(v[m],[1/3,2/3]))
one=np.ones((len(W),1))
Pb=np.stack([W[:,j]*(bands[:,j]==q) for j in range(K) for q in range(NB)],1)
Ptot=W.sum(1,keepdims=True)
A1=np.concatenate([W,I,one],1)
A2=np.concatenate([Pb,I,one],1)
A4=np.concatenate([Pb,I,one,W*Ptot/100.0],1)
PAIR=np.stack([W[:,i]*W[:,j]/100.0 for i,j in combinations(range(K),2)],1)
A5=np.concatenate([Pb,I,one,W*Ptot/100.0,PAIR],1)
rng=np.random.RandomState(0); rr=np.arange(N); rng.shuffle(rr)
tr_r=set(rr[:N*3//4].tolist()); tr=np.array([r in tr_r for r in rec]); te=~tr
def ev(A,y):
    out=np.zeros(te.sum(),complex)
    for p,get in ((0,np.real),(1,np.imag)):
        w,*_=np.linalg.lstsq(A[tr],get(y)[tr],rcond=None)
        e=get(y)[te]-A[te]@w
        out=out+(e if p==0 else 1j*e)
    return float(np.median(np.abs(out)))*1000
from src.model.net import harmonic_signatures
from src.synthesis.segment_pool import SegmentPool
pool=SegmentPool(npz_dir='processed_data/npz',time_split='train'); sg=harmonic_signatures(pool,apps); del pool
km=apps.index('minipc')
print('열수  ①%d  ②%d  ④%d  ⑤%d'%(A1.shape[1],A2.shape[1],A4.shape[1],A5.shape[1]))
print('%-5s %9s %11s %14s %14s | %9s'%('차수','① 지금','② 전력구간','④ +결합 PxPtot','⑤ +전쌍 PixPj','미니PC12W'))
for o in [1,3,5,7,9,11,13,15]:
    j=o-1; y=Y[:,j]
    r1,r2,r4,r5=ev(A1,y),ev(A2,y),ev(A4,y),ev(A5,y)
    mp=1000*abs(sg[km,j,0]+1j*sg[km,j,1])*12.0
    print('h%-4d %8.1f %10.1f %8.1f %4.0f%% %8.1f %4.0f%% | %8.1f'
          %(o,r1,r2,r4,100*(r4-r1)/r1,r5,100*(r5-r1)/r1,mp))
