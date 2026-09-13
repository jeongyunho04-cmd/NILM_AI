import sys, json, collections
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
T=yon.shape[1]; O=(obs[...,0]+1j*obs[...,1]).reshape(-1,15)
W=ypw.reshape(-1,len(apps)); ON=yon.reshape(-1,len(apps)); PL=ypl.reshape(-1,len(apps))
rec=np.repeat(np.arange(N),T)
STEP=5.0
key=collections.defaultdict(list)
for i in range(len(O)):
    k=(tuple(np.nonzero(ON[i])[0]), tuple(np.round(W[i][ON[i]]/STEP).astype(int)),
       tuple(np.nonzero(PL[i]&~ON[i])[0]))
    key[k].append(i)
groups=[v for v in key.values() if len(v)>=2 and len(set(rec[v]))>=2]
print('구성·전력(%gW 격자)·꽂힘이 같은 무리 %d개 (기록이 둘 이상인 것만)'%(STEP,len(groups)))
rng=np.random.RandomState(0)
same,diff=[],[]
for v in groups:
    v=np.asarray(v); r=rec[v]
    for _ in range(min(6,len(v))):
        a,b=rng.choice(len(v),2,replace=False)
        (same if r[a]==r[b] else diff).append(np.abs(O[v[a]]-O[v[b]])/np.sqrt(2))
same=np.asarray(same) if same else np.zeros((0,15)); diff=np.asarray(diff)
from src.model.net import harmonic_signatures
from src.synthesis.segment_pool import SegmentPool
pool=SegmentPool(npz_dir='processed_data/npz',time_split='train'); sg=harmonic_signatures(pool,apps); del pool
km=apps.index('minipc')
print('짝  같은 기록 %d · 다른 기록 %d'%(len(same),len(diff)))
print('\n%-5s %13s %13s %10s | %11s'%('차수','같은기록 안','다른기록 사이','차','미니PC12W'))
for o in [1,3,5,7,9,11,13,15]:
    j=o-1
    s=1000*np.median(same[:,j]) if len(same) else float('nan')
    d=1000*np.median(diff[:,j])
    mp=1000*abs(sg[km,j,0]+1j*sg[km,j,1])*12.0
    print('h%-4d %12.1f %13.1f %9.1f | %10.1f'%(o,s,d,d-s,mp))
print('\n같은기록 안 = 계측·시각 잡음 바닥 · 다른기록 사이 = 바닥 + **세션 이동**')
