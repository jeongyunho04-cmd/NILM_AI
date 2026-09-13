import sys, json
sys.path.insert(0,'.')
from src import env_guard
import numpy as np
from pathlib import Path
C=Path('cache/seqraw_v1'); meta=json.loads((C/'meta.json').read_text(encoding='utf-8'))
apps=meta['appliances']
grid=np.arange(meta['target_offset'], int(meta['record_s']*60)-13*60-1, int(meta['grid_s']*60))
raw=np.load(C/'raw.npy',mmap_mode='r'); yon=np.load(C/'y_on.npy',mmap_mode='r')
ypw=np.load(C/'y_power.npy',mmap_mode='r'); yst=np.load(C/'y_state.npy',mmap_mode='r')
SHORT={'electiric_kettle':'포트','hair_dryer':'드라이기','hotplate':'핫플','oven':'오븐',
       'beam_projector':'프로젝터','laptop_charger':'충전기','minipc':'미니PC'}
N=900
V=[];P=[];S=[];K=[]
for i in range(N):
    r=np.asarray(raw[i][32][grid])          # 격자 시각의 V 대리
    yo=np.asarray(yon[i]).astype(bool); yp=np.asarray(ypw[i]); st=np.asarray(yst[i])
    for k in range(len(apps)):
        m=yo[:,k]&(yp[:,k]>5)
        if not m.any(): continue
        V.append(r[m]); P.append(yp[m,k]); S.append(st[m,k]); K.append(np.full(m.sum(),k))
V=np.concatenate(V);P=np.concatenate(P);S=np.concatenate(S);K=np.concatenate(K)
print('합성 캐시 안에서 **참값 전력이 전압을 따라가나** — log P = k·log V + c')
print('  저항이면 k=2 여야 한다. k≈0 이면 생성기가 녹화 전력을 그대로 되뱉는 것이다.')
print('  %-10s %8s %8s %9s %9s | %8s'%('기기','상태','n','V p5~p95','P 중앙','k'))
for kk in range(len(apps)):
    nm=SHORT.get(apps[kk],apps[kk][:8])
    for s in np.unique(S[K==kk]):
        m=(K==kk)&(S==s)
        if m.sum()<400: continue
        v=V[m]; p=P[m]
        good=(v>50)&(p>5)
        if good.sum()<400 or v[good].std()/np.median(v[good])<0.01: continue
        slope=float(np.polyfit(np.log(v[good]),np.log(p[good]),1)[0])
        print('  %-10s %8d %8d %4.0f~%-4.0f %9.0f | %8.2f'
              %(nm,s,good.sum(),*np.percentile(v[good],[5,95]),np.median(p[good]),slope))
