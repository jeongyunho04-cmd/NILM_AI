import sys, json
sys.path.insert(0,'.')
from src import env_guard
import numpy as np, torch
from pathlib import Path
from src.model.inputs import build_inputs
from src.run_gate_check import load_model
dev='cuda' if torch.cuda.is_available() else 'cpu'
C=Path('cache/seqraw_v1'); meta=json.loads((C/'meta.json').read_text(encoding='utf-8'))
apps=meta['appliances']
grid=np.arange(meta['target_offset'], int(meta['record_s']*60)-13*60-1, int(meta['grid_s']*60))
raw=np.load(C/'raw.npy',mmap_mode='r'); yon=np.load(C/'y_on.npy',mmap_mode='r'); ypw=np.load(C/'y_power.npy',mmap_mode='r')
TOFF=meta['target_offset']; WC=meta['window_cycles']
RES=[apps.index(x) for x in ('electiric_kettle','hair_dryer','hotplate','oven')]
SM=[apps.index(x) for x in ('beam_projector','laptop_charger','minipc')]
rng=np.random.RandomState(0); WINS=[]; TP=[]
for i in rng.permutation(len(raw))[:600]:
    yo=np.asarray(yon[i]).astype(bool)
    ok=(yo[:,RES].sum(1)>=1)&(yo[:,SM].sum(1)==0)
    ts=np.nonzero(ok)[0][:3]
    if not len(ts): continue
    r=np.asarray(raw[i]); yp=np.asarray(ypw[i])
    for c,t in zip(grid[ts],ts):
        WINS.append(r[:,c-TOFF:c-TOFF+WC]); TP.append(yp[t][RES].sum())
    if len(WINS)>=450: break
WINS=np.stack(WINS); TP=np.array(TP)
print('합성 창 %d개 (저항 켜짐·SMPS 꺼짐) · 저항 참값 합 중앙 %.0fW'%(len(WINS),np.median(TP)))
print('\n전압을 α배 한 상황 (전류 α · 전압 α · 전력 α²) 에서 모델 출력이 α^k 로 따라가나')
print('  물리는 k=2 다. k<2 면 저전압에서 **과예측**한다.')
print('  %-10s %s'%('모델','   '.join('%7s'%('α=%.2f'%a) for a in (0.92,0.96,1.00,1.04,1.08))+'      k(적합)'))
V0=32
for nm in ('cnn_v31','cnn_v37','cnn_v32'):
    mo=load_model('results/%s.pt'%nm,dev)[0]
    out=[]
    for al in (0.92,0.96,1.00,1.04,1.08):
        W2=WINS.copy()
        W2[:,0:30]*=al          # 전류 고조파 Re/Im
        W2[:,30]*=al*al         # 전력
        W2[:,32:45]*=al         # 전압 실효 + 고조파
        f,w=build_inputs(W2)
        with torch.no_grad():
            P,G=[],[]
            for i in range(0,len(f),256):
                o=mo(torch.from_numpy(f[i:i+256]).to(dev),torch.from_numpy(w[i:i+256]).to(dev))
                P.append(o['power'].float()); G.append(o['on_logit'].float())
            p=torch.cat(P).cpu().numpy(); g=(torch.cat(G)>0).cpu().numpy()
        out.append(float(np.median((p*g)[:,RES].sum(1))))
    base=out[2]; rel=[o/base for o in out]
    al=np.array([0.92,0.96,1.00,1.04,1.08])
    k=float(np.polyfit(np.log(al),np.log(np.array(rel)),1)[0])
    print('  %-10s %s   k=%.2f'%(nm,'   '.join('%7.3f'%x for x in rel),k))
