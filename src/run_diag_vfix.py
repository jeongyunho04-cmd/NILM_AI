import sys, glob
sys.path.insert(0,'.')
from src import env_guard
import numpy as np, torch
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FS, FILES, real_windows
dev='cuda' if torch.cuda.is_available() else 'cpu'
b=torch.load('results/cnn_v37.pt',map_location='cpu',weights_only=False); apps=b['appliances']
real=real_windows(apps,2.0,dev)
#: 생성기에서 잰 참 지수 (합성 캐시 회귀). SMPS 는 0 으로 고정(음수는 적합 잡음)
E={'electiric_kettle':2.00,'hair_dryer':2.00,'hotplate':1.99,'oven':2.00,
   'fan':0.62,'air_conditioner':0.60,'beam_projector':0.0,'laptop_charger':0.0,'minipc':0.0}
VREC={'electiric_kettle':227.7,'hair_dryer':227.2,'hotplate':214.3,'oven':210.3,
      'fan':228.6,'air_conditioner':227.6,'beam_projector':226.5,'laptop_charger':225.2,'minipc':224.9}
print('사후 보정: p_k <- p_k · (V_파일/V_녹화,k)^(e_k − k_model)  — 기준을 **기기별 녹화 전압**으로')
print('  %-10s %6s | %s'%('모델','k_model',''.join('%9s'%f.replace('test_','t') for f in FILES)+'    |평균|'))
for nm,km in (('cnn_v31',0.81),('cnn_v37',0.85),('cnn_v32',0.91)):
    mo=load_model('results/%s.pt'%nm,dev)[0]
    for tag,use in (('보정 전',False),('보정 후',True)):
        row=[]
        for stem in FILES:
            d=real[stem]
            r_=load_nilm_npz('processed_data/composite_eval/%s.npz'%stem)
            pp=np.asarray(r_['power_features'])[:,0]
            Vc=np.abs(np.asarray(r_['voltage_harmonics_complex'])[:,0])
            idx=np.clip((d['t']*FS).astype(int),0,len(pp)-1)
            po=pp[idx]; V=Vc[idx]
            with torch.no_grad():
                P,G=[],[]
                for i in range(0,len(d['t']),512):
                    o=mo(torch.from_numpy(d['fine'][i:i+512]).to(dev),torch.from_numpy(d['wide'][i:i+512]).to(dev))
                    P.append(o['power'].float()); G.append(o['on_logit'].float())
                p=torch.cat(P).cpu().numpy(); g=(torch.cat(G)>0).cpu().numpy()
            pg=(p*g).copy()
            if use:
                for k,a in enumerate(apps):
                    e=E.get(a,0.0)
                    if abs(e-km)<1e-6: continue
                    pg[:,k]*= (V/VREC[a])**(e-km)
            row.append(float((pg.sum(1)-po).mean()))
        print('  %-10s %6.2f | %s %8.0f'%(nm if not use else '', km if not use else 0,
              ''.join('%+9.0f'%x for x in row),np.mean(np.abs(row))))
