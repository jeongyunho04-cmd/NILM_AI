"""vtemplate.py 검증: 혼합 3개에서 V15+템플릿이 V15 단독보다 h21 펌웨어 수준으로 개선되는지.
   실행:  python vtemplate_check.py  (원시 파일이 /mnt/user-data/uploads 에 있어야 함)
   위상 관례 주의: VTemplate 은 arg(V_h) − h·arg(V_1) (h배). 절대 위상 스펙트럼에 붙일 때
   exp(+j·h·φ₁) 로 되돌려야 한다. exp(+j·φ₁) 만 곱하면 (h−1)·φ₁ 이 빠져 6.3% 가 8.6% 로 나빠진다."""
import numpy as np, pandas as pd, pickle
from vtemplate import VTemplate
from circuit3 import _core3
from circuit5 import _core5
from sp_model import SPModel
U='/mnt/user-data/uploads/'; F=60.0; NPC=256; UP=12; NPH=NPC*UP; NCYC=24; N=NCYC*NPH; DT=1/(F*NPH)
CK={d:pickle.load(open('circ_%s.pkl'%d,'rb')) for d in ['laptop_charger','minipc','beam_projector']}
BG=SPModel.load('sp_curves_v2.npz')['background']
def load(f):
    d=pd.read_csv(f); v=d.v_v.values; i=d.i_a.values; rg=d['range'].values
    k0=np.argmax(v[:NPC]); st=(k0-NPC//4)%NPC; cyc=[c for c in range(st,len(v)-NPC,NPC) if rg[c:c+NPC].std()==0]
    V=np.median([v[c:c+NPC] for c in cyc],0); I=np.median([i[c:c+NPC] for c in cyc],0); return (V-np.roll(V,NPC//2))/2,I
def spec(x): return np.fft.rfft(x)/len(x)*2
def up(V): X=spec(V); Y=np.zeros(NPH//2+1,complex); Y[:len(X)]=X; return np.fft.irfft(Y*NPH/2,NPH)
def trunc(V,hmax): X=spec(V); X[hmax+1:]=0; return np.fft.irfft(X*len(V)/2,len(V))
def sim_dev(k,p,Vh):
    vsrc=np.tile(Vh,NCYC); ck=CK[k]
    if ck['topo']=='v3':
        C,R,L,Cx,rd=ck['params']; I=_core3(vsrc,DT,p,C,max(R,1e-3),L,Cx,1.4,max(rd,1e-3),vsrc.max()-1.4)
    else:
        C,L2,R2,Cx,rd,Rc,I0=ck['params']; Rs=R2+Rc
        for _ in range(3):
            I,ir=_core5(vsrc,DT,p,C,L2,max(Rs,1e-3),Cx,1.4,max(rd,1e-3),vsrc.max()-1.4)
            irms=np.sqrt(np.mean(ir[-4*NPH:]**2)); Rs=0.5*Rs+0.5*(R2+Rc*np.exp(-irms/max(I0,1e-4)))
        I,_=_core5(vsrc,DT,p,C,L2,max(Rs,1e-3),Cx,1.4,max(rd,1e-3),vsrc.max()-1.4)
    return I[-NPH:]
def bg_wave(Vh):
    I15=BG.current(1.7,vrms=np.sqrt(np.mean(Vh**2))); ph0=np.angle(spec(Vh)[1]); t=np.arange(NPH)/NPH; w=np.zeros(NPH)
    for h in range(1,16,2): w+=np.sqrt(2)*abs(I15[h-1])*np.cos(2*np.pi*h*t+np.angle(I15[h-1])+h*ph0)
    return w
def total(pw,Vh): return bg_wave(Vh)+sum(sim_dev(k,p,Vh) for k,p in pw.items())
def down(x): return x.reshape(NPC,UP).mean(1)
def rms_err(a,b,hmax=15): A=spec(a)[:hmax+1]; B=spec(b)[:hmax+1]; return np.sqrt(np.sum(abs(A-B)**2)/np.sum(abs(B)**2))
MIX={'smps3_1':({'beam_projector':48.0,'laptop_charger':19.7,'minipc':16.5},'1788536164336_raw_smps3_1.csv'),
     'beam_charger_1':({'beam_projector':48.3,'laptop_charger':23.9},'1788536164335_raw_beam_charger_1.csv'),
     'beam_minipc_1':({'beam_projector':47.4,'minipc':5.4},'1788536164335_raw_beam_minipc_1.csv')}
if __name__=='__main__':
    T=VTemplate.load('vtemplate_C.npz'); print(T); print()
    print('%-16s %8s %8s %10s %8s'%('혼합','V전체','V15','V15+템플릿','h21절단'))
    for name,(pw,f) in MIX.items():
        V,I=load(U+f); X=spec(V); ph0=np.angle(X[1])
        V15h=X[1:16]*np.exp(-1j*np.arange(1,16)*ph0)
        Xe=T.extend(V15h); Xabs=np.zeros(NPC//2+1,complex); n=min(len(Xe),len(Xabs)); Xabs[:n]=Xe[:n]*np.exp(1j*np.arange(n)*ph0)
        Vt=up(np.fft.irfft(Xabs*NPC/2,NPC))
        e=[rms_err(down(total(pw,v)),I) for v in (up(V),up(trunc(V,15)),Vt,up(trunc(V,21)))]
        print('%-16s %7.1f%% %7.1f%% %9.1f%% %7.1f%%'%(name,*[100*x for x in e]))
