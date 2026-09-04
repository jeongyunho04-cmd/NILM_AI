"""P1 대안 I: V15 에서 전체 파형을 복원하는 방법 비교 — 프로젝터 전류 재현으로 채점"""
import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize
from circuit3 import _core3
U='/mnt/user-data/uploads/'
F=60.0; NPC=3072; NCYC=24; NDFT=4; N=NCYC*NPC; DT=1/(F*NPC)
# ---- 실측 전압 (프로젝터 파일, 대칭화), 전체 스펙트럼
def load_V(f):
    d=pd.read_csv(U+f); v=d.v_v.values; npc=256
    k0=np.argmax(v[:npc]); st=(k0-npc//4)%npc
    V=np.mean([v[c:c+npc] for c in range(st,len(v)-npc,npc)],0); V=(V-np.roll(V,npc//2))/2
    X=np.fft.rfft(V)/npc*2; h=np.arange(len(X)); Xn=X*np.exp(-1j*h*np.angle(X[1])); Xn[0]=0
    return Xn                                                   # 진폭 (피크), V1 기준 위상, h0..h128
def wave(Xn,hmax):
    t=np.arange(N)*DT; v=np.zeros(N)
    for h in range(1,hmax+1,2): v+=abs(Xn[h])*np.cos(2*np.pi*F*h*t+np.angle(Xn[h]))
    return v
def sim_harm(par,P,vsrc):
    C,R,L,Cx,rd=par; I=_core3(vsrc,DT,P,C,max(R,1e-3),L,Cx,1.4,max(rd,1e-3),vsrc.max()-1.4)
    sl=slice(N-NDFT*NPC,N); Ih=np.fft.rfft(I[sl])[NDFT:NDFT*16:NDFT]; Vh=np.fft.rfft(vsrc[sl])[NDFT:NDFT*16:NDFT]
    h=np.arange(1,16); return np.abs(Ih)*np.exp(1j*(np.angle(Ih)-h*np.angle(Vh[0])))
def cond_angle(vsrc,par,P):
    C,R,L,Cx,rd=par; I=_core3(vsrc,DT,P,C,R,L,Cx,1.4,rd,vsrc.max()-1.4)[-NPC:]
    on=I>0.05*I.max(); return 360*on.sum()/NPC
def err(a,b): return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
par=pickle.load(open('circ_beam_projector.pkl','rb'))['params']; P=49.0
Xn=load_V('1788528564645_raw_beam_projector_1.csv')
V_full=wave(Xn,127); I_full=sim_harm(par,P,V_full)
print('전체 파형 기준: 피크 %.2fV  도통각 %.1f°'%(V_full.max(),cond_angle(V_full,par,P)))
print()
res=[]
def report(lab,v):
    I=sim_harm(par,P,v); e=err(I,I_full)
    hi=[abs(I[k-1])/abs(I_full[k-1]) for k in (9,11,13,15)]; ph=[np.degrees(np.angle(I[k-1]/I_full[k-1])) for k in (9,11,13,15)]
    res.append((lab,e)); print('  %-34s 피크 %+.2fV  도통각 %5.1f°  전류오차 %5.2f%%  |h9,11,13,15| %s  Δφ %s'%(
        lab,v.max()-V_full.max(),cond_angle(v,par,P),100*e,' '.join('%.2f'%x for x in hi),' '.join('%+.0f'%x for x in ph)))
# 1. 하드 절단
report('h15 하드 절단',wave(Xn,15)); report('h21 하드 절단',wave(Xn,21)); report('h31 하드 절단',wave(Xn,31))
# 2. Lanczos σ 인자 (절단 링잉 억제)
def lanczos(Xn,hmax):
    Y=Xn.copy(); 
    for h in range(1,hmax+1): 
        x=np.pi*h/(hmax+2); Y[h]*=np.sin(x)/x
    return wave(Y,hmax)
report('h15 + Lanczos σ',lanczos(Xn,15))
# 3. 꼭대기 평탄화 사전분포: v = A·sin − B·sin^n  (n 큰 홀수 → 피크 근처만 깎음). h3~h15 에 피팅
t=np.arange(NPC)/NPC*2*np.pi
def flattop(A,B,n,phi):
    v=A*np.cos(t)-B*np.cos(t)**n*np.sign(np.cos(t))**(n%2==0)   # n 홀수면 부호 자동
    return v
def ft_spec(A,B,n):
    v=A*np.cos(t)-B*np.cos(t)**int(n); X=np.fft.rfft(v)/NPC*2; return X
def ft_loss(x):
    A,B=abs(Xn[1]),x[0]; n=int(round(x[1]))|1
    Xf=ft_spec(A,B,n); Xf=Xf*np.exp(-1j*np.arange(len(Xf))*np.angle(Xf[1]))
    return sum(abs(Xf[h]-Xn[h])**2 for h in range(3,16,2))
best=min((minimize(ft_loss,[b0,n0],method='Nelder-Mead') for b0 in (5,10,20) for n0 in (5,9,15,21)),key=lambda r:r.fun)
B,n=best.x[0],int(round(best.x[1]))|1
Xf=ft_spec(abs(Xn[1]),B,n); Xf=Xf*np.exp(-1j*np.arange(len(Xf))*np.angle(Xf[1]))
print('  [평탄화 모델 피팅] B=%.1fV n=%d  →  h3..h15 재현: %s'%(B,n,' '.join('%.2f/%.2f'%(100*abs(Xf[h])/abs(Xn[1]),100*abs(Xn[h])/abs(Xn[1])) for h in (3,5,7,9,11,13,15))))
Y=Xn.copy()
for h in range(17,128,2): Y[h]=Xf[h] if h<len(Xf) else 0
report('h15 실측 + h17~ 평탄화모델',wave(Y,127))
# 4. 지수감쇠 외삽 (참고)
ks=np.arange(9,16,2); a,b=np.polyfit(ks,np.log([abs(Xn[k]) for k in ks]),1)
Y=Xn.copy()
for h in range(17,128,2): Y[h]=np.exp(a*h+b)*np.exp(1j*np.angle(Xn[15]))
report('h15 실측 + h17~ 지수외삽',wave(Y,127))
# 5. 피크 보정: h15 파형에서 피크 ±Δ 구간을 실측 통계로 깎기 — 장소 C 의 h17+ 가 만드는 파형 (v_full − v_15) 을 "형상 사전분포" 로
corr=V_full-wave(Xn,15)                                         # 이건 정답을 아는 것이지만, 형상만 빌려 다른 파일에 적용 가능한지 아래서 검증
print('  [h17+ 가 만드는 보정 파형] 피크 %.2fV, RMS %.2fV, 피크 ±30° 구간 평균 %.2fV'%(corr.min(),np.sqrt(np.mean(corr**2)),corr[-NPC:][np.abs(np.arange(NPC)-np.argmax(V_full[-NPC:]))<NPC*30/360].mean()))
pickle.dump(dict(corr=corr[-NPC:],Xn=Xn),open('vprior_C.pkl','wb'))
print()
print('전류 오차 순위:'); [print('  %5.2f%%  %s'%(100*e,l)) for l,e in sorted(res,key=lambda x:x[1])]
