"""fcm.py 검증: (1) 구조 (2) 선형화 정확도 (3) 결합 생성기 수렴 (4) 결합 효과 크기"""
import numpy as np, pandas as pd, pickle, time
from fcm import FCM, build_table, lookup, forward, wave_from_harmonics, harmonics_from_wave, H
U='/mnt/user-data/uploads/'
VH=['vh%d'%h for h in range(1,16)]; VD=['vhdeg%d'%h for h in range(1,16)]
def vbase_from_2hz(f):
    d=pd.read_csv(U+f); d=d[d.pll_locked==1]
    Vm=d[VH].values.astype(float).mean(0); Vp=np.array([np.angle(np.mean(np.exp(1j*np.deg2rad(d[c].values)))) for c in VD])
    return Vm*np.exp(1j*Vp)
V_C=vbase_from_2hz('1788516567454_beam_projector_4C.csv')          # 장소 C 전압 (홀수만 유효)
V_C[1::2]=0                                                          # 짝수 제거
print('기준 전압: V1=%.1fV  vh3=%.2f%%∠%+.0f°  vh5=%.2f%%'%(abs(V_C[0]),100*abs(V_C[2])/abs(V_C[0]),np.degrees(np.angle(V_C[2])),100*abs(V_C[4])/abs(V_C[0])))
DEV={'laptop_charger':45.0,'beam_projector':49.0,'minipc':18.0}
print()
print('=== (1) FCM 구조 ===')
FC={}; TAB={}
for dev,p in DEV.items():
    f=FCM.from_pickle('circ_%s.pkl'%dev); FC[dev]=f
    t0=time.time(); J,Y=f.extract(p,V_C); el=time.time()-t0
    Yn=np.abs(Y)*abs(V_C[0])/abs(J[0])          # 무차원: |Y_hk|·V1/|I1|  (k차 전압 1% 변화가 h차 전류를 I1 의 몇 % 바꾸나)
    diag=[Yn[h-1,h-1] for h in (1,3,5,7)]
    # 가장 큰 비대각 결합
    M=Yn.copy(); np.fill_diagonal(M,0); idx=np.unravel_index(np.argmax(M),M.shape)
    print('%-16s p=%4.0fW  추출 %.1f초  |J|=%.3fA'%(dev,p,el,abs(J[0])))
    print('   대각 |Y_hh|·V1/I1  h1..h7: %s'%' '.join('%.2f'%x for x in diag))
    print('   최대 비대각: Y[h=%d,k=%d] = %.2f   (k차 전압 1%% → h차 전류 %.2f%% of I1)'%(idx[0]+1,idx[1]+1,M[idx],M[idx]))
    print('   V3 열 (k=3) → h=1,3,5,7,9: %s'%' '.join('%.2f'%Yn[h-1,2] for h in (1,3,5,7,9)))
    TAB[dev]=build_table(f,[p*0.6,p,p*1.4] if dev!='beam_projector' else [p],V_C)

print()
print('=== (2) 선형화 정확도: J − Y·ΔV  vs  시뮬레이터(V+ΔV) ===')
print('    ΔV = h3 를 ±1%, ±3% 회전/증감')
for dev,p in DEV.items():
    f=FC[dev]; J,Y=lookup(TAB[dev],p)
    for lab,dv in [('h3 +1%',0.01),('h3 +3%',0.03),('h3 위상 +20°',None)]:
        Vp=V_C.copy()
        if dv is None: Vp[2]=V_C[2]*np.exp(1j*np.deg2rad(20))
        else: Vp[2]=V_C[2]+dv*abs(V_C[0])*np.exp(1j*np.angle(V_C[2]))
        Ilin=J-Y@(Vp-V_C); Isim=f.simulate(p,Vp)
        err=np.abs(Ilin-Isim)/np.abs(Isim); dJ=np.abs(Isim-J)/np.abs(J)
        print('   %-16s %-14s 변화량 h3=%4.1f%% h9=%4.1f%%  |  선형화 오차 h3=%.2f%% h9=%.2f%% h13=%.2f%%'%(
            dev,lab,100*dJ[2],100*dJ[8],100*err[2],100*err[8],100*err[12]))

print()
print('=== (3) 결합 생성기: 3기기 동시, 고정점 수렴 ===')
R_line,L_line=0.48,100e-6
pw={'laptop_charger':45.0,'beam_projector':49.0,'minipc':18.0}
for n in (1,2,3,5):
    I,Vt=forward(pw,R_line,L_line,V_C,tables=TAB,n_iter=n)
    print('   n_iter=%d  |I1|=%.4fA  I3=%.4f∠%+.0f°  I13=%.4f∠%+.0f°  V_term3=%.2fV'%(n,abs(I[0]),abs(I[2]),np.degrees(np.angle(I[2])),abs(I[12]),np.degrees(np.angle(I[12])),abs(Vt[2])))
I_lin,_=forward(pw,R_line,L_line,V_C,tables=TAB,n_iter=3)
t0=time.time(); I_ex,_=forward(pw,R_line,L_line,V_C,models=FC,n_iter=3); el=time.time()-t0
err=np.abs(I_lin-I_ex)/np.abs(I_ex)
print('   선형화 vs 정확(시뮬 고정점, %.1f초): 오차 h1=%.2f%% h3=%.2f%% h7=%.2f%% h11=%.2f%% h13=%.2f%%'%(el,*[100*err[h-1] for h in (1,3,7,11,13)]))

print()
print('=== (4) 결합 효과 크기: 독립 합 vs 결합 합 ===')
I_indep=sum(lookup(TAB[d],p)[0] for d,p in pw.items())
r=np.abs(I_ex)/np.abs(I_indep); dph=np.degrees(np.angle(I_ex/I_indep))
print('    %-6s'%''+''.join('%9s'%('h%d'%h) for h in (1,3,5,7,9,11,13,15)))
print('    %-6s'%'|비|'+''.join('%9.3f'%r[h-1] for h in (1,3,5,7,9,11,13,15)))
print('    %-6s'%'Δ위상'+''.join('%8.1f°'%dph[h-1] for h in (1,3,5,7,9,11,13,15)))
print('    → 이것이 12.181.4 에서 관측된 "합이 인수분해 안 됨" 의 모델 예측치')
pickle.dump(TAB,open('fcm_tables.pkl','wb'))
