"""v6: Shockley 다이오드.  V_bridge(I) = nVt·ln(1 + I/Is)   (nVt 는 브리지 전체 = 2 다이오드 합)
   rd, Vf 제거. 물리 범위: nVt 0.052~0.104 V (n=1~2, 2 다이오드), Is 1e-10~1e-6 A."""
import numpy as np
from numba import njit

@njit(cache=True)
def _core6(vsrc, dt, P, C_dc, R, L, Cx, nVt, Is, vc0):
    N=vsrc.size; iac=np.zeros(N)
    vc=vc0; iL=0.0; vn=vsrc[0]; ib=0.0
    inv=1.0/nVt
    for k in range(N):
        if L>1e-9: iL+=dt*(vsrc[k]-vn-iL*R)/L
        else:      iL=(vsrc[k]-vn)/R
        if Cx>1e-12:
            # 노드: Cx·(x−vn)/dt = iL − sgn·ib(|x|−vc),  Newton 3회
            x=vn
            # 지수 영역 깊숙이면 로그 초기 추정: ib ≈ iL 이 되는 vb 에서 시작
            sgn0=1.0 if x>=0.0 else -1.0
            if abs(x)-vc>30.0*nVt and sgn0*iL>Is:
                x=sgn0*(vc+nVt*np.log(sgn0*iL/Is+1.0))
            for _ in range(20):
                sgn=1.0 if x>=0.0 else -1.0
                vb=abs(x)-vc
                if vb>0.0:
                    e=np.exp(min(vb*inv,60.0)); ibx=Is*(e-1.0); dib=Is*e*inv
                else:
                    ibx=0.0; dib=0.0
                f=Cx*(x-vn)/dt-iL+sgn*ibx
                fp=Cx/dt+dib
                step=f/fp
                # 접합 전압 제한 (pnjlim): 지수 영역에서 한 번에 2·nVt 이상 못 움직임
                if vb>0.0 and abs(step)>2.0*nVt: step=2.0*nVt if step>0 else -2.0*nVt
                x-=step
                if abs(step)<1e-9: break
            vn=x
            vb=abs(vn)-vc
            ib=Is*(np.exp(min(vb*inv,60.0))-1.0) if vb>0.0 else 0.0
        else:
            # Cx 없음: 소스−R·iL 이 노드. 도통 전류를 Newton 으로 (iL = ±ib)
            vb=abs(vsrc[k]-iL*R)-vc
            ib=Is*(np.exp(min(vb*inv,60.0))-1.0) if vb>0.0 else 0.0
            iL=ib if vsrc[k]>=0.0 else -ib
        vc+=dt*(ib-P/max(vc,1.0))/C_dc
        iac[k]=iL
    return iac
