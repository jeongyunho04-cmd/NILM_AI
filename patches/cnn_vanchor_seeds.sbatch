#!/bin/bash
# 14.49 — **`harm_sig_vnorm` 의 기준전압을 기기별 sig 적합값으로.** 시드 셋 짝비교.
#
# 사용자: *"고전력 상황에서 과소예측되는 거부터 고쳐봐. 내가 계속 문제 있다고 5번은 말했는데."*
#
# 산수로 표적을 좁혔다: `test_1` 고부하 잔차 **+33.5W** = 오븐(−1.7%×1100=−19W) +
# 핫플(−2.0%×450=−9W) ≈ −28W. ⇒ **고전력 과소 = 오븐·핫플의 크기 편향 그 자체**다.
# 그리고 그 편향은 **참값이 정확한 합성 홀드아웃에도** 있다 (오븐 −21.3W · 핫플 −6.8W).
#
# 곱 분해 (합성 ON 창, 참 전력>50W, `run_diag_biasdecomp`, 시드 셋 소수 넷째까지 같다):
#     기기   power/참   σ(on)   p_st[참]/참   혼합 희석
#     오븐   0.969     1.0000   **0.970**    1.0000
#     핫플   0.981     1.0000   **0.985**    1.0000
#     포트   1.013     0.9999     1.014      1.0000
#     드라이  1.023     0.996      1.030      1.000
#   ⇒ 게이트도 혼합도 아니고 **상태의 학습된 전력값 하나**다.
#
# **원인 (14.49):** `sig = median(I_h/P)` 는 각 기기의 **격리 녹화 전압**에서 잰 값인데
#   `harm_sig_vnorm` 이 `sig × (V/V_CENTER=222)^e` 로 **전부 222V 에서 적합했다고 가정**한다.
#   실제 적합 전압이 기기마다 다르다:
#     오븐 210.4V · 핫플 214.4V · 미니PC 218.7V · 드라이 227.3V · 포트 227.7V ·
#     에어컨 227.7V · 프로젝 228.7V · 선풍 228.8V · 충전기 229.0V
#   어긋남은 `(222/V_적합)^e` 라는 **기기별 상수**다 (e=−1 이면 오븐 5.5%).
#     기기   적합 V   예상 편향   관측 편향
#     오븐   210.4V   −5.5%     **−3.0%**
#     핫플   214.4V   −3.5%     **−1.5%**
#     드라이  227.3V   +2.3%     **+3.0%**
#     포트   227.7V   +2.5%     **+1.4%**
#   **부호 4/4 · 순서 4/4.** (`L_power` 가 반대로 당기므로 크기는 절반쯤 평형이다.)
#   ⚠ SMPS 는 14.37 채택으로 지수가 0 이라 **이 축에 안 걸린다** — 충전기 −12.5% 가 이 표를
#     안 따르는 것이 오히려 **대조군**이다.
#
#   A 대조   그대로 (222V 고정)            = `cnn_vncls_res` 와 같은 조리법
#   B 고침   --harm-vnorm-anchor          기기별 적합 전압으로 기준을 옮긴다
#   나머지 열 **전부 같다.** 관문에서 잰 배수: 오븐 sig ×0.9479(전력 +5.5%) ·
#   핫플 ×0.9659(+3.5%) · 드라이 ×1.0239(−2.3%) · 포트 ×1.0259(−2.5%) · **나머지 다섯 ×1.0000**
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① 합성 `p_st[참]/참` 이 **1 쪽으로**: 오븐 0.970 -> 0.99 근처 · 핫플 0.985 -> 1.00 ·
#      드라이 1.030 -> 1.01 · 포트 1.014 -> 1.00
#   ② 합성 `편향(on)`: 오븐 **−21.3W -> −5W 안쪽** · 드라이 +8.3 -> 0 근처
#   ③ ★ **실측 고전력 과소가 준다** — 핫플통전·P>1500 잔차 중앙
#      test_1 +37.6 · test_3 +47.8 · test_5 +16.9 가 **눈에 띄게** 줄어야 한다.
#      **이것이 사용자가 다섯 번 말한 그것이고 이 판의 판정 숫자다.**
#   ④ ⚠ **SMPS 3종은 안 움직여야 한다** (지수 0). 움직이면 배선이 샌 것이다.
#   ⑤ ⚠ **과보정 위험** — `L_power` 가 이미 반대로 당겨 절반에서 평형이었으니 완전 보정이
#      과할 수 있다. 오븐이 +3% 로 **부호가 뒤집히면** 그것도 결론이고, 그때는 절반만 건다.
#   ⑥ ⚠ 판정 줄(on/off)은 **안 움직여도 된다** — 이 판은 크기를 고치지 검출을 안 고친다.
#
#SBATCH -J nilm_vanch
# ⚠ 파티션은 A6000 Ada/A6000 우선이지만 **밀리면 넓혀라** (HPC_RULES §3, A10 은 1.5배 느릴 뿐).
#   983243 에서 gpu3/4/5 만으로 128개 뒤에 섰다가 넓히자 6초에 잡혔다.
#SBATCH -p gpu3,gpu4,gpu5,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 00:45:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-vnorm-anchor" || {
  echo "--harm-vnorm-anchor 가 없다 — 코드를 안 올렸다"; exit 1; }
# ⚠⚠ **함수를 재지 말고 배선을 재라** (983506 이 45분을 버렸다). 아래 관문은
#   `harmonic_signature_vref` 를 따로 불러 배수만 찍었고 통과했는데, 정작 그 값을
#   `NILMLoss` 로 넘기는 줄이 **조용히 안 들어가** 두 팔이 비트 동일이었다
#   (`B−A = +0.00 ± 0.00`). `run_gate_vanchor` 가 `NILMLoss` 를 실제로 지어
#   `L_harm` 의 예측 고조파가 기기별로 예상 배수만큼 달라지는지 끝단에서 본다.
$PY -X utf8 -m src.run_gate_vanchor || { echo "vanchor 배선 관문 실패"; exit 1; }
# 관문: 저항 넷만 배수가 1 이 아니고 나머지 다섯은 정확히 1.0 이어야 한다
$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
import numpy as np, torch
from src.synthesis.segment_pool import SegmentPool
from src.model.net import harmonic_signature_vref
from src.run_baseline import S_I
from src.run_train_cnn import _vnorm_exp
from src.model.inputs import V_CENTER
apps=sorted(S_I); pool=SegmentPool(npz_dir='processed_data/npz', time_split='train')
v,_=harmonic_signature_vref(pool,apps); e=np.asarray(_vnorm_exp(apps,'RESISTIVE'),float)
k=np.where(v>0,(V_CENTER/np.maximum(v,1.0))**e,1.0)
bad=[a for a,x,y in zip(apps,k,e) if (y==0 and abs(x-1)>1e-6)]
if bad: sys.exit('지수 0 인데 배수가 1 이 아니다: %s' % bad)
res=[a for a,y in zip(apps,e) if y!=0]
if sorted(res)!=sorted(['electiric_kettle','oven','hotplate','hair_dryer']):
    sys.exit('지수가 걸린 기기가 %s — 저항 넷이어야 한다' % res)
print('관문 ✅ ' + ' '.join('%s %.1fV x%.4f'%(a[:4],vv,kk) for a,vv,kk in zip(apps,v,k) if kk!=1.0))
"
test -d "$CACHE" || { echo "캐시가 없다"; exit 1; }
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE \
    --tag cnn_vanch_off_s${SEED} > "results/cnn_vanch_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor \
    --tag cnn_vanch_on_s${SEED} > "results/cnn_vanch_on_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: 222V 고정 ==="; tail -32 "results/cnn_vanch_off_s${SEED}.log"
echo "=== B: 기기별 기준 ==="; tail -32 "results/cnn_vanch_on_s${SEED}.log"
ls -la results/cnn_vanch_*_s${SEED}.pt 2>/dev/null || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo "판정: **실측 고전력 과소**다 —"
echo "  python -X utf8 -m src.run_diag_biasdecomp --ckpt <둘>   (합성, 참값 정확)"
echo "  python -X utf8 src/run_diag_rollback.py --postproc off --ckpt <둘>"
exit $RC
