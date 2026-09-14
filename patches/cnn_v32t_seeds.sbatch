#!/bin/bash
# 14.33 ⓑ — **v32t** (전압 텍스처 표집 간격 60 -> 20초) 대 v32. 시드 셋 짝비교.
#
# 왜 (14.12): 생성기는 표류를 **이미 만들고 있었다**. 모자란 것은 텍스처 표집 간격이었다 —
#   텍스처가 `step_s` 구간의 **중앙값**이라 그보다 짧은 전압 변동이 뭉개진다.
#       step_s   텍스처   h3~h15 산포/원시   표류비
#         60      280        0.73          0.75
#         20    **819**    **0.95**      **0.80**
#   0.80 이 **정답**이다 — 14.11 이 표류 분산의 65.4% 가 회로로 계산된다고 쟀으므로
#   진폭비 √0.654 = 0.809. ⇒ 전압 쪽에서 더 넣을 것이 없다는 뜻이고,
#   `sibling_rotate` 를 그 위에 더하면 안 된다 (0.75 위에 1.0 을 더해 1.25배가 된 것이
#   `smps_dev1` 의 12~42배였다).
#
#   A 대조   cache/train60_v32  + processed_data/holdout60_v32
#   B 고침   cache/train60_v32t + processed_data/holdout60_v32t
#   나머지 열 **전부 같다.**
#
# ⚠⚠ **홀드아웃 숫자를 팔끼리 직접 비교하지 마라.** 각 팔이 **자기 분포**에서 잰 값이다
#   (12.168.4: 캐시와 홀드아웃은 같아야 한다). 두 홀드아웃은 `vtex_step_s` 하나만 다르고
#   나머지 meta 가 전부 같은 것을 관문이 확인했다 (983212).
#   **판정은 실측 `run_diag_rollback` 한 자로 한다.**
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① 겨냥은 **SMPS** 다 — 텍스처는 전압 고조파 쪽이고 14.12 가 잰 것도 h3~h15 산포다.
#      SMPS 3종 신원 · 미니PC 유령이 좋아져야 한다.
#   ② ⚠ **저항 4종 신원은 안 움직여야 한다** (0.9815). 움직이면 텍스처가 저항까지 흔든 것이고
#      그건 14.12 의 해석이 틀렸다는 뜻이다.
#   ③ ⚠ 판정 줄이 안 움직일 수도 있다 — 14.12 가 잰 것은 **생성기의 성질**이지 모델 성능이
#      아니다. 그래도 "표류비가 정답에 맞는다" 는 물리 근거로 채택할 수 있다 (14.27 ⑦ 논리).
#   ④ ⚠ 시드 셋으로 읽어라. 이 조리법의 판정 줄 시드 폭이 sd 0.0040 이다.
#
# ⚠ 파티션은 A6000 우선이되 **밀리면 넓혀라** (HPC_RULES §3 · A10 은 1.5배 느릴 뿐).
#   scontrol update jobid=<id> Partition=gpu2,gpu6,gpu4,gpu5,gpu3,gpu1 TimeLimit=00:45:00
#
#SBATCH -J nilm_v32t
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
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문: 네 쌍이 다 있고 v32t 가 실제로 20초인가 ─────────────────────────
for d in cache/train60_v32 cache/train60_v32t processed_data/holdout60_v32 processed_data/holdout60_v32t; do
  test -d "$d" || { echo "없다: $d"; exit 1; }
done
$PY -X utf8 -c "
import json, sys
c0=json.load(open('cache/train60_v32/meta.json',encoding='utf-8'))
c1=json.load(open('cache/train60_v32t/meta.json',encoding='utf-8'))
h0=json.load(open('processed_data/holdout60_v32/meta.json',encoding='utf-8'))
h1=json.load(open('processed_data/holdout60_v32t/meta.json',encoding='utf-8'))
for nm,a,b in (('캐시',c0,c1),('홀드아웃',h0,h1)):
    if a.get('shard') or b.get('shard'): sys.exit('%s 가 토막이다' % nm)
    if float(b.get('vtex_step_s',0))!=20.0: sys.exit('%s v32t 의 vtex_step_s 가 %r' % (nm,b.get('vtex_step_s')))
    if float(a.get('vtex_step_s',0))!=0.0:  sys.exit('%s v32 의 vtex_step_s 가 %r (0 이어야 한다)' % (nm,a.get('vtex_step_s')))
    if a['window_cycles']!=b['window_cycles']: sys.exit('%s 창 길이가 다르다' % nm)
if h0['target_index']!=h1['target_index']: sys.exit('홀드아웃 타깃 시점이 다르다')
print('관문 ✅ v32 %d창 / v32t %d창 · 홀드아웃 %d/%d창 · 타깃 %d'
      % (c0['n_windows'],c1['n_windows'],h0['n_windows'],h1['n_windows'],h0['target_index']))
"
BASE="--epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache cache/train60_v32  --holdout processed_data/holdout60_v32 \
    --tag cnn_v32t_off_s${SEED} > "results/cnn_v32t_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache cache/train60_v32t --holdout processed_data/holdout60_v32t \
    --tag cnn_v32t_on_s${SEED} > "results/cnn_v32t_on_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: v32 ===";  tail -32 "results/cnn_v32t_off_s${SEED}.log"
echo "=== B: v32t ==="; tail -32 "results/cnn_v32t_on_s${SEED}.log"
ls -la results/cnn_v32t_*_s${SEED}.pt 2>/dev/null || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo "⚠ 홀드아웃 숫자는 팔끼리 비교하지 마라 (각자 자기 분포다). 판정은 실측:"
echo "  python -X utf8 src/run_diag_rollback.py --postproc off --ckpt results/cnn_v32t_off_s0.pt results/cnn_v32t_on_s0.pt"
exit $RC
