#!/bin/bash
# 14.348 — **배관 연기시험**. 조합 머리를 본선에 던지기 전에 두 길이 다 도는지 본다.
#   팔 0: `--comb-tau 0`   배관을 붙였는데 **옛 경로가 안 깨졌나** (회귀)
#   팔 1: `--comb-tau 1.0` 새 경로가 도나 · 체크포인트에 `comb_tau` 가 적히나
# 에포크 둘 · 창 5,000 이라 몇 분이면 끝난다. **성적을 보는 판이 아니다.**
#SBATCH -J nilm_csmk
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -a 0-1
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v49}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v49}
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then TAU=0.0; NAME=csmk_off; else TAU=1.0; NAME=csmk_on; fi
echo "노드 $(hostname) · comb_tau=$TAU · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).

$PY -X utf8 -c "
import sys, numpy as np, torch
from src.model.inputs import VOLT_ORDERS, FINE_CHANNELS
from src.model.net import NILMNet, appliance_state_counts
from src.model.traincache import CachedWindows
c = CachedWindows('$CACHE')
print('  캐시 g_hat 있나 %s · 배치 길이 %d' % (c.has_ghat, len(c.batch(np.arange(4)))))
b = c.batch(np.arange(64))
g = b[-1]
print('  g_hat 중앙 %.2f mS · NaN %d/64' % (float(np.median(g)), int(np.isnan(g).sum())))
sys.exit(0 if (c.has_ghat and not np.isnan(g).any() and len(b) == 12) else '배관이 끊겼다')
"
test -d "$CACHE" || { echo "캐시가 없다"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다"; exit 1; }

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs 2 --epoch-windows 5000 --seed 0 \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 1 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40"
CAP="--p-state-cap 3"

echo "=== 학습 (연기시험) $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT --comb-tau $TAU \
    --tag cnn_${NAME} > "results/cnn_${NAME}.log" 2>&1
RC=$?
tail -20 "results/cnn_${NAME}.log"
$PY -X utf8 -c "
import torch, sys
ck = torch.load('results/cnn_${NAME}.pt', map_location='cpu', weights_only=False)
print('  체크포인트 comb_tau=%r · volt_orders=%r · fine=%r'
      % (ck.get('comb_tau'), ck.get('volt_orders'), ck.get('fine_channels')))
sd = ck['model'] if 'model' in ck else ck['state_dict']
print('  comb_beta 있나 %s · 파라미터 %d'
      % ('comb_beta' in sd, sum(v.numel() for v in sd.values())))
if abs(float(ck.get('comb_tau') or 0.0) - $TAU) > 1e-9: sys.exit('comb_tau 가 안 적혔다')
" || RC=1
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
