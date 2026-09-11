#!/bin/bash
#SBATCH -J nilm_t34
#SBATCH -p gpu3,gpu4,gpu2,gpu6,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
$PY -X utf8 -c "
import sys, torch
if not torch.cuda.is_available(): sys.exit('GPU 를 못 씁니다')
print('torch', torch.__version__, 'device:', torch.cuda.get_device_name(0))
"
# 관문: 새 코드(--vswap-p) + 캐시·홀드아웃 v33 짝
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vswap-p" || { echo "run_train_cnn 에 --vswap-p 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v33/meta.json', encoding='utf-8'))
b = json.load(open('processed_data/holdout60_v33/meta.json', encoding='utf-8'))
for k in ('smps_focus_off_p', 'recipe_mix', 'state_mix', 'carrier_apps', 'window_cycles', 'float_fill', 'steady_crop', 'standby_jitter_cap', 'sibling_rotate'):
    if a.get(k) != b.get(k): sys.exit('캐시와 홀드아웃의 %s 가 다릅니다: %r vs %r' % (k, a.get(k), b.get(k)))
print('짝 관문 통과 · 캐시 %d창 · 홀드아웃 %d창' % (a['n_windows'], b['n_windows']))
"

# v34 = v33 과 **같은 캐시·홀드아웃·시드·명령** + --vswap-p 0.5 (13.84.11).
#   학습 배치에서 전압 고조파 채널을 같은 자리의 다른 창 것으로 바꿔 끼워, 생성기의 정확한 V->I 법칙을
#   판별자로 배우는 것을 막는다. 짝지은 기준은 cnn_v33 (v33 캐시). 캐시를 다시 굽지 않는다.
time $PY -X utf8 -m src.run_train_cnn \
    --cache cache/train60_v33 \
    --holdout processed_data/holdout60_v33 \
    --epochs 300 --epoch-windows 50000 --w-z 0.3 --w-over 0.0 \
    --eval-every 10 --cache-workers 4 \
    --harm-even-magnitude --state-signatures --standby-operating session \
    --vswap-p 0.5 --tag cnn_v34

echo "=== 홀드아웃 v32(회전 없음) 와 v33(회전) 에 v31 · v32 · v33 · v34 를 나란히 ==="
for H in processed_data/holdout60_v32 processed_data/holdout60_v33; do
  echo "--- $H"
  $PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_v31.pt results/cnn_v32.pt results/cnn_v33.pt results/cnn_v34.pt \
      --holdout "$H" 2>&1 | tail -40 || true
done

echo "끝 $(date '+%F %H:%M:%S')"
ls -la results/cnn_v34.pt 2>/dev/null || true
