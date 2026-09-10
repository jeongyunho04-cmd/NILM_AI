#!/bin/bash
#SBATCH -J nilm_t31
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
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v31/meta.json', encoding='utf-8'))
b = json.load(open('processed_data/holdout60_v31/meta.json', encoding='utf-8'))
for k in ('sp_per_texture', 'vtail', 'couple_ext'):
    for nm, d in (('cache', a), ('holdout', b)):
        if not d.get(k): sys.exit('%s 에 %s 가 꺼져 있습니다' % (nm, k))
for k in ('smps_focus_off_p', 'recipe_mix', 'state_mix', 'carrier_apps', 'window_cycles', 'float_fill'):
    if a.get(k) != b.get(k): sys.exit('캐시와 홀드아웃의 %s 가 다릅니다: %r vs %r' % (k, a.get(k), b.get(k)))
pr = a.get('positive_rate', {})
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12}
bad = [(k, round(pr.get(k, 0), 3)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('캐시 양성률 하한 미달: %s' % bad)
print('짝 관문 통과 · 캐시 %d창 · 홀드아웃 %d창' % (a['n_windows'], b['n_windows']))
"

# cnn_v30 과 같은 명령. 캐시·홀드아웃·태그만 v31 (풀에 charger_6, 13.83.24).
time $PY -X utf8 -m src.run_train_cnn \
    --cache cache/train60_v31 \
    --holdout processed_data/holdout60_v31 \
    --epochs 300 --epoch-windows 50000 --w-z 0.3 --w-over 0.0 \
    --eval-every 10 --cache-workers 4 \
    --harm-even-magnitude --state-signatures --standby-operating session \
    --tag cnn_v31

echo "=== 같은 홀드아웃 v31 에 v29 · v30 · v31 을 나란히 ==="
$PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_v29.pt results/cnn_v30.pt results/cnn_v31.pt \
    --holdout processed_data/holdout60_v31 2>&1 | tail -50 || true

echo "끝 $(date '+%F %H:%M:%S')"
ls -la results/cnn_v31.pt 2>/dev/null || true
