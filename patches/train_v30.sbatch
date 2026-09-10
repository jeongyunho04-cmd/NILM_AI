#!/bin/bash
#SBATCH -J nilm_t30
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
print('torch', torch.__version__, 'cuda', torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit('GPU 를 못 씁니다')
print('device:', torch.cuda.get_device_name(0))
"

# 관문: 캐시(v30)와 홀드아웃(v26)이 같은 생성기 설정인가 (13.83.12 의 자). 홀드아웃은 v29 와
# 같은 v26 을 그대로 쓴다 — 합성 F1 을 v29 와 같은 자로 견주기 위해서다. float_fill 은 캐시에만 있다
# (그 효과는 실측 test_1 과 run_diag_float_syn 으로 잰다).
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v30/meta.json', encoding='utf-8'))
b = json.load(open('processed_data/holdout60_v26/meta.json', encoding='utf-8'))
for k in ('sp_per_texture', 'vtail', 'couple_ext'):
    for nm, d in (('cache', a), ('holdout', b)):
        if not d.get(k):
            sys.exit('%s 에 %s 가 꺼져 있습니다' % (nm, k))
for k in ('smps_focus_off_p', 'recipe_mix', 'state_mix', 'carrier_apps', 'window_cycles'):
    if a.get(k) != b.get(k):
        sys.exit('캐시와 홀드아웃의 %s 가 다릅니다: %r vs %r' % (k, a.get(k), b.get(k)))
if not a.get('float_fill'):
    sys.exit('캐시 meta 에 float_fill 이 없다 — v30 이 아니다')
pr = a.get('positive_rate', {})
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12}
bad = [(k, round(pr.get(k, 0), 3)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad:
    sys.exit('캐시 양성률 하한 미달: %s' % bad)
print('짝 관문 통과 · 캐시 %d창 · 홀드아웃 %d창 · float_fill %s'
      % (a['n_windows'], b['n_windows'], json.dumps(a.get('float_fill'))))
print('양성률', json.dumps({k: round(v, 3) for k, v in sorted(pr.items())}, ensure_ascii=False))
"

# cnn_v29 를 훈련한 명령 그대로. 캐시·태그만 v30 이다 (홀드아웃 v26 그대로).
time $PY -X utf8 -m src.run_train_cnn \
    --cache cache/train60_v30 \
    --holdout processed_data/holdout60_v26 \
    --epochs 300 --epoch-windows 50000 --w-z 0.3 --w-over 0.0 \
    --eval-every 10 --cache-workers 4 \
    --harm-even-magnitude --state-signatures --standby-operating session \
    --tag cnn_v30

echo "끝 $(date '+%F %H:%M:%S')"
ls -la results/cnn_v30.pt 2>/dev/null || true
