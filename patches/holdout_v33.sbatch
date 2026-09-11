#!/bin/bash
#SBATCH -J nilm_h33
#SBATCH -p cpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT="$HOME/NILM_AI/processed_data/holdout60_v33"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"
# 캐시 v33 와 병렬. **아무것도 지우지 않는다** (v32 홀드아웃은 회전 없는 자로 남긴다).

$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--sibling-rotate" || { echo "run_build_holdout 에 --sibling-rotate 가 없다"; exit 1; }
time $PY -X utf8 -m src.run_build_holdout \
  --out "$OUT" \
  --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
  --recipe-mix steady2 \
  --smps-focus-off-p 0.4 \
  --state-mix minipc_balanced \
  --carrier-on oven \
  --couple-ext \
  --sp-curves --sp-per-texture --vtail \
  --float-fill charger_float \
  --steady-crop smps_steady \
  --standby-jitter-cap 95 \
  --sibling-rotate smps_rot10

$PY -X utf8 -c "
import json, sys
import numpy as np
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
if not (b.get('float_fill') and b.get('steady_crop') and b.get('standby_jitter_cap') and b.get('sibling_rotate')): sys.exit('meta 에 v33 플래그가 없다')
apps = b['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy'); yp = np.load('$OUT/y_power.npy')
on = yo[:, kc] == 1; p = yp[on, kc]
print('충전기 ON 창 %d · <20W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), np.round(np.percentile(p, [10, 50, 90]), 1)))
print('positive_rate', json.dumps({k: round(v, 3) for k, v in b['positive_rate'].items()}, ensure_ascii=False))
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
