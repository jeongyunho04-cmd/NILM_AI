#!/bin/bash
#SBATCH -J nilm_h32
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
OUT="$HOME/NILM_AI/processed_data/holdout60_v32"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"
# 캐시 v32 와 병렬. 자리는 캐시 작업이 v31 캐시와 홀드아웃 v30·v31 을 지워 낸다 — 여기서는 v30 만 먼저 지운다
# (5G, 캐시 작업도 같은 폴더를 지우려 하지만 없으면 건너뛴다).
D="$HOME/NILM_AI/processed_data/holdout60_v30"; if [ -d "$D" ]; then echo "제거: $D"; rm -rf "$D"; fi

$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--steady-crop" || { echo "run_build_holdout 에 --steady-crop 이 없다"; exit 1; }
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
  --standby-jitter-cap 95

$PY -X utf8 -c "
import json, sys
import numpy as np
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
if not (b.get('float_fill') and b.get('steady_crop') and b.get('standby_jitter_cap')): sys.exit('meta 에 v32 플래그가 없다')
apps = b['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy'); yp = np.load('$OUT/y_power.npy')
on = yo[:, kc] == 1; p = yp[on, kc]
print('충전기 ON 창 %d · <20W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), np.round(np.percentile(p, [10, 50, 90]), 1)))
print('positive_rate', json.dumps({k: round(v, 3) for k, v in b['positive_rate'].items()}, ensure_ascii=False))
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
