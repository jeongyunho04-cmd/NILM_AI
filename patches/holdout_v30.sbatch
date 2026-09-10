#!/bin/bash
#SBATCH -J nilm_h30
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
OUT="$HOME/NILM_AI/processed_data/holdout60_v30"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"
# 캐시(v30)와 **병렬로** 돈다 — 홀드아웃은 캐시를 읽지 않는다. 짝 검사는 학습 작업이 한다.
# v26 홀드아웃과 다른 것은 --float-fill charger_float 하나다 (13.83.23). 축소 충전기 부류의
# 합성 F1 을 재려면 홀드아웃에도 그 창이 있어야 한다.

$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--float-fill" || { echo "run_build_holdout 에 --float-fill 이 없다 — 코드를 안 올렸다"; exit 1; }

time $PY -X utf8 -m src.run_build_holdout \
  --out "$OUT" \
  --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
  --recipe-mix steady2 \
  --smps-focus-off-p 0.4 \
  --state-mix minipc_balanced \
  --carrier-on oven \
  --couple-ext \
  --sp-curves --sp-per-texture --vtail \
  --float-fill charger_float

$PY -X utf8 -c "
import json, sys
import numpy as np
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
if not b.get('float_fill'): sys.exit('meta 에 float_fill 이 없다 — 옛 경로로 만들어졌다')
apps = b['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy'); yp = np.load('$OUT/y_power.npy')
on = yo[:, kc] == 1; p = yp[on, kc]
print('충전기 ON 창 %d · 목표전력 <20W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), np.round(np.percentile(p, [10, 50, 90]), 1)))
print('positive_rate', json.dumps({k: round(v, 3) for k, v in b['positive_rate'].items()}, ensure_ascii=False))
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
