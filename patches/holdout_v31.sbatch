#!/bin/bash
#SBATCH -J nilm_h31
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
OUT="$HOME/NILM_AI/processed_data/holdout60_v31"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"
# 캐시 v31 과 병렬. 자리는 캐시 작업이 v30 캐시를 지우는 동안 좁으므로 여기서는 v29 의 옛
# 홀드아웃 v26 을 먼저 지워 5G 를 확보한다 (캐시 작업도 같은 폴더를 지우려 하지만 없으면 건너뛴다).
D="$HOME/NILM_AI/processed_data/holdout60_v26"; if [ -d "$D" ]; then echo "제거: $D"; rm -rf "$D"; fi

$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.segment_pool import SegmentPool
p = SegmentPool(npz_dir='processed_data/npz', time_split='holdout', holdout_frac=0.2, carrier_apps=('oven',))
c = p.appliance_activations.get('laptop_charger', [])
assert any('laptop_charger_6' in str(a.source_file) for a in c), 'charger_6 가 홀드아웃 분할에 없다'
print('홀드아웃 분할 충전기 활성화 %d · charger_6 있음' % len(c))
"
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
if not b.get('float_fill'): sys.exit('meta 에 float_fill 이 없다')
apps = b['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy'); yp = np.load('$OUT/y_power.npy')
on = yo[:, kc] == 1; p = yp[on, kc]
print('충전기 ON 창 %d · <20W 몫 %.3f · 16~25W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), ((p >= 16) & (p < 25)).mean(), np.round(np.percentile(p, [10, 50, 90]), 1)))
print('positive_rate', json.dumps({k: round(v, 3) for k, v in b['positive_rate'].items()}, ensure_ascii=False))
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
