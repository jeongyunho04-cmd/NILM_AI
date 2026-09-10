#!/bin/bash
#SBATCH -J nilm_v31
#SBATCH -p cpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT="$HOME/NILM_AI/cache/train60_v31"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문 0: 자리. v30 캐시(23G)와 v29 의 홀드아웃 v26(5G)을 지운다 (사용자 승인, 2026-09-10 저녁).
#   cnn_v30.pt/json 은 results/ 에 남고 로컬에도 받는다. 캐시는 meta 로 다시 굽는다.
for D in "$HOME/NILM_AI/cache/train60_v30" "$HOME/NILM_AI/processed_data/holdout60_v26"; do
  if [ -d "$D" ]; then echo "제거: $D ($(du -sh "$D" 2>/dev/null | cut -f1))"; rm -rf "$D"; fi
done
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
if [ -z "${USED:-}" ]; then echo "홈 사용량을 못 재었다 — 관문 건너뜀"; else
  FREE=$((102400 - USED)); echo "여유 ${FREE} MB · 이 캐시는 약 23000 MB 필요"
  if [ "$FREE" -lt 26000 ] && [ -d "$HOME/NILM_AI/processed_data/holdout60_v30" ]; then
    echo "여유 부족 — 홀드아웃 v30 도 제거 (채점은 v31 홀드아웃에서 세 판을 나란히 한다)"
    rm -rf "$HOME/NILM_AI/processed_data/holdout60_v30"
    USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 ); FREE=$((102400 - USED)); echo "제거 뒤 여유 ${FREE} MB"
  fi
  if [ "$FREE" -lt 26000 ]; then echo "여유 부족 — 중단"; exit 1; fi
fi

$PY -X utf8 -c "
import numpy as np
from src.synthesis.coupling import SmpsCircuit
c = SmpsCircuit(); rel = np.zeros(15, complex); rel[0] = 1
c.current('laptop_charger', 45.0, rel, 224.0); print('선컴파일 OK')
"
$PY -X utf8 -c "
from src.run_build_sp_curves import TEX_CURVES
from src.synthesis.sp_curves import load_curves
t = load_curves(TEX_CURVES); assert len(t) >= 8, ('곡선 부족', len(t))
from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, load_vtail
tl = load_vtail(DEFAULT_VTAIL_NPZ); assert tl and 'E3' in tl
print('텍스처 곡선', len(t), '개 · 전압 꼬리 키', sorted(tl))
"
# 관문 2: **charger_6 (만충 녹화, 13.83.24) 가 풀에 있는가.** v30 과의 유일한 차이다.
$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.segment_pool import SegmentPool
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',))
c = p.appliance_activations.get('laptop_charger', [])
six = [a for a in c if 'laptop_charger_6' in str(a.source_file)]
assert six, 'laptop_charger_6 가 풀에 없다 — npz 나 등록부를 안 올렸다'
import numpy as np
p6 = np.concatenate([np.asarray(a.target_power_w) for a in six]); p6 = p6[p6 > 1]
print('충전기 활성화 %d · charger_6 %d개 · P p5/50/95 %s · 16~25W 몫 %.2f' % (len(c), len(six), np.round(np.percentile(p6, [5, 50, 95]), 1), ((p6 >= 16) & (p6 < 25)).mean()))
assert ((p6 >= 16) & (p6 < 25)).mean() > 0.5
"
$PY -X utf8 -c "
import sys, inspect; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.augmentor import FLOAT_FILL_PRESETS
from src.model.traincache import build_cache
assert 'charger_float' in FLOAT_FILL_PRESETS and 'float_fill' in inspect.signature(build_cache).parameters
print('float_fill 코드 OK')
"

# ── v30 에서 바뀐 것은 **풀에 charger_6 이 든 것 하나**다 (13.83.24). 축소(--float-fill)는 그대로.
time $PY -X utf8 -m src.run_build_traincache \
  --out "$OUT" \
  --windows 300000 --window-cycles 3600 --seed 0 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --power-scale-std measured \
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
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
for k in ('n_windows','seed','sp_per_texture','vtail','couple_ext','carrier_apps','state_mix','smps_focus_off_p','float_fill','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12, 'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
if not d.get('float_fill'): sys.exit('meta 에 float_fill 이 없다')
apps = d['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy', mmap_mode='r'); yp = np.load('$OUT/y_power.npy', mmap_mode='r')
on = np.asarray(yo[:, kc]) == 1; p = np.asarray(yp[:, kc])[on]
print('  충전기 ON 창 %d · <20W 몫 %.3f · 16~25W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), ((p >= 16) & (p < 25)).mean(), np.round(np.percentile(p,[10,50,90]),1)))
if (p < 20).mean() < 0.08: sys.exit('축소 충전기가 너무 적다')
print('  관문 통과')
"
du -sh "$OUT"
echo "홈 사용량: $(du -sh $HOME 2>/dev/null | cut -f1) / 쿼터 100G"
echo "끝 $(date '+%F %H:%M:%S')"
