#!/bin/bash
#SBATCH -J nilm_v33
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
OUT="$HOME/NILM_AI/cache/train60_v33"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문 0: 자리. **아무것도 지우지 않는다** — v32 캐시·홀드아웃은 짝지은 기준이라 남겨야 한다.
#   여유가 모자라면 중단하고 사람이 고른다 (공용 계정, 13.84.9).
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
if [ -z "${USED:-}" ]; then echo "홈 사용량을 못 재었다 — 관문 건너뜀"; else
  FREE=$((102400 - USED)); echo "여유 ${FREE} MB · 이 캐시는 약 23000 MB 필요"
  if [ "$FREE" -lt 26000 ]; then echo "여유 부족 — 중단 (지우지 않는다)"; exit 1; fi
fi

$PY -X utf8 -c "
import numpy as np
from src.synthesis.coupling import SmpsCircuit
c = SmpsCircuit(); rel = np.zeros(15, complex); rel[0] = 1
c.current('laptop_charger', 45.0, rel, 224.0); print('선컴파일 OK')
"
# 관문 1: 새 코드 (sibling_rotate 배관 5곳) + 풀 (v32 와 같아야 한다 — charger_6 포함)
$PY -X utf8 -c "
import sys, inspect; sys.path.insert(0,'.')
from src import env_guard
import numpy as np
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.augmentor import SIBLING_ROTATE_PRESETS, FLOAT_FILL_PRESETS, STEADY_CROP_PRESETS, DataAugmentor
from src.model.traincache import build_cache, _init
assert 'smps_rot10' in SIBLING_ROTATE_PRESETS, '코드를 안 올렸다 (augmentor)'
assert 'sibling_rotate' in inspect.signature(build_cache).parameters, '코드를 안 올렸다 (build_cache)'
assert 'sibling_rotate_json' in inspect.signature(_init).parameters, '코드를 안 올렸다 (_init)'
assert 'sibling_rotate' in inspect.signature(DataAugmentor.__init__).parameters, '코드를 안 올렸다 (DataAugmentor)'
a = DataAugmentor(sibling_rotate=SIBLING_ROTATE_PRESETS['smps_rot10'])
np.random.seed(1); c = (np.random.rand(8, 15) + 1j*np.random.rand(8, 15)).astype(np.complex64)
r = a._apply_sibling_rotate('beam_projector', c); ph = np.degrees(np.angle(r / c))
assert abs(ph[:, 0]).max() < 1e-4 and np.allclose(ph[0, 1:] / np.arange(2, 16), ph[0, 1] / 2, atol=0.05), '회전이 c·h 가 아니다'
assert np.array_equal(a._apply_sibling_rotate('minipc', c), c), '목록 밖 기기가 흔들린다'
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',), standby_jitter_cap_pct=95.0)
ch = p.appliance_activations.get('laptop_charger', [])
assert any('laptop_charger_6' in str(x.source_file) for x in ch), 'charger_6 가 풀에 없다'
assert 'smps_steady' in STEADY_CROP_PRESETS and 'charger_float' in FLOAT_FILL_PRESETS
print('관문 통과 · 충전기 활성화 %d · 회전 단위검사 OK (c=%.2f°/h)' % (len(ch), ph[0, 1] / 2))
"
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--sibling-rotate" || { echo "run_build_traincache 에 --sibling-rotate 가 없다"; exit 1; }

# ── v32 에서 바뀐 것 (13.84.8 ②): --sibling-rotate smps_rot10 하나. 나머지 인수는 cache_v32.sbatch 와 같다.
#   충전기·프로젝터 활성화의 고조파를 기본파에 대해 θ_h += c·h (c~U(±10°/h)) 로 돌린다. 항등: 플래그 off 에서 바이트 동일.
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
  --float-fill charger_float \
  --steady-crop smps_steady \
  --standby-jitter-cap 95 \
  --sibling-rotate smps_rot10

$PY -X utf8 -c "
import json, sys
import numpy as np
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
for k in ('n_windows','seed','sp_per_texture','vtail','couple_ext','carrier_apps','state_mix','smps_focus_off_p','float_fill','steady_crop','standby_jitter_cap','sibling_rotate','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12, 'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
if not d.get('sibling_rotate'): sys.exit('meta 에 sibling_rotate 가 없다')
# 회전이 실제로 들어갔는지: 충전기+프로젝터 ON 창의 φ3 산포가 v32 캐시보다 넓어야 한다
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
