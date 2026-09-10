#!/bin/bash
#SBATCH -J nilm_v30
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
OUT="$HOME/NILM_AI/cache/train60_v30"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문 0: 여유 공간. 모자라면 v29 캐시를 지운다 (사용자 승인, 2026-09-10 저녁).
#   cnn_v29.pt/json 은 로컬에 있고 meta 로 25분이면 다시 굽는다. 여유가 있으면 둔다.
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
if [ -z "${USED:-}" ]; then echo "홈 사용량을 못 재었다 — 관문 건너뜀"; else
  FREE=$((102400 - USED)); echo "여유 ${FREE} MB · 이 캐시는 약 23000 MB 필요"
  if [ "$FREE" -lt 26000 ] && [ -d "$HOME/NILM_AI/cache/train60_v29" ]; then
    echo "여유 부족 — v29 캐시 제거 ($(du -sh "$HOME/NILM_AI/cache/train60_v29" 2>/dev/null | cut -f1))"
    rm -rf "$HOME/NILM_AI/cache/train60_v29"
    USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 ); FREE=$((102400 - USED))
    echo "제거 뒤 여유 ${FREE} MB"
  fi
  if [ "$FREE" -lt 26000 ]; then echo "여유 부족 — 중단"; exit 1; fi
fi

# numba JIT 선컴파일 — 워커 폭주 전에 한 번
$PY -X utf8 -c "
import numpy as np
from src.synthesis.coupling import SmpsCircuit
c = SmpsCircuit(); rel = np.zeros(15, complex); rel[0] = 1
c.current('laptop_charger', 45.0, rel, 224.0); print('선컴파일 OK')
"

# 관문 1: 텍스처 곡선·전압 꼬리 (v29 와 같아야 한다)
$PY -X utf8 -c "
from src.run_build_sp_curves import TEX_CURVES
from src.synthesis.sp_curves import load_curves
t = load_curves(TEX_CURVES); assert len(t) >= 8, ('곡선 부족', len(t))
from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, load_vtail
tl = load_vtail(DEFAULT_VTAIL_NPZ); assert tl and 'E3' in tl, '전압 꼬리에 E3 가 없다'
print('텍스처 곡선', len(t), '개 · 전압 꼬리 키', sorted(tl))
"

# 관문 2: 풀 (v29 와 같은 풀이어야 한다)
$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.segment_pool import SegmentPool
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',))
c = p.appliance_activations.get('laptop_charger', []); m = p.appliance_activations.get('minipc', [])
assert len(c) >= 16 and len(m) >= 9, '풀이 v29 와 다르다'
print('충전기 활성화 %d · 미니PC %d · 관문 통과' % (len(c), len(m)))
"

# 관문 3: **새 코드가 실려 있는가.** 안 올렸으면 조용히 v29 와 같은 캐시가 나온다 — 여기서 선다.
$PY -X utf8 -c "
import sys, inspect; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.augmentor import FLOAT_FILL_PRESETS, DataAugmentor
from src.model.traincache import build_cache
assert 'charger_float' in FLOAT_FILL_PRESETS, 'FLOAT_FILL_PRESETS 가 없다 — augmentor.py 를 안 올렸다'
assert 'float_fill' in inspect.signature(build_cache).parameters, 'build_cache 에 float_fill 이 없다 — traincache.py 를 안 올렸다'
assert 'float_fill' in inspect.signature(DataAugmentor.__init__).parameters
d = FLOAT_FILL_PRESETS['charger_float']['laptop_charger']
print('charger_float: 상태 %d · p %.2f · 배율 %s' % (d['state'], d['p'], d['scale']))
"
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--float-fill" || { echo "run_build_traincache 에 --float-fill 이 없다"; exit 1; }

# ── v29 에서 바뀐 것은 **--float-fill charger_float 하나**다 (13.83.22/23) ─────────
#   실측 test_1 의 충전기는 만충 뒤 ~14W 부동인데 풀에는 8~20W 연속 구간이 최장 1초라
#   합성기가 그 창을 못 만들었다. 충전기 활성화의 25% 를 상태 1 로 채워 0.40~0.70 배로 내린다.
#   로컬 항등 검정: 플래그 off 에서 120창 sha256 이 옛 코드와 동일 (09af272d… / 3c03bb03…).
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

echo "=== meta (v29 와 float_fill 만 달라야 한다) ==="
$PY -X utf8 -c "
import json, sys
import numpy as np
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
for k in ('n_windows','window_cycles','seed','sp_curves','sp_per_texture','vtail','background',
          'couple_ext','carrier_apps','state_mix','level_scramble','time_split',
          'smps_focus_off_p','float_fill','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
print('  recipe_mix', json.dumps(d.get('recipe_mix'), ensure_ascii=False))
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12, 'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
if not d.get('float_fill'): sys.exit('meta 에 float_fill 이 없다 — 옛 경로로 구워졌다')
# 축소 충전기 관문 — 충전기 ON 창 중 목표 전력 < 20W 몫 (로컬 120창에서 0.17)
apps = d['appliances']; kc = apps.index('laptop_charger'); j = apps.index('minipc')
yo = np.load('$OUT/y_on.npy', mmap_mode='r'); yp = np.load('$OUT/y_power.npy', mmap_mode='r')
on = np.asarray(yo[:, kc]) == 1; p = np.asarray(yp[:, kc])[on]
share = float((p < 20.0).mean())
print('  충전기 ON 창 %d · 목표전력 <20W 몫 %.3f · p10/50/90 %s' % (on.sum(), share, np.round(np.percentile(p,[10,50,90]),1)))
if share < 0.08: sys.exit('축소 충전기가 너무 적다 (%.3f) — float_fill 이 안 걸렸다' % share)
print('  관문 통과')
"
du -sh "$OUT"
echo "홈 사용량: $(du -sh $HOME 2>/dev/null | cut -f1) / 쿼터 100G"
echo "끝 $(date '+%F %H:%M:%S')"
