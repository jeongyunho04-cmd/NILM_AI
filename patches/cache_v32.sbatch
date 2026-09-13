#!/bin/bash
#SBATCH -J nilm_v32
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
OUT="$HOME/NILM_AI/cache/train60_v32"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문 0: 자리. v31 캐시(23G)·홀드아웃 v30·v31 을 지운다 (사용자 승인 2026-09-11).
#   cnn_v31.pt/json 은 results/ 와 로컬에 있다. 채점은 v32 홀드아웃에 네 판을 나란히 한다.
for D in "$HOME/NILM_AI/cache/train60_v31" "$HOME/NILM_AI/processed_data/holdout60_v30" "$HOME/NILM_AI/processed_data/holdout60_v31"; do
  if [ -d "$D" ]; then echo "제거: $D ($(du -sh "$D" 2>/dev/null | cut -f1))"; rm -rf "$D"; fi
done
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
if [ -z "${USED:-}" ]; then echo "홈 사용량을 못 재었다 — 관문 건너뜀"; else
  FREE=$((102400 - USED)); echo "여유 ${FREE} MB · 이 캐시는 약 23000 MB 필요"
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
t = load_curves(TEX_CURVES); assert len(t) >= 8
from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, load_vtail
tl = load_vtail(DEFAULT_VTAIL_NPZ); assert tl and 'E3' in tl
print('텍스처 곡선', len(t), '개 · 전압 꼬리 키', sorted(tl))
"
# 관문 2: 풀 (v31 과 같아야 한다 — charger_6 포함) + 새 코드 셋 (steady_crop · standby_jitter_cap · float_fill)
$PY -X utf8 -c "
import sys, inspect; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.augmentor import FLOAT_FILL_PRESETS, STEADY_CROP_PRESETS, DataAugmentor
from src.model.traincache import build_cache
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',), standby_jitter_cap_pct=95.0)
c = p.appliance_activations.get('laptop_charger', [])
assert any('laptop_charger_6' in str(a.source_file) for a in c), 'charger_6 가 풀에 없다'
assert 'smps_steady' in STEADY_CROP_PRESETS and 'charger_float' in FLOAT_FILL_PRESETS
sig = inspect.signature(build_cache).parameters
assert all(k in sig for k in ('float_fill', 'steady_crop', 'standby_jitter_cap')), '코드를 안 올렸다'
assert 'steady_crop' in inspect.signature(DataAugmentor.__init__).parameters
import numpy as np
pr = p.get_standby_profile('laptop_charger'); j = np.abs(np.asarray(pr.jitter_pool)[:, 2]) * 1e3
print('충전기 활성화 %d · 대기 잔차 상한 뒤 충전기 지터 |h3| p99 %.1f mA (상한 전 15.8)' % (len(c), np.percentile(j, 99)))
assert np.percentile(j, 99) < 8.0, '대기 잔차 상한이 안 걸렸다'
print('관문 통과')
"
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--steady-crop" || { echo "run_build_traincache 에 --steady-crop 이 없다"; exit 1; }

# ── v31 에서 바뀐 것 (13.83.26): --steady-crop smps_steady · --standby-jitter-cap 95 ────────────
#   ① SMPS 활성화를 자를 때 전력이 평탄한 60초 구간을 90% 로 고르고 켜짐/꺼짐 갈래는 30% 로.
#   ② 대기 잔차 풀의 차수별 p95 초과 꼬리를 자른다 (합성 배경층 σ|I3| 18.6 -> 3.4mA).
#   float_fill·풀·믹스·focus_off·상태믹스·전압꼬리는 v31 그대로. 항등: 플래그 off 에서 sha256 동일.
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
  --standby-jitter-cap 95

$PY -X utf8 -c "
import json, sys
import numpy as np
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
for k in ('n_windows','seed','sp_per_texture','vtail','couple_ext','carrier_apps','state_mix','smps_focus_off_p','float_fill','steady_crop','standby_jitter_cap','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12, 'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
if not d.get('float_fill') or not d.get('steady_crop') or not d.get('standby_jitter_cap'): sys.exit('meta 에 v32 플래그가 없다')
apps = d['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$OUT/y_on.npy', mmap_mode='r'); yp = np.load('$OUT/y_power.npy', mmap_mode='r')
on = np.asarray(yo[:, kc]) == 1; p = np.asarray(yp[:, kc])[on]
print('  충전기 ON 창 %d · <20W 몫 %.3f · p10/50/90 %s' % (on.sum(), (p < 20).mean(), np.round(np.percentile(p,[10,50,90]),1)))
if (p < 20).mean() < 0.08: sys.exit('축소 충전기가 너무 적다')
print('  관문 통과')
"
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
