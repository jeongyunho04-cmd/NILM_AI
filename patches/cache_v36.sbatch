#!/bin/bash
#SBATCH -J nilm_v36
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
OUT="$HOME/NILM_AI/cache/train60_v36"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문 0: 자리. **아무것도 지우지 않는다.** 여유가 모자라면 중단하고 사람이 고른다 (공용 계정).
#   ⚠ 2026-09-11: 예전 판은 `102400 - du $HOME` 로 "홈 쿼터 100GB" 를 **가정**했다. 그것은 근거가 없다 —
#   `quota -s` 가 아무것도 안 내놓고(사용자 쿼터 미설정), 홈이 104GB 를 넘겨도 쓰기가 된다. 그 가정 때문에
#   42MB 차이로 두 번 막혔고 v33 캐시를 **불필요하게** 지웠다. 이제 파일시스템 실제 여유를 본다.
AVAIL=$( { df -m "$HOME" 2>/dev/null || true; } | tail -1 | awk '{print $4}' )
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
echo "파일시스템 여유 ${AVAIL:-?} MB · 홈 사용 ${USED:-?} MB · 이 캐시는 실측 22842 MB 필요"
if [ -n "${AVAIL:-}" ] && [ "$AVAIL" -lt 40000 ]; then
  echo "파일시스템 여유 부족 — 중단 (지우지 않는다)"; exit 1
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
assert 'smps_dev1' in SIBLING_ROTATE_PRESETS, '코드를 안 올렸다 (augmentor: smps_dev1)'
assert 'sibling_rotate' in inspect.signature(build_cache).parameters, '코드를 안 올렸다 (build_cache)'
assert 'sibling_rotate_json' in inspect.signature(_init).parameters, '코드를 안 올렸다 (_init)'
assert 'sibling_rotate' in inspect.signature(DataAugmentor.__init__).parameters, '코드를 안 올렸다 (DataAugmentor)'
a = DataAugmentor(sibling_rotate=SIBLING_ROTATE_PRESETS['smps_rot10'])
np.random.seed(1); c = (np.random.rand(8, 15) + 1j*np.random.rand(8, 15)).astype(np.complex64)
r = a._apply_sibling_rotate('beam_projector', c); ph = np.degrees(np.angle(r / c))
assert abs(ph[:, 0]).max() < 1e-4 and np.allclose(ph[0, 1:] / np.arange(2, 16), ph[0, 1] / 2, atol=0.05), '옛 회전 경로가 깨졌다'
assert np.array_equal(a._apply_sibling_rotate('minipc', c), c), '목록 밖 기기가 흔들린다'
b = DataAugmentor(sibling_rotate=SIBLING_ROTATE_PRESETS['smps_dev1'])
assert np.array_equal(b._apply_sibling_rotate('minipc', c), c), '목록 밖 기기가 흔들린다 (dev1)'
one = np.ones((1, 15), np.complex64)
np.random.seed(0)
G = np.array([np.abs(b._apply_sibling_rotate('laptop_charger', one.copy())[0]) for _ in range(3000)])
np.random.seed(0)
Pz = np.array([np.degrees(np.angle(b._apply_sibling_rotate('laptop_charger', one.copy())[0])) for _ in range(3000)])
wrap = lambda x: (x + 180.0) % 360.0 - 180.0
assert G[:, 0].std() < 1e-6 and abs(wrap(Pz[:, 0])).max() < 1e-4, 'h1 이 흔들린다 — P 라벨·phi 기준이 어긋난다'
s3, s9, s15 = wrap(Pz[:, 2]).std(), wrap(Pz[:, 8]).std(), wrap(Pz[:, 14]).std()
assert 12.0 < s3 < 24.0, 'h3 위상 산포가 회전 c*h 와 안 맞는다: %.1f' % s3
assert s9 > 85.0 and s15 > 85.0, 'h>=9 위상이 안 뒤섞였다: %.1f / %.1f' % (s9, s15)
r_mpc = {3: 0.202, 9: 0.311, 15: 0.277}     # 미니PC/형제 크기비 (run_diag_sibgap)
for h, rv in r_mpc.items():
    k = 3.0 * G[:, h - 1].std() / rv
    assert k < 1.5, 'h%d 크기 흔들림이 미니PC 기여를 묻는다 (3sigma/r=%.2f)' % (h, k)
print('  dev1 단위검사 OK · 위상 sigma h3 %.1f h9 %.1f h15 %.1f · 3sigma/r h3 %.2f h9 %.2f h15 %.2f'
      % (s3, s9, s15, 3*G[:, 2].std()/0.202, 3*G[:, 8].std()/0.311, 3*G[:, 14].std()/0.277))
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',), standby_jitter_cap_pct=95.0)
ch = p.appliance_activations.get('laptop_charger', [])
assert any('laptop_charger_6' in str(x.source_file) for x in ch), 'charger_6 가 풀에 없다'
assert 'smps_steady' in STEADY_CROP_PRESETS and 'charger_float' in FLOAT_FILL_PRESETS
print('관문 통과 · 충전기 활성화 %d' % len(ch))
"
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--sibling-rotate" || { echo "run_build_traincache 에 --sibling-rotate 가 없다"; exit 1; }

# ── v32 에서 바뀐 것 (13.84.8 ②): --sibling-rotate smps_dev1 하나. 나머지 인수는 cache_v32.sbatch 와 같다.
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
  --sibling-rotate smps_dev1

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
