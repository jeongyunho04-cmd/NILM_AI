#!/bin/bash
#SBATCH -J nilm_v29
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
OUT="$HOME/NILM_AI/cache/train60_v29"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 자리 만들기 (사용자 승인, 2026-09-10) ──────────────────────────────────
# v28 캐시·홀드아웃을 지운다. cnn_v28.pt/json 은 이미 로컬에 받아 뒀고 meta 로
# 언제든 다시 구울 수 있다 (16분).
for D in "$HOME/NILM_AI/cache/train60_v28" "$HOME/NILM_AI/processed_data/holdout60_v25"; do
  if [ -d "$D" ]; then echo "제거: $D ($(du -sh "$D" 2>/dev/null | cut -f1))"; rm -rf "$D"; fi
done

# 관문 0: 여유 공간. du 는 공용 계정에서 권한 거부로 비영을 낸다 — 감싼다.
USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
if [ -z "${USED:-}" ]; then echo "홈 사용량을 못 재었다 — 관문 건너뜀"; else
  FREE=$((102400 - USED))
  echo "여유 ${FREE} MB · 이 캐시는 약 23000 MB 필요"
  if [ "$FREE" -lt 26000 ]; then echo "여유 부족 — 중단"; exit 1; fi
fi

# numba JIT 선컴파일 — 워커 폭주 전에 한 번
$PY -X utf8 -c "
import numpy as np
from src.synthesis.coupling import SmpsCircuit
c = SmpsCircuit(); rel = np.zeros(15, complex); rel[0] = 1
c.current('laptop_charger', 45.0, rel, 224.0); print('선컴파일 OK')
"

# 관문 1: 텍스처 곡선·전압 꼬리
$PY -X utf8 -c "
from src.run_build_sp_curves import TEX_CURVES
from src.synthesis.sp_curves import load_curves
t = load_curves(TEX_CURVES); assert len(t) >= 8, ('곡선 부족', len(t))
print('텍스처 곡선', len(t), '개')
from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, load_vtail
tl = load_vtail(DEFAULT_VTAIL_NPZ)
assert tl and 'E3' in tl, '전압 꼬리에 E3 가 없다'
print('전압 꼬리 키', sorted(tl))
"

# 관문 2: 풀에 새 녹화가 들어와 있는가 (v28 과 같은 풀이어야 한다)
$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.segment_pool import SegmentPool
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',))
c = p.appliance_activations.get('laptop_charger', []); m = p.appliance_activations.get('minipc', [])
print('충전기 활성화 %d · 미니PC %d' % (len(c), len(m)))
assert len(c) >= 16 and len(m) >= 9, '새 녹화가 안 들어왔다'
assert max(x.duration_s for x in c) > 800 and max(x.duration_s for x in m) > 250
print('관문 통과')
"

# 관문 3: **새 레시피가 실려 있는가.** 코드를 안 올렸으면 여기서 선다 —
# 조용히 옛 거동으로 가면 v28 과 같은 캐시가 나온다.
$PY -X utf8 -c "
import sys; sys.path.insert(0,'.')
from src import env_guard
from src.synthesis.synthesizer import LoadSynthesizer
from src.run_recipe_mix_probe import PRESETS
assert hasattr(LoadSynthesizer, 'synthesize_steady_loaded_window'), 'steady_loaded 가 없다 — 코드를 안 올렸다'
m = PRESETS['steady2']; s = sum(m.values())
assert abs(s - 1.0) < 1e-9, ('믹스 합', s)
assert m['resistive_overlap'] > 0.15 and m['steady_loaded'] > 0.10, m
print('steady2 합 %.6f · resistive_overlap %.3f · steady_loaded %.3f'
      % (s, m['resistive_overlap'], m['steady_loaded']))
"

# ── v28 에서 바뀐 것은 **믹스 하나**다 (13.83.19) ───────────────────────────
#   --recipe-mix decorr -> steady2
#     · `steady_loaded` 0.121 추가 — 부하 실린 채 창 내내 정상상태인 창.
#       v28 은 그런 창이 133개 중 1개(0.8%)뿐이었고 실측은 11.7% 다.
#     · `resistive_overlap` 0.035 -> 0.162 — 핫플(0.883)·오븐(0.808)의 유일
#       공급원을 v27 수준 위로 되돌린다. decorr 가 그것을 굶기고 있었다.
#   --smps-focus-off-p 는 0.4 그대로.
# 굽기 전 점검(창 500): 오븐 0.201 · 핫플 0.220 · 에어컨 0.040 ·
#   phi 미↔충 +0.072 · SMPS>=2 0.568 · 전이0 몫 · σ 는 로컬 관문에 있다.
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
  --sp-curves --sp-per-texture --vtail

echo "=== meta (v28 과 recipe_mix 만 달라야 한다) ==="
$PY -X utf8 -c "
import json
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
for k in ('n_windows','window_cycles','seed','sp_curves','sp_per_texture','vtail','background',
          'couple_ext','carrier_apps','state_mix','level_scramble','time_split',
          'smps_focus_off_p','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
print('  recipe_mix', json.dumps(d.get('recipe_mix'), ensure_ascii=False))
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
# 굽고 나서의 관문 — 13.83.13 처럼 놓치지 않게 여기서 세운다.
import sys
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12,
      'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad:
    sys.exit('양성률 하한 미달: %s' % bad)
print('  양성률 하한 통과')
"
du -sh "$OUT"
echo "홈 사용량: $(du -sh $HOME 2>/dev/null | cut -f1) / 쿼터 100G"
echo "끝 $(date '+%F %H:%M:%S')"
