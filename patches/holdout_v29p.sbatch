#!/bin/bash
#SBATCH -J nilm_h29
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
OUT="$HOME/NILM_AI/processed_data/holdout60_v26"

echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠ 캐시와 **병렬로** 돈다 (사용자 지적, 2026-09-10). 홀드아웃은 캐시를 실제로
#   읽지 않는다 — 같은 생성기 설정으로 시드·time_split 만 달리 돌리는 것이다.
#   그래서 캐시 meta 를 보는 관문 0 을 뺐다. **짝 검사는 학습 작업(train_v29)에
#   그대로 있다** — 거기서 recipe_mix·focus_off·state_mix·carrier·window_cycles 를
#   전부 대조하고 하나라도 어긋나면 학습이 안 돈다.
#   자리: v28 삭제 뒤 여유 35,962 MB · 캐시 23,000 + 홀드아웃 5,000 = 28,000 MB.

USED=$( { du -sm "$HOME" 2>/dev/null || true; } | tail -1 | cut -f1 )
[ -n "${USED:-}" ] && echo "홈 ${USED} MB / 102400 MB · 홀드아웃은 약 5000 MB 필요"

# ── 홀드아웃: 캐시와 **같은 생성기 설정**. 다른 것은 시드와 time_split 뿐이다 ──
time $PY -X utf8 -m src.run_build_holdout \
  --out "$OUT" \
  --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
  --recipe-mix steady2 \
  --smps-focus-off-p 0.4 \
  --state-mix minipc_balanced \
  --carrier-on oven \
  --couple-ext \
  --sp-curves --sp-per-texture --vtail

if [ -f cache/train60_v29/meta.json ]; then
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v29/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
print('=== 캐시 대 홀드아웃 (달라도 되는 것: n_windows · seed · time_split) ===')
bad = []
for k in ('window_cycles','sp_curves','sp_per_texture','vtail','background',
          'couple_ext','carrier_apps','state_mix','smps_focus_off_p'):
    same = a.get(k) == b.get(k)
    if not same:
        bad.append(k)
    print('  %-20s %-28s %s%s' % (k, a.get(k), b.get(k), '' if same else '   <- 다름'))
print('  recipe_mix 같은가:', a.get('recipe_mix') == b.get('recipe_mix'))
if a.get('recipe_mix') != b.get('recipe_mix'):
    bad.append('recipe_mix')
if bad:
    sys.exit('캐시와 홀드아웃이 어긋난다: %s' % bad)
print('  짝 관문 통과')
"
else
  echo '캐시가 아직 없다 — 짝 검사는 학습 작업이 한다'
fi
du -sh "$OUT"
echo "끝 $(date '+%F %H:%M:%S')"
