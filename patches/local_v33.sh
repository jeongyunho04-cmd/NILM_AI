#!/bin/bash
# 13.84.9 — v33: v32 + 형제 전용 차수 비례 회전 (--sibling-rotate). HPC 터널이 막혀 **로컬 RTX 5070** 에서 돈다.
#   캐시·홀드아웃·학습 인수는 v32 (patches/cache_v32.sbatch · holdout_v32.sbatch · train_v32.sbatch) 와 같고
#   --sibling-rotate 하나만 다르다. 짝지은 기준은 cnn_v32 (HPC, 같은 시드·같은 플래그).
#   사용: bash patches/local_v33.sh [ROT_PRESET|JSON]   (기본 smps_rot10)
set -euo pipefail
cd "$(dirname "$0")/.."
ROT="${1:-smps_rot10}"
TAG=cnn_v33
CACHE=cache/train60_v33
HOLD=processed_data/holdout60_v33
PY="python -X utf8"

echo "시작 $(date '+%F %H:%M:%S') · 회전 $ROT"
$PY -m src.run_build_traincache --help | grep -q -- "--sibling-rotate" || { echo "run_build_traincache 에 --sibling-rotate 가 없다"; exit 1; }

if [ ! -f "$HOLD/meta.json" ]; then
  time $PY -m src.run_build_holdout \
    --out "$HOLD" --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
    --recipe-mix steady2 --smps-focus-off-p 0.4 --state-mix minipc_balanced --carrier-on oven \
    --couple-ext --sp-curves --sp-per-texture --vtail \
    --float-fill charger_float --steady-crop smps_steady --standby-jitter-cap 95 \
    --sibling-rotate "$ROT"
fi

if [ ! -f "$CACHE/meta.json" ]; then
  time $PY -m src.run_build_traincache \
    --out "$CACHE" --windows 300000 --window-cycles 3600 --seed 0 --workers 10 \
    --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4 --state-mix minipc_balanced \
    --carrier-on oven --couple-ext --sp-curves --sp-per-texture --vtail \
    --float-fill charger_float --steady-crop smps_steady --standby-jitter-cap 95 \
    --sibling-rotate "$ROT"
fi

$PY -c "
import json, sys
a = json.load(open('$CACHE/meta.json', encoding='utf-8'))
b = json.load(open('$HOLD/meta.json', encoding='utf-8'))
for k in ('smps_focus_off_p', 'recipe_mix', 'state_mix', 'carrier_apps', 'window_cycles', 'float_fill', 'steady_crop', 'standby_jitter_cap', 'sibling_rotate'):
    if a.get(k) != b.get(k): sys.exit('캐시와 홀드아웃의 %s 가 다릅니다: %r vs %r' % (k, a.get(k), b.get(k)))
if not a.get('sibling_rotate'): sys.exit('캐시 meta 에 sibling_rotate 가 없다')
print('짝 관문 통과 · 캐시 %d창 · 홀드아웃 %d창 · 회전 %s' % (a['n_windows'], b['n_windows'], a['sibling_rotate']))
"

# cnn_v32 와 같은 명령 (train_v32.sbatch). 캐시·홀드아웃·태그만 v33.
time $PY -m src.run_train_cnn \
    --cache "$CACHE" --holdout "$HOLD" \
    --epochs 300 --epoch-windows 50000 --w-z 0.3 --w-over 0.0 \
    --eval-every 10 --cache-workers 4 \
    --harm-even-magnitude --state-signatures --standby-operating session \
    --tag $TAG

echo "=== 홀드아웃 v32(회전 없음) 와 v33(회전) 에 v31 · v32 · v33 를 나란히 ==="
for H in processed_data/holdout60_v32 "$HOLD"; do
  [ -f "$H/meta.json" ] && $PY -m src.run_score_holdout --ckpt results/cnn_v31.pt results/cnn_v32.pt results/$TAG.pt --holdout "$H" 2>&1 | tail -40 || true
done
echo "=== 실측 ==="
$PY src/run_diag_smps_files.py $TAG cnn_v32 2>&1 | tail -20
$PY src/run_diag_stage1.py $TAG --stems test_1 test_2 test_3 test_4 2>&1 | grep -v "^①" | tail -40
echo "끝 $(date '+%F %H:%M:%S')"
