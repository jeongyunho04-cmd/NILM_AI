#!/bin/bash
# 14.44 ② — **v32t 홀드아웃**. `holdout60_v32` 와 조리법을 한 글자도 안 바꾸고
#           `--vtex-step-s 20` **하나만** 더한다.
#
# ⚠⚠ `run_build_holdout` 에 `--vtex-step-s` 가 **없었다** — 14.33 ⓑ 가 `run_build_traincache`
#   에서 고친 바로 그 배선 구멍이 이쪽에 그대로 남아 있었다. 14.44 에서 메웠다.
#   0 이면 그 블록을 통째로 건너뛰므로 **끄면 비트 동일**이다.
#
# ⚠ **판정용이 아니다.** v32 대 v32t 의 A/B 는 `run_diag_rollback`(실측)으로 한다.
#   합성 홀드아웃 숫자는 각 팔이 **자기 분포**에서 잰 값이라 팔끼리 직접 비교하면 안 된다.
#   이 셋은 v32t 팔의 학습 감시자이자 14.43 의 배치 비교용이다.
#
# 관문: 다 만든 뒤 `holdout60_v32` 의 meta 와 대조해 **`vtex_step_s` 말고 다 같은지** 본다.
#
#SBATCH -J nilm_ho32t
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT=processed_data/holdout60_v32t
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--vtex-step-s" || {
  echo "--vtex-step-s 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d processed_data/holdout60_v32 || { echo "대조할 v32 홀드아웃이 없다"; exit 1; }

$PY -X utf8 -m src.run_build_holdout --out "$OUT" \
    --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
    --recipe-mix steady2 \
    --smps-focus-off-p 0.4 \
    --state-mix minipc_balanced \
    --carrier-on oven \
    --couple-ext \
    --sp-curves --sp-per-texture --vtail \
    --float-fill charger_float \
    --steady-crop smps_steady \
    --standby-jitter-cap 95 \
    --vtex-step-s 20

echo "=== 관문: v32 와 **vtex_step_s 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
a = json.load(open('processed_data/holdout60_v32/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'vtex_step_s', 'content_sha256', 'bytes', 'built_at', 'positive_rate', 'recipe_counts'}
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(a.get(k), sort_keys=True, default=str)
        != json.dumps(b.get(k), sort_keys=True, default=str)]
if diff: sys.exit('❌ vtex_step_s 말고도 다르다: %s' % diff)
if float(b.get('vtex_step_s', 0)) != 20.0:
    sys.exit('❌ vtex_step_s 가 %r 이다 — 안 걸렸다' % b.get('vtex_step_s'))
if int(b['target_index']) != int(a['target_index']):
    sys.exit('❌ target_index 가 다르다')
print('관문 ✅ vtex_step_s %s 하나만 다르다 · 타깃 %d/%d · %d창'
      % (b['vtex_step_s'], b['target_index'], b['window_cycles'], b['n_windows']))
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
