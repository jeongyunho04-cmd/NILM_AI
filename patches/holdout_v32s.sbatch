#!/bin/bash
# 14.51 — **v32s 홀드아웃**. `holdout60_v32` 와 조리법을 한 글자도 안 바꾸고
#         `--vtex-step-s 10 --vtex-seg-s 10` 둘만 더한다. 학습 캐시와 **짝**이어야 한다.
#
# ★ **병렬로 굽는다** (`--workers`). 지금까지 이것은 단일 프로세스였고 `-t 03:00` 을 불렀다 —
#   v32s 는 창당 226ms 라 직렬이면 8,000창에 30분이다. 워커 24개면 몇 분이다.
#
#   ⚠ **워커 수를 바꿔도 같은 바이트**가 나온다. 시드를 **청크 번호**로 걸고 `imap`(순서 보장)
#     으로 받는다 — `traincache` 가 14.13 항등 검정으로 확인한 그 설계다.
#     `run_gate_holdout_par.py` ① 이 워커 2개와 5개를 비트 단위로 대 본다.
#   ⚠ **`--workers 0`(직렬)과 1 이상(병렬)은 서로 다른 홀드아웃이다** — 난수를 자르는 방식이
#     다르다. 옛 체크포인트 점수와 견주려면 그때 쓴 홀드아웃 폴더를 그대로 써라.
#     meta 에 `workers_chunked` 와 `chunk_windows` 가 남는다.
#   ⚠ `--chunk-windows` 를 **워커 수에 맞춰 바꾸지 마라** — 그러면 위 항등이 깨진다.
#     100 은 8,000창 = 80청크라 워커 24개에도 꼬리가 짧고, 청크 하나가 돌려주는 X 가 65MB 다.
#
# ⚠ **판정용이 아니다.** v32 대 v32s 의 A/B 는 `run_diag_rollback`(실측)으로 한다.
#   합성 홀드아웃 숫자는 각 팔이 **자기 분포**에서 잰 값이라 팔끼리 직접 비교하면 안 된다.
#
#SBATCH -J nilm_ho32s
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24           # 노드당 상한(gpu1·gpu6 은 40)보다 아래로 — backfill 이 잘 문다
#SBATCH --mem=64G       # 워커당 ~1.5GB + 청크 24 x 65MB + memmap
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). 옛 판이 `-t 03:00` 을 부른 것은 9배였다.
#   ⚠ `squeue --start`·`sbatch --test-only` 는 backfill 을 모델링하지 않는다.
#SBATCH -t 00:15:00     # 직렬 4.42 win/s 실측 -> 24워커 ≈ 3~5분. 그 3배로 잡아 둔다
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT=processed_data/holdout60_v32s
W=${WORKERS:-24}

echo "노드 $(hostname) · 워커 ${W} · $(date '+%F %H:%M:%S')"

# ── 관문: 배선이 닿았나 ────────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--vtex-seg-s" || {
  echo "--vtex-seg-s 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--workers" || {
  echo "--workers 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d processed_data/holdout60_v32 || { echo "대조할 v32 홀드아웃이 없다"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 재라.** 이 둘은 진짜 창을 합성해 본다.
$PY -X utf8 src/run_gate_vtexseg.py
$PY -X utf8 src/run_gate_holdout_par.py --n 120 --tmp "${TMPDIR:-/tmp}"

$PY -X utf8 -m src.run_build_holdout --out "$OUT" \
    --windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
    --workers "$W" --chunk-windows 100 \
    --recipe-mix steady2 \
    --smps-focus-off-p 0.4 \
    --state-mix minipc_balanced \
    --carrier-on oven \
    --couple-ext \
    --sp-curves --sp-per-texture --vtail \
    --float-fill charger_float \
    --steady-crop smps_steady \
    --standby-jitter-cap 95 \
    --vtex-step-s 10 --vtex-seg-s 10

echo "=== 관문: v32 와 **텍스처 손잡이·병렬 표시 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
a = json.load(open('processed_data/holdout60_v32/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'vtex_step_s', 'vtex_seg_s', 'workers_chunked', 'chunk_windows',
        'content_sha256', 'bytes', 'built_at', 'positive_rate', 'recipe_counts'}
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(a.get(k), sort_keys=True, default=str)
        != json.dumps(b.get(k), sort_keys=True, default=str)]
if diff: sys.exit('❌ 텍스처 손잡이 말고도 다르다: %s' % diff)
if float(b.get('vtex_step_s', 0)) != 10.0: sys.exit('❌ vtex_step_s = %r' % b.get('vtex_step_s'))
if float(b.get('vtex_seg_s', 0)) != 10.0:  sys.exit('❌ vtex_seg_s = %r' % b.get('vtex_seg_s'))
if not b.get('workers_chunked'):           sys.exit('❌ 병렬로 안 구웠다 — 워커가 0으로 떨어졌다')
if int(b['target_index']) != int(a['target_index']): sys.exit('❌ target_index 가 다르다')
print('관문 ✅ step_s %s · seg_s %s · 청크 %d창 · 타깃 %d/%d · %d창'
      % (b['vtex_step_s'], b['vtex_seg_s'], b['chunk_windows'],
         b['target_index'], b['window_cycles'], b['n_windows']))
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
