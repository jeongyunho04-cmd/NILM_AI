#!/bin/bash
# 14.54 — v32s 토막 병합 + **병렬** 홀드아웃. `--dependency=afterok:<굽기 배열>` 로 건다.
#
# ⚠⚠ **쿼터가 빠듯하다.** 병합은 토막(24G)을 읽어 통짜(23G)를 쓴다. `--rm-shards` 가
#   **파일 하나를 복사한 직후 그것을 지우므로** 최고점은 24G 가 아니라 ~26G 다.
#   그래도 제출 **전에** 로그인 노드에서 확인하라 (HPC_RULES §7-5 — 여기서 `mmlsquota` 를
#   부르면 안 된다. 980495 가 네 번째로 그렇게 죽었다):
#       /usr/lpp/mmfs/bin/mmlsquota --block-size auto
#   필요한 여유: **30GB 이상**.
#
# ⚠ 홀드아웃은 `--workers 24` 다 (14.51). 시드를 **청크 번호**로 걸고 `imap` 으로 받으므로
#   **워커 수를 바꿔도 같은 바이트**다. ⚠ `--chunk-windows` 를 워커 수에 맞춰 바꾸지 마라.
#
#SBATCH -J nilm_v32sm
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=64G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2).
#   병합 실측 ~2분(23G) + 홀드아웃 8,000창 / 24워커 ≈ 7분 = 9분. 그 1.5배보다 넉넉히.
#SBATCH -t 00:25:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
GEN=${GEN:-v32s}
NSHARD=${NSHARD:-8}
W=${WORKERS:-24}
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"

SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do
  d="cache/train60_${GEN}_s${i}"
  test -d "$d" || { echo "$d 가 없다 — 굽기가 다 안 끝났다"; exit 1; }
  SHARDS="$SHARDS $d"
done

echo "=== 병합 $NSHARD 토막 -> cache/train60_${GEN}  ($(date '+%H:%M:%S'))"
time $PY -X utf8 src/run_merge_traincache.py --out "cache/train60_${GEN}" $SHARDS --rm-shards
du -sh "cache/train60_${GEN}"
$PY -X utf8 -c "
import json, sys
d = json.load(open('cache/train60_${GEN}/meta.json', encoding='utf-8'))
assert d['shard'] is None, '병합본인데 shard 가 %r' % d['shard']
assert float(d.get('vtex_step_s',0)) == 10.0, 'vtex_step_s %r' % d.get('vtex_step_s')
assert float(d.get('vtex_seg_s',0))  == 10.0, 'vtex_seg_s %r'  % d.get('vtex_seg_s')
print('  관문 ✅ 창 %d · step_s %s · seg_s %s' % (d['n_windows'], d['vtex_step_s'], d['vtex_seg_s']))
"

echo
echo "=== 홀드아웃 (워커 ${W})  ($(date '+%H:%M:%S'))"
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--workers" || {
  echo "--workers 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 src/run_gate_vtexseg.py
$PY -X utf8 src/run_gate_holdout_par.py --n 120 --tmp "${TMPDIR:-/tmp}"
time $PY -X utf8 -m src.run_build_holdout --out "processed_data/holdout60_${GEN}" \
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

echo "=== 관문: v32 홀드아웃과 **텍스처 손잡이·병렬 표시 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
a = json.load(open('processed_data/holdout60_v32/meta.json', encoding='utf-8'))
b = json.load(open('processed_data/holdout60_${GEN}/meta.json', encoding='utf-8'))
skip = {'vtex_step_s','vtex_seg_s','workers_chunked','chunk_windows',
        'content_sha256','bytes','built_at','positive_rate','recipe_counts'}
d = [k for k in set(a)|set(b) if k not in skip and
     json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True, default=str)]
if d: sys.exit('❌ 텍스처 손잡이 말고도 다르다: %s' % d)
if not b.get('workers_chunked'): sys.exit('❌ 병렬로 안 구웠다 — 워커가 0 으로 떨어졌다')
print('관문 ✅ step_s %s · seg_s %s · 청크 %d창 · 타깃 %d/%d · %d창'
      % (b['vtex_step_s'], b['vtex_seg_s'], b['chunk_windows'],
         b['target_index'], b['window_cycles'], b['n_windows']))
"
du -sh "processed_data/holdout60_${GEN}"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
echo
echo "다음: sbatch patches/cnn_v32s_seeds.sbatch   (FRAC 을 983591 승자로)"
