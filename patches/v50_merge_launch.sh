#!/bin/bash
# 14.390 — v50 캐시를 **이어붙이고 Ĝ 를 넣고 학습을 던진다**. 로그인 노드에서 돌린다.
#
#   bash patches/v50_merge_launch.sh <굽기 작업번호>
#
# ⚠ `merge_and_submit.sh` 를 안 쓴다 — 그것은 **토막 2개**(`_s0` `_s1`) 전제로 박혀
#   있고 `cnn_signv.sbatch` 를 던진다. v32 시절 물건이라 전제가 바뀌었다
#   ([[find-patch-collisions-by-shape]] — "전제가 바뀐 땜질").
set -uo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
JOB="${1:?굽기 작업번호를 달라}"
GEN="${GEN:-v50}"
OUT="cache/train60_${GEN}"
NSHARD="${NSHARD:-8}"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

say "굽기 $JOB 을 기다린다"
while squeue -j "$JOB" -h -o '%T' 2>/dev/null | grep -q .; do sleep 60; done

ST=$(sacct -j "$JOB" --format=JobID%16,State -n -P | grep -vE 'batch|extern')
say "굽기 종료"; echo "$ST" | sed 's/^/    /'
if [ "$(echo "$ST" | grep -c 'COMPLETED')" -ne "$NSHARD" ]; then
  say "⚠ COMPLETED 가 $NSHARD 개가 아니다 — 멈춘다"; exit 1
fi

SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do
  d="cache/train60_${GEN}_s${i}"
  test -f "$d/meta.json" || { say "⚠ $d/meta.json 이 없다"; exit 1; }
  SHARDS="$SHARDS $d"
done

# ⚠ **굽기 설정이 실제로 걸렸나** — 토막 전부에서 확인한다
$PY -X utf8 -c "
import json, sys
bad = []
for i in range($NSHARD):
    d = json.load(open('cache/train60_${GEN}_s%d/meta.json' % i, encoding='utf-8'))
    if d.get('phase_jitter_map') != 'measured':
        bad.append((i, d.get('phase_jitter_map')))
if bad: sys.exit('phase_jitter_map 이 measured 가 아닌 토막: %s' % bad)
print('    토막 $NSHARD 개 전부 phase_jitter_map=measured')
" || exit 1

say "이어붙인다 -> $OUT"
# ⚠⚠ `--rm-shards` 는 **선택이 아니다** — 쿼터가 100G 인데 토막 25.7G + 합본 25.7G
#   를 같이 들고 있으면 123G 로 **하드 한계(110G)를 넘는다.** 986460 이 그렇게
#   Bus error(memmap 쓰기 실패)로 세 토막이 죽었다. 이 옵션은 파일마다 **바이트를
#   세고 나서** 지우므로 반쯤 복사된 것을 지우지 않는다.
$PY -X utf8 src/run_merge_traincache.py --out "$OUT" --rm-shards $SHARDS 2>&1 | sed 's/^/    /'
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "⚠ 이어붙이기 실패 — 멈춘다"; exit 1; }

$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
if d.get('shard'):           sys.exit('합본에 shard 가 남아 있다')
if d['n_windows'] != 300000: sys.exit('창 수가 30만이 아니다: %d' % d['n_windows'])
if d.get('phase_jitter_map') != 'measured': sys.exit('합본에 phase_jitter_map 이 없다')
if d.get('sibling_rotate'):  sys.exit('sibling_rotate 가 있다 (14.4 위반)')
for k in ('float_fill','steady_crop','standby_jitter_cap','vtail','couple_ext','sp_curves','sp_per_texture'):
    if not d.get(k): sys.exit('v49 설정이 빠졌다: %s' % k)
pr = d.get('positive_rate', {})
lo = {'air_conditioner':0.025,'oven':0.15,'hotplate':0.12,'electiric_kettle':0.12,'hair_dryer':0.10}
bad = [(k, round(pr.get(k,0),3)) for k,v in lo.items() if pr.get(k,0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
print('    합본 관문 통과 — 창 %d · 토막 %d개 · phase_jitter_map=%s'
      % (d['n_windows'], len(d.get('merged_from') or []), d.get('phase_jitter_map')))
" || { say "⚠ 합본 관문 실패 — 멈춘다"; exit 1; }

# ── Ĝ 를 넣는다 (학습 관문이 요구한다) ──────────────────────────────────────
say "Ĝ 굽기 제출"
GJ=$(CACHE="$OUT" sbatch --parsable --export=ALL,CACHE="$OUT" ghat_v49.sbatch) \
  || { say "⚠ Ĝ 제출 실패"; exit 1; }
say "Ĝ 작업 $GJ — 기다린다"
while squeue -j "$GJ" -h -o '%T' 2>/dev/null | grep -q .; do sleep 30; done
sacct -j "$GJ" --format=JobID%14,State -n -P | grep -vE 'batch|extern' | sed 's/^/    /'
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8')).get('g_hat') or {}
if not d.get('v_ref'): sys.exit('Ĝ 가 안 들어갔다')
print('    Ĝ 들어감 — 중앙 %.2f mS · 차수 %s' % (d.get('median_ms', -1), d.get('volt_orders')))
" || { say "⚠ Ĝ 실패 — 멈춘다"; exit 1; }

# ── 학습 ────────────────────────────────────────────────────────────────────
say "학습 제출 (cnn_pjit_seeds.sbatch · 씨앗 6)"
SB=$(sbatch --parsable cnn_pjit_seeds.sbatch) || { say "⚠ 제출 실패"; exit 1; }
say "제출됨 — 작업 $SB"
squeue -j "$SB" -o '%.10i %.12j %.7P %.2t %.9L %.4C %.6m %.9R' | sed 's/^/    /'
echo
say '토막은 --rm-shards 로 합치며 지워졌다 (쿼터 때문에 필수였다)'
