#!/bin/bash
# 굽기가 끝나면 이어붙이고 A/B 를 건다 — **로그인 노드에서** 돌린다 (14.26).
#
# ⚠ 계산 노드 스크립트가 아니므로 여기서는 `mmlsquota` 를 불러도 된다 (HPC_RULES §7-5 는
#   계산 노드 얘기다). 오히려 이어붙이기 전에 자리를 확인해야 한다.
#
#   bash patches/merge_and_submit.sh <굽기 작업번호>
set -uo pipefail          # ⚠ -e 는 안 건다 — 관문마다 직접 판정하고 메시지를 남긴다
cd "$HOME/NILM_AI"
PY=./env/bin/python
JOB="${1:?굽기 작업번호를 달라}"
GEN="${GEN:-v32}"
OUT="cache/train60_${GEN}"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── ① 굽기가 끝날 때까지 ────────────────────────────────────────────────
say "굽기 $JOB 을 기다린다"
while squeue -j "$JOB" -h -o '%T' 2>/dev/null | grep -q .; do sleep 60; done

ST=$(sacct -j "$JOB" --format=JobID%16,State -n -P | grep -vE 'batch|extern')
say "굽기 종료"
echo "$ST" | sed 's/^/    /'
if echo "$ST" | grep -qv 'COMPLETED'; then
  if ! echo "$ST" | grep -q 'COMPLETED'; then
    say "⚠ 어느 토막도 COMPLETED 가 아니다 — 멈춘다"; exit 1
  fi
fi
BAD=$(echo "$ST" | grep -c -v 'COMPLETED' || true)
if [ "${BAD:-0}" -gt 0 ]; then say "⚠ COMPLETED 아닌 토막이 있다 — 멈춘다"; exit 1; fi

for i in 0 1; do
  test -f "cache/train60_${GEN}_s${i}/meta.json" || { say "⚠ 토막 $i 의 meta.json 이 없다"; exit 1; }
done
say "토막 크기:"; du -sh cache/train60_${GEN}_s0 cache/train60_${GEN}_s1 | sed 's/^/    /'
say "이어붙이기 전 쿼터:"; /usr/lpp/mmfs/bin/mmlsquota --block-size auto | tail -1 | sed 's/^/    /'

# ── ② 이어붙이기 ────────────────────────────────────────────────────────
say "이어붙인다 -> $OUT"
$PY -X utf8 src/run_merge_traincache.py --out "$OUT" \
    "cache/train60_${GEN}_s0" "cache/train60_${GEN}_s1" 2>&1 | sed 's/^/    /'
MRC=${PIPESTATUS[0]}
[ "$MRC" -eq 0 ] || { say "⚠ 이어붙이기 실패 (rc=$MRC) — 멈춘다"; exit 1; }

# ── ③ 합본 검사 ─────────────────────────────────────────────────────────
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
if d.get('shard'):            sys.exit('합본에 shard 가 남아 있다')
if d['n_windows'] != 300000:  sys.exit('창 수가 30만이 아니다: %d' % d['n_windows'])
if d.get('sibling_rotate'):   sys.exit('sibling_rotate 가 있다 (14.4 위반)')
for k in ('float_fill','steady_crop','standby_jitter_cap','vtail','couple_ext','sp_curves','sp_per_texture'):
    if not d.get(k): sys.exit('v32 설정이 빠졌다: %s' % k)
pr = d.get('positive_rate', {})
lo = {'air_conditioner':0.025,'oven':0.15,'hotplate':0.12,'electiric_kettle':0.12,'hair_dryer':0.10}
bad = [(k, round(pr.get(k,0),3)) for k,v in lo.items() if pr.get(k,0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
print('    합본 관문 통과 — 창 %d · 이어붙인 토막 %s' % (d['n_windows'], d.get('merged_from')))
print('    양성률 %s' % json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
" 2>&1
[ "${PIPESTATUS[0]:-1}" -eq 0 ] || { say "⚠ 합본 관문 실패 — 멈춘다"; exit 1; }
say "이어붙인 뒤 쿼터:"; /usr/lpp/mmfs/bin/mmlsquota --block-size auto | tail -1 | sed 's/^/    /'

# ── ④ A/B 제출 ──────────────────────────────────────────────────────────
say "A/B 제출 (cnn_signv.sbatch)"
SB=$(sbatch --parsable patches/cnn_signv.sbatch) || { say "⚠ 제출 실패"; exit 1; }
say "제출됨 — 작업 $SB"
squeue -j "$SB" -o '%.10i %.12j %.7P %.2t %.9L %.4C %.6m %.9R' | sed 's/^/    /'
echo
say "⚠ 토막은 **안 지웠다** — 사람이 확인하고 지운다:"
echo "    rm -rf cache/train60_${GEN}_s0 cache/train60_${GEN}_s1   # 24GB 회수"
