#!/bin/bash
# 14.62 — **v32h 홀드아웃**: `holdout60_v32` 와 `--harmonic-z` 하나만 다르다.
#
# ⚠⚠ **학습 캐시와 반드시 같은 손잡이여야 한다.** 다르면 홀드아웃의 텍스처가 다른 Z 로
#   벗겨져 학습 분포와 평가 분포가 갈린다 (`holdout._w_init` 의 14.59 주석).
#   `cache_v32h_par.sbatch` 와 **짝**이다. 하나만 돌리지 마라.
#
# ⚠ 최종 판정은 실측(test_1~5)에서 한다 — 이 홀드아웃은 학습 중 검증용이다. 그래도
#   분포가 갈리면 조기 종료·최적 에포크 고르기가 다른 자를 쓰게 된다.
#SBATCH -J nilm_ho32h
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
OUT=processed_data/holdout60_v32h
W=${WORKERS:-24}

echo "노드 $(hostname) · 워커 ${W} · $(date '+%F %H:%M:%S')"

# ── 관문: 배선이 닿았나 ────────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--harmonic-z" || {
  echo "--harmonic-z 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--workers" || {
  echo "--workers 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d processed_data/holdout60_v32 || { echo "대조할 v32 홀드아웃이 없다"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 재라.** 이 둘은 진짜 창을 합성해 본다.
$PY -X utf8 src/run_gate_zharm.py
$PY -X utf8 src/run_gate_hzwire.py
$PY -X utf8 src/run_gate_holdout_par.py --n 120 --tmp "${TMPDIR:-/tmp}"

# ⚠⚠ 인자를 **한 곳에 모아** 굽기 **전에** 되짚는다. 984095 가 여기서 조용히 어긋났다 —
#   sbatch 를 만들 때 문자열 치환이 접혀 `--vtex-step-s 10 --vtex-seg-s 10` 이 그대로
#   남고 `--harmonic-z` 는 안 들어갔는데, 내 assert 가 **파일 전체**에서 `--harmonic-z`
#   를 찾아 (위 `--help | grep` 줄에) 걸려 통과했다. **굽기 명령 자체**를 봐야 한다.
FLAGS="--windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
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
        --harmonic-z"
echo "굽기 인자: $FLAGS"
case "$FLAGS" in *--vtex*) echo "vtex 가 남았다 — 학습 캐시는 seg 끔이다"; exit 1;; esac
case "$FLAGS" in *--harmonic-z*) ;; *) echo "--harmonic-z 가 없다 — 이 판의 목적이다"; exit 1;; esac

$PY -X utf8 -m src.run_build_holdout --out "$OUT" $FLAGS

echo "=== 관문: v32 와 **harmonic_z·병렬 표시 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
a = json.load(open('processed_data/holdout60_v32/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'harmonic_z', 'workers_chunked', 'chunk_windows',
        'content_sha256', 'bytes', 'built_at', 'positive_rate', 'recipe_counts'}
# ⚠ '키 없음' 과 '0/False' 는 **같은 뜻**이다 — 'train60_v32' 는 'vtex_step_s' 가 생기기
#   전에 구운 캐시라 키가 아예 없고, 새 캐시는 0.0 으로 적는다. 그대로 견주면 983992 처럼
#   자료가 멀쩡한데 관문이 운다 (자료 비트 동일은 굽기 관문 ③이 이미 증명했다).
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
        != json.dumps(nz(b.get(k)), sort_keys=True, default=str)]
if diff: sys.exit('❌ 텍스처 손잡이 말고도 다르다: %s' % diff)
if not b.get('harmonic_z'):                sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
if not b.get('workers_chunked'):           sys.exit('❌ 병렬로 안 구웠다 — 워커가 0으로 떨어졌다')
if int(b['target_index']) != int(a['target_index']): sys.exit('❌ target_index 가 다르다')
print('관문 ✅ harmonic_z %s · 청크 %d창 · 타깃 %d/%d · %d창'
      % (b['harmonic_z'], b['chunk_windows'],
         b['target_index'], b['window_cycles'], b['n_windows']))
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
