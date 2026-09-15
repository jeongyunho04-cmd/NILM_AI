#!/bin/bash
# 14.108 — **v32hs3 홀드아웃**: `holdout60_v32h` 와 **텍스처 토막 셋**만 다르다.
#   `--vtex-step-s 1.25 --vtex-seg-s 1.25 --vtex-coarse-s 25` (학습 캐시 train60_v32hs3 과 짝)
#
# ⚠⚠ 왜 이 판이 위험한가 — `--vtex-coarse-s` 를 만들 때 **학습 캐시 경로만** 배선했다.
#   홀드아웃은 `run_build_holdout` -> `build_holdout` -> `_build_generator` 라는 **다른
#   입구**라, 학습 쪽 관문이 전부 통과해도 홀드아웃은 조용히 **균일**로 구워진다. 그러면
#   학습 분포와 평가 분포가 갈리고 그 차이를 모델 성능으로 읽는다.
#   오늘만 같은 꼴을 셋 만났다(14.62 · `_init` 위치 밀림 · 이것).
#   그래서 `run_gate_hocoarse` 가 **진짜 CLI 로 40창을 세 판 구워** 견준다.
#
#SBATCH -J nilm_ho32hs3
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24           # 노드당 상한(gpu1·gpu6 은 40)보다 아래로 — backfill 이 잘 문다
#SBATCH --mem=64G
# 실측 앵커: 학습 캐시가 40코어에서 1.43 코어·초/창 (14.107). 8000창 / 24워커 ≈ 8분.
# 관문이 40창 x 4판 = 4~6분. 합 ~15분. HPC_RULES §2 로 3배.
#SBATCH -t 00:45:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT=processed_data/holdout60_v32hs3
REF=processed_data/holdout60_v32h
TRAIN=cache/train60_v32hs3
W=${WORKERS:-24}

echo "노드 $(hostname) · 워커 ${W} · $(date '+%F %H:%M:%S')"

# ── 관문: 배선이 닿았나 ────────────────────────────────────────────────────
for f in --harmonic-z --workers --vtex-coarse-s; do
  $PY -X utf8 -m src.run_build_holdout --help | grep -q -- "$f" \
    || { echo "$f 가 없다 — 코드를 안 올렸다"; exit 1; }
done
test -d "$REF"   || { echo "대조할 v32h 홀드아웃이 없다: $REF"; exit 1; }
test -d "$TRAIN" || { echo "짝 맞출 학습 캐시가 없다: $TRAIN"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 재라.** 아래 셋은 진짜 창을 합성해 본다.
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 src/run_gate_zharm.py
$PY -X utf8 src/run_gate_hzwire.py
$PY -X utf8 src/run_gate_bakeseg.py
$PY -X utf8 src/run_gate_hocoarse.py
$PY -X utf8 src/run_gate_holdout_par.py --n 120 --tmp "${TMPDIR:-/tmp}"

# ⚠⚠ 인자를 **한 곳에 모아** 굽기 **전에** 되짚는다. 984095 가 여기서 조용히 어긋났다.
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
        --harmonic-z \
        --vtex-step-s 1.25 --vtex-seg-s 1.25 --vtex-coarse-s 25"
echo "굽기 인자: $FLAGS"
# ⚠ v32h 판과 반대다 — 여기서는 vtex 가 **있어야** 한다
case "$FLAGS" in *"--vtex-step-s 1.25 --vtex-seg-s 1.25 --vtex-coarse-s 25"*) ;; *)
  echo "텍스처 토막 셋이 기대와 다르다"; exit 1;; esac
case "$FLAGS" in *--harmonic-z*) ;; *) echo "--harmonic-z 가 없다"; exit 1;; esac

$PY -X utf8 -m src.run_build_holdout --out "$OUT" $FLAGS

echo "=== 관문 ①: v32h 와 **텍스처 토막 셋 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
a = json.load(open('$REF/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'vtex_step_s', 'vtex_seg_s', 'vtex_coarse_s',
        'content_sha256', 'bytes', 'built_at', 'positive_rate', 'recipe_counts'}
# ⚠ '키 없음' 과 '0/False' 는 같은 뜻이다 — 옛 홀드아웃은 그 키가 생기기 전에 구웠다
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
        != json.dumps(nz(b.get(k)), sort_keys=True, default=str)]
if diff: sys.exit('❌ 텍스처 토막 셋 말고도 다르다: %s' % diff)
if not b.get('harmonic_z'):      sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
if not b.get('workers_chunked'): sys.exit('❌ 병렬로 안 구웠다 — 워커가 0으로 떨어졌다')
if int(b['target_index']) != int(a['target_index']): sys.exit('❌ target_index 가 다르다')
print('  관문 ① OK · 타깃 %d/%d · %d창' % (b['target_index'], b['window_cycles'], b['n_windows']))
"

echo '=== 관문 ②: **학습 캐시와 처치가 같은가** (이 판의 목적) ==='
$PY -X utf8 -c "
import json, sys
h = json.load(open('$OUT/meta.json', encoding='utf-8'))
t = json.load(open('$TRAIN/meta.json', encoding='utf-8'))
bad = []
for k in ('harmonic_z', 'vtex_step_s', 'vtex_seg_s', 'vtex_coarse_s'):
    hv, tv = h.get(k), t.get(k)
    if isinstance(tv, bool) or isinstance(hv, bool):
        same = bool(hv) == bool(tv)
    else:
        same = float(hv or 0) == float(tv or 0)
    print('    %-14s 홀드아웃 %-7r 학습 %-7r  %s' % (k, hv, tv, 'OK' if same else '** 다르다 **'))
    if not same: bad.append(k)
if bad: sys.exit('❌ 홀드아웃과 학습 캐시의 처치가 다르다: %s' % bad)
print('  관문 ② OK — 넷 다 같다')
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
