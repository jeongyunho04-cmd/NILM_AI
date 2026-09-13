#!/bin/bash
# 13.84.39 — 전력 헤드가 왜 깎였나: **사슬 목적함수**인가 **학습창 분포**인가 (2026-09-13)
#
# 사용자 지적으로 13.84.37~38 의 귀속이 틀린 것이 드러났다. v25~v37 은 전부 **창 캐시**,
# 사슬 판은 전부 **열 캐시**로 배웠는데 그 차이를 목적함수 탓으로 돌렸다. 둘이 같이 바뀌었다.
#
# 잰 것 (창 `train60_v25` 대 열 `seqraw_v1`, 전체 창 중 그 상태를 보는 비율):
#     오븐 500W+  15.57% -> 7.64%  ·  주전자 800W+ 18.71% -> 9.65%
#     핫플 200W+  16.98% -> 9.64%  ·  드라이기 400W+ 17.70% -> 10.79%
#     에어컨 200W+  2.01% -> 13.30% (6.6배)  ·  1500W 넘는 창 20.5% -> 6.6%
#     동시 3~5대 51.2% -> 27.0%
# 증상이 정확히 이것이다 — 고부하에서만 과소예측(test_3 0.76 · test_5 0.87), 저부하는 멀쩡(0.97).
#
# ⚠ 열 캐시가 통째로 나쁜 것이 **아니다.** 핫플 휴지가 창 캐시에서 **0.0%**, 열 캐시에서 52.6% 다.
#   등록부 실측은 통전 23.3% 이므로 **열 캐시 쪽이 맞다** ([[hotplate-duty-pause-is-on]] 의 그 결함).
#   그래서 `--mix hp` 는 듀티를 안 건드리고 **저항 기기 넷의 포함 확률만** 올린다.
#
#   A 대조      seq 캐시 · 사슬 그대로
#   B 사슬 뺌   seq 캐시 · --w-crf 0 --keep-on 1.0   -> loss = parts["total"] (순수 창별 손실)
#   C 고전력    hp  캐시 · 사슬 그대로
#   D 고전력+상한 hp 캐시 · --w-over 0.1
#
#   A−B = 사슬 목적함수 · A−C = 캐시 분포 · C−D = 빠진 물리 상한 힌지
#
# ⚠ B 의 **사슬 점수는 뜻이 없다** — w_crf=0 이면 머리가 초기값에 머문다. B 는 **전력 헤드**와
#   **창별 점수**로만 읽는다.
# ⚠ 채점은 파일별로 짝지어 읽는다 (13.59 함정). 그리고 ±0.03 은 epoch 대응비교 전에 읽지 않는다.
#SBATCH -J nilm_cfix
#SBATCH -p gpu6,gpu5,gpu4,gpu3,gpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40          # gpu6 QOS 상한 (HPC_RULES 3절)
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH -t 03:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

RECORDS=${RECORDS:-18000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM_A="/dev/shm/$USER/seqraw_seq"
SHM_B="/dev/shm/$USER/seqraw_hp"

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 기록 $RECORDS x2 · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ⚠ 공용 계정이다. 쓰고 나면 램을 반드시 돌려준다.
cleanup() { echo "[정리] 램 반납"; rm -rf "$SHM_A" "$SHM_B"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에 둔다 — HPC_RULES §7-3) ───────────────────────────────────────
$PY -X utf8 -m src.run_build_seqraw --help | grep -q -- "--mix"      || { echo "run_build_seqraw 에 --mix 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq    --help | grep -q -- "--w-over"   || { echo "run_train_seq 에 --w-over 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# 두 판이 **실제로 다른지** 확인한다 ([[verify-the-gate-runs-that-path]]).
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
from src.synthesis.sequence import MIXES
a, b = MIXES['seq'], MIXES['hp']
d = {k: (a[k], b[k]) for k in a if abs(a[k] - b[k]) > 1e-9}
if not d: sys.exit('seq 와 hp 가 같다 — mix 가 안 걸렸다')
print('관문: mix 두 판이 다르다 · ' + ' · '.join('%s %.2f->%.2f' % (k, v[0], v[1]) for k, v in sorted(d.items())))
"

NEED=$(( RECORDS * 4 / 1000 * 2 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB (캐시 둘)"
[ "$FREE" -gt $(( NEED * 12 / 10 )) ] || { echo "램이 모자란다"; exit 1; }

# ── ① 캐시 둘을 램에 굽는다 ────────────────────────────────────────────────
for M in seq hp; do
  D=$([ "$M" = seq ] && echo "$SHM_A" || echo "$SHM_B")
  mkdir -p "$D"
  echo "=== 캐시 $M -> $D ==="
  time $PY -X utf8 -m src.run_build_seqraw \
    --out "$D" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
    --gen v36 --mix "$M" --workers 30 --seed 0
  $PY -X utf8 -c "
import json, sys
m = json.load(open('$D/meta.json', encoding='utf-8'))
if m.get('gen') != 'v36': sys.exit('생성기가 v36 이 아니다')
if m.get('mix') != '$M': sys.exit('mix 가 $M 이 아니다: %r' % m.get('mix'))
o = m.get('gen_opts', {})
for k in ('sibling_rotate','sp_curves','sp_per_texture','vtail','float_fill','steady_crop','state_mix','power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('생성기 설정 %s 가 빠졌다' % k)
print('  캐시 관문 통과 · mix %s · 기록 %d · 단계 %d' % (m['mix'], m['records'], m['steps']))
"
done

# 두 캐시의 **고전력 노출**을 실제로 재서 낸다 — 여기서 안 갈리면 실험이 뜻이 없다.
$PY -X utf8 -c "
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from pathlib import Path
apps = json.loads((Path('$SHM_A') / 'meta.json').read_text(encoding='utf-8'))['appliances']
out = {}
for nm, d in (('seq', '$SHM_A'), ('hp', '$SHM_B')):
    w = np.load(Path(d) / 'y_power.npy', mmap_mode='r')[:3000].reshape(-1, len(apps))
    p = np.asarray(np.load(Path(d) / 'p_observed.npy', mmap_mode='r')[:3000]).ravel()
    out[nm] = (np.asarray(w), p)
print('관문: 고전력 노출 (전체 창 대비) · 목표는 창 캐시 값')
for a, t, tgt in (('oven', 500, 15.57), ('electiric_kettle', 800, 18.71), ('hotplate', 200, 16.98), ('air_conditioner', 200, 2.01)):
    k = apps.index(a)
    print('  %-18s seq %6.2f%%  hp %6.2f%%   (창 캐시 %5.2f%%)'
          % (a, 100 * (out['seq'][0][:, k] > t).mean(), 100 * (out['hp'][0][:, k] > t).mean(), tgt))
for t in (1000, 1500):
    print('  %5dW 넘는 창       seq %6.2f%%  hp %6.2f%%' % (t, 100 * (out['seq'][1] > t).mean(), 100 * (out['hp'][1] > t).mean()))
"
du -sh "$SHM_A" "$SHM_B"; df -h /dev/shm | tail -1
cp "$SHM_A/meta.json" results/seqraw_seq_meta.json
cp "$SHM_B/meta.json" results/seqraw_hp_meta.json

# ── ② 네 판을 동시에 ────────────────────────────────────────────────────────
# 초기점·씨앗·예산 전부 같다. 바뀌는 것은 **캐시**와 **손실 항**뿐이다.
BASE="--epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
      --workers 8 --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
      --init results/cnn_v37.pt --ref results/cnn_v37.pt --lr 1e-4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $BASE --cache "$SHM_A"                             --tag seq_cf_base > "results/seq_cf_base_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $BASE --cache "$SHM_A" --w-crf 0 --keep-on 1.0     --tag seq_cf_nochain > "results/seq_cf_nochain_s${SEED}.log" 2>&1 &
B=$!
CUDA_VISIBLE_DEVICES=2 $PY -X utf8 -m src.run_train_seq $BASE --cache "$SHM_B"                             --tag seq_cf_hp > "results/seq_cf_hp_s${SEED}.log" 2>&1 &
C=$!
CUDA_VISIBLE_DEVICES=3 $PY -X utf8 -m src.run_train_seq $BASE --cache "$SHM_B" --w-over 0.1                --tag seq_cf_hpover > "results/seq_cf_hpover_s${SEED}.log" 2>&1 &
D=$!
echo "A(대조) $A · B(사슬뺌) $B · C(고전력) $C · D(고전력+상한) $D — $(date '+%F %H:%M:%S')"
RC=0
wait $A || RC=1
wait $B || RC=1
wait $C || RC=1
wait $D || RC=1

echo "=== A: 대조 (seq 캐시 · 사슬) ===";        tail -20 "results/seq_cf_base_s${SEED}.log"
echo "=== B: 사슬 뺌 (seq 캐시 · 창별 손실만) ==="; tail -20 "results/seq_cf_nochain_s${SEED}.log"
echo "=== C: 고전력 캐시 (사슬) ===";            tail -20 "results/seq_cf_hp_s${SEED}.log"
echo "=== D: 고전력 캐시 + 상한 힌지 ===";        tail -20 "results/seq_cf_hpover_s${SEED}.log"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
