#!/bin/bash
# 13.84.40 — 녹화별 대기 지문을 기록당 하나씩. `--bg-head` 에게 **배울 것을 준다**. 짝지은 2판.
#
# 13.84.39 가 잰 것: `--bg-head` 는 슬랙이 아니라 거의 **죽은 항**이었다 (파일당 0.04~2.45mA,
# 상한 50mA 의 0~5%). 합성이 준 기록당 배경이 계측계 잡음 참조 3개의 산포 1.26~1.70mA 뿐이라
# 배울 것이 없었다. 머리는 제 훈련 분포를 맞게 배운 것이다.
#
# 13.84.40 이 그 양을 갈랐다. 실측 전부-OFF 배경의 **파일 간 편차**를 분해하면:
#     h1 은 관측 19.20 을 꽂힌 기기 대기가 19.61 로 이미 다 설명한다 -> bg-head 가 할 일 없음
#     h9~h15 는 관측 8.35/5.64/4.49/7.48 중 모형 몫이 4.06/3.32/3.32/3.16,
#     **남는 몫이 7.00/5.42/4.71/6.27** — 모형 몫보다 크거나 같다. 여기가 진짜 구멍이고
#     하필 미니PC 를 가르는 차수다 (13.84.32 의 h11·h13).
#
# 처방: 대기 지문은 단독 녹화마다 뽑히는데 `_merge_standby_profiles` 가 기기당 하나로 평균해
# 버린다. 그 평균을 없애고 **기록당 하나를 뽑는다** (`--gen v36r`). 손실의 `standby_sig` 는
# 기기당 상수 그대로라 그 산포가 곧 미모형 세션 배경이 된다.
# 관문(`src.run_gate_sbpr`)이 잰 실제 크기: h9~h15 에서 4.95/3.53/3.50/4.12mA
# = v36 의 2.3~3.3배 · 목표의 65~74%. **전부 측정값이고 지어낸 값이 없다.**
#
#   A: 대조   v36r 캐시 · bg-head 없음
#   B: bg     v36r 캐시 · --bg-head (w_bg 3.0)
#
# ⚠ 판정은 **로컬에서** 한다 (`run_score_seq.py` · `run_diag_bghead.py`). 여기서는 `--no-real`.
#   13.84.38 ④ 에서 합성 홀드아웃 창별 순서가 실측과 **부호가 반대**였다.
# ⚠ 13.59 함정 — 총점만 보지 말고 파일별로 짝지어 읽는다. 13.84.38 의 bg 가 총점은 올리고
#   test_5 는 1.000 -> 0.942 였다.
#SBATCH -J nilm_sbpr
#SBATCH -p gpu5
#SBATCH -N 1
#SBATCH -n 1
# ⚠ 코어 54 = 판당 워커 26 x 2 + 주 프로세스 2.
#   §5 실측이 2판x워커14(-c 32)에서 GPU 54~55% · load 31/32 로 **CPU 포화**였다.
#   54~55% 를 채우려면 판당 워커가 약 1.85배 필요하다 -> 26. 그 값이 QOS 상한을 넘지 않는
#   파티션은 gpu5(60) 뿐이라 후보가 **6노드**로 준다 (§2 의 backfill 대가).
#   ⚠ 안 잡히면 취소하지 말고 고친다 (§2): scontrol update jobid=<id> NumCPUs=38 MinCPUsNode=38
#      Partition=gpu4,gpu6,gpu1,gpu2,gpu3,gpu5 — 38 이면 여섯 파티션 95노드가 다 열린다.
#      워커 수는 아래에서 nproc 으로 계산하므로 코어만 바꿔도 따라온다.
#SBATCH -c 54
#SBATCH --gres=gpu:2
# 캐시 98GB + 워커 52개 + 주 프로세스 2 = 약 170GB. 노드 램 1TB 의 24% 다 (§4).
#SBATCH --mem=240G
# 실측 근거: 969137 이 2판x워커14 로 12 epoch **1:57:27**. 워커 26 이면 낙관 1:06 · 비관 1:30.
# + 캐시 굽기(970310 은 워커 30 에 9.8분, 여기선 50) => 1:15~1:40. 2:30 은 1.5~2배다 (§2).
#SBATCH -t 02:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_v3"

# 워커는 **실제 할당분**에서 계산한다 — scontrol 로 코어를 바꿔도 그대로 따라온다.
NCPU=$(nproc)
W=$(( (NCPU - 2) / 2 ))
[ "$W" -ge 4 ] || { echo "코어가 너무 적다 (nproc=$NCPU)"; exit 1; }
BW=$(( NCPU - 4 ))

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · nproc $NCPU · 판당 워커 $W · 굽기 워커 $BW"
echo "기록 $RECORDS · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ⚠ 공용 계정이다. 쓰고 나면 램을 반드시 돌려준다 (§7-7).
cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에 둔다, §7-3) ──────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_seqraw --help | grep -q -- "--gen"       || { echo "run_build_seqraw 에 --gen 이 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq    --help | grep -q -- "--bg-head"   || { echo "run_train_seq 에 --bg-head 가 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq    --help | grep -q -- "--keep-last" || { echo "run_train_seq 에 --keep-last 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }
$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src.synthesis.genopts import PRESETS
o = PRESETS.get('v36r')
if not o: sys.exit('genopts 에 v36r 이 없다 — 코드를 안 올렸다')
if not o.get('standby_per_record'): sys.exit('v36r 에 standby_per_record 가 없다')
for k in ('sibling_rotate','sp_curves','sp_per_texture','vtail','float_fill','steady_crop',
          'state_mix','power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('v36r 이 V36 의 %s 를 잃었다' % k)
print('관문: v36r = V36 + standby_per_record')
"

# ── ① 캐시를 램에 굽는다 ────────────────────────────────────────────────────
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다 (${FREE}GB < ${NEED}GB x1.3)"; exit 1; }

mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen v36r --workers "$BW" --seed 0

# 캐시가 **실제로** v36r 로 구워졌는지 되짚는다 ([[verify-the-gate-runs-that-path]]).
$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != 'v36r': sys.exit('캐시가 v36r 이 아니다: %r' % m.get('gen'))
o = m.get('gen_opts', {})
if not o.get('standby_per_record'): sys.exit('캐시 메타에 standby_per_record 가 없다')
for k in ('sibling_rotate','sp_curves','sp_per_texture','vtail','float_fill','steady_crop',
          'state_mix','power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('생성기 설정 %s 가 빠졌다' % k)
print('캐시 관문 통과 · 기록 %d · 단계 %d · 기기 %d' % (m['records'], m['steps'], len(m['appliances'])))
"
du -sh "$SHM"; df -h /dev/shm | tail -1
cp "$SHM/meta.json" "results/seqraw_v3_meta.json"

# ── ② 두 판을 동시에 ────────────────────────────────────────────────────────
# 같은 캐시·씨앗·초기점·예산. 바뀌는 것은 **bg-head 하나**뿐이다.
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers $W --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init results/cnn_v37.pt --ref results/cnn_v37.pt --lr 1e-4 \
        --keep-last 4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON             --tag seq_sbpr_base > "results/seq_sbpr_base_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --bg-head   --tag seq_sbpr_bg   > "results/seq_sbpr_bg_s${SEED}.log"   2>&1 &
B=$!
echo "A(대조) $A · B(bg) $B — $(date '+%F %H:%M:%S')"

# 코어 54 가 값을 했는지 **재서** 남긴다 (§5). 969137 은 워커 14 에 GPU 54~55% · load 31/32 였다.
( sleep 420
  for i in 1 2 3; do
    echo "[점유 $i] $(date '+%H:%M:%S') $(uptime | sed 's/.*average/load/')"
    nvidia-smi | grep Default || true
    sleep 900
  done ) &
MON=$!

RC=0
wait $A || RC=1
wait $B || RC=1
kill $MON 2>/dev/null || true

echo "=== A: 대조 (v36r 캐시, bg-head 없음) ==="; tail -20 "results/seq_sbpr_base_s${SEED}.log"
echo "=== B: --bg-head ===";                     tail -20 "results/seq_sbpr_bg_s${SEED}.log"
ls -la results/seq_sbpr_*_ep*.pt 2>/dev/null || echo "(ep 체크포인트가 없다 — --keep-last 확인)"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5). 969002·969137·969590 이 그 한 줄 때문이었다.
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
