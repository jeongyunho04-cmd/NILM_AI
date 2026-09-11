#!/bin/bash
# 13.84.28 — **한 판으로 끝나는가** + 가림을 풀면 나은가 (2026-09-12)
#
# 969137 이 잰 것 둘:
#  ① 처음부터 배운 판은 게이트가 **한 번도 감독을 못 받아**(창별 0.777 대 v37판 0.913) 방출이
#     base 와 emit 의 **큰 값 상쇄**로 풀린다 — 항 규모 ±0.6 대 ±0.05 인데 잔차는 둘 다 −0.05~−0.09.
#     합성에서는 오히려 이기고(0.9901 대 0.9886) 실측에서 부서진다. 켜짐 BCE 를 일부 남기면
#     그 규모가 묶여 **v37 없이 한 판**으로 끝날 수 있다 (`--keep-on`).
#  ② v37 의 고차위상 가림은 **생성기가 h9~15 위상을 못 만들어서** 걸었던 것인데(13.84.12),
#     지금 캐시는 `sibling_rotate smps_dev1` 로 그 단서를 원천에서 없앴고(13.84.16),
#     v36(가림 없음)이 실패 ② AUC 역대 최고였다(0.66 대 v37 0.56, 13.84.20).
#     게다가 사슬 구조에서 **전이 머리는 원시를 안 가리고 본다** — 가림은 방출 경로에만 걸린다.
#     ⚠ 가림 해제는 **처음부터 배우는 판만의 선택지**다. 물려받은 가중치엔 본 적 없는 입력이다.
#
#   A: --keep-on 0.3              조금만 남겨도 규모가 잡히나
#   B: --keep-on 1.0              창 손실 그대로 + CRF
#   C: --keep-on 0.3 --no-mask    거기에 고차위상 채널까지 살리면
#
# ⚠ 대조군은 새로 안 돌린다. 캐시가 **재현되므로**(chunk_seed, 13.84.27) 같은 인수로 구운 캐시는
#   969137 것과 바이트 단위로 같다. 969137 의 seq_v37(v37 이어) · seq_scratch(scratch, keep-on 0)
#   이 그대로 대조군이다.
#SBATCH -J nilm_keep3
#SBATCH -p gpu5,gpu4,gpu3,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 39
#SBATCH --gres=gpu:3
#SBATCH --mem=220G
#SBATCH -t 03:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
MM=/usr/lpp/mmfs/bin/mmlsquota

RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_v2"   # 969137 과 **같은 인수** -> 바이트 동일

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 기록 $RECORDS · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ⚠ 공용 계정이다. 쓰고 나면 램을 반드시 돌려준다.
cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 ────────────────────────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_seqraw --help | grep -q -- "--gen"     || { echo "run_build_seqraw 에 --gen 이 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq   --help | grep -q -- "--keep-on"  || { echo "run_train_seq 에 --keep-on 이 없다 — 코드를 안 올렸다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# GPFS 쿼터 — 여기는 체크포인트·로그 10MB 만 쓴다. 하드 한도까지 1GB 는 남아 있어야 한다.
# ⚠ `quota -s` 는 gpfs 에 아무것도 안 내놓는다. 자리는 **mmlsquota** 로만 보인다 (13.84.27).
# ⚠⚠ 그런데 **이 관문으로 작업을 죽이지 마라** — 969002 는 n055 에서 "GPFS is down on this node"
#    (mm 클라이언트 데몬 무응답)로 6초 만에 죽었다. 파일 읽기는 멀쩡했다. 우리가 쓰는 건 10MB 다.
if Q=$($MM --block-size 1M 2>/dev/null | awk '$1=="gpfs" && $2=="USR" {print $3+0, $5+0}') && [ -n "$Q" ]; then
  read -r USED LIM <<< "$Q"
  echo "[쿼터] 홈 ${USED}MB / 소프트 100G / 하드 ${LIM}MB — 하드까지 $((LIM-USED))MB"
  [ $((LIM - USED)) -gt 1024 ] || { echo "하드 한도까지 1GB 미만이다 — 정리부터 하라"; exit 1; }
else
  echo "[쿼터] ⚠ 이 노드에서 mmlsquota 를 못 읽는다 — 10MB 만 쓰므로 그냥 간다"
fi

# 램 자리: 기록당 3.269MB (원시 45x18000x4 = 3.240 + 라벨 0.029).
# ⚠ SLURM cgroup 이 /dev/shm 을 `--mem` 에 잡는다. 실측 상한으로 잡은 예산 (13.84.27):
#     캐시 98GB + 워커 44개 53GB(상한, 공유 페이지 중복 포함) + 주 프로세스 4.5GB = 156GB -> 200G 요청.
#   ⚠ 과하게 요청하면 여유 400GB 짜리 노드에서 **남의 backfill 을 막는다**. 공용 계정이다.
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다 (${FREE}GB < ${NEED}GB x1.3)"; exit 1; }

# ── ① 캐시를 램에 굽는다 ────────────────────────────────────────────────────
mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen v36 --workers 30 --seed 0

$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != 'v36': sys.exit('캐시가 v36 생성기가 아니다: %r' % m.get('gen'))
o = m.get('gen_opts', {})
for k in ('sibling_rotate','sp_curves','sp_per_texture','vtail','float_fill','steady_crop','state_mix','power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('생성기 설정 %s 가 빠졌다' % k)
print('캐시 관문 통과 · 기록 %d · 단계 %d · 기기 %d' % (m['records'], m['steps'], len(m['appliances'])))
"
du -sh "$SHM"; df -h /dev/shm | tail -1
cp "$SHM/meta.json" "results/seqraw_v2_meta.json"

# ── ② 두 판을 동시에 ────────────────────────────────────────────────────────
# 같은 캐시·같은 씨앗·같은 예산. 바뀌는 것은 **초기점 하나**다.
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers 13 --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON --lr 3e-4 --keep-on 0.3     --init scratch --ref results/cnn_v37.pt --tag seq_keep03 > "results/seq_keep03_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --lr 3e-4 --keep-on 1.0     --init scratch --ref results/cnn_v37.pt --tag seq_keep10 > "results/seq_keep10_s${SEED}.log" 2>&1 &
B=$!
CUDA_VISIBLE_DEVICES=2 $PY -X utf8 -m src.run_train_seq $COMMON --lr 3e-4 --keep-on 0.3 --no-mask     --init scratch --ref results/cnn_v37.pt --tag seq_keep03nm > "results/seq_keep03nm_s${SEED}.log" 2>&1 &
C=$!
echo "A(keep03) $A · B(keep10) $B · C(keep03+가림해제) $C — $(date '+%F %H:%M:%S')"
RC=0
wait $A || RC=1
wait $B || RC=1
wait $C || RC=1

echo "=== A: keep-on 0.3 ==="; tail -20 "results/seq_keep03_s${SEED}.log"
echo "=== B: keep-on 1.0 ==="; tail -20 "results/seq_keep10_s${SEED}.log"
echo "=== C: keep-on 0.3 + 가림해제 ==="; tail -20 "results/seq_keep03nm_s${SEED}.log"
# ⚠ 여기서도 죽이지 마라 — 969137 이 끝의 이 한 줄 때문에 FAILED 로 찍혔다 (결과는 멀쩡했다).
$MM --block-size auto 2>/dev/null | tail -2 || echo "[쿼터] 이 노드에서 못 읽는다"
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
