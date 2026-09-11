#!/bin/bash
# 13.84.27 — 처음부터 대 v37 이어학습, 짝지어 한 노드에서 (2026-09-12)
#
#   ① 시퀀스 원시 캐시를 **--gen v36** 으로 **/dev/shm(램)** 에 굽는다
#      ⚠ 램이어야 한다 — 홈 쿼터가 **100G(하드 110G)이고 지금 101.6G** 로 이미 넘었다.
#        94GB 캐시를 GPFS 에 둘 자리가 없다. 램은 노드 것이라 쿼터와 무관하다.
#      ⚠ seqraw_v1 은 생성기 설정 열이 빠진 판이었다 (13.84.27). 그걸로 처음부터 배우면
#        "이미 나쁘다고 아는 생성기로 몸통이 배울 수 있는가" 를 재게 된다.
#   ② 같은 캐시·같은 씨앗·같은 예산으로 두 판을 **같은 노드에서 동시에** 돌린다
#        A: --init results/cnn_v37.pt --lr 1e-4   (이어학습 — 좋은 초기점을 3e-4 로 흔들면 망친다)
#        B: --init scratch            --lr 3e-4   (무작위 — v37 이 처음부터 배울 때 쓴 값)
#      둘 다 --vswap-p 0.5 — v37 이 그걸로 배웠는데 시퀀스 학습기가 빠뜨리고 있었다 (13.84.11).
#      ⚠ **몸통 학습률은 일부러 다르게 준다.** 맞춰야 할 것은 날 학습률이 아니라 *그 초기점에 맞는*
#        학습률이다. 무작위 몸통에 미세조정용 1e-4 를 물리면 총 학습량이 v37 의 62% 밖에 안 되어
#        (54,000스텝 x1e-4 대 v37 29,100스텝 x3e-4) **거짓 음성**이 난다. 머리는 양쪽 3e-4 로 같다.
#        그래서 이 판은 "초기점 하나만 다른" 엄밀한 대조가 아니라 **각자 제 학습률에서의 대결**이다.
#   ③ **실측은 여기 없다** (--no-real). 실측은 원래 채점 전용이고 공용 계정에 두지 않는다.
#      여기서는 합성 홀드아웃으로 곡선을 보고, 체크포인트(2.5MB)를 받아 로컬에서 실측을 채점한다.
#
# 예산: 기록 30,000 · 학습 27,000 -> epoch 당 27,000x64 = 1.73M 창.
#       v37 은 300ep x 5만 = 15M 창을 봤다. 12 epoch = 20.7M 이라 양쪽에 넉넉하다.
#
# ⚠ 자리 잡기 (13.84.27 실측): 이 클러스터는 **FIFO** 다 (priority/basic, sprio 출력 없음,
#   fairshare 0). 앞에 120건이 있어 순서로는 못 앞지르고 **backfill 만이 길**이다. 그래서
#   ① 시간을 짧게(3h) ② 코어를 적게(32 — 후보 노드 8개 -> 20개) ③ 파티션을 넓게(A10 포함).
#   실제로 이렇게 바꾸자 1분 안에 잡혔다. GPU 는 원래 남아돈다 (전체 380장 중 110장 유휴).
#   ⚠ 시간 초과로 죽어도 **epoch 마다 저장**하므로 거기까지는 건진다.
#SBATCH -J nilm_seq
#SBATCH -p gpu5,gpu4,gpu3,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
MM=/usr/lpp/mmfs/bin/mmlsquota

RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_v2"

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 기록 $RECORDS · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ⚠ 공용 계정이다. 쓰고 나면 램을 반드시 돌려준다.
cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 ────────────────────────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_seqraw --help | grep -q -- "--gen"     || { echo "run_build_seqraw 에 --gen 이 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq   --help | grep -q -- "--no-real"  || { echo "run_train_seq 에 --no-real 이 없다 — 코드를 안 올렸다"; exit 1; }
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
        --workers 14 --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON --lr 1e-4 \
    --init results/cnn_v37.pt --tag seq_v37 > "results/seq_v37_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --lr 3e-4 \
    --init scratch --ref results/cnn_v37.pt --tag seq_scratch > "results/seq_scratch_s${SEED}.log" 2>&1 &
B=$!

echo "A(v37 이어) pid $A · B(처음부터) pid $B — $(date '+%F %H:%M:%S')"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: v37 이어학습 ==="; tail -20 "results/seq_v37_s${SEED}.log"
echo "=== B: 처음부터 ===";    tail -20 "results/seq_scratch_s${SEED}.log"
$MM --block-size auto | tail -2
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
