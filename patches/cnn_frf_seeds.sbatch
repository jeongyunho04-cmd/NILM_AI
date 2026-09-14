#!/bin/bash
# 14.78/14.79 — **세밀 갈래가 미래 계단의 위치를 알게 한다.**
#
# 사용자: *"세밀창의 수용영역을 10초로 완전히 늘려버리면 되는 거 아니야?"*
#         *"mean/amax를 유지할 이유가 있어?"*
#
# 겨냥하는 병 (14.75, test_5 78~82초):
#   관측이 **완전히 평평한데**(2242 -> 2247 -> 2240 -> 2247, 폭 7W) 모델이 오븐의
#   1072W 를 포트로 통째로 옮겼다가 되돌린다. 포트의 진짜 on 은 82.47초 — 세밀 창
#   [t-3.98, t+6.00] 이 t=78.5 에서 이미 그 계단을 품는다. **곧 올 포트의 요구를
#   지금 있는 오븐 와트를 다시 라벨링해서 만족시킨다.** 총합이 맞으니 L_power 는 조용하다.
#   그게 공짜인 까닭은 지문 축퇴다 (오븐<->포트 코사인 0.99944, 전력 1085 대 1100W).
#   => **타이밍이 방아쇠, 축퇴가 총알.**
#
# 무엇을 바꾸나:
#   A  --fine-dilations 1,3,9,27,81
#      RF 187 -> **727 사이클 (+-6.06초)**. 블록 수가 같아 **파라미터가 안 는다**(624,917).
#      `run_gate_finerf` 실측 — 미래 +2초 계단과 +4초 계단의 **구별율**:
#         기본 **17.4%**  ->  1,3,9,27,81 **68.3%**  (3.9배)
#      ⚠ 기본도 +4초에 "반응" 은 한다 (GroupNorm 이 시간축 전체로 정규화). 다만 그 응답은
#        **오프셋에 평평하다**(변동계수 0.052) — 위치 불변 스칼라 둘이라 '언제' 를 못 담는다.
#   B  A + --fine-pool amax
#      위치를 모르는 `h.mean` 을 뺀다. 수용영역이 창을 덮으면 conv 가 어떤 가중평균이든
#      만들 수 있어 중복이고, 남기면 합성에서 잘 듣는 지름길만 준다
#      (14.47: 미래를 다 지우면 mix 0.939 인데 **수용영역 밖만** 지워도 0.941).
#      `h.amax` 는 max 라 conv 가 표현 못 하고 듀티 기기에 필요해서 **남긴다**(11.1).
#      원시 `fp.amax/amin` 과 물리 프라이어도 그대로다 (12.9.8).
#      trunk 입력 637 -> 509 (파라미터 −32,768).
#
# 기준선: 오늘 돌린 `cnn_v32h_off_s{0,1,2}` (옛 `cnn_vanch_on_*` 과 숫자가 같다).
# 캐시 재굽기 **없음**.
#
# ★ 미리 적는 판정 자 (오늘 겨냥을 네 번 헛짚었다)
#   1 ★ 축퇴 — 포트 참ON·오븐 OFF 창(61개)의 **오븐 헛detect**
#       기준 **42.1%** · hz 단독 30.6% · vhrel f=.5 33.3% · wide-target 41.0%
#   2 ★ test_5 78~82초 — 관측이 평평한데 오븐<->포트 와트가 오가는가 (눈으로 본다)
#   3 전환 +-20초 띠 오븐 헛detect  기준 17.8 +- 5.8 % (짝차 −2.5%p 문턱)
#   4 지킴: 판정 줄 0.9133 · 핫플통전 P>1500 −16.3+-7.5W · 저항 4종 신원 0.9815
#
#SBATCH -J nilm_frf
# ⚠⚠ **Slurm 은 `-p` 목록의 순서를 안 본다.** 2026-09-14 에 이걸 틀리게 적었다가
#   `gpu3,gpu4,gpu5,gpu1,gpu6` 으로 던졌더니 제일 느린 **gpu6(A10)** 으로 갔다.
#   까닭: 파티션 `PriorityTier` 가 다섯 다 **1 로 같고**, 같으면 Slurm 은
#   *"가장 빨리 시작되는 곳"* 을 고른다 (이름 순서 무시).
#   그리고 이 클러스터는 **노드 특성(feature)이 아예 없어서** `--prefer` 도 못 쓴다
#   (`Invalid preferred feature specification`, slurm 23.11.4).
#   ⇒ *"빠른 것부터, 없으면 그 아래"* 를 Slurm 으로 표현할 수단이 없다.
#      **느린 파티션을 목록에서 빼는 것**이 유일한 수단이다.
#
#   파티션별 GPU (`sinfo -O Gres,GresUsed`):
#     gpu3 **a6000ada** x4/노드 (10노드)  |  gpu4·gpu5 a6000  |  gpu1 rtx3090  |  gpu2·gpu6 a10
#   14.47 ④ 실측: A6000 **13.6 ep/분** · A10 **9.3 ep/분** (1.46배). Ada 는 그보다 빠르다.
#
#   ⚠ 자리가 없으면 **대기한다.** 그게 느린 데서 도는 것보다 나은지는 그때 보고 판단하라 —
#     A10 34분 대 A6000 23분이니, 11분 넘게 대기할 것 같으면 gpu1 을 더해라.
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 984191 이 3090 에서 27~34분이었다.
#SBATCH -t 00:55:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 src/run_gate_finerf.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠⚠ 인자를 한 곳에 모아 **학습 전에** 되짚는다 (14.69 가 이 자리에서 어긋났다).
#   파일 전체가 아니라 **이 명령줄 자체**를 본다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
A_FLAGS="--fine-dilations 1,3,9,27,81"
B_FLAGS="--fine-dilations 1,3,9,27,81 --fine-pool amax"
echo "  BASE: $BASE"
echo "  갈래 A: $A_FLAGS"
echo "  갈래 B: $B_FLAGS"
case "$A_FLAGS" in *"--fine-dilations 1,3,9,27,81"*) ;; *) echo "A 에 dilation 이 없다"; exit 1;; esac
case "$B_FLAGS" in *"--fine-pool amax"*) ;; *) echo "B 에 fine-pool 이 없다"; exit 1;; esac
case "$BASE" in *--fine-dilations*|*--fine-pool*) echo "BASE 에 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_frf_s${SEED} > "results/cnn_frf_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_frf_am_s${SEED} > "results/cnn_frf_am_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: dilation 1,3,9,27,81 ==="; tail -32 "results/cnn_frf_s${SEED}.log"
echo "===================================================================================="
echo "=== B: + --fine-pool amax ==="; tail -32 "results/cnn_frf_am_s${SEED}.log"
ls -la results/cnn_frf*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
