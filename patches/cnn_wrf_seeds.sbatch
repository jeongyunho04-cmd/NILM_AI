#!/bin/bash
# 14.91 — **광역 몸통의 수용영역을 창 전체로** (`--wide-extra-dilations`). 시드 셋 짝비교.
#
# 사용자: *"광역갈래를 통해 할 방법을 더 찾아보자, 한 번 실패했다고 버리는 건 아닌 것 같아"*
#         *"이전에 광역갈래랑 세밀갈래를 통합하는 곳에 약간 문제가 있었던 것 같은데"*
#   -> 맞다. **14.42 (1)** 이 그 자리를 이미 짚어 뒀다.
#
# 14.42 (1) 이 쟀던 것 (오븐 혼합, 짝수차를 구간별로 덮기):
#     타깃 이후 전부 덮기  0.939   |  수용영역 **안**만  0.126  (효과 없음)
#     수용영역 **밖**만  **0.941** |  **광역만 덮기  0.203**
#   => 수용영역 밖 증거가 머리에 닿는 길은 **전역 평균/최대뿐**이고 과거와 미래를 못 가른다.
#
# 처방이 둘이었는데 하나만 다 해봤다:
#   (a) 전역 풀링을 쪼갠다  --seg-pool(14.46, 2 에서 못 읽음) / --wide-seg-pool(14.88, 6 기각)
#   (b) 수용영역을 창 전체로 넓혀 **위치를 아는 탭**으로 받는다
#        세밀에는 했고(--fine-extra-dilations, 14.80) **그것만 들었다** (14.90)
#        광역에는 **한 번도 안 했다**   <- 이 판
#
# 관문(run_gate_wextra.py)이 **재서** 확인한 것 - 산술이 아니라 경험적 수용영역:
#     지금 (1,2,4)   타깃에서 **14블록 = +-7.0초**  (창은 +-30초 -> **13%**)
#     + 8            24블록 = 12.0초
#     + 8,16       **54블록 = 27.0초**
#   파라미터 624,917 -> 666,261 (**+6.6%**) / trunk_in 은 안 변한다 / 비우면 비트 동일
#
#   A  --wide-extra-dilations 8,16                  (수용영역만. 머리는 여전히 mean)
#   B  --wide-extra-dilations 8,16 --wide-target    (거기에 **타깃 자리 탭**)
#   대조는 **이미 로컬에 있는** cnn_v32h_off_s{0,1,2} - BASE 가 글자 그대로 같다.
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   14.90 이 잰 기전은 "타깃 자리에서 창 전체를 읽는 탭" 이다. 그러면
#   **B 가 A 보다 훨씬 나아야 한다** - A 는 넓게 보지만 평균으로만 흘리므로 위치가 죽는다.
#   A 와 B 가 같으면 그 기전 설명이 틀린 것이다.
#
# ★ 미리 적는 판정 자 (14.89 와 **같은 자**를 쓴다)
#   1 ★★ test_5 128~142초 오븐 통전확률의 **면적** (참값 0)
#         기준선 **3.814 +- 0.402** / wtap A 1.529 / wtap B 0.915 / **wsp 는 못 읽었다**
#         -> 짝차가 **3/3 음수**여야 하고 산포보다 커야 한다
#   2 ★ 축퇴 - 포트 참ON/오븐 OFF 61창의 오븐 헛detect
#         기준 게이트 42.1 +- 5.6% / >100W 31.7 +- 0.8%   (wtap A 22.4 / 19.1)
#   3 run_diag_fptrig - (전이 없음) 칸이 **0.0%** 를 지켜야 한다
#   4 ⚠⚠ 지킴: 저항 4종 신원 0.9815 (한 시드도 0.97 밑이면 안 된다)
#         SMPS 3종 0.9518 / 판정 줄 0.9133 / 절대 잔차 20.2W
#      ⚠ **미니PC 를 꼭 본다** - 광역 제거 실험에서 미니PC 만 게이트의 9.6~13.0% 를
#        광역이 정했다. 광역을 건드리는 판이므로 거기가 제일 먼저 흔들린다.
#
#SBATCH -J nilm_wrf
# ⚠⚠ SLURM 은 `-p` 순서를 안 본다 (PriorityTier 가 다 1). 느린 칸을 **빼는 것**이
#   유일한 수단이다. gpu3 a6000ada · gpu4·gpu5 a6000 (A10 대비 1.46배).
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# 파라미터 증가는 작다 ((64+2)x5 = 330 차원). 984295 가 A6000 에서 25분이었다.
#SBATCH -t 01:00:00
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
$PY -X utf8 src/run_gate_wextra.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--wide-extra-dilations" \
  || { echo "--wide-seg-pool 이 없다"; exit 1; }

# ⚠ BASE 는 cnn_v32h_seeds.sbatch 와 **글자 그대로 같아야** 한다 — 대조를 새로 안 굽고
#   로컬의 cnn_v32h_off_s{0,1,2} 를 그대로 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
A_FLAGS="--wide-extra-dilations 8,16"
B_FLAGS="--wide-extra-dilations 8,16 --wide-target"
echo "  BASE: $BASE"
echo "  갈래 A: $A_FLAGS"
echo "  갈래 B: $B_FLAGS"
# 처치가 BASE 로 새면 대조가 무너진다 (cnn_v32h_off 와 BASE 가 달라진다)
case "$BASE" in *--wide-extra*|*--wide-target*|*--seg-pool*)
  echo "BASE 에 처치가 섞였다"; exit 1;; esac
# 세밀 쪽은 한 글자도 안 건드린다 — 14.78 의 실패가 거기였다
case "$A_FLAGS$B_FLAGS" in *--fine-*|*--tap-layers*|*--seg-pool\ *)
  echo "세밀 쪽 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_wrf_s${SEED} > "results/cnn_wrf_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_wrf_wt_s${SEED} > "results/cnn_wrf_wt_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: 광역 수용영역 8,16 ==="; tail -32 "results/cnn_wrf_s${SEED}.log"
echo "===================================================================================="
echo "=== B: + --wide-target ==="; tail -32 "results/cnn_wrf_wt_s${SEED}.log"
ls -la results/cnn_wsp*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
