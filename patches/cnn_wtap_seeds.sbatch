#!/bin/bash
# 14.80 — **넓은 탭을 더한다** (14.78 은 대체해서 실패했다).
#
# 14.78/14.79 판정 (984279, 3시드) — 둘 다 기각:
#   판정 줄 0.9133 -> A 0.8871 · B 0.8911  |  저항 4종 신원 한 시드 **0.9321**
#   SMPS 3종 0.9518 -> A 0.9219  |  절대 잔차 20.2 -> 21.7 / 22.5W
#   그리고 test_5 78~82초의 오븐<->포트 교환이 **더 커졌다** (78.5초 시작 -> 74.5초 시작).
#
# 까닭 (`run_gate_finerf`, 무작위 가중치):
#   깊은 타깃 슬라이스 반응   +0        +90
#     기본 (1,2,4,8,16)   **2.393**   0.448    <- 타깃에서 5.34배 뾰족
#     교체 (1,3,9,27,81)    1.494   **1.959**  <- **어깨가 꼭대기보다 높다**
#   도달거리와 국소성을 맞바꿨고 **국소성이 더 비쌌다.** 신원은 순시 고조파 벡터가
#   있어야 풀리는데 그게 12초짜리 평균으로 뭉개졌다.
#
# 고침: 앞 다섯 블록을 **그대로 두고** 뒤에 (32,64) 를 붙이고 `tap_layers` 에 **4** 를 넣어
#   옛 마지막 블록(RF 187)의 타깃 슬라이스도 같이 뽑는다.
#     탭 4 (국소) +0 **2.393** · +90 0.448 · +240 0.391   <- 지금 판과 **비트 동일**
#     탭 6 (넓음) +0 1.426 · +90 1.430 · **+240 1.030**   <- 미래 4초를 본다
#   파라미터 624,917 -> 887,829 (+42%). 끄면 비트 동일.
#
# 겨냥 (14.75, test_5 78~82초): 관측이 **평평한데**(폭 7W) 오븐의 1072W 가 포트로
#   통째로 옮겨갔다 되돌아온다. 포트의 진짜 on 은 82.47초 — 창이 이미 그 계단을 품는다.
#   타이밍이 방아쇠, 축퇴(오븐<->포트 코사인 0.99944)가 총알.
#
# ★ 미리 적는 판정 자
#   1 ★ 축퇴 — 포트 참ON·오븐 OFF 창(61개)의 오븐 헛detect. 기준 **42.1%**
#       (hz 30.6 · vhrel.5 33.3 · **14.78 A 36.1±13.9 못 읽음** · wtgt 41.0)
#   2 ★ test_5 78~82초 교환 — 눈으로 본다 (기준: 78.5초부터 2창, 오븐 1072->219W)
#   3 전환 ±20초 띠 오븐 헛detect 17.8 ± 5.8 % (짝차 −2.5%p 문턱)
#   4 ⚠⚠ **지킴 — 14.78 이 무너뜨린 자리를 반드시 본다**:
#        저항 4종 신원 0.9815 (한 시드도 0.97 아래로 가면 안 된다)
#        SMPS 3종 0.9518 · 판정 줄 0.9133 · 절대 잔차 20.2W · 핫플통전 −16.3±7.5W
#
#SBATCH -J nilm_wtap
# ⚠⚠ SLURM 은 `-p` 순서를 안 본다 (PriorityTier 가 다 1). 느린 칸을 **빼는 것**이
#   유일한 수단이다. gpu3 a6000ada · gpu4·gpu5 a6000 (A10 대비 1.46배).
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# ⚠ 파라미터가 42% 늘어 조금 느려진다. 984279 가 A6000 에서 22~24분이었다.
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
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_finerf.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
A_FLAGS="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
B_FLAGS="--fine-extra-dilations 32,64 --tap-layers 0,1,4 --wide-target"
echo "  BASE: $BASE"
echo "  갈래 A: $A_FLAGS"
echo "  갈래 B: $B_FLAGS"
# ⚠ 14.78 의 실수를 막는다 — 탭에 4 가 없으면 국소 탭이 사라진다.
case "$A_FLAGS" in *"--tap-layers 0,1,4"*) ;; *) echo "A 의 탭에 4 가 없다"; exit 1;; esac
case "$B_FLAGS" in *"--tap-layers 0,1,4"*) ;; *) echo "B 의 탭에 4 가 없다"; exit 1;; esac
case "$A_FLAGS" in *--fine-dilations*) echo "A 에 dilation **교체** 가 섞였다 (14.78 실패)"; exit 1;; esac
case "$BASE" in *--fine-*|*--tap-layers*|*--wide-target*) echo "BASE 에 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_wtap_s${SEED} > "results/cnn_wtap_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_wtap_wt_s${SEED} > "results/cnn_wtap_wt_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: 넓은 탭 추가 ==="; tail -32 "results/cnn_wtap_s${SEED}.log"
echo "===================================================================================="
echo "=== B: + --wide-target ==="; tail -32 "results/cnn_wtap_wt_s${SEED}.log"
ls -la results/cnn_wtap*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
