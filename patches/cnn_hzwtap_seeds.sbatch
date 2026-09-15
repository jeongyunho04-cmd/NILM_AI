#!/bin/bash
# 14.95 — **hz 캐시 위에서 wtap**. 두 처치가 더해지나. 시드 셋 짝비교.
#
# 왜 이 판인가: 지금까지 모든 갈래가 hz 를 **끈** 캐시(train60_v32)로 돌았다.
#   그런데 hz 는 이 세션 초에 ★2 축퇴에서 확인된 승리였고(−11.5 +- 5.7 %p),
#   hz 캐시(train60_v32h)와 홀드아웃이 **이미 구워져 있다**. 조합을 한 번도 안 봤다.
#
# ★ 던지기 전에 **잰** 기준값 (3시드, 로컬 체크포인트)
#     판                    ★1 면적(test_5 128~142초)   ★2 축퇴 게이트
#     v32 off (hz 끔)          3.814 +- 0.402            42.1 +- 5.6 %
#     **v32h on (hz 단독)**    **4.454 +- 1.005**        **30.6 +- 1.5 %**   <- 이 판의 대조
#     wtap A (v32 위)          1.529 +- 1.257            22.4 +- 2.0 %
#
#   ⇒ **둘이 고치는 것이 다르다.** hz 는 ★2 를 크게 고치는데 ★1 은 오히려 **나빠진다**
#     (3.814 -> 4.454). wtap 은 ★1 을 고친다 (−2.285 +- 0.896, 3/3).
#     hz 는 **지문 쪽**(자리 전압 왜곡, 14.84)을, wtap 은 **증거 쪽**(창 전체의 평평함,
#     14.90)을 고친다. 기전이 달라 더해질 여지가 있는데, 확인이 안 돼 있다.
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① 가산적이면 ★1 이 4.454 − 2.285 = **약 2.2** 로 내려간다. 짝차가 **3/3 음수**여야 한다.
#   ② ★2 는 30.6% 에서 더 내려가야 한다 — **25% 밑**, 3/3.
#   ③ 둘 다 안 움직이면 두 처치가 **같은 자리를 다투는 것**이고, 그러면 hz 를 켠 뒤
#      wtap 의 여지가 이미 없다는 뜻이다.
#
#   A  --fine-extra-dilations 32,64 --tap-layers 0,1,4                (= wtap A)
#   B  A + --wide-target                                              (= wtap B)
#   대조는 **이미 로컬에 있는** cnn_v32h_on_s{0,1,2} — BASE 가 글자 그대로 같다.
#
# ⚠⚠ 지킴: 저항 4종 신원 0.9815 (한 시드도 0.97 밑이면 안 된다)
#      SMPS 3종 · 판정 줄 · 절대 잔차는 **v32h_on 과 짝으로** 뺀다 (v32 기준선이 아니다)
#
#SBATCH -J nilm_hzwt
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
CACHE=${CACHE:-cache/train60_v32h}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_wsegpool.py
$PY -X utf8 src/run_gate_finerf.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--fine-extra-dilations" \
  || { echo "--wide-seg-pool 이 없다"; exit 1; }

# ⚠ BASE 는 cnn_v32h_seeds.sbatch 와 **글자 그대로 같아야** 한다 — 대조를 새로 안 굽고
#   로컬의 cnn_v32h_off_s{0,1,2} 를 그대로 짝으로 쓰기 때문이다.
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
# 처치가 BASE 로 새면 대조가 무너진다 (cnn_v32h_off 와 BASE 가 달라진다)
case "$BASE" in *--fine-*|*--wide-target*|*--tap-layers*)
  echo "BASE 에 처치가 섞였다"; exit 1;; esac
# 세밀 쪽은 한 글자도 안 건드린다 — 14.78 의 실패가 거기였다
case "$A_FLAGS" in *"--tap-layers 0,1,4"*) ;; *) echo "A 의 탭에 4 가 없다 (14.78)"; exit 1;; esac
case "$B_FLAGS" in *"--tap-layers 0,1,4"*) ;; *) echo "B 의 탭에 4 가 없다 (14.78)"; exit 1;; esac
case "$A_FLAGS" in *--fine-dilations*) echo "dilation **교체** 가 섞였다 (14.78 실패)"; exit 1;; esac
case "$CACHE" in *v32h*) ;; *) echo "hz 캐시가 아니다: $CACHE"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_hzwt_s${SEED} > "results/cnn_hzwt_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_hzwt_wt_s${SEED} > "results/cnn_hzwt_wt_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: hz + 넓은 탭 ==="; tail -32 "results/cnn_hzwt_s${SEED}.log"
echo "===================================================================================="
echo "=== B: + --wide-target ==="; tail -32 "results/cnn_hzwt_wt_s${SEED}.log"
ls -la results/cnn_wsp*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
