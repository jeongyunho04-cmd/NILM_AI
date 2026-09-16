#!/bin/bash
# 14.246 — **짝수차 모양 흔들기** `--even-jitter`. 사용자 숏컷 가설의 직접 처방.
#
# 까닭 (14.243, 진짜 학습 캐시 300,000창 중 70,000창 표본):
#   ① **벌의 부재** — R²(채널 | 9기기 전력·켜짐) 가 반파 43·44 **0.931** ·
#      짝수 |I2|~|I6| **0.883** (|I2| 0.906) · 비율 26~28 0.801 이다. 같은 켜짐 조합 안
#      잔차 퍼짐은 전체의 **0.12~0.16** 뿐. 그 채널을 믿어도 합성 안에선 손실이 한 번도
#      안 오르니 가중치를 깎을 −∇L 이 **생기지 않는다**
#   ② **증폭** — 채널별 z-score 는 없다. `arcsinh(x·20)` 이 같은 일을 한다:
#      ch0 Re(I1) 2.55A -> 0.39/A 인데 ch22 |I14| 0.0046A -> 19.92/A = **50.8배**
#   ③ **취약** — 짝수차 7채널을 **12.1° 돌리기만** 해도 건강한 판이 죽는다 (14.230):
#      hdrop_s0 0.979->0.003 · hdrop_s2 0.939->0.012 · ondet_s2 1.000->0.003.
#      같은 각도 무작위도 똑같이 파괴하고 **홀수 블록은 0.995~1.000 무사**다
#
# 눈금의 근거 (14.234·14.237): 분절 풀은 반파 드라이를 **녹화 하나**(hair_dryer_1,
#   149.7초)에서만 뽑아 모양이 구간 사이 **0.06°** · 안 0.15° 로 얼어 있다. 실측 반파는
#   에피소드 사이 **2.2~9.2°** · 안 0.7~4.9° 로 돈다. σ=0.25 -> 3.55°(p90 7.66°) 가
#   그 폭 한가운데고, σ=0.40 -> 5.56°(p90 13.9°) 가 12.1° 파괴점까지 덮는다.
#
# ⚠ **반례를 먼저 적는다.** 전이 낙차(AUC 합성−실측)는 ch23 asinh(P/100) 0.943->0.521 과
#   ch0 Re(I1) 0.994->0.589 가 압도적이고 **고차 고조파는 반대로** 간다
#   (홀수 I7~I15 0.553->0.877). 즉 증폭은 짝수차에 걸렸지만 숏컷의 주력은 **절대 전력
#   수준**일 수 있다. 그쪽은 `cnn_finedc` 가 시험 중이다. 이 판은 ①의 시험이다.
#
# 관문 `src/run_gate_evenjit.py` 여섯 줄:
#   [1] σ=0 비트 동일 + 난수 흐름 불변   [2] 경로 census 를 **측정**으로 지어 표와 대조
#   [3] ★ 원시를 f 배 해 다시 지은 것과 채널공간 연산이 같다 (세밀 7.45e-08)
#   [4] k=5 에서도 성립   [5] 기하평균 1 (크기 불변)   [6] σ 눈금
#
# **반증 가능한 예측**
#   ① 체크포인트에 `even_jitter` 가 적히고 파라미터는 짝과 **동일**하다 (증강이라 구조 무변)
#   ② ★ 좋은 바닥(포트 게이트>0.8) 비율 p 가 **0.46 보다 커야** 한다 (씨앗 여섯 이항)
#   ③ ⚠ **미탐율이 안 올라야** 한다 — 모양만 돌리고 크기는 뒀으니 오를 이유가 없다
#   ④ 짝수차 12.1° 회전에 대한 게이트 붕괴(0.979->0.003)가 **누그러져야** 한다
#
# ⚠ 씨앗 **여섯**이다. 셋이면 표준오차가 0.29 라 어떤 처치도 못 읽는다 (14.205).
#SBATCH -J nilm_ej
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -a 0-11
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32h}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=$((SLURM_ARRAY_TASK_ID % 6))
ARM=$((SLURM_ARRAY_TASK_ID / 6))
if [ "$ARM" -eq 0 ]; then SIG=0.25; NAME=evenjit25; else SIG=0.40; NAME=evenjit40; fi
echo "노드 $(hostname) · 팔 $NAME (σ=$SIG) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_evenjit
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_dzn`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH=""
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter $SIG"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT \
    --tag cnn_${NAME}_s${SEED} > "results/cnn_${NAME}_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_${NAME}_s${SEED}.log"
ls -la results/cnn_${NAME}_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
