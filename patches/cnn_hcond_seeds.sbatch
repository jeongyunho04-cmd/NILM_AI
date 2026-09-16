#!/bin/bash
# 14.285 — **전도도 머리** `--head-conductance`. 짝은 `cnn_dzn`(씨앗 0~2).
#   ⚠ 짝이 evenjit 이 아니다 — 이건 지터(구조 무변)가 아니라 **머리의 목표**를 바꾸는 것이라,
#     다른 게 하나만 되려면 아무 지터도 안 걸린 바닥과 견줘야 한다.
#
# 까닭 (14.282~283 측정):
#   오븐 통전 R **42.08Ω** · 포트 R **35.67Ω** -> 214V 에서 1088W 대 1284W = **196W** 차이.
#   갈리고도 남는데 결정까지 못 온다:
#     · 두 가설의 재구성 잔차 26W 대 29W -> **둘 다 deadzone 100W 안** = 손실 0
#     · `--w-swap` · `--w-gate-cond` 가 **둘 다 0** -> 맞바꿈에 마진이 없다
#     · 같은 18% 저항 오차의 손실값이 1.4kW 급 0.011608 대 11W 급 **0.000001**
#       (옛 `Huber(P/s_i)` 가 **11,600배** 싸게 본다)
#   그리고 14.16 이 남긴 진단: *"전압 효과가 전부 상태 **안**에 있는데 헤드는 상태 **사이**
#   100배에 손실이 지배당해 7%를 잡음으로 취급한다. 헤드를 먼저 고쳐야 한다."*
#
# 처방 — 목표를 **V 불변 전도도 영역**으로:
#   순전파  p_raw *= (V/222)^e_k        (vexp 와 같다)
#   목표    log(P / (V/222)^e_k)        <- **V 가 사라진다**
#   ⇒ `vexp` 가 죽은 이중 계산(구조 +2.02 + 헤드 0.66 = 2.68)이 **목표 쪽에서** 막힌다
#
# ⚠ **반증을 먼저 적는다.** 합성 홀드아웃은 이미 AUC 0.9999 · F1 0.933 이다.
#   **학습에서 안 나는 오차는 학습 손실로 못 고친다.** 이게 통한다면 손실이 오차를 잡아서가
#   아니라 표현이 달라져 전이가 나아져서고, 그건 결과가 아니라 **가설**이다.
#
# ⚠ `run_adapt`(2단계) 배선이 아직이다 — `run_gate_lossparity` 의 빚 목록에 등재했다.
#   **이 팔이 채택되면 즉시 갚아야 한다** (안 갚으면 adapt 이 옛 와트 목표로 되돌린다).
#
# 관문 `src/run_gate_hcond.py` 여덟 줄. 핵심 둘:
#   [3] 저항 물리로 V 를 ±8% 흔들어도 목표가 1109.00/1109.00/1109.00 — 변동 **0.0000%**
#   [6] 신원 오차가 전력 수준과 무관 (0.007013 대 0.007013)
#
# **반증 가능한 예측**
#   ① 학습 파라미터가 짝과 **동일** (v_exp 는 버퍼다) · `head_conductance: true` 기록
#   ② ★ 실측 오븐↔포트 맞바꿈이 준다 — test_3 114초 포트 오탐 18창(evenjit25_s0 기준)과
#      네 방향 오차표의 C(포트 오탐 15)·B(오븐 미탐 134)가 줄어야 한다
#   ③ ⚠ 합성 홀드아웃이 **안 나빠진다** (지금 MAE 4.65 · F1 0.933)
#   ④ 안 줄면 **틀린 것이다**
#
# ⚠ -t 01:20:00 — 14.246 이 00:55:00 으로 씨앗 둘을 TIMEOUT 냈다 (측정 38분 x 1.5 = 57분).
#SBATCH -J nilm_hc
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:20:00
#SBATCH -a 0-5
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
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_hcond
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_dzn`)과 **글자 그대로 같아야** 한다 — 다른 것은 ARCH 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--head-conductance"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  처치: $ARCH $TREAT"
case "$BASE$WTAP$CAP" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--head-conductance*) echo "처치가 바닥으로 샜다"; exit 1;; esac
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--fine-dc*|*--even-jitter*|*--odd-phase-jitter*)
  echo "기각된 손잡이나 다른 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT \
    --tag cnn_hcond_s${SEED} > "results/cnn_hcond_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_hcond_s${SEED}.log"
ls -la results/cnn_hcond_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
