#!/bin/bash
# 14.116 — **세밀 몸통을 타깃에서 둘로** (`--fine-time-split`). 시드 셋 짝비교.
#
# 사용자: "미래를 더 많이 알수록 혼동이 는다는 게 말이 되냐. 이게 말이 된다는 것 자체가
#   모델 구조에 결함이 있다는 거잖아."  맞다. 정보가 늘어서 나빠질 수는 없다.
#   나빠진다면 그 정보가 **"미래의 것" 이라는 표시 없이** 들어온다는 뜻이다.
#
#   세밀 창 600 · 타깃 239 · 미래 361사이클 = 6.02초
#   깊은 탭 수용영역   wtap 없음 187 (±1.56초)  ·  **wtap 켬 763 (±6.36초)**
#   -> `--fine-extra-dilations 32,64` 를 켜면 `h[:,:,239]` 라는 "지금의 특징" 안에
#      **미래 6.02초가 이미 섞여** 있다. 모델은 그것을 미래라 부를 길이 없어 무시도 못 한다.
#
# ★ 개입으로 확인한 것 (미래 6.02초를 타깃값으로 덮는다, cnn_hzwt_s0)
#   test_5 260.0초  핫플 헛게이트 **0.878 -> 0.004**  (참 통전은 263.0초)
#   test_5 128~138초 **포트 게이트 0.003 -> 0.938** · 오븐 0.998 -> 0.431
#     -> 사용자가 열 번 짚은 오븐<->포트 혼동이 **미래 누수**다. 모델은 현재만 보면 포트인 줄 안다.
#   전체 5파일   오븐 **헛detect 의 22.0%** 가 미래에 기댄다 (참detect 는 **0.8%**)
#                핫플 참detect 의 **30.1%** 는 미래가 필요하다 (듀티 위상)
#   -> 미래를 **없애면 안 되고**, **시간 부호를 붙여야** 한다.
#
# 손잡이: `--fine-time-split` — 몸통을 `[0:t+1]` 과 `[t+1:]` 에 **같은 가중치로** 따로
#   태운다. 타깃 탭은 과거만, 미래는 별도 요약(mean/amax)으로 머리에 간다.
#   정보를 하나도 안 버리고 **시간 부호만** 준다. 몸통 파라미터 불변(공유). 끄면 비트 동일.
#   ⚠ `--seg-pool` 과 다르다 — 저쪽은 전역 풀링을 쪼갠다. 14.42 가 그 처방을 낼 때 몸통은
#     RF 187 이라 미래가 풀링으로만 샜고, 14.72 가 `--seg-pool 2` 를 기각(-0.9 ± 2.0)한 것도
#     그 구조에서의 결과다. wtap 을 켠 지금은 **새는 길이 탭 안**이라 풀링으로는 못 막는다.
#
# ★ 던지기 전에 **잰** 기준값 (3시드, 로컬 `cnn_hzwt_s{0,1,2}`)
#     ★1 면적 2.908 +- 1.271   ★2 축퇴 33.9 +- 3.9 %
#     ★3 현행 라벨 오븐 F1 0.914 +- 0.018 (재현 0.985 / 정밀 0.853)
#     ★4 물리마스크 오븐 F1 0.859 +- 0.024 (재현 0.988 / 정밀 0.761)
#     오븐 헛게이트 10.1% (A 축퇴 57창 · B 팬조명 196창 · C 초과 6창)
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① ★ **직격** — `test_5` 128~138초에서 포트를 고른다. 지금은 포트 게이트가 0.003~0.136
#      인데 **0.5 넘어야** 한다. 3시드 중 **3/3**. 이게 이 판의 통과 조건이다.
#   ② ★2 축퇴 33.9% -> **25% 밑**, 3/3. (미래 누수가 원인이면 여기가 같이 내려간다)
#   ③ **핫플 참detect 가 안 내려간다** — 미래를 버리는 게 아니라 부호를 붙이는 것이므로
#      듀티 위상 정보는 그대로 있어야 한다. 핫플 재현 짝차가 **-0.03 안**.
#      내려가면 "부호만 준다" 는 내 주장이 틀린 것이다.
#   ④ `test_5` 260.0초에 핫플을 3초 일찍 켜지 않는다 (게이트 0.878 -> 0.5 밑).
#   ⑤ 지킴: 저항 4종 신원 0.97 (한 시드도 밑이면 안 된다) · SMPS 3종 · 판정 줄 · 절대 잔차
#
#SBATCH -J nilm_tsp
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
# 14.112 측정대로 갈래당 1GPU·3코어. 몸통을 두 번 태우므로 세밀 연산이 2배지만
# 조각 길이가 240/361 이라 실제로는 1.2배 안쪽이다.
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:30:00
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
$PY -X utf8 src/run_gate_timesplit.py
$PY -X utf8 src/run_gate_finerf.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--fine-time-split" \
  || { echo "--fine-time-split 가 없다"; exit 1; }

# ⚠ BASE + WTAP 이 cnn_hzwtap_seeds.sbatch 의 `BASE + A_FLAGS` 와 **글자 그대로 같아야**
#   한다 — 대조를 새로 안 굽고 로컬 `cnn_hzwt_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
TREAT="--fine-time-split"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP"
echo "  처치: $TREAT"
# 손잡이가 섞이면 이 판이 무엇을 재는지 알 수 없다
case "$BASE$WTAP" in *--fine-time-split*) echo "처치가 BASE 로 샜다"; exit 1;; esac
case "$BASE$WTAP$TREAT" in *--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "다른 손잡이가 섞였다"; exit 1;; esac
# 처치의 전제: wtap 이 켜져 있어야 RF 가 763 이고, 그래야 이 병이 존재한다
case "$WTAP" in *"--fine-extra-dilations 32,64"*) ;; *)
  echo "wtap 이 없다 — 이 처치의 전제가 무너진다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $TREAT \
    --tag cnn_tsp_s${SEED} > "results/cnn_tsp_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_tsp_s${SEED}.log"
ls -la results/cnn_tsp_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
