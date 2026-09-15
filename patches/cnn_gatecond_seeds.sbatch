#!/bin/bash
# 14.106 — **게이트를 통전에 묶는다** (`--w-gate-cond`). 시드 셋 짝비교.
#
# 왜 이 판인가 (14.105 의 진단)
#   학습 자료 20만 창의 상태 라벨: 오븐만 `on=1` 인데 전력이 없는 창이 35.3% 다.
#       오븐  ON 62897  state 1 **35.3%** 중앙 **15.0W**  | state 2 64.7% 중앙 1125W
#       핫플  ON 43417  state 2 100.0% 중앙 455W    <- state 1 이 **아예 없다**
#       포트  ON 45439  state 1 100.0% 중앙 1272W
#       드라이 ON 46228  state 1 43.6% 442W / state 2 56.4% 863W
#   `power = 시그모이드(on_logit) x p_raw` 라 게이트가 서면 전력 통로가 열린다.
#   실측에서 오븐 헛게이트 **10.1%** (포트 0.2 / 핫플 0.3 / 드라이 1.1), 그 **75.7%가
#   전력 없는 팬·조명**이고 **22.0%가 축퇴**(참으로 켜진 저항 기기를 못 켜고 그 와트를
#   오븐이 받음, 짝은 포트 77%).
#
#   ⚠ 이 병은 **세 번째** 만남이다. 12.162.4 가 재고 이유까지 적고 "**안 했다**" 로
#     남겼고(라벨 정의를 바꿔야 해서), 14.32 는 `res_cond_state` 로 **조합 탐색만**
#     피하게 했다. 여기서는 **라벨이 아니라 손실에서** 한다 — 채점기를 안 건드리므로
#     옛 판과 비교가 선다.
#
# 손잡이: `--w-gate-cond W` — `res_cond_state>0` 인 기기의 `on_logit` 을
#   `1{y_state == 통전상태}` 에 맞추는 BCE. 팬·조명 창의 최적 게이트가
#   `w_on/(w_on+W)` 다 (`w_on=0.3`). 관문 [5] 가 0.3/0.9/2.7 에서 확인했다.
#   **핫플에는 구조적으로 아무 일도 안 일어난다** (state 1 이 자료에 없다) — 오븐 전용이다.
#
# ★ 던지기 전에 **잰** 기준값 (3시드, 로컬 `cnn_hzwt_s{0,1,2}` = hz + wtap A)
#     ★1 면적 2.908 +- 1.271   ★2 축퇴 33.9 +- 3.9 %
#     ★3 현행 라벨 오븐 F1 0.914 +- 0.018 (재현 0.985 / 정밀 0.853)
#     ★4 물리마스크 오븐 F1 0.859 +- 0.024 (재현 0.988 / **정밀 0.761**)
#   ⚠ 대조를 `cnn_hzwt` 로 잡은 까닭: 14.105 의 진단(헛게이트 10.1% · B 75.7% ·
#     A 22.0%)을 **바로 그 판에서** 쟀다. 진단과 처치가 같은 바닥 위에 서야 한다.
#     그러므로 BASE 에 **wtap A 깃발이 들어간다** — 안 넣으면 짝이 안 선다.
#     (참고) `cnn_v32h_on`(wtap 없음): ★1 4.454 +- 1.005 · ★2 30.6 +- 1.5
#   ⚠ 12.162 때는 마스크가 F1 을 **올렸다**(0.530 -> 0.867, 재현 0.361). 지금은 반대다 —
#     재현이 0.975 로 올라가 병이 **재현에서 정밀로 옮겨갔다**. 이 손잡이가 겨냥하는 것이
#     바로 그 정밀이다.
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (걸렸나 검사, 예측 아님) 오븐 헛게이트가 10.1% -> **5% 밑**. 목표를 직접 내렸으니
#      이건 손잡이가 실렸다는 확인이지 과학적 예측이 아니다. 안 되면 배선이 틀린 것이다.
#   ② ★ **진짜 물음** — 게이트가 비싸지면 **전력 통로도 좁아지나**.
#         ★2 축퇴 33.9% -> **25% 밑**, 3/3.   ★1 면적 2.908 -> **2.0 밑**, 3/3.
#      (여태 제일 좋았던 판이 `hz끔 wtapB` 의 ★1 0.915 / ★2 33.3 이다)
#      안 움직이면 A 유형은 게이트 싸구려가 아니라 **순수 지문 축퇴**이고, 그러면
#      이 축을 닫고 컨덕턴스/지문 쪽으로 간다.
#   ③ 비용 — ★3 은 **내려간다**(예상된 일이다. 라벨이 팬·조명도 ON 으로 적는다).
#      ★4 는 **안 내려가야 한다** (0.859 밑이면 진짜 악화). 정밀 0.761 이 올라야 성공이다.
#   ④ B(W=2.7)가 A(W=0.9)보다 ①은 세고, ③의 ★3 은 더 많이 내려간다. ②에서 B>A 이면
#      "게이트 비용" 가설이 맞고, A~B 로 평평하면 문턱 효과다.
#
# ⚠⚠ 지킴 (**cnn_hzwt 와 짝으로**): 저항 4종 신원 0.97 (한 시드도 밑이면 안 된다) ·
#     SMPS 3종 · 판정 줄 · 절대 잔차
#
#SBATCH -J nilm_gcond
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# 984489 가 A6000 에서 25분이었다. 손잡이는 BCE 한 줄이라 비용이 안 는다.
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
$PY -X utf8 src/run_gate_gatecond.py
$PY -X utf8 src/run_gate_swap1.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--w-gate-cond" \
  || { echo "--w-gate-cond 가 없다"; exit 1; }

# ⚠ BASE + WTAP 이 cnn_hzwtap_seeds.sbatch 의 `BASE + A_FLAGS` 와 **글자 그대로
#   같아야** 한다 — 대조를 새로 안 굽고 로컬 `cnn_hzwt_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
# 대조(`cnn_hzwt`)가 쓴 처치. 이제 **바닥**이므로 두 갈래 모두에 들어간다.
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
A_FLAGS="$WTAP --w-gate-cond 0.9"
B_FLAGS="$WTAP --w-gate-cond 2.7"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP"
echo "  갈래 A: $A_FLAGS   (팬·조명 창 목표 게이트 0.25)"
echo "  갈래 B: $B_FLAGS   (팬·조명 창 목표 게이트 0.10)"
# 처치가 BASE 로 새면 대조가 무너진다
case "$BASE" in *--w-gate-cond*) echo "BASE 에 처치가 섞였다"; exit 1;; esac
# `res_cond_state` 기본값이 손잡이의 전제다 — 비면 항이 통째로 안 걸린다
case "$BASE" in *--res-cond-state*) echo "BASE 가 res-cond-state 를 덮어쓴다"; exit 1;; esac
# wtap 이 빠지면 대조(`cnn_hzwt`)와 짝이 안 선다. 14.78: dilation 을 **교체**하면 안 된다
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "wtap 의 탭에 4 가 없다 (14.78)"; exit 1;; esac
case "$WTAP" in *--fine-dilations*) echo "dilation **교체** 가 섞였다 (14.78 실패)"; exit 1;; esac
for f in "$A_FLAGS" "$B_FLAGS"; do
  case "$f" in *--fine-extra-dilations*) ;; *) echo "갈래에 wtap 이 빠졌다"; exit 1;; esac
done

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_gc09_s${SEED} > "results/cnn_gc09_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_gc27_s${SEED} > "results/cnn_gc27_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: --w-gate-cond 0.9 ==="; tail -32 "results/cnn_gc09_s${SEED}.log"
echo "===================================================================================="
echo "=== B: --w-gate-cond 2.7 ==="; tail -32 "results/cnn_gc27_s${SEED}.log"
ls -la results/cnn_gc09_s${SEED}.pt results/cnn_gc27_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
