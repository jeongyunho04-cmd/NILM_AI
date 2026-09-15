#!/bin/bash
# 14.128 — 머리 입력 **196칸을 뺀다** (`--head-drop rawstat,tap0,pastmean`). 시드 셋 짝비교.
#
# 사용자: *"세밀갈래를 처음부터 만드는 건 어때. 너무 많은 정보가 헤드에 덕지덕지
#   붙어 있는 모양새인데 이게 문제일 수도 있지 않을까? 여러 입력이 섞이면서 불안정해질
#   수도 있나?"*
#
# 중복은 **실재한다** (실측 2753창, cnn_pcap_s0, 머리 입력 765칸):
#   유효 차원   분산 90%까지 79칸 · 95% 144칸 · 99% 339칸 · **참여비 20.2**
#   덩이별 R^2 (그 덩이를 **나머지 전부**로 맞힌 값)
#     과거평균 0.982 · 원시창통계 0.982 · 탭0 0.980 · 탭1 0.976 · 원시타깃 0.968
#     과거최대 0.947 · 광역평균 0.929 · 탭4 0.925 · 깊은탭 0.905
#   -> **어느 덩이를 빼도 나머지로 90~98% 복원된다.**
#
# ⚠ 그런데 **불안정의 원인이라는 증거는 없다.** 같은 팔 3시드의 오븐 on_logit 상관이
#   hzwt 0.958 · pcap 0.956 · tsp 0.960 · fseg 0.945, 판정 일치 0.94~0.95 다.
#   해가 여럿으로 갈라지지 않는다.
#   ⇒ 그래서 이 판은 **"고침" 이 아니라 "빼도 되나" 를 묻는 실험**이고,
#     통과 조건도 **"좋아진다" 가 아니라 "안 나빠진다"** 다.
#
# ⚠⚠ 추론에서 0 으로 죽여 보는 것으로는 못 묻는다 — 14.123 이 그걸로 원시 창통계를
#   -3.6%p 로 읽었는데 `asinh(P/100)=0` 은 실측에 없는 입력이라 **분포 이탈**이었다.
#
# 뺀 것 (R^2 가 가장 높은 셋): 원시 창통계 4 + 탭0 64 + 과거 평균 128 = **196칸**
#   머리 입력 765 -> **569** (26% 감축) · 머리 파라미터 195,840 -> 145,664
#   몸통은 **한 칸도 안 건드린다** (관문 [5] 가 확인).
#
# ★ 던지기 전에 **잰** 기준값 — 짝은 `cnn_pcap_s{0,1,2}` (같은 캐시·BASE·WTAP·상한,
#   `--head-drop` **하나만** 다르다)
#     ★1 면적 4.462 +- 1.820   ★2 축퇴 35.519 +- 8.709
#     ★3 현행 F1 0.931 +- 0.004   ★4 마스크 F1 0.893 +- 0.004
#     저항 4종 신원 0.9774 +- 0.0029   SMPS 3종 0.9807 +- 0.0087
#     판정 줄(7종) 0.8820 +- 0.0106    절대 잔차 21.27 +- 1.40 W
#     오븐 헛게이트율 7.1% · 전이 ±20초 안(큰부하) 23.3%
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 머리 입력이 569 다. 학습이 돈다.
#   ② ★ **진짜 물음 — 안 나빠지나.** 저항 4종 신원 · SMPS 3종 · 판정 줄 · ★4 의
#      짝차가 **전부 씨앗 바닥 안**에 머문다. 그러면 중복은 무해하고, **머리를 처음부터
#      다시 짓는 근거**가 생긴다.
#   ③ 무너지면 (어느 하나라도 짝차가 바닥 밖으로 내려가면) **그 중복이 일을 하고
#      있었다**는 뜻이다. 그것도 소득이고, 그때는 어느 덩이가 필수인지 하나씩 뺀다.
#   ④ **좋아질 것으로는 보지 않는다.** 3시드 판정 일치가 0.95 라 해가 갈라지고 있지
#      않다. 좋아지면 내 "중복은 무해하다" 읽기가 틀린 것이다.
#
#SBATCH -J nilm_hdrop
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984729(같은 바닥)가 6초/에포크, 300에포크 약 30분이었다. 머리가 작아져 조금 빠르다.
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
$PY -X utf8 src/run_gate_headdrop.py
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--head-drop" \
  || { echo "--head-drop 이 없다"; exit 1; }

# ⚠ BASE + WTAP + CAP 이 cnn_pcap_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_pcap_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--head-drop rawstat,tap0,pastmean"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP" in *--head-drop*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116 · 14.122)
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# 처치의 전제: 탭0 을 빼려면 탭에 0 이 있어야 한다
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "탭 배치가 다르다 — 덩이 경계가 어긋난다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_hdrop_s${SEED} > "results/cnn_hdrop_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_hdrop_s${SEED}.log"
ls -la results/cnn_hdrop_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
