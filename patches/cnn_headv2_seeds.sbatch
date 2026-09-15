#!/bin/bash
# 14.130 — 머리를 **처음부터 다시 짓는다** (`--head-layout v2`). 시드 셋 짝비교.
#
# 사용자: *"지금 너무 많은 요소가 있어서 너도 나도 모델 구조를 완벽히 파악을 못하고
#   있잖아. 기존 구조 백업해 두고 한번 깔끔하게 처음부터 해 보자."*
#
# **규칙 하나로 머리를 설명한다 — "모든 요약은 타깃 기준 2.0초 격자로 낸다."**
#   v1 (765칸 · 아홉 가지)              v2 (1475칸 · 다섯 가지)
#   원시타깃57 탭0 64 탭1 64 탭4 128    [1] 타깃 순간  원시 57 · 얕은탭 64 · 깊은탭 128
#   h.mean 128   <- 순서 불변           [2] 세밀 5토막 얕은탭(RF 19) mean+amax = 640
#   h.amax 128   <- 순서 불변           [3] 깊은 요약  과거 mean+amax · 미래 mean+amax = 512
#   깊은탭 128                          [4] 광역 전역 평균 64   <- 14.94 로 닫힌 축 (예외)
#   hw.mean 64   <- 순서 불변           [5] 원시 5토막 P max/min = 10
#   원시창통계 4 <- 순서 불변
#
# ★★ **오늘 처음 잰 기전 — 핵심은 풀링이 아니라 정규화다.**
#   `_block` 이 `Conv1d -> GroupNorm(C, T 전체) -> GELU` 라 **수용영역과 무관하게**
#   통계가 창 전체에서 나온다. 그래서 풀링만 토막내면 토막이 안 갈린다:
#     한 토막을 흔들자 다섯 토막이 다 움직였다 — 대각 1.435 대 비대각 0.830 = **1.73배**
#   `--seg-pool`(14.72) · `--fine-future-segs`(14.122) · v2 첫 판이 전부 이 이유로 떨어졌다.
#   반대로 `--fine-time-split` 이 통한 까닭도 여기다 — 몸통을 **따로 태워** GroupNorm 도
#   따로 돌린 것이다 (과거<->미래 Δ가 **0.000**).
#   ⇒ v2 는 그 처방을 **2조각에서 5조각으로 일반화**한다. 얕은 스택(RF 19 < 120)을
#     토막마다 따로 태운다. 재검: 대각 **0.253** 대 비대각 **0.0000**, argmax 5/5.
#   깊은 스택은 과거/미래 둘로만 — 120칸에 dilation 64 를 태우면 거의 패딩이다.
#
# 세밀 600 = **정확히 5 x 120** 이고 타깃 239 가 두 번째 토막의 끝이다.
#   K=4 는 150사이클 · K=6 은 100사이클이라 둘 다 240 이 경계가 아니다 — **K=5 가 유일**하다.
#
# ★ 던지기 전에 **잰** 기준값 — 짝은 `cnn_pcap_s{0,1,2}` (채택된 팔. 같은 캐시·BASE·
#   WTAP·상한, `--head-layout` **하나만** 다르다)
#     ★1 면적 4.462 +- 1.820   ★2 축퇴 35.519 +- 8.709
#     ★3 현행 F1 0.931 +- 0.004   ★4 마스크 F1 0.893 +- 0.004
#     저항 4종 신원 0.9774 +- 0.0029   SMPS 3종 0.9807 +- 0.0087
#     판정 줄(7종) 0.8820 +- 0.0106    절대 잔차 21.27 +- 1.40 W
#     3시드 오븐 on_logit 상관 0.956 · 판정 일치 0.950   <- 씨앗 바닥
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 머리 입력 1475 · 파라미터 888k -> 1,070k. 학습이 돈다.
#   ② ★ **좋아진다고 예측하지 않는다.** 14.128 이 머리를 196칸 줄였다가 씨앗 하나가
#      통째로 무너졌다 (저항 신원 0.975 -> 0.821). 중복은 **해를 고정하는 값**을 한다.
#      v2 는 중복을 격자로 **바꾸는** 것이지 줄이는 게 아니지만, 배치를 통째로 갈았으니
#      같은 위험이 있다. **지킴 넷이 씨앗 바닥 안에 머물면 그것만으로 성공**이다.
#   ③ ★ **진짜 물음** — 전이 근처가 나아지나. 14.124 가 잰 바로 오븐 헛게이트는
#      **전이 ±20초 안에서만** 난다 (안 23.3% · 밖 0.0%, 큰 저항 부하 켜진 창).
#      v2 가 전이 시각을 진짜로 주는 첫 배치이므로 **전이 안 23.3% 가 내려가야** 한다.
#      안 내려가면 시간 귀속은 이 병의 원인이 아니고, 그러면 이 줄을 놓는다.
#   ④ 씨앗 판정 일치가 0.95 밑으로 떨어지면 배치가 해를 갈라놓은 것이다 (14.128 의 재판).
#
#SBATCH -J nilm_hv2
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 얕은 스택을 5번 더 태우므로 세밀 연산이 늘지만 조각이 120칸이라 실제로는 1.3배 안쪽이다.
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
$PY -X utf8 src/run_gate_headv2.py
$PY -X utf8 src/run_gate_finepad.py
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--head-layout" \
  || { echo "--head-layout 이 없다"; exit 1; }

# ⚠ BASE + WTAP + CAP 이 cnn_pcap_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_pad_s{0,1,2}` 를 짝으로 쓴다 (14.131).
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--head-layout v2 --fine-pad replicate"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP" in *--head-layout*|*--fine-pad*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116 · 14.122 · 14.128)
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--head-drop*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# v2 의 전제: 얕은 탭이 `tap_layers` 의 두 번째다 — 배치가 다르면 RF 가 어긋난다
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "탭 배치가 다르다 — 얕은 탭 RF 가 어긋난다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_hv2_s${SEED} > "results/cnn_hv2_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_hv2_s${SEED}.log"
ls -la results/cnn_hv2_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
