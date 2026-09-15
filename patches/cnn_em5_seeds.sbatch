#!/bin/bash
# 14.160 — **짝수차 크기에 5탭 이동중앙값.** `--even-median 5` 하나만. 짝 `cnn_pcap`.
#
# 사용자: *"일단 확실한 거부터 고치자. 반파 피처를 중앙값으로 주면 되는 거 아니야?"*
#
# 겨냥은 맞았고 **자리만 달랐다**:
#   · ch43/44 는 이미 61주기 **평균**인데, 그쪽을 되돌려도 유령이 402 -> 398.6W 로 안 죽는다
#   · 모델이 읽는 것은 평활 없는 **ch16~22 (순시 짝수차 크기)** 다 -> 402 -> 67.5W
#   · 그래서 중앙값을 `mag` 의 **짝수 행**에 건다. 한 자리에서 ch16~22 · 28 · 39 · 43 · 44 가 전부 robust
#
# 병 (14.159): 릴레이가 반주기에서 끊기면 **그 한 사이클이 물리적으로 반파**다.
#   test_1 핫플 듀티에서 |I2| 4.9 -> **483mA** (h2/h1 0.0018 -> 0.27, 구간 최대 0.96).
#   사전에서 반파는 **드라이기 s1 (0.4326)** 하나뿐이라, 드라이기가 **없는 파일**에
#   208~402W 유령이 7번 선다.
#
# k=5 인 까닭 — 과도는 **1사이클**이다 (실측):
#   |I2| 최대  과도 1176.7 -> **12.8mA** (92배) · 진짜 반파 891.5 -> 884.4 (**0.8% 손실**)
#   순시값에서는 과도가 진짜보다 **커서** 분리가 안 된다. k=5 에서 **69배** 갈린다.
#   비용 250창x3600x7채널에 **0.2초** (k=61 이면 3.9초 — 옛 주석의 "25분" 이 그것이다).
#
# ★ 추론에만 걸어 미리 잰 짝비교 (`cnn_pcap` 3시드, 학습 안 한 상태 = **분포 밖**)
#   드라이 유령 스파이크 **7개 -> 0개**
#   절대 잔차 19.3/22.1/22.4 -> **17.0/18.6/19.9W**  (평균 21.3 -> **18.5W**, 씨앗마다 개선)
#   판정줄 .8820 -> .8774 · SMPS .9807 -> .9693 · 혼자켜짐 31.8 -> 32.9%
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① 학습이 돈다. 입력이 바뀌므로 첫 에포크 손실이 대조와 **다르다**.
#   ② ★ **유령 스파이크가 안 돌아와야 한다** — `run_diag_ghost` 로 test_1 드라이 0개.
#   ③ ★ **절대 잔차가 낮아야 한다** — 21.3W 에서 18~19W 대로. 이게 이 판의 본전이다.
#   ④ **검출 손해가 회복돼야 한다** — 분포 밖이라 생긴 −0.005 / −0.011 이면, 학습하면
#      판정줄 .8820 · SMPS .9807 근처로 돌아와야 한다. **안 돌아오면 진짜 손해**다.
#   ⑤ 오븐<->포트 축퇴(14.158)는 **안 건드린다** — 그건 짝수차와 무관하다. 관찰만.
#
# ⚠ 채점 규약: 체크포인트에 `even_median` 을 기록하고 `load_model` 이 어긋나면 경고한다.
#SBATCH -J nilm_em5
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
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
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_pcap`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH=""
TREAT="--even-median 5"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_em5_s${SEED} > "results/cnn_em5_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_em5_s${SEED}.log"
ls -la results/cnn_em5_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
