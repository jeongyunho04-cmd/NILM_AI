#!/bin/bash
# 14.224 — **시간축 DC 분리** `--fine-dc split`. 짝 `cnn_dzn`.
#
# 까닭 (14.220~221 측정):
#   · 같은 L2 크기 섭동에 대한 첫 conv 의 응답이 **DC 가 AC 의 776배** (15판 중앙, 63~938)
#   · 첫 conv 커널 시간합 |Σw|/|w| = **4.33** (무작위 기대 2.12의 2배) — 막도록 배운 적 없다
#   · 실측 실패 창의 변위는 창 전체에 고르고(에너지 8~15%/칸) 맞히는 창은 타깃에 **78.3%**
#     몰려 있다. 그래서 **실패만 DC 통로로 직통**한다
#   · 변위를 갈라 주입: **DC 만 +5.7 -> −9.8 (게이트 0.000)** · AC 만 +2.7 (0.938)
#     **가짜약**(맞히는 구간의 DC) +2.7 (0.938) — "DC 면 뭐든" 이 아니다
#   · DC 를 통째로 빼면 게이트는 살지만(0.005 -> 0.993) **미탐 17.5% -> 36.8%** 다.
#     그래서 **버리지 않고 머리로 옮긴다**
#
# ⚠⚠ **반례를 먼저 적는다.** `leak4` 는 인과 GroupNorm 덕에 DC/AC 가 이미 **63~75배**
#    (다른 판의 1/10)인데 실측에서 **가장 깊게 무너졌다** (s0 0.002 · s1 0.006).
#    "DC 민감도를 낮추면 낫는다" 는 **아직 증명 안 됐다.** 이 판이 그 가설의 시험이다.
#
# 관문 `src/run_gate_finedc.py` 여섯 줄이 못 박는다:
#   [1] keep 비트 동일 0.000e+00   [2] 파라미터 +14,592 = n_dc(57) x 256 정확
#   [3] conv 입력의 채널별 시간평균 7.15e-09   [4] ★ **DC/AC 이득비 191배 -> 0.00배**
#   [5] 머리 뒤 57칸 = 떼어 낸 DC (정보 보존)   [6] keep 가중치를 못 싣는다
#
# **반증 가능한 예측**
#   ① 파라미터가 dzn 보다 **정확히 +14,592** (체크포인트에 `fine_dc: "split"` 이 적힌다)
#   ② 실측 실패 창에 DC 만 주입해도 게이트가 **안 뒤집혀야** 한다
#   ③ ★ 포트 게이트가 좋은 바닥(>0.8)에 떨어지는 비율 p 가 **0.46 보다 커야** 한다
#   ④ ⚠ **미탐율이 안 올라야** 한다 — 전량 제거의 17.5% -> 36.8% 가 이 설계로 사라지는지
#
# ⚠ 씨앗 **여섯**이다 (`-a 0-5%3`). 셋으로는 못 읽는다 — 판정이 p≈0.46 이항이고
#   씨앗 셋이면 표준오차가 0.29 다 (14.205). 동시 3개로 묶은 것은 QOS 몰림 때문이다
#   ⚠ 2026-09-16 정정: `%3` 은 **불필요했다.** QOS 는 `node=10` 이고 GPU 노드가 한 대에
#     4장이라 넷이 한 노드에 붙는 것은 정상 배치다 (n009 FreeMem 243GB, 여섯 다 무사).
#     `scontrol update JobId=<id> ArrayTaskThrottle=6` 으로 풀었다. 다음엔 `-a 0-5` 로.
#SBATCH -J nilm_finedc
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
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
$PY -X utf8 -m src.run_gate_finedc
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_pcap`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--fine-dc split"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_finedc_s${SEED} > "results/cnn_finedc_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_finedc_s${SEED}.log"
ls -la results/cnn_finedc_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
