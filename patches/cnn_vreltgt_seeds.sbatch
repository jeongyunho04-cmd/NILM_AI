#!/bin/bash
# 14.53 — **창 전압을 타깃 사이클에서 읽는다** (`--vrel-target`). 3시드 x 2팔.
#
# 무엇이 틀려 있었나:
#   `L_harm` 이 맞추는 것은 **타깃 사이클 하나**의 `obs_harm` 인데, 거기 쓰는 전압이
#   `fine[:, 25].mean(-1)` — 세밀 창 600사이클(**10초**) 평균이었다.
#   핫플은 2.00초 주기로 0.47초만 통전하므로(14.48) 평균이 **통전 안 하는 구간의 높은
#   전압**을 섞는다. 실측 (`run_diag_vrel.py`, test_1~5):
#     전체 창          −0.01%      <- 정확히 0
#     P>1500           +0.11%
#     핫플통전·P>1500  **+0.28%**
#     오븐통전·P>1000  **+0.30%**
#   V 를 높게 읽으면 `sig ∝ 1/V` 라 지문이 작아지고 `L_harm` 이 전력을 **더** 요구한다
#   = **과대예측 쪽**, 그리고 **고전력 창에서만**.
#
# ⚠⚠ **두 입구가 갈려 있었다.** 2단계 `run_adapt` 는 `RealWindows.v_observed` =
#   `power_features[targets, 4]` 로 **이미 타깃 사이클**을 쓴다 (`realdata.py:199`).
#   1단계만 평균이었다. `run_gate_vreltgt.py` ③ 이 실측 창에서 둘을 맞대 본다
#   (최대 차 **0.0000V** 로 일치). [[pin-the-two-entry-points-against-each-other]]
#
# ⚠⚠⚠ **효과가 씨앗 바닥 아래일 수 있다 — 미리 적어 둔다.**
#   1500W 창에서 +0.3% = **+4.5W** 다. 14.51 의 3시드 짝차 표준편차가 `핫플통전·P>1500`
#   에서 **9.8W** 였다. 즉 이 판은 **3시드로는 못 읽는다**(차/폭 = 0.46 < 2).
#   그래도 거는 까닭은 셋이다:
#     ① 물리가 명백히 이쪽이다 (맞추는 대상이 타깃 사이클 하나다)
#     ② 1·2단계 입구 불일치를 없앤다
#     ③ 공짜다 — 굽기도 학습 시간도 안 변한다
#   **판정을 "좋아졌다" 로 쓰지 마라.** 읽을 수 있는 것은 "나빠지지 않았다" 까지다
#   ([[measure-the-seed-floor-before-reading-any-effect]]). 크기를 읽고 싶으면 시드를
#   늘려라 — 4.5W 를 2배 분해능으로 보려면 짝차 표준편차가 2.25W 여야 하고,
#   9.8/sqrt(n) < 2.25 면 **n >= 19** 다.
#
# ⚠ `FRAC` 을 **983591 의 승자**로 맞춰라. 기본 1.0 은 983573 의 `on` 팔과 짝이 맞는다.
#
#SBATCH -J nilm_vreltg
#SBATCH -p gpu3,gpu4,gpu5,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 983573 이 A10 에서 30:21~30:31 이었다.
#SBATCH -t 00:45:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
FRAC=${FRAC:-1.0}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · frac ${FRAC} · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문: **함수가 아니라 배선을 잰다** ────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vrel-target" || {
  echo "--vrel-target 이 없다 — 코드를 안 올렸다"; exit 1; }
# ③ 이 핵심이다 — 1단계 세밀채널@239 == 2단계 `v_observed` 인가 (색인이 밀리면 잡힌다)
$PY -X utf8 -m src.run_gate_vreltgt || { echo "vrel-target 배선 관문 실패"; exit 1; }
$PY -X utf8 -m src.run_gate_vanchor || { echo "vanchor 배선 관문 실패"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다"; exit 1; }

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor --harm-vnorm-frac $FRAC"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --tag cnn_vrt_off_s${SEED} > "results/cnn_vrt_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --vrel-target \
    --tag cnn_vrt_on_s${SEED} > "results/cnn_vrt_on_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: 10초 평균 (옛 경로) ==="; tail -32 "results/cnn_vrt_off_s${SEED}.log"
echo "===================================================================================="
echo "=== B: 타깃 사이클 ==="; tail -32 "results/cnn_vrt_on_s${SEED}.log"
ls -la results/cnn_vrt_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 (로컬에서, 실측 복합 5파일):"
echo "  python -X utf8 src/run_diag_hipow.py --pair 3 --ckpt \\"
echo "      results/cnn_vrt_off_s{0,1,2}.pt results/cnn_vrt_on_s{0,1,2}.pt"
echo "  ⚠⚠ 예상 크기는 **+4.5W** 이고 3시드 짝차 폭이 **9.8W** 다 — **못 읽는다.**"
echo "     읽을 수 있는 것은 '나빠지지 않았다' 까지다. '좋아졌다' 라고 쓰지 마라."
echo "     크기를 읽으려면 시드 19개가 필요하다 (9.8/sqrt(n) < 2.25)."
exit $RC
