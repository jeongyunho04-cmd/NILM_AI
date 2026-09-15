#!/bin/bash
# 14.153 — **`--w-harm` 0.1 -> 0.03 하나만.** 시드 셋 짝비교. 짝 `cnn_hv2`.
#
# 사용자: *"w-harm 0.03으로 한번 돌려봐"*
#
# ★ 왜 이것이 새 판인가 — **설계만 되고 한 번도 안 돌았다.**
#   `cnn_wharm_seeds.sbatch`(0 대 0.3)도 `cnn_wharm.sbatch`(0.10 대 0.02)도 산출물이 없다.
#   14.84 는 **올리는 쪽만** 기각했다 (0.1 -> 0.3 이면 축퇴가 심해진다).
#
# 근거 둘
#   ① 14.84 의 대수. 축퇴 창에서 Δ = L_harm(오븐답) − L_harm(포트답) = **−0.11~−0.13**
#      (포트 승률 5%). `Δtotal = w_harm·ΔL_harm` 이라 **부호가 w_harm 과 무관**하다.
#      올리면 오븐 편애가 세지고 **내리면 약해진다.**
#   ② 14.151 의 측정. 켜진 칸에서 게이트 경사의 몫이 BCE 53% / **harm 42%** / power 5% 다.
#      파라미터 수준에서 L_harm 은 게이트로 **11~15배**를 보낸다 (4판 전부). 그리고
#      13.84.2 가 잰 대로 그 방향은 **라벨과 무관한 동전 던지기**다.
#
# ⚠ 알려진 위험: `L_harm` 은 **신원을 담는 항**이다. 12.9절 ② 가 0.01 에서 핫플 F1
#   0.602 로 실패했다 — 다만 **150에포크(미수렴, 12.9.6)·수용영역 4배 이전**이라
#   만료된 진단이다 ([[a-diagnosis-expires-when-the-architecture-changes]]).
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① 학습이 돈다. harm 가중이 다르니 첫 에포크 total 이 대조와 **다르다**.
#   ② ★ **진짜 물음 — 게이트 결손이 줄어야 한다.** 홀드아웃 `log(출력/참)` 이 0 쪽으로.
#      기준선은 오븐 −0.21 · 미니PC −0.60 · 선풍 −0.40 · 빔 −0.28 이다.
#      **이게 안 움직이면 harm 가설도 아니다** (14.152 에서 게이트 가설이 그렇게 죽었다).
#   ③ ⚠ **신원이 1차 위험.** 저항 4종·SMPS 3종·핫플 F1 을 짝으로 본다.
#   ④ 오븐 통전 헛율은 **관찰만** — 유효 표본이 3 이다 (14.152 정정).
#   ⑤ 통과 조건: ② 가 움직이고 ③ 이 씨앗 바닥 안.
#
# ★ 짝의 기준값: 판정줄 .9020 / SMPS .9851 / 잔차 18.6W / 혼자켜짐 19.7%
#SBATCH -J nilm_wh03v2
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:10:00
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

# ⚠ BASE 는 짝(`cnn_hv2`)과 **--w-harm 만** 달라야 한다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--head-layout v2 --fine-pad replicate"
TREAT="--w-harm 0.03"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--w-harm*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_wh03v2_s${SEED} > "results/cnn_wh03v2_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_wh03v2_s${SEED}.log"
ls -la results/cnn_wh03v2_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
