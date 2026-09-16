#!/bin/bash
# 14.295 — **궤적 평균** `--swa-start` (SWA, Izmailov 외 UAI 2018). 사용자 물음에서 나왔다.
#
# 사용자: *"손실이 최저점이 되는 곳이 안정되지 않은 것 같다."*  그 증상의 이름이
# **underspecification** 이고 (D'Amour 외, JMLR 2022) 이 프로젝트는 이미 재 놨다:
#
#   §22.2  실측 행동이 정반대인 판 여섯 쌍을 라벨 있는 학습 분포에서 맞대니
#          포트 AUC 가 12판 전부 0.9999~1.0000 이고, **실측에서 무너지는 쪽의 L_on 이 낮다**
#          -> 지역최소가 아니다. 둘 다 목적함수의 똑같이 좋은 해이고,
#             우리가 신경 쓰는 축이 그 **해집합 안의 자유방향**이다
#   §19.4  씨앗 0·1 의 궤적은 결정경계 **위에** 서 있고 s2 만 떨어져 있다
#   §19.5  그래서 "궤적을 경계에서 떼는 것"을 처치로 — 게이트마진·SAM·**EMA**·LR 스케줄
#          (등록만 해두고 **안 던졌다**)
#   14.294 자유방향의 크기: 씨앗이 갈리는 창이 **1.8%** 인데 그게 불일치의 70~88% 를 진다.
#          나머지 98% 는 씨앗 간 폭 중앙이 **0W** 다. 그리고 `--even-jitter` 는
#          그 1.8% 를 **안 줄였다**(1.80->1.73 / 1.73->1.80) — 넘어지는 쪽만 바꿨다
#
# 원리: LR 이 0 이 아닌 동안 SGD 는 한 점이 아니라 분지 바닥 주변의 **정상분포**에 든다.
#       마지막 가중치는 거기서 뽑은 표본 하나다. 궤적을 평균하면 중심을 재는 것이고,
#       비대칭 골짜기(He·Huang·Yuan, NeurIPS 2019)에서 **학습** 손실의 최소점은 가파른
#       벽 쪽으로 치우쳐 있어 평균이 평평한 쪽으로 더 들어간다. 분포 이동 = 손실 곡면이
#       밀리는 것이므로, 평평한 안쪽 점이 덜 무너진다.
#
# ⚠ **앙상블(§19.1)과 다른 물건이다.** 앙상블은 서로 다른 분지의 판들의 *출력* 을
#   평균해 2대1 다수결이 되고, 확신에 차서 틀린 다수가 살아남았다. 여기서는 **한 궤적
#   안의 가중치**를 평균해 판 **하나**를 만든다. 씨앗이 다르면 초기값이 달라 분지가
#   다르므로 **체크포인트끼리는 절대 평균하면 안 된다**.
#
# ★ **짝이 공짜로 나온다.** `swa_start` 부터 LR 을 고정하면 `cnn_evenjit40` 과 다른 것이
#   **둘**(스케줄 꼬리·평균)이 되어 이겨도 어느 쪽인지 모른다. 그래서 학습이 같은 궤적의
#   **끝점**을 `cnn_swa_s{N}_last.pt` 로 따로 저장한다 — 씨앗·자료순서·스케줄이 전부
#   같고 **평균만** 다르다. 판정이 두 걸음으로 갈린다:
#       evenjit40 -> swa_last   = **LR 꼬리 고정**의 몫
#       swa_last  -> swa        = **평균**의 몫
#
# 눈금: ep220 부터. 코사인이 거기서 내는 5.08e-05 (lr0 의 16.9%) 로 고정하고 81 점 평균.
#       12.9.6 이 "곡선이 ep210 부터 평평하다" 고 쟀고, SWA 원 논문의 관례가 예산의
#       마지막 ~25% 를 상수 LR 로 도는 것이다 (300 의 27%).
#
# 관문 `src/run_gate_swa.py` 여덟 줄. 그중 둘이 이 팔의 급소다:
#   [2] ★★ **무동작 감지** — 코사인 꼬리를 그냥 평균하면 5.16e-08 인데 LR 을 고정하면
#       1.35e-04 다 (**2,622배**). LR 고정이 처치의 일부라는 증거다
#   [3] ★ **BN 재추정 불필요** — GroupNorm 10 개 · running stat **0 개**
#
# **반증 가능한 예측**
#   ① 학습 로그에 `||w̄ − w_T||/||w_T||` 가 **1e-4 보다 크게** 찍힌다
#      (작으면 무동작이다 — 그 자리에서 기각하고 swa_lr 을 올린다)
#   ② ★★ **씨앗이 갈리는 창 1.8% 가 준다.** 이게 이 팔의 존재 이유다.
#      안 줄면 씨앗이 서로 **다른 분지**라는 뜻이고, 그러면 평균으로는 못 합의시킨다
#   ③ 게이트 6/6 **유지** · 네 방향 합이 evenjit40 의 186 보다 **안 는다**
#   ④ 홀드아웃 MAE(4.84)·F1(0.9318) 이 **안 나빠진다**
#      ⚠ 평평한 해는 학습 분포 손실을 **조금 손해** 볼 수 있다. ±0.1W 는 봐준다
#   ⑤ `swa_last` 가 `evenjit40` 과 ②③④ 에서 **구별 안 된다**
#      (구별되면 이득이 평균이 아니라 **LR 꼬리**에서 온 것이다 — 그것도 답이다)
#
#SBATCH -J nilm_swa
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:20:00
#SBATCH -a 0-5
#SBATCH -o logs/swa_%A_%a.out
#SBATCH -e logs/swa_%A_%a.err

set -euo pipefail
# ⚠ 이 네 줄은 짝(`patches/cnn_evenjit_seeds.sbatch`)에서 **그대로 베낀 것**이다.
#   처음에 `PY=python` 과 없는 캐시 경로를 적었다가 잡았다 — 로그인 노드엔 torch 가
#   없고 `processed_data/train60_v33` 는 **존재하지 않는다**.
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32h}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_evenjit
$PY -X utf8 -m src.run_gate_swa
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE·WTAP·CAP·TREAT 는 짝(`cnn_evenjit40`)과 **글자 그대로 같다**. 다른 것은 SWA 셋뿐이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40"
SWA="--swa-start 220 --swa-every 1 --select final"

echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $TREAT"
echo "  처치: $SWA   (짝 evenjit40 은 이 셋만 없다)"
case "$BASE$WTAP$CAP$TREAT" in *--swa-start*|*--swa-lr*|*--swa-every*) echo "처치가 바닥으로 샜다"; exit 1;; esac
case "$BASE$WTAP$CAP$TREAT$SWA" in *--odd-phase-jitter*|*--fine-dc*|*--head-conductance*|*--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이나 다른 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT $SWA \
    --tag cnn_swa_s${SEED} > "results/cnn_swa_s${SEED}.log" 2>&1
echo "=== 끝 $(date '+%F %H:%M:%S')"

# ① 를 그 자리에서 읽는다 — 무동작이면 여기서 알아야 한다
grep -E "가중치 평균 [0-9]+ 점" "results/cnn_swa_s${SEED}.log" || echo "⚠ 평균 줄이 없다"
tail -4 "results/cnn_swa_s${SEED}.log"
ls -la results/cnn_swa_s${SEED}.pt results/cnn_swa_s${SEED}_last.pt
