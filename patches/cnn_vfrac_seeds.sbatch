#!/bin/bash
# 14.52 — **앵커를 몇 할만 걸면 영점인가** (`--harm-vnorm-frac`). 3시드 x 2팔.
#
# 왜 (14.51 이 3시드로 잰 것):
#   `--harm-vnorm-anchor` 는 겨냥한 것을 **맞혔다** — 실측 `핫플통전·P>1500` 잔차 중앙이
#   **+47.5 ± 4.1W (과소) -> −16.3 ± 7.5W (과대)**. 짝차 −63.8 ± 9.8W, 씨앗 폭의 6.5배다.
#   오븐통전·P>1000 도 +43.6 -> −15.5 (짝차 −59.1 ± 3.3W, 18배). **그런데 부호를 넘겼다.**
#
#   왜 넘겼나 (14.52 가 쟀다):
#     앵커가 넣은 보정   오븐 5.2% x 통전 1357W ≈ **70W**
#     실측에 나타나 있던 오차            **+47.5W** = 그 **68%**
#     보정 걸고 예측이 움직인 양          **−63.8W** = 그 **91%**
#   `L_power`(가중 1.0)가 `L_harm`(0.1)을 32% 되당기고 있어 **오차는 68%만 드러나 있었는데
#   보정은 91%가 통과했다.** 14.49 의 "예상 −5.5% 대 관측 −3.0%" 가 그 32% 다.
#
# 선형이라 보면 영점 (14.51 의 3시드 값에서):
#   실측 고전력 f = 47.5/63.8 = **0.745**
#   합성 편향(on) 영점  오븐 0.77 · 드라이 0.71 · 핫플 0.60 · 포트 0.32   (중앙 0.66)
#   ⇒ **f = 0.5 와 0.7 을 건다.** 둘과 이미 있는 f=0(983573 off)·f=1(983573 on)로
#     네 점이 생겨 **선형인지 자체가 검정된다.**
#
# ⚠ 선형은 **가정**이다. 네 점이 직선에 안 앉으면 그것이 결론이고, 그때는 영점을
#   내삽하지 말고 f 를 더 촘촘히 걸어야 한다.
# ⚠ 983573 의 두 팔과 **같은 씨앗·같은 캐시·같은 나머지 설정**이어야 네 점을 이을 수 있다.
#   `--harm-vnorm-frac` 이 그것 하나만 가른다 (`k.pow(1.0)` 은 항등이라 f=1 은 983573 on 과 같다).
# ⚠ 같이 고칠 것이 하나 더 있는데 **이 판에 섞지 않는다** ([[count-how-many-things-differ-before-attributing]]):
#   `vrel` 이 타깃 사이클이 아니라 **10초 창 평균**이다 (`run_train_cnn.py:312`).
#   고전력 창에서 V 를 +0.28~0.30% 높게 읽어 `L_harm` 이 전력을 그만큼 더 요구한다 —
#   방향은 과대예측 쪽이고 크기는 앵커의 1/17 이다 (`run_diag_vrel.py`). **따로 건다.**
#
#SBATCH -J nilm_vfrac
# ⚠ A6000 Ada/A6000 우선이되 **밀리면 넓혀라** (HPC_RULES §3, A10 은 1.46배 느릴 뿐).
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
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문: **함수가 아니라 배선을 잰다** (983506 이 45분을 버렸다) ──────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-vnorm-frac" || {
  echo "--harm-vnorm-frac 이 없다 — 코드를 안 올렸다"; exit 1; }
# `run_gate_vanchor` 가 `NILMLoss` 를 실제로 지어 f=0/0.5/0.7/1.0 의 배수가
# `(V_CENTER/V_적합)^(e·f)` 와 맞는지 끝단에서 본다. f=0 이 **정확히** 1.0 인 것도 본다.
$PY -X utf8 -m src.run_gate_vanchor || { echo "vanchor 배선 관문 실패"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다"; exit 1; }

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-frac 0.5 \
    --tag cnn_vfrac_50_s${SEED} > "results/cnn_vfrac_50_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-frac 0.7 \
    --tag cnn_vfrac_70_s${SEED} > "results/cnn_vfrac_70_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== f=0.5 ==="; tail -32 "results/cnn_vfrac_50_s${SEED}.log"
echo "===================================================================================="
echo "=== f=0.7 ==="; tail -32 "results/cnn_vfrac_70_s${SEED}.log"
ls -la results/cnn_vfrac_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 (로컬에서, 실측 복합 5파일):"
echo "  python -X utf8 src/run_diag_hipow.py --ckpt \\"
echo "      results/cnn_vanch_off_s{0,1,2}.pt results/cnn_vfrac_50_s{0,1,2}.pt \\"
echo "      results/cnn_vfrac_70_s{0,1,2}.pt results/cnn_vanch_on_s{0,1,2}.pt"
echo "  ★ **핫플통전·P>1500** 잔차 중앙이 f 에 대해 직선인가, 그리고 영점이 어디인가."
echo "     f=0 +47.5 ± 4.1 · f=1 −16.3 ± 7.5 (14.51 실측). 선형이면 f=0.745 에서 0."
echo "  ⚠ 네 점이 직선에 안 앉으면 그것이 결론이다 — 내삽하지 마라."
exit $RC
