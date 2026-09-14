#!/bin/bash
# 14.32 — **저항 조합 맞바꿈 `L_swap` 을 1단계에.** 시드 셋 짝비교.
#
# 무엇을 고치려는가 (14.30~14.31):
#   `test_2` 에는 오븐이 **아예 없는데**(참 0%) 게이트가 16% 선다. 231~260초에서
#   참 포트 1430W 를 오븐이 1200W 로 가져간다. 고조파로는 못 가른다 —
#   오븐↔포트 지문 각도 **1.91°**, 겹친 구간에서 몫을 통째로 뒤집어도 모양이
#   **0.1~0.2%** 밖에 안 변한다 (지문 표류가 분 단위로 9~11% 다).
#   10초 창 듀티 리플도 못 가른다 (통전 중 CV 오븐 0.002 · 포트 0.001).
#
#   **컨덕턴스로는 갈린다.** 새 계측기 격리 녹화:
#       전기포트 35.65Ω (p5~p95 35.21~35.73, 폭 1.5%)
#       오븐     40.15Ω (39.64~40.57, 2.3%)   2번 녹화 40.12Ω — **녹화 사이 0.08%**
#       핫플     101.13Ω / 100.87Ω            — 녹화 사이 0.26%
#       드라이기 강 53.30Ω (0.4%) · 약(반파) 106.31Ω (0.8%)
#   두 기기 구간이 **겹치지 않는다.** 14.31 이 이 축을 후처리로 써 봤는데 판정 줄
#   −0.005~−0.016 · 저항 신원 −0.006~−0.049 였다 — 창 하나만 보고 이산 결정을
#   내려 되돌릴 길이 없기 때문이다. **손실로 넣으면 기울기라 되돌려진다.**
#
# 왜 `L_swap` 이고 `L_res` 가 아닌가 (기울기를 직접 뽑아 쟀다):
#   `L_res`   비저항 일곱의 **게이트**로 못 박은 것의 25~37% 크기로 샌다.
#             대기 헤드에는 아홉 전부 똑같은 기울기가 간다.
#   `L_swap`  조합 탐색이 argmin 이라 목표가 상수, BCE 는 저항 열에만
#             -> 비저항 일곱 **정확히 0.000e+00**
#
# 통전 상태 (`--res-cond-state oven:2,hotplate:2`):
#   `σ(on)` 을 "통전" 으로 읽는 것이 포트·드라이기에는 맞고 오븐·핫플에는 틀리다.
#   실측에서 오븐 게이트가 켜진 **2322창 중 997창만** 통전이고, 나머지는 전력
#   중앙 **16.0W** 인 팬·조명이다. 구분 안 하면 조합 탐색이 "오븐 끔" 을 정답으로
#   내 `L_on`(y_on=1) 과 정면으로 싸운다.
#
#   A 대조   --w-swap 0       (기본. 완전히 꺼짐)
#   B 고침   --w-swap 0.3     나머지 **열 전부 같다**
#
# ⚠ **`--w-swap 0.3` 은 근거 있는 값이 아니다.** `w.on` 과 같은 크기로 잡았을
#   뿐이다. 이 판이 재는 것은 "이 축이 듣나" 이지 최적 가중이 아니다.
#   듣는다는 신호가 나오면 그 다음에 0.1 / 0.3 / 1.0 을 훑는다.
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① `swap_frac` 이 0 이 아니어야 한다. 0 이면 항이 안 걸린 것이고 나머지는
#      볼 필요가 없다 — 학습 로그가 매 에포크 찍는다.
#   ② **SMPS 3종 신원이 안 움직여야 한다** — 기울기가 0 이므로 움직이면
#      목표 오염(`g_need` 에 든 `p_other`)이거나 배선이 샌 것이다.
#   ③ 개선은 **저항이 둘 이상 겹친 창**에서. `test_4`(SMPS 전용)가 움직이면 누수다.
#   ④ 오븐 정밀도가 오르고 `test_2` 의 유령 오븐(참 0W, 지금 46.1W)이 줄어야 한다.
#   ⑤ ⚠ 판정 줄이 내려갈 수도 있다 — 14.31 의 후처리가 그랬다. 그러면 이 축은
#      손실로도 안 되는 것이고, 그것도 결론이다.
#
# ⚠ 시드 하나로 읽지 마라. 14.29 가 시드 하나로 +0.018 을 읽었다가 셋으로
#   −0.0031±0.0042 였다 ([[one-seed-cannot-rank-small-classes]]).
#
#SBATCH -J nilm_swap
# A6000 Ada / A6000 만 (14.28 과 같은 목록). a10·rtx3090 은 뺀다.
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 01:15:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
WSWAP=${WSWAP:-0.3}
SEED=${SLURM_ARRAY_TASK_ID}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 캐시 $CACHE · w_swap $WSWAP"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문은 전부 앞에 (HPC_RULES §7-3) ─────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--w-swap" || {
  echo "run_train_cnn 에 --w-swap 이 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 토막이다 — run_merge_traincache 로 먼저 이어붙여라')
if d.get('sibling_rotate'): sys.exit('v32 인데 sibling_rotate 가 있다 (14.4 위반)')
print('관문: 캐시 %d창 · v32 통짜' % d['n_windows'])
"
# 관문 다섯 — 비저항 기울기 0 · 통전 상태 · 조용한 0 · 끄면 항등 · 2단계 불변
# (`--skip-real` : 계산 노드에는 실측 npz 가 없을 수 있다)
$PY -X utf8 -m src.run_gate_swap1 --skip-real || {
  echo "run_gate_swap1 실패 — 배선이 깨졌다"; exit 1; }

# ── 두 판. 바뀌는 것은 **--w-swap 하나** ──────────────────────────────────
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"
# `--harm-sig-vnorm` 은 14.29 에서 **기본 켜짐**이 됐다. 두 팔 다 켜진 채로 간다.

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --w-swap 0 --tag cnn_swap_ctl_s${SEED} > "results/cnn_swap_ctl_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --w-swap "$WSWAP" \
    --tag cnn_swap_on_s${SEED} > "results/cnn_swap_on_s${SEED}.log" 2>&1 &
B=$!
echo "A(대조 w_swap=0) $A · B(w_swap=$WSWAP) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

# ── ① 항이 실제로 물었나 — **이것이 0 이면 나머지는 볼 필요가 없다** ──────
echo "=== 예측 ① swap_frac 이 0 이 아닌가 ==="
grep -o "swap [0-9.]*/[0-9.]*" "results/cnn_swap_on_s${SEED}.log" | head -3
grep -o "swap [0-9.]*/[0-9.]*" "results/cnn_swap_on_s${SEED}.log" | tail -3
NZ=$(grep -o "swap [0-9.]*/[0-9.]*" "results/cnn_swap_on_s${SEED}.log" \
     | awk -F/ '$2+0 > 0' | wc -l)
echo "  swap_frac > 0 인 에포크 ${NZ}개"
if [ "$NZ" -eq 0 ]; then
  echo "  ❌ **항이 한 번도 안 물었다.** tol(0.02) 이 너무 빡빡하거나 배선이 샜다."
  echo "     v_rms 가 타깃에 들어갔는지부터 볼 것 (`to_targets`)."
  RC=1
fi
echo "  ⚠ 대조 팔(A)에는 이 줄이 없는 것이 정상이다 (w_swap=0 이면 안 찍는다)"

echo "=== A: 대조 ===";  tail -22 "results/cnn_swap_ctl_s${SEED}.log"
echo "=== B: swap ===";  tail -22 "results/cnn_swap_on_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 ==="
$PY -X utf8 -m src.run_score_holdout \
    --ckpt results/cnn_swap_ctl_s${SEED}.pt results/cnn_swap_on_s${SEED}.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_swap_ctl_s${SEED}.pt results/cnn_swap_on_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬 실측**으로, 같은 자 하나로:"
echo "  python -X utf8 src/run_diag_rollback.py --ckpt results/cnn_swap_ctl_s0.pt results/cnn_swap_on_s0.pt"
echo "  python -X utf8 src/run_diag_ovenkettle.py --ckpt results/cnn_swap_on_s0.pt --stem test_2"
echo "  python -X utf8 -m src.run_plot_real --ckpt results/cnn_swap_on_s0.pt --tag swap"
exit $RC
