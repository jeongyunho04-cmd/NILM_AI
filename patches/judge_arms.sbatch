#!/bin/bash
# 14.416 — **실측 판정을 HPC 에서 돌린다** (사용자 지시 2026-09-18).
#
#   BASE=v52base ARMS="v52dp v52mg" sbatch --export=ALL,BASE,ARMS patches/judge_arms.sbatch
#
# 왜 옮겼나
#   팔이 셋씩 늘면서 이 노트북에서 판당 `forward_file` **6.4초** x 파일 6 x 판 15
#   = **10분**이 되고, 그동안 노트북이 묶인다. CUDA(RTX 2050)는 1.45배뿐이라 답이 아니다
#   (창 512개 묶음이라 GPU 가 논다). 사용자: *"여기 노트북은 성능이 좀 제한적이네"*.
#
# ⚠⚠ **`composite_eval` 업로드 금지는 폐지됐다** (HPC_RULES §0, 2026-09-18).
#   원래 논거가 *"학습에 안 들어가니 둘 이유가 없다"* 는 **필요성**이었는데 필요가 생겼다.
#   사용자 확인: *"무슨 기밀정보도 아니고"*.
#   ⚠ 그래도 **상주시키지 않는다** — 이 스크립트가 끝나면 `trap` 으로 지운다.
#     공용 계정이라 예의 문제다. 채점 자료는 **필요한 순간에만** 거기 있는다.
#
# ⚠ 봉인은 그대로다 — `evaluation.sealing` 이 `test.csv` 를 어디서도 못 읽게 한다.
#
#SBATCH -J nilm_judge
#SBATCH -p cpu1,cpu2,gpu6,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

BASE=${BASE:?BASE 를 달라 (기준선 팔 이름)}
ARMS=${ARMS:?ARMS 를 달라 (공백으로 구분)}
SEEDS=${SEEDS:-"0 1 2 3 4"}
EVAL="processed_data/composite_eval"

# ⚠⚠ **반납은 선택이 아니다** (HPC_RULES §0 — 공용 계정).
#   `trap` 은 정상 종료·오류·시간초과(SIGTERM) 전부에서 돈다.
cleanup() {
  echo "[정리] $EVAL 반납"
  rm -rf "$EVAL"
  ls -d "$EVAL" 2>/dev/null && echo "⚠ 아직 남아 있다 — 손으로 지워라" || echo "[정리] 지웠다"
}
trap cleanup EXIT

#: ⚠ torch 는 기본으로 코어를 다 안 쓴다. 명시한다 (HPC_RULES: gpu1·gpu6 은 40 이 상한).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "노드 $(hostname) · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"
echo "기준선 $BASE · 팔 $ARMS · 씨앗 $SEEDS"

# ── 관문 ⓪ 채점 자료가 올라왔나 ─────────────────────────────────────────────
#   ⚠ 이걸 안 보면 파일이 0개인 채로 "전부 NaN" 표가 나온다 — 조용한 실패다.
test -d "$EVAL" || { echo "✖ $EVAL 이 없다. 먼저 올려라:
  tar cf - $EVAL processed_data/real_events.json | ssh gate1_External 'cd ~/NILM_AI && tar xf -'"; exit 1; }
test -f processed_data/real_events.json || { echo "✖ real_events.json 이 없다"; exit 1; }
N=$(ls "$EVAL"/*.npz 2>/dev/null | wc -l)
echo "  채점 파일 ${N}개"
[ "$N" -ge 5 ] || { echo "✖ 채점 파일이 ${N}개뿐이다"; exit 1; }

# ── 관문 ① 체크포인트가 다 있나 ────────────────────────────────────────────
MISS=""
for nm in $BASE $ARMS; do
  for s in $SEEDS; do
    [ -f "results/cnn_${nm}_s${s}.pt" ] || MISS="$MISS cnn_${nm}_s${s}.pt"
  done
done
[ -z "$MISS" ] || { echo "✖ 없는 체크포인트:$MISS"; exit 1; }
echo "  체크포인트 전부 있다"

echo "=== 판정 $(date '+%H:%M:%S')"
time $PY -X utf8 -m src.run_judge_arms --base "$BASE" --arms $ARMS --seeds $SEEDS --dev cpu
echo "=== 끝 $(date '+%F %H:%M:%S')"
