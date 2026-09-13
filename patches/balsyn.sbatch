#!/bin/bash
# 13.84.43 — 짝 점수가 가짜 전이를 거르는가. 작은 합성 홀드아웃으로 제대로 센다.
#
# 13.84.42 가 실측에서 짝 AUC 0.819 · 켜짐 −ΔP 0.924 를 냈는데 **거짓 전이가 2개뿐**이라
# 못 닫았다 ([[one-seed-cannot-rank-small-classes]]). 13.84.33 과 같은 수 — 합성은 표본이 무제한이다.
#
# ⚠ 작은 작업이다. 기록 400개면 1.3GB · 굽기 1분 안쪽이고, 전이는 수백~수천 개 나온다.
#   30,000 기록짜리 학습 캐시(92GB)를 다시 구울 이유가 없다.
# ⚠ 972281 이 도는 중이므로 **그 노드를 건드리지 않는다** — 따로 작은 자리를 받는다.
#   자체 제한 3노드 안이다 (지금 1노드 사용 중).
#SBATCH -J nilm_balsyn
#SBATCH -p gpu6,gpu1,gpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
# 작게 부른다 — §2 의 backfill. 굽기만 병렬이고 채점은 GPU 한 장이다.
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

RECORDS=${RECORDS:-200}
SHM="/dev/shm/$USER/seqraw_bal"

NCPU=$(nproc)
echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · nproc $NCPU · 기록 $RECORDS"
nvidia-smi --query-gpu=index,name --format=csv,noheader

cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에, §7-3) ───────────────────────────────────────────────────────
$PY -X utf8 -c "import src.run_diag_balsyn" 2>/dev/null || { echo "run_diag_balsyn 을 안 올렸다"; exit 1; }
test -f results/seq_h38_base.pt || { echo "results/seq_h38_base.pt 가 없다"; exit 1; }
test -f results/cnn_v37.pt      || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# ── ① 작은 합성 홀드아웃 ────────────────────────────────────────────────────
# ⚠ 씨앗을 학습 캐시(0)와 **다르게** 준다 — 같은 기록을 다시 보면 안 된다.
mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen v36 --workers $(( NCPU - 2 )) --seed 777
du -sh "$SHM"

# ── ② 방출 배율 α 훑기 ───────────────────────────────────────────────────────────
# 체크포인트는 `seq_h38_base`(v36 캐시로 학습한 대조판)를 쓴다 — 캐시 생성기도 v36 으로 맞췄다.
$PY -X utf8 -m src.run_diag_balsyn --cache "$SHM" --ck results/seq_h38_base.pt \
    --records "$RECORDS" | tee "results/balsyn_${SLURM_JOB_ID}.log"

# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5).
echo "끝 $(date '+%F %H:%M:%S')"
