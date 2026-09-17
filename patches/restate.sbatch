#!/bin/bash
# 14.397 — 판정줄 `re_on_median` 을 **상태로 가른다** (사용자 질문).
# 계산 노드에서 돈다 (HPC_RULES §0 — 로그인 노드에서 무거운 계산 금지).
#SBATCH -J nilm_rest
#SBATCH -p cpu1,cpu2,gpu6,gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).
for CK in results/cnn_comb_s0.pt results/cnn_pjit_s0.pt; do
  [ -f "$CK" ] || { echo "없다: $CK"; continue; }
  echo "=============================== $CK"
  $PY -X utf8 -m src.run_diag_restate --ckpt "$CK" \
      --holdout processed_data/holdout60_v49
done
echo "끝 $(date '+%F %H:%M:%S')"
