#!/bin/bash
# 14.40 ① — 상태 혼합 붕괴가 **합성 홀드아웃에도** 있는가. 참 `y_state` 로 잰다.
#SBATCH -J nilm_mixsl
#SBATCH -p gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
test -d processed_data/holdout60_v32 || { echo "홀드아웃이 없다"; exit 1; }
$PY -X utf8 -m src.run_diag_mixslide \
    --ckpt results/cnn_vswap_on_s0.pt results/cnn_vncls_res_s0.pt \
           results/cnn_vncls_res_s1.pt results/cnn_vncls_res_s2.pt \
           results/cnn_sigc_vn_s0.pt
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
