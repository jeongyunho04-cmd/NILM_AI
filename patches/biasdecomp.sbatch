#!/bin/bash
# 14.49 — 합성 홀드아웃의 기기별 편향을 **곱으로 분해**. 참값이 정확하다.
#SBATCH -J nilm_bdec
#SBATCH -p gpu2,gpu6,gpu4,gpu5,gpu3,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 00:15:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
./env/bin/python -X utf8 -m src.run_diag_biasdecomp \
    --ckpt results/cnn_vncls_res_s0.pt results/cnn_vncls_res_s1.pt results/cnn_vncls_res_s2.pt
echo "끝 $(date '+%F %H:%M:%S')"
