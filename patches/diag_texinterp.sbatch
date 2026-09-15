#!/bin/bash
# 14.101 — 연속 텍스처 토막을 양 끝만 풀고 보간해도 되나.
#SBATCH -J nilm_ti
#SBATCH -p cpu1,cpu2,gpu6,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
echo "노드 $(hostname) · $(date '+%F %T')"
./env/bin/python -X utf8 src/run_diag_texinterp.py --nseg 20 --runs 40 --step-s 3
echo "끝 $(date '+%F %T')"
