#!/bin/bash
# 14.33 ⓑ 검사 — v32t 가 의도대로 흔들리나. 모델 없이 자료만.
#SBATCH -J nilm_vtd
#SBATCH -p cpu1,cpu2,gpu6,gpu2,gpu4,gpu5,gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
./env/bin/python -X utf8 -m src.run_diag_vtexdrift \
    --caches cache/train60_v32 cache/train60_v32t \
    --apps beam_projector laptop_charger minipc electiric_kettle oven hotplate hair_dryer
echo "끝 $(date '+%F %H:%M:%S')"
