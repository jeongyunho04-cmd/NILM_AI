#!/bin/bash
# 14.33 ⓑ 검사 ② — v32t 의 **전압 채널**이 실제로 더 흔들리나. 모델 없이 자료만.
#SBATCH -J nilm_vtv
#SBATCH -p cpu1,cpu2,gpu6,gpu2,gpu4,gpu5,gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=100G
#SBATCH -t 00:25:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
./env/bin/python -X utf8 -m src.run_diag_vtexvolt --n 20000
echo "끝 $(date '+%F %H:%M:%S')"
