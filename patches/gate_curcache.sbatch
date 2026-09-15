#!/bin/bash
# 14.99 관문 — current() 메모이즈가 굽기 산출물을 **바이트 한 톨도** 안 바꾸는지.
#SBATCH -J nilm_cc
#SBATCH -p gpu4,gpu3,gpu6,cpu1,cpu2,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --mem=60G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
echo "노드 $(hostname) · $(date '+%F %T')"
GATE_WORKERS=${SLURM_CPUS_PER_TASK} ./env/bin/python -X utf8 src/run_gate_curcache.py
echo "끝 $(date '+%F %T')"
