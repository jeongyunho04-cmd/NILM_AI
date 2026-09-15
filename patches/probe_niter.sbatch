#!/bin/bash
# 14.103 — 고정점 반복 수를 줄여도 되나. `run_coupling_check` 로 n_iter 를 훑는다.
# 비용은 n_iter 에 정비례한다 (호출당 n_iter x SMPS수 번의 simulate_true).
#SBATCH -J nilm_ni
#SBATCH -p cpu2,cpu1,gpu6,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 00:50:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %T')"
for NI in 1 2 3 4 6; do
  echo "============================ n_iter=$NI (외삽 켬) ============================"
  S=$(date +%s)
  $PY -X utf8 -m src.run_coupling_check --windows 300 --n-iter $NI --seed 0 2>&1 | tail -14
  echo "  [벽시계 $(( $(date +%s) - S ))초]"
done
echo "============================ n_iter=4 외삽 **끔** (대조) ============================"
$PY -X utf8 -m src.run_coupling_check --windows 300 --n-iter 4 --no-extrapolate --seed 0 2>&1 | tail -14
echo "끝 $(date '+%F %T')"
