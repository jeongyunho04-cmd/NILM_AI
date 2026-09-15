#!/bin/bash
# seg_s 별 굽기 비용을 **잰다** (14.96). 250창 토막 하나로 core-s/window 를 낸다.
#SBATCH -J nilm_segp
#SBATCH -p cpu2,cpu1,gpu6,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=60G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
C=${SLURM_CPUS_PER_TASK}
echo "노드 $(hostname) · 코어 $C · $(date '+%F %T')"
COMMON="--windows 300000 --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95 --harmonic-z"
for SEG in 0 10 5 3; do
  OUT="/tmp/segprobe_$SEG"
  rm -rf "$OUT"
  if [ "$SEG" = "0" ]; then VT=""; else VT="--vtex-step-s $SEG --vtex-seg-s $SEG"; fi
  S=$(date +%s)
  $PY -X utf8 -m src.run_build_traincache --out "$OUT" --shard "0/1200" \
      $COMMON $VT --workers "$C" > /dev/null 2>&1
  E=$(date +%s)
  N=$($PY -X utf8 -c "import json;print(json.load(open('$OUT/meta.json'))['n_windows'])")
  echo "seg_s=$SEG  창 $N  벽시계 $((E-S))초  core-s/window $($PY -X utf8 -c "print('%.3f'%(($E-$S)*$C/$N))")"
  rm -rf "$OUT"
done
echo "끝 $(date '+%F %T')"
