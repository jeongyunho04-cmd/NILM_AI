#!/bin/bash
# 14.151 — 연결점 여섯이 각각 `p_raw` 에 무엇을 하나. **합성 홀드아웃**(학습 분포)에서 잰다.
#SBATCH -J nilm_gpath
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_diag_gatepath \
    --holdout processed_data/holdout60_v32h --n 6000 --n-grad 1024 \
    --ckpt results/cnn_pcap_s0.pt results/cnn_hv2_s0.pt
echo "끝 $(date '+%F %H:%M:%S')"
