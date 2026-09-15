#!/bin/bash
#SBATCH -J nilm_sdeg
#SBATCH -p gpu3,gpu4,gpu5
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
./env/bin/python -X utf8 -m src.run_diag_synthdegen \
    results/cnn_pcap_s0.pt results/cnn_pcap_s1.pt results/cnn_pcap_s2.pt
