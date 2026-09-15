#!/bin/bash
# 14.125 — 부푼 슬롯이 **공유 몸통**에 얼마나 큰 기울기를 되돌리나.
#
# 로컬에서 야코비안만 쟀더니 빔은 전체의 **3%** 뿐이었다 (0.089 / 2.904).
# 그런데 학습 기울기는 `Huber'(잔차) x 야코비안` 이고, 빔은 **잔차 쪽이 부풀어**
# 있었다 (1% 혼합 오차가 참값의 13.1%). 그 곱을 재려면 캐시의 `y_power` 가 있어야 한다.
# ⚠ 로그인 노드에서 무거운 계산 금지 (HPC_RULES 0) — 그래서 작업으로 던진다.
#SBATCH -J nilm_grad
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
./env/bin/python -X utf8 gradprobe.py 3072
