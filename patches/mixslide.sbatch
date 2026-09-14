#!/bin/bash
# 14.40 ① — 상태 혼합 붕괴가 **합성 홀드아웃에도** 있는가. 참 `y_state` 로 잰다.
#SBATCH -J nilm_mixsl
#SBATCH -p gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). 짧을수록 backfill 이 잘 문다 —
#   `-t 01:15` 로 3.2배를 불렀다가 시드 1·2 가 한 시간을 섰다 (983243, 2026-09-14).
#   ⚠ `squeue --start`·`sbatch --test-only` 는 **backfill 을 모델링하지 않는다** —
#     모양을 바꿔도 같은 날짜가 나온다. 그 값으로 '자리가 없다' 고 읽지 마라.
#SBATCH -t 00:10:00     # 실측 3분(983151)
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
test -d processed_data/holdout60_v32 || { echo "홀드아웃이 없다"; exit 1; }
$PY -X utf8 -m src.run_diag_mixslide \
    --ckpt results/cnn_vswap_on_s0.pt results/cnn_vncls_res_s0.pt \
           results/cnn_vncls_res_s1.pt results/cnn_vncls_res_s2.pt \
           results/cnn_sigc_vn_s0.pt
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
