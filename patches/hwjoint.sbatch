#!/bin/bash
# 14.44 ③ — **결합**을 센다: 오븐이 통전 중 **이면서** 반파가 미래에 처음 켜지는 창.
#
# 14.44 에서 셋이 연달아 반증됐다 (생성기가 배치를 안 만드는 게 아니다):
#   전환 일반      합성 47.9% 대 실측 49.9%   — 같다
#   핫플 듀티      합성 100% 중앙 21 대 실측 100% 중앙 18 — 합성이 더 많다
#   날카로운 계수   합성 4.17% 대 실측 3.46%   — 합성이 더 많다
# 남은 것은 **결합**이다. 14.43 ② 의 표가 그것을 가리킨다 (오븐 통전 창 안에서만 갈렸다).
#
# **실측 기준선** (조인 자 · `--sharp` · 336창):
#   미래 전환 있음 **111창 -> 붕괴 25.2 / 17.1 / 22.5%**
#   미래 전환 없음  225창 -> 붕괴 **0.0 / 0.0 / 0.0%**
#   ⇒ 붕괴가 **전부** 그 111창 안에 있다.
#
# **반증 가능한 예측**:
#   ① 합성에서 그 결합 창이 실측 111/336=33% 보다 **훨씬 적으면** 원인은 결합 빈도다.
#   ② 결합 창이 비슷한데 **붕괴율만 낮으면** 빈도가 아니다 — 같은 배치를 합성에서는
#      제대로 처리한다는 뜻이고, 남는 차이는 증거의 **크기·지속**이다 (`_stats` 줄).
#   ③ ⚠ 합성의 '전환 없음' 붕괴가 실측처럼 0.0% 인지 봐라. 그래야 자가 같은 것을 잰다.
#
#SBATCH -J nilm_hwj
#SBATCH -p gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_diag_hwfuture --help | grep -q -- "--sharp" || {
  echo "--sharp 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_diag_hwfuture --sharp --holdout processed_data/holdout60_v32 \
    --ckpt results/cnn_vncls_res_s0.pt results/cnn_vncls_res_s1.pt results/cnn_vncls_res_s2.pt
echo "끝 $(date '+%F %H:%M:%S')"
