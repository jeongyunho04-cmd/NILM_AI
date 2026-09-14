#!/bin/bash
# 14.43 — **실측의 '미래 반파 증거' 가 합성의 것과 무엇이 다른가.** 참 y_state 로 쪼갠다.
#
# **실측 기준선은 이미 쟀다** (조인 자: 오븐 라벨 ON · 다른 저항 전부 OFF · P>=0.9·V²/R):
#     오븐 단독통전 336창 · 붕괴 5.7~8.3%
#       │ 미래 전환 있음  30창 -> **붕괴 93.3 / 50.0 / 73.3% = 72 ± 22%**
#       │ 미래 전환 없음 306창 -> **붕괴 0.0 / 1.3 / 1.0% = 0.8%**
#     반파 증거 크기: 타깃 |I2| 0.213 · 미래 |I2| 중앙 2.060 (최대 3.795) · 미래 구간 점유 56%
#   ⇒ **실측 오븐 붕괴는 전부 미래 전환 하나로 설명된다** (8.9% x 72% = 6.4% ≈ 7.3%).
#
# **반증 가능한 예측**:
#   ① 합성에서도 '미래 전환 있음' 창의 붕괴가 **72% 근처**면, 생성기는 그 배치를 만드는데
#      **빈도가 달라** 전체 수치가 갈린 것이다 (합성 4.3% 대 실측 1.5%... 그런데 합성이 더
#      많으므로 이 경우 합성 전체 붕괴가 **더** 높아야 하는데 0.7% 다 -> 모순. 즉 ① 은
#      이미 거의 죽어 있다).
#   ② **합성에서 '미래 전환 있음' 창의 붕괴가 낮으면**(예: 5% 아래) 같은 배치를 모델이
#      합성에서는 **제대로 처리한다**는 뜻이다. 그러면 다른 것은 **증거의 성질**이다 —
#      `_stats` 줄의 |I2| 크기·미래 구간 점유를 실측(2.060 / 최대 3.795 / 56%)과 맞대라.
#   ③ **동시성 줄**을 봐라 — 오븐 HEATING 과 드라이 약풍(반파)이 **동시에** 있는 창이
#      합성에 몇 % 인가. 0 에 가까우면 모델은 그 조합을 **배운 적이 없다**.
#   ④ ⚠ 합성 붕괴 표본이 작을 수 있다 (참 통전 1,235창 중 미래 전환은 일부). n 을 먼저 봐라.
#
#SBATCH -J nilm_hwf
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
test -d processed_data/holdout60_v32 || { echo "홀드아웃이 없다"; exit 1; }
$PY -X utf8 -m src.run_diag_hwfuture --holdout processed_data/holdout60_v32 \
    --ckpt results/cnn_vncls_res_s0.pt results/cnn_vncls_res_s1.pt results/cnn_vncls_res_s2.pt
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
