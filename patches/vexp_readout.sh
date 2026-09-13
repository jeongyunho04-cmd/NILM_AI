#!/bin/bash
# 14.9 판정 — 979865 의 두 체크포인트를 받아 **실측**으로 잰다. 로컬에서 돌린다.
#
# ⚠ 합성 홀드아웃으로 판정하지 마라. 합성 창은 전압이 녹화 전압과 같아 α≈1 이라
#   k 가 절반 이하로 틀려도 잔차가 −0.7~−1.2% 로 깨끗하다 (14.6 ④).
# ⚠ 개선은 **test_5·test_2 에서만** 보여야 정상이다. 나머지 셋이 움직이면 잘못된 것이다.
#
#   bash patches/vexp_readout.sh          # 받고 재기
#   bash patches/vexp_readout.sh --local  # 이미 받았으면 재기만
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="python -X utf8"
LOG="results/_vexp_readout.log"

if [ "${1:-}" != "--local" ]; then
  echo "=== 체크포인트 받기 (2.5MB 씩)"
  for T in cnn_vexp_ctl cnn_vexp_on; do
    scp "gate1_External:~/NILM_AI/results/${T}.pt" "results/${T}.pt" || { echo "못 받았다: $T"; exit 1; }
  done
fi
ls -la results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt

{
echo "################ 14.9 판정 $(date '+%F %H:%M:%S') ################"
echo
echo "#### ① 전압 지수가 2 에 앉았나 (run_gate_vexp)"
echo "     기준선 cnn_v37: 저항 k=0.74 · SMPS k=−1.33  (물리 2 / 0)"
for C in results/cnn_v37.pt results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt; do
  echo; echo "--- $C"
  $PY src/run_gate_vexp.py --ckpt "$C" 2>&1 | grep -E "저항 4종|SMPS 3종|끔|켬|더해진|끄면 항등"
done

echo
echo "#### ② 실측 파일별 잔차 (run_plot_real) — 부호는 관측−예측, 음수가 과예측"
echo "     기준선 cnn_v37: test_1 +1.5 · test_2 +15.2 · test_3 −17.3 · test_4 +2.5 · test_5 −112.5 W"
for C in results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt; do
  echo; echo "--- $C"
  $PY src/run_plot_real.py --ckpt "$C" --tag "$(basename "$C" .pt)" 2>&1 \
    | grep -vE "저장 |Warning|warn" | tail -8
done

echo
echo "#### 읽는 법"
echo "  통과  test_5 의 과예측(−112.5W)이 B 에서 크게 줄고, test_1·3·4 는 ±5W 안에서 그대로"
echo "  실패  test_1/3/4 가 같이 움직인다   -> 지수가 아니라 다른 것을 바꾼 것이다"
echo "  실패  A 와 B 가 같다               -> --vexp 가 안 걸렸다 (①의 k 를 봐라)"
echo "  ⚠ A(대조)와 cnn_v37 은 자료 파이프라인이 달라 절대값을 나란히 놓으면 안 된다."
echo "     재는 것은 **A 대 B** 하나다. cnn_v37 은 ①의 k 참조용이다."
} 2>&1 | tee "$LOG"
echo
echo "기록: $LOG"
