#!/bin/bash
# 14.33 ⓒ — `sig ∝ 1/V` 보정을 **유도로 정확한 곳에만** 남긴다. 시드 셋 짝비교.
#
# 사용자: *"근데 sig ∝ 1/V 이 맞는 원리야? SMPS 는 그 관계가 상당히 복잡할 건데"*
#
# 두 절반의 근거가 다르다:
#   **저항 — 유도다.** `I_h = V_h/R` · `P = V²/R` 이라 전압 *모양*이 고정된 채 크기만
#     변하면 **모든 차수에서** `I_h/P ∝ 1/V` 다. 적합이 필요 없다.
#   **SMPS — h1 에서만 선다.** `P ≈ V·I₁·cosφ` 이므로 `I₁/P ≈ 1/(V·cosφ)` 는 거의
#     **항등식**이다 (기기 물리와 무관하게 −1 이 나온다. 조건수가 엉망인 저항 녹화에서도
#     h1 만은 −0.51~−1.06 으로 얌전하다. R²=1.00 이 그 증거다).
#     `h>=3` 은 전압이 오르면 **도통각이 좁아져** 고차 함량이 오르므로 `V^e` 꼴이 아니다.
#
# 실측으로 차수별 지수를 재 봤다 (10초 블록 · `log(I_h/P) ~ e·logV + b·t`, 시간 통제):
#   SMPS 는 조건화가 좋다 (std logV 0.027~0.028 · 전압 폭 6.4~7.3% · 130~236블록).
#   그런데 **녹화 사이에 재현이 안 된다** —
#       충전기 h1  −0.48 / −1.05 / −1.88 / −1.45      (폭 1.40)
#       충전기 h11 +47.4 / −12.5 / −32.2 /  −5.9      (폭 **79.5**)
#       미니PC h1  −0.83 / −3.06 / +4.62 / −4.54      (폭 **9.2**, R² 0.04)
#   통합 적합은 "h11 에서 +10~+13" 을 내지만 그것은 **녹화 간 차이를 전압 기울기로
#   읽은 값**이다. 규칙 1 로 버린다. ⇒ SMPS 의 −1 은 **확인도 반증도 안 된 가정**이다.
#   (저항 녹화는 std logV 0.0008~0.0076 이라 애초에 못 읽는다 — 유도가 답이다.)
#
# 그런데 14.29 는 왜 크게 이겼나: 이득이 **저항에 몰려 있다**. 합성 홀드아웃 저항 4종
#   12/12 칸 개선, 실측 이득도 test_5(저항 파일)에서 나왔다. 그리고 14.29 ④ 가 적어 둔
#   대가가 정확히 SMPS 쪽이다 — F1 −0.0029/−0.0003/−0.0015 · 충전기 MAE
#   +0.06/+0.31/+0.18W, **셋 다 같은 부호**.
#
#   A 대조   그대로 (전 분류에 −1/0)          = 지금 채택본
#   B 제한   --harm-vnorm-classes RESISTIVE   -> 프로젝터·충전기·미니PC 지수 0
#            (모터는 원래 0 이라 안 바뀐다. 저항 넷은 −1 그대로.)
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① **저항 4종 지표가 안 움직여야 한다** — 저항 지수는 양쪽 −1 로 같다.
#      움직이면 배선이 샌 것이다.
#   ② 전압 지수(상태 라벨로 잰 것)가 B 에서도 **2.0 근처**여야 한다. 그 축은 저항이다.
#   ③ SMPS 3종 신원 · 충전기 MAE · F1 평균이 B 에서 **좋아지거나 같아야** 한다.
#      나빠지면 SMPS 의 −1 이 실제로 쓸모가 있었다는 뜻이고, 그때는 그대로 둔다.
#   ④ ⚠ 셋 다 0 근처일 수도 있다 (효과가 작아서). 그러면 "근거 없는 절반을 뺐는데
#      손해가 없다" 이므로 **물리를 이유로** B 를 택한다 (14.27 ⑦ 과 같은 논리).
#
# ⚠ 시드 하나로 읽지 마라 — 14.29 가 시드 하나로 +0.018 을 읽었다가 셋으로 −0.0031 이었다.
# ⚠ 원리적으로 옳은 길은 지수가 아니라 **회로 야코비 `∂I_h/∂V_h`** 다 (14.11 이 그것으로
#   표류 분산 65%를 자유도 0 에 설명했다). 이 판은 그 전에 "근거 없는 절반을 빼면
#   어떻게 되나" 를 싸게 재는 것이다.
#
#SBATCH -J nilm_vncls
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 01:15:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 캐시 $CACHE"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문 ──────────────────────────────────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-vnorm-classes" || {
  echo "--harm-vnorm-classes 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src import env_guard
from src.run_train_cnn import _vnorm_exp
from src.run_baseline import S_I
apps = sorted(S_I)
a = _vnorm_exp(apps); b = _vnorm_exp(apps, 'RESISTIVE')
print('관문 A:', ' '.join('%s=%+.0f' % (x[:4], e) for x, e in zip(apps, a)))
print('관문 B:', ' '.join('%s=%+.0f' % (x[:4], e) for x, e in zip(apps, b)))
chg = [x for x, u, v in zip(apps, a, b) if float(u) != float(v)]
want = ['beam_projector', 'laptop_charger', 'minipc']
if sorted(chg) != sorted(want):
    sys.exit('바뀌어야 할 기기가 %s 인데 %s 가 바뀐다' % (want, chg))
for x in ('electiric_kettle', 'oven', 'hotplate', 'hair_dryer'):
    if float(b[apps.index(x)]) != -1.0:
        sys.exit('저항 %s 의 지수가 %.1f — B 에서도 −1 이어야 한다' % (x, float(b[apps.index(x)])))
print('관문: SMPS 셋만 0 으로 · 저항 넷은 −1 유지 ✅')
"

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"
# `--harm-sig-vnorm` 은 14.29 에서 기본 켜짐. 두 팔 다 켜진 채로 간다.

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
# ⚠ 14.37 에서 `--harm-vnorm-classes` **기본이 `RESISTIVE` 로 바뀌었다.** A 팔은
#   전 분류를 **명시로** 줘야 한다 — 안 주면 두 팔이 같아진다. 그리고 앞의 관문은
#   `_vnorm_exp(apps)` 의 **함수 기본값**을 보므로 이 실수를 못 잡는다 (argparse
#   기본값이 바뀐 것이라 자리가 다르다).
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes "" \
    --tag cnn_vncls_all_s${SEED} > "results/cnn_vncls_all_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE \
    --tag cnn_vncls_res_s${SEED} > "results/cnn_vncls_res_s${SEED}.log" 2>&1 &
B=$!
echo "A(전 분류) $A · B(저항만) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: 전 분류 ==="; tail -22 "results/cnn_vncls_all_s${SEED}.log"
echo "=== B: 저항만 ===";  tail -22 "results/cnn_vncls_res_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 ==="
$PY -X utf8 -m src.run_score_holdout \
    --ckpt results/cnn_vncls_all_s${SEED}.pt results/cnn_vncls_res_s${SEED}.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_vncls_*_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬 실측**으로, 같은 자 하나로:"
echo "  python -X utf8 src/run_diag_rollback.py --ckpt results/cnn_vncls_all_s0.pt results/cnn_vncls_res_s0.pt"
echo "  ⚠ 예측 ① — **저항 4종 신원이 안 움직여야 한다**. 움직이면 배선이 샌 것이다."
exit $RC
