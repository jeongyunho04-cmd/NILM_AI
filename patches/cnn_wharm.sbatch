#!/bin/bash
# 14.21 — **기울기 희석이 전압 지수를 막는가.** `--w-harm` 짝 비교.
#
# 14.20 이 잰 것 (수렴한 `cnn_vexp_ctl` 한 점, 학습과 같은 s_state·상태별 지문):
#   L_harm   기울기 노름 0.10654 · 지수 압력 **+0.012 (중립)**
#   L_power  기울기 노름 0.00034 · 지수 압력 **+0.670**
#   가중 뒤 노름 비가 **31배** — 헤드 갱신을 L_harm 이 지배하고 그 힘은 지수가 아닌 곳에 쓰인다.
# 14.19 ① 이 그 반대 방향을 보였다 — 전력 손실 **만으로** 800스텝 돌리자(희석 제거)
#   지수가 0.79~1.14 -> 1.56~1.94 로 바로 갔다 (참값 1.46~1.92).
#
#   A 대조   --w-harm 0.1     (지금 본선 값)
#   B 낮춤   --w-harm 0.02    (5배 낮춤 — 노름 비 31 -> ~6)
#
# 반증 가능한 예측 (학습 전에 적는다):
#   ① B 의 저항 전압 지수가 A 보다 **뚜렷이 높다** (`run_gate_vexp` 의 k_끔, 또는 분포 안 탐침)
#      -> 안 오르면 희석 가설이 틀렸고 원인은 다른 데 있다
#   ② B 는 **SMPS 신원을 잃을 수 있다** — L_harm 이 그 일을 한다. 대가를 같이 잰다
#      ([[signature-unchanged-is-not-enough]], [[ambiguity-transfer-means-structural-bug]])
#   ③ 실측 test_5 의 과예측이 줄어야 이득이다. 합성 홀드아웃으로는 못 본다 (14.6 ④)
#
# ⚠ 쿼터가 풀렸다 (AIP 42G 반납, 38.31G/100G). `/dev/shm` 우회를 안 쓰고 **GPFS 캐시**를
#   쓴다 — 굽기는 cpu2 에서 따로 하고(14.13) 여기는 **학습만** 한다.
#SBATCH -J nilm_wharm
#SBATCH -p gpu3,gpu4,gpu5,gpu2,gpu6,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16          # 캐시 학습은 코어를 거의 안 쓴다 (train_v38 도 16)
#SBATCH --gres=gpu:2
#SBATCH --mem=96G      # 실측: 캐시 학습 MaxRSS 104GB 가 96G 안에서 완료 (963828)
#SBATCH -t 01:15:00    # 실측 판당 ~19분, 두 판 병렬 + 채점. 1.5배
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
SEED=${SEED:-0}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 캐시 $CACHE"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문 (앞에, HPC_RULES §7-3) ────────────────────────────────────────────
test -d "$CACHE" || { echo "캐시가 없다: $CACHE — cache_v32_par.sbatch 로 굽고 이어붙였나"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 **토막**이다 — run_merge_traincache 로 먼저 이어붙여라: %s' % d['shard'])
print('캐시 창 %d · 시드 %s · sibling_rotate %s' % (d['n_windows'], d['seed'], d.get('sibling_rotate')))
if d.get('sibling_rotate'): sys.exit('v32 인데 sibling_rotate 가 있다 (14.4 위반)')
for k in ('float_fill','steady_crop','standby_jitter_cap','vtail','couple_ext','sp_curves'):
    if not d.get(k): sys.exit('캐시에 v32 설정이 빠졌다: %s' % k)
print('관문: 캐시가 v32 통짜다')
"

# ── 두 판. 바뀌는 것은 **--w-harm 하나** ──────────────────────────────────
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE --w-harm 0.10 \
    --tag cnn_wh10 > "results/cnn_wh10_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --w-harm 0.02 \
    --tag cnn_wh02 > "results/cnn_wh02_s${SEED}.log" 2>&1 &
B=$!
echo "A(w_harm 0.10) $A · B(w_harm 0.02) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: w_harm 0.10 ==="; tail -22 "results/cnn_wh10_s${SEED}.log"
echo "=== B: w_harm 0.02 ==="; tail -22 "results/cnn_wh02_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 (참고 — 전압 지수는 여기서 안 보인다, 14.6 ④) ==="
$PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_wh10.pt results/cnn_wh02.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_wh10.pt results/cnn_wh02.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬에서 실측·분포안 지수**로 한다:"
echo "  python -X utf8 src/run_diag_vindist.py --ckpt results/cnn_wh02.pt   # 지수가 올랐나"
echo "  python -X utf8 src/run_plot_real.py   --ckpt results/cnn_wh02.pt    # test_5 가 닫혔나"
echo "  python -X utf8 src/run_diag_rollback.py                             # SMPS 신원 대가"
exit $RC
