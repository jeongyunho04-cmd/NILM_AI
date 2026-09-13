#!/bin/bash
# 14.7 — **전압 지수를 구조에 박으면 저전압 과예측이 고쳐지나.** 짝지은 2판.
#
# 14.6 이 잰 원인: 저항 부하는 `P = V²/R` 인데 모델이 배운 지수가 **0.85** 다.
#   α 훑기(전류 α · 전압 α · 전력 α²)로 v31 0.81 · v37 0.85 · v32 0.91.
#   모델은 **상태 명목값**은 기울기 1.00 으로 잘 내는데(오븐 13W~1326W) **같은 상태 안의
#   V² 의존**을 0.33~0.83 로만 읽는다. 그리고 고정 저항에서 상태 안 전력 변화는 **오직 전압**이다
#   (합성 참값 회귀: 포트 2.01 · 드라이기 2.00 · 핫플 2.02 · 오븐 2.00 · SMPS ~0).
# 증상: 드라이기 녹화 227.2V · test_5 는 210.6V -> 물리 0.859 대 모델 0.937 = **+9.1% 과예측**.
#   실측 test_5 가 v25 이후 **모든 모델**에서 +2.3~7.9% 다. test_1 은 핫플·오븐 녹화 전압과
#   파일 전압이 같아 깨끗하고(+10W), test_4 는 저항이 0W 라 편향 0 이다.
#
#   A 대조   그대로
#   B vexp   --vexp   p_raw *= (V/222)^e_k  · 저항 2.0 · SMPS 0.0 · 유도기 0.6
#
# ⚠ **캐시를 안 굽는다** (`--cache none`, 14.8). 쿼터가 79.5G/100G 라 23GB 캐시를 못 쓴다.
#   실시간 합성은 굽는 35~55분이 0 이고 **창을 한 번도 재사용하지 않는다** (12.1절).
#   생성기는 `--gen v32` = 사슬 이전 판(= v36 − sibling_rotate, 14.4 의 롤백 결정).
#   ⚠ 그래서 `cnn_v31`·`v32` 와 **절대값을 나란히 놓으면 안 된다** — 자료 파이프라인이 다르다.
#   이 판이 재는 것은 **A 대 B** 하나다.
#
# ⚠ 판정은 **실측 전력**으로 한다. 합성 홀드아웃은 이것을 **못 본다** — 합성 창의 전압이
#   녹화 전압과 같아 α≈1 이라 k 가 절반 이하로 틀려도 잔차가 −0.7~−1.2% 로 깨끗하다 (14.6 ④).
#   체크포인트를 받아 **로컬에서** `run_gate_vexp.py` · `run_plot_real.py` 로 잰다.
# ⚠ 개선은 **test_5·test_2 에서만** 보여야 정상이다. 나머지 셋(기울기 1.00 · 절편 ±3W)이
#   움직이면 무언가 잘못된 것이다.
#SBATCH -J nilm_vexp
#SBATCH -p gpu6,gpu5,gpu4,gpu3,gpu2,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40          # gpu1·gpu6 의 노드당 상한이 40 이다 (HPC_RULES §2). 판당 워커 18 + 주 2
#SBATCH --gres=gpu:2
#SBATCH --mem=140G
#SBATCH -t 02:30:00    # 실측 32분(캐시)·실시간 합성은 더 걸린다. 넉넉히 1.5~2배 (§2)
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

EPOCHS=${EPOCHS:-300}
SEED=${SEED:-0}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · epoch $EPOCHS · 씨앗 $SEED · 홀드아웃 $HOLDOUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── 관문 (앞에 둔다 — HPC_RULES §7-3) ───────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vexp" || { echo "run_train_cnn 에 --vexp 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--gen"  || { echo "run_train_cnn 에 --gen 이 없다"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ① 실시간 합성이 **실제로 v32 설정을 쓰는가** ([[verify-the-gate-runs-that-path]]).
#    13.84.27 과 14.8 이 같은 결함을 두 번 찾았다 — 맨몸 생성기는 조용히 약하다.
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
from src.run_train_cnn import SynthBatchDataset
ds = SynthBatchDataset(1, 8, seed=0, gen_spec='v32', recipe_mix='steady2', smps_focus_off_p=0.4)
ds[0]
a = ds.gen.synthesizer.augmentor
need = {'sp_curves': bool(getattr(a, '_sp', None)), 'sp_per_texture': bool(getattr(a, '_sp_tex', None)),
        'float_fill': bool(getattr(a, 'float_fill', None)), 'steady_crop': bool(getattr(a, 'steady_crop', None)),
        'state_mix': bool(getattr(a, 'state_mix', None)), 'power_scale_std': bool(getattr(a, 'power_scale_std_map', None))}
bad = [k for k, v in need.items() if not v]
if bad: sys.exit('실시간 합성에 설정이 안 걸렸다: %s' % bad)
if getattr(a, 'sibling_rotate', None): sys.exit('v32 인데 sibling_rotate 가 걸렸다 (14.4 롤백 위반)')
print('관문: 실시간 합성 v32 설정 통과 · sibling_rotate 없음')
"

# ② `--vexp` 가 실제로 지수를 더하는가. 끄면 항등인가.
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
import torch
from src.model.net import NILMNet, V_EXP
from src.run_baseline import S_I
from src.model.traincache import appliance_state_counts
apps = sorted(S_I)
off = NILMNet(apps, appliance_state_counts(apps), vexp=False)
on = NILMNet(apps, appliance_state_counts(apps), vexp=True)
if float(off.v_exp.abs().sum()) != 0.0: sys.exit('끈 판의 v_exp 가 0 이 아니다')
want = [V_EXP.get(x, 0.0) for x in apps]
if [float(x) for x in on.v_exp] != want: sys.exit('켠 판의 v_exp 가 V_EXP 와 다르다')
print('관문: v_exp 버퍼 %s' % ' '.join('%.1f' % x for x in on.v_exp))
"

# ── 두 판을 동시에 ──────────────────────────────────────────────────────────
# 같은 씨앗·같은 예산·같은 생성기. 바뀌는 것은 **--vexp 하나**다.
BASE="--cache none --gen v32 --recipe-mix steady2 --smps-focus-off-p 0.4 \
      --holdout $HOLDOUT --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --eval-every 10 --workers 18 \
      --harm-even-magnitude --state-signatures --standby-operating session"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE          --tag cnn_vexp_ctl  > "results/cnn_vexp_ctl_s${SEED}.log"  2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --vexp   --tag cnn_vexp_on   > "results/cnn_vexp_on_s${SEED}.log"   2>&1 &
B=$!
echo "A(대조) $A · B(vexp) $B — $(date '+%F %H:%M:%S')"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: 대조 ===";  tail -25 "results/cnn_vexp_ctl_s${SEED}.log"
echo "=== B: --vexp ==="; tail -25 "results/cnn_vexp_on_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 ==="
$PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
