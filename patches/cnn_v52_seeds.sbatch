#!/bin/bash
# 14.411 — **v52 두 팔.** 캐시는 `train60_v52`(자리 F 가 든 풀), 손잡이는 `--harm-dprime` 하나.
#
#   sbatch --export=ALL,NAME=v52base,DPRIME=0   patches/cnn_v52_seeds.sbatch
#   sbatch --export=ALL,NAME=v52dp,DPRIME=1.0   patches/cnn_v52_seeds.sbatch
#   sbatch --export=ALL,NAME=v52mg,DPRIME=0,MARGIN=1.0  patches/cnn_v52_seeds.sbatch
#
# 팔 3 `--w-harm-margin` (14.412) — 사용자 지적에서 나왔다:
#   *"상수로 손실을 가려내는 게 이중으로 왜곡을 만들 것 같다."*
#   `pred_ij = pred + δ·(sig_j − sig_i)` 로 δ=5W 를 옮긴 예측을 짓고 오차가
#   `m_ij = 0.5·(pred=obs 에서 달성되는 간격)` 이상 커지길 요구한다.
#   절대적합이 잔차의 **95%**(순방향 오차)를 배분에 흡수시키는 것을 피한다.
#   14.412 가 잰 것: 정답이 오답보다 잔차가 작은 창이 SMPS **83.1%**(40W 옮김),
#   간격 0.389 = 참값의 37%. **저항은 47.8% (동전 아래)** 라 SMPS 쌍에만 건다.
#   관문 `run_gate_hmargin` 6/6 — 제일 좁은 쌍이 **충전기/빔**(§62 의 22.6°)이다.
#
# 왜 기준선이 따로 필요한가
#   캐시가 v49 -> v52 로 바뀌었다. v49 위의 `cnn_comb` 과 견주면 **풀과 손잡이가 같이**
#   움직여 무엇이 이겼는지 못 가른다 ([[count-how-many-things-differ-before-attributing]]).
#   `v52base` 가 **자리 F 를 넣은 효과만** 을 재고, `v52dp` 가 그 위에서 손잡이를 잰다.
#
# 손잡이 `--harm-dprime W` (14.411)
#   `den_h = harm_scale_h · (d_ref/d'_h)^W · c`,  `c = mean(harm_scale/den)` 로 규모 보존.
#   `d'_h` = 1W 오배분이 그 차수에 내는 신호 / 그 차수의 잔차 바닥 (`run_diag_orderdprime`).
#   W=0 은 **비트 동일**이다. 관문 `run_gate_hdprime` 7/7 이 잰 것:
#     짝수차 기울기 몫 **90.2% -> 72.3%** · 판별 차수 h3~h11 **4.4% -> 18.1%** · 단조
#   ⓘ 규모는 완전히는 안 잡힌다 — 남은 **+6.7%** (w_harm 0.1 이 사실상 0.107).
#
# ⚠ 되살리지 마라 (14.411 에서 측정으로 죽은 것들)
#   `--harm-grad-balance smps`  SMPS 불균형이 1.39배 -> **1.16배** 로 닫혔다 (개입 ±11%)
#   `--harm-resid-scale`        잔차로 백색화하면 h2 가 h1 의 **80배** 가중을 받는다
#                               (짝수 몫 90.2% -> 90.7% 로 **늘었다**). 코드에서 뺐다
#   `--harm-deadzone`           `cnn_dz03`/`dz06` 이 안 움직였고(§65), 게다가 그 판들은
#                               **옛 프로파일**(삭제된 계측기 자료)을 썼다. 다시 재려면
#                               14.411 의 새 표로 다시 굽는 판을 따로 세워라
#
#SBATCH -J nilm_v52
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#: 씨앗 5개 (사용자 지시 2026-09-17). 두 팔이 **같은 씨앗**을 써야 짝 검정이 선다.
#SBATCH -a 0-4
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v52}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v49}
EPOCHS=${EPOCHS:-300}
SEED=$SLURM_ARRAY_TASK_ID
SIG=0.40
NAME=${NAME:?NAME 을 달라 (v52base | v52dp)}
#: ★ 이 판의 유일한 손잡이. 0 = **비트 동일**.
DPRIME=${DPRIME:?DPRIME 을 달라 (0 | 1.0)}
#: ★ 14.412 — SMPS 쌍 맞바꿈 **여유**. 0 = 비트 동일. 팔마다 **하나만** 켠다.
MARGIN=${MARGIN:-0}
echo "노드 $(hostname) · 팔 $NAME (σ=$SIG · evenjit40 과 같은 처치) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).

# ── 관문 ⓪ **코드가 올라왔나** (985942 의 교훈) ─────────────────────────────
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS
print('  차수 %s · RAW %d · FINE %d · WIDE %d'
      % (tuple(VOLT_ORDERS), RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS))
ok = (tuple(VOLT_ORDERS) == (1, 3, 5, 7, 9, 11, 13, 15)
      and (RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS) == (49, 61, 51))
sys.exit(0 if ok else '코드를 안 올렸다 — 14.331 이전 배치다')
"

echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_evenjit
$PY -X utf8 -m src.run_gate_comb
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ── 관문 ① **캐시와 홀드아웃이 둘 다 새 배치인가** ──────────────────────────
#   ⚠ 홀드아웃 meta 에는 채널 수가 없다. `volt_orders` 가 없으면 옛 6차수 판이고,
#     그걸 섞으면 학습 분포와 평가 분포가 갈린다 (14.333 에서 이 키를 넣은 까닭).
$PY -X utf8 -c "
import json, sys
import numpy as np
c = json.load(open('$CACHE/meta.json', encoding='utf-8'))
h = json.load(open('$HOLDOUT/meta.json', encoding='utf-8'))
want = [1, 3, 5, 7, 9, 11, 13, 15]
print('  캐시   volt_orders=%r · fine=%r · wide=%r · 창 %d'
      % (c.get('volt_orders'), c.get('fine_shape'), c.get('wide_shape'), c['n_windows']))
print('  홀드아웃 volt_orders=%r · harmonic_z=%r · 창 %d'
      % (h.get('volt_orders'), h.get('harmonic_z'), h['n_windows']))
if c.get('volt_orders') != want: sys.exit('캐시가 새 배치가 아니다')
if h.get('volt_orders') != want: sys.exit('홀드아웃이 새 배치가 아니다')
if list(c.get('fine_shape') or []) != [61, 600]: sys.exit('캐시 fine_shape 가 [61,600] 이 아니다')
if list(c.get('wide_shape') or []) != [51, 120]: sys.exit('캐시 wide_shape 가 [51,120] 이 아니다')
if not c.get('harmonic_z') or not h.get('harmonic_z'): sys.exit('harmonic_z 가 한쪽에 없다')
if c.get('vtex_seg_s') or h.get('vtex_seg_s'): sys.exit('vtex_seg_s 가 실렸다 — 접은 축이다')
x = np.load('$HOLDOUT/X.npy', mmap_mode='r')
print('  홀드아웃 X %s' % (x.shape,))
if x.shape[1] != 49: sys.exit('홀드아웃 X 채널이 %d 다' % x.shape[1])
if not (c.get('g_hat') or {}).get('v_ref'): sys.exit('캐시에 g_hat 이 없다 — run_build_ghat')
print('  관문 ① 통과 — 학습과 평가가 같은 배치다 · g_hat 중앙 %.2f mS'
      % (c.get('g_hat') or {}).get('median_ms', -1))
"

# ⚠ BASE 는 짝(`cnn_evenjit40`)과 **글자 그대로 같아야** 한다 — 캐시·홀드아웃만 다르다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH=""
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter $SIG"
#: ★ 14.411 — 두 팔을 가르는 **단 하나**. BASE/WTAP/CAP/TREAT/COMB 는 글자까지 같다.
DP="--harm-dprime $DPRIME --w-harm-margin $MARGIN"
#: ⚠ 두 손잡이를 **같이 켜면 안 된다** — 무엇이 이겼는지 못 가른다.
case "$DPRIME:$MARGIN" in 0:0|0:*|*:0) : ;; *) echo "DPRIME 과 MARGIN 을 같이 켰다"; exit 1;; esac
#: ★ 14.348 — **이 판의 유일한 손잡이.** 물리 잔차의 눈금 (mS).
#  14.343 의 λ 쓸기: 0(Ĝ 단독) 84.5%% · **0.6~1.0 꼭지 99.6%%** · >=2 모델 단독 94.9%%
COMB="--comb-tau ${TAU:-1.0}"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT $COMB"
echo "  손잡이: $DP"
case "$BASE$WTAP$CAP$ARCH" in *--harm-dprime*) echo "손잡이가 바닥으로 샜다"; exit 1;; esac
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.315·14.331)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--wide-dg*|*--swa-start*|*--head-conductance*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT $COMB $DP \
    --tag cnn_${NAME}_s${SEED} > "results/cnn_${NAME}_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_${NAME}_s${SEED}.log"
ls -la results/cnn_${NAME}_s${SEED}.pt || true
# 체크포인트가 새 배치로 적혔는지 확인한다 (14.331 의 막이 ⓒ)
$PY -X utf8 -c "
import torch, sys
ck = torch.load('results/cnn_${NAME}_s${SEED}.pt', map_location='cpu', weights_only=False)
print('  체크포인트 volt_orders=%r · fine_channels=%r · even_median=%r · even_jitter=%r'
      % (ck.get('volt_orders'), ck.get('fine_channels'), ck.get('even_median'),
         ck.get('even_jitter')))
if ck.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('체크포인트에 새 차수가 안 적혔다')
" || RC=1
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
