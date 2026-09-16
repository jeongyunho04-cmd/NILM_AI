#!/bin/bash
# 14.355 — ★★ **Z 주입 + 꺾기** 를 한 판에. 짝은 `cnn_comb`(씨앗 여섯, 같은 캐시).
#
# 이 판에서 바뀌는 것 (짝 `cnn_comb` 기준):
#   --comb-over 0.05     14.352 의 꺾기. **추론에서 이미 채택**했고 학습으로 옮긴다
#   --z-input            14.354 의 선로 임피던스 주입 (z_proj 0 초기화)
#   --z-drop 0.3         못 잰 자리 대비책
#
# ⚠ **손잡이가 둘이다.** 나빠지면 어느 쪽인지 못 가른다 — 그래서 꺾기는 이미
#   추론에서 **합 8 · |r| 9.2W** 로 잰 값이 있고, 그 수에서 더 나빠지면 Z 쪽이다.
#   [[count-how-many-things-differ-before-attributing]] 를 이렇게 피한다.
#
# ★★ **미리 적는 자** (사용자 결정 순서)
#   1 ⚠ **충전기·미니PC 실측 AUC** — 이 판의 진짜 표적이다.
#     13.55 가 `w_z 0.3` 으로 충전기 **+0.077** · 미니PC **+0.068** 을 적었는데
#     14.350 이 그 이득의 **기전이 Z 가 아님**을 보였다 (실측 Z 오차 48~60%).
#     맞는 Z 를 주면 올라야 한다. **안 오르면 Z 가설이 틀린 것이다**
#   2 test_2 의 **미니PC 게이트 22% (참 91%)** — §39.7ⓐ 의 구멍. 이게 목표다
#   3 A/B/C/D 합 · |r| — **추론 꺾기 8 / 9.2W** 아래여야 한다 (기각선)
#   4 ‖z_proj.weight‖ — **0 에서 시작한다. 안 움직였으면 안 배운 것이다**
#     같이 읽을 것: log_z 실측 오차(지금 48~60%) · 모름 창 따로 · 배운 β 넷
#
# ⚠ 홀드아웃은 `holdout60_v49z`(14.353)라야 Z 갈래를 잰다. 옛 것이면 전부 "모름"
#   으로 돌아 **대비책 갈래만** 재게 된다 — 아래 관문 ① 이 막는다.
#
#SBATCH -J nilm_zin
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:10:00
#SBATCH -a 0-5
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v49}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v49z}
EPOCHS=${EPOCHS:-300}
SEED=$SLURM_ARRAY_TASK_ID
SIG=0.40
NAME=zin
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
$PY -X utf8 -m src.run_gate_zin
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
import os
if not os.path.exists('$HOLDOUT/z_grid.npy'):
    sys.exit('홀드아웃에 z_grid.npy 가 없다 — 14.353 의 holdout_v49_z 를 먼저 구워라 '
             '(없으면 평가가 전부 "모름" 으로 돌아 Z 갈래를 못 잰다)')
zg = np.load('$HOLDOUT/z_grid.npy')
print('  홀드아웃 z_grid %s · r %.3f~%.3f · 서로 다른 값 %d개'
      % (zg.shape, zg[:,0].min(), zg[:,0].max(), len(np.unique(np.round(zg[:,0],4)))))
if np.isnan(zg).any(): sys.exit('홀드아웃 z_grid 에 NaN 이 있다')
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
#: ★ 14.348 — **이 판의 유일한 손잡이.** 물리 잔차의 눈금 (mS).
#  14.343 의 λ 쓸기: 0(Ĝ 단독) 84.5%% · **0.6~1.0 꼭지 99.6%%** · >=2 모델 단독 94.9%%
#: ★ 14.352 — **이 판의 유일한 손잡이.** 초과 주장 벌점의 눈금 (mS).
#  τ=1.0 은 그대로 두고 꺾기만 더한다. 0.05 면 215초 띠의 1.22 mS 초과가 벌점 24.4 다.
COMB="--comb-tau ${TAU:-1.0} --comb-over ${OVER:-0.05} --comb-over-margin ${OVMARGIN:-2.0}"
#: ★ 14.355 — 선로 임피던스 주입. `z_proj` 는 0 초기화라 **출발이 짝과 비트 동일**이다.
ZIN="--z-input --z-drop ${ZDROP:-0.3}"
COMB="$COMB $ZIN"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT $COMB"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.315·14.331)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--wide-dg*|*--swa-start*|*--head-conductance*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT $COMB \
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
