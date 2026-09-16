#!/bin/bash
# 14.352 — ★ **물리 벌점을 한쪽만 꺾는다** `--comb-over`. 짝은 `cnn_comb`(씨앗 여섯).
#
# ⚠ **바뀌는 것은 `--comb-over` 하나뿐이다.** 캐시·홀드아웃·처치·씨앗이 전부 같다.
#
# 왜 — 14.351 이 사용자 지목(test_5 215초)을 열어 병의 이름을 냈다: Ĝ 가
#   24.87~24.93 으로 **미동도 안 하는데**(오븐 단독 24.957 에서 0.06 mS) 포트 후보
#   (28.122, **3.2 mS 밖**)가 699W 를 가져간다. 합은 맞으니 손실이 조용하다
#   ([[a-sum-constraint-launders-misattribution]]). 벌점이 겨우 3.2 (τ=1.0) 라
#   β_포트 0.337 의 확신에 진다.
#
#   over_c = relu( ΣG(c) − Ĝ − 2.0 ) ;  score_c −= over_c / comb_over
#
# ⚠⚠ **한쪽만 꺾는다.** 초과 주장(없는 전류)은 불가능하지만 미달은 가능하다 —
#   선풍기가 명목의 53~78%% 를 Ĝ 에 흘리고(§37) 오븐에 9.94 mS 짜리 결손 상태가
#   있다(§39.7ⓑ). 양쪽을 다 막으면 그 창에서 **후보가 전멸한다**.
# ⚠ 그래서 이 팔은 §39.7ⓑ(오븐↔핫플 duty)를 **안 고친다.** 겨냥은 초과 주장뿐이다.
#
# ★★ **기각선 — 이 항은 추론에서만 켜도 된다.** 재학습은 그 수를 넘어야 뜻이 있다.
#   같은 `cnn_comb` 체크포인트 여섯에 `model.comb_over` 만 켜서 잰 값 (14.352):
#
#     팔                  A  B  C  D   합   |r|W   1.8%자     씨앗별 합
#     조합 단독             0  4  9  2   15   8.7  0.54/0.46  [21,12,15,12,17,15]
#     조합+거부권(후처리)     0  4  6  2   12   9.0  0.35/0.46  [11,12,8,12,17,15]
#     ★ 추론 꺾기 0.05      0  4  2  2  **8**  9.2  0.13/0.11  [8,12,8,8,9,9]
#     꺾기 0.05+거부권       0  4  2  2    8   9.2  0.13/0.11   <- 거부권이 **아무것도 더 안 한다**
#
#   ⇒ **합 8 · |r| 9.2W · 1.8% 0.13/0.11 을 못 넘으면 재학습을 접고 추론 켜기로 간다.**
#   ⇒ 재학습이 노릴 자리는 **|r|** 다 — 지금 8.7 -> 9.2 로 0.5W 늘었는데, 학습하면
#     포트에서 뺀 전력을 오븐이 제대로 받도록 β·크기가 같이 적응할 수 있다.
#   ⚠ 0.05 와 0.20 이 같은 값이고 0.01 은 되레 나쁘다 (합 9). 평평한 자리를 골랐다.
#
# 저항 4종을 독립 게이트가 아니라 **24개 조합 위의 softmax** 로 낸다. Ĝ 는 계량기로
# 쓰고(총 전력 −0.39%) 분류는 모델이 한다(94.9%) — 14.343 이 잰 결합이 **99.69%** 다.
#
# ★ **기각선이 있다**: 후처리 사중+총량이 **합 31 · |r| 11.0W** 다 (`run_gate_pintot`).
#   조합 머리가 그 수를 못 넘으면 **구조로 할 이유가 없다.** 접고 후처리로 간다.
#
# (아래는 짝이 된 새 바닥의 설명이다)
#
# 왜 — 14.331 이 `VOLT_ORDERS` 를 여덟로 넓혀 RAW 45->49 · FINE 57->**61** · WIDE 47->**51**
#   이 됐다. **지금까지의 체크포인트가 전부 무효**다. 오늘까지 잰 후처리 수치
#   (거부권 C오탐 32->24 · 고정 총잔차 6.4->5.5 · 삼중 합 60->32)는 전부 **옛 배치 위의
#   값**이고, 기전의 증거이지 새 배치의 성적이 아니다. 비교선을 다시 세우는 판이다.
#
# ⚠ **처치는 `cnn_evenjit_seeds.sbatch` 의 ARM 1 과 글자 그대로 같다** (σ=0.40).
#   달라지는 것은 `--cache`·`--holdout`·`--tag` 뿐이고, 그건 배치가 바뀌었기 때문이다.
#   딴 손잡이를 여기서 같이 넣지 마라 — 그러면 새 바닥이 바닥이 아니게 된다.
#
# ⚠ **이름을 `cnn_evenjit40_s*` 로 쓰지 않는다.** 옛 체크포인트를 덮어쓰면 §29·§31~§35 의
#   수치를 낸 물건이 사라진다. 옛것은 57채널이라 `load_model` 이 못 읽지만, 기록은 남긴다.
#
# 이 판 뒤에 할 일 (14.335):
#   `src/model/gbudget.py` 의 네 칸(거부권·바닥·천장·고정)을 **이 바닥 위에서** 다시 잰다.
#   ⚠ 그 관문(`run_gate_gbudget`)은 `processed_data/composite_eval` 을 읽으므로
#     **HPC 에서 못 돌린다** (HPC_RULES §0 — 실측은 올리지 않는다). 로컬에서 잰다.
#
#SBATCH -J nilm_covr
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
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v49}
EPOCHS=${EPOCHS:-300}
SEED=$SLURM_ARRAY_TASK_ID
SIG=0.40
NAME=combover
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
#: ★ 14.348 — **이 판의 유일한 손잡이.** 물리 잔차의 눈금 (mS).
#  14.343 의 λ 쓸기: 0(Ĝ 단독) 84.5%% · **0.6~1.0 꼭지 99.6%%** · >=2 모델 단독 94.9%%
#: ★ 14.352 — **이 판의 유일한 손잡이.** 초과 주장 벌점의 눈금 (mS).
#  τ=1.0 은 그대로 두고 꺾기만 더한다. 0.05 면 215초 띠의 1.22 mS 초과가 벌점 24.4 다.
COMB="--comb-tau ${TAU:-1.0} --comb-over ${OVER:-0.05} --comb-over-margin ${OVMARGIN:-2.0}"
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
