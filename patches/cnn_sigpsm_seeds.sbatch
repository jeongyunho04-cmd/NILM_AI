#!/bin/bash
# 14.405 — ★ **손잡이는 `--w-harm-smps 0.3` 하나다.** 짝은 `cnn_sigp`(986882) 이고
# `--load-rot` 을 **둘 다 켜고** 본문 차이가 그 한 줄뿐이다.
#
# 무엇을 묻나 (사용자 제안):
#   `err_smps = | Σ_smps + 대기 + 잡음 − ( obs − **Ĝ·V_h** ) |`  (h3 이상)
#   저항 몫을 빼서 SMPS 잔차를 **보이게** 만든다. `Ĝ·V_h` 는 입력만의 함수라
#   저항과의 결합이 **원리상 없다**.
#
# ⚠⚠ **이 항은 §46 에서 한 번 기각됐다** (`cnn_hsmps`):
#     충전기 홀드아웃 RE 0.198 -> **0.395** · 미니PC 0.304 -> **0.719** · 빔 0.170 -> **0.524**
#     통전 편향 충전기 **+20.8%** · 미니PC **−33.6%** · 빔 **−36.5%** 인데 **합은 5.5W 변화**
#     ⇒ 저항을 빼니 압력이 전부 SMPS 셋으로 몰렸고, 그 셋이 거의 평행이라
#       압력이 통째로 **맞바꿈 방향(영공간)** 으로 흘렀다
#
# ★ 그런데 그 판정을 **다시 묻는 이유가 둘** 있다:
#   ① `cnn_hsmps` 는 손잡이가 **둘**이었다 — `w_harm_smps=0.3` **그리고**
#      `pow_sig_instate=True`. 기각이 어느 쪽 탓인지 **안 갈렸다**
#   ② **(나) `--load-rot` 이 기하를 바꾼다.** §46 의 실패 기전이 *"압력이 흘러들 곳이
#      영공간뿐"* 이었는데, `sig(P)` 는 같은 기기를 전력대마다 **다른 방향**으로 보낸다.
#      압력이 흘러들 **조건 좋은 방향**이 생겼다면 결과가 달라질 수 있다
#
# ⚠ `(load_rot ✗, harm_smps ○)` 칸은 **안 던진다** (사용자 선택). 그래서 이 판은
#   *"load_rot 이 있을 때 harm_smps 가 사나"* 만 답하고, **상호작용은 못 가른다**.
#
# 자 (짝 `cnn_sigp` · 씨앗 6 · §60.3 의 SMPS 축)
#   ★1 실측 AUC 셋 — 미니PC · 충전기 · 빔 (짝지은 씨앗차 + t)
#   ★2 홀드아웃 통전 RE 셋 — ⚠ §46 에서 **2.0~2.4배 터진** 자다. 여기가 표적이다
#   ★3 **기기별 통전 편향/정격 표** — 맞바꿈이 줄었나 (RE 만 보면 합에 묻힌다)
#    4 저항 가드레일 — 합 ≤ 33 · **놓침 B·D 불변** · 신원 0.997
#
#SBATCH -J nilm_sgsm
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -a 0-4
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
NAME=sigpsm
echo "노드 $(hostname) · 팔 $NAME (짝 cnn_sigp · 손잡이 --w-harm-smps 0.3) · 씨 $SEED · $(date '+%F %H:%M:%S')"
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
#: ⚠⚠ 14.406 — **한 줄로 쓴다** (986882 이 리터럴 `\n` 으로 죽었다. 셸 인용 네 번째)
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter $SIG --load-rot --w-harm-smps 0.3"
#: ★ 14.348 — **이 판의 유일한 손잡이.** 물리 잔차의 눈금 (mS).
#  14.343 의 λ 쓸기: 0(Ĝ 단독) 84.5%% · **0.6~1.0 꼭지 99.6%%** · >=2 모델 단독 94.9%%
COMB="--comb-tau ${TAU:-1.0}"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT $COMB"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.315·14.331)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--wide-dg*|*--swa-start*|*--head-conductance*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

# ── 관문: **이 판의 유일한 손잡이**가 실제로 닿나 ───────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--load-rot" || {
  echo "코드를 안 올렸다 — --load-rot 이 없다"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 잰다** — 진짜 손실 객체를 지어 본다 (7/7 이어야 한다).
$PY -X utf8 -m src.run_gate_sigload
# ⚠ 두 학습기가 같은 순방향 모형을 쓰나 (14.171 이 남긴 관문)
$PY -X utf8 -m src.run_gate_lossparity

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
#: 14.402 — **손잡이가 체크포인트에 남았나** (2단계가 이걸 읽어 같은 물리를 쓴다)
if not ck.get('load_rot'):
    sys.exit('load_rot 이 체크포인트에 안 적혔다 — 손잡이가 안 걸렸다')
print('  load_rot=%r' % ck.get('load_rot'))
#: 14.405 — 이 판의 손잡이도 체크포인트에서 확인한다 (§29.5 부류 빚을 안 만든다)
if float(ck.get('w_harm_smps') or 0.0) <= 0.0:
    sys.exit('w_harm_smps 가 0 이다 — 손잡이가 안 걸렸다')
print('  w_harm_smps=%r' % ck.get('w_harm_smps'))
if ck.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('체크포인트에 새 차수가 안 적혔다')
" || RC=1
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
