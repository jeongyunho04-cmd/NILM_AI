#!/bin/bash
# 14.395 (§52 (가)) — ★ **손잡이는 `--align-sig-recordings` 하나다.**
# 짝은 `cnn_pjit`(986561) 이고 캐시·홀드아웃·다른 인자가 **글자 그대로 같다**.
#
# 무엇을 고치나:
#   `sig = median(I_h/P)` 가 그 기기의 **모든 녹화를 섞어** 복소 중앙값을 낸다.
#   그런데 녹화마다 정류 도통 타이밍이 달라 페이저가 `h 비례`로 돌아 있다 (§52.1:
#   녹화 간 σ 미니PC 0.71 · 충전기 **3.06** · 빔 **3.63** 도/차수). 돌아 있는 것들의
#   복소 중앙값은 **크기를 잃는다** — 충전기 coh h15 **0.599**.
#   그러면 `L_harm` 이 고차 크기를 맞추려고 **충전기 와트를 올린다**
#   (§46.2 의 +4.1%/+20.8% 가 정확히 그 방향이다).
#
# 고침 = **녹화마다 회전 하나를 빼고** 사전을 다시 굽는다. 손실 **구조는 안 바뀐다**.
#   참 배분에서도 못 줄이던 잔차 (판별차수 h9·h11·h13 평균, 관문 [5] 가 찍는다):
#     충전기 0.593 -> **0.394 (−33.6%)** · 빔 0.136 -> **0.106 (−22.4%)**
#     미니PC 0.292 -> 0.280 (−4.3%)   <- 녹화 간 σ 가 0.67 뿐이라 이게 맞다
#     ⚠ 대조군 오븐·핫플 **정확히 +0.0%** (R² 0.14/0.02 라 채택에서 걸러진다)
#
# ⚠ 캐시는 **안 바뀐다** (실제 조각을 쓰므로). 사전만 바뀐다 -> 손잡이가 하나다.
# ⚠ **(나)는 이 판이 아니다** — §48.3 의 `exp(−j·h·a_k·(lnP−lnP̄))` 를 같이 켜면
#   §46 처럼 어느 쪽 덕인지 못 가른다.
#
# 자 (§45.3 규약) — 짝 `cnn_pjit` 대비
#   ① 저항 넷 합 **8** · |r| 유지 (사전이 SMPS 만 바뀌므로 저항은 원리상 불변)
#   ② ★ 충전기 홀드아웃 RE — **여기가 표적**이다 (사전의 크기 편향을 걷었다)
#   ③ 미니PC 실측 AUC 안 내려가나
#   ④ ★ **기기별 통전 편향/정격** 표에서 SMPS 셋의 맞바꿈이 줄었나 (§46.2 꼴)
#      ⇒ RE 만 보면 맞바꿈이 합에 묻힌다
#
#SBATCH -J nilm_sgal
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -a 0-5
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v50}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v49}
EPOCHS=${EPOCHS:-300}
SEED=$SLURM_ARRAY_TASK_ID
SIG=0.40
NAME=sgal
echo "노드 $(hostname) · 팔 $NAME (짝 cnn_pjit · 손잡이 --align-sig-recordings) · 씨 $SEED · $(date '+%F %H:%M:%S')"
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
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter $SIG \n       --align-sig-recordings"
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
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--align-sig-recordings" || {
  echo "코드를 안 올렸다 — --align-sig-recordings 가 없다"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 잰다** — 진짜 손실 객체를 지어 본다 (7/7 이어야 한다).
$PY -X utf8 -m src.run_gate_sigalign

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
#: 14.395 — **손잡이가 체크포인트에 남았나** (§29.5 부류 빚을 안 만든다)
if not ck.get('align_sig_recordings'):
    sys.exit('align_sig_recordings 가 체크포인트에 안 적혔다 — 손잡이가 안 걸렸다')
print('  align_sig_recordings=%r' % ck.get('align_sig_recordings'))
if ck.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('체크포인트에 새 차수가 안 적혔다')
" || RC=1
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
