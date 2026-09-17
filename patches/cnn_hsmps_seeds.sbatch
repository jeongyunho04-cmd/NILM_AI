#!/bin/bash
# 14.376 — ★★ **SMPS 전용 고조파 항** `--w-harm-smps` + 전력대 보정 `--pow-sig-instate`.
#
# 사용자 지적에서 나왔다: *"L_harm 에서 저항성 부하 분리되어 있어? 이미 총량을 알면
#   거기서 저항성 부하 몫의 고조파 빼는 건 쉬운 일이잖아"*.
#
#   지금  pred = Σ(기기 전부) + 대기 + 잡음 ; err = |pred − obs|   <- **잔차가 하나**
#   새 항 err_smps = | Σ_smps + 대기 + 잡음 − ( obs − **Ĝ·V_h** ) |   (h3 이상)
#
# ★ 왜 (14.374 측정) — 그 잔차를 **저항이 지배한다**:
#     test_5 관측 |I1| **7,915mA** 중 저항 몫 **7,896mA (100%%)** · SMPS 77mA (**1%%**)
#   ⇒ SMPS 목표를 고쳐도 잔차의 1%% 를 건드리는 것이고, 거꾸로 저항이 흔들리면 묻힌다.
#   14.373 의 C포트오탐 **2 -> 12** 가 정확히 그 누출 경로다.
#   `Ĝ·V_h` 는 **입력만의 함수**라 결합이 원리상 끊긴다 (관문 [2]: 저항 기울기 **0.000e+00**).
#   ⚠ h1 은 뺀다 — SMPS 없는 test_5 의 뺄셈 오차가 h1 76.9mA 라 신호:오차 **1.8:1** 이다
#     (h9 **5.6:1** · h11 **5.3:1**).
#
# ★★ **이 판은 어제(14.373)의 2×2 를 채운다** — 손잡이가 둘이지만 귀속이 된다:
#     cnn_comb    보정✗ smps항✗   합 **8**  · 충전기 RE 0.249
#     cnn_powsig  보정○ smps항✗   합 **19** · 충전기 RE **0.201** (−19%%, 진짜 이득)
#     ★ 이 판      보정○ smps항○   -> **합이 8 로 돌아오고 RE 0.20 이 남으면 가설 확정**
#       합이 19 에 머물면 새 항이 누출을 못 막은 것 · RE 가 0.249 로 돌아가면
#       어제 이득이 **누출 자체에서 온 것**이었다는 뜻이다
#
# ⚠ `w_harm_smps 0.3` 은 **측정에서 나왔다** — 새 항이 합쳐진 항의 **0.27배** 크기라
#   `w_harm 0.1` 과 같은 기여를 주는 값이 0.365 다. 임의로 고른 값이 아니다.
#
# ★★ **미리 적는 자** (짝 `cnn_comb` 끝점 대비)
#   1 ⚠ **저항 넷이 안 상하나** — 합 **8** · |r| **9.2W**. 관문이 기울기 0 을 보증했으니
#     여기가 19 로 가면 **관문이 못 잡은 경로가 또 있다**는 뜻이다
#   2 **충전기 홀드아웃 RE 0.249 -> 0.201 이 남나** (어제의 진짜 이득)
#   3 ⚠ **미니PC 실측 AUC 0.789** — 표적. 어제 손잡이 넷이 전부 못 움직였다
#   4 test_2 미니PC 게이트 **19.6%%(참 90.3%%)**
#   5 충전기 실측 AUC **0.9965** 안 내려가나
#
#SBATCH -J nilm_hsmp
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:10:00
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
NAME=hsmps
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
$PY -X utf8 -m src.run_gate_powsig_instate
$PY -X utf8 -m src.run_gate_harmsmps
$PY -X utf8 src/run_gate_swa.py
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
#: 14.369 - Z 주입 기각으로 홀드아웃 관문을 뻐다 (짝이 cnn_comb 이다)
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
COMB="--comb-tau ${TAU:-1.0} --comb-over ${OVER:-0} --comb-over-margin ${OVMARGIN:-2.0}"
#: ★ 14.369 — 꺾기는 **학습에 안 넣는다** (합 8 대 11). 추론 운영점으로 켠다.
#: ★ 14.355 — 선로 임피던스 주입. `z_proj` 는 0 초기화라 **출발이 짝과 비트 동일**이다.
ZIN="${ZIN_FLAGS:-}"                       # ★ 14.369 기본 **끔** (Z 주입 기각)
#: ★ 14.368 — **이 판의 손잡이.** 상태 안 전력대 보정 + SWA(공짜 대조).
PSIG="--pow-sig-instate --pow-bands ${PBANDS:-3} --pow-rel-floor ${PFLOOR:-0.05}"
#: ★ 14.376 — **이 판의 새 손잡이.** 저항 몫을 `Ĝ·V_h` 로 빼고 SMPS 만 맞춘다.
HSMPS="--w-harm-smps ${WHS:-0.3} --harm-smps-min-order ${HSMIN:-3}"
SWA="--swa-start ${SWASTART:-240}"
COMB="$COMB $ZIN $PSIG $HSMPS $SWA"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT $COMB"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.315·14.331)
# ★ 14.368 — **`--swa-start` 를 목록에서 뺐다.** 몰래 지나가면 안 되므로 이유를 적는다:
#   29.12 의 판정은 *"해롭지 않지만 아무것도 안 샀다"* 였지 **해롭다**가 아니다
#   (네 방향 증분 p=0.36~1.00 · 홀드아웃 MAE 4.843 -> 4.819 로 6 중 5 개 개선).
#   새 바닥에서 뺀 것은 **배치 변경을 고립시키려던 것**이고, 지금은 `{tag}_last.pt` 가
#   같은 궤적의 끝점을 따로 저장해 **손잡이가 둘이 되지 않는다**. 판정은 끝점으로 한다.
#   ⚠ 목록의 나머지는 그대로다. 그리고 **검사 문자열에 `$COMB` 를 넣어** 앞으로는
#     `$COMB` 로 숨겨서 지나갈 수 없게 한다 (지금까지 그 구멍이 있었다).
case "$BASE$WTAP$CAP$ARCH$TREAT$COMB" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--wide-dg*|*--head-conductance*)
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
