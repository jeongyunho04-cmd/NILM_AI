#!/bin/bash
# 14.368 — ★★ `L_harm` **상태 안 전력대 보정** `--pow-sig-instate` (+ SWA 공짜 대조).
#   짝은 `cnn_zin`(986163) — 같은 캐시·홀드아웃·처치, **이 손잡이만** 다르다.
#
# 왜 (14.358·14.363 이 잰 것)
#   `L_harm` 은 `sig = median(I/P)` **상수표**를 써서 "모양은 전력과 무관" 을 가정한다.
#   상태로 가른 뒤에도 남는 오차가 **차수에 따라 가파르다** (minipc s2):
#     h1 12.0%% · h7 26.1%% · **h9 38.0%%** · **h11 63.9%%** · h13 129.6%% · h15 284.2%%
#   ★ 그리고 그 h9·h11 이 미니PC↔충전기를 가르는 **판별 차수**다 (+2.51°/+2.54°).
#   즉 **정보가 있는 차수에서 목적함수가 제일 틀린다.** 망은 틀린 목적함수를 못 피한다.
#
# ★ 14.172 가 적어 둔 처방 그대로다 — *"대역을 상태 안에서 잡아라 (sig_state 를 분모로)"*.
#   앞판 `--pow-sig` 는 대역 경계 == 상태 전력이라 같은 축을 두 번 재서 드라이 s2 h2/h1 을
#   **x0.000** 으로 죽였고 `--state-signatures` 와의 조합이 하드 차단돼 있다.
#
# ★ 손잡이가 **스스로 겨냥한다** (풀 측정 · 관문 `run_gate_powsig_instate` 6/6):
#     minipc s2 h11 **1.179/1.009/0.781** · charger s2 **1.400/0.899/0.668**
#     beam 0.971~1.070 · fan **1.000** · 저항 넷 **1.000** · 드라이 h2 **0.9989~1.0005**
#   (상태 안 전력 폭 minipc 2.35배 · charger 1.94배 · beam 1.05 · fan 1.01)
#
# ★ SWA 는 **공짜 대조**다 (14.295·29.12) — `{tag}_last.pt` 에 같은 궤적의 **끝점**을 따로
#   저장하므로 손잡이가 둘이 되지 않는다. 판정은 **끝점**으로 하고 SWA 이득은 그 위에서
#   따로 읽는다. ⚠ 29.12 기준 기대치는 **홀드아웃 MAE 0.5%%** 뿐이고 실측은 0 이었다.
#
# ★★ **미리 적는 자** — 짝(`cnn_zin`)의 끝점 대비
#   1 ⚠ **미니PC 실측 AUC** (지금 0.789 · F1 0.691) — 이 판의 표적이다
#   2 test_2 **미니PC 게이트 22%% (참 91%%)**
#   3 **충전기 AUC 0.9965 가 안 내려가나** (보정이 걸리는 나머지 한 기기다)
#   4 홀드아웃 미니PC **nMAE 0.220 · RE중앙 0.284**
#   5 ⚠ **저항 넷이 안 상하나** — 합 **8** · |r| **9.2W** (14.352 의 기각선)
#      g=1.000 이라 원리상 안 움직여야 한다. 움직이면 배선이 샌 것이다
#
# ⚠ 씨앗 수: 사용자 규칙은 **쓸기 셋 · 판정 다섯**이다 (14.356). 이건 채택/기각이라 **다섯**.
#   ⚠ 다만 미니PC AUC 의 **씨앗 바닥을 아직 모른다** — 986163 이 그걸 재 준다.
#     그 수를 보고 다섯이 모자라면 올려라.
#
# ★★ 14.369 — **986163 이 판정났고 짝을 바꿨다.**
#   ① **Z 주입 기각.** 같은 무게에서 Z 를 껐다 켜도 **모든 자가 동일**하다
#      (합 11 = 11 · |r| 9.3 = 9.3 · minipc AUC 0.7804 대 0.7800 · test_2 18.1 대 18.0%%).
#      ‖z_proj.W‖ 가 4.88~5.15 로 0 이 아닌데도 그렇다 — 배운 것의 대부분이 편향
#      (‖b‖ 8.76 > ‖W‖ 4.88) 이라 **Z 값과 무관한 상수**다.
#      ★ 왜 원리상 안 듣나 (사용자 지적): Z 가 크게 먹는 곳은 **고부하**인데
#        (포트 ΔV=Z·I 가 **7.79V=3.62%%** 대 미니PC **0.10V=0.05%%**, **75배**)
#        그 강하는 측정한 V₁ 에 이미 들어 있고 조합 머리가 **G(c)·|V₁|²** 로 쓴다.
#        따로 주면 **같은 것을 두 번** 주는 것이다.
#   ② **꺾기를 학습에 넣은 것도 기각.** 추론 적용이 합 **8** 인데 학습내장이 **11** 이다
#      (14.352 sbatch 의 기각선이 그대로 작동했다). 그래서 `OVER=0` 으로 끄고
#      **추론 운영점**(`gbudget.COMB_OVER_OP=0.05`)으로 간다.
#   ⇒ 짝은 **`cnn_comb`**(986106) 이고 손잡이는 **`--pow-sig-instate` 하나**다.
#
#SBATCH -J nilm_psig
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
NAME=powsig
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
SWA="--swa-start ${SWASTART:-240}"
COMB="$COMB $ZIN $PSIG $SWA"
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
