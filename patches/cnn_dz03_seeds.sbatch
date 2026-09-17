#!/bin/bash
# 14.408 — ★ **손잡이는 `--harm-deadzone 0.3` 하나다.** 짝은 `cnn_comb`(986106) 이고
# 캐시·홀드아웃·다른 인자가 **글자 그대로 같다** (v49 / holdout60_v49).
#
# ★★ 방향이 앞의 여덟과 **반대**다. 앞의 여덟은 전부 *"사전을 더 정확하게"* 였고
#   전부 미니PC 를 못 움직였다. 이건 *"못 맞추는 걸 그만 벌해라"* 다.
#
# 왜 (사용자 질문에서 나왔다 — *"많이 흔들리는 고조파를 상수로 잡고 벌하는 설계가
# 잘못된 게 아닐까"*):
#   `HARM_DEADZONE_PROFILE` = **정답 배분을 넣어도 남는 차수별 잔차 중앙값** (12.122.16)
#     홀수 h1 .191 h3 .303 h5 .320 h7 .270 h9 .265 h11 .298 h13 .378 h15 **.798**
#     짝수 .843 .851 .772 .879 **1.138** 1.068 **1.545**   전 차수 평균 **0.661**
#   학습이 도달하는 harm = **0.767**  ->  **86%가 원리상 못 줄이는 몫**인데 전액 벌한다
#   줄일 손잡이가 **배분뿐**이라 배분이 밀린다 (12.120.3 의 '가장 싼 기기')
#   ⇒ 오늘 판 다섯이 같은 모양으로 실패했다: harm 0.767->0.746->0.685 인데
#     pw 0.0029->0.0037->**0.0062**. **고조파를 더 맞출수록 와트가 틀린다**
#
# ⚠⚠ **여태 1단계에 안 걸려 있었다.** `unlabeled()`(2단계)에만 있었고 주석이
#   *"1단계는 합성이라 순방향이 정확하다"* 고 적어 뒀다. **그 전제가 틀렸다** —
#   §51 이 합성 풀에서 직접 쟀다: 참 배분에서도 충전기 h11 **0.592** · h13 0.737 이다.
#   사전은 **실측 풀**에서 굽는다. 합성이라고 맞는 게 아니다
#
# ⚠ 프로파일이 **12.122.16 = 13절 이전 계측기**다 (§55.5). 절대값이 아니라
#   **상대 쓸기**로만 읽는다. 다시 재는 것은 따로 할 일이다
# ⚠ 12.12.2 의 경고 — *"L_harm 이 죽으면 합만 맞추는 해로 무너진다"*. 그래서
#   배수 1.0(−83%)이 아니라 **0.3(−45%) · 0.6(−70%)** 로 쓸었다
#
# 자 (§60.3 SMPS 축 · 짝 cnn_comb · 씨앗 0~4)
#   ★1 실측 AUC 셋 — 미니PC **0.7780** 넘나 (손잡이 여덟이 못 넘은 그 수)
#   ★2 홀드아웃 통전 RE 셋 — 충전기 0.2495 · 미니PC 0.2917 · 빔 0.1262
#   ★3 기기별 통전 편향/정격 표 — 맞바꿈이 줄었나
#    4 저항 가드레일 — 합 <= 33 · **놓침 B·D 불변** · 신원 0.997
#   ⚠ 학습 로그의 `harm` 이 **뚝 떨어져야** 한다 (손잡이가 걸렸다는 증거).
#     안 떨어지면 배선이 안 닿은 것이다
#
#SBATCH -J nilm_dz03
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
NAME=dz03
echo "노드 $(hostname) · 팔 $NAME (짝 cnn_comb · 손잡이 --harm-deadzone 0.3) · 씨 $SEED · $(date '+%F %H:%M:%S')"
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
#: ⚠⚠ 14.406 — **한 줄로 쓴다.** 이어쓰기를 쓰려다 파이썬 힙독에서 역슬래시+n 이
#:   **두 글자 그대로** 들어가 986882 가 `unrecognized arguments: \n` 으로 죽었다
#:   (관문 여섯을 다 통과한 **뒤**, 학습 시작 42초 만에). 오늘 셸 인용 **네 번째**다.
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter $SIG --harm-deadzone 0.3"
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
if abs(float(ck.get('harm_deadzone') or 0.0) - 0.3) > 1e-9:
    sys.exit('harm_deadzone 이 %r 다 — 손잡이가 안 걸렸다' % ck.get('harm_deadzone'))
print('  harm_deadzone=%r' % ck.get('harm_deadzone'))
if ck.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('체크포인트에 새 차수가 안 적혔다')
" || RC=1
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
