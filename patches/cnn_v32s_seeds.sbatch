#!/bin/bash
# 14.54 — **v32s 판**: 창 하나에 그 녹화의 연속 텍스처 6장. 3시드 x 2팔.
#         `--vrel-target`(14.53)을 **양쪽 팔 모두**에 묻어서 같이 켠다 (사용자 지시).
#
# ⚠⚠ **가르는 것은 캐시 하나다.** `--vrel-target` 은 학습 플래그이지 캐시 설정이 아니라
#   (캐시에는 `fine` 45채널만 있고 `vrel` 은 `to_targets` 가 채널 25 에서 계산한다),
#   **두 팔 다 켜야** A/B 가 캐시 하나만 가른다 ([[count-how-many-things-differ-before-attributing]]).
#   ⇒ 그래서 이 판의 A팔은 983573/983591 의 어느 팔과도 **짝이 아니다.** 새 기준선이다.
#
# 고친 것 (14.51):
#   `_terminal_voltage_harmonics` 가 `rel[None,:] * v_open` — 텍스처를 창 전체에 **정적으로**
#   곱했다. 창-안 `V_h/|V_1|` 변동이 **정확히 0** 이었고, 합성이 내던 13~35% 는 전부
#   `−Z·I` 항이었다. 창 하나에 연속 텍스처 6장을 얹으면 (실측 복합 대비):
#       Re(V_h/|V1|) 창-안 표준편차   17% -> **71%**
#       Im(V_h/|V1|)                 12% -> **62%**
#   과보정 아니다 (중앙이 100% 아래). ⚠ h11 의 Im 하나만 134% 로 넘는다.
#
# ★ **반증 가능한 예측** — 자와 씨앗 바닥을 **미리** 쟀다 (`run_diag_vblock.py`):
#   ① ★ **전압 블록 민감도가 준다.** 같은 파일 다른 창에서 전압 고조파 블록(세밀 45~56)만
#      끼워 넣고 `p_raw` 가 얼마나 흔들리는지. 물리 기대치는 ~0% 다 (SMPS 는 정전압 제어,
#      저항의 V² 는 안 건드린 채널 25 가 나른다).
#          지금 본선 (3시드)   SMPS **3.4 ± 0.2%** · 저항 1.06 ± 0.10% · 뒤집힘 1.36 ± 0.11%
#          14.39 대조         sigc_vn(vswap 0) 9.2/7.7/1.4 · v37(vswap 0.5) 1.7/0.9/0.4
#      **짝차 폭이 0.54%p 라 1.1%p 넘게 줄어야 읽힌다.** 3.4 -> 2.3% 면 읽히고, 3.0% 면 못 읽는다.
#      까닭: 모델은 "창 안에서 V_h 는 안 움직인다" 는 세계만 봤고 그래서 V_h 블록을 **창의
#      지문(판별자)** 으로 읽었다. 실측은 그렇지 않으니 실측에서 흔들린다.
#   ② SMPS 3종 신원이 오르면 그것도 이 축이다 (h11 창-간 다양성 +12%, 14.50).
#   ③ ⚠ **판정 줄(on/off 듀티제외 7종)은 나빠질 수 있다** — 창 안 변동이 늘면 과제가
#      어려워진다. 씨앗 폭(0.006~0.018) 안이면 무승부로 읽는다.
#   ④ ⚠ **`--vswap-p` 를 여기에 더하지 마라.** 14.50 이 그것을 **증상 억제**로 판정했고
#      이 판은 같은 증상을 **뿌리**에서 고치는 쪽이다. 둘을 겹치면 무엇이 고쳤는지 못 읽는다.
#   ⑤ ⚠ 합성 홀드아웃 숫자로 **팔끼리 비교하지 마라** — 각 팔이 **자기 분포**에서 잰 값이다.
#      판정은 `run_diag_rollback`·`run_diag_hipow`·`run_diag_vblock`(전부 실측)이다.
#
# ⚠ `FRAC` 을 **983591(분수 쓸기)의 승자**로 맞춰라. 기본 1.0 은 14.51 이 쓴 값이다.
#
#SBATCH -J nilm_v32s_t
#SBATCH -p gpu3,gpu4,gpu5,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 983573 이 A10 에서 30:21~30:31 이었다.
#SBATCH -t 00:45:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
EPOCHS=${EPOCHS:-300}
FRAC=${FRAC:-1.0}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · frac ${FRAC} · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문: 배선 · 캐시가 진짜 v32s 인가 ─────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vrel-target" || {
  echo "--vrel-target 이 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_gate_vreltgt || { echo "vrel-target 배선 관문 실패"; exit 1; }
$PY -X utf8 -m src.run_gate_vanchor || { echo "vanchor 배선 관문 실패"; exit 1; }
for d in cache/train60_v32 cache/train60_v32s processed_data/holdout60_v32 processed_data/holdout60_v32s; do
  test -d "$d" || { echo "$d 가 없다"; exit 1; }
done
# ⚠⚠ **캐시가 진짜 갈렸는지 meta 로 본다.** 이름만 다른 같은 캐시를 두 팔에 물리면
#   A/B 가 ±0.00 으로 나오고 그것을 결과로 읽는다 (983506 이 그랬다).
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v32/meta.json', encoding='utf-8'))
b = json.load(open('cache/train60_v32s/meta.json', encoding='utf-8'))
if float(b.get('vtex_seg_s', 0)) != 10.0: sys.exit('v32s 의 vtex_seg_s 가 %r' % b.get('vtex_seg_s'))
if float(b.get('vtex_step_s', 0)) != 10.0: sys.exit('v32s 의 vtex_step_s 가 %r' % b.get('vtex_step_s'))
if float(a.get('vtex_seg_s', 0)) != 0.0:  sys.exit('v32 에 vtex_seg_s 가 걸려 있다')
skip = {'vtex_step_s','vtex_seg_s','build_seconds','positive_rate','bytes','n_windows_total','shard','shard_chunks'}
d = [k for k in set(a)|set(b) if k not in skip and
     json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True, default=str)]
if d: sys.exit('텍스처 손잡이 말고도 다르다: %s' % d)
print('관문 ✅ 캐시가 vtex_step_s/seg_s 둘만 다르다 (v32 60/0 · v32s %s/%s) · 창 %d' %
      (b['vtex_step_s'], b['vtex_seg_s'], b['n_windows']))
"

BASE="--epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor --harm-vnorm-frac $FRAC \
      --vrel-target"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')  (--vrel-target 은 **양쪽 다**)"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache cache/train60_v32 --holdout processed_data/holdout60_v32 \
    --tag cnn_v32s_off_s${SEED} > "results/cnn_v32s_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache cache/train60_v32s --holdout processed_data/holdout60_v32s \
    --tag cnn_v32s_on_s${SEED} > "results/cnn_v32s_on_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: v32 (정적 텍스처) ==="; tail -32 "results/cnn_v32s_off_s${SEED}.log"
echo "===================================================================================="
echo "=== B: v32s (창 안 6장) ==="; tail -32 "results/cnn_v32s_on_s${SEED}.log"
ls -la results/cnn_v32s_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 (로컬, 실측 복합 5파일). **이 순서로 읽어라**:"
echo "  ① ★ python -X utf8 src/run_diag_vblock.py --pair 3 --ckpt \\"
echo "         results/cnn_v32s_off_s{0,1,2}.pt results/cnn_v32s_on_s{0,1,2}.pt"
echo "     겨냥: SMPS |Δp_raw| 3.4%% -> ? · **1.1%%p 넘게 줄어야 읽힌다**"
echo "  ② python -X utf8 src/run_diag_rollback.py --postproc off --ckpt <여섯>"
echo "  ③ python -X utf8 src/run_diag_hipow.py --pair 3 --ckpt <여섯>"
echo "  ⚠ 합성 홀드아웃 숫자로 팔끼리 비교하지 마라 — 각 팔이 자기 분포에서 잰 값이다."
exit $RC
