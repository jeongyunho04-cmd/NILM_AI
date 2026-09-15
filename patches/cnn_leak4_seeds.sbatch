#!/bin/bash
# 14.183 ⓐⓑⓒⓓ — **구멍 넷을 한꺼번에.** 짝 `cnn_dzn`.
#
# 14.182 의 결론: 미래가 **최소 네 군데로 독립적으로** 새고, 학습된 모델은 막은 곳을
# 우회한다. dzn_s0 은 **넷을 다 막아도 39%** 인데 셋(ⓐⓑⓒ)만 막으면 56% 다 —
# 더하기가 안 되고 더 막으면 덜 막힌다. 판마다 지배 구멍이 다르다.
#   그래서 `--fine-time-split`(ⓑ+ⓓ) · `--fine-future-segs`(ⓒ) · `--fine-pad`(ⓓ) ·
#   `--seg-pool`(ⓒ) · `--head-drop`(ⓒⓓ) · 전방탭(ⓐ) · 광역룩어헤드가 **전부** 널이었다.
#
# **정보를 버리지 않는다. 시간 부호를 붙인다** (인계 2026-09-15b 의 결론).
#   ⓐ `--fine-derive both`  41·42(p(t+3.0s)·p(t+5.5s), 수용영역 1) 의 **인과 짝 4개**를
#                            `ch23` 에서 되살려 몸통 입력에 **더한다**. 캐시 재굽기 없음
#   ⓑ `--fine-norm causal`  GroupNorm 통계를 **0..타깃**에서만 (기준선만 과거로)
#   ⓒ `--fine-tpool split`  `h.mean/amax` 를 **과거/미래 따로** — `amax` 에는 위치가
#                            없어서 오탐 창의 argmax 가 128칸 중 **104칸이 미래**였다
#   ⓓ `--fine-conv causal`  세밀 conv 를 **왼쪽 패딩** (타깃 RF 가 미래로 못 간다)
#
# 관문 `run_gate_leaks` 일곱 줄이 **막힌 것과 일부러 남긴 것**을 둘 다 못 박는다:
#   [1] 넷 다 기본값 = 비트 동일 (최대차 0.000e+00)
#   [2] ⓓ conv 날출력이 타깃에서 **정확히 불변** (대칭은 1.1162)
#   [3] ⚠ **ⓑ 없이 ⓓ 만이면 타깃이 여전히 샌다 0.0861** — GroupNorm 이 다시 연다
#   [5] ⓐ 새 네 채널 불변 · 원래 41·42 는 **4.8729** 움직인다 (실측 창)
#   [6] 넷 다 켜도 미래가 답을 0.0784 움직인다 — **0 이면 오히려 의심**해야 한다
#
# **반증 가능한 예측**
#   ① 파라미터가 **955,414** (dzn 888,086). 머리 765 -> **1021**, 첫 conv 57 -> **61**.
#      체크포인트에 네 키가 다 적힌다.
#   ② ★ 오븐 참OFF 유령초가 크게 줄어야 한다. **안 줄면 넷이 전부가 아니다**
#      (남은 후보: ⑥ 광역 · ⑦ 원시 전력통계 `fp.amax` · 학습자료 쪽).
#   ③ ⚠ **가장 큰 위험 — 인과 conv 가 타깃의 수용영역을 반으로 줄인다.**
#      15c 가 *"세밀 룩어헤드 6초는 쓰인다(짧을수록 나쁘다)"* 를 쟀다. ⓒ 가 미래를
#      요약으로 되돌려 주지만 **깊은 탭의 미래는 사라진다**. 미탐·판정줄이 무너지면 그것이다.
#   ④ ⚠ 14.45: *"미래 6초를 그냥 가리면 나빠진다"*. 이 판은 가리는 게 아니라
#      **부호를 붙이는** 것이지만, 그 구별이 학습에서도 유지되는지는 이 판이 처음 잰다.
#
# **미리 등록한 판정법** — `cnn_cnorm`(ⓑ 하나)과 **같은 자**로 본다
#   이득  오븐 **참OFF 유령초** (dzn s0/s1/s2 = **44.5 / 28.5 / 19.0초**)· 수백W 띠 · p99
#   비용  판정줄 7종 (.8497/.8368/.8518) · 저항 4종 (.9815 x3) ·
#           SMPS (.9368/.9711/.8500) · 절대 잔차 (16.0/19.5/15.2W) ·
#           혼자켜짐 (20.9/45.5/44.4%)
#   ⚠ **판정줄이나 저항 4종이 무너지면 이득이 뭐든 기각** — 12.9 의 붕괴 감시다.
#SBATCH -J nilm_leak4
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:15:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32h}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_leaks
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_pcap`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--fine-norm causal --fine-conv causal --fine-tpool split --fine-derive both"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

case "$ARCH" in *--fine-norm*) : ;; *) echo "처치가 안 걸렸다"; exit 1;; esac
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_leak4_s${SEED} > "results/cnn_leak4_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_leak4_s${SEED}.log"
ls -la results/cnn_leak4_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC

