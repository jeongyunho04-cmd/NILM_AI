#!/bin/bash
# 14.183 ⓑ — **인과 정규화 하나만.** 짝 `cnn_dzn` — `--fine-norm` 하나만 다르다.
#
# 14.182 가 센 것 (test_2 176.0초 경계 · 두 사이클 차이):
#   구멍 넷의 막은 %          dzn_s0  dzn_s1  osig_s0
#     ⓐ 파생채널 41·42·29·30    −3%     93%     −6%
#     **ⓑ 정규화 μ**           45%     47%     −5%
#     ⓒ 전역최대 amax           40%     67%     68%
#     ⓓ 수용영역                35%     15%     32%
#   -> **판마다 지배 구멍이 다르다.** 그래서 한 손잡이 처방이 전부 널이었다.
#
# 이 판은 그중 **ⓑ 하나**다. `GroupNorm` 통계가 (C_g, **T 전체**) 라 수용영역과
# 무관하게 미래 6초가 **기준선**에 들어간다. 경계에서 범인은 σ 가 아니라 **μ** 다:
#     μ만 갈아끼움 729 -> **45W** · σ만 729 -> 597W · 둘 다 729 -> 29W  (s1 796 -> 249/530/278)
#     σ 부풀이는 최대 +12%(블록3) 뿐이고 μ 이동이 0.02~0.13σ 다
#     **광역 갈래의 정규화는 무관** (724W = 기준 729W) — 그래서 세밀에만 건다
#
# ⚠ 추론에서 미리 재 본 것 (분포 이탈이라 **상한이지 예측이 아니다**):
#     ①시작 경계에서 729 -> **31W** (s1 796 -> 43W) · 위약(같은 240자리 미래) 1234W
#     그런데 **5파일 집계는 널**이고 절대 잔차가 15.6 -> 23.3W 로 **악화**한다.
#     학습에서 해야 읽히는지 보는 것이 이 판의 전부다.
#
# **반증 가능한 예측**
#   ① 학습이 돈다. 정규화가 다르므로 첫 에포크 total 이 `cnn_dzn` 과 **다르다**.
#      파라미터 수는 **같다**(888,086). 체크포인트에 `fine_norm: causal` 이 적힌다.
#   ② ★ 오븐 참OFF 유령초가 줄어야 한다. 안 줄면 **ⓑ 하나로는 안 된다**가 확인된다
#      (그러면 짝인 `cnn_leak4` 가 답이다).
#   ③ ⚠ 절대 잔차가 오르면 비용이다. 추론판에서 +49% 였다.
#
# **미리 등록한 판정법**
#   이득  오븐 **참OFF 유령초** (dzn s0/s1/s2 = **44.5 / 28.5 / 19.0초**,
#           test_2 31.5/13.0/5.0 + test_5 13.0/15.5/14.0) · 수백W 띠 · p99
#   비용  판정줄 7종 (.8497/.8368/.8518) · 저항 4종 (.9815 x3) ·
#           SMPS (.9368/.9711/.8500) · 절대 잔차 (16.0/19.5/15.2W) ·
#           혼자켜짐 (20.9/45.5/44.4%)
#   ⚠ 15c: 세밀 룩어헤드 6초는 **쓰인다**(짧을수록 나쁘다). 미탐이 오르면 그 대가다.
#SBATCH -J nilm_cnorm
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
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
ARCH="--fine-norm causal"
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
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_cnorm_s${SEED} > "results/cnn_cnorm_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_cnorm_s${SEED}.log"
ls -la results/cnn_cnorm_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC

