#!/bin/bash
# 14.269 — **짝수차 모양 + 홀수차 위상** 을 같이. 짝은 이미 있는 `cnn_evenjit40`(씨앗 여섯).
#
# 근거 (14.266 — 파일이 여럿인 기기로 잰 **세션 간** 변이):
#   자유도                 파일 안   파일 사이   비
#   **차수별 위상 φ rms°    6.33   **18.73**  3.0**   <- 절대 크기 1위
#   짝수차 크기 모양°        3.63     5.99    1.7     <- evenjit 가 덮는 자리
#   크기 수준 %            0.49     3.24    6.6      · 어드미턴스 % 0.17 / 1.43 / 8.3
#
# ★ **포트 s1 · 드라이 s1 · 드라이 s2 는 녹화가 하나뿐**이라 이 변이가 풀에 **구조적으로 0** 이다
#   (오븐 2 · 핫플 2 · 미니PC 4 · 충전기 6 · 빔 3). 실패 조합의 **두 기기만** 그렇다.
#
# 눈금: 짝수차는 측정된 5.99° 에 맞춰 **σ=0.40** (5.56°). σ=0.25 는 3.55° 로 낮다.
#       위상은 측정된 **18.7° rms** 그대로.
#
# ⚠ **반례를 먼저 적는다.** `evenjit40` 이 이미 유령을 57 -> 1 로 잡았다. 바닥이 거의 닿아
#   있으니 위상 지터가 더 얹을 자리가 없을 수 있다. 그러면 ①은 **동률**로 나오고, 그때
#   읽을 것은 ②③ (미탐·흔들림·MAE 가 **안 나빠졌나**) 뿐이다. 동률이면 **증분 없음**이 답이다.
#
# 관문 `src/run_gate_phasejit.py` 여덟 줄:
#   [2] 경로 census 측정 = 표 (세밀 1~7,9~15,31~38 · 광역 27~34)
#   [3] ★★ 라벨 안전 — P·Q·V·크기비·역률·짝수블록 4.77e-07 · |I_h| 불변 3.58e-07
#   [7] even_jitter 와 **채널 겹침 없다**   [8] h=1 정확히 불변
#
# **반증 가능한 예측**
#   ① 게이트 6/6 **유지** (evenjit40 이 이미 6/6 이면 동률이 상한이다)
#   ② ★ 유령창이 evenjit40 의 중앙 1 **아래로** 가거나 최소한 안 는다
#   ③ ★ 미탐(통전검출 83.7%)·흔들림(2.53)·MAE(~4.8)·F1(~0.932) 이 **안 나빠진다**
#   ④ 체크포인트에 `odd_phase_jitter: 18.7` 과 `even_jitter: 0.4` 가 같이 적힌다
#
# ⚠ -t 는 01:20:00 이다 — 14.246 이 00:55:00 으로 씨앗 둘을 TIMEOUT 냈다 (측정 38분 x 1.5 = 57분).
#SBATCH -J nilm_both
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:20:00
#SBATCH -a 0-5
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
$PY -X utf8 -m src.run_gate_evenjit
$PY -X utf8 -m src.run_gate_phasejit
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_evenjit40`)과 **글자 그대로 같아야** 한다 — 다른 것은 위상 지터 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40 --odd-phase-jitter 18.7"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $TREAT   (짝 evenjit40 은 --odd-phase-jitter 만 없다)"
case "$BASE$WTAP$CAP" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--even-jitter*|*--odd-phase-jitter*) echo "처치가 바닥으로 샜다"; exit 1;; esac
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--fine-dc*)
  echo "기각된 손잡이나 다른 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_bothjit_s${SEED} > "results/cnn_bothjit_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_bothjit_s${SEED}.log"
ls -la results/cnn_bothjit_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
