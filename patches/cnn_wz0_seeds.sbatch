#!/bin/bash
# 14.181 — **선로 임피던스 보조 감독을 끈다.** 짝 `cnn_dzn` — `--w-z` 하나만 다르다.
#
# 왜 이 판이 없었나: 56판 전부 `aux_z=True` 다. `--w-z 0.3` 이 **모든 팔의 바닥**에
# 박혀 있어 **한 번도 안 꺼봤다**.
#
# 14.180 이 잰 것 (dzn_s0 · 실측 5파일 5,487창):
#   · 임피던스 추정이 **"창 안에 계단이 있나"** 에 붙어 있다
#       계단 있음  log_z −0.228 (Z **0.796Ω**)    없음 −0.850 (Z **0.427Ω**)   차 +0.622
#       **미래 6초에만** 계단이 있어도 차 **+0.605** — 거의 같다. **위치와 무관**하다
#   · `log_z` 와 **오븐 예측W 의 상관 +0.501** · cos(z_head, 오븐on) +0.264/+0.239/+0.257 (3판)
#   => 보조과제가 몸통에게 *"창 안 어디든 계단을 기억해라"* 를 **타임스탬프 없이** 요구한다.
#      그리고 오븐 오탐의 방아쇠가 바로 **미래 2사이클 계단**이다 (14.178).
#
# ⚠ **반대 증거도 같이 적는다** — 내 예측은 **오븐에 대해 널**이다:
#     경계에서 Δlog_z +0.027 (Z 0.399->0.410Ω) · **cos(Δz, z_head) = +0.005 (직교)**
#     |z_head 가중치| 가 병목칸에서 전체의 **하위 10~19% 분위**
#     추론에서 임피던스 축 1칸을 지우면 오븐 729 -> **976W 악화** (3판 모두)
#   그러나 추론에서 축을 지우는 것과 **학습에서 과제를 빼는 것은 다른 물건**이다
#   (14.153 [[the-intervention-is-not-the-treatment]]). 그래서 한 판이 필요하다.
#
# 문서 확인 (던지기 전에 뒤졌다):
#   13.55 의 제목이 **"진단은 틀렸고 부산물이 남았다"** 다.
#   13.55.1 이 원래 근거(13.54 의 Z 86% 폭 · 관문 붕괴 0.137)를 **분포 밖 외삽**으로
#   반증했다 — *"v23 은 이미 자기 학습 범위 안에서 Z 불변이었다 … 보조 헤드가 그 폭을
#   못 줄인 것이 아니라 **줄일 것이 없었다**"*.
#   그런데 인계 2026-09-08 은 `cnn_v24b(w_z 0.3)` 이 **운영점**이 됐다고 적는다 —
#   충전기 **+0.077** · 미니PC **+0.068** (실측 AUC, **씨앗 3개 전부 양수**).
#   => 근거는 죽었고 **이득은 남았다**. 그러니 이것은 공짜가 아니다.
#
# **미리 등록한 판정법** — 이득과 비용을 갈라서:
#   이득: 수백W 띠 초 (dzn 22.4/10.4/8.9) · 오븐 유령 게이트율 (dzn 6.8/5.0/3.4%) · p99
#   비용: ★ **충전기·미니PC 실측 AUC** (13.55 가 3/3 으로 적은 값이다. 여기가 무너지면
#            이득이 뭐든 기각) · 판정줄 7종 · 절대 잔차 (16.0/19.5/15.2W) · 혼자켜짐
#   ⚠ **오븐 유령 게이트율이 오르면 즉시 기각** (14.167 이 무너진 방식)
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① 학습이 돈다. `z` 항이 0 이므로 첫 에포크 total 이 `cnn_dzn` 과 **다르다**.
#      파라미터가 **257개 적다** (Linear(256,1)). 체크포인트에 `aux_z: false` 가 적힌다.
#   ② ★ 내 예측은 **오븐 유령·수백W 띠가 안 변한다**(널)이다. 직접 경로가 직교하므로.
#      **널이면 임피던스 결합 가설이 죽는다** — 그것이 이 판의 값이다.
#   ③ 비용 쪽은 **내려갈 것으로 예상**한다 (충전기·미니PC). 그 크기를 재는 것도 값이다.
#   ④ 만약 ②가 크게 좋아지면 — 그때는 **보조과제에 위치를 요구하는 쪽**(과거 계단만으로
#      Z 를 맞추게)이 다음 처방이다. 통째로 끄는 것이 아니라.
#
# 짝의 기준값 `cnn_dzn` (s0/s1/s2):
#   판정줄 .8497/.8368/.8518 · 저항 .9815 x3 · SMPS .9368/.9711/.8500 ·
#   절대 잔차 16.0/19.5/15.2W · 혼자켜짐 20.9/45.5/44.4%
#SBATCH -J nilm_wz0
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
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_dzn`)과 **`--w-z` 말고는 글자 그대로 같다.**
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH=""
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 이 판의 **유일한 차이**가 실제로 걸렸는지 확인한다 — 안 걸리면 짝과 같은 판이 된다.
case "$BASE" in *"--w-z 0 "*) : ;; *) echo "--w-z 0 이 안 걸렸다"; exit 1;; esac
case "$BASE" in *"--w-z 0.3"*) echo "--w-z 0.3 이 남아 있다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_wz0_s${SEED} > "results/cnn_wz0_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_wz0_s${SEED}.log"
ls -la results/cnn_wz0_s${SEED}.pt || true
$PY -X utf8 -c "import torch,sys; d=torch.load('results/cnn_wz0_s${SEED}.pt',map_location='cpu',weights_only=False); print('aux_z =', d.get('aux_z')); sys.exit(0 if d.get('aux_z') is False else 1)"
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
