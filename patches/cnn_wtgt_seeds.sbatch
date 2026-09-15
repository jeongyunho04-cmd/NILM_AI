#!/bin/bash
# 14.73 — **광역에 "지금" 을 알려준다** (`--wide-target`, 13.44). 200판 만에 처음 켠다.
#
# 무엇을 고치려는가 (14.72 가 잰 것):
#   오븐 헛detect 가 **전환에 묶여 있다** — ±10초 안 25~32%, **±20초 밖 0.0%**.
#   부호 있는 오프셋으로 보면 전환 **전**부터 오른다 (−12초 17% · −6~−4초 31%).
#   고원의 심장 [−6, +4]초가 세밀 창 [t−3.98, t+6.00] 이 계단을 **품는** 구간과 같다.
#   => 모델은 "이 창 어딘가에 큰 계단이 있다" 까지만 알고 **그게 미래인지 모른다.**
#
# 시간 정보를 세어보면 (`net.py` forward):
#   fine[:,:,t] 1사이클 · 탭0 7 · 탭1 19 · 깊은탭 187(±1.56초)   <- 위치를 안다
#   h.mean/h.amax · fp.amax/amin · hw.mean                      <- **창 전체, 위치를 모른다**
#   광역 타깃 슬라이스 · 광역 amax/창끝                           <- **아예 없다**
#   => 광역이 머리에 주는 것은 `hw.mean(-1)` 하나고 60초에 대해 **순서 불변**이다.
#      60초 맥락을 주면서 "지금이 그 안 어디인지" 를 안 준다. seq2point 인데.
#
# 체크포인트 200개를 훑었다: `wide_target` 이 **200/200 전부 False** 다. 한 번도 안 켰다.
# 13.44 가 재둔 값: 미니PC 0.795(60초 문맥만) / 0.680(순시값만) -> **둘 다면 0.937**.
#
# ⚠ `--seg-pool 2` 는 **이미 이 병에 대해 널로 확인했다** (14.72): `cnn_segp_2_*` 3시드로
#   0~20초 띠 짝차 **−0.9 ± 2.0 %p** (시드별 −3.2 +0.5 +0.0). 시드0만 보면 −3.2 라 솔깃하다.
#   그래서 여기서는 **`--wide-target` 을 혼자 세우고**, seg-pool 은 그 위에 얹어 사다리로 본다.
#
# 기준선: 오늘 22:00 에 같은 조리법으로 돌린 `cnn_v32h_off_s{0,1,2}`.
#   (그것이 옛 `cnn_vanch_on_*` 과 아홉 줄 전부 숫자가 같은 것을 확인했다 — 표류 없음.)
#   그래서 GPU 두 장을 **둘 다 처치에 쓴다.**
#
# ★ 미리 적는 판정 (나중에 고르지 않으려고)
#   자:   `run_diag_fpage.py` 의 **전환 이후 0~20초 띠** (오븐이 안 켜진 파일만)
#   기준: 17.8 ± 5.8 % (시드별 22.9 · 11.5 · 18.9) — 씨앗 폭이 크니 **짝차로만** 읽는다
#   문턱: 짝차 폭이 2.0%p 였으니 3시드 SE 1.15%p. **−2.5%p 를 넘어야 읽은 것이다.**
#   ⚠ 같이 볼 것 (한 자만 보고 채택하지 않는다):
#     · `run_diag_rollback --postproc off` 판정 줄 (기준 0.9133 ± 0.0054, 씨앗 바닥 .006~.018)
#     · 상대 잔차 (기준 3.93%) · 핫플통전 P>1500 (기준 −16.3 ± 7.5W, **되돌아가면 안 된다**)
#     · `run_diag_resconf` 자리 E 어긋난 창 (기준 59창 중 53창이 오븐 헛detect)
#
#SBATCH -J nilm_wtgt
#SBATCH -p gpu1,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 984136 이 3090 에서 27분이었다.
#SBATCH -t 00:50:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--wide-target" || {
  echo "--wide-target 이 없다 — 코드를 안 올렸다"; exit 1; }

# ⚠⚠ 인자를 한 곳에 모아 **학습 전에** 되짚는다 (14.69 가 이 자리에서 조용히 어긋났다).
#   파일 전체가 아니라 **이 명령줄 자체**를 본다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
A_FLAGS="--wide-target"
B_FLAGS="--wide-target --seg-pool 2"
echo "  BASE: $BASE"
echo "  갈래 A: $A_FLAGS"
echo "  갈래 B: $B_FLAGS"
case "$A_FLAGS" in *--wide-target*) ;; *) echo "A 에 --wide-target 이 없다"; exit 1;; esac
case "$B_FLAGS" in *--seg-pool*) ;; *) echo "B 에 --seg-pool 이 없다"; exit 1;; esac
case "$BASE" in *--wide-target*|*--seg-pool*) echo "BASE 에 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_wtgt_on_s${SEED} > "results/cnn_wtgt_on_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_wtgt_sp_s${SEED} > "results/cnn_wtgt_sp_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: --wide-target ==="; tail -32 "results/cnn_wtgt_on_s${SEED}.log"
echo "===================================================================================="
echo "=== B: --wide-target --seg-pool 2 ==="; tail -32 "results/cnn_wtgt_sp_s${SEED}.log"
ls -la results/cnn_wtgt_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 — 기준선은 이미 있는 results/cnn_v32h_off_s{0,1,2}.pt 다."
echo "  1 ★ run_diag_fpage.py  전환 이후 0~20초 띠 (기준 17.8 +- 5.8%, 짝차 -2.5%p 넘어야 읽음)"
echo "  2 run_diag_rollback --postproc off   판정 줄 (기준 0.9133) · 상대 잔차 (기준 3.93%)"
echo "  3 run_diag_hipow --arms  핫플통전 P>1500 (기준 -16.3 +- 7.5W, 되돌아가면 안 된다)"
echo "  4 run_diag_resconf  자리 E 어긋난 창 (기준 59창)"
