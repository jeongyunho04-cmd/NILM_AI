#!/bin/bash
# 14.251 — **DC 를 크기 경로에만 되돌린다** `--fine-dc mag`. 짝은 **기존** `cnn_dzn`(씨앗 0~2).
#
# 14.247 이 잰 것 — `split` 은 샀고, 동시에 값을 치렀다:
#   샀다   실측 포트 게이트 **6/6** (이항 p=0.009, 기준 0.46) · DC 만 주입해도 안 뒤집힘
#          (짝은 같은 주입에 0.0005) · 편향 아님 (참OFF 게이트 0.000 · ON/OFF AUC 0.9922)
#   치렀다 홀드아웃 MAE 4.65 -> **7.75** · F1 0.933 -> 0.904 (여섯 대 셋, **겹침 0**)
#          예측 흔들림 정지창 |ΔP| 3.06 -> **7.40** W/0.5초 · **계단근처 4.74배** (14.250)
#          유령창은 안 줄었다 (56 -> 64) — 맞바꿈은 고쳤지만 유령은 다른 병이다
#
# 까닭 (14.243·14.244): `ch23` 의 창 평균 = 절대 전력 수준이 **두 일을 한꺼번에** 한다 —
#   숏컷이다 (오븐/큰저항부하 신원 AUC 합성 **0.943** -> 실측 **0.521**)
#   신호다   (진짜 크기 정보라 빼면 MAE 가 오른다)
# 그러니 물음은 "빼느냐 두느냐"가 아니라 **어느 경로에 주느냐**다. RevIN(ICLR 2022)의
# 표준이 정확히 이 대칭이고(입력에서 빼고 **출력에서 도로 넣는다**), `split` 은 앞 절반만
# 했다. `mag` 가 뒤 절반이다 — DC 는 몸통에 안 들어가고 `p_states`(크기)에만 더해진다.
#
# ⚠ **반례를 먼저 적는다.** `mag` 은 신원 경로에서 DC 를 **split 과 똑같이** 끊는다.
#   그러니 ①이 유지되는 것은 예측이 아니라 **요구**다. 만약 ②·③이 안 돌아오면
#   비용의 원인이 "DC 를 신원에서 끊은 것" 자체이고, 이 처방은 **틀린 것이다.**
#
# 관문 `src/run_gate_finedcmag.py` 여덟 줄. 핵심 둘:
#   [3] dc_pow 를 N(0,5) 로 흔들어도 on_logit·state·plugged·standby·power_mix 가
#       **전부 정확히 0.0e+00** 움직이고 power_states 만 12.390 움직인다
#   [4] autograd — ∂|on_logit|/∂dc_pow = **0.000e+00** · ∂|power_raw|/∂dc_pow = 6.85e-02
#   옛 판과 안 겹침도 따로 못 박았다 — 옛/새 net.py 로 이미 구운 finedc·dzn 을 같은
#   입력에 돌려 **12개 비교 전부 0.000e+00**.
#
# **반증 가능한 예측**
#   ① 게이트 6/6 이 **유지**돼야 한다 (split 이 산 것을 안 잃는다)
#   ② ★ 홀드아웃 MAE 가 7.75 -> **4.65 쪽으로** 돌아와야 한다 (되돌림이 값을 갚는가)
#   ③ ★ 정지창 흔들림 7.40 -> **3.06 쪽으로** (몸통이 DC 를 안 보니 창 위치 민감도가 없다)
#   ④ 파라미터가 짝보다 **정확히 +2,610** (= 57 x 45 + 45) · `fine_dc: "mag"` 기록
#
# ⚠ 짝 `cnn_dzn` 은 **다시 안 굽는다.** 씨앗 셋으로 충분하다 — ①은 57판에서 확립된
#   p0=0.46 에 견주고(dzn 이 기준이 아니다), ②·③은 dzn 씨앗 폭이 MAE 0.29 · 흔들림
#   0.19 로 아주 좁아 이미 6대3 짝짓기 18쌍 전부 겹침이 0 이었다.
#SBATCH -J nilm_mag
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
$PY -X utf8 -m src.run_gate_finedc
$PY -X utf8 -m src.run_gate_finedcmag
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_dzn`)과 **글자 그대로 같아야** 한다 — 다른 것은 ARCH 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--fine-dc mag"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP"
echo "  처치: $ARCH $TREAT"
case "$BASE$WTAP$CAP" in *--even-median*|*--w-cons*|*--cons-deadzone*|*--fine-dc*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*|*--even-jitter*)
  echo "기각된 손잡이나 다른 처치가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT \
    --tag cnn_finedcmag_s${SEED} > "results/cnn_finedcmag_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_finedcmag_s${SEED}.log"
ls -la results/cnn_finedcmag_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
