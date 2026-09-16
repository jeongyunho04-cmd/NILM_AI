#!/bin/bash
# 14.317 — **쌓기 시험** `--wide-dg` + `--swa-start 220`. 짝은 `cnn_swa`(씨앗 여섯, 이미 있다).
#
# 사용자: *"swa 를 기준으로 비교하면 되잖아"* — 맞는 지적이다. `cnn_swa` 가 `evenjit40` 위에
# `--swa-start 220` **하나만** 얹은 판이라 셋이 같은 바닥을 쓴다. 이 팔이 넷째 칸을 채운다:
#
#     바닥 evenjit40 | +swa | +wide_dg | **+둘 다**   -> 2x2 가 완성된다
#
# 읽는 법 — 2x2 라 **쌓이는지**를 직접 본다:
#   (둘다 − swa)  대  (widedg − 바닥)   같으면 **가법**이다 (SWA 와 ΔĜ 가 독립)
#   (둘다 − swa) > (widedg − 바닥)      SWA 가 ΔĜ 를 **도운다**
#   (둘다 − swa) < (widedg − 바닥)      겹친다 — 둘이 같은 것을 고치고 있었다
#
# ⚠ SWA 판정은 §29.12 에서 **"해롭지 않지만 아무것도 안 샀다"** 였다. 산 것은 합성
#   홀드아웃 MAE 하나뿐이고(완전한 짝 swa_last 4.913 -> swa 4.819, 6 중 5, p=0.094),
#   1.8% 자·네 방향·게이트·통전은 **전부 동률**이었다. 추론 비용이 0 이라 얹어 보는 것이지
#   **SWA 가 이득이라는 전제로 던지는 게 아니다.**
#
# ⚠ `--select final` 은 짝(`cnn_swa`)에서 그대로 베꼈다 — SWA 는 마지막 에포크의 평균을
#   써야 하는데 `--select` 가 홀드아웃 최고점을 고르면 평균이 아닌 판이 저장된다.
#
#SBATCH -J nilm_wds
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
# ⚠ §7-3 — **코드가 올라왔는지 먼저.** 파이프로 grep 하면 `set -o pipefail` 아래에서
#   조기 종료가 SIGPIPE 로 잡힐 수 있어 변수로 받아서 본다.
HELP=$($PY -X utf8 -m src.run_train_cnn --help)
case "$HELP" in *--wide-dg*) : ;; *) echo "run_train_cnn 에 --wide-dg 가 없다 — 코드를 안 올렸다"; exit 1;; esac
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_evenjit
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
# ⚠ `run_gate_widedg` 는 원시 열 캐시(`cache/seqraw_k`)가 있어야 돈다. HPC 에 없으면
#   건너뛴다 — 그 관문은 **노트북에서 8/8 로 통과시켰다** (14.315).
if [ -d cache/seqraw_k ]; then $PY -X utf8 -m src.run_gate_widedg; else echo "  (seqraw_k 없음 — widedg 관문 건너뜀)"; fi

# ⚠ BASE·WTAP·CAP·TREAT 는 짝(`cnn_evenjit40`)과 **글자 그대로 같다**. 다른 것은 WDG 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40"
WDG="--wide-dg --swa-start 220 --swa-every 1 --select final"

echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $TREAT"
echo "  처치: $WDG   (짝 cnn_swa 는 --wide-dg 만 없다)"
case "$BASE$WTAP$CAP$TREAT" in *--wide-dg*|*--swa-start*|*--swa-every*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.296·14.303)
case "$BASE$WTAP$CAP$TREAT$WDG" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--head-conductance*|*--hcond-*|*--odd-phase-jitter*|*--fine-dc*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# ⚠ `--wide-dg` 는 `--head-layout v2` 와 못 쓴다 (v2 는 `_conv_in` 을 안 탄다). 생성자가 막지만
#   여기서도 막아 **던지기 전에** 잡는다.
case "$BASE$WTAP$CAP$TREAT$WDG" in *"--head-layout v2"*) echo "wide_dg 는 head_layout=v2 와 못 쓴다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT $WDG \
    --tag cnn_wdgswa_s${SEED} > "results/cnn_wdgswa_s${SEED}.log" 2>&1
RC=$?
echo "=== 끝 $(date '+%F %H:%M:%S')"

# ⑤ 를 그 자리에서 읽는다
$PY -X utf8 -c "
import torch, sys
p = torch.load('results/cnn_wdgswa_s${SEED}.pt', map_location='cpu', weights_only=False)
print('  체크포인트: wide_dg=%r even_jitter=%r fine_derive=%r head_layout=%r'
      % (p.get('wide_dg'), p.get('even_jitter'), p.get('fine_derive'), p.get('head_layout')))
print('  swa_start=%r swa_n=%r' % (p.get('swa_start'), p.get('swa_n')))
ok = (p.get('wide_dg') is True and float(p.get('even_jitter', 0)) == 0.4
      and p.get('head_layout') == 'v1' and int(p.get('swa_start', 0) or 0) == 220
      and int(p.get('swa_n', 0) or 0) > 1)
print('  ⑤ %s' % ('통과' if ok else '**실패 — 처치가 안 실렸다**'))
sys.exit(0 if ok else 1)
"
tail -32 "results/cnn_wdgswa_s${SEED}.log"
ls -la results/cnn_wdgswa_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
