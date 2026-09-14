#!/bin/bash
# 14.56 — **`sig` 의 파형 몫도 앵커한다** (`--harm-vhrel-anchor`). 3시드 x 2팔 (f=0.25 / 0.5).
#
# 무엇이 틀려 있었나:
#   순저항은 `sig_h = median(I_h/P) = v_h_rel,h / V_1` 인데 14.49 의 앵커는 **`1/V_1` 쪽만**
#   고쳤다. `v_h_rel = V_h/V_1` 은 **그 녹화 세션 값 그대로 박제**돼 있었다:
#       녹화 h3   오븐 0.73%% · 핫플 0.68%%  (자리 D)   |   나머지 여섯 2.81~3.67%%  (자리 E)
#   14.55 가 실측 대 녹화 비를 쟀다 (고전력 창은 전부 자리 D):
#       오븐   h7 0.81 · h11 0.61        <- 같은 자리인데 **세션**이 다르다
#       포트   h3 **0.22** · 드라이 h3 **0.24**   <- E 녹화 · D 창, **4.6배**
#   13.69 이 **생성기**에서 없앤 바로 그 가짜 판별자인데 (`apply_site_distortion`),
#   **손실의 `sig` 에는 그 고침이 없었다.**
#
# 고침: 생성기와 **같은 덧셈 꼴**로 건다 — `sig_h += sig_1 · f · (rel_창 − rel_녹화)`.
#   곱셈이 아니라 덧셈인 까닭: 기기 자신의 지문(I_1 의 0.04~0.50%%)을 **안 건드린다** (13.69).
#   `rel_창` 은 **모델 입력에 이미 있다** (세밀 45~56) — 캐시를 다시 안 굽는다.
#   `rel_녹화` 는 `vtexture.file_rel(stem)` — 생성기가 `rel_rec` 으로 쓰는 **바로 그 객체**다.
#
# ⚠⚠ **처음부터 분수다.** 14.49 가 전량을 걸었다가 부호를 넘겼다 (오븐 −2.9%% -> +2.3%%).
#   기본 `--harm-vhrel-frac 0` = 끔 = **비트 동일**. 여기서 0.25 와 0.5 를 건다.
#
# ★ **반증 가능한 예측** — 자와 씨앗 바닥을 **미리** 쟀다:
#   ① ★ **전압 블록 민감도가 준다** (`run_diag_vblock.py`). 손실이 자리 차이를 직접 다루면
#      모델이 V_h 를 **창의 지문**으로 쓸 이유가 준다.
#          지금 본선 (3시드)  SMPS **3.4 ± 0.2%%** · 저항 1.06 ± 0.10%% · 뒤집힘 1.36 ± 0.11%%
#      **짝차 폭 0.54%%p 라 1.1%%p 넘게 줄어야 읽힌다.**
#   ② ★ **신원이 오른다.** 포트·드라이의 h3 이 D 창에서 4.6배 틀려 있었으므로 이것은
#      *수준*이 아니라 **판별** 문제다. `run_diag_rollback` 의 `저항 4종 신원`(지금 0.9815,
#      네 팔 전부 같다)과 `SMPS 3종 신원`(0.9640)이 겨냥이다.
#   ③ ⚠⚠ **오븐의 수준은 오히려 나빠질 수 있다 — 미리 적어 둔다.** 오븐은 `|sig|` 가 h5 에
#      지배되는데 실측 h5 가 녹화보다 **작다**(1.603 대 1.753). 보정하면 `sig` 가 작아지고
#      `L_harm` 은 전력을 **더** 요구한다 — 지금 +24.9 ± 4.5W 과대인 것이 **더 커진다.**
#      그래도 거는 까닭: 그것은 **또 하나의 상쇄 오차를 걷어내는 것**이고 (14.52·14.55 와
#      같은 형태), 그 아래를 봐야 진짜 원인에 닿는다. **나빠져도 그것이 결론이다.**
#   ④ ⚠ 판정 줄(on/off 듀티제외 7종)은 씨앗 폭(0.006~0.018) 안이면 무승부로 읽는다.
#   ⑤ ⚠ **SMPS 세 종은 이 보정을 안 받는다** (`I_h = V_h/R` 이 없다). `_vhrel_on` 이
#      정확히 0 인 것을 `run_gate_vhrel` ④ 가 확인한다 — 움직이면 배선이 샌 것이다.
#
# ⚠ `VFRAC`(14.51 의 크기 앵커)은 **983591 의 승자**로 맞춰라. 두 손잡이가 겹치므로
#   한 판에 둘을 바꾸지 않는다 ([[count-how-many-things-differ-before-attributing]]).
#
#SBATCH -J nilm_vhrel
#SBATCH -p gpu3,gpu4,gpu5,gpu2,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 983591 이 21:56~22:23 이었다.
#SBATCH -t 00:35:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}
EPOCHS=${EPOCHS:-300}
VFRAC=${VFRAC:-1.0}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · vnorm-frac ${VFRAC} · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문: **함수가 아니라 배선을 잰다** (983506 이 45분을 버렸다) ──────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-vhrel-anchor" || {
  echo "--harm-vhrel-anchor 가 없다 — 코드를 안 올렸다"; exit 1; }
# ② 가 제일 위험한 줄이다 — `_vhrel_from_fine` 이 `build_inputs` 를 정확히 되돌리는가.
#   (허용 1e-6. 처음에 기준 색인을 틀려 1.36e-4 가 나왔는데, `build_inputs` 가 창의
#    **뒤 600사이클**만 쓴다는 것을 놓쳐서였다. 지금은 1.9e-9 다.)
$PY -X utf8 -m src.run_gate_vhrel  || { echo "vhrel 배선 관문 실패"; exit 1; }
$PY -X utf8 -m src.run_gate_vanchor || { echo "vanchor 배선 관문 실패"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다"; exit 1; }

BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor --harm-vnorm-frac $VFRAC \
      --harm-vhrel-anchor"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vhrel-frac 0.25 \
    --tag cnn_vhrel_25_s${SEED} > "results/cnn_vhrel_25_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vhrel-frac 0.5 \
    --tag cnn_vhrel_50_s${SEED} > "results/cnn_vhrel_50_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== f=0.25 ==="; tail -32 "results/cnn_vhrel_25_s${SEED}.log"
echo "===================================================================================="
echo "=== f=0.50 ==="; tail -32 "results/cnn_vhrel_50_s${SEED}.log"
ls -la results/cnn_vhrel_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 (로컬, 실측 복합 5파일). **이 순서로**:"
echo "  ① ★ python -X utf8 src/run_diag_vblock.py --pair 3 --ckpt <끔 셋> <켬 셋>"
echo "       겨냥: SMPS 3.4%% -> ? · **1.1%%p 넘게 줄어야 읽힌다**"
echo "  ② ★ python -X utf8 src/run_diag_rollback.py --postproc off --ckpt <여섯>"
echo "       겨냥: 저항 4종 신원 0.9815 · SMPS 3종 0.9640"
echo "  ③ python -X utf8 src/run_diag_levelerr.py --ckpt <여섯>   (오븐은 **나빠질 수 있다**)"
echo "  ④ python -X utf8 src/run_diag_hipow.py --pair 3 --ckpt <여섯>"
exit $RC
