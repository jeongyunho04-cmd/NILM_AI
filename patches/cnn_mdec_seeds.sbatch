#!/bin/bash
# 14.147B — **마스크 독립 손실** (`--on-power-praw`). 시드 셋 짝비교.
#
# 사용자: *"전력 회귀 헤드에 게이트를 곱해서 손실을 계산하지 않습니다. 대신 실제 정답
#   라벨이 ON 인 구간에서만 p̂ 에 회귀 손실을 부여합니다. 이렇게 하면 게이트가 불완전하게
#   열려 있더라도 전력 헤드가 값을 억지로 부풀리는 보상 왜곡이 원천 차단됩니다."*
#
# ★ 이것이 `--on-detach-gate`(A) 를 **포함하면서 더 간다**. 관문 [10] 이 잰 값:
#     A  L = huber(**gate.detach()** x p_raw, y)  -> p_raw 경사에 gate 가 곱해져 있다
#        => p_raw 가 **y/gate** 로 밀린다 (게이트 0.5 면 2배로 부풀려 보상) = 왜곡이 남는다
#     B  L = 1(y_on) x huber(**p_raw**, y)        -> 게이트가 식에 없다. p_raw -> y 로 곧장
#     참ON 창 p_raw 경사 합: A 0.000297 -> B **0.000576 (1.94배)**. 그 1.94 가 곧 1/gate 다.
#   둘은 같이 못 쓴다 (관문 [13]). A 는 985024 로 같이 돌아 **대조**가 된다.
#
# 사용자: *"게이트랑 전력이랑 결합되어 있으니까, 학습할 때도 그렇고 필연적으로
#   불확실성이 생길 수밖에 없는 전이 순간에 게이트가 전력 예측을 흔들면서 문제를
#   발생시킬 수도 있나?"*
#
# **그렇다. 14.146 이 쟀다.** `power = s(on_logit)*p_raw` 라 전력 손실의 하강 방향이
# 둘이다 — `p_raw` 를 낮추거나 **게이트를 닫거나**. 합성 홀드아웃 8000창·3시드에서
# **참ON 창**의 게이트와 (참전력/슬롯 비) 상관:
#   에어컨 **0.777** · **오븐 0.618** · 포트 0.333 · 선풍 0.261 · 미니PC 0.233
#   (순수 검출기면 0 이어야 한다 · 참OFF 게이트는 0.0000~0.0020 으로 멀쩡하다)
#   참ON 인데 게이트<0.9 인 창: 미니PC 46.2% · 에어컨 39.3% · 선풍 22.2%
# 그리고 13.98 이 이미 한 번 봤다 — "작은 와트 슬롯이 없으면 '켜짐인데 0W' 를 표현할
#   수 없어 게이트가 0 으로 간다". 핫플 재현 0.41 -> 0.990 이 슬롯을 옮기기만 해서 났고
#   전력 출력은 그대로였다.
#
# 처치는 `--off-detach-praw`(13.11) 의 **짝**이다:
#   참OFF 창   p_raw 를 끊는다   -> 게이트가 0 으로 만드는 일을 맡는다  (기존, 지금 꺼져 있다)
#   참ON  창   **게이트를 끊는다** -> p_raw 가 진폭을 맡는다            (이번)
# ⚠ **값은 비트 동일이다.** detach 는 경사만 끊는다. 추론 경로도 저장 가중치의 뜻도 그대로다.
#
# ★ 왜 v1(pcap) 위인가: pcap 이 **채택된 기준선**이라 다른 게 하나뿐이다. hv2 는 보류이고
#   (★1 이 씨앗별 0.039/0.023/6.057 로 갈리고 통전 자가 나빠졌다) 거기 얹으면 둘이 다르다.
#   통과하면 그때 hv2 짝을 얹어 2x2 로 본다.
#
# ★ 던지기 전에 **잰** 기준값 — 짝은 `cnn_pcap_s{0,1,2}` (BASE·WTAP·상한 전부 같고
#   `--on-detach-gate` **하나만** 다르다)
#     ★1 면적 4.462 +- 1.820   ★2 축퇴 35.52 +- 8.71   ★4 마스크 F1 0.893 +- 0.004
#     저항 4종 신원 0.9774 +- 0.0029   SMPS 3종 신원 0.9807 +- 0.0087
#     판정 줄(7종) 0.8820 +- 0.0106    절대 잔차 21.27 +- 1.40W
#     포트창 오븐통전 24.20 +- 7.95%   겹침창 오븐통전 6.48 +- 2.11%
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 학습이 돈다. 값이 아니라 경사만 바뀌므로 첫 에포크 손실이 대조와 **같다**.
#   ② ★ **진짜 물음 — 게이트가 검출기로 돌아오나.** 참ON 창에서 게이트가 1 에 더 붙고
#      (게이트<0.9 비율이 내려간다) 게이트-비 상관 r 이 오븐 0.618 에서 내려간다.
#      이것이 **처치가 걸렸다는 증거**이고, 여기서 안 움직이면 나머지 해석은 무의미하다.
#   ③ **통과 조건은 '안 나빠진다'** 다 — 저항 4종 신원 · 판정 줄 · 절대 잔차 · ★4 의
#      짝차가 전부 씨앗 바닥 안. 통전 자(포트창·겹침창)는 **관찰**한다.
#   ④ **통전 헛detect 가 줄어들 것으로는 보지 않는다.** 14.146 은 상관만 보였고 인과는
#      안 보였다. 줄면 소득이고, 안 줄면 "게이트는 증폭기였지 방아쇠가 아니다" 가 된다.
#   ⑤ ⚠ 위험: `p_raw` 가 참ON 창의 전력 손실을 혼자 지면 상태 전력 머리가 흔들릴 수 있다.
#      13.11 이 반대 방향에서 본 병이다 (드라이 s2 가 0W 로 죽었다). **혼자켜짐 전력오차**와
#      **상태별 지문**을 같이 본다.
#
#SBATCH -J nilm_mdec
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984953(같은 바닥)이 45분 20초였다. 경사만 바뀌므로 속도는 같다.
# ⚠ **-t 를 넉넉히 잡으면 백필이 안 된다** (14.147). 90분으로 뒀더니 206개 뒤에 서서
#   `Reason=Priority` 로 멈췄는데, HPC_RULES §2 대로 1.5배(45:20 -> 68분)에 맞춰
#   **70분으로 줄이자 즉시 배치됐다.** 파티션을 넓히는 것(gpu1~gpu6)만으로는 안 풀렸다.
#SBATCH -t 01:10:00
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
$PY -X utf8 src/run_gate_ondetach.py
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--on-power-praw" \
  || { echo "--on-power-praw 가 없다"; exit 1; }

# ⚠ BASE + WTAP + CAP 이 cnn_pcap_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_pcap_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--on-power-praw"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP" in *--on-power-praw*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116 · 14.122 · 14.128 · 14.131 · 14.16)
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--fine-pad*|*--vexp*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# 처치의 전제: 사영이 꺼져 있어야 한다 (관문 [8] 이 같이 못 씀을 확인한다)
case "$BASE$WTAP$CAP$TREAT" in *--proj*) echo "사영과 같이 못 쓴다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_mdec_s${SEED} > "results/cnn_mdec_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_mdec_s${SEED}.log"
ls -la results/cnn_mdec_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
