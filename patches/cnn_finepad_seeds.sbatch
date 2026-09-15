#!/bin/bash
# 14.131 — 세밀 conv 패딩을 **복제**로 (`--fine-pad replicate`). 시드 셋 짝비교.
#
# ★★ 14.116 은 진단 **개입**과 학습 **처치**를 같은 것으로 취급했다. **아니었다.**
#   (나) 240칸으로 잘라서 — **0 패딩**        `--fine-time-split` 이 한 것
#   (다) 600칸이되 미래를 **타깃값 복제**      진단 **개입**이 한 것
# 둘 다 "미래를 안 본다" 인데 오른쪽 경계값이 **0** 이냐 **마지막 값**이냐가 다르다.
# `asinh` 눈금에서 0 은 *"고조파가 0"* 이라 실측 창에 없는 값이고, 타깃은 조각의
# **맨 끝**이라 바로 옆이 전부 그 값이 된다. 실측에서 잰 차이 (cnn_pcap_s0, 타깃 위치):
#   블록 0 (RF **7**)   (나)0패딩 대 (다)복제  cos **0.825**  <- RF 7 인데 이미 갈린다
#                       (다)복제 대 (가)600    cos **0.994**
# 그리고 결과가 갈렸다 — 개입은 고쳤고(포트 0.003 -> **0.938** · 핫플 0.878 -> 0.004),
# 처치는 ★2 를 **악화**시켰다 (+7.10 +- 3.09).
#
# 패딩 비중도 크다. 수용영역이 창을 **넘친다**:
#   블록  d    RF   | 600(창)   360(미래)  240(과거)  <- RF/길이
#   5    32   379  |  63.2%    105.3%    157.9%
#   6    64   763  | 127.2%    211.9%    317.9%
#   탭 중 패딩 비율  |  18.3%     30.5%     45.7%
# 넘치는 만큼이 **전부 0** 이다.
#
# 관문 `run_gate_finepad.py` [3] 이 결정적이다 — `replicate` 로 240칸을 태운 블록 0 의
# **conv 출력**(GroupNorm 전)이 개입(600 복제)과 **최대차 0** 으로 비트 동일이다.
# 0 패딩은 최대차 1.12 다. ⚠ GroupNorm 뒤로는 결코 같아질 수 없다 (통계가 다르다).
#
# ★ 기준값 — 짝은 **채택된 `cnn_pcap_s{0,1,2}`** (`--fine-pad` 하나만 다르다)
#     ★1 4.462 +- 1.820 · ★2 35.519 +- 8.709 · ★4 0.893 +- 0.004
#     저항 4종 0.9774 +- 0.0029 · SMPS 3종 0.9807 +- 0.0087
#     판정 줄 0.8820 +- 0.0106 · 절대 잔차 21.27 +- 1.40W
#     전이 ±20초 안 오븐 헛게이트 **23.3%** (밖은 0.0%)
#
# **반증 가능한 예측**
#   ① (확인) 파라미터 불변(887,829). 학습이 돈다.
#   ② ★ **진짜 물음** — 0 패딩이 만들던 가짜 계단이 사라지니 지킴이 **오른다**.
#      저항 4종 · SMPS 3종 · ★4 중 **하나라도 3/3** 으로 오르면 이 줄이 산다.
#   ③ ★2 는 **안 움직일 것으로 본다** — 패딩은 `--fine-time-split` 을 안 켠 v1 에서는
#      창 600 기준 18.3% 라 영향이 작다. 움직이면 그만큼 큰 소득이다.
#   ④ **이 판이 통과해야 v2 를 읽을 수 있다** — v2 는 조각을 더 잘게 내므로 패딩
#      비중이 더 크다. 패딩이 범인이면 v2 판정이 통째로 오염된다.
#
#SBATCH -J nilm_pad
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 얕은 스택을 5번 더 태우므로 세밀 연산이 늘지만 조각이 120칸이라 실제로는 1.3배 안쪽이다.
#SBATCH -t 01:30:00
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
$PY -X utf8 src/run_gate_finepad.py
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--fine-pad" \
  || { echo "--fine-pad 가 없다"; exit 1; }

# ⚠ BASE + WTAP + CAP 이 cnn_pcap_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_pcap_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--fine-pad replicate"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP" in *--fine-pad*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116 · 14.122 · 14.128)
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--head-drop*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# v2 의 전제: 얕은 탭이 `tap_layers` 의 두 번째다 — 배치가 다르면 RF 가 어긋난다
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "탭 배치가 다르다 — 얕은 탭 RF 가 어긋난다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_pad_s${SEED} > "results/cnn_pad_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_pad_s${SEED}.log"
ls -la results/cnn_pad_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
