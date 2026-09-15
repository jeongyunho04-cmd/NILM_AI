#!/bin/bash
# 14.122 — **미래에 시간축을 준다** (`--fine-future-segs 3`). 시드 셋 짝비교.
#
# 사용자: *"미래를 보고 그게 지금 판정 포인트랑 다르다는 걸 모델이 몰라? 그걸 알게
#   구조를 바꾼 거 아니었어?"*
#
# 바꿨고 모델은 **안다** — `run_gate_timesplit` [4] 가 타깃 탭이 과거만 넣은 것과 비트
# 동일임을 확인했다. 그런데 14.116 이 머리에 준 미래는 딱 둘이었다:
#     hf.mean(-1)   미래 360사이클 **전체** 평균
#     hf.amax(-1)   미래 360사이클 **전체** 최대
# **둘 다 순서에 불변**이라 "앞으로 6초 안에 큰 게 있다" 는 말해도 **"언제"** 는 못
# 말한다 — **+0.2초 뒤의 계단과 +5.9초 뒤의 계단이 같은 값**이다. "이건 미래다" 는
# 알려 줬는데 **"미래의 언제냐" 는 안 알려 줬다.**
#
# ★ 그리고 미래를 빼앗으면 **안 된다** (cnn_tsp 3시드 · 실측 5파일에서 잰 것):
#     머리 첫 층 차원당 기여 — `hf.amax` 가 **단일 덩이 1위 19.2%** (타깃 탭 14.9%)
#     미래 덩이 둘을 죽이면   오븐 헛게이트 10.8% -> **23.0%** (두 배)
#     세밀 미래 **입력**을 지우면              10.8% -> 12.7%
#   ⇒ 미래는 **순이득**이다. 지난 판에서 "미래 드롭아웃" 을 제안했다가 이 측정으로 접었다.
#
# ⚠⚠ **깊은 `hf` 를 토막내는 것은 소용없다.** 처음에 그렇게 짰다가 관문 [4] 가 굽기
#   전에 잡았다 — 미래 몸통의 수용영역이 **763** 인데 미래는 360칸뿐이라 `hf` 의 모든
#   열이 이미 미래 전체를 본다 (앞쪽만 흔들어도 토막 셋이 0.397/0.404/0.360 으로 똑같이
#   움직였다). `--seg-pool` 이 실패한 것과 같은 이유다. 그래서 **원시 미래 입력**을
#   토막낸다 — 수용영역이 정의상 **1** 이라 토막이 반드시 구별된다.
#   관문 [4] 재검: 앞흔듦 **4.085 / 0.000 / 0.000** · 뒤흔듦 **0.000 / 0.000 / 4.181**
#
# ★ 던지기 전에 **잰** 기준값 — 짝은 `cnn_tsp_s{0,1,2}` (같은 캐시·같은 글자, 깃발 하나 차이)
#     ★1 면적 4.435 +- 1.824   ★2 축퇴 40.984 +- 3.541 %
#     오븐 헛게이트율 10.8% (7.2 11.1 14.1)
#     저항 4종 신원 0.9815 +- 0.0000   SMPS 3종 0.9675 +- 0.0087
#     판정 줄(7종) 0.8952 +- 0.0028    절대 잔차 19.9 +- 0.6 W
#     test_5 127.9~142초 (참: 오븐 OFF · 포트 ~1200W) -> 지금 **오븐 1.00 · 포트 0.00** 으로 뒤바뀜
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 머리 입력 1021 -> **1363** (+2x3x57). 학습이 돈다.
#   ② ★ **직격** — `test_5` 130.0/135.0초에서 **오븐 게이트 <0.5 · 포트 >0.5**.
#      지금 tsp 는 오븐 1.00 / 포트 0.00 인데 **미래를 지우면** 오븐 0.41 / 포트 1.00 이
#      된다. 즉 그 창의 답은 미래의 **시각**에 달려 있다. 3시드 중 **2/3** 이상.
#      이게 이 판의 통과 조건이다.
#   ③ ★2 축퇴가 tsp(41.0%)보다 내려간다. ⚠ **크게 기대하지 않는다** — 오븐 헛게이트
#      중 미래 탓은 약 29% 뿐이고 나머지 71% 는 저항 4종 축퇴(0.1~1.1도, 조건수 2124)다.
#      hzwt 기준선 33.9% 밑까지 가면 큰 소득이고, 41 -> 36 만 돼도 방향은 맞다.
#   ④ 오븐 헛게이트율이 10.8% 에서 내려간다.
#   ⑤ 지킴: 저항 4종 신원 · SMPS 3종 · 판정 줄 · 절대 잔차 (전부 짝차)
#   ⑥ ★ **안 움직이면 이 줄을 놓는다.** ②가 0/3 이면 미래의 시간 귀속이 원인이 아니라는
#      뜻이고, 그러면 남은 것은 저항 축퇴 하나다. 그쪽으로 옮긴다.
#
#SBATCH -J nilm_fseg
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
# 14.112 측정대로 갈래당 1GPU·3코어. 더하는 것은 원시 슬라이스 풀링뿐이라 비용 ~0.
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984682(같은 바닥 + time-split)가 6초/에포크, 300에포크 약 33분이었다. HPC_RULES 2 로 1.5배 이상.
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
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_timesplit.py
$PY -X utf8 src/run_gate_futsegs.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--fine-future-segs" \
  || { echo "--fine-future-segs 가 없다"; exit 1; }

# ⚠ BASE + WTAP + TSP 가 cnn_timesplit_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_tsp_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
TSP="--fine-time-split"
TREAT="--fine-future-segs 3"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $TSP"
echo "  처치: $TREAT"
# 손잡이가 섞이면 이 판이 무엇을 재는지 알 수 없다
case "$BASE$WTAP$TSP" in *--fine-future-segs*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ `--p-state-cap` 은 **일부러 뺐다** — 984729 가 아직 판정 전이다. 섞으면 분리가 안 된다.
case "$BASE$WTAP$TSP$TREAT" in *--p-state-cap*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "다른 손잡이가 섞였다"; exit 1;; esac
# 처치의 전제: 미래 조각은 시간분할에서만 생긴다
case "$TSP" in *--fine-time-split*) ;; *) echo "시간분할이 없다 — 전제가 무너진다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $TSP $TREAT \
    --tag cnn_fseg_s${SEED} > "results/cnn_fseg_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_fseg_s${SEED}.log"
ls -la results/cnn_fseg_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
