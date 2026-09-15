#!/bin/bash
# 14.88 — **광역 갈래에 위치를 준다** (`--wide-seg-pool`). 시드 셋 짝비교.
#
# 사용자: *"근데 이건 애초에 광역이 비중이 낮아서 그런 거 아니야?
#            정보를 주고 학습을 시키면 일하지 않을까?"*  -> 맞는 지적이라 이 판을 낸다.
#
# 무엇을 고치려는가 (14.86~14.87):
#   test_5 의 오븐 통전->팬·조명 계단은 123.0초다. 세밀 창은 **과거 3.98초**뿐이라
#   123.0+3.98 = 127.0초, 즉 **t=128.0초 창부터** 그 계단이 창 밖으로 나간다.
#   세 모델 전부 **정확히 거기서** 무너진다:
#       초        124    126   **128**   132    136    138    140
#       기준선   0.001  0.000  **0.806** 0.949  0.995  0.432  0.008
#       wtap A   0.001  0.001  **0.051** 0.233  0.780  0.951  0.419
#       wtap B   0.000  0.000  **0.174** 0.305  0.760  0.532  0.014
#   광역은 `[t−54초, t+6초]` 라 그 계단을 177초까지 갖고 있는데 머리가 받는 것이
#   `hw[:, :, a:b].mean(-1)` — **시간 평균**이라 위치가 거의 지워진다.
#
#   14.87 이 **개입**으로 쟀다 (계단을 왼쪽으로 밀고 특징이 얼마나 움직이나,
#   `full` 대비 담는 몫):
#       mean **22%** · seg2 49% · seg4 67% · **seg6 78%** · **seg8 86%**   <- N 에 단조 증가
#
#   `--seg-pool` 은 세밀·광역·원시전력을 **한꺼번에** 바꾸고 세밀 쪽은 14.74 에서 2 로
#   해보고 못 읽었다. 필요한 곳은 광역뿐이라 거기만 쪼갠다.
#
# 관문 (`run_gate_wsegpool.py`, 전부 통과):
#   [1][2] 0/1 이면 **출력 비트 동일**   [3] trunk_in 이 정확히 (w2+2)x(N−1) 만 는다
#   [4] **세밀 유래 차원 571 -> 571** (광역만 늘었다)   [5] 켜면 출력이 실제로 달라진다
#   ⚠ [3] 을 처음에 `w2x(N−1)` 로 잡았다가 6·10·14 씩 어긋났다 — 구간마다 원시 광역
#     전력 통계 2개가 같이 는다. **기댓값은 구현에서 나와야 한다.**
#
#   A  --wide-seg-pool 6                 (78% 담는다)
#   B  --wide-seg-pool 6 --wide-target   (거기에 타깃 블록까지)
#   대조는 **이미 로컬에 있는** cnn_v32h_off_s{0,1,2} 다 — BASE 가 글자 그대로 같아서
#   GPU 를 대조에 안 쓴다. 같은 씨앗끼리 짝으로 뺀다.
#
# ★ 미리 적는 판정 자
#   1 ★★ **기전 직격** — test_5 128~142초 오븐 통전확률의 **면적** (참값 0, 작을수록 좋다)
#         기준선 **4.983** · wtap A 3.142 · wtap B 2.544
#         -> 광역 위치가 듣는다면 **3.0 밑**으로 가야 한다. 안 내려가면 이 축은 끝이다.
#   2 ★ 축퇴 — 포트 참ON·오븐 OFF 창(61개)의 오븐 헛detect
#         기준 게이트 **42.1 ± 5.6%** · >100W **31.7 ± 0.8%**   (wtap A 22.4 / 19.1)
#   3 전환 정체별 헛detect (`run_diag_fptrig`) — (전이 없음) 칸이 **0.0%** 를 지켜야 한다
#   4 `run_gate_wideage` 마지막 칸 — 광역을 흔들 때 오븐 상태가 **더 크게** 움직여야 한다
#        (기준선 +0.025 · A +0.019 · B +0.016 -> 이보다 커야 "쓰기 시작했다")
#   5 ⚠⚠ **지킴 — 14.78 이 무너뜨린 자리**:
#        저항 4종 신원 0.9815 (한 시드도 **0.97 아래로 가면 안 된다**)
#        SMPS 3종 0.9518 · 판정 줄 0.9133 · 절대 잔차 20.2W
#
#SBATCH -J nilm_wsp
# ⚠⚠ SLURM 은 `-p` 순서를 안 본다 (PriorityTier 가 다 1). 느린 칸을 **빼는 것**이
#   유일한 수단이다. gpu3 a6000ada · gpu4·gpu5 a6000 (A10 대비 1.46배).
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# 파라미터 증가는 작다 ((64+2)x5 = 330 차원). 984295 가 A6000 에서 25분이었다.
#SBATCH -t 01:00:00
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
$PY -X utf8 src/run_gate_wsegpool.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--wide-seg-pool" \
  || { echo "--wide-seg-pool 이 없다"; exit 1; }

# ⚠ BASE 는 cnn_v32h_seeds.sbatch 와 **글자 그대로 같아야** 한다 — 대조를 새로 안 굽고
#   로컬의 cnn_v32h_off_s{0,1,2} 를 그대로 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
A_FLAGS="--wide-seg-pool 6"
B_FLAGS="--wide-seg-pool 6 --wide-target"
echo "  BASE: $BASE"
echo "  갈래 A: $A_FLAGS"
echo "  갈래 B: $B_FLAGS"
# 처치가 BASE 로 새면 대조가 무너진다 (cnn_v32h_off 와 BASE 가 달라진다)
case "$BASE" in *--wide-seg-pool*|*--wide-target*|*--seg-pool*)
  echo "BASE 에 처치가 섞였다"; exit 1;; esac
# 세밀 쪽은 한 글자도 안 건드린다 — 14.78 의 실패가 거기였다
case "$A_FLAGS$B_FLAGS" in *--fine-*|*--tap-layers*|*--seg-pool\ *)
  echo "세밀 쪽 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE $A_FLAGS \
    --tag cnn_wsp_s${SEED} > "results/cnn_wsp_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE $B_FLAGS \
    --tag cnn_wsp_wt_s${SEED} > "results/cnn_wsp_wt_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== A: 광역 구간 6 ==="; tail -32 "results/cnn_wsp_s${SEED}.log"
echo "===================================================================================="
echo "=== B: + --wide-target ==="; tail -32 "results/cnn_wsp_wt_s${SEED}.log"
ls -la results/cnn_wsp*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
