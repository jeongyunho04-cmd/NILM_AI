#!/bin/bash
# 14.46 — **구간별 풀링.** 전역 `mean`/`amax` 를 타깃 경계로 쪼갠다. 시드 셋 짝비교.
#
# 무엇을 고치려는가 (14.41~14.42, 반사실로 확정):
#   세밀 창 600 중 **360(6초)이 타깃보다 뒤**인데 깊은 탭의 수용영역은 **±93** 뿐이다.
#   그 바깥 증거가 머리에 닿는 길이 **`h.mean(-1)`·`h.amax(-1)` 같은 전역 요약 하나**이고,
#   그것은 과거와 미래를 **구별하지 못한다.**
#       미래를 지우면 오븐 혼합 0.137 -> 0.939 · 과거를 미래로 덮으면 0.001
#       **수용영역 밖만** 지워도 0.941  <- 전부 지운 것과 같다
#   실측 증상: 오븐 단독통전 336창 중 붕괴가 **전부** 결합 창(111창) 안에 있다.
#             전환 없는 225창은 **0.0%**. 붕괴 시 중앙 **1066W** 를 놓친다.
#
# 왜 '없애기' 가 아니라 '쪼개기' 인가:
#   · 14.45 가 **미래를 가리면 나빠지는 것**을 쟀다 (판정 −0.0026±0.0013 · 잔차 +5.8±0.4W).
#     -> "미래를 없앤다" 는 문이 닫혔다. 남은 길은 **시간 부호를 주는 것**이다.
#   · `amax` 는 시간축 최대라 **conv+점별 비선형으로 표현되지 않는다.** 12.9.8 이 원시 창
#     통계를 넣은 이유가 *"학습된 amax 로만 있었더니 총전력을 1/10 로 줄여도 핫플 on
#     로짓이 0.09 밖에 안 움직였다"* 였다. 없애면 그 자리가 다시 빈다.
#   · **`fp_max` 는 물리 프라이어(12.9.8)가 전역으로 쓴다** — 그것은 그대로 두고 머리에
#     주는 통계만 쪼갠다.
#
#   A 대조   --seg-pool 0     창 전체 한 구간 = **지금과 비트 동일**
#   B 고침   --seg-pool 2     [0,240) 과거 · [240,600) 미래 (광역은 [0,108)·[108,120))
#   나머지 열 **전부 같다**. 머리 입력 **637 -> 961**.
#
# 관문 (`src/run_gate_segpool.py`, 여섯 전부 통과 확인):
#   [1] 구간이 타깃 직후를 경계로 두고 창을 빠짐없이·겹침없이 덮는다
#   [2] seg_pool 0 과 1 이 **출력 비트 동일**
#   [3] 옛 체크포인트가 seg_pool=0 으로 **그대로 실린다** (머리 637)
#   [4] seg_pool=2 는 옛 가중치를 **거부한다** (조용히 받으면 위험)
#   [5] `fine_dim_mask` 길이 == trunk_in (829/1217)
#   [6] autocast(bf16) **역전파**가 돈다 (982877 이 여기서 죽었다)
#   끝단: `run_diag_rollback` 아홉 줄이 14.45 기록과 **한 글자도 안 다르다** (비트 동일).
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① **오븐 상태 붕괴가 준다.** 실측 조인 자 **7.3±1.1%** (`run_diag_hwfuture --real --sharp`,
#      결합 창 111개에서 17~25% · 나머지 225창 0.0%). 이것이 겨냥한 것이다.
#      **결합 창의 붕괴율**이 판정 숫자다.
#   ② **판정 줄이 안 떨어져야 한다.** 정보를 하나도 안 버리고 시간 부호만 붙이므로 떨어질
#      이유가 없다. 떨어지면 머리 입력이 637->961 로 커진 **용량 쪽 부작용**이다.
#   ③ **절대 잔차가 준다** (지금 23.6W). 14.40 의 유령 반파 드라이기(test_1 46창 · −300~−660W)도
#      같은 축이라 같이 줄어야 한다.
#   ④ ⚠ **A 팔은 `cnn_vncls_res` 자리에 와야 한다** (.9020/.9099/.9078). 안 오면 비트 동일이
#      학습 경로에서 깨진 것이고, 그러면 이 판 전체를 못 읽는다.
#   ⑤ ⚠ **안 움직이면 구조가 아니라 자료다.** 14.44 ② 가 *"합성 결합 창에서는 안 무너진다
#      (0~1.2%)"* 를 보였으므로, 그때는 생성기 쪽(실측 결합이 무엇이 다른가)을 판다.
#
# ⚠ 시드 하나로 읽지 마라. 판정 줄 시드 폭이 이 조리법에서 0.004 다.
#
#SBATCH -J nilm_segp
# ⚠ 파티션은 **A6000 Ada/A6000 우선**이다 (gpu3 는 나머지의 2.4배, HPC_RULES §3).
#   그런데 그래서 제일 밀린다. **밀려 있으면 넓혀라** — A6000·RTX3090·A10 은 서로
#   **25% 안쪽**이고 이 작업은 VRAM 을 11% 밖에 안 쓴다 (§3·§7). 기다림이 25% 보다 비싸다.
#       scontrol update jobid=<id> Partition=gpu2,gpu6,gpu4,gpu5,gpu3,gpu1 TimeLimit=00:45:00
#   983243 에서 gpu3/4/5 만으로 128개 뒤에 서서 한 시간 설 참이었는데, 넓히자 **6초**에 잡혔다.
#   ⚠ 넓힐 때 `-t` 도 25% 만큼 늘려라 (35분 -> 45분). 팔 둘이 같은 노드에 있으면
#     **짝비교는 그대로 유효**하다 — 시드끼리 하드웨어가 달라도 B−A 는 노드 안에서 난다.
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). 짧을수록 backfill 이 잘 문다 —
#   `-t 01:15` 로 3.2배를 불렀다가 시드 1·2 가 한 시간을 섰다 (983243, 2026-09-14).
#   ⚠ `squeue --start`·`sbatch --test-only` 는 **backfill 을 모델링하지 않는다** —
#     모양을 바꿔도 같은 날짜가 나온다. 그 값으로 '자리가 없다' 고 읽지 마라.
#SBATCH -t 00:35:00     # 실측 19~24분(982936·982937)의 1.5배
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

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 캐시 $CACHE"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문은 전부 앞에 (HPC_RULES §7-3) ─────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--seg-pool" || {
  echo "--seg-pool 이 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_gate_segpool || { echo "구간별 풀링 관문 실패"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 토막이다')
print('관문: 캐시 %d창' % d['n_windows'])
"

# ── 두 판. 바뀌는 것은 **--seg-pool 하나** ────────────────────────────────
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"
# ⚠ 14.37 에서 `--harm-vnorm-classes` 기본이 RESISTIVE 로 바뀌었다. `vncls_res` 와 같은
#   조리법이 되도록 **명시로 박는다** (BASE 변수에는 안 넣는다 — 따옴표 사고 14.37).

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE --seg-pool 0 \
    --tag cnn_segp_off_s${SEED} > "results/cnn_segp_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE --seg-pool 2 \
    --tag cnn_segp_2_s${SEED} > "results/cnn_segp_2_s${SEED}.log" 2>&1 &
B=$!
echo "A(seg 0) $A · B(seg 2) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: seg 0 ===";  tail -32 "results/cnn_segp_off_s${SEED}.log"
echo "=== B: seg 2 ===";  tail -32 "results/cnn_segp_2_s${SEED}.log"
ls -la results/cnn_segp_*_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
# ⚠⚠ `grep ... | head` 도 쓰지 않는다 (SIGPIPE + pipefail 이 982881 을 죽였다).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬 실측**으로, 같은 자 하나로:"
echo "  python -X utf8 src/run_diag_rollback.py --postproc off --ckpt results/cnn_segp_off_s0.pt results/cnn_segp_2_s0.pt"
echo "  python -X utf8 -m src.run_diag_hwfuture --real --sharp --ckpt <위 둘>"
echo "  ⚠ 예측 ① 결합 창(111개)의 붕괴율 17~25% 가 줄어야 한다"
echo "  ⚠ 예측 ④ A 팔이 vncls_res 자리(.9020/.9099/.9078)에 와야 한다"
exit $RC
