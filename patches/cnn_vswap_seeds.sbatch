#!/bin/bash
# 14.33 ⓐ — **프로젝터가 관측 변화 없이 내려앉는 원인**이 `--vswap-p` 인가. 시드 셋 짝비교.
#
# 사용자가 `test_4` 그림에서: *"전체적으로 과소예측되고 그래프 상에 큰 변화가 있는 것도
# 아닌데 참값이 46W 인 프로젝터가 갑자기 내려앉는다."* — 둘 다 맞았다.
#
# 재 보니 (cnn_sigc_vn_s0, test_4):
#   내려앉는 5구간에서 **관측 총전력은 −0.4~−2.7W 밖에 안 변하고** 관측 지문(|I1|,
#   |I3|/|I1|)은 소수점 셋째 자리까지 그대로다. 움직이는 것은 **게이트**뿐이다
#   (0.93 -> 0.21). `power = σ(on)·p_raw` 라 전력이 따라 죽고 충전기가 일부를 가져간다.
#
#   갈래를 반사실로 갈랐다 — **세밀 갈래**다 (광역만 바꾸면 오히려 더 낮아진다).
#   채널 분리도 상위가 `fine[47,48,50,55]` = **전압 고조파 블록(45~56)** 이고,
#   그 블록만 0 으로 만들면 다섯 구간 게이트가 전부 **0.99~1.00** 으로 살아난다.
#
# 그리고 분포 **안에서**(같은 파일 다른 창으로 전압 블록만 교체) 재면:
#       체크포인트        vswap   SMPS p_raw |Δ|   저항 p_raw |Δ|   게이트 뒤집힘
#       cnn_sigc_vn_s0     0.0        9.2%           7.7%           1.4%
#       cnn_v37            0.5        1.7%           0.9%           0.4%
#   물리 기대치는 **~0%** 다 — SMPS 는 정전압 제어라 전압이 ±2% 흔들려도 전력이 안 변하고
#   (14.6 ③), 저항의 V² 의존은 **안 건드린 채널 25(V_rms)** 가 나른다.
#   전압 블록을 통째로 빼도 test_4 추종 r 이 0.964 -> 0.976 으로 **안 떨어진다** —
#   물리로 쓰고 있었다면 떨어져야 한다.
#
# `vswap` 이 깨는 것은 **텍스처 단위의 정확한 (V_h, I_h) 짝 하나**다 —
#   `V_rms`(세밀 25 · 광역 2)는 **안 건드리고** 같은 **자리 안에서만** 섞으므로
#   자리·Z 수준 전압 정보는 산다. 13.84.11 이 그것을 넣은 이유가 *"모델이 생성기의
#   정확한 V→I 법칙을 판별자로 배운다"* 였고, 실측의 (V,I) 짝은 그 법칙에서 벗어나므로
#   잔차가 "설명 안 되는 SMPS 전류 = 미니PC" 로 읽힌다 (유령 8~12% -> 24~32%).
#   실제로 전압 블록을 빼면 **미니PC** 게이트가 제일 크게 움직인다 (sigc +0.31 대 v37 +0.01).
#
#   A 대조   --vswap-p 0     = 지금 채택본의 조리법
#   B 고침   --vswap-p 0.5   = v37 이 쓴 값. **나머지 열 전부 같다**
#
# ⚠ `cnn_v37` 은 이 셋이 다 다르다 (vswap · zero_channels · harm_sig_vnorm). 이 판은
#   **vswap 하나만** 갈라 낸다 — `zero_channels` 는 양쪽 다 안 건다.
#
# **반증 가능한 예측**:
#   ① B 의 전압 블록 민감도가 9.2% -> **2% 안쪽**으로 (로컬에서 `vh_phys` 로 잰다)
#   ② `test_4` 의 프로젝터 게이트<0.5 창이 68 -> **20대**로, |잔차| 8.7 -> **4W 대**로
#   ③ 내려앉는 5구간이 사라지거나 크게 준다
#   ④ ⚠ **저항 쪽(test_5)이 나빠지면 안 된다** — vswap 은 V_rms 를 안 건드리므로
#      `sig ∝ 1/V` 와 V² 지수는 그대로여야 한다. 나빠지면 내 해석이 틀린 것이다
#   ⑤ ⚠ 안 움직이면 원인은 `zero_channels` 쪽이고, 그때는 그것을 따로 건다
#
#SBATCH -J nilm_vswap
# A6000 Ada / A6000 만 (14.28 과 같은 목록). a10·rtx3090 은 뺀다.
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
VSWAP=${VSWAP:-0.5}
SEED=${SLURM_ARRAY_TASK_ID}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 캐시 $CACHE · vswap $VSWAP"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── 관문은 전부 앞에 (HPC_RULES §7-3) ─────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vswap-p" || {
  echo "run_train_cnn 에 --vswap-p 가 없다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 토막이다 — run_merge_traincache 로 먼저 이어붙여라')
if d.get('sibling_rotate'): sys.exit('v32 인데 sibling_rotate 가 있다 (14.4 위반)')
print('관문: 캐시 %d창 · v32 통짜' % d['n_windows'])
"
# ── 두 판. 바뀌는 것은 **--vswap-p 하나** ─────────────────────────────────
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"
# `--harm-sig-vnorm` 은 14.29 에서 **기본 켜짐**이 됐다. 두 팔 다 켜진 채로 간다.

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --vswap-p 0 --tag cnn_vswap_off_s${SEED} > "results/cnn_vswap_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --vswap-p "$VSWAP" \
    --tag cnn_vswap_on_s${SEED} > "results/cnn_vswap_on_s${SEED}.log" 2>&1 &
B=$!
echo "A(vswap 0) $A · B(vswap $VSWAP) $B"
RC=0
wait $A || RC=1
wait $B || RC=1


echo "=== A: vswap 0 ===";  tail -22 "results/cnn_vswap_off_s${SEED}.log"
echo "=== B: vswap 0.5 ===";  tail -22 "results/cnn_vswap_on_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 ==="
$PY -X utf8 -m src.run_score_holdout \
    --ckpt results/cnn_vswap_off_s${SEED}.pt results/cnn_vswap_on_s${SEED}.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_vswap_off_s${SEED}.pt results/cnn_vswap_on_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬 실측**으로, 같은 자 하나로:"
echo "  python -X utf8 src/run_diag_rollback.py --ckpt results/cnn_vswap_off_s0.pt results/cnn_vswap_on_s0.pt"
echo "  python -X utf8 -m src.run_plot_real --ckpt results/cnn_vswap_on_s0.pt --tag vswap   # 내려앉음이 사라졌나"
echo "  python -X utf8 -m src.run_plot_real --ckpt results/cnn_vswap_on_s0.pt --tag swap"
exit $RC
