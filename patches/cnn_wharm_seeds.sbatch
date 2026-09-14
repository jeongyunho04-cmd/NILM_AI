#!/bin/bash
# 14.38 — **저항 통전 중 과소가 `L_harm` 때문인가.** `--w-harm` 하나만. 시드 셋 짝비교.
#
# 사용자: *"열려 있는 것 1번을 파자"* (저항 통전 중 2~4% 과소).
#
# 판 앞에서 자를 바꿨더니 문제가 달라졌다 (14.38 본문):
#   ① 곱 분해 — `power = σ(on)·Σmix·p_states`. 실측 통전 창에서 **σ(on)=1.000 ·
#      mix_max=1.000** 이다. 게이트도 혼합도 아니고 **상태 전력값 하나**다.
#   ② 자가 아니다 — **합성 홀드아웃(참값 정확)에도 같은 편향이 있다.**
#      학습 로그의 `편향(on)` 열이 이미 재고 있었다.
#   ③ 씨앗 바닥 위다 — 같은 팔 안 시드 sd 0.1~1.8W 인데 팔 사이가 10~32W 다.
#   ④ `harm_sig_vnorm` 하나가 포트를 **+40.9 -> +8.5W**, 오븐을 **−11.1 -> −21.2W**
#      밀었다 -> `L_harm` 이 이 편향의 지렛대인 것은 확실하다.
#   ⑤ 그런데 **지문 표의 크기는 저항에서 안 틀렸다** — 격리 녹화에서
#      `a = <c,sig>/<sig,sig>/P` 가 포트 1.0000 · 드라이 1.0000 · 핫플 1.0001 ·
#      오븐 1.0010 이다. (SMPS 는 프로젝 0.957 · 충전기 0.933 · 미니PC 0.932 로
#      4~7% 어긋나 있고 **부호가 관측 편향과 맞는다** — 미니PC −14.6%.)
#
# 남은 물음 하나: **vnorm 뒤에 남은 저항 편향도 `L_harm` 이 만드는가.**
#   크기가 아니면 **기하**다 — 기기 지문들이 직교가 아니라, 한 기기의 전력을 내리고
#   다른 기기를 올려도 `obs_harm` 이 같게 맞는 방향이 있다. 그 방향으로 `L_power`
#   (Huber, w=1.0) 와 `L_harm` (w=0.1) 이 평형을 이루면 **계통 편향**이 남는다.
#
#   A 탐침  --w-harm 0      항을 통째로 끈다
#   B 탐침  --w-harm 0.3    3배로 키운다
#   기준선은 **이미 있다** — `--w-harm 0.1` 을 12판 쟀다 (sigc_vn · vncls_all/res).
#
# ⚠⚠ **이것은 조리법 후보가 아니라 기전 탐침이다.** `w_harm 0` 은 배분을 못 하게
#   만들므로 F1·저항3종 정확도가 떨어질 것이다. 그것은 이 판의 실패가 아니다.
#
# ⑥ ★ **합은 이미 맞는다 — 틀린 것은 배분이다.** 1단계에는 보존 항이 **없다**
#   (`--w-cons` 기본 0.0 · 조리법이 `--w-over 0.0`). 그런데 합성 홀드아웃의
#   **부호 있는** 총전력 잔차가 `vncls_res` 3판 평균 **+0.96W** 다 (절대평균 27.2W).
#   보존을 안 걸었는데도 합이 맞으니 **`--w-cons` 를 1단계에 올리는 길은 닫혔다.**
#   그리고 오븐이 잃는 21.3W 는 사라지지 않고 **같은 창의 다른 기기 유령**으로 간다 —
#   `전가`(고부하 창 FA − 저부하 창 FA) 합이 **+19.66W** 다:
#       오븐 +7.01 · 포트 +6.07 · 드라이 +2.74 · 핫플 +1.96 · 프로젝 +0.94 · 나머지 +0.9
#   ⇒ 이 판이 재는 것은 **배분(전가)** 이지 수준이 아니다.
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① `L_harm` 은 배분을 식별 가능하게 만드는 **유일한 항**이다 (3.4절의 30차원).
#      그것이 원인이면 **편향과 전가가 `w_harm` 에 단조**여야 한다:
#          w=0    -> 전가 합 +19.7W 가 **커지고** 오븐 −21.3W 가 **더 음수로**
#          w=0.3  -> 전가가 **줄고** 오븐 편향이 **0 쪽으로**
#      ⚠ 방향을 한쪽으로 못 박지 않는다 — vnorm 판은 전체 보정이 좋아지면서
#        (잔차 −13.0 -> +0.9W) 오븐만 −11 -> −21 로 나빠졌다. **단조인지**가 판정이다.
#   ② 가장 크게 움직이는 것은 **SMPS** 여야 한다 (지문 크기 오차 4~7% 가 거기 있다).
#      미니PC −3.17W (시드 sd 0.12) 는 0.5W 만 움직여도 읽힌다.
#   ③ ⚠ **부호 있는 총전력 잔차는 세 팔 다 0 근처(±3W)여야 한다** — 보존 항이
#      없는데도 0 인 것이 ⑥ 의 관측이다. 크게 벌어지면 내 해석이 틀린 것이다.
#   ④ ⚠ **안 움직이면 `L_harm` 은 무죄**고, 다음 탐침은 `--power-delta`
#      (Huber 문턱, 지금 정격의 0.1 = 오븐 121W) 다.
#   ⑤ ⚠ 판정은 **편향(on)·전가·부호 있는 잔차** 세 열이다. F1 이 떨어지는 것은
#      예상된 대가지 이 판의 실패가 아니다.
#
# ⚠ 읽을 수 있는 분해능: 같은 팔 시드 sd 가 오븐 0.6~1.6 · 드라이 0.1~0.9 ·
#   핫플 0.4~1.8 · 포트 2.2~4.6W 다. **2W 보다 작은 이동은 읽지 마라.**
#
#SBATCH -J nilm_wharm
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 01:15:00
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
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--w-harm" || {
  echo "--w-harm 이 없다"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-vnorm-classes" || {
  echo "--harm-vnorm-classes 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 토막이다 — run_merge_traincache 로 먼저 이어붙여라')
if d.get('sibling_rotate'): sys.exit('v32 인데 sibling_rotate 가 있다 (14.4 위반)')
print('관문: 캐시 %d창 · v32 통짜' % d['n_windows'])
"

# ── 두 판. 바뀌는 것은 **--w-harm 하나** ──────────────────────────────────
# ⚠ `--harm-vnorm-classes RESISTIVE` 를 **명시로 박는다** — 14.37 이 기본값을
#   뒤집었으므로, 코드를 올렸든 안 올렸든 이 판이 `vncls_res` 기준선(12판 중 3판)과
#   **같은 조리법**이 되게 못박아야 한다. BASE 변수에 넣지 않는다 (14.37 의
#   따옴표 사고 — 확장 뒤 두 글자가 값으로 넘어간다).
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE --w-harm 0.0 \
    --tag cnn_wharm_0_s${SEED} > "results/cnn_wharm_0_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --harm-vnorm-classes RESISTIVE --w-harm 0.3 \
    --tag cnn_wharm_3_s${SEED} > "results/cnn_wharm_3_s${SEED}.log" 2>&1 &
B=$!
echo "A(w_harm 0) $A · B(w_harm 0.3) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

# 판정 열은 **홀드아웃 표의 `편향(on)`** 이다. 표 전체가 보이게 넉넉히 자른다.
echo "=== A: w_harm 0 ===";   tail -32 "results/cnn_wharm_0_s${SEED}.log"
echo "=== B: w_harm 0.3 ==="; tail -32 "results/cnn_wharm_3_s${SEED}.log"
ls -la results/cnn_wharm_0_s${SEED}.pt results/cnn_wharm_3_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
# ⚠⚠ `grep ... | head` 도 쓰지 않는다 (SIGPIPE + pipefail 이 982881 을 죽였다).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정: **로그의 편향(on) 열만** 본다. 기준선은 이미 있다 —"
echo "  w_harm 0.1 (vncls_res 3판): 포트 +8.48 오븐 −21.26 핫플 −6.83 드라이 +8.28"
echo "                              프로젝 −1.09 충전기 −0.16 미니PC −3.17 선풍 −2.73"
echo "  전가(고부하FA−저부하FA) 합 +19.66W · 부호 있는 잔차 +0.96W"
echo "  ⚠ 2W 보다 작은 이동은 시드 바닥 안이다."
exit $RC
