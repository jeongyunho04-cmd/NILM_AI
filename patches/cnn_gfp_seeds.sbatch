#!/bin/bash
# 14.150 — ★ **게이트와 전력을 완전히 분해한다** (`--gate-free-power`). 시드 셋 짝비교.
#
# 사용자: *"또 다른 게이트와의 연결점이 있는지 검토하고, 완전히 게이트랑 전력예측을
#   분해하고 학습을 돌려봐야 하지 않겠어?"*  그리고 *"v2 에도 똑같이 하고 둘 다 돌려보자."*
#
# 전수조사에서 연결점이 **여섯**이었다 (14.150):
#   (1) 값 `power = s(on)*p_raw`      (2) `dpower/dp_raw = s(on)` **흡수 상태**(13.80 주석)
#   (3) `dpower/don = s'*p_raw`       (4) 프라이어가 on_logit 을 밀어 전력을 민다
#   (5) on_logit/state/p_states 가 **같은 머리**에서 나온다 (구조적 — 안 끊는다)
#   (6) L_harm(w 0.1) 도 out["power"] 를 쓴다
# 14.147 의 A·B 는 (2)(3)을 **참ON 창에서만** 끊었고 (1)(4)는 그대로였다. 그리고
# 14.149 가 잰 결과 출력 결손의 **100%가 게이트 항**이고 p_raw 는 이미 정직했다(-0.0005).
# **끊은 곳이 틀렸다.**
#
# ⚠ 곱을 그냥 뗄 수 없다 — `power_mix_mask` 가 state 0 을 빼서 p_raw 는 구조적으로
#   "켜졌다면 얼마" 이고 0 이 못 된다. 그래서 **state 0 을 혼합에 넣고 그 전력을 0** 으로
#   둔다. OFF 일을 **상태 머리**가 맡고 게이트는 순수 검출기가 된다.
#   관문 `run_gate_gatefree.py` 아홉 — [6] power 경사가 on_logit 에 **0** · [7] 프라이어
#   κ 8 대 0 에서 power **동일** · [1] 끄면 비트 동일.
#
# ★ 던지기 전에 잰 기준값 (짝)
#   v1 짝 `cnn_pcap`  판정줄 0.8820 · 저항 0.9774 · SMPS 0.9807 · 잔차 21.27W ·
#                     ★4 0.893 · 혼자켜짐 31.8% · 포트창 통전 24.20% · 게이트-진폭 r(오븐) 0.618
#   v2 짝 `cnn_hv2`   판정줄 0.9020 · 저항 0.9815 · SMPS 0.9851 · 잔차 18.63W ·
#                     ★4 0.912 · 혼자켜짐 19.7% · 포트창 통전 30.71% · r(오븐) 0.776
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 학습이 돈다. 값이 바뀌므로 첫 에포크 손실이 대조와 **다르다**.
#   ② ★ **진짜 물음 — 게이트를 떼도 OFF 가 되나.** 참OFF 창의 예측 전력이 0 근처에
#      머물러야 한다. 상태 머리가 그 일을 못 하면 여기서 무너지고, 그게 이 판의 핵심이다.
#   ③ ⚠ **13.11 의 재발 감시.** 이제 OFF 창이 `p_raw` 를 직접 0 으로 민다 — 드라이 s2 가
#      0W 로 죽었던 그 병이다. **혼자켜짐 전력오차**와 **상태별 지문**을 같이 본다.
#   ④ 통과 조건은 **'안 나빠진다'** — 저항 4종 신원·판정줄·잔차·★4 가 씨앗 바닥 안.
#   ⑤ 출력 결손이 줄어야 한다 — `log(출력/참)` 이 -0.170(v1)/-0.142(v2) 에서 0 쪽으로.
#      이게 안 움직이면 14.149 의 진단이 틀린 것이다.
#   ⑥ 통전 자는 **관찰**한다.
#SBATCH -J nilm_gfp
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
$PY -X utf8 src/run_gate_gatefree.py
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--gate-free-power" \
  || { echo "--gate-free-power 가 없다"; exit 1; }

# ⚠ BASE + WTAP + CAP 이 cnn_pcap_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_pcap_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--gate-free-power"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP $CAP"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP" in *--gate-free-power*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116 · 14.122 · 14.128 · 14.131 · 14.16)
case "$BASE$WTAP$CAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--fine-pad*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# 처치의 전제: 사영이 꺼져 있어야 한다 (관문 [8] 이 같이 못 씀을 확인한다)
case "$BASE$WTAP$CAP$TREAT" in *--proj*) echo "사영과 같이 못 쓴다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT \
    --tag cnn_gfp_s${SEED} > "results/cnn_gfp_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_gfp_s${SEED}.log"
ls -la results/cnn_gfp_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
