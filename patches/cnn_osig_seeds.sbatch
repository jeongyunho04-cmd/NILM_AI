#!/bin/bash
# 14.167 — **오븐 휴지 지문을 고친 뒤 그대로 다시.** 짝 `cnn_dzn`.
#   처치는 문자 하나 안 다르다 — 달라진 것은 `harmonic_signatures_by_state` 의
#   분모뿐이다 (oven s1 이 히터 지문 대신 자기 지문을 갖는다).
#   관문 `run_gate_statesig` 이 그것을 매 판마다 확인한다.
#
# 사용자: *"em5에 이번 수정 더해서 학습돌려줘"*
#
# 겨냥: **오븐<->포트 신원 축퇴** (14.156~158). 오늘 잰 것 —
#   · 신원이 입력에 **없다**. 기본파가 답을 정하고 고조파는 무관
#   · 두 안정점이 **정확히 두 컨덕턴스 답** (226V 에서 1,259W 대 1,428W)
#   · 관측 총합은 **포트 답에 1~6W · 오븐 답에 140~190W** 어긋난다 (8/8 창)
#   · 치환 방향에 따라 잔차 **부호가 뒤집힌다** (test_5 +97W · test_2 −174W)
#     -> `L_over`(과잉만)로는 못 잡는다. **양방향**이라야 한다
#   · 그런데 그 자를 보는 손실이 1단계에 **하나도 없다** (w_cons 0 · w_over 0 · proj 꺼짐)
#
# 왜 이제 `L_cons` 를 켤 수 있나 — 12.9 의 기각 사유 둘이 **다 만료됐다**:
#   (가) *"합성에서 P관측 ≡ Σ라벨 은 항등식이라 정보를 안 더한다"* 는 **배분이 자유롭다**는
#        전제 위에 있다. 12.9 **이후** `state_power_init`(13.84.68)·`--p-state-cap`(14.121)이
#        기기별 전력을 거의 고정된 몇 값으로 묶었다. 실측 반사실(14.161): 맞춘 창에서
#        포트<->오븐을 바꾸면 |r| 27.8 -> **150.0W**. 합이 배분을 **고른다**
#   (나) *"경사가 35~3000배 세다"* 는 옛 식이 **절대 와트**(`|r|.mean()`)라서다.
#        관문 [1] 이 그것을 보여 준다 — 같은 배치에서 옛 식 **919** 대 새 식 **1.72**
#
# δ=100W 는 **재서** 정했다 (14.161):
#   합성 홀드아웃 |r| p50 8.5W · p90 49.4W  ->  δ=100W 에서 정상 창의 **96.9%** 가
#   죽은구역 안이라 기울기가 **정확히 0** (실측은 96.5%).
#   가장 싼 치환(포트<->오븐, 최소 **139W**)은 **100%** 걸린다. 나머지 다섯 쌍은 266~958W
#
# w_cons=1.0 의 뜻: 걸리는 창에서 (170−100)/2000 = **0.035** 를 낸다. `L_power` 의
#   평소 값이 0.0030 이니 **그 창에서만 10배**로 세고, 나머지 97%에서는 **정확히 0** 이다
#
# 관문 `run_gate_deadzone.py` 여섯 줄 전부 통과 (끄면 비트 동일 · 죽은구역 안 기울기 0 ·
#   양방향 대칭 · 진짜 모델 출력으로 검산).
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① 학습이 돈다. `cons` 항이 생기므로 첫 에포크 total 이 `cnn_em5` 와 **다르다**.
#   ② ★ **진짜 물음 — 축퇴가 줄어야 한다.** test_2 오븐 유령(46초·1,254W)과
#      test_5 포트 유령(9초)이 짧아지거나 없어져야 한다 (`run_diag_ghost` 최장초 ·
#      `run_diag_flipedge`). **안 움직이면 컨덕턴스 가설이 틀린 것이다.**
#   ③ ★ **절대 잔차가 낮아야 한다** — 손실이 직접 겨냥하는 양이다. 안 내려가면 구현 문제다.
#   ④ ⚠ **12.9 의 붕괴 감시** — "합만 맞추고 배분 포기" 가 재발하면 **혼자켜짐 전력오차**와
#      **저항 4종 신원**이 무너진다. 죽은구역이 그걸 막는다는 것이 이 판의 전제다.
#   ⑤ em5 의 효과(스파이크 0개)는 **유지**돼야 한다.
#
# ★ 짝의 기준값은 `cnn_em5` 가 끝나면 채운다 (985148). 할아버지 `cnn_pcap`:
#   판정줄 .8820 · 저항 .9774 · SMPS .9807 · 절대잔차 21.3W · 혼자켜짐 31.8% ·
#   유령 게이트율 0.96% · 유령 최장 48초(test_2 오븐)
#SBATCH -J nilm_osig
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
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
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_statesig
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_pcap`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH=""
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_osig_s${SEED} > "results/cnn_osig_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_osig_s${SEED}.log"
ls -la results/cnn_osig_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
