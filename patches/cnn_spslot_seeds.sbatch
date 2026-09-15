#!/bin/bash
# 14.186 ★★★ — **초기값 슬롯표를 라벨 중앙값으로.** 짝 `cnn_dzn` — `--state-power-src` 하나만.
#
# 14.185 가 찾은 것: 오탐 자리가 **씨앗·팔과 무관**하다 (15판 Jaccard 같은팔 0.533 ·
# 다른팔 0.504 · test_5 는 15판 전부 틀린 창이 19개). 그러면 범인은 **모두가 공유하는 것**이다.
#
#   `losses.S_STATE` 는 12.9.9 의 **`L_power` Huber 척도**(p90)다. 척도로는 p90 이 맞다.
#   그런데 13.84.68 이 같은 값을 **머리 바이어스 초기값**으로 재사용했고, 듀티 기기는
#   p90 과 중앙이 20% 넘게 벌어진다. 14.186 측정:
#
#     기기  상태   S_STATE   라벨중앙   학습값  | 표->라벨 경로의 %  · 남은오차(Huber 눈금)
#     포트  s1    1534.5   1456.8  1468.9 |   **84%**        0.008
#     드라이 s2   1022.5    964.1   978.6 |   **75%**        0.014
#     드라이 s1    529.2    486.9   506.3 |   **54%**        0.037
#     핫플  s2     549.6    454.8   525.6 |   **25%**      **0.129**
#     오븐  s2    1357.1   1100.6  1298.9 |   **23%**      **0.146**
#
#   δ 안(0.008~0.037)인 셋은 54~84% 수렴했고 δ **밖**인 둘은 23~25% 만 왔다.
#   Huber 는 δ 밖에서 기울기가 **포화**되므로 초기값이 10% 넘게 틀리면 300에포크로 못 간다.
#   **라벨이 옳다** — 라벨 중앙과 실측 중앙이 0.1W 안에서 같다 (오븐 1100.6 대 1100.1).
#
# 결과: 오븐 통전 슬롯이 1298.9W(참 1100.6) 라 **오븐↔포트 간격이 170W(11.6%)** 로 눌린다.
#   **실측 계단은 352W(24.6%)** 다. 두 순저항의 고조파 사이각이 **0.30~1.91도**(오븐s2↔핫플s2
#   0.30도, cos 1.0000)라 가를 것이 없고, 그 동률을 창 안의 계단이 기울인다 (§7 의 구멍 넷).
#
# **추론 선검증** (14.186, 모델 없이 사전만으로 관측을 설명):
#   test_2 231~259초  표: 1~3등 **전부 오븐 통전** -> 실측 슬롯: 1~3등 **전부 포트**   뒤집힘
#   test_2 174~178초  표: 1~3등 **전부 오븐**(|ΔP| 0.6W) -> 실측: 전부 포트(고조파 7.4%->2.1%)  뒤집힘
#   test_5 128~141초  실측 슬롯에서도 1~3등이 오븐                                   **안 뒤집힘**
#   => **2/3**. 사전이 확인된 원인이지만 **충분하지는 않다**. 그것이 이 판의 예측이다.
#
# 바꾸는 것 — **큰 슬롯(>=300W) 일곱뿐**:
#   에어컨 s3 549.6->473.5 · s4 794.0->624.4 · 포트 s1 1534.5->1456.8 ·
#   드라이 s1 529.2->486.9 · s2 1022.5->964.1 · 핫플 s2 549.6->454.8 · **오븐 s2 1357.1->1100.6**
#   ⚠ **오븐 s1(FAN_LIGHT)은 안 건드린다.** `is_on=0` 이라 라벨이 **100% 0W** 인데 실측은
#     **14.6W** 다 (핫플 s1 은 라벨 0 · 실측 0.3W 로 맞다). 그 13.4W 가 라벨 없이 총전력에만
#     들어가는 것이 14.173 의 대기 축퇴이자 §7 의 '16W 발판' 뿌리다. 14.167 이 그 근처를
#     건드렸다가 오븐 유령을 **3.9배**로 키웠다.
#   ⚠ **척도(`build_state_scales`)와 상한(`p_state_cap`)은 안 바꾼다** — 한 번에 하나만.
#
# 관문 `run_gate_spslot` 여섯 줄 전부 통과:
#   [1] `table` 이면 **비트 동일** [2] 큰 슬롯 일곱만 [3] 값이 풀 라벨 중앙과 일치
#   [4] 바꾼 슬롯은 라벨과 실측이 5% 안 [5] ★ 새 초기값이 **Huber δ 안** (표는 넷이 밖)
#   [6] `load_state_dict` 가 덮으므로 **학습된 판에 영향 없다**
#
# **반증 가능한 예측**
#   ① 학습이 돈다. 초기값이 다르므로 첫 에포크 total 이 `cnn_dzn` 과 **다르다**.
#      파라미터 수는 **같다**(888,086). 체크포인트에 `state_power_src: label` 이 적힌다.
#   ② ★ **학습된 오븐 s2 슬롯이 1100 근처로 가야 한다** (dzn 은 1298.9 ± 0.1).
#      안 가면 초기값 가설이 틀린 것이다 — 그것만으로도 이 판의 값이다.
#   ③ ★ 오븐↔포트 간격이 **350W 대**로 벌어져야 한다 (dzn 170W).
#   ④ 오븐 참OFF 유령초가 줄어야 한다. 선검증이 2/3 이니 **완전히 죽지는 않을 것**이다.
#   ⑤ ⚠ 슬롯이 전부 내려가므로 **절대 잔차가 오를** 수 있다 (예측합이 작아진다).
#      L_cons 의 죽은구역 δ=100W 가 그것을 얼마나 막는지가 이 판이 처음 재는 것이다.
#
# **미리 등록한 판정법**
#   이득  오븐 **참OFF 유령초** (dzn s0/s1/s2 = **44.5 / 28.5 / 19.0초**) ·
#           **학습된 오븐 s2 슬롯** · **오븐↔포트 간격** · 수백W 띠 · p99
#   비용  판정줄 7종 (.8497/.8368/.8518) · 저항 4종 (.9815 x3) · SMPS (.9368/.9711/.8500) ·
#           절대 잔차 (16.0/19.5/15.2W) · 혼자켜짐 (20.9/45.5/44.4%)
#   ⚠ 판정줄이나 저항 4종이 무너지면 이득이 뭐든 기각 (12.9 붕괴 감시).
#SBATCH -J nilm_spslot
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
$PY -X utf8 -m src.run_gate_spslot
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE 는 짝(`cnn_pcap`)과 **글자 그대로 같아야** 한다 — 다른 것은 TREAT 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT       --epochs $EPOCHS --epoch-windows 50000 --seed $SEED       --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4       --harm-even-magnitude --state-signatures --standby-operating session       --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
ARCH="--state-power-src label"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100"
echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $ARCH"
echo "  처치: $TREAT"
case "$BASE$WTAP$CAP$ARCH" in *--even-median*|*--w-cons*|*--cons-deadzone*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152)
case "$BASE$WTAP$CAP$ARCH$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac

case "$ARCH" in *"--state-power-src label"*) : ;; *) echo "처치가 안 걸렸다"; exit 1;; esac
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $ARCH $TREAT     --tag cnn_spslot_s${SEED} > "results/cnn_spslot_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_spslot_s${SEED}.log"
ls -la results/cnn_spslot_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC

