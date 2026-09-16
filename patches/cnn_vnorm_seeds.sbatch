#!/bin/bash
# 14.299 — **V 불변 와트 목표** `--hcond-scale watt`. 짝은 `cnn_evenjit40`(씨앗 여섯).
#
# 사용자 물음 둘이 이 팔을 만들었다:
#   *"저항값 손실이 저항성 기기에만 영향을 주게 할 방법이 있나?"*
#   *"이제 저항성만 영향을 받는데 로그가 있어야 하나?"*   -> 답은 **아니다**.
#
# log 가 하던 일이 **둘**인데 뗄 수 있다:
#   ① **V 불변**    목표를 vrel^e 로 나눈다   <- 14.16 이 요구한 진짜 진단
#   ② **척도 불변**  log 를 씌운다            <- "11,600배" 명분이자 **피해의 원인**
# 저항 기기만 남기면 ②는 살 이유가 없다 — 11,600배는 **1.4kW 대 11W** 이야기인데 그
# 11W 짜리가 갈래에서 빠지고, 남는 넷은 529~1534W 라 `s_i` 정규화가 이미 공평하다.
# 한 기기 **안에서** 상태끼리(드라이 529 대 1022)는 log 가 2배를 사는데, 그 2배 때문에
# 아래 두 값을 지불했던 것이다.
#
# 14.296 이 잰 `--head-conductance`(=log) 의 실제 결과:
#   ✗ 겨냥한 **오븐->포트가 0.17 대 0.17** — 한 칸도 안 움직였다
#   ✗ 홀드아웃 MAE 4.65->6.37 (+37%) · 총잔차 20.1->35.8 (+78%) · harm +0.027(완전분리)
#   ✗ 씨앗이 갈리는 창 1.73/1.80% -> **3.12/3.54%** (거의 두 배)
#   ✗ 저항 넷이 전부 ~1% 아래로 치우쳤다 (포트 편향 −9.2 -> **−25.8W**)
#
# 관문 `src/run_gate_hcondwatt.py` 여덟이 **왜 watt 가 나은지**를 수치로 박았다:
#   [2] ★★ 오븐↔포트 196W 를 가르는 와트당 기울기
#       바닥 8.27e-05 · **log 3.89e-05 (0.47배)** · **watt 8.90e-05 (1.08배)**
#       -> log 는 분리력을 **절반으로 깎는다**. watt 는 바닥보다 세다 (1/vp 만큼)
#   [3] ★★ 오븐 s1 17W 가 1W 로 무너진 창
#       바닥 1.09e-05 · **log 5.00e-02 (4,568배)** · watt 1.27e-05 (1.16배)
#   [4] ⚠ 폭주는 **클래스로 안 잘린다** — 전역 1위가 오븐 s1(저항, 4,626배)이다.
#       그래서 `--hcond-on-w 100` 이 **처치의 일부**다 (오븐 s1 17W · 핫플 s1 10W 를 뺀다)
#   [6] 문턱 아래·마스크 밖은 옛 Huber 그대로 · 위는 V 나눈 Huber (미니PC 1000W 도 옛 것)
#
# ⚠ **반례를 먼저 적는다.** watt 는 바닥과 기울기가 **1.08배**밖에 안 다르다. 그 8% 는
#   `1/vp` 그 자체이고, 저항 V_REL_CLAMP 안에서 최대 ±29% 다. **아무 차이도 안 날 수 있다.**
#   그러면 "V 불변 목표는 이 자료에서 안 읽힌다" 가 답이고 14.16 의 진단이 과했던 것이다.
#   이 팔은 "hcond 를 고치는 것"이 아니라 **"hcond 가 원래 했어야 할 것"** 이다.
#
# ⚠ `run_adapt` 배선이 아직이다 — `run_gate_lossparity` 의 빚 목록에 넷이 같이 올라 있다
#   (`head_conductance` · `hcond_scale` · `hcond_on_w` · `hcond_classes`).
#   **채택되면 넷을 한꺼번에 갚아야 한다** — 안 갚으면 adapt 이 log 기본값으로 되돌린다.
#
# **반증 가능한 예측**
#   ① 게이트 6/6 **유지** (evenjit40 이 6/6 이다. 무너지면 그 자리에서 기각)
#   ② ★ 홀드아웃 MAE(4.84)·총잔차(20.3)·F1(0.9318)·harm 이 **안 나빠진다**
#      (log 판은 넷 다 유의하게 나빴다. 여기서 또 나쁘면 V 불변 자체가 문제다)
#   ③ ★ 씨앗이 갈리는 창 **1.73/1.80% 가 안 는다** (log 판은 3.12/3.54 로 두 배)
#   ④ 저항 넷의 편향이 **안 나빠진다** (log 판은 포트 −9.2 -> −25.8W)
#   ⑤ 체크포인트에 `hcond_scale: 'watt'` · `hcond_on_w: 100.0` ·
#      `hcond_classes: 'RESISTIVE'` · `even_jitter: 0.4` 가 **같이** 적힌다
#   ⇒ ①~④가 전부 "안 나빠진다" 이고 ★ 진짜 이득 가설은 **오븐↔포트 네 방향**뿐이다.
#     그것도 안 움직이면 V 불변은 이 자료에서 못 읽는 것이다.
#
#SBATCH -J nilm_vn
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 01:20:00
#SBATCH -a 0-5
#SBATCH -o logs/vnorm_%A_%a.out
#SBATCH -e logs/vnorm_%A_%a.err

set -euo pipefail
# ⚠ 이 다섯 줄은 짝(`patches/cnn_evenjit_seeds.sbatch`)에서 **그대로 베낀 것**이다.
#   14.295b 에서 `PY=python` 과 없는 캐시 경로를 적었다가 던지기 직전에 잡았다.
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32h}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
$PY -X utf8 src/run_gate_deadzone.py
$PY -X utf8 -m src.run_gate_evenjit
$PY -X utf8 -m src.run_gate_hcond
$PY -X utf8 -m src.run_gate_hcondwatt
$PY -X utf8 -m src.run_gate_lossparity
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ⚠ BASE·WTAP·CAP·TREAT 는 짝(`cnn_evenjit40`)과 **글자 그대로 같다**. 다른 것은 VN 넷뿐이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40"
VN="--head-conductance --hcond-scale watt --hcond-classes RESISTIVE --hcond-on-w 100"

echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $TREAT"
echo "  처치: $VN   (짝 evenjit40 은 이 넷만 없다)"
case "$BASE$WTAP$CAP$TREAT" in *--head-conductance*|*--hcond-*) echo "처치가 바닥으로 샜다"; exit 1;; esac
case "$BASE$WTAP$CAP$TREAT$VN" in *--odd-phase-jitter*|*--fine-dc*|*--swa-start*|*--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이나 다른 처치가 섞였다"; exit 1;; esac
# ⚠ `--vexp` 는 `--head-conductance` 와 같이 못 쓴다 (같은 곱을 두 번 건다). 위 case 가 막는다.

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT $VN \
    --tag cnn_vnorm_s${SEED} > "results/cnn_vnorm_s${SEED}.log" 2>&1
echo "=== 끝 $(date '+%F %H:%M:%S')"

# ⑤ 를 그 자리에서 읽는다
$PY -X utf8 -c "
import torch, sys
p = torch.load('results/cnn_vnorm_s${SEED}.pt', map_location='cpu', weights_only=False)
print('  체크포인트: hcond=%r scale=%r on_w=%r classes=%r even_jitter=%r'
      % (p.get('head_conductance'), p.get('hcond_scale'), p.get('hcond_on_w'),
         p.get('hcond_classes'), p.get('even_jitter')))
ok = (p.get('head_conductance') and p.get('hcond_scale') == 'watt'
      and float(p.get('hcond_on_w', 0)) == 100.0
      and p.get('hcond_classes') == 'RESISTIVE' and float(p.get('even_jitter', 0)) == 0.4)
print('  ⑤ %s' % ('통과' if ok else '**실패 — 처치가 안 실렸다**'))
sys.exit(0 if ok else 1)
"
tail -4 "results/cnn_vnorm_s${SEED}.log"
