#!/bin/bash
# 14.315 — ★ **전이 컨덕턴스 계단** `--wide-dg` (사용자 제안 ③). 짝은 `cnn_evenjit40`(씨앗 여섯).
#
# 사용자 제안의 ③ 만 떼어 왔다. 나머지(①물리분해 ②κ가중 ④어드미턴스 ⑤사전대조)는
# 한 번에 다섯이 바뀌면 무엇이 이겼는지 못 가르기 때문에 **이것이 통한 뒤**에 간다
# ([[count-how-many-things-differ-before-attributing]]).
#
# 무엇을 하나 — 광역 60초에서 `G(τ) = P(τ)/V(τ)²` 의 **가장 큰 계단**을 찾아 그 앞뒤
# 차이를 채널 넷(|ΔĜ| · 부호 · 자리 · 유효)으로 **세밀 갈래 몸통 입력**에 더한다.
#   ⓐ `G` 는 전압으로 나눈 값이라 전이 때 생기는 **전압 계단이 이미 보정**돼 있다
#   ⓑ 실측 참 전이 101개가 전부 **한 번에 한 대**다 -> 그 차이가 **기기 하나의 G** 다
#   ⓒ 크기는 **선형**(/20mS)이다. arcsinh 면 오븐 24.6 ↔ 포트 27.9 의 13.4% 를 압축하는데,
#      그게 오늘 `--head-conductance`(log)를 죽인 바로 그 짓이다
#
# **왜 이 팔인가** — 잰 값 둘이 근거다 (합성 300기록 · `cache/seqraw_k`):
#   [가용성] 기기가 켜진 창 안에 **그 기기 자신의** 전이가 있는 비율
#       세밀 10초  오븐(가열) 47.9% · 포트 10.3%
#       광역 60초  오븐(가열) **89.1%** · 포트 57.8%
#     => "오븐이 켜졌다" 에 **반증 가능한 물증**이 생긴다. 24.6mS 계단이 창 안에 없으면
#        오븐이 아니다. 지금 최대 항이 **C포트오탐 9.5**(오븐 쪽 헛게이트)다
#   [정확도] 관문 [4] — ΔĜ 가 **어느 한 기기의 참 G** 와 얼마나 맞나
#       가장 가까운 기기와의 상대거리 중앙 **2.3%** · 15% 안에 드는 창 **77%**
#     => 아무 수나 나오는 게 아니라 진짜 기기 하나를 집는다
#
# 그리고 이건 **이미 통한 것의 모델 안 판**이다 — 14.305 의 스위치 희소성 디코더가
# 사후에 한 일(합 60 -> 31.5, p=0.031)을 입력 채널로 앞당긴다.
#
# ⚠⚠ **반례를 먼저 적는다.** 14.94 가 광역 축을 닫았다 — 손잡이 셋(풀링·수용영역·
#   타깃탭)이 전부 실패했고 **넷 다 씨앗 산포가 기준선의 2~7배**였다. 결론이
#   *"신호가 없는 갈래에 용량을 주면 해가 여러 개 생긴다"* 였다.
#   그 셋은 광역에 **용량**을 준 것이고 이건 **정보**다. 그래서 광역에는 파라미터를
#   한 톨도 안 준다 — 광역은 **숫자의 출처**로만 쓰고 파생 넷은 세밀에 넣는다.
#   ⇒ 그래도 씨앗 산포가 늘면 **그 자리에서 기각**이다. 14.94 가 옳았던 것이다.
#
# ⚠ 삭제 실험("광역을 지워도 저항 넷 게이트가 0.0~0.2%")으로 이 팔을 미리 기각하면
#   **순환논증**이다 (사용자 지적). 그건 **지금 실린 내용**이 쓸모없다는 뜻이지 그
#   갈래가 새 정보를 못 나른다는 뜻이 아니다.
#
# ⚠ `run_adapt` 배선이 없다 — 그런데 이건 **이 팔 이전부터** 깨져 있다. `run_adapt` 은
#   모델 인자를 아홉 개만 물려받아 `--fine-extra-dilations`(우리 처방 전체가 쓴다)를
#   못 읽는다: `Unexpected key(s): fine.5.0.weight` 로 **크게** 죽는다(조용히가 아니다).
#   `wide_dg` 는 그 목록에 하나 더 얹힐 뿐이다. 2단계를 되살릴 때 같이 갚는다.
#
# **반증 가능한 예측**
#   ① 게이트 6/6 **유지** (evenjit40 이 6/6 이다. 무너지면 그 자리에서 기각)
#   ② ★ 씨앗이 갈리는 창 **1.73/1.80% 가 안 는다** — 14.94 의 "산포만 는다" 의 직접 시험
#   ③ ★ 진짜 이득 가설은 **C포트오탐**이다 (디코더 뒤 9.5). 오븐 쪽 헛게이트가 준다
#   ④ 홀드아웃 MAE(4.84)·총잔차(20.3)·F1(0.9318)·harm 이 **안 나빠진다**
#   ⑤ 체크포인트에 `wide_dg: True` · `even_jitter: 0.4` 가 **같이** 적힌다
#   ⇒ ②가 무너지면 14.94 의 결론이 정보 채널에도 적용되는 것이고, 축을 완전히 닫는다.
#
#SBATCH -J nilm_wdg
#SBATCH -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -a 0-5
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
$PY -X utf8 -m src.run_gate_evenjit
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
# ⚠ `run_gate_widedg` 는 원시 열 캐시(`cache/seqraw_k`)가 있어야 돈다. HPC 에 없으면
#   건너뛴다 — 그 관문은 **노트북에서 8/8 로 통과시켰다** (14.315).
if [ -d cache/seqraw_k ]; then $PY -X utf8 -m src.run_gate_widedg; else echo "  (seqraw_k 없음 — widedg 관문 건너뜀)"; fi

# ⚠ BASE·WTAP·CAP·TREAT 는 짝(`cnn_evenjit40`)과 **글자 그대로 같다**. 다른 것은 WDG 하나다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
CAP="--p-state-cap 3"
TREAT="--even-median 5 --w-cons 1.0 --cons-deadzone 100 --even-jitter 0.40"
WDG="--wide-dg"

echo "  BASE: $BASE"
echo "  바닥(짝과 공통): $WTAP $CAP $TREAT"
echo "  처치: $WDG   (짝 evenjit40 은 이것만 없다)"
case "$BASE$WTAP$CAP$TREAT" in *--wide-dg*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ 기각된 손잡이는 섞지 않는다 (14.116·14.122·14.128·14.131·14.16·14.152·14.296·14.303)
case "$BASE$WTAP$CAP$TREAT$WDG" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*|*--head-drop*|*--vexp*|*--head-conductance*|*--hcond-*|*--odd-phase-jitter*|*--fine-dc*|*--swa-start*|*--on-power-praw*|*--on-detach-gate*|*--off-detach-praw*|*--gate-free-power*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
# ⚠ `--wide-dg` 는 `--head-layout v2` 와 못 쓴다 (v2 는 `_conv_in` 을 안 탄다). 생성자가 막지만
#   여기서도 막아 **던지기 전에** 잡는다.
case "$BASE$WTAP$CAP$TREAT$WDG" in *"--head-layout v2"*) echo "wide_dg 는 head_layout=v2 와 못 쓴다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $CAP $TREAT $WDG \
    --tag cnn_widedg_s${SEED} > "results/cnn_widedg_s${SEED}.log" 2>&1
RC=$?
echo "=== 끝 $(date '+%F %H:%M:%S')"

# ⑤ 를 그 자리에서 읽는다
$PY -X utf8 -c "
import torch, sys
p = torch.load('results/cnn_widedg_s${SEED}.pt', map_location='cpu', weights_only=False)
print('  체크포인트: wide_dg=%r even_jitter=%r fine_derive=%r head_layout=%r'
      % (p.get('wide_dg'), p.get('even_jitter'), p.get('fine_derive'), p.get('head_layout')))
ok = (p.get('wide_dg') is True and float(p.get('even_jitter', 0)) == 0.4
      and p.get('head_layout') == 'v1')
print('  ⑤ %s' % ('통과' if ok else '**실패 — 처치가 안 실렸다**'))
sys.exit(0 if ok else 1)
"
tail -32 "results/cnn_widedg_s${SEED}.log"
ls -la results/cnn_widedg_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
