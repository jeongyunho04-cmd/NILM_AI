#!/bin/bash
# 14.121 — **상태 전력 슬롯의 상한** (`--p-state-cap 3`). 시드 셋 짝비교.
#
# 사용자: *"test_4 의 310~371초에서 빔프로젝터의 소비전력이 쪼그라드는 건지, 그리고
#   빔은 켜지면 46W 로 일정한 기기인데 그게 반영돼 있는지."*
#
# 원인을 팠더니 **슬롯이 위로 표류해 있었다.** 실측 창에서 9개 체크포인트를 재면:
#   beam_projector s1  초기  10.0W -> **83.6배** (자리채움 슬롯, 주석은 "p90 4.5")
#   laptop_charger s1  초기  36.4W ->   4.31배
#   minipc         s1  초기  10.7W ->   2.17배
#   그 밖 17개 슬롯                ->   1.52배 이하 (대부분 0.8~0.9배)
#
# 기전은 13.84.68 주석의 **거울상**이다. 저쪽은 혼합이 안 고르는 슬롯이 아래로 죽어
# 못 살아나는 것을 막았다 (`state_power_init`). 위쪽은 안 막혀 있었다 — 기울기
# `mix[s]·∂L/∂p_raw` 가 작지만 **부호가 일정**하고 `softplus` 에 상한이 없어서 300에포크
# 동안 쌓인다. 그러고 나면 상태 머리가 6%만 그쪽을 골라도 `p_raw` 가 통째로 튄다:
#   test_4 310~371초  빔 `p_raw` **99.2W** (참 40.6W) x 게이트 0.346 = 26.6W
#   -> "쪼그라들었다" 로 보이지만 **게이트가 죽고 수준은 2.4배로 튄 것**이다.
#
# ⚠ 상한을 **추론에서만** 씌워 보면 안 갈린다 — 모델이 표류를 껴안고 배웠다.
#   222~250초에서 빔이 46.6 -> 38.9W 로 참값(40.6)에 가까워지는데 잔차는 -3.0 ->
#   -10.7W 로 나빠진다 (빔의 부풀림이 놓친 미니PC 9W 를 가리고 있었다). 학습까지
#   켜야 공정하다.
#
# ★ R=3 은 **위 측정에서 온 값**이다 — 통과시키려고 고른 값이 아니다.
#   병적인 둘(83.6배·4.31배)만 걸리고 나머지 18개 슬롯은 손도 안 탄다.
#
# ★ 던지기 전에 **잰** 기준값 (3시드, 로컬 `cnn_hzwt_s{0,1,2}` · 같은 캐시·같은 글자)
#     ★1 면적 2.908 +- 1.271   ★2 축퇴 33.9 +- 3.9 %
#     저항 4종 신원 0.9527 +- 0.0029   SMPS 3종 0.9237 +- 0.0261
#     판정 줄(7종) 0.8885 +- 0.0020    절대 잔차 21.1 +- 2.3 W
#     빔 전력 (참 ON 전체) 39.7 +- **9.9** W   (참 40.6W · 확신할 때는 sd 0.6W)
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인 검사) 학습 뒤 빔 s1 이 **30W 에 붙어** 있다. 안 붙으면 표류가 학습
#      과정의 산물이 아니라는 뜻이고, 그러면 이 처방의 전제가 틀린 것이다.
#   ② ★ **진짜 물음** — 빔 전력의 변동이 준다. 참 ON 동안 sd **9.9 -> 6 밑**, 3/3.
#      표류한 슬롯이 변동의 원인이면 여기가 내려간다. 안 내려가면 원인은 딴 데 있다.
#   ③ ★ test_4 310~371초에서 빔 `p_raw` 가 **99.2 -> 60W 밑**. (참 40.6W)
#   ④ **★2 축퇴는 안 변할 것으로 본다.** 이건 SMPS 슬롯 문제이고 오븐<->포트 축퇴는
#      저항 기기 문제다. ★2 짝차가 씨앗 바닥(+-3.9) 안에 머문다. 내려가면 내 인과가
#      틀린 것이고 그건 그것대로 큰 소득이다.
#   ⑤ 지킴: 저항 4종 신원 · SMPS 3종 · 판정 줄 · 절대 잔차 (전부 짝차로)
#
#SBATCH -J nilm_pcap
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
# 14.112 측정대로 갈래당 1GPU·3코어. 상한은 `minimum` 하나라 비용이 사실상 0이다.
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984682 가 같은 바닥에서 6초/에포크, 300에포크에 약 33분이었다. HPC_RULES 2 로 1.5배 이상.
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
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--p-state-cap" \
  || { echo "--p-state-cap 이 없다"; exit 1; }

# ⚠ BASE + WTAP 이 cnn_hzwtap_seeds.sbatch 의 `BASE + A_FLAGS` 와 **글자 그대로 같아야**
#   한다 — 대조를 새로 안 굽고 로컬 `cnn_hzwt_s{0,1,2}` 를 짝으로 쓰기 때문이다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
TREAT="--p-state-cap 3"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP"
echo "  처치: $TREAT"
# 손잡이가 섞이면 이 판이 무엇을 재는지 알 수 없다
case "$BASE$WTAP" in *--p-state-cap*) echo "처치가 BASE 로 샜다"; exit 1;; esac
# ⚠ `--fine-time-split` 은 **일부러 뺐다.** 14.116 판정에서 지킴은 다 올랐지만
#   ★2 축퇴가 +7.10 +- 3.09 로 **악화**했다 — 채택 여부가 아직 미정이다.
#   여기 섞으면 상한의 효과와 분리가 안 된다.
case "$BASE$WTAP$TREAT" in *--fine-time-split*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "다른 손잡이가 섞였다"; exit 1;; esac
case "$WTAP" in *"--fine-extra-dilations 32,64"*) ;; *)
  echo "wtap 이 없다 — 대조와 바닥이 달라진다"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $TREAT \
    --tag cnn_pcap_s${SEED} > "results/cnn_pcap_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_pcap_s${SEED}.log"
ls -la results/cnn_pcap_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
