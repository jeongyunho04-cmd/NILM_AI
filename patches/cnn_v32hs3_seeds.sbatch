#!/bin/bash
# 14.113 — **창 안 전압 텍스처**를 넣은 캐시로 학습. 시드 셋 짝비교.
#
# 처치는 깃발이 아니라 **캐시**다. BASE 는 글자 하나 안 건드리고 `--cache`/`--holdout`
# 만 바꾼다. 대조는 로컬 `cnn_hzwt_s{0,1,2}` (= 같은 BASE+WTAP, 캐시만 train60_v32h).
#
#   train60_v32h    vtex 끔        세밀 창(10초) 안 텍스처 변동 = 실측의 **22%**
#   train60_v32hs3  step=seg=1.25 · coarse=25   〃              = 실측의 **96%**
#   (14.111, 복합 실측 composite_eval 기준. 단일기기 기준으로는 31% -> 121%)
#
# 왜 이것을 사려 했나 (14.50/14.51/14.65)
#   `_terminal_voltage_harmonics` 의 `rel` 이 창 전체에서 **상수**였다. 그래서 세밀
#   갈래가 보는 10초 안에 전압 파형 변화가 **구조적으로 없었다**. 실측에는 있다.
#   모델이 "창 안에서 전압 텍스처는 안 변한다" 를 가짜 단서로 쓸 수 있었고, 텍스처가
#   움직일 때 전류가 어떻게 반응하는지는 아예 못 배웠다.
#
# ★ 던지기 전에 **잰** 기준값 (3시드, 로컬 `cnn_hzwt_s{0,1,2}`)
#     ★1 면적 2.908 +- 1.271   ★2 축퇴 33.9 +- 3.9 %
#     ★3 현행 라벨 오븐 F1 0.914 +- 0.018 (재현 0.985 / 정밀 0.853)
#     ★4 물리마스크 오븐 F1 0.859 +- 0.024 (재현 0.988 / 정밀 0.761)
#   ⚠⚠ **홀드아웃 지표는 비교하지 마라.** 홀드아웃도 같이 갈렸다
#     (holdout60_v32h -> holdout60_v32hs3). ★1~★4 는 **실측**이라 비교가 선다.
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인 검사) 캐시가 달라졌으니 모델도 달라진다. ★1~★4 중 하나라도 씨앗 바닥
#      밖으로 움직이면 처치가 닿은 것이다. 하나도 안 움직이면 **세밀 갈래가 전압
#      텍스처를 애초에 안 본다**는 뜻이고, 그 자체가 결론이다.
#   ② ★ **진짜 물음** — 이것이 오븐↔포트 축퇴를 줄이나.
#      **줄이지 않을 것으로 본다.** 저항 부하는 선형이라 전압 텍스처가 흔들리면
#      오븐도 포트도 **같은 비로** 반응한다 — 둘을 가르는 정보가 안 생긴다.
#      그러므로 ★2 는 33.9% 에서 **씨앗 바닥(±3.9) 안에 머문다**고 적는다.
#      내려가면 내 물리 추론이 틀린 것이고, 그건 그것대로 큰 소득이다.
#   ③ ★ **정말 겨냥한 곳은 SMPS 다.** SMPS 는 비선형이라 텍스처가 바뀌면 전류
#      고조파가 **기기마다 다르게** 움직인다 (회로로 잼: |I3|/|I1| 상대 0.10%).
#      그러므로 **SMPS 3종 신원이 올라야 한다.** 이게 이 판의 성공 조건이다.
#   ④ 저항 4종 신원은 안 내려가야 한다 (한 시드도 0.97 밑이면 안 된다).
#
#SBATCH -J nilm_hs3t
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
# ⚠ 갈래당 **1 GPU · 3 코어**로 쪼갰다 (14.112 의 측정). 2GPU x 8코어 덩어리는
#   빈틈에 못 들어가 3시간 대기했고, 4코어로 줄이니 CPU 가 꽉 차 GPU 가 30% 놀았다
#   (프로세스 10개 / 코어 4개). 작업 하나에 학습 하나면 프로세스가 5개(본체 1 +
#   워커 4)라 3코어로 충분하고, 작은 작업 셋이 빈틈에 훨씬 잘 들어간다.
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984629 가 4코어 나눠 쓰며 7.5초/에포크였다. 전용 3코어면 그보다 빠를 것이고,
# 대조(-c 8, 2학습)가 5.0초/에포크였다. 300에포크 = 25~40분. HPC_RULES 2 로 3배.
#SBATCH -t 01:30:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v32hs3}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32hs3}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
$PY -X utf8 src/run_gate_fmt.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
# ⚠ 이 판의 전부다 — **캐시와 홀드아웃의 처치가 같아야** 한다. 다르면 학습 분포와
#   평가 분포가 갈리고 그 차이를 모델 성능으로 읽는다.
$PY -X utf8 -c "
import json, sys
t = json.load(open('$CACHE/meta.json', encoding='utf-8'))
h = json.load(open('$HOLDOUT/meta.json', encoding='utf-8'))
bad = []
for k in ('harmonic_z', 'vtex_step_s', 'vtex_seg_s', 'vtex_coarse_s'):
    a, b = t.get(k), h.get(k)
    same = (bool(a) == bool(b)) if isinstance(a, bool) or isinstance(b, bool) \
        else (float(a or 0) == float(b or 0))
    print('    %-14s 학습 %-7r 홀드아웃 %-7r  %s' % (k, a, b, 'OK' if same else '** 다르다 **'))
    if not same: bad.append(k)
if bad: sys.exit('캐시와 홀드아웃의 처치가 다르다: %s' % bad)
for k, v in (('vtex_step_s', 1.25), ('vtex_seg_s', 1.25), ('vtex_coarse_s', 25.0)):
    if float(t.get(k) or 0) != v: sys.exit('학습 캐시 %s 가 %r 이다 (기대 %s)' % (k, t.get(k), v))
if not t.get('harmonic_z'): sys.exit('학습 캐시에 harmonic_z 가 없다')
print('  관문 OK — 처치 넷이 학습·홀드아웃에서 같고 기대값과 맞는다')
"

# ⚠ BASE + WTAP 이 cnn_hzwtap_seeds.sbatch 의 `BASE + A_FLAGS` 와 **글자 그대로 같아야**
#   한다 (캐시·홀드아웃만 빼고). 대조를 새로 안 굽고 `cnn_hzwt_s{0,1,2}` 를 짝으로 쓴다.
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
WTAP="--fine-extra-dilations 32,64 --tap-layers 0,1,4"
echo "  BASE: $BASE"
echo "  바닥(대조와 공통): $WTAP"
# 처치는 캐시뿐이다 — 손잡이가 섞이면 이 판이 무엇을 재는지 알 수 없다
case "$BASE$WTAP" in *--w-gate-cond*) echo "손잡이가 섞였다 (--w-gate-cond)"; exit 1;; esac
case "$BASE$WTAP" in *--w-swap*) echo "손잡이가 섞였다 (--w-swap)"; exit 1;; esac
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "wtap 의 탭에 4 가 없다 (14.78)"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP \
    --tag cnn_hs3_s${SEED} > "results/cnn_hs3_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_hs3_s${SEED}.log"
ls -la results/cnn_hs3_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
