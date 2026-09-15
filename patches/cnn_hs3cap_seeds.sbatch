#!/bin/bash
# 14.126 — **2x2 의 빈 칸**: 새 캐시(텍스처) + 상한. 시드 셋 짝비교.
#
# 사용자: *"새로운 캐시로 돌린 판은 없어?"* — `cnn_hs3` 하나뿐이었고 그 뒤 손잡이
# 셋(tsp/pcap/fseg)은 전부 옛 캐시였다.
#
#                 옛 캐시 v32h        새 캐시 v32hs3
#   상한 없음      cnn_hzwt            cnn_hs3
#   상한 3        cnn_pcap            **여기**
#
# 두 처치가 **각각** 저항 4종 신원을 3/3 올렸다:
#   캐시(hs3, 짝 hzwt)  +0.0226 +- 0.0058   SMPS +0.0176 +- 0.0387 (2/3, 바닥 안)
#   상한(pcap, 짝 hzwt) +0.0247 +- 0.0000   SMPS +0.0570 +- 0.0224 (3/3)
# **가산인지 같은 자리를 다투는지 모른다.** 이 칸이 그것을 가른다.
#
# ★ 던지기 전에 **잰** 기준값 — 짝은 `cnn_hs3_s{0,1,2}` (같은 캐시·홀드아웃·BASE·WTAP,
#   `--p-state-cap 3` **하나만** 다르다)
#     ★1 면적 4.499 +- 1.506   ★2 축퇴 36.612 +- 4.089
#     ★3 현행 F1 0.927 +- 0.004   ★4 마스크 F1 0.881 +- 0.008
#     저항 4종 신원 0.9753 +- 0.0051   SMPS 3종 0.9412 +- 0.0140
#     판정 줄(7종) 0.8783 +- 0.0066    절대 잔차 20.97 +- 3.66 W
#     빔 s1 `p_states` 평균 553/543/548W (상한 30W 에 한참 위)
#
# **반증 가능한 예측** (돌리기 전에 적는다)
#   ① (확인) 빔 s1 이 **30W 에 붙는다**. 안 붙으면 전제가 틀린 것이다.
#   ② ★ **진짜 물음 — 가산인가.** 저항 4종 신원이 0.9753 에서 더 오른다.
#      옛 캐시에서 상한이 +0.0247 을 냈으니 가산이면 ~0.9975 다. 천장(1.0)이 가까워
#      전부는 못 받는다 — **0.985 넘으면 가산**, 0.9753 근처면 **같은 자리를 다툰** 것이다.
#   ③ ★ SMPS 3종 신원 — 0.9412 에서 오른다. 옛 캐시 상한이 +0.0570 이었으니
#      가산이면 ~0.998. **0.980 넘어야** 가산으로 읽는다.
#   ④ ★2 축퇴는 **안 내려갈 것으로 본다** — 상한은 SMPS 슬롯에만 걸리고(저항 넷은
#      상한의 0.38배 이하) 축퇴는 저항 문제다. 36.6 +- 4.1 안에 머문다.
#      내려가면 14.125 에서 못 찾은 간접 기전이 실재한다는 뜻이고 그건 큰 소득이다.
#   ⑤ 지킴: 판정 줄 · 절대 잔차 · ★4
#
#SBATCH -J nilm_h3cap
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
# 984668(같은 캐시)이 6초/에포크, 300에포크 약 30분이었다. HPC_RULES 2 로 1.5배 이상.
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
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
$PY -X utf8 src/run_gate_pstatecap.py
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
# ⚠ 캐시와 홀드아웃의 **처치가 같아야** 한다 (14.113 이 세운 관문 그대로).
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
print('  관문 OK — 텍스처 처치 셋이 학습·홀드아웃에서 같고 기대값과 맞는다')
"

# ⚠ BASE + WTAP 이 cnn_v32hs3_seeds.sbatch 와 **글자 그대로 같아야** 한다 —
#   대조를 새로 안 굽고 로컬 `cnn_hs3_s{0,1,2}` 를 짝으로 쓰기 때문이다.
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
case "$BASE$WTAP" in *--p-state-cap*) echo "처치가 바닥으로 샜다"; exit 1;; esac
# ⚠ `--fine-time-split`(14.116 기각) · `--fine-future-segs`(14.122 기각) 은 뺐다.
case "$BASE$WTAP$TREAT" in *--fine-time-split*|*--fine-future-segs*|*--w-gate-cond*|*--w-swap*|*--seg-pool*)
  echo "기각된 손잡이가 섞였다"; exit 1;; esac
case "$WTAP" in *"--tap-layers 0,1,4"*) ;; *) echo "wtap 의 탭에 4 가 없다 (14.78)"; exit 1;; esac

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
$PY -X utf8 -m src.run_train_cnn $BASE $WTAP $TREAT \
    --tag cnn_h3cap_s${SEED} > "results/cnn_h3cap_s${SEED}.log" 2>&1
RC=$?
tail -32 "results/cnn_h3cap_s${SEED}.log"
ls -la results/cnn_h3cap_s${SEED}.pt || true
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
