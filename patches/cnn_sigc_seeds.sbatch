#!/bin/bash
# 14.28 — **기기별 지수로 고친 판**을 시드 셋으로. (980700 의 균일 판은 취소했다.)
#
# 14.27 ⑦ 이 찾은 것: 균일 `sig/vrel` 은 **모터에 틀리다.**
#   `I/P` 의 전압 지수 = `i_exp − p_exp`  ->  저항 (1,2) 과 SMPS (−1,0) 은 **−1**,
#   **모터 (0.7,0.7) 은 0** 이다 — 와트당 전류가 전압과 무관하다.
#   균일 −1 을 걸자 에어컨 MAE 가 **17.19 -> 19.35** 로 혼자 나빠졌다 (선풍기는 −0.13 로 제자리).
#   이 판은 `harm_vnorm_exp` 로 기기마다 다른 지수를 쓴다 — 9종 중 **팬·에어컨 둘만** 바뀐다.
#
# 14.27 (시드 0) 이 얻은 것: 판정 줄 on/off **+0.018** · 잔차 41.3 -> **21.9W** ·
#   실측 test_5 −113.2 -> **+16.7W** · 전압 지수 1.0 -> **2.0**.
# ⚠ 그런데 **SMPS 3종 신원이 −0.0105** 다. [[one-seed-cannot-rank-small-classes]] 가
#   **시드 하나로 ±0.03 을 못 읽는다**고 못박았으므로 이 크기는 결론이 못 된다.
#   짝지은 시드 셋으로 확정한다 — 시드마다 A·B 가 **같은 캐시·같은 배치 순서**를 쓰고
#   `--harm-sig-vnorm` **하나만** 다르다.
#
# 읽는 법: 시드별 **차이(B−A)** 셋을 놓고 본다. 평균만 보지 말 것 —
#   줄 하나가 반대면 그것이 결론이다 ([[validate-both-directions]]).
#
# 왜 (14.21~14.22):
#   상태 라벨로 묶어 재면 참값 지수가 **정확히 2.0** (1.97~2.13) 인데 모델은 **1.0** (0.89~1.09) 다.
#   물리 `P = V·I` 인데 모델은 `P = c·I` — **V 하나가 빠졌다**.
#   출처는 `L_harm` 이다: `obs_harm ≈ Σ power_k·sig_k`, `sig_k = median(I/P)` 가 **상수**다.
#   그런데 `I = P/V` 라 **`I/P ∝ 1/V`** — 상수 sig 는 "전압이 녹화 때 그대로" 라는 가정이고,
#   뒤집으면 `power ≈ 관측전류/sig ∝ V¹` 을 민다. 전력 손실은 V² 를 민다. 모델은 1 에 앉는다.
#   고치면 `power ≈ I·V/(sig·V_ref) ∝ V²` 로 **두 항의 부호가 맞는다**.
#
#   지수 1 이 test_5 에서 만들 과예측 = **91W** (기기별 α 가중) · 실측 **95W**. 맞아떨어진다.
#
#   A 대조   그대로
#   B 고침   --harm-sig-vnorm     sig -> sig · (V_CENTER / V_창)
#
# **반증 가능한 예측** (학습 전에 적는다):
#   ① B 의 저항 지수가 1.0 -> **2.0 근처**로 (상태 라벨로 잰다, `run_diag_vindist` 계열)
#      -> 안 오르면 이 설명이 틀렸다
#   ② 실측 **test_5 의 과예측이 −95W 에서 크게 줄고**, test_2 도 준다
#   ③ ⚠ **test_1·3·4 는 ±10W 안에서 그대로여야 한다** — 전압이 녹화와 가까운 파일들이다.
#      다섯이 다 움직이면 `--vexp` 때와 같은 실패다 (14.16)
#   ④ SMPS 신원이 떨어지면 안 된다 — `I/P ∝ 1/V` 는 SMPS 에도 물리적으로 옳다
#
# ⚠ 캐시가 먼저 있어야 한다 (`cache_v32_par.sbatch` 로 굽고 `run_merge_traincache` 로 이어붙인다).
#   그 캐시에는 14.25 의 상태별 지수(오븐 FAN_LIGHT 1.75)가 **묻어 들어간다** — 평균 0.128W 라
#   이 비교에는 영향이 없고, A·B 가 **같은 캐시**를 쓰므로 짝 비교는 깨끗하다.
#SBATCH -J nilm_sigc
# 사용자 지시 (2026-09-14): **A6000 Ada / A6000 만.** gpu3=a6000ada · gpu4·gpu5=a6000.
#   a10(gpu2·gpu6)·rtx3090(gpu1)은 뺀다 — A10 은 2020 Ampere 라 느리다 (HPC_RULES §2).
#   ⚠ 대신 대기가 길어질 수 있다. 980573 은 이 목록의 첫판으로 **n071(RTX 6000 Ada)** 에 즉시 붙었다.
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 01:15:00
#SBATCH -a 0-2          # ← 시드 셋. 자체 제한 3노드에 딱 맞는다 (HPC_RULES §0)
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
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--harm-sig-vnorm" || {
  echo "run_train_cnn 에 --harm-sig-vnorm 이 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }
$PY -X utf8 -c "
import json, sys
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
if d.get('shard'): sys.exit('캐시가 **토막**이다 — run_merge_traincache 로 먼저 이어붙여라')
if d.get('sibling_rotate'): sys.exit('v32 인데 sibling_rotate 가 있다 (14.4 위반)')
for k in ('float_fill','steady_crop','standby_jitter_cap','vtail','couple_ext','sp_curves'):
    if not d.get(k): sys.exit('캐시에 v32 설정이 빠졌다: %s' % k)
print('관문: 캐시 %d창 · v32 통짜' % d['n_windows'])
"
# 끄면 비트 동일 · 켜면 달라진다 · **모터는 보정이 안 걸린다** (14.28)
$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src import env_guard
import torch, numpy as np
from src.model.losses import NILMLoss, LossWeights
from src.run_train_cnn import _vnorm_exp
from src.run_baseline import S_I
apps = sorted(S_I); K = len(apps)
ex = _vnorm_exp(apps)
want = {'fan': 0.0, 'air_conditioner': 0.0, 'oven': -1.0, 'laptop_charger': -1.0}
for a, w in want.items():
    got = float(ex[apps.index(a)])
    if abs(got - w) > 1e-6: sys.exit('%s 의 지수가 %.2f 인데 %.2f 여야 한다' % (a, got, w))
print('관문: 기기별 지수 %s' % ' '.join('%s=%+.0f' % (a[:4], e) for a, e in zip(apps, ex)))
sig = np.random.default_rng(0).normal(size=(K, 15, 2)).astype(np.float32)
def mk(vn, e):
    return NILMLoss(s_i=torch.tensor([S_I[x] for x in apps]),
                    signatures=torch.from_numpy(sig), standby_sig=torch.from_numpy(sig * 0),
                    noise_sig=torch.zeros(15, 2), harm_scale=torch.ones(15),
                    harm_sig_vnorm=vn, harm_vnorm_exp=e, weights=LossWeights(harm=0.1))
a, b, u = mk(False, None), mk(True, ex), mk(True, None)
out = {'power': torch.rand(8, K) * 100, 'power_raw': torch.rand(8, K) * 100 + 1}
p0 = a._harm_pred_active(out, out['power'])
b._vrel = None
assert torch.equal(p0, b._harm_pred_active(out, out['power'])), 'vrel 없는데 달라졌다'
b._vrel = torch.full((8,), 1.05); p2 = b._harm_pred_active(out, out['power'])
u._vrel = torch.full((8,), 1.05); p3 = u._harm_pred_active(out, out['power'])
assert not torch.equal(p0, p2), 'vrel 을 줬는데 안 변했다'
assert not torch.equal(p2, p3), '균일판(-1 전부)과 같다 — 기기별 지수가 안 걸렸다'
print('관문: 끄면 항등 · 켜면 바뀜 · 균일판과 다름 (최대차 %.4f)' % float((p2 - p0).abs().max()))
"

# ── 두 판. 바뀌는 것은 **--harm-sig-vnorm 하나** ──────────────────────────
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"

echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --tag cnn_sigc_ctl_s${SEED} > "results/cnn_sigc_ctl_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --harm-sig-vnorm \
    --tag cnn_sigc_vn_s${SEED}  > "results/cnn_sigc_vn_s${SEED}.log"  2>&1 &
B=$!
echo "A(대조) $A · B(sig 고침) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: 대조 ===";      tail -22 "results/cnn_sigc_ctl_s${SEED}.log"
echo "=== B: sig 고침 ==="; tail -22 "results/cnn_sigc_vn_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 (참고 — 지수는 여기서 안 보인다, 14.6 ④) ==="
$PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_sigc_ctl_s${SEED}.pt results/cnn_sigc_vn_s${SEED}.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_sigc_ctl_s${SEED}.pt results/cnn_sigc_vn_s${SEED}.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정은 **로컬 실측**으로:"
echo "  python -X utf8 src/run_diag_sigv.py --ckpt results/cnn_sig_vn.pt   # 지수가 2 로 갔나"
echo "  python -X utf8 src/run_plot_real.py --ckpt results/cnn_sig_vn.pt   # test_5 가 닫혔나"
exit $RC
