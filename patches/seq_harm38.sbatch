#!/bin/bash
# 13.84.38 — 순방향 모형의 구멍 둘을 메운다 (2026-09-12). 2x2 요인 4판.
#
# 13.84.35 가 잰 것: 참값 라벨·참값 와트를 **그대로** 넣어도 `L_harm` 이 h11~h15 에서
# 미니PC 12W 신호의 1.0~3.1배를 못 맞추고, 그것이 기록마다 CV 92~96% 로 움직인다.
# 환경(R,X,V,vh)은 그 중 1~7% 만 설명한다. 두 구멍이 확인됐다:
#
#  ① 전력 의존 (--pow-sig). 손실은 `Σ_k sig_k·power_k` 라 **전력에 선형**인데 와트당 고차
#     함량이 동작점에 따라 47~109% 변한다 (13.84.32). 전력 구간별로 바꾸면 순방향 잔차 −13~34%.
#     잰 보정비: 충전기 h13 이 전력대마다 1.60 / 0.86 / 0.49 (3.3배) · h1 은 1.00 고정.
#  ② 세션 배경 (--bg-head). `noise_sig` 는 **전역 상수 하나**인데 실측 배경은 파일 **안**
#     0.19~0.66mA 로 거의 완벽히 일정하고 파일 **간** 3.6~22.6mA 다. h15 에서는 미니PC 보다 크다.
#     ⚠ 창마다 자유로우면 슬랙이 된다 — 조각의 z 평균 **하나**에서 값 하나를 낸다.
#
#   A: (대조)                  969137 의 seq_v37 재현이어야 한다 — 관문 노릇도 한다
#   B: --pow-sig
#   C: --bg-head
#   D: --pow-sig --bg-head
#
# ⚠ 초기점은 **v37 이어받기**로 고정한다 (13.84.37 에서 네 scratch 판 전부에게 유의하게 이겼다,
#   0승8패~1승7패 p=0.0078~0.07). 이 판이 재는 것은 **손실**이지 초기점이 아니다.
# ⚠ 13.59 의 함정 — 유령만 보면 안 된다. 채점은 파일별로 **짝지어** 읽는다
#   (실패 ② test_3·test_4 유령 · 실패 ③ test_1 미탐). 한쪽만 좋아지면 문턱이 움직인 것이다.
#SBATCH -J nilm_harm38
#SBATCH -p gpu6,gpu5,gpu4,gpu3,gpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 44
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH -t 03:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_v2"   # 969137·969590 과 **같은 인수** -> 바이트 동일

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 기록 $RECORDS · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ⚠ 공용 계정이다. 쓰고 나면 램을 반드시 돌려준다.
cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 ────────────────────────────────────────────────────────────────────
# ⚠ 관문은 **앞에** 둔다. 없으면 세 시간 뒤에야 틀린 것을 안다 (HPC_RULES §7-3).
$PY -X utf8 -m src.run_build_seqraw --help | grep -q -- "--gen"      || { echo "run_build_seqraw 에 --gen 이 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq    --help | grep -q -- "--pow-sig"  || { echo "run_train_seq 에 --pow-sig 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq    --help | grep -q -- "--bg-head"  || { echo "run_train_seq 에 --bg-head 가 없다 — 코드를 안 올렸다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# 새 손실이 **실제로 달라지는지** 먼저 확인한다 ([[verify-the-gate-runs-that-path]]).
# 계측기가 0 이면 그 코드는 안 돈 것이다 — "변화 없음"은 통과가 아니다.
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
import torch
from src.model.lossbuild import build_loss
from src.model.chain import BgHead
apps = ['air_conditioner','beam_projector','electiric_kettle','fan','hair_dryer','hotplate','laptop_charger','minipc','oven']
c0 = build_loss(apps, 'cpu', verbose=False)
c1 = build_loss(apps, 'cpu', power_signatures=True, verbose=False)
K = len(apps); S = int(c1.sig_state.shape[1]); B = 32
torch.manual_seed(0)
o = {'power': torch.rand(B, K) * 60, 'power_raw': torch.rand(B, K) * 60 + 1,
     'on_logit': torch.randn(B, K), 'plugged_logit': torch.randn(B, K),
     'power_mix': torch.softmax(torch.randn(B, K, S), -1), 'power_states': torch.rand(B, K, S) * 60}
p0 = c0._harm_pred_active(o, o['power']); p1 = c1._harm_pred_active(o, o['power'])
if not int(c1.use_pow_sig): sys.exit('pow_sig 가 안 걸렸다')
if float((p1 - p0).abs().max()) < 1e-6: sys.exit('pow_sig 가 예측을 **안 바꾼다** — 보정비가 전부 1 이다')
g = c1.pow_gain[..., 0] + 1j * c1.pow_gain[..., 1]
j = apps.index('laptop_charger')
print('관문: pow_sig 산다 · 충전기 h13 보정비 %s' % ' '.join('%.2f' % abs(g[j, b, 12]) for b in range(g.shape[1])))
bh = BgHead(256)
if float(bh(torch.randn(2, 8, 256)).abs().max()) != 0.0: sys.exit('배경 머리가 0 에서 출발하지 않는다')
print('관문: 배경 머리 0 출발 · 상한 %.0f mA' % (1000 * bh.cap))
"

# ── 램 자리 ─────────────────────────────────────────────────────────────────
# 기록당 3.269MB (원시 45x18000x4 = 3.240 + 라벨 0.029).
# ⚠ SLURM cgroup 이 /dev/shm 을 `--mem` 에 잡는다: 캐시 98GB + 워커 44개 + 주 프로세스 4판.
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다 (${FREE}GB < ${NEED}GB x1.3)"; exit 1; }

# ── ① 캐시를 램에 굽는다 ────────────────────────────────────────────────────
mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen v36 --workers 30 --seed 0

$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != 'v36': sys.exit('캐시가 v36 생성기가 아니다: %r' % m.get('gen'))
o = m.get('gen_opts', {})
for k in ('sibling_rotate','sp_curves','sp_per_texture','vtail','float_fill','steady_crop','state_mix','power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('생성기 설정 %s 가 빠졌다' % k)
print('캐시 관문 통과 · 기록 %d · 단계 %d · 기기 %d' % (m['records'], m['steps'], len(m['appliances'])))
"
du -sh "$SHM"; df -h /dev/shm | tail -1
cp "$SHM/meta.json" "results/seqraw_v2_meta.json"

# ── ② 네 판을 동시에 ────────────────────────────────────────────────────────
# 같은 캐시·같은 씨앗·같은 초기점·같은 예산. 바뀌는 것은 **손실 항 둘**뿐이다.
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers 10 --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init results/cnn_v37.pt --ref results/cnn_v37.pt --lr 1e-4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON                       --tag seq_h38_base > "results/seq_h38_base_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --pow-sig             --tag seq_h38_pow  > "results/seq_h38_pow_s${SEED}.log"  2>&1 &
B=$!
CUDA_VISIBLE_DEVICES=2 $PY -X utf8 -m src.run_train_seq $COMMON --bg-head             --tag seq_h38_bg   > "results/seq_h38_bg_s${SEED}.log"   2>&1 &
C=$!
CUDA_VISIBLE_DEVICES=3 $PY -X utf8 -m src.run_train_seq $COMMON --pow-sig --bg-head   --tag seq_h38_both > "results/seq_h38_both_s${SEED}.log" 2>&1 &
D=$!
echo "A(대조) $A · B(pow) $B · C(bg) $C · D(둘다) $D — $(date '+%F %H:%M:%S')"
RC=0
wait $A || RC=1
wait $B || RC=1
wait $C || RC=1
wait $D || RC=1

echo "=== A: 대조 (969137 seq_v37 재현이어야 한다) ==="; tail -20 "results/seq_h38_base_s${SEED}.log"
echo "=== B: --pow-sig ===";               tail -20 "results/seq_h38_pow_s${SEED}.log"
echo "=== C: --bg-head ===";               tail -20 "results/seq_h38_bg_s${SEED}.log"
echo "=== D: 둘 다 ===";                   tail -20 "results/seq_h38_both_s${SEED}.log"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5). 969002·969137·969590 이 그 한 줄 때문이었다.
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
