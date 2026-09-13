#!/bin/bash
# 13.84.49 — 형제 위상 폭을 실측에 맞추고, v35 가림을 푼다. 짝지은 2판.
#
# 13.84.48 이 잰 것: 전력 19~28W 로 맞추고 **녹화 하나 빼기**로 채점하면 충전기와 미니PC 가
# h1 기준 **상대 위상** 하나로 **AUC 1.000 (18짝 전부, 표준편차 0)** 로 갈린다.
# 도 단위로는 h5 에서 60°, h9 에서 96° 떨어져 있고 녹화 간 산포는 2~18° 뿐이다.
#
# 그런데 모델은 그것을 못 본다 — 우리가 두 번 지웠다:
#   13.84.13 (v35) --zero-channels 4,5,6,7,12,13,14,15,37,38  = Re/Im h9,11,13,15 를 **가림**
#   13.84.16 (v36) --sibling-rotate smps_dev1 의 scramble_from:9 = 형제 h9+ 위상을 **뒤섞음**
# `run_gate_phase.py` 실측: smps_dev1 이 만드는 상대위상 산포가 실측의 **12~42배**다
# (h9 171.8° 대 실측 4.1°). 회전만 걸어도 12배다.
#
# 고침: `smps_dev2` = 회전 c_max 10.0 -> **0.9**, 뒤섞기 **제거**, 크기 항은 그대로.
#       관문이 실측의 0.6~1.1배로 확인했다. 생성기 설정은 `v36p` = V36 + smps_dev2.
#
#   A: 가림 유지 (대조)
#   B: **--no-mask** (가림 해제)  <- 이 판이 재는 것
#
# ⚠ 둘 다 **처음부터** 배운다. v37 몸통에 가림을 풀면 물려받은 가중치에 본 적 없는 입력이
#   들어간다 (13.84.27). 초기점을 맞춰야 재는 것이 가림 하나가 된다.
# ⚠ `--keep-on 1.0` — 처음부터 배우는 판은 켜짐 BCE 를 빼면 게이트가 무감독이 된다 (13.84.27).
#   13.84.37 은 keep-on 이 창별을 공짜로 살린다고 쟀다 (0승8패 p=0.0078).
# ⚠ 13.84.37 이 가림만 풀어(keep03nm) 아무것도 못 얻은 것은 **위상이 뒤섞인 캐시**였기 때문이다.
#   생성기를 먼저 고쳐야 가림 해제가 뜻을 갖는다 — 이 판이 그 순서다.
#SBATCH -J nilm_phase
#SBATCH -p gpu4,gpu3,gpu2,gpu6,gpu1,gpu5
#SBATCH -N 1
#SBATCH -n 1
# 판당 워커 18 x 2 + 주 2 = 38. §5 의 교차점이고 가장 빡빡한 QOS 상한(gpu6·gpu1 의 40) 아래다.
# 972281 이 이 모양으로 n071 에 90초 만에 잡혔고 판당 4,180창/초를 냈다.
#SBATCH -c 38
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
# 972281 실측 411~420초/epoch x 12 = 84분 + 캐시 10분 = 1:34. 3:00 은 1.9배다 (§2).
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_v4"

NCPU=$(nproc)
W=$(( (NCPU - 2) / 2 ))
[ "$W" -ge 4 ] || { echo "코어가 너무 적다 (nproc=$NCPU)"; exit 1; }
BW=$(( NCPU - 4 ))
echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · nproc $NCPU · 판당 워커 $W · 굽기 $BW"
nvidia-smi --query-gpu=index,name --format=csv,noheader

cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에, §7-3) ───────────────────────────────────────────────────────
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--keep-last" || { echo "--keep-last 가 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--no-mask"   || { echo "--no-mask 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다 (--ref 로 구조를 가져온다)"; exit 1; }

# ⚠⚠ 핵심 관문 — 위상 폭이 실측과 맞는가. 안 맞으면 구울 이유가 없다.
# 대조로 smps_dev1 도 찍는다 — 그쪽은 **막히는 것이 정상**이라 종료코드를 보지 않는다.
$PY -X utf8 -m src.run_gate_phase smps_dev1 || true
$PY -X utf8 -m src.run_gate_phase smps_dev2 || { echo "위상 폭 관문 실패"; exit 1; }

$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src.synthesis.genopts import PRESETS
o = PRESETS.get('v36p')
if not o: sys.exit('genopts 에 v36p 가 없다 — 코드를 안 올렸다')
if o.get('sibling_rotate') != 'smps_dev2': sys.exit('v36p 가 smps_dev2 가 아니다: %r' % o.get('sibling_rotate'))
from src.synthesis.augmentor import SIBLING_ROTATE_PRESETS
c = SIBLING_ROTATE_PRESETS['smps_dev2']['laptop_charger']
if 'scramble_from' in c: sys.exit('smps_dev2 에 scramble_from 이 아직 있다')
if abs(c['c_max'] - 0.9) > 1e-9: sys.exit('c_max 가 0.9 가 아니다: %r' % c['c_max'])
for k in ('sp_curves','sp_per_texture','vtail','float_fill','steady_crop','state_mix',
          'power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('v36p 가 V36 의 %s 를 잃었다' % k)
print('관문: v36p = V36 + smps_dev2 (c_max 0.9 · 뒤섞기 없음)')
"

# ── ① 캐시 ──────────────────────────────────────────────────────────────────
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다"; exit 1; }

mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen v36p --workers "$BW" --seed 0

$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != 'v36p': sys.exit('캐시가 v36p 가 아니다: %r' % m.get('gen'))
o = m.get('gen_opts', {})
if o.get('sibling_rotate') != 'smps_dev2': sys.exit('캐시 메타의 sibling_rotate 가 %r' % o.get('sibling_rotate'))
print('캐시 관문 통과 · 기록 %d · 단계 %d' % (m['records'], m['steps']))
"
du -sh "$SHM"; cp "$SHM/meta.json" "results/seqraw_v4_meta.json"

# ── ② 두 판 — 바뀌는 것은 **가림 하나**뿐이다 ────────────────────────────────
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers $W --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init scratch --ref results/cnn_v37.pt --lr 1e-4 --keep-on 1.0 --keep-last 4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON              --tag seq_ph_mask  > "results/seq_ph_mask_s${SEED}.log"  2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --no-mask    --tag seq_ph_open  > "results/seq_ph_open_s${SEED}.log"  2>&1 &
B=$!
echo "A(가림 유지) $A · B(가림 해제) $B — $(date '+%F %H:%M:%S')"

( sleep 600; echo "[점유] $(date '+%H:%M:%S') $(uptime | sed 's/.*average/load/')"; nvidia-smi | grep Default || true ) &
MON=$!

RC=0
wait $A || RC=1
wait $B || RC=1
kill $MON 2>/dev/null || true

echo "=== A: 가림 유지 ===";  tail -20 "results/seq_ph_mask_s${SEED}.log"
echo "=== B: 가림 해제 ===";  tail -20 "results/seq_ph_open_s${SEED}.log"
ls -la results/seq_ph_*_ep*.pt 2>/dev/null || echo "(ep 체크포인트 없음 — --keep-last 확인)"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
