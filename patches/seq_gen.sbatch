#!/bin/bash
# 13.84.49 + 13.84.61 — 형제 증강을 고치고 v35 가림을 푼다. **2x2 요인설계**.
#
# 이 파일 하나를 `GEN` 만 바꿔 두 번 낸다:
#     GEN=v36p  sbatch patches/seq_gen.sbatch     # 손으로 맞춘 폭 (smps_dev2)
#     GEN=v36dp sbatch patches/seq_gen.sbatch     # **실측 표류** (smps_driftp)
# 각 작업이 한 노드에서 가림 A/B 두 판을 돌린다 -> 칸 네 개.
#
# ── 무엇을 재나 ────────────────────────────────────────────────────────────
# 13.84.48: 전력 19~28W 로 맞추고 녹화하나빼기로 채점하면 충전기와 미니PC 가 h1 기준
#   **상대 위상** 하나로 AUC 1.000 으로 갈린다. 그런데 모델은 그것을 못 본다 — 두 번 지웠다:
#     13.84.13 (v35) --zero-channels ... = Re/Im h9,11,13,15 를 **가림**
#     13.84.16 (v36) smps_dev1 의 scramble_from:9 = 형제 h9+ 위상을 **뒤섞음**
# 13.84.49 가 고쳤고(smps_dev2: c_max 0.9, 뒤섞기 제거) 972468 이 관문 버그로 죽어 **안 걸렸다**.
#
# 13.84.61 이 더 나간다 — 폭만 맞추지 말고 **실측 표류를 그대로 넣는다**.
#   관문(`run_gate_driftaug.py`)이 잰 것, 실측 표류 부분공간 대비:
#     프리셋        주각            부분공간 밖    으뜸축-미니PC   폭(실측=1.00)
#     smps_dev1    32/64/78°         52.6%          70°        6.6~12.9배
#     smps_dev2    13/26/82°         23.4%          58°        0.62~1.10
#     smps_driftp   0/ 6/32°       **3.2%**       **85°**      0.88~0.94
#     (실측 표류)        —             0.0%          89°           1.00
#   smps_driftp 는 자유도 9개 모수가 아니라 13.84.52 의 **3차원 기저**에서 뽑아 **더한다**.
#
# ⚠ 왜 손실이 아니라 생성기인가: 13.84.60 이 같은 부분공간을 `L_harm` 에서 사영으로 지워
#   봤고 **졌다** — 표류 PC1 이 미니PC-형제 판별축과 24~25° 라 표류와 함께 판별력이 간다.
#   지울 수 없으면 가르쳐야 한다.
# ⚠ 둘 다 **처음부터** 배운다 (13.84.27). v37 몸통에 가림을 풀면 물려받은 가중치에 본 적
#   없는 입력이 들어간다. 초기점을 맞춰야 재는 것이 가림 하나가 된다.
# ⚠ `--keep-on 1.0` — 처음부터 배우는 판은 켜짐 BCE 를 빼면 게이트가 무감독이 된다.
# ⚠ 13.84.37 이 가림만 풀어 아무것도 못 얻은 것은 **위상이 뒤섞인 캐시**였기 때문이다.
#   생성기를 먼저 고쳐야 가림 해제가 뜻을 갖는다 — 이 판이 그 순서다.
#SBATCH -J nilm_gen
#SBATCH -p gpu3,gpu2,gpu4,gpu6,gpu1
#SBATCH -N 1
#SBATCH -n 1
# ── 코어 (2026-09-12 재조사) ────────────────────────────────────────────────
# ⚠ 앞선 기록의 "gpu6·gpu1 QOS 상한 40" 은 **틀렸다.** 실제 QOS 는 node10_cpu640_gpu12 =
#   사용자당 640코어·12GPU·10노드다. 파티션별 코어 상한은 없다. 한계는 **노드 크기**뿐이고
#   gpu1·gpu6 이 48, gpu2·gpu3·gpu4 가 56, gpu5 가 64 다.
# 제출 시각 실황: 유휴 >=52코어 4노드 · >=48 9노드 · >=44 9노드 · >=40 16노드 · >=56 0노드.
# 48 은 44 와 후보 수가 같으므로(둘 다 9노드) **48 이 공짜다.** 52 로 올리면 후보가 4노드로
# 준다 — +4코어(판당 워커 +2, 약 +8%) 값이 안 된다.
# CPU 가 병목이다: 972281 에서 GPU 32~43% · VRAM 5% · load 36/38. 카드 종류는 거의 무의미하니
# 파티션은 카드가 아니라 **여유 코어** 순으로 늘어놓았다 (gpu5 는 지금 자리가 없어 뺐다).
#SBATCH -c 48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
# 972281 실측 413초/epoch(워커 18). 워커 23 이면 약 330초 x 12 = 66분 + 캐시 10분 = 1:16.
# 2:30 은 2배다 (§2 — 짧게 걸어야 backfill 에 들어간다).
#SBATCH -t 02:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

GEN=${GEN:-v36dp}
RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_${GEN}"

NCPU=$(nproc)
W=$(( (NCPU - 2) / 2 ))
[ "$W" -ge 4 ] || { echo "코어가 너무 적다 (nproc=$NCPU)"; exit 1; }
BW=$(( NCPU - 4 ))
echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · GEN $GEN · nproc $NCPU · 판당 워커 $W · 굽기 $BW"
nvidia-smi --query-gpu=index,name --format=csv,noheader || nvidia-smi | head -12

cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에, §7-3) ───────────────────────────────────────────────────────
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--keep-last" || { echo "--keep-last 가 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--no-mask"   || { echo "--no-mask 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# 표류 기저는 **여기서** 만든다 (형제 격리 녹화만 쓴다 — 실측 복합은 HPC 에 없다)
if [ "$GEN" = "v36dp" ]; then
  $PY -X utf8 -m src.run_build_drift --block 600 --k 3 || { echo "표류 기저 생성 실패"; exit 1; }
  test -f results/drift_basis.npz || { echo "drift_basis.npz 가 없다"; exit 1; }
fi

# ⚠⚠ 핵심 관문 — 증강 폭·모양이 실측과 맞는가. 안 맞으면 구울 이유가 없다.
WANT=$([ "$GEN" = "v36dp" ] && echo smps_driftp || echo smps_dev2)
$PY -X utf8 -m src.run_gate_driftaug -n 2000 --check "$WANT" || { echo "증강 관문 실패 ($WANT)"; exit 1; }
# 대조로 옛 판도 찍는다 — **떨어지는 것이 정상**이라 종료코드를 보지 않는다
$PY -X utf8 -m src.run_gate_driftaug -n 400 --check smps_dev1 || true

$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src.synthesis.genopts import PRESETS
from src.synthesis.augmentor import SIBLING_ROTATE_PRESETS
gen = '$GEN'
o = PRESETS.get(gen)
if not o: sys.exit('genopts 에 %s 가 없다 — 코드를 안 올렸다' % gen)
want = {'v36p': 'smps_dev2', 'v36dp': 'smps_driftp'}[gen]
if o.get('sibling_rotate') != want: sys.exit('%s 가 %s 가 아니다: %r' % (gen, want, o.get('sibling_rotate')))
c = SIBLING_ROTATE_PRESETS[want]['laptop_charger']
if 'scramble_from' in c: sys.exit('%s 에 scramble_from 이 아직 있다' % want)
if want == 'smps_driftp' and not c.get('drift_scale'): sys.exit('drift_scale 이 없다')
for k in ('sp_curves','sp_per_texture','vtail','float_fill','steady_crop','state_mix',
          'power_scale_std','couple_ext','standby_jitter_cap'):
    if not o.get(k): sys.exit('%s 가 V36 의 %s 를 잃었다' % (gen, k))
print('관문: %s = V36 + %s' % (gen, want))
"

# ── 1 캐시 ──────────────────────────────────────────────────────────────────
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다"; exit 1; }

mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen "$GEN" --workers "$BW" --seed 0

$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != '$GEN': sys.exit('캐시가 $GEN 이 아니다: %r' % m.get('gen'))
want = {'v36p': 'smps_dev2', 'v36dp': 'smps_driftp'}['$GEN']
o = m.get('gen_opts', {})
if o.get('sibling_rotate') != want: sys.exit('캐시 메타의 sibling_rotate 가 %r' % o.get('sibling_rotate'))
print('캐시 관문 통과 · 기록 %d · 단계 %d' % (m['records'], m['steps']))
"
du -sh "$SHM"

# ── 2 두 판 — 바뀌는 것은 **가림 하나**뿐이다 ────────────────────────────────
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers $W --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init scratch --ref results/cnn_v37.pt --lr 1e-4 --keep-on 1.0 --keep-last 4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON           --tag seq_${GEN}_mask > "results/seq_${GEN}_mask_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --no-mask --tag seq_${GEN}_open > "results/seq_${GEN}_open_s${SEED}.log" 2>&1 &
B=$!
echo "A(가림 유지) $A · B(가림 해제) $B — $(date '+%F %H:%M:%S')"

( sleep 900; echo "[점유] $(date '+%H:%M:%S') $(uptime | sed 's/.*average/load/')"; nvidia-smi | grep Default || true ) &
MON=$!

RC=0
wait $A || RC=1
wait $B || RC=1
kill $MON 2>/dev/null || true

echo "=== A: 가림 유지 ===";  tail -20 "results/seq_${GEN}_mask_s${SEED}.log"
echo "=== B: 가림 해제 ===";  tail -20 "results/seq_${GEN}_open_s${SEED}.log"
ls -la results/seq_${GEN}_*_ep*.pt 2>/dev/null || echo "(ep 체크포인트 없음 — --keep-last 확인)"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
