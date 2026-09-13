#!/bin/bash
# 13.84.70 — **빈 상태 전력 슬롯** 고침을 건다. 한 노드에서 두 팔.
#
#     sbatch patches/seq_slot.sbatch
#
# ── 무엇을 재나 ────────────────────────────────────────────────────────────
# 13.84.68: 전력은 `p_raw = Σ_s mix[s]·p_states[s]` 인 **곱**인데 슬롯을 따로 붙잡는 항이
#   없다 (`L_state_power` 가 0 이고, 게다가 `lossbuild` 가 그 가중을 **안 넘겨서** 시퀀스
#   학습기에서는 닿을 수조차 없었다). 드라이기 약풍 슬롯이 죽어서, 시퀀스 학습이 `L_state`
#   로 분류를 정확하게 만들자(`mix[1] = 0.997`) 전력이 **466W -> 3W** 로 무너졌다.
#   **분류가 좋아져서 전력이 나빠진다.** 드라이기만이 아니다 — 오븐 상태1 15->1W,
#   포트 상태1 1456->919W.
#
# 13.84.70 의 고침 둘과 이 판의 사다리:
#     기준선   seq_v36p_mask (**이미 있다**)   초기화 옛것 · state_power 0
#     팔 A     seq_slot_init                  초기화 **새것** · state_power 0
#     팔 B     seq_slot_both                  초기화 **새것** · state_power **0.3**
#   기준선은 다시 안 굽는다. 13.84.69 의 네 판이 이미 `--init scratch` 라 설정이 같다
#   (v36p · 가림 유지 · keep_on 1.0 · 12 epoch · seed 0) — **바뀌는 것이 고침 하나**다.
#
# ⚠ 왜 `--init scratch` 인가: 고침 ① 은 **새 판에만** 듣는다. 물려받으면 `load_state_dict`
#   가 바이어스를 덮고, 이미 죽은 슬롯은 `σ(−30) ≈ 9e-14` 라 ② 로도 **못 살아난다**
#   (관문 [4]: 죽은 슬롯 기울기 2e-18 = 산 슬롯의 1e-13 배). 그래서 warm start 로는 이
#   결함을 못 고친다.
# ⚠ 생성기는 **v36p** 다. 13.84.69 가 v36dp(실측 표류)를 기각했고 가림은 유지가 낫다.
# ⚠ 이 판은 13.84.63 의 주 루프 고침(crf CPU + pin_memory)을 **처음으로** 태운다.
#   값은 같다(관문에서 1.2e-07) 지만 **속도가 다르니 epoch 시간을 앞 판과 직접 견주지 마라**.
# ⚠ 값이 좋아진다는 보장은 없다. 12.35 가 `state_power` 로 유령을 42.1 -> 86.9W 로 키웠다고
#   쟀다 (옛 판, 다른 구성). 팔 A 와 B 를 나눈 이유가 그것이다 — ① 만으로 되는지 먼저 본다.
#SBATCH -J nilm_slot
#SBATCH -p gpu3,gpu2,gpu4,gpu6,gpu1
#SBATCH -N 1
#SBATCH -n 1
# 코어 근거는 `seq_gen.sbatch` 와 같다 (QOS node10_cpu640_gpu12, 노드 크기가 한계).
#SBATCH -c 48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
# 972759(v36p, A10) 가 1:46:08 이었다. 주 루프 고침이 얹히니 더 짧을 것이다. 2:30 은 여유다.
#SBATCH -t 02:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

GEN=${GEN:-v36p}
RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
WSP=${WSP:-0.3}
SHM="/dev/shm/$USER/seqraw_${GEN}"

NCPU=$(nproc)
W=$(( (NCPU - 2) / 2 ))
[ "$W" -ge 4 ] || { echo "코어가 너무 적다 (nproc=$NCPU)"; exit 1; }
BW=$(( NCPU - 4 ))
echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · GEN $GEN · w_state_power $WSP · nproc $NCPU · 판당 워커 $W"
nvidia-smi --query-gpu=index,name --format=csv,noheader || nvidia-smi | head -12

cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에, §7-3) ───────────────────────────────────────────────────────
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--w-state-power" || { echo "--w-state-power 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--no-state-init"  || { echo "--no-state-init 가 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--keep-last"      || { echo "--keep-last 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# ⚠⚠ 핵심 관문 — 고침이 꺼진 채로는 항등이고 켜면 동작하는가. 안 되면 구울 이유가 없다.
$PY -X utf8 -m src.run_gate_states || { echo "빈 슬롯 관문 실패"; exit 1; }

# 증강 관문 (seq_gen.sbatch 와 같은 것 — 생성기가 그대로인지 본다)
$PY -X utf8 -m src.run_gate_driftaug -n 2000 --check smps_dev2 || { echo "증강 관문 실패"; exit 1; }

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
if m.get('gen_opts', {}).get('sibling_rotate') != 'smps_dev2':
    sys.exit('캐시 메타의 sibling_rotate 가 %r' % m.get('gen_opts', {}).get('sibling_rotate'))
print('캐시 관문 통과 · 기록 %d · 단계 %d' % (m['records'], m['steps']))
"
du -sh "$SHM"

# ── 2 두 팔 — 바뀌는 것은 **state_power 하나**뿐이다 ────────────────────────
# 기준선(seq_v36p_mask)과 이 둘 사이에 바뀌는 것은 ① 초기화 하나. A 와 B 사이는 ② 하나.
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers $W --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init scratch --ref results/cnn_v37.pt --lr 1e-4 --keep-on 1.0 --keep-last 4"

CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_seq $COMMON \
  --tag seq_slot_init > "results/seq_slot_init_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_seq $COMMON --w-state-power "$WSP" \
  --tag seq_slot_both > "results/seq_slot_both_s${SEED}.log" 2>&1 &
B=$!
echo "A(초기화만) $A · B(초기화+state_power $WSP) $B — $(date '+%F %H:%M:%S')"

( sleep 900; echo "[점유] $(date '+%H:%M:%S') $(uptime | sed 's/.*average/load/')"; nvidia-smi | grep Default || true ) &
MON=$!

RC=0
wait $A || RC=1
wait $B || RC=1
kill $MON 2>/dev/null || true

echo "=== A: 초기화만 ===";              tail -20 "results/seq_slot_init_s${SEED}.log"
echo "=== B: 초기화+state_power ===";    tail -20 "results/seq_slot_both_s${SEED}.log"
ls -la results/seq_slot_*_ep*.pt 2>/dev/null || echo "(ep 체크포인트 없음 — --keep-last 확인)"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
