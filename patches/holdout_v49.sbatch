#!/bin/bash
# 14.333 — **v49 홀드아웃**: `holdout60_v32h` 와 **전압 차수** 하나만 다르다.
#   VOLT_ORDERS (1,3,5,7,9,11) -> **(1,3,5,7,9,11,13,15)** · RAW 45 -> **49**
#
# ⚠⚠ **학습 캐시와 반드시 같은 손잡이여야 한다** (`holdout._w_init` 의 14.59 주석).
#   짝은 `cache/train60_v49` (985953, 굽기 끝나고 병합했다). 하나만 돌리지 마라.
#
# ⚠⚠ **이 판에는 옛 관문이 원리적으로 못 미친다.** 홀드아웃 meta 에는 채널 수도 shape 도
#   안 적히므로, 차수만 바뀐 v32h 와 v49 의 meta 가 **한 글자도 안 달라진다** — 짝 대조
#   관문이 **다른 배치를 조용히 통과시킨다.** 그래서 14.333 에서 meta 에 `volt_orders` 를
#   적게 고쳤고, 아래 관문 ⓪ 가 **굽기 전에** 코드가 올라왔는지 먼저 본다.
#
# ⚠ 985942 의 교훈: 베낀 sbatch 의 **짝 경로가 살아 있는지** 먼저 본다. 여덟 토막이
#   `cache/train60_v32` 가 없어서 통째로 죽었다 (그 캐시는 14.62 물건이었다).
#SBATCH -J nilm_ho49
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24           # 노드당 상한(gpu1·gpu6 은 40)보다 아래로 — backfill 이 잘 문다
#SBATCH --mem=64G       # 워커당 ~1.5GB + 청크 24 x 65MB + memmap
#SBATCH -t 00:15:00     # v32h 실측 3~5분(24워커). 채널이 4개 늘 뿐이라 그대로 둔다
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
OUT=processed_data/holdout60_v49
PAIR=processed_data/holdout60_v32h
W=${WORKERS:-24}

echo "노드 $(hostname) · 워커 ${W} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).

# ── 관문 ⓪ **코드가 올라왔나** — 이 판에서는 이게 제일 중요한 관문이다 ──────────
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS
print('  차수 %s · RAW %d · FINE %d · WIDE %d'
      % (tuple(VOLT_ORDERS), RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS))
ok = (tuple(VOLT_ORDERS) == (1, 3, 5, 7, 9, 11, 13, 15)
      and (RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS) == (49, 61, 51))
sys.exit(0 if ok else '코드를 안 올렸다 — 14.331 이전 배치다')
"
$PY -X utf8 -c "
import sys, inspect
from src.evaluation import holdout as H
if 'volt_orders' not in inspect.getsource(H):
    sys.exit('holdout.py 가 meta 에 volt_orders 를 안 적는다 — 14.333 코드를 안 올렸다')
print('  holdout.py 가 volt_orders 를 적는다')
"

# ── 관문: 배선이 닿았나 ────────────────────────────────────────────────────
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--harmonic-z" || {
  echo "--harmonic-z 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--workers" || {
  echo "--workers 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$PAIR" || { echo "짝 홀드아웃 $PAIR 가 없다 — 대조를 못 한다"; exit 1; }
test -f "$PAIR/meta.json" || { echo "$PAIR/meta.json 이 없다"; exit 1; }
test -d cache/train60_v49 || { echo "짝 학습 캐시 cache/train60_v49 가 없다"; exit 1; }
# ⚠⚠ **함수가 아니라 배선을 재라.** 이 둘은 진짜 창을 합성해 본다.
$PY -X utf8 src/run_gate_zharm.py
$PY -X utf8 src/run_gate_hzwire.py
$PY -X utf8 src/run_gate_holdout_par.py --n 120 --tmp "${TMPDIR:-/tmp}"

# ⚠⚠ 인자를 **한 곳에 모아** 굽기 **전에** 되짚는다 (984095 가 여기서 조용히 어긋났다).
#   `holdout_v32h.sbatch` 에서 **글자 그대로** 베꼈다 — 달라지는 것은 차수뿐이고
#   그건 인자가 아니라 `inputs.py` 의 상수다.
FLAGS="--windows 8000 --window-cycles 3600 --holdout-frac 0.2 --seed 20260821 \
        --workers "$W" --chunk-windows 100 \
        --recipe-mix steady2 \
        --smps-focus-off-p 0.4 \
        --state-mix minipc_balanced \
        --carrier-on oven \
        --couple-ext \
        --sp-curves --sp-per-texture --vtail \
        --float-fill charger_float \
        --steady-crop smps_steady \
        --standby-jitter-cap 95 \
        --harmonic-z"
echo "굽기 인자: $FLAGS"
case "$FLAGS" in *--vtex*) echo "vtex 가 남았다 — 학습 캐시는 seg 끔이다"; exit 1;; esac
case "$FLAGS" in *--harmonic-z*) ;; *) echo "--harmonic-z 가 없다 — 짝이 v32h 다"; exit 1;; esac

$PY -X utf8 -m src.run_build_holdout --out "$OUT" $FLAGS

echo "=== 관문: v32h 와 **차수 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
import numpy as np
a = json.load(open('$PAIR/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'volt_orders', 'content_sha256', 'bytes', 'built_at', 'positive_rate',
        'recipe_counts'}
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
        != json.dumps(nz(b.get(k)), sort_keys=True, default=str)]
if diff: sys.exit('❌ 차수 말고도 다르다: %s — 손잡이가 하나가 아니다' % diff)
if b.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('❌ meta 의 volt_orders 가 %r 다 — 옛 코드로 구웠다' % b.get('volt_orders'))
if a.get('volt_orders') is not None:
    sys.exit('❌ 짝($PAIR)에 volt_orders 가 있다 — 짝이 v32h 가 아니다')
if not b.get('harmonic_z'):      sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
if not b.get('workers_chunked'): sys.exit('❌ 병렬로 안 구웠다 — 워커가 0으로 떨어졌다')
if int(b['target_index']) != int(a['target_index']): sys.exit('❌ target_index 가 다르다')
# ⚠ meta 에 채널 수가 안 적히므로 **자료를 직접 본다** — 이게 이 판의 핵심 관문이다.
X = np.load('$OUT/X.npy', mmap_mode='r')
Y = np.load('$PAIR/X.npy', mmap_mode='r')
print('  X %s  (짝 %s)' % (X.shape, Y.shape))
if X.shape[1] != 49:   sys.exit('❌ X 의 채널이 %d 다 — 49 여야 한다' % X.shape[1])
if Y.shape[1] != 45:   sys.exit('❌ 짝의 채널이 %d 다 — 45 여야 한다' % Y.shape[1])
if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
    sys.exit('❌ 창 수·길이가 짝과 다르다')
# 전압 블록 **앞**(0..32 = 고조파 30 + P·Q·V)은 한 바이트도 안 달라져야 한다.
n = min(200, X.shape[0])
if not np.array_equal(np.asarray(X[:n, :33]), np.asarray(Y[:n, :33])):
    sys.exit('❌ 비전압 채널 0..32 가 짝과 다르다 — 손잡이가 하나가 아니다')
print('  ✅ 차수만 다르다 · 비전압 33채널 비트 동일 (앞 %d창) · volt_orders %s'
      % (n, b['volt_orders']))
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
echo
echo "이제 두 팔을 던질 수 있다:  CACHE=cache/train60_v49 HOLDOUT=$OUT"
