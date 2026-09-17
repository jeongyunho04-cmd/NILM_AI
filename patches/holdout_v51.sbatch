#!/bin/bash
# 14.401 — 홀드아웃 v51 (§58). `holdout_v49.sbatch` 와 `--phase-jitter-map measured` **하나**만 다르다.
#
# 왜 따로 굽나 (§48.3b·§49):
#   v49 홀드아웃에는 **일괄 ±4도 무작위 위상 지터**가 들어 있다. 그래서 그 위에서는
#   부하 의존 위상 법칙이 **묻혀 있다** — cnn_pjit 이 그 법칙을 배웠어도 v49 홀드아웃은
#   그것을 **보여 줄 수가 없다.** 자로서 약하다.
#
# ⚠⚠ **짝 비교는 여전히 v49 로 한다.** 두 팔이 같은 자를 써야 손잡이가 하나다.
#   이 홀드아웃은 **추가 자**이고, 다 구운 뒤 두 체크포인트를 여기서 **따로** 채점한다.
#   (학습 중 평가는 기울기에 안 쓰이므로 `--holdout` 을 바꿔도 모델은 안 바뀐다.)
#
#SBATCH -J nilm_ho51
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
OUT=processed_data/holdout60_v51
PAIR=processed_data/holdout60_v49
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
case "$FLAGS" in *--harmonic-z*) ;; *) echo "--harmonic-z 가 없다"; exit 1;; esac

# ★ 14.390 — **이 판의 유일한 손잡이.** 학습 캐시 v50 과 같은 값이어야 한다.
$PY -X utf8 -m src.run_build_holdout --help | grep -q -- "--phase-jitter-map" || {
  echo "코드를 안 올렸다 — --phase-jitter-map 이 없다"; exit 1; }
$PY -X utf8 -m src.run_build_holdout --out "$OUT" $FLAGS --phase-jitter-map measured
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
if d.get('phase_jitter_map') != 'measured':
    sys.exit('메타에 phase_jitter_map 이 안 적혔다 — 손잡이가 안 걸렸다')
print('  phase_jitter_map=%s' % d.get('phase_jitter_map'))
"

#: ⚠⚠ 14.390 — **짝을 v32h 에서 v49 로 바꿨다.** 원본(`holdout_v49.sbatch`)은 45채널
#:   v32h 와 견주는데 그 홀드아웃은 **지웠다**(쿼터). 986487 이 정확히 거기서 죽었다 —
#:   내가 `cache_v49_par.sbatch` 에 *"사라진 짝"* 경고를 적어 놓고 **같은 함정에 빠졌다**
#:   ([[find-patch-collisions-by-shape]] — "전제가 바뀐 땜질". 오늘 두 번째다).
#: ★ 그리고 v49 가 **진짜 짝**이다 — 같은 채널 배치에 손잡이가 위상 지터 하나뿐이다.
echo "=== 관문: v49 와 **위상 지터 말고** 다 같은가 ==="
$PY -X utf8 -c "
import json, sys
import numpy as np
a = json.load(open('$PAIR/meta.json', encoding='utf-8'))
b = json.load(open('$OUT/meta.json', encoding='utf-8'))
skip = {'content_sha256', 'bytes', 'built_at', 'positive_rate', 'recipe_counts',
        'phase_jitter_map'}
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
diff = [k for k in set(a) | set(b) if k not in skip
        and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
        != json.dumps(nz(b.get(k)), sort_keys=True, default=str)]
if diff: sys.exit('❌ 위상 지터 말고도 다르다: %s — 손잡이가 하나가 아니다' % diff)
if a.get('phase_jitter_map'): sys.exit('❌ 짝(v49)에 phase_jitter_map 이 있다')
if b.get('phase_jitter_map') != 'measured': sys.exit('❌ 새 판에 지터가 안 걸렸다')
if b.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('❌ volt_orders 가 %r 다' % b.get('volt_orders'))
if not b.get('harmonic_z'):      sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
if not b.get('workers_chunked'): sys.exit('❌ 병렬로 안 구웠다')
if int(b['target_index']) != int(a['target_index']): sys.exit('❌ target_index 가 다르다')

X = np.load('$OUT/X.npy', mmap_mode='r')
Y = np.load('$PAIR/X.npy', mmap_mode='r')
print('  X %s  (짝 %s)' % (X.shape, Y.shape))
if X.shape != Y.shape: sys.exit('❌ 모양이 짝과 다르다')
if X.shape[1] != 49:   sys.exit('❌ X 의 채널이 %d 다' % X.shape[1])

# ★ **SMPS 가 켜진 창만** 달라져야 한다 — 캐시 관문 ④ 와 같은 자다.
apps = list(b['appliances'])
SM = [apps.index(x) for x in ('minipc', 'laptop_charger', 'beam_projector') if x in apps]
yon = np.asarray(np.load('$OUT/y_on.npy', mmap_mode='r'))
ya  = np.asarray(np.load('$PAIR/y_on.npy', mmap_mode='r'))
if not np.array_equal(yon, ya): sys.exit('❌ y_on 이 짝과 다르다 — 라벨이 움직였다')
#: ⚠⚠ 14.390e — **'SMPS 없는 창' 을 y_on(타깃 사이클)으로 정의하면 안 된다.**
#:   창은 60초인데 타깃은 한 사이클이라, 창 어딘가에 SMPS 가 있는 창이 섞인다.
#:   캐시 관문 ④ 가 같은 실수로 986473_0 을 죽였고 y_plugged|y_on 으로 고치니
#:   그런 창의 최대차가 **정확히 0.000e+00** 이었다. 여기도 같은 정의를 쓴다.
try:
    ypl = np.asarray(np.load('$OUT/y_plugged.npy', mmap_mode='r'))
except FileNotFoundError:
    sys.exit('❌ y_plugged.npy 가 없다 — SMPS 유무를 제대로 정의할 수 없다')
n = min(400, X.shape[0])
d = np.abs(np.asarray(X[:n], np.float64) - np.asarray(Y[:n], np.float64)).reshape(n, -1).max(1)
pres = (ypl[:n][:, SM].any(1)) | (yon[:n][:, SM].any(1))
print('  앞 %d창 · SMPS 있는 창 %d · 없는 창 %d' % (n, int(pres.sum()), int((~pres).sum())))
print('  SMPS 있는 창 최대차 **%.3e** · **SMPS 없는 창 최대차 %.3e**'
      % (d[pres].max() if pres.any() else 0.0, d[~pres].max() if (~pres).any() else 0.0))
if (~pres).any() and d[~pres].max() != 0.0:
    sys.exit('❌ SMPS 없는 창이 달라졌다 — 손잡이가 하나가 아니다')
if pres.any() and d[pres].max() == 0.0:
    sys.exit('❌ SMPS 있는 창이 하나도 안 달라졌다 — 손잡이가 안 걸렸다')
print('  ✅ 라벨 불변 · **SMPS 없는 창은 비트 동일** · SMPS 창만 달라진다')
"
du -sh "$OUT"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
echo
echo "이 홀드아웃은 **추가 자**다 — 짝 비교(cnn_comb 대 cnn_pjit)는 v49 로 한다"
