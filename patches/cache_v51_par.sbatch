#!/bin/bash
# 14.401 — **캐시 v51.** 손잡이는 `PHASE_JITTER_DEG_MEASURED` **표 하나**다.
# v50 과 인자가 **글자까지 같고** 코드의 그 표만 다르다.
#
# 무엇을 고쳤나 (§58):
#   ① `U(−a,a)` 의 std 는 **a/√3** 인데 잰 σ 를 `a` 에 그대로 넣었다
#      -> 실제 주입 std 가 **0.577σ** 였다
#   ② σ 를 나쁜 추정기로 쟀다 — h13·h15 는 **세션 배경이 기기 전류보다 크다**(13.84.38).
#      §53 의 추정기(h3~h11 · 크기 가중)로 다시 재면 σ = 0.67 / 2.73 / 2.75
#   ⇒ 표  미니PC **1.16** · 충전기 **4.73** · 빔 **4.76**   (v50 은 0.94 / 2.13 / 3.66)
#
# 왜 (판정이 시켰다):
#   `cnn_pjit`(v50) 이 **깎은 만큼 졌다** — 미니PC 4.3배 깎음 -> 실측 AUC **−0.0378**(t −4.5),
#   빔 거의 안 깎음 -> **+0.0165**(t +3.5). 완벽한 단조다. 덜 흔들면 캐시 위상이
#   너무 깨끗해져 모델이 **전이 안 되는 위상**을 외운다 (§54: 녹화 회전은 관측 불가)
#
# ⚠ 빔은 3.66 -> 4.76 이라 **처음으로 더 흔드는** 기기다. 자를 따로 둔다 (§58.3 ④)
# ⚠ 관문 `run_gate_phaselaw` 6/6 — [4] 는 **활성화 안 법칙 불변식**이다 (14.401 에서 고쳤다)
#
#SBATCH -J nilm_v51
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 01:20:00     # 14.390e — 토막 0 은 관문 ④(두 번 굽기)에 8분을 더 쓴다
#SBATCH -a 0-7
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v51}
#: ⚠⚠ 14.390e — **`--array=0` 으로 한 토막만 다시 돌리면 `SLURM_ARRAY_TASK_COUNT`
#:   가 1 이 된다.** 그러면 토막 '0/1' = **30만 창 전부**가 되어 `fine.npy` 가
#:   21.96GB 로 부풀고 타임아웃 + 쿼터 초과로 죽는다 (986491 이 그랬다).
#:   그래서 **`NSHARD` 를 밖에서 덮어쓸 수 있게** 한다: NSHARD=8 sbatch --array=0 ...
NSHARD=${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-8}}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).

# ── 관문 ⓪ **코드가 올라왔나** ──────────────────────────────────────────────
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS
from src.synthesis.augmentor import PHASE_JITTER_DEG_MEASURED as T
print('  차수 %s · RAW %d · FINE %d · WIDE %d'
      % (tuple(VOLT_ORDERS), RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS))
print('  위상 지터 표 %s' % T)
ok = (tuple(VOLT_ORDERS) == (1, 3, 5, 7, 9, 11, 13, 15)
      and (RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS) == (49, 61, 51)
      and set(T) == {'minipc', 'laptop_charger', 'beam_projector'})
#: ⚠⚠ 14.401 — **값까지 본다.** 열쇠만 보면 v50 표로 조용히 v51 을 굽는다.
from src.synthesis.augmentor import PHASE_JITTER_DEG_V50 as OLD
if T == OLD:
    sys.exit('표가 아직 v50 이다 (%s) — 코드를 안 올렸다' % T)
if not (1.0 < T['minipc'] < 1.4 and 4.4 < T['laptop_charger'] < 5.1
        and 4.4 < T['beam_projector'] < 5.2):
    sys.exit('표 값이 14.401 이 잰 범위 밖이다: %s' % T)
sys.exit(0 if ok else '코드를 안 올렸다 — 14.389 이전 배치다')
"
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- --phase-jitter-map \
  || { echo 'CLI 에 --phase-jitter-map 이 없다'; exit 1; }

# `--phase-jitter-map` 말고는 v49 와 **글자까지 같아야 한다**.
COMMON="--windows $WINDOWS --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95"

echo "=== 관문 ① Z 표 배선 ==="
$PY -X utf8 src/run_gate_zharm.py
echo "=== 관문 ② 굽기가 타는 그 경로 (traincache._init) ==="
$PY -X utf8 src/run_gate_hzwire.py
echo "=== 관문 ③ 위상 지터 배선·표 (run_gate_phaselaw) ==="
$PY -X utf8 -m src.run_gate_phaselaw

# ── 관문 ④ **손잡이가 하나인가** — 같은 씨로 두 번 굽고 견준다 ───────────────
#   v49 합본과 견주는 대신 **여기서 두 판을 짓는다**. 그러면 짝이 지금 이 코드로
#   구워진 것이라 "옛 캐시가 어떤 코드로 구워졌나" 를 안 믿어도 된다
#   (14.332b 가 사라진 짝 캐시로 여덟 토막을 통째로 죽인 자리다).
#   ★ 걸어야 할 것: **라벨은 비트 동일** · 달라지는 창은 **SMPS 가 켜진 창뿐**.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  A="$HOME/NILM_AI/cache/_pj_off"; B="$HOME/NILM_AI/cache/_pj_on"
  rm -rf "$A" "$B"
  echo "=== 관문 ④ 손잡이 하나 확인 (청크 0/1200 을 두 번) ==="
  $PY -X utf8 -m src.run_build_traincache --out "$A" --shard "0/1200"       $COMMON --workers "${SLURM_CPUS_PER_TASK}" --harmonic-z > /dev/null
  $PY -X utf8 -m src.run_build_traincache --out "$B" --shard "0/1200"       $COMMON --workers "${SLURM_CPUS_PER_TASK}" --harmonic-z       --phase-jitter-map measured > /dev/null
  $PY -X utf8 -c "
import numpy as np, sys, json
#: ⚠⚠ 14.390d — **첫 판의 자가 틀렸다.** 986473_0 이 여기서 죽었는데 코드가 아니라
#:   자가 틀린 것이었다. 잰 것:
#:     · 'SMPS 없는 창' 을 y_on(**타깃 사이클**)으로 정의했다 -> 창 60초 **어딘가에**
#:       SMPS 가 있는 창이 거기 섞였다. y_plugged|y_on 으로 고치니 그런 창은
#:       최대차가 **정확히 0.000e+00** 이다
#:     · y_power 가 최대 0.0287W 달라졌는데 **상대차 2.3e-05**(중앙 3.1e-06)이고
#:       달라진 창 155개가 **전부** SMPS 있는 창이다 -> 합성이 float32 라 한 기기의
#:       위상을 바꾸면 3600사이클 누적에서 그만큼 흔들린다. 허용오차를 **구현
#:       정밀도**에서 가져온다 (자를 늘리는 것이 아니라 **정의를 고치는** 것이다)
bad = []
for nm in ('y_on', 'y_state', 'y_plugged'):
    a = np.asarray(np.load('$A/%s.npy' % nm, mmap_mode='r'))
    b = np.asarray(np.load('$B/%s.npy' % nm, mmap_mode='r'))
    ok = np.array_equal(a, b)
    print('    %-12s %s' % (nm, '비트 동일' if ok else '**다르다**'))
    if not ok: bad.append(nm)
apps = json.load(open('$B/meta.json', encoding='utf-8'))['appliances']
SM = [apps.index(x) for x in ('minipc', 'laptop_charger', 'beam_projector') if x in apps]
yon = np.asarray(np.load('$B/y_on.npy', mmap_mode='r'))
ypl = np.asarray(np.load('$B/y_plugged.npy', mmap_mode='r'))
pres = ypl[:, SM].any(1) | yon[:, SM].any(1)          # 창 **어디에라도** SMPS 가 있나
p0 = np.asarray(np.load('$A/y_power.npy', mmap_mode='r'))
p1 = np.asarray(np.load('$B/y_power.npy', mmap_mode='r'))
rel = np.abs(p0 - p1) / np.maximum(np.abs(p0), 1e-9)
chg = np.abs(p0 - p1).max(1) > 0
print('    y_power      상대차 최대 %.2e · 달라진 창 %d (그중 SMPS 있는 창 %d)'
      % (rel.max(), int(chg.sum()), int(pres[chg].sum())))
if rel.max() > 1e-4:             bad.append('y_power 상대차가 float32 한계를 넘는다')
if chg.sum() != pres[chg].sum(): bad.append('SMPS 없는 창의 y_power 가 달라졌다')
f0 = np.asarray(np.load('$A/fine.npy', mmap_mode='r'))
f1 = np.asarray(np.load('$B/fine.npy', mmap_mode='r'))
d  = np.abs(f0 - f1).reshape(len(f0), -1).max(1)
print('    창 %d · SMPS 있는 창 %d · 없는 창 %d' % (len(d), int(pres.sum()), int((~pres).sum())))
print('    fine  SMPS 있는 창 %.3e · **없는 창 %.3e**'
      % (d[pres].max() if pres.any() else 0.0, d[~pres].max() if (~pres).any() else 0.0))
if (~pres).any() and d[~pres].max() != 0.0:
    bad.append('SMPS 없는 창이 달라졌다')
if pres.any() and d[pres].max() == 0.0:
    bad.append('SMPS 있는 창이 하나도 안 달라졌다 — 손잡이가 안 걸렸다')
if bad: sys.exit('관문 ④ 실패: %s' % bad)
print('  관문 ④ 통과 — 라벨 불변 · **SMPS 없는 창은 비트 동일** · SMPS 창만 달라진다')
"
  rm -rf "$A" "$B"
fi

# ── 굽기 ────────────────────────────────────────────────────────────────────
time $PY -X utf8 -m src.run_build_traincache \
    --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}" \
    $COMMON --workers "${SLURM_CPUS_PER_TASK}" --harmonic-z \
    --phase-jitter-map measured

du -sh "$OUT"
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
print('  창 %d · fine %s · phase_jitter_map=%r'
      % (d['n_windows'], d['fine_shape'], d.get('phase_jitter_map')))
if d.get('phase_jitter_map') != 'measured':
    sys.exit('메타에 phase_jitter_map 이 안 적혔다 — 손잡이가 안 걸렸을 수 있다')
"
echo "=== 끝 $(date '+%F %H:%M:%S')"
