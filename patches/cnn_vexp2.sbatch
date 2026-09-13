#!/bin/bash
# 14.9 — 14.7 의 `--vexp` A/B 를 **다시** 건다. 979793 이 왜 죽었는지부터.
#
# ⚠⚠ 979793 은 `--cache none` 이라 **14분에 1에폭도 못 끝냈다.** 300에폭은 75시간이다.
#   원인은 굽기가 느려서가 **아니라 굽기를 없애서**다. 캐시가 있는 이유는 창을 50번
#   재사용하기 때문이다 (30만창 캐시 ÷ 에폭당 5만창 × 300에폭 = 50.0회).
#   `--cache none` 은 그 재사용을 0 으로 만들어 **생성 일감을 50배**로 키운다.
#     캐시    30만창 한 번 굽기 19분(64코어) + 학습 19분  = ~40분
#     실시간  1,500만창 생성                               = 16시간+
#   14.8 이 근거로 든 "12.1절: 캐시보다 워커가 낫다(창 재사용 0)" 는 **13.10.5 가 이미 뒤집었다**
#   — 13.2 의 회로모델 델타(`SmpsCircuit`, `randomize_r=True`)가 생성기를 430 -> 56창/초로
#   10배 늦췄다. `run_train_cnn.py:59` 주석이 아직 옛 수치를 이고 있다.
#   [[derived-limits-outlive-their-reason]] [[legacy-artifacts-shadow-new-files]]
#
# ⚠⚠ 쿼터는 **실재한다** — `mmlsquota` 로만 보인다. 79.51G / 소프트 100G / 하드 110G.
#   `df` 는 파일시스템 전체(466T)를 보여줄 뿐 이 질문에 답을 **못 한다**. 세 번 틀렸다.
#   그래서 22.8GB 캐시를 GPFS 에 못 둔다 -> **`/dev/shm` 에 굽는다** (노드 램, 쿼터 무관).
#   `/dev/shm` 은 **노드 국소**라 CPU 노드에서 굽고 GPU 노드에서 읽을 수 없다.
#   그래서 **한 작업 안에서** 굽고 바로 학습한다. cgroup 이 `--mem` 에 잡으므로 넉넉히
#   요청하고 `trap` 으로 반드시 반납한다.
#
# 재는 것 (14.6): 저항 부하는 `P = V²/R` 인데 모델이 배운 지수가 **0.85** 다.
#   드라이기 녹화 227.2V · test_5 는 210.6V -> 물리 0.859 대 모델 0.937 = **+9.1% 과예측**.
#   A 대조   그대로
#   B vexp   --vexp   p_raw *= (V/222)^e_k  · 저항 2.0 · SMPS 0.0 · 유도기 0.6
#   두 판은 **같은 캐시**를 읽는다 — `--vexp` 하나만 다르다.
#
# ⚠ 판정은 **실측 전력**으로 로컬에서 한다. 합성 홀드아웃은 창 전압이 녹화 전압과 같아
#   α≈1 이라 k 가 절반 이하로 틀려도 잔차가 −0.7~−1.2% 로 깨끗하다 (14.6 ④).
# ⚠ 개선은 **test_5·test_2 에서만** 보여야 정상이다. 대조(cnn_v37) 기준선:
#     test_1 +1.5W · test_2 +15.2W · test_3 −17.3W · test_4 +2.5W · test_5 **−112.5W**
#   (부호는 관측−예측이라 음수가 과예측이다.) 나머지 셋이 움직이면 무언가 잘못된 것이다.
#SBATCH -J nilm_vexp2
#SBATCH -p gpu6,gpu5,gpu4,gpu3,gpu2,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40           # gpu1·gpu6 의 노드당 상한 40 (HPC_RULES §2). 굽기가 이 코어를 다 쓴다
#SBATCH --gres=gpu:2
#SBATCH --mem=200G      # /dev/shm 캐시 22.8G 가 여기 잡힌다 + 두 판. 넉넉히
#SBATCH -t 02:30:00     # 굽기 ~30분(40코어) + 두 판 병렬 ~25분 + 채점. 1.5~2배 여유 (§2)
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

EPOCHS=${EPOCHS:-300}
SEED=${SEED:-0}
HOLDOUT=${HOLDOUT:-processed_data/holdout60_v32}

# ── /dev/shm 자리. 반드시 반납한다 ─────────────────────────────────────────
#   ⚠ 이 경로는 **이 작업이 만든 것만** 가리킨다 ($SLURM_JOB_ID). 공용 계정이라
#     다른 것을 지울 수 있는 모양을 아예 만들지 않는다.
SHM="/dev/shm/nilm_${SLURM_JOB_ID}"
CACHE="$SHM/train60_v32"
cleanup() { rm -rf "$SHM" 2>/dev/null || true; echo "반납 $SHM"; }
trap cleanup EXIT INT TERM
mkdir -p "$CACHE"

echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 코어 ${SLURM_CPUS_PER_TASK} · epoch $EPOCHS · 씨앗 $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "캐시 자리 $CACHE"
df -h /dev/shm | tail -1

# ── 관문은 전부 앞에 (HPC_RULES §7-3) ──────────────────────────────────────
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vexp" || { echo "run_train_cnn 에 --vexp 가 없다 — 코드를 안 올렸다"; exit 1; }
test -d "$HOLDOUT" || { echo "홀드아웃이 없다: $HOLDOUT"; exit 1; }

# ① 굽기 플래그가 **`genopts.V32` 와 같은가.** 두 입구를 서로 맞댄다.
#    genopts 가 있는 이유가 이것이다 — 창 캐시와 다른 경로가 조용히 갈라지는 것을 막는다
#    (13.84.27). 여기서 갈리면 A/B 가 재는 것이 `--vexp` 가 아니게 된다.
#    [[pin-the-two-entry-points-against-each-other]]
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
from src.synthesis.genopts import V32
flags = dict(power_scale_std='measured', state_mix='minipc_balanced', carrier_apps=('oven',),
             couple_ext=True, sp_curves=True, sp_per_texture=True, vtail=True,
             float_fill='charger_float', steady_crop='smps_steady', standby_jitter_cap=95.0)
if flags != V32:
    sys.exit('굽기 플래그가 genopts.V32 와 다르다: %s' % (set(flags.items()) ^ set(V32.items())))
if 'sibling_rotate' in V32: sys.exit('V32 에 sibling_rotate 가 있다 (14.4 롤백 위반)')
print('관문: 굽기 플래그 == genopts.V32 · sibling_rotate 없음')
"

# ② `--vexp` 가 실제로 지수를 더하는가. 끄면 항등인가.
$PY -X utf8 -c "
import sys
sys.path.insert(0, '.')
from src import env_guard
from src.model.net import NILMNet, V_EXP, appliance_state_counts
from src.run_baseline import S_I
apps = sorted(S_I)
off = NILMNet(apps, appliance_state_counts(apps), vexp=False)
on = NILMNet(apps, appliance_state_counts(apps), vexp=True)
if float(off.v_exp.abs().sum()) != 0.0: sys.exit('끈 판의 v_exp 가 0 이 아니다')
want = [V_EXP.get(x, 0.0) for x in apps]
# ⚠ float32 라 **정확한 등호를 쓰지 마라** — 0.6 이 0.60000002 로 들어간다.
if max(abs(float(x) - y) for x, y in zip(on.v_exp, want)) > 1e-5:
    sys.exit('켠 판의 v_exp 가 V_EXP 와 다르다')
print('관문: v_exp 버퍼 %s' % ' '.join('%.1f' % x for x in on.v_exp))
"

# ③ 풀·곡선·꼬리 (cache_v32.sbatch 의 관문 그대로)
$PY -X utf8 -c "
import numpy as np
from src.synthesis.coupling import SmpsCircuit
c = SmpsCircuit(); rel = np.zeros(15, complex); rel[0] = 1
c.current('laptop_charger', 45.0, rel, 224.0); print('선컴파일 OK')
"
$PY -X utf8 -c "
import sys, inspect
sys.path.insert(0, '.')
from src import env_guard
from src.run_build_sp_curves import TEX_CURVES
from src.synthesis.sp_curves import load_curves
t = load_curves(TEX_CURVES); assert len(t) >= 8
from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, load_vtail
tl = load_vtail(DEFAULT_VTAIL_NPZ); assert tl and 'E3' in tl
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.augmentor import FLOAT_FILL_PRESETS, STEADY_CROP_PRESETS
from src.model.traincache import build_cache
p = SegmentPool(npz_dir='processed_data/npz', time_split='train', carrier_apps=('oven',),
                standby_jitter_cap_pct=95.0)
c = p.appliance_activations.get('laptop_charger', [])
assert any('laptop_charger_6' in str(a.source_file) for a in c), 'charger_6 가 풀에 없다'
assert 'smps_steady' in STEADY_CROP_PRESETS and 'charger_float' in FLOAT_FILL_PRESETS
sig = inspect.signature(build_cache).parameters
assert all(k in sig for k in ('float_fill', 'steady_crop', 'standby_jitter_cap')), '코드를 안 올렸다'
print('텍스처 곡선', len(t), '· 전압 꼬리 키', sorted(tl), '· 충전기 활성화', len(c))
print('관문 통과')
"

# ── 굽기: /dev/shm 에 30만창 (`cache_v32.sbatch` 와 **같은 값**) ────────────
echo "=== 굽기 시작 $(date '+%F %H:%M:%S') · 40코어 · 예상 ~30분 (64코어 19분)"
time $PY -X utf8 -m src.run_build_traincache \
    --out "$CACHE" \
    --windows 300000 --window-cycles 3600 --seed 0 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --power-scale-std measured \
    --recipe-mix steady2 \
    --smps-focus-off-p 0.4 \
    --state-mix minipc_balanced \
    --carrier-on oven \
    --couple-ext \
    --sp-curves --sp-per-texture --vtail \
    --float-fill charger_float \
    --steady-crop smps_steady \
    --standby-jitter-cap 95
echo "=== 굽기 끝 $(date '+%F %H:%M:%S')"
du -sh "$CACHE"; df -h /dev/shm | tail -1

# ④ 구운 것이 v32 인가 (`cache_v32.sbatch` 의 뒷관문 그대로)
$PY -X utf8 -c "
import json, sys
import numpy as np
d = json.load(open('$CACHE/meta.json', encoding='utf-8'))
for k in ('n_windows','seed','sp_per_texture','vtail','couple_ext','carrier_apps','state_mix',
          'smps_focus_off_p','float_fill','steady_crop','standby_jitter_cap','build_seconds'):
    print('  %-20s %s' % (k, d.get(k)))
if d.get('sibling_rotate'): sys.exit('구운 캐시에 sibling_rotate 가 있다 (14.4 위반)')
pr = d.get('positive_rate', {})
print('  positive_rate', json.dumps({k: round(v,3) for k,v in pr.items()}, ensure_ascii=False))
lo = {'air_conditioner': 0.025, 'oven': 0.15, 'hotplate': 0.12, 'electiric_kettle': 0.12, 'hair_dryer': 0.10}
bad = [(k, pr.get(k)) for k, v in lo.items() if pr.get(k, 0) < v]
if bad: sys.exit('양성률 하한 미달: %s' % bad)
if not d.get('float_fill') or not d.get('steady_crop') or not d.get('standby_jitter_cap'):
    sys.exit('meta 에 v32 플래그가 없다')
apps = d['appliances']; kc = apps.index('laptop_charger')
yo = np.load('$CACHE/y_on.npy', mmap_mode='r'); yp = np.load('$CACHE/y_power.npy', mmap_mode='r')
on = np.asarray(yo[:, kc]) == 1; p = np.asarray(yp[:, kc])[on]
print('  충전기 ON 창 %d · <20W 몫 %.3f · p10/50/90 %s'
      % (on.sum(), (p < 20).mean(), np.round(np.percentile(p,[10,50,90]),1)))
if (p < 20).mean() < 0.08: sys.exit('축소 충전기가 너무 적다')
print('  관문 통과')
"

# ── 두 판을 같은 캐시로, 동시에 ────────────────────────────────────────────
#   `mmap_mode='r'` 이라 tmpfs 쪽 22.8GB 는 두 판이 **나눠 쓴다** (한 번만 잡힌다).
BASE="--cache $CACHE --holdout $HOLDOUT \
      --epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session"

echo "=== 학습 시작 $(date '+%F %H:%M:%S') · 예상 판당 ~19분"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE        --tag cnn_vexp_ctl > "results/cnn_vexp_ctl_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE --vexp --tag cnn_vexp_on  > "results/cnn_vexp_on_s${SEED}.log"  2>&1 &
B=$!
echo "A(대조) $A · B(vexp) $B"
RC=0
wait $A || RC=1
wait $B || RC=1

echo "=== A: 대조 ===";   tail -25 "results/cnn_vexp_ctl_s${SEED}.log"
echo "=== B: --vexp ==="; tail -25 "results/cnn_vexp_on_s${SEED}.log"
echo "=== 같은 홀드아웃에 나란히 (참고용 — 이것으로 판정하지 마라, 14.6 ④) ==="
$PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt \
    --holdout "$HOLDOUT" 2>&1 | tail -40 || true
ls -la results/cnn_vexp_ctl.pt results/cnn_vexp_on.pt 2>/dev/null || true
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 — 작업 셋이 그렇게 죽었다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
