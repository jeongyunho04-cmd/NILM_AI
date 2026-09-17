#!/bin/bash
# 14.332 — **캐시 재굽기**. 전압 홀수 8차수 (14.331). 짝은 `cache/train60_v32h` 다.
#
# 왜 다시 굽나 — `VOLT_ORDERS` 를 (1,3,5,7,9,11) 에서 **(1,3,5,7,9,11,13,15)** 로 넓혔다
# (14.331, 사용자 결정). 채널 수가 바뀌므로 **기존 캐시와 체크포인트가 통째로 못 쓴다**:
#     RAW 45 -> **49** · FINE 57 -> **61** · WIDE 47 -> **51**
#
# 잰 근거 (파일 13개 = 단독 8 + 실측 5 의 `|V_h|/|V_1|` 중앙):
#     홀수 h13 **1.76e-3** · h15 **1.76e-3**  <- h7(1.88e-3)·h11(1.08e-3) 과 같은 급
#     짝수 전부 8.5e-5 ~ 4.2e-4               <- 홀수의 1/10~1/100. **넣지 않는다**
#   선형 저항은 전압에 성분이 있는 차수에서만 전류를 내므로 짝수차 G 열은 0 이 되고,
#   짝수차의 관측 전류는 전부 비선형 몫이라 템플릿이 가져가야 맞다.
#
# ⚠ `Ĝ_sum` 추정 자체에는 이득이 **없다** (6차수 0.111 대 8차수 이상 0.107 mS, 실측).
#   넓힌 이유는 **어드미턴스 특징 `Y_h = I_h/V_h`** 가 그 차수의 전압을 요구해서다 (14.320).
#
# ⚠⚠ **`vtex_seg_s` 는 안 넣는다.** 사용자가 처음에 요청했고 나도 넣으려 했는데,
#   14.94 를 다시 읽으니 **그 축은 이미 접혀 있다**:
#     `vblock`(세밀) 짝차  SMPS −0.57±0.88 · 저항 +0.11±0.10 · 게이트 +0.00±0.13 — **전부 잡음**
#     => "입력을 8% 바꿔도 모델이 전압 블록에 기대는 정도가 안 변한다.
#         **phase 2(seg_s=5)의 선행 조건이 떨어졌다.** 지렛대도 **60배 작다**"
#        (hz 는 Re(V3) 를 44% 바꿨는데 seg_s=5 는 |V3/V1| 의 0.67% 다)
#   ⇒ 되살린다면 **이 캐시로 재학습한 뒤 `run_diag_vblock --block fine` 을 다시 재고 나서**다.
#     차수가 6 -> 8 로 바뀌었으니 그 측정이 새 배치에서도 성립하는지는 **모른다.**
#   ⚠ 되살릴 때는 `--vtex-seg-s` 균일이 아니라 **`--vtex-coarse-s`** 를 써라 —
#     세밀 갈래는 3600사이클 중 **마지막 600(10초)** 만 보므로
#     `--vtex-seg-s 1.25 --vtex-coarse-s 25` 가 **비용 절반에 세밀 해상도 2.7배**다.
#     (균일 `seg_s=10` 은 세밀 창 안에 경계를 **하나도** 안 만든다 — 겨냥이 세밀인데 무의미하다.)
#
#SBATCH -J nilm_v49
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 00:50:00
#SBATCH -a 0-7
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v49}
NSHARD=${SLURM_ARRAY_TASK_COUNT:-8}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5). 980495 가 네 번째로 그렇게 죽었다.

# ── 관문 ⓪ **코드가 올라왔나** (§7-3) ───────────────────────────────────────
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS
print('  차수 %s · RAW %d · FINE %d · WIDE %d'
      % (tuple(VOLT_ORDERS), RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS))
ok = (tuple(VOLT_ORDERS) == (1, 3, 5, 7, 9, 11, 13, 15)
      and (RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS) == (49, 61, 51))
sys.exit(0 if ok else '코드를 안 올렸다 — 14.331 이전 배치다')
"

# `--harmonic-z` 말고는 v32 와 **글자까지 같아야 한다** (`cache_v32h_par` 에서 그대로 베꼈다).
COMMON="--windows $WINDOWS --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95"

echo "=== 관문 ① Z 표 배선 ==="
$PY -X utf8 src/run_gate_zharm.py
echo "=== 관문 ② 굽기가 타는 그 경로 (traincache._init) ==="
$PY -X utf8 src/run_gate_hzwire.py

# ── 관문 ③ **전압 블록 말고는 옛 캐시와 비트 동일** ─────────────────────────
#   옛 관문은 `train60_v32` 와 **통째로** 비트 동일을 요구했는데 배치가 바뀌어 못 쓴다.
#   대신 *"안 건드린 채널은 한 바이트도 안 달라졌다"* 를 증명한다 — 이 재굽기가
#   **손잡이 하나짜리**임을 보증하는 자리다. 새 차수는 **끼워 넣는** 것이라 색인이 밀린다:
#     세밀  옛 45..50 = V_re(1..11) -> 새 45..50 (같다) · 새 51,52 = V_re(13,15) **새 것**
#           옛 51..56 = V_im(1..11) -> 새 53..58          · 새 59,60 = V_im(13,15) **새 것**
#     광역  옛 35..40 -> 새 35..40 · 옛 41..46 -> 새 43..48
#: 14.332b — ⚠ **짝 캐시를 고쳤다.** 985942 가 여덟 토막 전부 여기서 죽었다 (2분 39초):
#      FileNotFoundError: 'cache/train60_v32/y_power.npy'
#   `cache_v32h_par` 에서 글자 그대로 베낀 경로인데, 그 캐시는 14.62 물건이고 **지금 없다**.
#   베낀 자리의 **전제가 바뀐 것**을 안 봤다.
#   ⚠ 짝을 `v32h` 로 바꾸면 **재현도 `--harmonic-z` 로 구워야 한다** (v32h 는 harmonic_z=True).
#     결과적으로 관문이 **실제로 굽는 그 배선**을 재게 되어 옛 관문보다 세다.
#: ⚠⚠ 14.390 — **이 짝 캐시는 이제 없다.** 쿼터가 꽉 차서(110/110G) 지웠다
#:   (fine=57 배치라 14.331 이후 체크포인트가 원리상 못 쓴다). 이 파일을 다시
#:   돌리려면 관문 ③ 을 `cache_v50_par.sbatch` 의 관문 ④ 처럼 **두 판을 여기서
#:   짓는 방식**으로 바꿔라 — 985942 가 사라진 짝으로 여덟 토막을 통째로 죽였다.
PAIR="cache/train60_v32h"
[ -f "$PAIR/y_power.npy" ] || { echo "짝 캐시 $PAIR 가 없다 — 관문 ③을 못 세운다"; exit 1; }
REPRO="$HOME/NILM_AI/cache/_repro_${SLURM_ARRAY_TASK_ID}"
rm -rf "$REPRO"
echo "=== 관문 ③ 전압 블록 외 v32h 와 비트 동일 (청크 0) ==="
$PY -X utf8 -m src.run_build_traincache --out "$REPRO" --shard "0/1200" \
    $COMMON --workers "${SLURM_CPUS_PER_TASK}" --harmonic-z > /dev/null
$PY -X utf8 -c "
import numpy as np, sys
bad = []
def cmp(nm, A, B):
    same = np.array_equal(np.asarray(A), np.asarray(B))
    print('    %-22s %s' % (nm, '비트 동일' if same else '**다르다**'))
    if not same: bad.append(nm)
for nm in ('y_power', 'y_on', 'y_state', 'obs_harm'):
    a = np.asarray(np.load('$REPRO/%s.npy' % nm, mmap_mode='r'))
    b = np.asarray(np.load('$PAIR/%s.npy' % nm, mmap_mode='r'))[:len(a)]
    cmp(nm, a, b)
f = np.asarray(np.load('$REPRO/fine.npy', mmap_mode='r'))
g = np.asarray(np.load('$PAIR/fine.npy', mmap_mode='r'))[:len(f)]
w = np.asarray(np.load('$REPRO/wide.npy', mmap_mode='r'))
x = np.asarray(np.load('$PAIR/wide.npy', mmap_mode='r'))[:len(w)]
cmp('fine 비전압 0..44', f[:, :45],    g[:, :45])
cmp('fine V_re h1..11',  f[:, 45:51],  g[:, 45:51])
cmp('fine V_im h1..11',  f[:, 53:59],  g[:, 51:57])
cmp('wide 비전압 0..34', w[:, :35],    x[:, :35])
cmp('wide V_re h1..11',  w[:, 35:41],  x[:, 35:41])
cmp('wide V_im h1..11',  w[:, 43:49],  x[:, 41:47])
if bad: sys.exit('전압 블록 말고도 v32h 와 다르다: %s — 손잡이가 하나가 아니다' % bad)
print('  관문 ③ 통과 — 새 차수(h13·h15) 말고는 한 바이트도 안 달라졌다')
"
rm -rf "$REPRO"

# ── 굽기 ────────────────────────────────────────────────────────────────────
time $PY -X utf8 -m src.run_build_traincache \
    --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}" \
    $COMMON --workers "${SLURM_CPUS_PER_TASK}" --harmonic-z

du -sh "$OUT"
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
print('  토막 %s · 창 %d · volt_orders=%r · fine_shape=%r · wide_shape=%r'
      % (d['shard'], d['n_windows'], d.get('volt_orders'), d.get('fine_shape'),
         d.get('wide_shape')))
if not d.get('harmonic_z'): sys.exit('meta 에 harmonic_z 가 안 적혔다')
if d.get('volt_orders') != [1, 3, 5, 7, 9, 11, 13, 15]:
    sys.exit('meta 의 volt_orders 가 %r 다 — 옛 코드로 구웠다' % d.get('volt_orders'))
if d.get('vtex_seg_s'): sys.exit('vtex_seg_s 가 실렸다 — 이 팔은 안 넣는다')
print('  ⑤ 통과 — 8차수로 구워졌고 seg_s 는 안 실렸다')
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
echo "여덟 토막이 다 끝나면 로그인 노드에서 (15초):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN} cache/train60_${GEN}_s{0..7} --rm-shards"
echo "⚠ **홀드아웃도 같은 배치로 다시 구워야 한다** — \`processed_data/holdout60_v32h\` 는 못 쓴다."
