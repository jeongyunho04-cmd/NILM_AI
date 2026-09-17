#!/bin/bash
# 14.410 — **캐시 v52 = v49 조리법 + 자리 F 를 넣은 세그먼트 풀.** 손잡이는 **풀 하나**다.
#
# ⚠⚠ **v51 에서 뜨지 마라.** v51 은 v49 에 `--phase-jitter-map measured` 를 더한 것이고,
#   그 손잡이는 `cnn_pjit`(v50)로 **이미 기각됐다** (§58: 미니PC 실측 AUC t **−4.5**,
#   깎은 만큼 지는 완전 단조). 그리고 오늘까지의 판정(`cnn_comb`·`dz03`·`dz06`·`sigp` …)은
#   **전부 `train60_v49`** 위에서 났다 — 체크포인트가 `cache='cache/train60_v49'` 로 적고 있다.
#   v51 에서 뜨면 풀과 지터가 **동시에** 바뀌어 무엇이 이겼는지 못 가른다.
#   ⇒ 굽기 인자는 v49 와 **글자까지 같다**: `$COMMON --harmonic-z` (지터 표 안 건다).
#
# 왜 굽나 (사용자 결정):
#   시연이 **자리 F 와 같은 건물**에서 열린다. F 는 일반화 시험대가 아니라 **배포 도메인**이다.
#   그리고 지금 모델은 거기서 무너진다 (14.409, `cnn_comb` 씨앗 3개, test_7 사람 라벨):
#     게이트 AUC  미니PC 0.784 -> **0.445~0.583**(동전) · 빔 0.988 -> **0.579~0.696**
#                 충전기 0.997 -> 0.921~0.966
#     예측합/관측  D/E 0.90~1.26 -> **F 0.723~0.786** (잔차 평균 +21.0W = 절대 21.0W, 순수 편향)
#   손실 손잡이 **열**이 미니PC 를 못 움직였는데(§58~65) 자리 하나가 0.28~0.34 를 움직였다.
#
# 풀에 새로 든 넷 (세션 F1, 2026-09-17. vrms 218.1 · vh3 1.01% · vh7 1.65% · Z 0.68Ω):
#   `minipc_5`(활성화 2) · `laptop_charger_7`(1) · `beam_projector_8`(2) · `electric_kettle_5`(12)
#   ⇒ 풀이 자리 **둘(D·E) -> 셋**. 둘일 때는 '자리'와 '녹화'를 원리상 못 가른다.
#   ⓘ 사전 자체는 거의 안 움직였다 (홀수차 지문 이동: 미니PC **0.4%** · 주전자 0.1% ·
#     빔 4.8% · 충전기 14.9% · 나머지 0.0%, `harm_scale` 노름 +6.4%).
#     **이득은 지문이 아니라 합성 재료에서 와야 한다.**
#
# ⚠ `electric_kettle_5` 는 규칙 74 를 통과 못 한다 (∠I−∠V(h1) +2.23°, D/E 는 −0.12~+0.06°).
#   지문 거리 영향은 0%로 쟀지만 원인 미상이다 (`file_registry` 주석). 저항 대조 지표가
#   나빠지면 **그 파일만 빼고 다시 구워라**.
#
# ⚠⚠ 이 판의 유일한 사고는 **풀을 안 올리고 굽는 것**이다 — 그러면 v49 를 v52 이름으로 굽는다.
#   관문 ⓪ 이 그것만 본다.
#
#SBATCH -J nilm_v52
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 00:50:00
#: HPC_RULES §0 의 자체 제한은 동시 3노드였다. **2026-09-17 사용자가 풀었다**
#:   (*"3노드 제한 상관없어"*). 987048 은 `%3` 으로 던진 뒤 `scontrol update
#:   arraytaskthrottle=8` 로 풀어 8토막을 6노드(n023·n025x2·n026x2·n029·n030·n037)에
#:   동시에 올렸다. 되돌리려면 `%3` 으로 바꾸면 된다.
#SBATCH -a 0-7%8
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v52}
#: 14.390e — `--array=0` 으로 한 토막만 다시 돌리면 COUNT 가 1 이 되어
#:   토막 '0/1' = 30만 창 전부가 된다 (986491 이 그렇게 죽었다).
NSHARD=${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-8}}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5). 980495 가 네 번째로 그렇게 죽었다.

# ── 관문 ⓪ **풀에 자리 F 가 올라왔나** (이 판의 유일한 손잡이다) ──────────────
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS
from src.preprocessing.file_registry import site_of, SITE_PROFILE
from src.synthesis.segment_pool import SegmentPool
print('  차수 %s · RAW %d · FINE %d · WIDE %d'
      % (tuple(VOLT_ORDERS), RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS))
if (tuple(VOLT_ORDERS) != (1, 3, 5, 7, 9, 11, 13, 15)
        or (RAW_CHANNELS, FINE_CHANNELS, WIDE_CHANNELS) != (49, 61, 51)):
    sys.exit('코드를 안 올렸다 — 14.331 이전 배치다')
if 'F' not in SITE_PROFILE:
    sys.exit('등록부에 자리 F 가 없다 — src 를 안 올렸다')
NEW = ('minipc_5', 'laptop_charger_7', 'beam_projector_8', 'electric_kettle_5')
bad = [s for s in NEW if site_of(s) != 'F']
if bad:
    sys.exit('등록부가 %s 를 자리 F 로 안 본다' % (bad,))
pool = SegmentPool(npz_dir='processed_data/npz', time_split='train')
seen, sites = {}, {}
for app, acts in pool.appliance_activations.items():
    for a in acts:
        st = getattr(a, 'source_file', '')
        seen[st] = seen.get(st, 0) + 1
        sites.setdefault(site_of(st) or '?', set()).add(st)
print('  풀 자리 분포: ' + ' · '.join('%s %d녹화' % (k, len(v)) for k, v in sorted(sites.items())))
miss = [s for s in NEW if seen.get(s, 0) == 0]
if miss:
    sys.exit('풀에 자리 F 녹화가 없다: %s — processed_data/npz 를 안 올렸다. '
             '이대로 구우면 v49 를 v52 이름으로 굽는다' % (miss,))
print('  자리 F 활성화: ' + ' · '.join('%s %d개' % (s, seen[s]) for s in NEW))
if len(sites) < 3:
    sys.exit('자리가 %d 뿐이다 — 셋이어야 한다 (D·E·F)' % len(sites))
print('  관문 ⓪ 통과 — 풀이 **자리 셋**이다')
"

# ⚠⚠ 14.410 — **여기가 내가 한 번 날린 자리다.** 관문 ⓪ 을 갈아 끼울 때 범위를
#   `관문 ⓪ ~ 관문 ①` 으로 잡았는데 **그 사이에 `COMMON` 이 살고 있었다**.
#   987048 이 여덟 토막 전부 `line 110: COMMON: unbound variable` 로 죽었다.
#   범위를 바꾸기 전에 그 안에 무엇이 사는지 봐라 ([[find-patch-collisions-by-shape]]).
#
# `--harmonic-z` 말고는 v32 와 **글자까지 같아야 한다**. v51 과 다른 점은 **없다** —
# v51 은 여기에 `--phase-jitter-map measured` 를 굽기 줄에서 더했을 뿐이고, 이 판은 안 건다.
COMMON="--windows $WINDOWS --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95"

echo "=== 관문 ① Z 표 배선 ==="
$PY -X utf8 src/run_gate_zharm.py
echo "=== 관문 ② 굽기가 타는 그 경로 (traincache._init) ==="
$PY -X utf8 src/run_gate_hzwire.py

# ── 관문 ③ 은 **없다** (14.410에서 뗐다) ─────────────────────────────────────
#   v49 의 관문 ③ 은 짝 캐시 `train60_v32h` 와 **비트 동일**을 요구했다. 두 가지로 못 쓴다:
#     ① 그 짝 캐시가 HPC 에 없다 (14.390 에서 쿼터 때문에 지웠다. 985942 가 사라진
#        짝으로 여덟 토막을 통째로 죽인 자리이므로 **되살리지 말고 뗀다**)
#     ② 이 판은 **풀이 바뀌는 판**이다. 옛 캐시와 비트 동일하면 그것이 곧 **실패**다
#        (자리 F 가 안 들어갔다는 뜻이니까)
#   ⇒ 손잡이 확인은 **관문 ⓪** 이 맡는다 — 풀이 자리 셋인지, 새 녹화 넷이 실렸는지.

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
