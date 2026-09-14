#!/bin/bash
# 14.33 ⓑ — **v32t 캐시**(전압 텍스처 표집 간격 60 -> 20초)를 노드 둘로 나눠 굽는다.
#
# 왜 (14.12): 생성기는 표류를 **이미 만들고 있었다**. 모자란 것은 텍스처 표집 간격이었다 —
#   텍스처가 `step_s` 구간의 **중앙값**이라 그보다 짧은 전압 변동이 뭉개진다.
#       step_s   텍스처   h3~h15 산포 / 원시
#         60      280        0.73        <- 지금 기본값
#         20      819        **0.95**
#   그리고 표류비가 0.75 -> **0.80** 이 되는데 그것이 **정답**이다:
#   14.11 이 표류 **분산의 65.4%** 가 회로로 계산된다고 쟀으므로 진폭비 √0.654 = **0.809**.
#   ⇒ 전압 쪽에서 더 넣을 것이 없다는 뜻이고, `sibling_rotate` 를 **그 위에 더하면 안 된다**
#     (0.75 위에 1.0 을 더해 1.25배가 된 것이 `smps_dev1` 의 12~42배였다).
#
# ⚠⚠ **배선이 없었다.** `genopts.build_synthesizer` 는 `vtex_step_s` 를 걸지만
#   `traincache.build_cache` 는 `set_default_vtail` 만 부르고 있었다 — 창 캐시 경로에는
#   이 손잡이가 닿은 적이 없다 (14.33 ⓑ 에서 고쳤다). 두 입구가 갈렸던 또 한 자리다.
#   풀이 `spawn` 이라 **워커마다** `set_default_step_s` 를 다시 건다. 관문이 그것을 찍는다.
#
# 왜 나눠도 되나 (14.13 항등 검정): `_chunk` 가 **청크 번호**로 시드하고 `imap` 이 순서를
#   지킨다. 토막내 다른 노드에서 구워 번호순으로 이어붙이면 통짜와 **비트 동일**이다.
#   ⚠ `--windows`·`--seed` 는 **전체 기준**을 양쪽 다 똑같이 줘야 한다.
#
# ⚠ 쿼터: 토막당 ~12GB. 이어붙이면 한때 토막 23G + 통짜 23G 가 같이 있다.
#   **이어붙인 뒤 토막을 사람이 확인하고 지운다** — 이 스크립트는 안 지운다.
# ⚠ 텍스처가 2.9배라 `SmpsCircuit` 캐시 키 공간이 커진다. 굽기가 느려질 수 있어
#   시간을 45분 -> 70분으로 늘렸다 (14.12 ④ 의 256창 냉캐시 시험은 차이가 없었지만
#   30만창에서 적중률이 어떻게 되는지는 실제로 구워 봐야 안다고 적혀 있다).
#
#SBATCH -J nilm_v32t
# ⚠⚠ **상한에 딱 붙여 달라고 하지 마라** (2026-09-14, 982938 이 23시간 섰다).
#   `-p cpu2 -c 64` 는 `cpu2_limit` 의 노드당 상한 **그대로**라, 한 노드에 64코어가
#   통째로 비어야 앉는다. 그날 cpu2 는 전체 116코어가 놀고 있었는데 **한 노드 최대
#   구멍이 56** 이라 어디에도 못 앉았고, Slurm 이 n115 를 **다음날 13:09 로 예약**해
#   버렸다. 코어를 줄이고 토막을 늘리면 그 구멍들에 나눠 들어간다.
#   ⚠ 코어 수는 **캐시 내용을 안 바꾼다** — `_chunk` 가 청크 번호로 시드하고 `imap` 이
#     순서를 지킨다 (14.13). `--workers` 는 병렬도일 뿐이다.
#SBATCH -p cpu1,gpu6,cpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24           # 상한(cpu2 64 · gpu6 40 · cpu1 48)보다 **한참 아래**로
#SBATCH --mem=100G      # 실측 MaxRSS: 64워커 113GB · 40워커 54GB. 24워커면 넉넉하다
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). 짧을수록 backfill 이 잘 문다 —
#   `-t 01:15` 로 3.2배를 불렀다가 시드 1·2 가 한 시간을 섰다 (983243, 2026-09-14).
#   ⚠ `squeue --start`·`sbatch --test-only` 는 **backfill 을 모델링하지 않는다** —
#     모양을 바꿔도 같은 날짜가 나온다. 그 값으로 '자리가 없다' 고 읽지 마라.
#SBATCH -t 00:40:00     # 실측 25분(983103)의 1.5배     # 토막당 7.5만창. 느린 노드(1.95 win/s/코어)에서도 ~27분
#SBATCH -a 0-3          # ← **배열 작업**. NSHARD 는 아래에서 이 크기로부터 나온다
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v32t}
VTEX=${VTEX:-20.0}
# ⚠⚠ **토막 수를 손으로 적지 마라.** 배열 크기와 어긋나면 조용히 절반만 구워진다
#   (`-a 0-1` 인데 NSHARD=4 면 청크의 절반이 빠지고 아무도 안 죽는다).
NSHARD=${SLURM_ARRAY_TASK_COUNT:-2}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문: 코드가 올라왔나 · 자리가 있나 ────────────────────────────────────
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--shard" || {
  echo "run_build_traincache 에 --shard 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--vtex-step-s" || {
  echo "run_build_traincache 에 --vtex-step-s 가 없다 — 배선을 안 올렸다"; exit 1; }
# 14.12 의 관문: 간격을 줄이면 텍스처가 **400개를 넘어야** 한다. 안 넘으면 안 걸린 것이다.
$PY -X utf8 -c "
import sys; sys.path.insert(0, '.')
from src import env_guard
from src.synthesis import vtexture
n0 = len(vtexture.default_library().textures)
vtexture.set_default_step_s(float('$VTEX'))
n1 = len(vtexture.default_library().textures)
print('관문: 텍스처 %d(기본 60초) -> %d(%ss)' % (n0, n1, '$VTEX'))
if n1 <= 400: sys.exit('텍스처가 %d 뿐이다 — 간격이 안 걸렸다 (14.12 ④)' % n1)
if n1 <= n0: sys.exit('간격을 줄였는데 텍스처가 안 늘었다')
"
test -f src/run_merge_traincache.py || { echo "run_merge_traincache.py 가 없다"; exit 1; }
# ⚠⚠ **여기에 `mmlsquota` 를 넣지 마라** (HPC_RULES §7-5). 980495 가 **네 번째**로 그렇게
#   죽었다 (앞의 셋: 969002 · 969137 · 969590). `set -euo pipefail` 아래에서 계산 노드의
#   GPFS 데몬이 무응답이면 그 대입문 하나가 **종료코드 50** 으로 작업을 죽인다 —
#   `2>/dev/null` 도 `${USED:-?}` 도 소용없다. 죽는 것은 대입문 자체다.
#   **쿼터는 제출 전에 로그인 노드에서 본다.** 이 굽기는 토막당 약 12GB 를 쓴다.

# ── 굽기: 전체 기준값은 **양쪽 같게**, 토막만 다르게 ───────────────────────
#   `cache_v32.sbatch` 와 같은 값 (= genopts.V32 + 창 층 둘)
time $PY -X utf8 -m src.run_build_traincache \
    --out "$OUT" \
    --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}" \
    --windows "$WINDOWS" --window-cycles 3600 --seed 0     --vtex-step-s "$VTEX" \
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

du -sh "$OUT"
$PY -X utf8 -c "
import json
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
print('  토막 %s · 청크 %s · 창 %d (전체 기준 %d)'
      % (d['shard'], d['shard_chunks'], d['n_windows'], d['n_windows_total']))
assert float(d.get('vtex_step_s', 0)) == float('$VTEX'),     'meta 의 vtex_step_s 가 %r 이다 — 안 걸렸다' % d.get('vtex_step_s')
print('  관문: meta vtex_step_s = %s ✅' % d['vtex_step_s'])
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do SHARDS="$SHARDS cache/train60_${GEN}_s${i}"; done
echo "토막 ${NSHARD}개가 다 끝나면 이어붙여라 (로그인 노드에서 1~2분):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN}${SHARDS}"
echo "  그 뒤 토막 삭제는 **사람이 확인하고** — rm -rf cache/train60_${GEN}_s[0-9]" 
