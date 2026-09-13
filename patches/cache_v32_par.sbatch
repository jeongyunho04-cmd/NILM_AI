#!/bin/bash
# 14.13 — 30만창 v32 캐시를 **노드 둘로 나눠** 굽는다. 배열 작업 + 이어붙이기.
#
# 왜 나누나 (사용자 제안):
#   979865 는 GPU 노드(gpu6 n037)에서 40코어로 구웠다 — 85 win/s · 59분.
#   cpu2 는 256코어 노드고 64코어로 213 win/s 가 나온다 (963826, 30만창 23분 30초).
#   CPU 가 다르다 — GPU 노드 안에서 코어만 늘려봐야 1.2배다 (14.9).
#   둘로 나누면 **노드당 15만창 x 64코어 ≈ 12분** 이다.
#
# 왜 나눠도 되나 (14.13 항등 검정):
#   `build_cache` 의 `_chunk` 가 **청크 번호**로 시드하고(`chunk_seed`) `imap` 이 순서를
#   지킨다. 그래서 청크 목록을 토막내 다른 노드에서 구워도 같은 창이 나오고, 번호순으로
#   이어붙이면 통짜와 **비트 동일**하다. 500창 2분할로 11개 배열 sha256 이 전부 같았다.
#   ⚠ `--windows`·`--seed` 는 **전체 기준**을 양쪽 다 똑같이 줘야 한다. 번호가 밀린다.
#
# ⚠ 쿼터 (`mmlsquota` 로만 보인다 — `df` 는 답을 못 한다):
#   AIP 42G 를 지워 79.5G -> 37.5G 가 됐다. 캐시 23G + 토막 23G = 46G 가 한때 같이 있으므로
#   피크가 ~83G 다. 소프트 100G 아래지만 **이어붙인 뒤 토막을 지워야** 다음 판이 들어간다.
#   토막 삭제는 이 스크립트가 **안 한다** — 사람이 확인하고 지운다.
#SBATCH -J nilm_v32par
#SBATCH -p cpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64           # cpu2_limit 의 노드당 상한 (HPC_RULES §2)
#SBATCH --mem=100G      # 실측: 64코어 굽기 MaxRSS 113GB 가 100G 안에서 완료 (963826)
#SBATCH -t 00:45:00     # 실측 15만창 ≈ 12분. 1.5배 + 여유
#SBATCH -a 0-1          # ← **배열 작업**. 토막 둘이 노드 둘에서 동시에 돈다
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v32}
NSHARD=2
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문: 코드가 올라왔나 · 자리가 있나 ────────────────────────────────────
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--shard" || {
  echo "run_build_traincache 에 --shard 가 없다 — 코드를 안 올렸다"; exit 1; }
test -f src/run_merge_traincache.py || { echo "run_merge_traincache.py 가 없다"; exit 1; }
USED=$(/usr/lpp/mmfs/bin/mmlsquota --block-size auto 2>/dev/null | awk '/gpfs/{print $3}' | tr -d 'G')
echo "쿼터 사용 ${USED:-?}G / 소프트 100G — 이 굽기가 토막당 ~12G 를 쓴다"

# ── 굽기: 전체 기준값은 **양쪽 같게**, 토막만 다르게 ───────────────────────
#   `cache_v32.sbatch` 와 같은 값 (= genopts.V32 + 창 층 둘)
time $PY -X utf8 -m src.run_build_traincache \
    --out "$OUT" \
    --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}" \
    --windows "$WINDOWS" --window-cycles 3600 --seed 0 \
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
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
echo "두 토막이 다 끝나면 이어붙여라 (로그인 노드에서 1~2분):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN} \\"
echo "      cache/train60_${GEN}_s0 cache/train60_${GEN}_s1"
echo "  그 뒤 토막 삭제는 **사람이 확인하고** — rm -rf cache/train60_${GEN}_s[01]"
