#!/bin/bash
# 14.97 — **hz + 창 안 텍스처 변동(seg_s=3)** 캐시. v32h 를 대체하는 것이 아니라 **한 판 더**.
#
# 왜 seg_s 인가 (14.50/14.51, 그리고 오늘 14.92~14.94):
#   `Texture.rel` 은 창 하나에 **정적으로 한 번** 얹힌다 -> 창-안 `V_h/|V_1|` 변동이 **정확히 0**.
#   합성이 실측의 13~35% 밖에 못 낸 것은 전부 `-Z*I` 항이 낸 것이고 **텍스처가 소유한 축이
#   통째로 비어 있었다**. 14.50 이 표집 간격(step_s 60->20)만 줄여 봤지만 표류비가 안 움직였다 —
#   고칠 곳은 **개수가 아니라 적용**이었다.
#
#   그리고 세밀 창이 창의 **마지막 10초**에 붙어 있어서, 토막 경계가 3000사이클(50초)에
#   정확히 떨어지는 seg_s=10 까지는 **세밀이 텍스처를 한 장만 본다**:
#       seg_s   토막   세밀이 보는 텍스처   광역 회수율
#         10      6          1장            78.2%
#          5     12          2장          **86.7%**
#        **3     20        **4장**          (5에서 이미 포화)
#   14.84 가 잰 병이 **세밀 쪽 자리 의존성**(자리 D 에서 L_harm 부호가 반대)이므로
#   세밀이 보는 장수를 늘리는 쪽으로 간다.
#
# ⚠ `vtex_seg_s` 는 `vtex_step_s` 와 **같아야 한다** — 텍스처는 그 녹화의 step_s 구간
#   중앙값이라, 20초 중앙값을 10초씩 틀면 변화를 2배로 빨리 감는 것이 된다. genopts.check() 가 막는다.
#
# 병렬 (14.96 에서 한도를 확인했다):
#   QOS node10_cpu640_gpu12  ->  **노드 10 · 코어 640 · 동시작업 10**  ("3" 제한은 없었다)
#   ⚠⚠ 14.96 이 **재서** 정한 값이다 (짐작이 두 번 틀렸다):
#     탐침 (고정 시작부하 115초를 뺀 한계비용, 250창 x 24코어)
#        seg_s=0   토막  1 ·  0.96 core-s/window
#        seg_s=10  토막  6 ·  9.79 core-s/window   <- seg0 의 **10.2배**
#        seg_s=5   토막 12 · 19.4  core-s/window   <- 외삽 19.6 과 일치 (모델 확인)
#        seg_s=3   토막 20 · **32.6** (외삽)
#     토막 수에 **거의 정비례**한다 — 문서가 이미 적어 뒀다: "토막마다 회로를 다시
#     부르는데 texture_delta/coupling_delta 캐시 키에 v1 이 있어 창 사이 적중이 없다".
#
#   `-c` 는 **지금 잡히는 노드 수**로 정했다 (-c 40 이 단맛 자리다):
#        -c 40 -> 13노드 · -c 48 -> 3노드 · -c 56 -> 1노드 · **-c 64 -> 0노드**
#     그래서 cpu2 큰 노드(256코어) 안은 못 쓴다 — 유휴가 노드당 8~28 뿐이다.
#
#   20토막 x 40코어, 동시 10 (QOS node=10):  토막당 15,000창 -> **3.4시간/토막** · 총 6.8시간
#   10토막으로 하면 총 시간은 같은데 한 토막이 6.8시간이라, 죽으면 손실이 두 배다.
#
#SBATCH -J nilm_hs3
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --mem=160G
#SBATCH -t 06:00:00   # 14.96 실측 3.4시간 x 1.76 (HPC_RULES 2 는 1.5 이상)
#SBATCH -a 0-19%10
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v32hs3}
NSHARD=${SLURM_ARRAY_TASK_COUNT:-20}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5). 980495 가 네 번째로 그렇게 죽었다.

# `--harmonic-z` 말고는 v32 와 **글자까지 같아야 한다.** 한 군데에 적고 둘 다 여기서 쓴다.
COMMON="--windows $WINDOWS --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95"

# 이 판의 **처치** — 한 군데에만 적는다
TREAT="--harmonic-z --vtex-step-s 3 --vtex-seg-s 3"
echo "  처치: $TREAT"
case "$TREAT" in *"--vtex-step-s 3"*"--vtex-seg-s 3"*) ;; *)
  echo "step_s 와 seg_s 가 안 맞는다 (genopts.check 가 요구한다)"; exit 1;; esac
case "$COMMON" in *--vtex*|*--harmonic-z*) echo "COMMON 에 처치가 섞였다"; exit 1;; esac

echo "=== 관문 ① Z 표 배선 (네 소비처 + 흘러나온 메서드) ==="
$PY -X utf8 src/run_gate_zharm.py
echo "=== 관문 ② **굽기가 타는 그 경로** (traincache._init) ==="
$PY -X utf8 src/run_gate_hzwire.py
echo "=== 관문 ①-b 캐시 모양 (14.93) ==="
$PY -X utf8 src/run_gate_cacheshape.py

# ── 관문 ③ 표를 끄면 `train60_v32` 를 **비트 그대로** 되만든다 ────────────────
#   함수 수준이 아니라 **자료 수준**의 항등 검정이다. 14.59 의 `zline` 이동, 14.51 의 토막
#   배선, 14.62 의 표 배선 — 그동안 합성 사슬을 여러 번 건드렸다. 그 전부가 `harmonic_z=끔`
#   에서 옛 캐시와 한 바이트도 안 다름을 여기서 증명한다. 청크 0(250창)만 구워 1~2분이다.
#   ⚠ 이게 깨지면 v32h 대 v32 비교는 **손잡이 하나짜리가 아니다** — 굽지 마라.
#   ⚠⚠ 14.98 — `cache/train60_v32` 본체(23G)는 쿼터 때문에 지웠다. 이 관문이 읽는 것은
#      코드상 **앞 250행뿐**이라(`b[:n]`), 앞 512행 슬라이스 `cache/_v32_head512`(39MB)로
#      바꿨다. 지우기 **전에** 여섯 배열이 앞 250행 비트 동일임을 확인했다.
REPRO="$HOME/NILM_AI/cache/_repro_${SLURM_ARRAY_TASK_ID}"
rm -rf "$REPRO"
echo "=== 관문 ③ 표 끔 = train60_v32 비트 동일 (청크 0) ==="
$PY -X utf8 -m src.run_build_traincache --out "$REPRO" --shard "0/1200"     $COMMON --workers "${SLURM_CPUS_PER_TASK}" > /dev/null
$PY -X utf8 -c "
import numpy as np, sys
bad = []
for nm in ('fine', 'wide', 'y_power', 'y_on', 'y_state', 'obs_harm'):
    a = np.load('$REPRO/%s.npy' % nm, mmap_mode='r')
    b = np.load('cache/_v32_head512/%s.npy' % nm, mmap_mode='r')
    n = len(a)
    same = np.array_equal(np.asarray(a), np.asarray(b[:n]))
    print('    %-9s %6d창  %s' % (nm, n, '비트 동일 ✅' if same else '**다르다** ❌'))
    if not same: bad.append(nm)
if bad: sys.exit('❌ 표를 꺼도 v32 와 다르다: %s — 손잡이가 하나가 아니다' % bad)
print('  관문 ③ ✅ 합성 사슬이 v32 이후로 안 움직였다')
"
rm -rf "$REPRO"

# ── 굽기 ────────────────────────────────────────────────────────────────────
time $PY -X utf8 -m src.run_build_traincache     --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}"     $COMMON --workers "${SLURM_CPUS_PER_TASK}"     $TREAT

du -sh "$OUT"
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
a = json.load(open('cache/_v32_head512/meta.json', encoding='utf-8'))
print('  토막 %s · 청크 %s · 창 %d (전체 기준 %d)'
      % (d['shard'], d['shard_chunks'], d['n_windows'], d['n_windows_total']))
if not d.get('harmonic_z'): sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
if float(d.get('vtex_step_s') or 0) != 3.0 or float(d.get('vtex_seg_s') or 0) != 3.0:
    sys.exit('❌ meta 의 vtex_step_s/seg_s 가 3.0 이 아니다: %r / %r'
             % (d.get('vtex_step_s'), d.get('vtex_seg_s')))
skip = {'harmonic_z', 'shard', 'shard_chunks', 'n_windows', 'build_seconds', 'bytes',
        'merged_from', 'merged_partial', 'positive_rate', 'recipe_counts', 'built_at',
        'content_sha256'}
# ⚠ '키 없음' 과 '0/False' 는 **같은 뜻**이다 — 'train60_v32' 는 'vtex_step_s' 가 생기기
#   전에 구운 캐시라 키가 아예 없고, 새 캐시는 0.0 으로 적는다. 그대로 견주면 983992 처럼
#   자료가 멀쩡한데 관문이 운다 (자료 비트 동일은 굽기 관문 ③이 이미 증명했다).
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
diff = [k for k in set(a) | set(d) if k not in skip
        and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
        != json.dumps(nz(d.get(k)), sort_keys=True, default=str)]
skip |= {'vtex_step_s', 'vtex_seg_s'}
diff = [k for k in diff if k not in skip]
if diff: sys.exit('❌ 처치 셋 말고도 v32 와 다르다: %s' % diff)
print('  관문 ④ ✅ v32 와 **harmonic_z · vtex_step_s · vtex_seg_s 셋만** 다르다')
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do SHARDS="$SHARDS cache/train60_${GEN}_s${i}"; done
echo "네 토막이 다 끝나면 로그인 노드에서 (15초):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN}${SHARDS} --rm-shards"
