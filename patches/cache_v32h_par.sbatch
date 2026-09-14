#!/bin/bash
# 14.62 — **v32h 캐시**: `train60_v32` 와 `--harmonic-z` **하나만** 다르다.
#
# 무엇을 켜나 (14.59 가 실측한 것):
#   생성기는 `z_h = r + j·h·x` 를 썼다. 실측 계단 복소 최소제곱(`run_diag_zharm --snr 8`)은
#   그 **함수 형태 자체가 틀렸다**고 말한다 — h3 은 유도성(+4.67j)인데 h5 는 용량성(−2.36j)이라
#   부호가 뒤집힌다. `x_grid` 를 올리는 것으로는 절대 재현이 안 된다.
#     자리 D  h1 1.140 (R² 0.996) · h3 6.059+4.672j (**Z/Z_1 6.71**, R² 0.742)
#                                  · h5 0.263−2.364j (2.09, R² 0.580)
#     자리 E  h3 2.37 (R² 0.635)
#     h7 위는 R² 0.04~0.43 이라 **안 넣는다** — 거기는 텍스처가 나른다 (강하 기여 0.3~1.2%).
#
# 입력이 얼마나 바뀌나 (`run_gate_hzwire`, 창 6개):
#   세밀 8.02% · 광역 9.44% (신호 RMS 대비). 겨냥한 채널에만 떨어진다 —
#   c46 Re(V3) **44%** · c53 Im(V5) 8% · c52 Im(V3) 7% · c15(전류) 11% 는 결합으로 번진 몫.
#
# ⚠⚠ **2026-09-14 에 이 굽기를 던졌으면 통째로 버렸다.** `traincache._init` 에만
#   `grid_sim.harmonic_z_table` 대입이 빠져 있었다 — 텍스처는 **새 Z 로 벗기고** 단자 전압은
#   **옛 Z 로 다시 입혔다.** 안 지워지고 `(Z_새 − Z_옛)·I` 가 남는다. `run_gate_hzwire` ④ 가
#   잰 그 반쪽의 오차는 **손잡이 효과의 100%** 였다 (반쪽↔온전 0.1345 대 끔↔온전 0.1342).
#   그런데 굽기 전 관문(`run_gate_vtexseg`)은 `genopts` 로 **따로 지은** 물건을 재서 통과했다.
#   ⇒ 아래 관문 ②가 **굽기가 실제로 타는 `traincache._init`** 을 잰다.
#
# ⚠⚠ 같은 날 `coupling.py` 에서 `zline` 을 클래스 한가운데 0열에 두어 `SmpsCircuit` 이
#   거기서 끝나 `solve_terminal`·`_compute_coupling`·`stats` 셋이 통째로 사라져 있었다.
#   합성 첫 창에서 `AttributeError` 로 죽는다. `run_gate_zharm` ⑦ 이 이제 막는다.
#
# 쿼터: 이 캐시는 23G · 토막당 5.8G. **굽기 전에 로그인 노드에서 자리를 만들어라** —
#   `/usr/lpp/mmfs/bin/mmlsquota --block-size auto`. 이 스크립트는 아무것도 안 지운다.
#
# 시간 (실측 앵커): v32 meta `build_seconds 555.8` = 15만창/64코어 -> **창당 0.237 코어·초**.
#   ⚠ **이 추정이 2.35배 틀렸다** — 아래 실측을 봐라.
#
# ⚠⚠ **코어를 올리면 스케줄이 느려진다 — 처음에 `-c 48 · 4토막` 으로 던졌더니 `(Priority)`
#   로 섰다.** `-c 48` 은 cpu1(노드 48코어)에서 **노드가 통째로 비어야** 잡히는데, 그때
#   cpu1 의 노드별 여유가 36·35·35·24·18·8·8·3·2·0 이라 **48짜리가 하나도 없었다.**
#   cpu2 도 64·60 둘뿐이라 4토막을 못 채웠다.
#   ⇒ **토막을 잘게, 코어는 노드 여유 아래로.** 24코어면 cpu1 4대 + cpu2 3대 + gpu6 다수가
#     받는다. 총 144코어라 QOS 640 의 1/4 이고, 토막이 6개라 MaxJobs 10 안에도 든다
#     (굽기 8 <= MaxJobs 10). [[widen-the-partition-dont-wait-the-queue]]
# ⚠ gpu6·gpu1 은 노드당 상한이 **40** 이라 24는 안전하다.
#
# ⚠⚠ **코어당 속도가 파티션마다 다르다 — 983910 이 이것 때문에 잘릴 뻔했다.**
#   시간 앵커로 v32 meta 의 `build_seconds 555.8`(15만창/64코어 = 창당 **0.237** 코어·초)을
#   썼는데, 그 굽기는 **cpu2** 에서 돌았다. cpu1 노드에서 실측하니 **43 win/s / 24코어 =
#   창당 0.558 코어·초** 로 **2.35배** 느렸다. 7.5만창이 29분이 되어 `-t 25:00` 을 넘겼고,
#   `scontrol update TimeLimit` 은 사용자에게 **권한이 없다**(Access/permission denied) —
#   늘릴 수 없고 취소 후 재제출뿐이다.
#   ⇒ **앵커를 쓸 때 그 앵커가 어느 파티션에서 났는지 확인하라.**
#   ⇒ 지금 값: 3.75만창 / 24코어 / 43 win/s = 872초 ≈ 14.5분 + 관문 2.5분 = **17분**.
#     `-t 35:00` 은 그 2배다 (1.5배 규칙보다 넉넉하게 잡는다 — 잘리면 판 전체를 다시 굽는데
#     `-t` 를 늘려 잃는 것은 backfill 확률 조금뿐이라 **비대칭**이다).
#SBATCH -J nilm_v32h
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 00:35:00
#SBATCH -a 0-7
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v32h}
NSHARD=${SLURM_ARRAY_TASK_COUNT:-8}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5). 980495 가 네 번째로 그렇게 죽었다.

# `--harmonic-z` 말고는 v32 와 **글자까지 같아야 한다.** 한 군데에 적고 둘 다 여기서 쓴다.
COMMON="--windows $WINDOWS --window-cycles 3600 --seed 0
        --power-scale-std measured --recipe-mix steady2 --smps-focus-off-p 0.4
        --state-mix minipc_balanced --carrier-on oven --couple-ext
        --sp-curves --sp-per-texture --vtail --float-fill charger_float
        --steady-crop smps_steady --standby-jitter-cap 95"

echo "=== 관문 ① Z 표 배선 (네 소비처 + 흘러나온 메서드) ==="
$PY -X utf8 src/run_gate_zharm.py
echo "=== 관문 ② **굽기가 타는 그 경로** (traincache._init) ==="
$PY -X utf8 src/run_gate_hzwire.py

# ── 관문 ③ 표를 끄면 `train60_v32` 를 **비트 그대로** 되만든다 ────────────────
#   함수 수준이 아니라 **자료 수준**의 항등 검정이다. 14.59 의 `zline` 이동, 14.51 의 토막
#   배선, 14.62 의 표 배선 — 그동안 합성 사슬을 여러 번 건드렸다. 그 전부가 `harmonic_z=끔`
#   에서 옛 캐시와 한 바이트도 안 다름을 여기서 증명한다. 청크 0(250창)만 구워 1~2분이다.
#   ⚠ 이게 깨지면 v32h 대 v32 비교는 **손잡이 하나짜리가 아니다** — 굽지 마라.
REPRO="$HOME/NILM_AI/cache/_repro_${SLURM_ARRAY_TASK_ID}"
rm -rf "$REPRO"
echo "=== 관문 ③ 표 끔 = train60_v32 비트 동일 (청크 0) ==="
$PY -X utf8 -m src.run_build_traincache --out "$REPRO" --shard "0/1200"     $COMMON --workers "${SLURM_CPUS_PER_TASK}" > /dev/null
$PY -X utf8 -c "
import numpy as np, sys
bad = []
for nm in ('fine', 'wide', 'y_power', 'y_on', 'y_state', 'obs_harm'):
    a = np.load('$REPRO/%s.npy' % nm, mmap_mode='r')
    b = np.load('cache/train60_v32/%s.npy' % nm, mmap_mode='r')
    n = len(a)
    same = np.array_equal(np.asarray(a), np.asarray(b[:n]))
    print('    %-9s %6d창  %s' % (nm, n, '비트 동일 ✅' if same else '**다르다** ❌'))
    if not same: bad.append(nm)
if bad: sys.exit('❌ 표를 꺼도 v32 와 다르다: %s — 손잡이가 하나가 아니다' % bad)
print('  관문 ③ ✅ 합성 사슬이 v32 이후로 안 움직였다')
"
rm -rf "$REPRO"

# ── 굽기 ────────────────────────────────────────────────────────────────────
time $PY -X utf8 -m src.run_build_traincache     --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}"     $COMMON --workers "${SLURM_CPUS_PER_TASK}"     --harmonic-z

du -sh "$OUT"
$PY -X utf8 -c "
import json, sys
d = json.load(open('$OUT/meta.json', encoding='utf-8'))
a = json.load(open('cache/train60_v32/meta.json', encoding='utf-8'))
print('  토막 %s · 청크 %s · 창 %d (전체 기준 %d)'
      % (d['shard'], d['shard_chunks'], d['n_windows'], d['n_windows_total']))
if not d.get('harmonic_z'): sys.exit('❌ meta 에 harmonic_z 가 안 적혔다')
skip = {'harmonic_z', 'shard', 'shard_chunks', 'n_windows', 'build_seconds', 'bytes',
        'merged_from', 'merged_partial', 'positive_rate', 'recipe_counts', 'built_at',
        'content_sha256'}
diff = [k for k in set(a) | set(d) if k not in skip
        and json.dumps(a.get(k), sort_keys=True, default=str)
        != json.dumps(d.get(k), sort_keys=True, default=str)]
if diff: sys.exit('❌ harmonic_z 말고도 v32 와 다르다: %s' % diff)
print('  관문 ④ ✅ v32 와 **harmonic_z 하나만** 다르다')
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do SHARDS="$SHARDS cache/train60_${GEN}_s${i}"; done
echo "네 토막이 다 끝나면 로그인 노드에서 (15초):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN}${SHARDS} --rm-shards"
