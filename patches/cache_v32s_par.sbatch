#!/bin/bash
# 14.51 — **v32s 캐시**: 창 하나에 그 녹화의 **연속된 텍스처 6장**을 얹는다 (`--vtex-seg-s 10`).
#
# 왜 (14.50 이 남긴 구멍):
#   `_terminal_voltage_harmonics` 는 `rel[None,:] * v_open` 이었다 — 텍스처를 창 전체에
#   **한 번 정적으로** 곱한다. 그래서 창-안 `V_h/|V_1|` 변동이 **정확히 0** 이고, 합성이 내던
#   변동은 전부 `−Z·I` 항이었다. 14.50 이 "합성 창-안 V_h 변동이 실측의 13~35%" 라고 쟀던
#   그 구멍이다. v32t(표집 간격 60->20초)가 텍스처를 280->819개로 늘리고도 표류비를 못 올린
#   까닭도 이것이다 — **자료는 있었고 배선이 정적이었다.**
#
# 자 (전부 실측):
#   ① 천장 — 실측 격리 녹화 24개·256창에서 60초 창을 n토막 내 회수되는 창-안 변동
#      (`run_diag_vtexseg.py`):   n=1 **0.0%** · n=2 35% · n=3 55% · n=6 **78%** · n=12 87%
#   ② 실측 복합 대 합성 (`run_diag_vtexreal.py`, test_1~5 94창 대 합성 120창):
#      창-안 `V_h/|V_1|` 표준편차 / 실측     seg 끔 **Re 17% · Im 12%**  ->  seg=10 **Re 71% · Im 62%**
#      ⇒ 14.50 의 13~35% 를 독립으로 재현했고, **과보정이 아니다** (중앙이 100% 아래).
#      ⚠ h11 의 Im 하나만 134% 로 넘는다. 실측 창이 94개뿐이라 그 한 줄은 표본 탓일 수 있다.
#   ③ 전류로도 넘어간다 — 연속 텍스처 6장이 만드는 창-안 SMPS 전류 변동이
#      **지금 넣고 있는 텍스처 델타 자체의 48~73%** 다 (`probe_texrun`). 2차 항이 아니라서
#      `apply_voltage_texture`·`apply_smps_coupling` 도 토막마다 다시 부른다.
#
# ⚠ **`--vtex-seg-s` 는 `--vtex-step-s` 와 같은 값이어야 한다.** 텍스처는 그 녹화의 `step_s`
#   구간 **중앙값**이라, 20초 중앙값을 10초씩 틀면 변화를 2배로 빨리 감는 것이 된다.
#   `genopts.check()` 와 `run_gate_vtexseg.py` 가 둘 다 막는다.
# ⚠ **끄면 비트 동일**이다 — `--vtex-seg-s 60`(60초 창 = 한 장)이 끔과 입력·정답 전부 같은
#   것을 `run_gate_vtexseg.py` ⑤ 가 확인한다. 난수도 한 칸 안 밀린다 (④, y_on·y_state·y_plugged).
#
# ⚠⚠ **쿼터.** 이 캐시는 23GB 다. 굽기 시작 전 로그인 노드에서 확인하라 —
#     `/usr/lpp/mmfs/bin/mmlsquota --block-size auto`
#     2026-09-14 기준 88.5G/100G 라 **자리가 없다.** `cache/train60_v32t` (23G) 가
#     지울 자리다 (14.50 이 그 판을 무효로 만들었고 v32s 가 그것을 대신한다).
#     ⚠ 이 스크립트는 **아무것도 안 지운다.** 사람이 지운다.
#
# 굽는 시간 (실측): 창당 118ms(v32t) -> **226ms**(v32s) = **1.92배**. 토막을 8개로 늘려
#   토막당 37,500창으로 줄였다 — 짧을수록 backfill 이 잘 문다.
#
#SBATCH -J nilm_v32s
#SBATCH -p cpu1,gpu6,cpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24           # 노드당 상한(cpu2 64 · gpu6 40 · cpu1 48)보다 **한참 아래**로
#SBATCH --mem=100G      # 실측 MaxRSS: 64워커 113GB · 40워커 54GB. 24워커면 넉넉하다
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). `squeue --start`·`sbatch --test-only` 는
#   backfill 을 모델링하지 않는다 — 그 값으로 '자리가 없다' 고 읽지 마라.
#SBATCH -t 00:40:00     # 토막당 37,500창 / 24코어 ≈ 24분 (v32t 25분 x 1.92 / 2)의 1.5배
#SBATCH -a 0-7          # ← **배열 작업**. NSHARD 는 아래에서 이 크기로부터 나온다
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

WINDOWS=${WINDOWS:-300000}
GEN=${GEN:-v32s}
VTEX=${VTEX:-10.0}
SEG=${SEG:-10.0}
# ⚠⚠ **토막 수를 손으로 적지 마라.** 배열 크기와 어긋나면 조용히 절반만 구워진다.
NSHARD=${SLURM_ARRAY_TASK_COUNT:-8}
OUT="$HOME/NILM_AI/cache/train60_${GEN}_s${SLURM_ARRAY_TASK_ID}"

echo "노드 $(hostname) · 토막 ${SLURM_ARRAY_TASK_ID}/${NSHARD} · 코어 ${SLURM_CPUS_PER_TASK} · $(date '+%F %H:%M:%S')"

# ── 관문: 코드가 올라왔나 · 배선이 닿았나 ───────────────────────────────────
# ⚠⚠ **함수를 재지 말고 배선을 재라** (983506 이 45분을 버렸다). `run_gate_vtexseg` 는
#   진짜 생성기로 진짜 창을 합성해 모델이 보는 입력 텐서에서 창-안 전압 변동을 잰다.
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--vtex-seg-s" || {
  echo "run_build_traincache 에 --vtex-seg-s 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_build_traincache --help | grep -q -- "--shard" || {
  echo "run_build_traincache 에 --shard 가 없다"; exit 1; }
$PY -X utf8 src/run_gate_vtexseg.py
test -f src/run_merge_traincache.py || { echo "run_merge_traincache.py 가 없다"; exit 1; }
# ⚠⚠ **여기에 `mmlsquota` 를 넣지 마라** (HPC_RULES §7-5). 980495 가 네 번째로 그렇게 죽었다.
#   쿼터는 제출 전에 로그인 노드에서 본다. 이 굽기는 토막당 약 3GB 를 쓴다.

# ── 굽기: 전체 기준값은 **토막마다 같게**, 토막만 다르게 ─────────────────────
#   `cache_v32t_par.sbatch` 와 같은 값 + `--vtex-step-s 10 --vtex-seg-s 10`
time $PY -X utf8 -m src.run_build_traincache \
    --out "$OUT" \
    --shard "${SLURM_ARRAY_TASK_ID}/${NSHARD}" \
    --windows "$WINDOWS" --window-cycles 3600 --seed 0 \
    --vtex-step-s "$VTEX" --vtex-seg-s "$SEG" \
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
assert float(d.get('vtex_step_s', 0)) == float('$VTEX'), 'meta vtex_step_s = %r' % d.get('vtex_step_s')
assert float(d.get('vtex_seg_s', 0)) == float('$SEG'),   'meta vtex_seg_s = %r' % d.get('vtex_seg_s')
print('  관문: meta vtex_step_s=%s · vtex_seg_s=%s ✅' % (d['vtex_step_s'], d['vtex_seg_s']))
"
echo "끝 $(date '+%F %H:%M:%S')"
echo
SHARDS=""
for i in $(seq 0 $((NSHARD - 1))); do SHARDS="$SHARDS cache/train60_${GEN}_s${i}"; done
echo "토막 ${NSHARD}개가 다 끝나면 이어붙여라 (로그인 노드에서 1~2분):"
echo "  \$PY -X utf8 src/run_merge_traincache.py --out cache/train60_${GEN}${SHARDS} --rm-shards"
echo "  (--rm-shards 는 바이트 수를 확인한 **뒤에** 토막을 지운다 — 쿼터가 빠듯할 때 쓴다)"
