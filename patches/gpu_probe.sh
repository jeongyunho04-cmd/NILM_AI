#!/bin/bash
# 도는 작업의 GPU·CPU 점유를 잰다 (HPC_RULES §5 의 절차).
#
#   bash patches/gpu_probe.sh 972758 972759
#
# ⚠ 컴퓨트 노드의 nvidia-smi 는 구버전이라 `--query-gpu` 를 모른다. 평문을 grep 한다.
# ⚠ 학습이 **시작된 뒤에** 재야 한다. 캐시 굽는 동안은 GPU 가 0% 다.
set -uo pipefail
cd "$HOME/NILM_AI"

JOBS=("$@")
[ ${#JOBS[@]} -gt 0 ] || { echo "작업 번호를 달라"; exit 1; }

# ── ① 학습이 붙을 때까지 기다린다 ────────────────────────────────────────────
echo "[대기] 학습 시작을 기다린다 (캐시 굽는 중이면 GPU 가 0% 라 재도 의미가 없다)"
for i in $(seq 1 60); do
  N=0
  for j in "${JOBS[@]}"; do
    F=$(ls -t *_"$j".out 2>/dev/null | head -1)
    [ -n "$F" ] && grep -q "가림 유지" "$F" 2>/dev/null && N=$((N+1))
  done
  [ "$N" -eq ${#JOBS[@]} ] && { echo "[대기] 모두 시작됨 ($(date '+%H:%M:%S'))"; break; }
  sleep 15
done

echo "[예열] 120초 — 첫 배치의 워커 기동이 섞이지 않게"
sleep 120

# ── ② 표본 ───────────────────────────────────────────────────────────────────
for j in "${JOBS[@]}"; do
  F=$(ls -t *_"$j".out 2>/dev/null | head -1)
  echo ""
  echo "===== $j  ($F)"
  grep -m1 "^노드 " "$F" 2>/dev/null
  for k in 1 2 3 4 5; do
    srun --jobid="$j" --overlap -n1 --cpus-per-task=1 bash -c '
      L=$(uptime | sed "s/.*average[: ]*//")
      echo "  [$(date +%H:%M:%S)] $(hostname -s) load $L"
      nvidia-smi | grep -E "Default|W /|MiB /" | sed "s/^/    /"
    ' 2>/dev/null
    sleep 20
  done
done

# ── ③ 처리량 (로그에서) ──────────────────────────────────────────────────────
echo ""
echo "===== 처리량 — 견줌: 972281 은 워커 18 에서 판당 4,180 창/초 · 413초/epoch"
for f in results/seq_v36*_s0.log; do
  [ -f "$f" ] || continue
  echo "  --- $f"
  grep -E "창/초|epoch|초\)" "$f" 2>/dev/null | tail -4 | sed 's/^/    /'
done
echo ""
echo "끝 $(date '+%F %H:%M:%S')"
