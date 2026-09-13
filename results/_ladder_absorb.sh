#!/usr/bin/env bash
# 12.150 절제 사다리 — 흡수 기제 셋을 하나씩 얹는다 (규칙 3).
#   A 현행               cos, 보정 없음, 무제한   (= gc_cap10/pc_cap10, 이미 있다)
#   B + Norton 보정      cos
#   C + NNLS 배분        Norton
#   D + 증거 제한        Norton + NNLS
set -u
ZI="results/adapt_zi_s0.pt results/adapt_zi_s1.pt results/adapt_zi_s2.pt"
C="--postproc on --resmatch 0.02 --rm-snap --squelch 0.1 --absorb 1.0"
N="--absorb-norton results/norton_coef.npz"

run () {   # $1=name  $2..=extra
  local n="$1"; shift
  echo "### $n  ($*)"
  python -X utf8 -m src.run_gate_check  --ckpt $ZI $C "$@" \
      --out "results/gc_$n.json" > "results/_gc_$n.log" 2>&1 ; echo "    gc exit=$?"
  python -X utf8 -m src.run_power_check --ckpt $ZI $C "$@" \
      --out "results/pc_$n.json" > "results/_pc_$n.log" 2>&1 ; echo "    pc exit=$?"
}

run absB $N
run absC $N --absorb-mode nnls
run absD $N --absorb-mode nnls --absorb-limit
# 귀속용 — 제한만 (보정·NNLS 없이)
run absL --absorb-limit
echo "### done"
