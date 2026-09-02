#!/usr/bin/env bash
# 12.149.4 — 흡수 천장을 기기별로 확장. 값 둔감성 배수 쓸기 (1.0 / 1.25 / 1.5).
# 배수 1.0 판만 기준선 팔을 같이 넣는다 (귀속용).
set -u
ZI="results/adapt_zi_s0.pt results/adapt_zi_s1.pt results/adapt_zi_s2.pt"
ALL="results/adapt_ovh.pt results/adapt_ovh_s1.pt results/adapt_ovh_s2.pt $ZI"
C="--postproc on --resmatch 0.02 --rm-snap --squelch 0.1 --absorb 1.0"

run () {   # $1=name  $2=ckpts  $3=scale
  echo "### $1  (cap-scale=$3)"
  python -X utf8 -m src.run_gate_check  --ckpt $2 $C --absorb-cap-scale "$3" \
      --out "results/gc_$1.json" > "results/_gc_$1.log" 2>&1 ; echo "    gc exit=$?"
  python -X utf8 -m src.run_power_check --ckpt $2 $C --absorb-cap-scale "$3" \
      --out "results/pc_$1.json" > "results/_pc_$1.log" 2>&1 ; echo "    pc exit=$?"
}

run cap10 "$ALL" 1.0
run cap125 "$ZI" 1.25
run cap150 "$ZI" 1.5
echo "### done"
