#!/usr/bin/env bash
# 12.149 값 둔감성 + 귀속 쓸기. 복소 Z·I 3시드만 (기준선 팔은 gc_sqab 에 있다).
#   ① 스켈치 단독 (흡수 0)      -> 귀속: 어느 쪽이 일하는가
#   ② tau 0.05 / 0.1 / 0.2      -> 한 점만 좋으면 튜닝 잔향 (12.102 의 검사)
#   ③ 흡수 0.5                  -> frac 둔감성
set -u
CK="results/adapt_zi_s0.pt results/adapt_zi_s1.pt results/adapt_zi_s2.pt"
COMMON="--postproc on --resmatch 0.02 --rm-snap"

run () {   # $1=name  $2=squelch  $3=absorb
  local n="$1" sq="$2" ab="$3" sqf="" abf=""
  [ "$sq" != "0" ] && sqf="--squelch $sq"
  [ "$ab" != "0" ] && abf="--absorb $ab"
  echo "### $n  (squelch=$sq absorb=$ab)"
  python -X utf8 -m src.run_gate_check --ckpt $CK $COMMON $sqf $abf \
      --out "results/gc_$n.json"  > "results/_gc_$n.log" 2>&1
  echo "    gc exit=$?"
  python -X utf8 -m src.run_power_check --ckpt $CK $COMMON $sqf $abf \
      --out "results/pc_$n.json"  > "results/_pc_$n.log" 2>&1
  echo "    pc exit=$?"
}

run sq50_ab0    0.5   0
run sq05_ab1    0.05  1.0
run sq10_ab1    0.1   1.0
run sq20_ab1    0.2   1.0
run sq50_ab05   0.5   0.5
echo "### done"
