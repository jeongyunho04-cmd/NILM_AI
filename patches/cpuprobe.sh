#!/bin/bash
echo "노드 $(hostname -s) · 전체코어 $(nproc --all)"
ps -u "$USER" -o pcpu=,comm= > /tmp/ps_$$.txt
TOT=$(awk '{s+=$1} END {printf "%.1f", s/100}' /tmp/ps_$$.txt)
MAIN=$(awk '$2=="python"{s+=$1} END {printf "%.1f", s/100}' /tmp/ps_$$.txt)
WRK=$(awk '$2=="pt_data_worker"{s+=$1} END {printf "%.1f", s/100}' /tmp/ps_$$.txt)
NW=$(grep -c pt_data_worker /tmp/ps_$$.txt)
echo "  우리 CPU 합계 ${TOT}코어  (주 프로세스 ${MAIN} · 워커 ${WRK} · 워커 프로세스 ${NW}개)"
awk '$2=="pt_data_worker"{print $1}' /tmp/ps_$$.txt | sort -n | awk '{a[NR]=$1} END {printf "  워커 점유 최소 %.0f%% 중앙 %.0f%% 최대 %.0f%%\n", a[1], a[int(NR/2)], a[NR]}'
rm -f /tmp/ps_$$.txt
