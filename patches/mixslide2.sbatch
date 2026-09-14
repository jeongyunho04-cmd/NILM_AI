#!/bin/bash
# 14.41 ⑤ — **상태 혼합 붕괴의 13배 차이가 '타깃 이후 전환' 빈도로 설명되는가.**
#
# 14.41 이 확정: 세밀 창 600 중 **360(6초)이 타깃보다 뒤**이고, 그 뒤에 켜지는 반파 기기가
#   현재 상태를 뒤집는다 (미래를 지우면 mix 0.137 -> 0.939, 과거를 미래로 덮으면 0.001).
# 14.40 ⑤: 오븐 붕괴가 **실측 8.0% 대 합성 0.5%** 로 13배다.
#
# **실측 기준선은 이미 쟀다** (같은 도구 `run_diag_mixslide --real`, 16,496창):
#     '타깃 이후 |I2| 상승 > 0.5' 인 창 = **1.5%**
#     오븐  전체 8.0%  │ 전환 있는 창 **23.7%**(118창)  │ 전환 없는 창 **7.4%**(2869창)
#     드라이 전체 23.3% │ 전환 있는 창 **41.2%**(102창)  │ 전환 없는 창 **22.3%**
#
# **반증 가능한 예측** (돌리기 전에 적는다):
#   ① 빈도로 설명되면: 합성의 '전환 있는 창' 비율이 실측 1.5% 보다 **훨씬 낮고**,
#      그 창 안에서의 붕괴율은 실측 23.7% 와 **비슷**해야 한다 -> **생성기를 고친다**.
#   ② 빈도가 비슷한데 붕괴율만 낮으면(예: 전환 있는 창에서도 2%) 빈도가 아니다 ->
#      실측의 증거가 더 헷갈리거나(반파 기기가 실제로 더 자주/크게 겹친다) **구조**다.
#   ③ ⚠ 합성의 '전환 없는 창' 붕괴율이 실측 7.4% 보다 훨씬 낮으면(예상: ~0.5%),
#      **전환으로 설명 안 되는 몫이 실측에만 있다**는 뜻이고 그것이 진짜 남는 숙제다.
#   ④ ⚠ 대리자 검증 줄을 봐라 — 물리 대리자(`P>=0.9·V²/R`)가 참 `y_state` 통전 창을
#      몇 % 나 잡는지 합성에서만 찍는다. 80% 아래면 **두 자가 다른 것을 재고 있다.**
#
# ⚠ 문턱 0.5 는 실측 p98~99 자리다. 0.25 / 1.0 도 같이 찍어 **문턱 의존성**을 본다.
#
#SBATCH -J nilm_mixs2
#SBATCH -p gpu3,gpu4,gpu5,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
# ⚠⚠ `-t` 는 **실측의 1.5배**다 (HPC_RULES §2). 짧을수록 backfill 이 잘 문다 —
#   `-t 01:15` 로 3.2배를 불렀다가 시드 1·2 가 한 시간을 섰다 (983243, 2026-09-14).
#   ⚠ `squeue --start`·`sbatch --test-only` 는 **backfill 을 모델링하지 않는다** —
#     모양을 바꿔도 같은 날짜가 나온다. 그 값으로 '자리가 없다' 고 읽지 마라.
#SBATCH -t 00:15:00     # 실측 6분(983163)
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
test -d processed_data/holdout60_v32 || { echo "홀드아웃이 없다"; exit 1; }
$PY -X utf8 -m src.run_diag_mixslide --help | grep -q -- "--onset" || {
  echo "--onset 이 없다 — 코드를 안 올렸다"; exit 1; }
for THR in 0.25 0.5 1.0; do
  echo; echo "################ 문턱 $THR ################"
  $PY -X utf8 -m src.run_diag_mixslide --onset "$THR" \
      --holdout processed_data/holdout60_v32 \
      --ckpt results/cnn_vncls_res_s0.pt results/cnn_vncls_res_s1.pt \
             results/cnn_vncls_res_s2.pt
done
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (HPC_RULES §7-5).
echo "끝 $(date '+%F %H:%M:%S')"
