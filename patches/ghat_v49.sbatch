#!/bin/bash
# 14.346 — 학습 캐시에 **`g_hat.npy`** 를 얹는다 (창마다 타깃 순간의 Ĝ_sum, mS).
#
# 조합 머리(`--comb-tau`, 14.347)가 쓸 물건이다. Ĝ 는 **입력만의 함수**라 학습 전에 한 번
# 풀어 두면 되고 **캐시를 다시 굽지 않는다** — `obs_harm` 과 세밀 45~60 이 이미 있다.
# `run_gate_gbudget_cache` 가 그 되짚기를 8/8 로 확인했다 (float16 이어도 |잔차| 0.191 mS).
#
# ⚠ `fine` 전체가 300,000 x 61 x 600 x 2B = **22GB** 다. 창 축으로 토막내 읽는다.
#   그래서 로그인 노드가 아니라 **작업으로** 돌린다 (HPC_RULES §0).
# ⚠ 타깃 순간만 푼다 — 창 최대는 **게이트 프라이어**용이고 다른 판이다 (14.335).
#SBATCH -J nilm_ghat
#SBATCH -p cpu1,gpu6,cpu2,gpu3,gpu4,gpu5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:25:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE=${CACHE:-cache/train60_v49}

echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
# ⚠⚠ 여기에 mmlsquota 를 넣지 마라 (HPC_RULES §7-5).

# 관문 ⓪ 코드가 올라왔나
$PY -X utf8 -c "
import sys
from src.model.inputs import VOLT_ORDERS, FINE_CHANNELS
from src.model import gbudget as GB
ok = (tuple(VOLT_ORDERS) == (1,3,5,7,9,11,13,15) and FINE_CHANNELS == 61
      and hasattr(GB, 'pin_total') and 'oven' in GB.FLOOR_APPS)
print('  차수 %s · FINE %d · pin_total %s · FLOOR_APPS %s'
      % (tuple(VOLT_ORDERS), FINE_CHANNELS, hasattr(GB,'pin_total'), GB.FLOOR_APPS))
sys.exit(0 if ok else '코드를 안 올렸다 (14.345 이전 판)')
"
test -d "$CACHE" || { echo "캐시가 없다: $CACHE"; exit 1; }

$PY -X utf8 -m src.run_build_ghat --cache "$CACHE"

ls -la "$CACHE/g_hat.npy"
echo "끝 $(date '+%F %H:%M:%S')"
