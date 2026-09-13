#!/bin/bash
#SBATCH -J nilm_t35
#SBATCH -p gpu3,gpu4,gpu2,gpu6,gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

echo "노드 $(hostname) · $(date '+%F %H:%M:%S')"
$PY -X utf8 -c "
import sys, torch
if not torch.cuda.is_available(): sys.exit('GPU 를 못 씁니다')
print('torch', torch.__version__, 'device:', torch.cuda.get_device_name(0))
"
$PY -X utf8 -m src.run_train_cnn --help | grep -q -- "--vswap-p" || { echo "run_train_cnn 에 --vswap-p 가 없다"; exit 1; }
$PY -X utf8 -c "
import json, sys
a = json.load(open('cache/train60_v33/meta.json', encoding='utf-8'))
b = json.load(open('processed_data/holdout60_v33/meta.json', encoding='utf-8'))
for k in ('smps_focus_off_p', 'recipe_mix', 'state_mix', 'carrier_apps', 'window_cycles', 'float_fill', 'steady_crop', 'standby_jitter_cap', 'sibling_rotate'):
    if a.get(k) != b.get(k): sys.exit('캐시와 홀드아웃의 %s 가 다릅니다: %r vs %r' % (k, a.get(k), b.get(k)))
print('짝 관문 통과 · 캐시 %d창 · 홀드아웃 %d창' % (a['n_windows'], b['n_windows']))
"

# v35 = v34 (v33 캐시 + --vswap-p 0.5) + **고차(h9~15) 위상 채널 차폐** (13.84.12).
#   pair4 이식에서 v34 유령의 필요·충분조건이 실측 전류의 h9~15 **위상**이었다 (합성+실측 arg h9..15 → 0.58,
#   실측+합성 arg h9..15 → 0.48; 크기만·저차·전압은 아니다). 실측 고차 위상은 창마다 IQR 1° 로 안정한데 합성은
#   IQR 200° 로 제멋대로라, 생성기가 못 만드는 양을 모델이 판별자로 쓴다. 세밀: Re/Im h9,11,13,15 (4~7·12~15) + φ9 cos/sin
#   (37·38). 광역: φ9 cos/sin (33·34). 크기(광역 |I_h| 12~26 · 세밀 |I9|/|I3| 40)는 남긴다. 체크포인트에 zero_channels 가
#   적혀 추론도 같이 0 으로 만든다 (13.80.10).
time $PY -X utf8 -m src.run_train_cnn \
    --cache cache/train60_v33 \
    --holdout processed_data/holdout60_v33 \
    --epochs 300 --epoch-windows 50000 --w-z 0.3 --w-over 0.0 \
    --eval-every 10 --cache-workers 4 \
    --harm-even-magnitude --state-signatures --standby-operating session \
    --vswap-p 0.5 \
    --zero-channels 4,5,6,7,12,13,14,15,37,38 --zero-wide-channels 33,34 \
    --tag cnn_v35

echo "=== 홀드아웃 v32(회전 없음) 와 v33(회전) 에 v33 · v34 · v35 를 나란히 ==="
for H in processed_data/holdout60_v32 processed_data/holdout60_v33; do
  echo "--- $H"
  $PY -X utf8 -m src.run_score_holdout --ckpt results/cnn_v33.pt results/cnn_v34.pt results/cnn_v35.pt \
      --holdout "$H" 2>&1 | tail -40 || true
done

echo "끝 $(date '+%F %H:%M:%S')"
ls -la results/cnn_v35.pt 2>/dev/null || true
