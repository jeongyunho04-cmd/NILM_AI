#!/bin/bash
# 14.62 — **차수별 Z_h 판**: 캐시 하나만 다른 두 갈래를 **같은 판·같은 씨**로 돌린다.
#
#   끔  `cache/train60_v32`   (z_h = r + j·h·x)         -> cnn_v32h_off_s{0,1,2}
#   켬  `cache/train60_v32h`  (실측 표: D h3·h5 · E h3) -> cnn_v32h_on_s{0,1,2}
#
# ⚠ 손잡이는 **캐시 하나뿐**이다. 나머지 인자는 `cnn_vanchor_seeds.sbatch` 의 B 갈래와
#   글자까지 같다 (= 지금 본선 조리법). 그래서 `cnn_v32h_off_*` 는 옛 `cnn_vanch_on_*` 과
#   같은 물건이어야 한다 — **그 둘이 다르면 그 사이에 코드가 움직인 것이다.** 표류 검정이다.
#
# 왜 기준선을 새로 돌리나: 이 판은 노드 하나에 GPU 두 장을 잡는데 한 장이 어차피 논다.
#   같이 돌리면 **같은 코드·같은 씨의 짝비교**가 되어 공짜로 더 세진다.
#
# 입력이 얼마나 바뀌었나 (`run_gate_hzwire`, 창 6개): 세밀 8.02% · 광역 9.44% RMS.
#   c46 Re(V3) **44%** · c53 Im(V5) 8% · c52 Im(V3) 7% · c15(전류) 11%.
#
# ★ 미리 적는 예측 (나중에 고르지 않으려고):
#   ① 저항 4종 신원 — 기준 0.9815. Z_h 는 저항기기 전압강하의 **파형**을 고치니 여기가
#      움직여야 한다. 씨앗 바닥이 0.043~0.062 라 **0.062 를 넘어야 읽은 것**이다.
#   ② `run_diag_vblock --block fine` — 입력이 8% 바뀌었는데 **안 움직이면** 모델이 전압
#      블록을 거의 안 쓴다는 뜻이다 (그것도 결론이다).
#   ③ 오븐 수준 오차 +24.9 ± 4.5W — 오븐은 자리 D(Z_1 1.15Ω) 녹화고 h3 이 6.71배가 된다.
#      **여기가 겨냥이다.**
#   ④ `핫플통전·P>1500` −16.3 ± 7.5W — 이미 고친 자리다. **되돌아가면 안 된다.**
#
#SBATCH -J nilm_v32ht
#SBATCH -p gpu1,gpu6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
# ⚠⚠ `-t` 는 **실측의 1.5배** (HPC_RULES §2). 983785 가 A10 에서 ~31분이었다.
#SBATCH -t 00:50:00
#SBATCH -a 0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python
CACHE_OFF=${CACHE_OFF:-cache/train60_v32}
CACHE_ON=${CACHE_ON:-cache/train60_v32h}
HO_OFF=${HO_OFF:-processed_data/holdout60_v32}
HO_ON=${HO_ON:-processed_data/holdout60_v32h}
EPOCHS=${EPOCHS:-300}
SEED=${SLURM_ARRAY_TASK_ID}
echo "노드 $(hostname) · 씨 $SEED · $(date '+%F %H:%M:%S')"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== 관문 ==="
# 14.66 — `%` 형식 자리 수. 983771 이 이 버그로 **1분 만에** 죽었다 (`run_gate_help` 는
#   argparse help 만 보고 런타임 print 는 안 본다). 1초짜리라 여기 건다.
$PY -X utf8 src/run_gate_fmt.py
$PY -X utf8 -m src.run_gate_lossparity
$PY -X utf8 -m src.run_gate_ctordefault
for d in "$CACHE_OFF" "$CACHE_ON" "$HO_OFF" "$HO_ON"; do
  test -d "$d" || { echo "없다: $d"; exit 1; }
done
# ⚠ 두 캐시가 **harmonic_z 하나만** 다른지 여기서 다시 본다. 굽기 때 봤어도 또 본다 —
#   굽기와 학습 사이에 사람이 뭘 덮어썼을 수 있다.
$PY -X utf8 -c "
import json, sys
a = json.load(open('$CACHE_OFF/meta.json', encoding='utf-8'))
b = json.load(open('$CACHE_ON/meta.json', encoding='utf-8'))
if not b.get('harmonic_z'): sys.exit('켬 캐시 meta 에 harmonic_z 가 없다')
if a.get('harmonic_z'):     sys.exit('끔 캐시 meta 에 harmonic_z 가 있다')
skip = {'harmonic_z','shard','shard_chunks','build_seconds','bytes','merged_from',
        'merged_partial','positive_rate','recipe_counts','built_at','content_sha256'}
# ⚠ '키 없음' 과 '0/False' 는 **같은 뜻**이다 — 'train60_v32' 는 'vtex_step_s' 가 생기기
#   전에 구운 캐시라 키가 아예 없고, 새 캐시는 0.0 으로 적는다. 그대로 견주면 983992 처럼
#   자료가 멀쩡한데 관문이 운다 (자료 비트 동일은 굽기 관문 ③이 이미 증명했다).
def nz(v):
    return None if v is None or v == 0 or v == '' or v is False else v
d = [k for k in set(a)|set(b) if k not in skip
     and json.dumps(nz(a.get(k)), sort_keys=True, default=str)
      != json.dumps(nz(b.get(k)), sort_keys=True, default=str)]
if d: sys.exit('두 캐시가 harmonic_z 말고도 다르다: %s' % d)
print('  관문 OK 두 캐시는 harmonic_z **하나만** 다르다 (창 %d)' % b['n_windows'])
"

BASE="--epochs $EPOCHS --epoch-windows 50000 --seed $SEED \
      --w-z 0.3 --w-over 0.0 --w-harm 0.1 --eval-every 10 --cache-workers 4 \
      --harm-even-magnitude --state-signatures --standby-operating session \
      --harm-vnorm-classes RESISTIVE --harm-vnorm-anchor"
echo "=== 학습 시작 $(date '+%F %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache "$CACHE_OFF" --holdout "$HO_OFF" \
    --tag cnn_v32h_off_s${SEED} > "results/cnn_v32h_off_s${SEED}.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -X utf8 -m src.run_train_cnn $BASE \
    --cache "$CACHE_ON" --holdout "$HO_ON" \
    --tag cnn_v32h_on_s${SEED} > "results/cnn_v32h_on_s${SEED}.log" 2>&1 &
B=$!
RC=0; wait $A || RC=1; wait $B || RC=1
echo "=== 끔 (r + j*h*x) ==="; tail -32 "results/cnn_v32h_off_s${SEED}.log"
echo "===================================================================================="
echo "=== 켬 (실측 Z_h 표) ==="; tail -32 "results/cnn_v32h_on_s${SEED}.log"
ls -la results/cnn_v32h_*_s${SEED}.pt
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
echo
echo "판정 (로컬, 실측 복합 5파일). **이 순서로**:"
echo "  1 ★ run_diag_rollback --postproc off — 저항 4종 신원 (기준 0.9815, 씨앗 바닥 0.062)"
echo "  2 run_diag_vblock --block fine --pair 3"
echo "  3 run_diag_levelerr — 오븐 (기준 +24.9 +- 4.5W)"
echo "  4 run_diag_hipow --arms — 핫플통전 P>1500 (기준 -16.3 +- 7.5W, **되돌아가면 안 된다**)"
echo "  5 표류 검정: cnn_v32h_off_* 대 옛 cnn_vanch_on_* — 같아야 한다"
