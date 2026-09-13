#!/bin/bash
# 14.3 / 13.87 — **합 정합성 사영** (계획 A). 한 노드에 한 팔, 세 번 낸다.
#
#     ARM=ctl   sbatch patches/seq_proj.sbatch     # 대조군      proj 0
#     ARM=size  sbatch patches/seq_proj.sbatch     # A-크기      proj 1 · resp power
#     ARM=alloc sbatch patches/seq_proj.sbatch     # A-배분      proj 1 · resp head
#
# ── 왜 한 노드에 한 팔인가 (HPC_RULES §2) ──────────────────────────────────
# "작업을 나누면 코어가 곱절이 된다. CPU 가 병목인 이 작업에서는 그것이 유일한
# 증속 수단이다." 한 노드에 두 팔이면 판당 워커가 (48-2)/2 = 23 인데, 세 작업으로
# 나누면 판당 44 다. 2026-09-12 18:5x 실측 문턱: >=52 5노드 · **>=48 10노드** ·
# >=44 10노드 · >=40 19노드 — **48 과 44 의 후보가 같으므로 48 이 공짜**다.
# 3노드는 자가제한(§0)의 상한이고, 144코어/3GPU 는 QOS(640/12) 안이다.
#
# ── 무엇을 재나 ────────────────────────────────────────────────────────────
# 13.84.74: 2단계가 큰 부하 잔차를 59 -> 105W 로 만든다. 합을 맞추라는 제약이
#   사슬 어디에도 없다 (`cons=0.0` 이 1·2단계 모두).
# 13.86: **CO-P 가 잔차 1.2W 를 낸다** — 이산 상태공간 안에 총전력을 맞추는 해가
#   실재한다. 우리 40.7W 는 물리 한계가 아니라 모델 실패다.
# 13.87: 사영을 넣었다. 다만 `resp=power` 는 **창별 곱셈 재조정**이라 크기만 고친다
#   (창내 배율 표준편차 4e-8). 배분을 옮기는 것은 `resp=head` 뿐이다.
#   그리고 잔차의 **97% 가 2대 이상 켜진 창**에 있다 (혼자 켜진 창은 12.0W).
#
#   ⇒ 그래서 세 팔이다. ctl -> size 는 **합 제약 하나**, size -> alloc 은
#     **배분 자유 하나**만 바뀐다.
#
# ⚠⚠ **잔차 숫자로 판정하지 마라** (13.87 [3]). proj=1 이면 잔차가 정의상 0 에
#   가깝다 — 13.86 에서 CO-P 의 1.2W 를 품질로 읽지 말라고 적은 그 함정이다.
#   판정은 **신원 · 검출 · 혼자켜짐 전력**으로 한다. 채점은 실측이 있는 로컬에서.
#
# ⚠ 대조군을 새로 굽는다. `seq_v36p_mask` 는 13.84.68 의 `state_power_init` **이전**
#   판이라, 그걸 대조군으로 쓰면 바뀌는 것이 둘이 된다 (사영 + 슬롯 초기화).
#   덤으로 ctl 대 seq_v36p_mask 가 13.84.70 ① 의 측정이 된다.
#SBATCH -J nilm_proj
# ⚠ **gpu3·gpu4 만** 쓴다 (A6000 Ada / A6000). 앞 판에서 코어 문턱을 40 으로
#   낮춰 gpu6(A10)을 받았더니 **1.36배** 느렸다 (249 대 339 s/epoch, 같은 코어).
#   코어 상한도 여기가 48·52 라 -c 48 이 그대로 선다 (gpu1·gpu6 은 40 이 끝).
#SBATCH -p gpu3,gpu4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
# 972759(v36p, A10) 가 2팔 2:30 이었다. 1팔이고 워커가 곱절이라 짧아야 정상이지만
# 캐시를 팔마다 새로 굽는 대가가 있다. epoch 마다 덮어쓰므로 초과해도 직전까지 건진다.
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out
set -euo pipefail
cd "$HOME/NILM_AI"
PY=./env/bin/python

ARM=${ARM:-ctl}
GEN=${GEN:-v36p}
RECORDS=${RECORDS:-30000}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
SHM="/dev/shm/$USER/seqraw_${GEN}_${ARM}_s${SEED}"

case "$ARM" in
  ctl)    EXTRA="";                                 TAG="seq_proj_ctl"   ;;
  size)   EXTRA="--proj 1.0 --proj-resp power";     TAG="seq_proj_size"  ;;
  alloc)  EXTRA="--proj 1.0 --proj-resp head";      TAG="seq_proj_alloc" ;;
  # ── 2026-09-13 판 — **고친 S_STATE 위에서** 다시 세운다 (13.92) ───────────
  # 앞 세 팔(974259/260/390)은 `S_STATE["hotplate"]` 슬롯 버그가 있는 채로 구웠다.
  # 팔 사이 비교는 유효하지만 **기준선 자체가 틀렸고**, 하필 고친 것이 핫플이라
  # `alloc` 이 이득을 주장한 저항 4종과 겹친다. 그래서 기준선부터 다시 굽는다.
  ctlfix) EXTRA="";                                 TAG="seq_fix_ctl"    ;;
  alloc2) EXTRA="--proj 1.0 --proj-resp head";      TAG="seq_fix_alloc"  ;;
  attn)   EXTRA="--appl-attn 64";                   TAG="seq_fix_attn"   ;;
  # ── 사슬 롤백 (2026-09-13) ────────────────────────────────────────────────
  # `--w-crf 0` 이면 손실이 `창 손실 + 0·CRF` 라 **정확히 1단계 학습**이다 —
  # 사슬이 목적함수에 안 들어간다. `--keep-on 1.0` 이라 켜짐 BCE 는 그대로 산다.
  # 생성기(v36p)·고친 S_STATE·나머지 설정은 다른 팔과 **완전히 같다**.
  #   -> 이것이 "생성기를 안 바꾸는 롤백" 이고, 창 캐시 300k(2h) + 홀드아웃(3h) +
  #      300epoch(6h) = 9시간짜리 정공법을 80분으로 줄인 것이다.
  # ⚠ 진짜 1단계(`run_train_cnn`)와 **같지 않다**: 자료 파이프라인(사슬 기록을 창으로
  #   자름)과 epoch/lr 일정이 다르다. 가르려는 것은 **CRF 항이 몸통을 망치는가** 하나다.
  nochain) EXTRA="--w-crf 0";                       TAG="seq_fix_nochain" ;;
  # ── 13.99 뒤 첫 판 — 사슬이 **정말 배우는** 첫 학습 ───────────────────────
  # `seq_fix_*` 여섯 판은 옵티마이저 무리가 조용히 비어 `ChainHeads` 가 12에포크
  # 내내 **초기값 그대로**였다. 설정은 `ctlfix` 와 **완전히 같고** 바뀐 것은
  # `run_train_seq` 의 조립 한 줄뿐이다 — 그래서 `seq_fix_ctl` 과 나란히 놓으면
  # **"사슬을 배우는 것이 값을 하나" 를 한 가지 차이로** 잰다.
  # ⚠ 꼬리표를 따로 둔다. `seq_fix_*` 는 13.96~13.99 의 증거라 **덮으면 안 된다.**
  chainfix) EXTRA="";                               TAG="seq_chain_ctl"  ;;
  *) echo "ARM 은 ctl|size|alloc|ctlfix|alloc2|attn|nochain|chainfix 이다: $ARM"; exit 1 ;;
esac

# ⚠ 씨앗이 0 이 아니면 **꼬리표에 붙인다** — 안 그러면 `results/$TAG.pt` 가
#   씨앗 0 판을 덮어쓴다 (12.39 처럼 씨앗 반복을 할 때 곧바로 문다).
[ "$SEED" = "0" ] || TAG="${TAG}_s${SEED}"

NCPU=$(nproc)
# ⚠ 고정 숫자로 박지 않는다 (§2) — 코어가 줄어든 노드에 잡히면 과구독된다.
W=$(( NCPU - 4 ))
[ "$W" -ge 8 ] || { echo "코어가 너무 적다 (nproc=$NCPU)"; exit 1; }
BW=$W
echo "노드 $(hostname) · $(date '+%F %H:%M:%S') · 팔 $ARM · nproc $NCPU · 워커 $W"
echo "  추가 인수: [$EXTRA]"
nvidia-smi --query-gpu=index,name --format=csv,noheader || nvidia-smi | head -6

cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT

# ── 관문 (앞에, §7-3) ───────────────────────────────────────────────────────
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--proj-resp" || { echo "--proj-resp 가 없다 — 코드를 안 올렸다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--w-cons"    || { echo "--w-cons 가 없다"; exit 1; }
$PY -X utf8 -m src.run_train_seq --help | grep -q -- "--keep-last" || { echo "--keep-last 가 없다"; exit 1; }
test -f results/cnn_v37.pt || { echo "results/cnn_v37.pt 가 없다"; exit 1; }

# ⚠⚠ 13.99 — 파라미터 무리를 리스트로 굳히는가. 제너레이터면 중복검사가 소진시켜
#   AdamW 가 **빈 무리**를 받고 사슬이 **한 걸음도 안 배운다.** 예외가 안 나므로
#   여기서 소스를 직접 본다. (연기 시험도 잡지만 그건 캐시 굽기 7분 뒤다.)
grep -q 'g\["params"\] = list(g\["params"\])' src/run_train_seq.py \
  || { echo "13.99 고침이 안 올라왔다 — 사슬이 안 배운다"; exit 1; }
# ⚠ `set -e` 라 `grep && {...}` 를 쓰면 **못 찾았을 때**(정상) 스크립트가 죽는다.
if grep -q '{"params": heads.parameters()' src/run_train_seq.py; then
  echo "13.99 옛 코드가 남아 있다 — 제너레이터를 무리에 넣는다"; exit 1
fi

# ⚠⚠ 핵심 관문 — 사영이 꺼지면 항등이고 켜면 동작하는가, head 가 power 의 정확한
#   일반화인가. 실측 npz 는 HPC 에 없으므로 자료 없는 갈래로 돈다 (§0).
$PY -X utf8 src/run_gate_proj.py || { echo "사영/어텐션 관문 실패"; exit 1; }

# ⚠ S_STATE 슬롯이 풀의 상태별 실측과 맞나 (13.92 [3]) — 핫플이 어긋나 있었다
$PY -X utf8 -m src.run_gate_states || { echo "상태 슬롯 관문 실패"; exit 1; }

# 생성기가 그대로인지 (seq_slot.sbatch 와 같은 관문)
$PY -X utf8 -m src.run_gate_driftaug -n 2000 --check smps_dev2 || { echo "증강 관문 실패"; exit 1; }

# ── 1 캐시 ──────────────────────────────────────────────────────────────────
NEED=$(( RECORDS * 4 / 1000 ))
FREE=$(df -BG /dev/shm | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "[램] /dev/shm 여유 ${FREE}GB · 필요 어림 ${NEED}GB"
[ "$FREE" -gt $(( NEED * 13 / 10 )) ] || { echo "램이 모자란다"; exit 1; }

mkdir -p "$SHM"
time $PY -X utf8 -m src.run_build_seqraw \
  --out "$SHM" --records "$RECORDS" --record-s 300 --grid-s 2.0 \
  --gen "$GEN" --workers "$BW" --seed 0

$PY -X utf8 -c "
import json, sys
m = json.load(open('$SHM/meta.json', encoding='utf-8'))
if m.get('gen') != '$GEN': sys.exit('캐시가 $GEN 이 아니다: %r' % m.get('gen'))
if m.get('gen_opts', {}).get('sibling_rotate') != 'smps_dev2':
    sys.exit('캐시 메타의 sibling_rotate 가 %r' % m.get('gen_opts', {}).get('sibling_rotate'))
print('캐시 관문 통과 · 기록 %d · 단계 %d' % (m['records'], m['steps']))
"
du -sh "$SHM"

# ── 1b 연기 시험 — 조립까지 서는가 (13.88) ─────────────────────────────────
# 974261 이 **옵티마이저 조립**에서 죽었다 ("some parameters appear in more than one
# parameter group"). 캐시 굽기 4분과 관문을 다 지나고 나서였다. 몇 초로 미리 잡는다.
$PY -X utf8 -m src.run_train_seq --cache "$SHM" --epochs 1 --no-real     --init scratch --ref results/cnn_v37.pt --records-per-batch 6 --chunk 64     --workers 2 --holdout-records 8 $EXTRA --tag dryrun_$ARM --dry-run   || { echo "연기 시험 실패 — 학습을 안 낸다"; exit 1; }

# ── 2 한 팔 ────────────────────────────────────────────────────────────────
# 세 팔 사이에 바뀌는 것은 EXTRA 뿐이다. 나머지는 13.84.69 의 설정 그대로다.
COMMON="--cache $SHM --epochs $EPOCHS --seed $SEED --records-per-batch 6 --chunk 64 \
        --workers $W --vswap-p 0.5 --eval-every 1 --no-real --holdout-records 300 \
        --init scratch --ref results/cnn_v37.pt --lr 1e-4 --keep-on 1.0 --keep-last 4"

$PY -X utf8 -m src.run_train_seq $COMMON $EXTRA --tag "$TAG" \
  > "results/${TAG}_s${SEED}.log" 2>&1 &
J=$!
echo "학습 시작 pid $J — $(date '+%F %H:%M:%S')"

( sleep 900; echo "[점유] $(date '+%H:%M:%S') $(uptime | sed 's/.*average/load/')"; nvidia-smi | grep Default || true ) &
MON=$!

RC=0
wait $J || RC=1
kill $MON 2>/dev/null || true

echo "=== $ARM ($TAG) ==="
tail -25 "results/${TAG}_s${SEED}.log"
ls -la "results/${TAG}"*.pt 2>/dev/null || echo "(체크포인트 없음)"
# ⚠⚠ 끝단에서 mmlsquota 를 부르지 않는다 (§7-5, 세 번 당했다).
echo "끝 $(date '+%F %H:%M:%S') · rc=$RC"
exit $RC
