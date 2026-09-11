# UBAI gate1 HPC 사용 수칙

접속 **설정**은 `gate1-access-setup.md` 를 본다. 이 문서는 **거기서 무엇을 어떻게 돌리는가**다.
여기 적힌 숫자는 전부 실측이고, 잰 날짜를 달아 둔다. 추측으로 고치지 말고 **다시 재서** 고친다.

---

## 0. 가장 중요한 것 — 공용 계정이다

`zetin348` 은 **공용 계정**이다. 모든 활동이 계정 소유자(김광민) 이름으로 기록되고 책임도 그쪽으로 간다.

**금지**
- 로그인 노드에서 무거운 연산 (반드시 스케줄러를 거친다)
- 대용량 스크래치·램 무단 점유 (쓰고 나면 **반드시** 반납)
- 키를 git 에 커밋 (`gate1-access/` 는 `.gitignore` 에 있다. 개인키 평문이다)
- 남의 디렉터리를 지우거나 옮기기 — 홈의 `AIP`·`containers`·`Defence_Robot`·`miniconda3` 는 **우리 것이 아니다**
- **실측 녹화(`processed_data/composite_eval`)를 올려 두기** — 채점 전용 자료다.
  학습에는 안 들어가므로 거기 둘 이유가 없다. 체크포인트(2.5MB)를 받아 로컬에서 채점한다.

**우리 것은 `~/NILM_AI` 하나뿐이다.** `rm -rf ~/NILM_AI` 로 전부 되돌아간다.

### 자체 제한: 동시 3노드 (2026-09-09 사용자 지시)

QOS 는 `node=10 / cpu=640 / gpu=12` 까지 허용하지만 **우리는 3노드로 제한한다.**
"허락은 받았지만 조심히 쓴다" 의 구체적 형태다. 큰 작업은 사용자 동의를 먼저 받는다.

---

## 1. 자리 — 쿼터는 **실재한다**

```bash
/usr/lpp/mmfs/bin/mmlsquota --block-size auto      # 이것만이 정본이다
```
```
gpfs USR   79.2G 사용 / 100G 소프트 / 110G 하드 / grace 7일 | files 644k / 10.2M
```

⚠ **다른 도구는 이 질문에 답을 못 한다** (2026-09-12 에 두 번 틀리고 확인):

| 도구 | 나오는 것 | 왜 틀린가 |
|---|---|---|
| `quota -s` | 출력 없음 | gpfs 를 못 읽는다. "쿼터 없음"이 **아니다** |
| `which mmlsquota` | 없음 | PATH 에 없을 뿐. 전체 경로로 부른다 |
| `df /gpfs` | 466T 중 249T 여유 | 파일시스템 전체지 **우리 몫이 아니다** |
| `du $HOME` | 102G | 남의 것까지 센다. 빼기의 기준으로 쓰지 마라 |

소프트(100G)를 넘으면 **유예 7일**이 시작되고, 그동안은 하드(110G)까지 쓰기가 된다.
유예가 끝나면 막힌다. `mmlsquota` 의 `grace` 칸이 `none` 이어야 정상이다.

**참고 크기**
```
창 캐시 300k창       22.8 GB      홀드아웃 8k창      4.9 GB (원시 X.npy 를 담아 구조가 다르다)
시퀀스 원시 기록당   3.269 MB     conda env          6.3 GB
npz                  283 MB       체크포인트 하나    2.5 MB
```

**지우기 전에**
1. 사용자 승인을 받는다.
2. `meta.json` 만 `results/cache_meta/` 로 옮겨 둔다 — 2.3KB 면 19분에 다시 굽는다.
3. 짝지은 채점에 쓰는 홀드아웃은 남긴다 (`cnn_v30~v37` 을 다시 채점하려면 필요하다).

---

## 2. 스케줄러 — **FIFO 다. backfill 만이 길이다**

```bash
sprio -j <id>                  # 출력 없음  -> 다중요소 우선순위 꺼짐 (priority/basic)
sshare -U                      # FairShare 0.000000, RawUsage 0
scontrol show job <id>         # Priority=4293987143  (제출 순서로 정해지는 값)
```

**먼저 낸 순서대로만 돈다.** 2026-09-12 기준 앞에 120건이 있었다.
자원이 남아돌아도(전체 GPU 380장 중 110장 유휴, 우리를 당장 받을 수 있는 노드 8개) 순서는 못 앞지른다.

**유일한 길은 backfill** — 앞선 작업이 시작되기 전 틈에 끼워 넣는 것. 들어가려면 셋이 필요하다:

| | 나쁨 | 좋음 |
|---|---|---|
| 시간 (`-t`) | 08:00:00 | **실제 예상의 1.5~2배** (예 03:00:00) |
| 코어 (`-c`) | 48 (후보 8노드) | **32 (후보 20노드)** · 16 이면 29노드 |
| 파티션 (`-p`) | `gpu5,gpu4,gpu3` | **다섯 개 다** (A10 포함) |

이렇게 바꾸자 **1분 안에** 잡혔다. 그 전 추정은 "내일 02:35" 였다.

⚠ **`sbatch --test-only` 의 추정을 믿지 마라.** 그것은 주 스케줄러의 FIFO 순서만 모델링하고
**backfill 을 모델링하지 않는다.** 그래서 모양을 어떻게 바꿔도 같은 값이 나온다 — 실제로
CPU 48↔16, GPU 2↔1, 메모리 320↔170G, 시간 8h↔1.5h 전부 같은 값이었다. 그 신호는 무의미하다.

⚠ **짧은 시간을 걸면 초과 위험이 생긴다.** 그래서 학습기는 **epoch 마다 체크포인트를 덮어쓴다**.
죽어도 직전 epoch 은 건진다. 시간을 줄이기 전에 이것부터 확인한다.

### 대기 중인 작업은 다시 내지 말고 **고쳐라**

```bash
scontrol update jobid=<id> TimeLimit=03:00:00
scontrol update jobid=<id> Partition=gpu5,gpu4,gpu3,gpu2,gpu6
scontrol update jobid=<id> MinMemoryNode=204800      # ⚠ 단위는 MB. "200G" 는 거부된다
scontrol update jobid=<id> NumCPUs=32 MinCPUsNode=32
```

`SubmitTime` 이 유지되므로 **FIFO 자리를 안 잃는다.** 취소하고 다시 내면 맨 뒤로 가고,
2026-09-12 기준 그 차이가 **5일**이었다 (09-13 대 09-18).
⚠ `NumCPUs` 만 바꾸면 `CPUs/Task` 가 남아 어긋난다. 모양을 크게 바꿀 거면 차라리 새로 낸다.

---

## 3. 자원 — 무엇이 얼마나 있나 (2026-09-12)

| 파티션 | 노드 | GPU | 대수 | 노드 코어 | **QOS 코어 상한** | 노드 램 |
|---|---|---|---|---|---|---|
| gpu4 | 29 | A6000 | 116 | 56 | 52 (mem 800G) | 1 TB |
| gpu6 | 25 | A10 | 100 | 48 | 40 | 768 GB |
| gpu1 | 14 | RTX 3090 | 56 | 48 | 40 | – |
| gpu2 | 11 | A10 | 44 | 56 | 48 | 1 TB |
| gpu3 | 10 | **A6000 Ada** | 40 | 56 | 48 | 1 TB |
| gpu5 | 6 | A6000 | 24 | **64** | **60** | 1 TB |
| cpu2 | 10 | – | – | 256 | 64 | 1 TB |

- **노드 코어 수보다 QOS 상한이 먼저 막는다.** `cpu2` 에 256 을 부르면 `QOSMaxCpuPerNode` 로 PENDING 이 된다.
- A6000 Ada 가 나머지의 약 2.4배다. 그래서 gpu3 는 40장 중 1장만 놀고 84건이 밀려 있다.
  A6000 · RTX3090 · A10 셋은 서로 25% 안쪽이다.
- gpu5 는 **GPU 가 통째로 비어도 CPU 는 꽉 차 있는** 일이 잦다 (CPU 전용 작업이 앉아 있다).

---

## 4. 램에 굽기 (`/dev/shm`)

큰 캐시를 GPFS 에 둘 자리가 없을 때의 정답이다. **노드 램이라 쿼터와 무관하다.**

```
컴퓨트 노드 n101 실측:  /dev/shm 504 GB · 노드 램 1,007 GB
```

```bash
SHM="/dev/shm/$USER/seqraw_v2"
cleanup() { echo "[정리] $SHM 반납"; rm -rf "$SHM"; df -h /dev/shm | tail -1; }
trap cleanup EXIT              # ⚠ 공용 계정이다. 반납은 선택이 아니다
```

⚠ **SLURM cgroup 이 `/dev/shm` 사용량을 `--mem` 에 잡는다.** 캐시 크기를 반드시 포함해 요청한다.
⚠ 그렇다고 과하게 부르지 마라 — 여유 400GB 짜리 노드에서 320G 를 잡으면 **남의 backfill 을 막는다.**

**예산 잡는 법** (기록 30,000 × 300초 판, 실측 기반):
```
캐시            98 GB   (기록당 3.269MB = 원시 45x18000x4 + 라벨)
워커 36개       46 GB   (상한. 측정: 워커 12개 14.36 GiB — 공유 페이지 중복 포함이라 실제는 더 작다)
주 프로세스 2개  4.5 GB  (측정 2.09 GiB x2)
--------------------------
합              149 GB  ->  --mem=200G (34% 여유)
```

---

## 5. 무엇이 빨라지고 무엇이 안 빨라지나 (실측)

```
캐시 300k창      로컬 2.6시간  ->  HPC 64코어 19분      8배 빠르다
1단계 훈련       로컬 RTX5070 26분  ->  A10 30분        오히려 느리다
```

**HPC 의 값은 단일 작업 가속이 아니라 코어 수와 병렬 실험이다.** GPU 한 장짜리 훈련만 하려면 로컬이 낫다.

**병목을 먼저 재라.** 시퀀스 학습기는 GPU 점유가 19~50% 였다 — 병목은 GPU 가 아니라
워커가 `build_inputs` 로 입력을 만드는 **CPU** 다 (워커당 약 380 창/초).
- 점유가 낮으면 → 코어를 늘린다 (gpu5 의 60 이 상한)
- 90% 넘게 붙어 있으면 → 코어를 늘려도 소용없다. A6000 Ada(gpu3)로 간다
- 교차점은 판당 워커 18개쯤이다. 그 너머로는 코어를 더 줘도 안 빨라진다

⚠ 짝지은 A/B 는 **GPU 종류와 무관하다.** 같은 자료·같은 스텝·같은 씨앗이면 GPU 속도는
벽시계만 바꾼다. 두 판을 **같은 노드**에 올리면 그마저도 같아진다.

---

## 6. 전송

```
업로드 1.66 MB/s · 다운로드 2.3 MB/s     (3홉 터널: 내 PC -> Oracle VPS -> 라즈베리파이 -> gate1)
```

⚠ **캐시를 가져오지 마라** — 23GB 면 2.8시간으로 로컬 빌드(2.3h)보다 느리다.
**거기서 만들고 거기서 훈련해 체크포인트(2.5MB)만 가져오는** 구조여야 한다.

`rsync` 는 로컬(Git Bash)에 없다. tar 를 파이프로 흘린다:
```bash
tar czf - src patches | ssh gate1_External 'cd ~/NILM_AI && tar xzf -'     # 8MB, 1분
```

---

## 7. sbatch 규약

1. **LF 로 만든다.** `tr -d '\r' < patches/x.sbatch > x.sbatch` 를 거쳐 제출한다. CRLF 면 안 돈다.
2. **한국어 heredoc 을 피한다.** 패치는 파일로 만들어 올린다.
3. **관문을 앞에 둔다** — 코드가 올라왔는지(`--help | grep -q`), 캐시 설정이 실제로 걸렸는지,
   짝지은 캐시·홀드아웃이 같은지. 관문이 없으면 몇 시간 뒤에야 틀린 것을 안다.
4. **⚠ 관문으로 작업을 죽이지 마라 — 그 관문이 없어도 되는 것이라면.**
   쿼터 관문이 n055 에서 `mmlsquota: GPFS is down on this node` 로 실패해 `set -e` 가
   작업을 6초 만에 죽였다. 파일 읽기는 멀쩡했고 우리가 쓰는 건 10MB 였다. 비치명으로 바꿨다.
5. **epoch 마다 체크포인트를 덮어쓴다.** 짧은 시간을 걸어 backfill 에 끼는 대가다.
6. **`trap ... EXIT` 로 램을 반납한다.**
7. 명령은 `ssh gate1_External` **그대로**. `-p 8000` 을 붙이면 라즈베리파이 키의 `permitopen`
   이 거부한다 (`administratively prohibited`). 포트는 `~/.ssh/config` 에 있다.

---

## 8. 재현성

캐시를 굽는 코드는 **작업 번호로 전역 난수를 시드**해야 한다.

```python
from src.synthesis.dataset import chunk_seed
np.random.seed(chunk_seed(seed_base, index))     # traincache._chunk 와 같은 규약
```

⚠ 생성기는 **전역 난수**를 먹는다 (환경 표집·잡음·회로 되먹임). 일정만 지역 `RandomState` 로
뽑아서는 재현이 안 된다 — 워커 수와 `imap_unordered` 가 집어 가는 순서에 따라 달라진다.
`run_build_seqraw` 가 이 규약을 안 베껴서 `seqraw_v1` 이 재현 불가였다 (2026-09-12 에 고침).
확인: 워커 4개와 7개로 구운 것이 **바이트 단위로 같아야** 한다.

그리고 **새 학습기를 만들 때 캐시 굽는 인수를 sbatch 에서 통째로 베껴라.**
`run_build_seqraw` 가 생성기를 맨몸으로 만들어 `cache_v36.sbatch` 의 설정 열하나가 빠져 있었다.
`src/synthesis/genopts.py` 의 `V36` 이 정본이고 `check()` 가 관문이다.

---

## 9. 자주 쓰는 명령

```bash
ssh gate1_External                                   # 접속 (-p 붙이지 말 것)
squeue -u zetin348 -o "%.9i %.9P %.4t %.9M %.14R"    # 내 작업
squeue -j <id> --start                               # 예상 시작 (FIFO 기준. backfill 은 안 보인다)
sacct -j <id> --format=State,ExitCode,Elapsed,NodeList,ReqCPUS,AllocCPUS,ReqMem
scontrol show job <id>                               # 자세히
scancel <id>
/usr/lpp/mmfs/bin/mmlsquota --block-size auto        # 자리
sinfo -p gpu5 -N -o "%.9N %.5c %.10m %.14G %.6t %.20C"
sinfo -h -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -N -O "Partition:8,Gres:20,GresUsed:22"   # GPU 여유
```
