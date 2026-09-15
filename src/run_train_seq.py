# -*- coding: utf-8 -*-
"""끝까지 함께 — 몸통 + 방출 머리 + 전이 머리를 사슬 손실로 같이 학습한다 (13.84.26).

```
지금   입력 -> 2갈래 CNN -> 기기별 머리(켜짐·전력) -> 창마다 독립 판정
바꿈   입력 -> 같은 CNN  -> 머리 둘(방출·전이)     -> 사슬 디코딩 -> 파일 전체 상태 열
```

손실은 **켜짐 항만** 바뀐다. `NILMLoss` 의 `total` 에서 `w.on * parts["on"]` 을 빼고
`w_crf * CRF` 를 더한다. 전력·상태·플러그·대기·고조파·z 는 창마다 정답이 있으므로 그대로다.

13.84.24 의 얼린 몸통 판이 실측 0.852 -> 0.912~0.922 를 냈다(시드 6판). 이 판은 몸통도 배운다.

⚠ **실측 채점은 매번 지금 몸통으로 다시 돈다.** 얼린 판의 z 를 재사용하면 몸통이 바뀐 뒤로
   옛 표현을 채점하는 꼴이 된다.

    python -X utf8 src/run_train_seq.py --cache cache/seqraw_v1 --epochs 40
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.chain import BgHead, ChainHeads, crf_nll, viterbi
from src.model.inputs import build_inputs
from src.model.lossbuild import build_loss
from src.model.losses import LossWeights
from src.model.transition import N_FEAT, feat_at
from src.model.realdata import RealWindows
from src.run_gate_check import load_model
from src.run_train_cnn import vswap

FS = 60
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
PRE = 13
LABELS = ("y_on", "y_power", "y_plugged", "y_standby", "y_state",
          "obs_harm", "p_noise", "p_observed", "z_grid")


def dfeat_seq(H, P, cycles):
    df = np.zeros((len(cycles), N_FEAT + 2), np.float32)
    for q, c in enumerate(cycles):
        f_, dp_ = feat_at(H, P, int(c))
        df[q, :N_FEAT] = f_
        df[q, -2] = np.sign(dp_)
        df[q, -1] = np.log10(abs(dp_) + 1e-3)
    return df


class SeqChunks(Dataset):
    """기록 하나에서 연속 `chunk` 단계를 잘라 입력을 **그때 만든다**.

    ⚠ 입력을 미리 구우면 격자 2초·세밀 10초라 같은 사이클이 다섯 번 저장돼 5.9배다
      (단계당 155.6KB · 120만 단계면 187GB). 워커가 만드는 편이 싸다 (코어당 572단계/초).
    """

    def __init__(self, cache, idx, chunk, tgt_off, w_cyc, grid, fixed_start=None):
        #: 홀드아웃 채점은 **자리를 고정**한다 — 매번 다른 조각을 보면 epoch 사이 값이 흔들려
        #: 평평해졌는지 못 읽는다.
        self.fixed_start = fixed_start
        self.c = Path(cache)
        self.idx = np.asarray(idx)
        self.chunk, self.tgt_off, self.w_cyc = int(chunk), int(tgt_off), int(w_cyc)
        self.grid = np.asarray(grid)
        self._a = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self._a is None:
            self._a = {n: np.load(self.c / (n + ".npy"), mmap_mode="r")
                       for n in ("raw",) + LABELS}
        i = int(self.idx[j])
        T = len(self.grid)
        if self.chunk >= T:
            s = 0
        elif self.fixed_start is not None:
            s = int(self.fixed_start) % (T - self.chunk + 1)
        else:
            s = int(np.random.randint(0, T - self.chunk + 1))
        sl = slice(s, s + min(self.chunk, T))
        g = self.grid[sl]
        raw = np.asarray(self._a["raw"][i])
        win = np.stack([raw[:, c - self.tgt_off:c - self.tgt_off + self.w_cyc] for c in g])
        fine, wide = build_inputs(win)
        out = {"fine": fine, "wide": wide,
               "dfeat": dfeat_seq((raw[0:15] + 1j * raw[15:30]).T, raw[30], g)}
        for n in LABELS:
            out[n] = np.asarray(self._a[n][i][sl])
        return out


def collate(b):
    return {k: torch.from_numpy(np.stack([x[k] for x in b])) for k in b[0]}


def _alloff_base(ev_file, P, n):
    """그 파일의 **전부-꺼짐 기준선** (W). 라벨상 아무것도 안 켜진 사이클의 중앙.

    ⚠ 라벨이 못 잡은 사건이 섞이므로 **중앙**을 쓰고 상위 10% 를 떨군다 (13.84.64 ③).
    """
    t = np.arange(n) / FS
    off = np.ones(n, bool)
    for v in ev_file.get("intervals", {}).values():
        for t0, t1 in v.get("on", []):
            off &= ~((t >= t0) & (t < t1))
    if off.sum() < 60:
        return float("nan")
    off &= P <= np.percentile(P[off], 90)
    return float(np.median(P[off]))


def real_windows(apps, grid_s, dev):
    """실측 파일의 입력·Δ특징·참값을 **한 번만** 만든다. 몸통은 채점 때마다 새로 돈다."""
    from src.preprocessing import load_nilm_npz
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    stride = int(grid_s * FS)
    out = {}
    for stem in FILES:
        rw = RealWindows(stems=[stem], stride=stride, require_valid=False)
        tgt = rw.target_cycle
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        ok = (tgt >= PRE * FS + 1) & (tgt < n - PRE * FS - 1)
        sel = np.nonzero(ok)[0]
        F, W, OH, PN = [], [], [], []
        for i in range(0, len(sel), 512):
            f, w, _, oh, pn = rw.batch(sel[i:i + 512])
            F.append(f); W.append(w); OH.append(oh); PN.append(pn)
        t = tgt[sel] / FS
        y = np.zeros((len(t), len(apps)), np.int8)
        for k, a in enumerate(apps):
            for t0, t1 in ev[stem]["intervals"].get(a, {}).get("on", []):
                y[(t >= t0) & (t < t1), k] = 1
        out[stem] = dict(fine=np.concatenate(F), wide=np.concatenate(W),
                         dfeat=dfeat_seq(H, P, tgt[sel]), y=y, t=t,
                         # 창별 **관측 총전력**. 저항 무리 신원 채점의 전력 하한이 이걸 쓴다
                         # (`run_score_seq` 의 `RESISTIVE_MIN_TRUE_W` 주석).
                         p_obs=P[tgt[sel]].astype(np.float32),
                         # 그 파일의 **전부-꺼짐 기준선** (배경 + 꽂힌 것들의 대기, W).
                         # ⚠ **사이클 단위**로 잡는다 — 창 단위는 전부-꺼짐이 파일당 1~12개뿐이라
                         #   중앙이 안 선다. 사이클로는 999~2,922개다 (13.84.64 ③ 이 쓴 그 양).
                         #   `run_score_seq` 의 전력 채점이 이것을 참값의 기준으로 쓴다.
                         p_base=_alloff_base(ev[stem], P, n),
                         # 후처리가 쓰는 셋 (14.31). **추가 키라 옛 호출부는 그대로 돈다.**
                         # `run_diag_rollback --postproc` 가 `resistive_match` 를 부르려면
                         # 관측 고조파·계측 바닥·창 전압이 있어야 한다.
                         obs_harm=np.concatenate(OH), p_noise=np.concatenate(PN),
                         v_obs=np.asarray(rw.v_observed, np.float64)[sel],
                         present=np.array([a in ev[stem]["appliances_present"] for a in apps]))
    return out


def _stage1_physics(a, apps):
    """1단계 체크포인트가 적어 둔 손실 물리를 `build_loss` 인자로 바꾼다 (14.171).

    ⚠ **옛 체크포인트에는 키가 없다.** 그때는 빈 dict 를 내어 **옛 동작과 비트 동일**로
    둔다 — 없는 값을 지어내면 지난 판과 비교가 끊긴다.
    """
    import torch as _T
    path = getattr(a, "init", "") or getattr(a, "ref", "") or getattr(a, "ckpt", "")
    if not path:
        return {}
    try:
        ck = _T.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    out = {}
    if ck.get("harm_sig_vnorm") is not None and ck.get("harm_vnorm_anchor"):
        from src.run_train_cnn import _vnorm_exp
        out["harm_sig_vnorm"] = bool(ck["harm_sig_vnorm"])
        out["harm_vnorm_exp"] = _vnorm_exp(apps, ck.get("harm_vnorm_classes", ""))
        out["harm_vnorm_frac"] = float(ck.get("harm_vnorm_frac", 1.0))
    if ck.get("cons_deadzone") is not None:
        out["cons_deadzone"] = float(ck["cons_deadzone"])
    if ck.get("res_apps") is not None:
        from src.run_train_cnn import _res_cond, _res_ohm
        out["res_ohm"] = _res_ohm(apps, ck["res_apps"], half=False)
        out["res_ohm_half"] = _res_ohm(apps, ck["res_apps"], half=True)
        out["res_cond_state"] = _res_cond(apps, ck.get("res_cond_state", ""))
        out["swap_tol"] = float(ck.get("swap_tol", 0.02))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--init", default="results/cnn_v37.pt",
                    help="몸통 초기점. 'scratch' 면 **무작위 초기화**로 처음부터 배운다 (13.84.27)")
    ap.add_argument("--ref", default="results/cnn_v37.pt",
                    help="--init scratch 일 때 **구조·가림**을 가져올 체크포인트")
    ap.add_argument("--no-mask", action="store_true",
                    help="v37 이 가린 고차 위상 채널(세밀 4~7,12~15,37,38 · 광역 33,34)을 **살린다** "
                         "(13.84.27). ⚠ --init scratch 에서만 쓸 것 — 물려받은 가중치에는 본 적 없는 "
                         "입력이 된다. 가린 이유는 생성기가 h9~15 위상을 못 만들어서였는데(13.84.12), "
                         "지금 캐시는 sibling_rotate 로 그 단서를 원천에서 없앴고 전이 머리는 어차피 "
                         "원시를 가리지 않고 본다")
    ap.add_argument("--vswap-p", type=float, default=0.0, metavar="P",
                    help="학습 배치에서 전압 고조파 채널을 다른 창 것으로 바꿔 끼운다 (13.84.11). "
                         "⚠ v37 은 0.5 로 배웠다 — 0 으로 이어 학습하면 그 규제가 풀린다")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--records-per-batch", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-heads", type=float, default=3e-4)
    ap.add_argument("--vexp", action="store_true",
                    help="전압 지수를 구조에 박는다 (14.7). `p_raw *= (V/V_CENTER)^e_k` 로 "
                         "저항 e=2 · SMPS 0 · 유도기 0.6. 모델은 상태 명목값은 기울기 1.00 으로 "
                         "잘 내는데 **같은 상태 안의 V² 의존**을 0.33~0.83 로만 읽어 순 지수가 "
                         "0.85 다(물리는 2). 저전압에서 과예측한다 (14.6). 끄면 비트 동일")
    ap.add_argument("--w-crf", type=float, default=0.3)
    ap.add_argument("--pow-sig", action="store_true",
                    help="전력 의존 지문 (13.84.38). `L_harm` 이 전력에 선형인 것을 고친다 — "
                         "와트당 고차가 동작점에 따라 47~109%% 변한다 (13.84.32). 순방향 잔차 −13~34%%")
    ap.add_argument("--bg-head", action="store_true",
                    help="기록(=세션)당 배경 전류 하나 (13.84.38). `noise_sig` 는 전역 상수 하나인데 "
                         "실측 배경은 파일 간 3.6~22.6mA 로 다르고 h15 에서 미니PC 보다 크다 (13.84.35)")
    ap.add_argument("--w-bg", type=float, default=3.0, metavar="W",
                    help="배경 항의 크기 벌점. 0 이면 자유 항이 되어 **슬랙**이 된다")
    ap.add_argument("--keep-last", type=int, default=0, metavar="N",
                    help="마지막 N epoch 의 체크포인트를 `results/<tag>_ep{N}.pt` 로 **따로** 남긴다 "
                         "(0=안 남김). 13.84.37 의 판정법(후반 epoch 을 **짝지어** 재기)은 판마다 "
                         "여러 epoch 이 있어야 하는데, 기본 저장은 한 파일에 덮어써서 작업이 끝나면 "
                         "마지막 epoch 하나만 남는다 — 그때는 도는 동안 매 epoch 내려받는 수밖에 없었다. "
                         "판 4개 x 4 epoch 이 약 48MB 라 터널로 6분이다")
    ap.add_argument("--keep-on", type=float, default=0.0, metavar="F",
                    help="켜짐 BCE 를 얼마나 남길지 (0=통째로 뺌, 1=그대로 두고 CRF 를 더함). "
                         "처음부터 배우는 판은 0 이면 게이트가 무감독이 된다 (13.84.27)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--score-norm", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-real", action="store_true",
                    help="실측 채점을 건너뛴다 — 실측 npz 가 없는 곳에서 (13.84.27). "
                         "대신 **합성 홀드아웃**으로 사슬 대 창별을 잰다. 실측은 체크포인트를 "
                         "받아 로컬에서 채점한다 (실측은 원래 학습에 안 들어간다)")
    ap.add_argument("--holdout-records", type=int, default=300,
                    help="--no-real 일 때 채점에 쓸 홀드아웃 기록 수")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="seq_v1")
    ap.add_argument("--freeze-trunk", action="store_true", help="몸통을 얼린다 (대조군)")
    ap.add_argument("--no-state-init", action="store_true",
                    help="상태 전력 슬롯을 **옛 방식**(전부 0)으로 초기화한다 (13.84.70 ①의 A/B용). 기본은 `S_STATE` 의 잰 값에서 출발한다. ⚠ `--init scratch` 에서만 뜻이 있다 — 물려받는 판은 가중치가 곧바로 덮어쓴다")
    # ── 계획 A: 합 정합성 (14.3) ──────────────────────────────────────────
    ap.add_argument("--proj", type=float, default=0.0, metavar="A",
                    help="합 정합성 **사영 강도** [0,1] (계획 A, 14.3). 0 = 지금과 정확히 같다. "
                         "설명 안 된 전력 r 을 켜진 기기에 에너지 비례로 나눠 싣는다. "
                         "**매개변수를 안 늘린다** — 순수 재매개화라 바뀐 것이 구조 하나다. "
                         "⚠ 이 모드는 **창별 곱셈 재조정과 같아서 크기만 고치고 배분은 못 "
                         "옮긴다** (13.87 [2]). 배분을 옮기려면 배운 책임 팔이 따로 필요하다. "
                         "12.12.2 의 벌점판(w_cons 0.05)이 붕괴한 것과 다른 점은 "
                         "`L_power` 가 **사영된 출력**을 채점한다는 것이다 (net._project 독스트링)")
    ap.add_argument("--proj-cap", type=float, default=0.5, metavar="C",
                    help="|r| 을 관측 전력의 이 비율로 자른다. 순방향 모형 오차가 큰 창이 "
                         "배분을 독식하지 않게 한다 (13.84.23 이 잰 SMPS 잔차 17~27%%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="캐시·모델·손실·옵티마이저까지 **조립만** 하고 끝낸다 (13.88). "
                         "sbatch 가 캐시 뒤에 부르는 연기 시험이다 — 974261 이 옵티마이저 "
                         "조립에서 죽는 데 4분 걸렸다")
    ap.add_argument("--appl-attn", type=int, default=0, metavar="D",
                    help="**기기 축 자기어텐션** 토큰 차원 (13.93). 0 이면 완전히 꺼진다. "
                         "기기 9개를 토큰으로 놓고 머리들이 서로 보게 한다 — 지금은 "
                         "`crf_nll` 이 적은 대로 **기기 축이 서로 독립**이고, FHMM 이 "
                         "구조로 갖는 결합이 우리 구조 어디에도 없다 (13.84.74 [7] · 13.86). "
                         "⚠ 시간 축이 아니다 — 시간 문맥은 12.8·12.44 가 세 번 반증했다. "
                         "토큰이 9개뿐이라 자료 요구가 거의 없다 (권장 64)")
    ap.add_argument("--appl-attn-heads", type=int, default=4, metavar="N")
    ap.add_argument("--proj-resp", default="power", choices=("power", "head"),
                    help="책임 가중 (13.87 [5]). "
                         "`power` = A-크기 팔: 매개변수 0개인데 **창별 곱셈 재조정과 같아** "
                         "크기만 고치고 배분을 못 옮긴다. "
                         "`head` = **A-배분 팔**: `z` 에서 책임을 배운다 — 배분을 옮길 수 "
                         "있는 유일한 모드다. `Linear(h,K)` 를 0 으로 초기화해 **첫 스텝이 "
                         "에너지 비례와 같은 자리**에서 출발한다. ⚠ 새 키를 만든다")
    ap.add_argument("--proj-floor", type=float, default=5.0, metavar="W",
                    help="책임 분모의 하한 (W). **이것이 로버스트 슬랙이다** — 모델이 "
                         "아무도 안 켜졌다고 보면 아무에게도 안 싣는다 (검출 문제를 배분으로 "
                         "안 푼다). AFAMAP 의 로버스트 성분에 해당")
    ap.add_argument("--w-cons", type=float, default=0.0, metavar="W",
                    help="재구성 **벌점** `|Σ P̂ + Σ Ŝ + 잡음 − P_관측|` 의 가중 (3.3절). "
                         "⚠ 지금까지 시퀀스 학습기는 이것을 **0.0 으로 박아 놨다** — "
                         "잔차를 손실에 넣은 적이 한 번도 없다 (13.84.74 [2]). "
                         "⚠ 12.12.2 에서 0.05 로 붕괴한 항이다 (포트 편향 −879W). "
                         "`--proj` 와 **다른 것**이다: 이쪽은 경쟁하는 벌점, 저쪽은 재매개화. "
                         "귀속을 가르려면 따로 켜서 잰다. "
                         "⚠ `--proj 1.0` 이면 이 항이 **정확히 0 이 되어 무효**다 (측정 확인). "
                         "둘을 같이 재려면 proj<1 이어야 한다")
    ap.add_argument("--w-state-power", type=float, default=0.0, metavar="W",
                    help="상태별 전력 출력 `p_states[참상태]` 를 참 전력에 **직접** 묶는다 "
                         "(13.84.68). 기본 0 = 옛 동작. 전력은 `Σ_s mix[s]·p_states[s]` 인 "
                         "곱이라 혼합이 안 고르는 슬롯은 기울기가 0 이고, 음의 포화로 가면 "
                         "**못 살아난다** — 드라이기 약풍이 그렇게 죽었다. 이 항은 혼합과 "
                         "무관하게 슬롯마다 기울기를 준다. "
                         "⚠ 12.35 가 이 항으로 유령이 42.1 -> 86.9W 가 됐다고 쟀다 (옛 판). "
                         "관문 `src/run_gate_states.py` 로 재고 켜라")
    ap.add_argument("--no-crf-cpu", action="store_true",
                    help="CRF 를 GPU 에서 센다 (옛 동작). 기본은 **CPU** 다 — T 루프가 "
                         "54개짜리 텐서에 커널을 수백 번 날려 배치 크기와 무관하게 30ms 를 "
                         "먹는다 (13.84.63). CPU 가 4~4.8배 빠르고 값은 같다")
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--drift-proj", default="",
                    help="고정 표류 사영을 L_harm 에 굽는다 (13.84.52 ⓐ). "
                         "**13.84.60 에서 측정하고 기각했다 — 켜지 마라.** 표류 PC1 이 "
                         "미니PC-형제 판별축과 24~25° 라 표류와 함께 판별력이 간다. "
                         "배분 모의에서 미니PC 오차 산포가 4.11 -> 7.87W 로 두 배가 됐다. "
                         "재현용으로만 남긴다 (`src/run_gate_drift.py`)")
    ap.add_argument("--drift-k", type=int, default=1,
                    help="버릴 차원 수. 위 경고를 먼저 읽어라")
    ap.add_argument("--w-z", type=float, default=0.3)
    ap.add_argument("--w-over", type=float, default=0.0)
    ap.add_argument("--no-window-loss", action="store_true",
                    help="CRF 만 쓴다 (절제용). 기본은 창 단위 항을 다 쓰고 켜짐만 CRF 로 바꾼다")
    a = ap.parse_args()

    c = Path(a.cache)
    meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    K = len(apps)
    N = int(meta["record_s"] * FS)
    grid = np.arange(meta["target_offset"], N - PRE * FS - 1, int(meta["grid_s"] * FS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # `--init scratch` 는 **v37 과 같은 구조·같은 가림에 무작위 가중치**다 (13.84.27).
    # 구조는 체크포인트에서 읽되 가중치만 새로 뽑아, 바뀐 것이 초기점 하나가 되게 한다.
    scratch = str(a.init).lower() in ("scratch", "random", "none")
    if scratch:
        torch.manual_seed(a.seed)
    if a.no_mask and not scratch:
        raise SystemExit("--no-mask 는 --init scratch 에서만 쓴다 (물려받은 가중치엔 본 적 없는 입력이다)")
    model, apps_m = load_model(a.ref if scratch else a.init, dev,
                               weights=not scratch, mask=not a.no_mask,
                               state_power_init=not a.no_state_init,
                              # 전압 지수는 **이 판의 인수**가 정한다 (14.7). 체크포인트 값을 덮는다.
                              proj_from={"vexp": bool(a.vexp)})[:2]
    assert list(apps_m) == list(apps), "기기 열 순서가 캐시와 다르다"
    # 사영은 **매개변수가 아니라 상수**다 — `load_state_dict` 뒤에 꽂아도 안전하고,
    # `proj=0` 이면 `forward` 가 `_project` 를 아예 안 부른다 (완전한 하위호환).
    # ⚠ 여기서는 `proj_resp="power"` 만 쓴다. `head` 는 새 키를 만들어 "바뀐 것이
    #   구조 하나" 라는 조건을 깬다 — 그 팔은 따로 세울 것.
    model.proj = float(a.proj)
    model.proj_cap = float(a.proj_cap)
    model.proj_floor = float(a.proj_floor)
    model.proj_resp = str(a.proj_resp)
    # 몸통에 **새로 붙는** 모듈들의 파라미터. 사영 머리와 기기 축 어텐션이 여기 모인다 —
    # 둘은 서로 독립이고 (어텐션은 머리 **앞**에서 z 를 보정, 사영은 머리 **뒤**에서
    # 전력 합을 맞춘다) 같이 켤 수도 따로 켤 수도 있다. 무리를 따로 두는 이유는
    # `model.parameters()` 와 **겹치면 AdamW 가 거부하기 때문**이다 (13.88).
    new_params = []
    if a.appl_attn:
        # `load_model` 은 `ref` 로 짓는다 — 어텐션이 없다. **`load_state_dict` 뒤에** 붙인다.
        # `attn_out` 이 0 이라 첫 스텝이 어텐션 없는 판과 **정확히 같은 자리**다.
        import torch.nn as _nn
        _h = model.trunk[-2].out_features
        _d = int(a.appl_attn)
        model.appl_attn = _d
        model.attn_in = _nn.Linear(_h, _d).to(dev)
        model.attn_tok = _nn.Parameter(torch.randn(len(apps), _d, device=dev) * 0.02)
        model.attn = _nn.MultiheadAttention(_d, int(a.appl_attn_heads),
                                            batch_first=True).to(dev)
        model.attn_out = _nn.Linear(_d, _h).to(dev)
        _nn.init.zeros_(model.attn_out.weight)
        _nn.init.zeros_(model.attn_out.bias)
        new_params += ([model.attn_tok] + list(model.attn_in.parameters())
                        + list(model.attn.parameters()) + list(model.attn_out.parameters()))
        print("[seq] **기기 축 어텐션** d=%d · 머리 %d · 새 파라미터 %d개 (13.93)"
              % (_d, a.appl_attn_heads, sum(q.numel() for q in new_params)), flush=True)
    if a.proj_resp == "head":
        if a.proj <= 0:
            raise SystemExit("--proj-resp head 인데 --proj 가 0 이다 — 사영이 안 돈다")
        import torch.nn as _nn
        # `load_model` 은 `ref`(cnn_v37)로 지으므로 이 머리가 없다. **`load_state_dict`
        # 가 끝난 뒤에** 붙인다 — 0 으로 시작하므로 첫 스텝이 에너지 비례와 같은 자리다.
        h = model.trunk[-2].out_features
        model.proj_head = _nn.Linear(h, len(apps)).to(dev)
        _nn.init.zeros_(model.proj_head.weight)
        _nn.init.zeros_(model.proj_head.bias)
        new_params += list(model.proj_head.parameters())
    if a.proj > 0:
        print("[seq] **합 정합성 사영** 강도 %.2f · 책임 %s · 자르기 %.2f·P · 분모하한 %.1fW"
              % (a.proj, a.proj_resp, a.proj_cap, a.proj_floor), flush=True)
        if a.proj_resp == "power":
            print("      ⚠ power 모드는 **창별 곱셈 재조정**이라 배분을 못 옮긴다 (13.87 [2])",
                  flush=True)
    # ⚠ **물려받은 판의 죽은 슬롯은 여기서 못 고친다** (13.84.70).
    #   `p_states = softplus(W·z + b)` 이고 바이어스는 어느 판이나 0 근처다 — 값을 내는
    #   것은 **가중치**다. 그래서 "바이어스가 작으면 죽은 것" 이라는 판정은 틀린다:
    #   cnn_v37 에서 그 기준을 걸면 **잘 도는 슬롯 16개**가 걸린다 (드라이기 s2 는
    #   바이어스 1.21 인데 출력이 976W 다). 죽었는지는 **자료를 통과시켜 봐야** 안다.
    #   지금 고침은 **새 판에만** 듣는다 (`--init scratch` + `state_power_init`).
    #   물려받아 고치려면 자료로 슬롯 출력을 재서 세우는 별도 절차가 필요하다 — 미착수.
    with torch.no_grad():
        zdim = int(model(torch.zeros(1, model.fine_channels, 600, device=dev),
                         torch.zeros(1, 47, 120, device=dev))["z"].shape[1])
    torch.manual_seed(a.seed)
    heads = ChainHeads(zdim, N_FEAT + 2, K, hidden=a.hidden, score_norm=a.score_norm).to(dev)
    bghead = BgHead(zdim).to(dev) if a.bg_head else None

    n_rec = meta["records"]
    n_ho = max(1, int(n_rec * a.holdout_frac))
    idx = np.random.RandomState(0).permutation(n_rec)
    tr_i = idx[n_ho:]
    ho_i = idx[:n_ho]
    ds = SeqChunks(a.cache, tr_i, a.chunk, meta["target_offset"], meta["window_cycles"], grid)
    # `pin_memory` 가 없으면 H2D 가 **동기**이고 중간 버퍼를 한 번 더 거친다 (13.84.63 ③).
    # 배치 하나가 62MB 라 그 복사가 주 루프 CPU 의 약 10ms 다.
    dl = DataLoader(ds, batch_size=a.records_per_batch, shuffle=True,
                    num_workers=a.workers, collate_fn=collate, drop_last=True,
                    pin_memory=(dev != "cpu"),
                    persistent_workers=a.workers > 0)
    print("[seq] 기록 %d (학습 %d) · 격자 %d단계 · 조각 %d · z %d · 기기 %d · %s%s"
          % (n_rec, len(tr_i), len(grid), a.chunk, zdim, K, dev,
             (" · 몸통 얼림" if a.freeze_trunk else "")
             + (" · **처음부터**(무작위 몸통)" if scratch else " · 초기점 " + Path(a.init).stem)
             + (" · vswap %.2f" % a.vswap_p if a.vswap_p else "")
             + (" · 가림 해제" if a.no_mask else "")
             + (" · keep-on %.2f" % a.keep_on if a.keep_on else "")
             + (" · 상태슬롯 옛초기화" if a.no_state_init else "")
             + (" · state_power %.2f" % a.w_state_power if a.w_state_power else "")
             + (" · 전력의존지문" if a.pow_sig else "")
             + (" · 세션배경(벌점 %.1f)" % a.w_bg if a.bg_head else "")), flush=True)

    if a.freeze_trunk:
        for p in model.parameters():
            p.requires_grad_(False)
    # ⚠⚠ **2026-09-13 고침 (13.99) — `list(...)` 를 빼면 사슬이 안 배운다.**
    #   아래 중복검사가 `for q in g["params"]` 로 무리를 훑는데, 여기가 **제너레이터**면
    #   그 훑기가 **소진시켜** AdamW 는 빈 무리를 받는다. 예외도 경고도 없다.
    #   증상: 몸통만 배우고 `ChainHeads` 는 12에포크 뒤에도 **초기값 그대로**.
    #   `seq_fix_*` 여섯 판이 전부 그렇게 나왔다 (갓 초기화와 차 7.45e-09).
    #   그러므로 **파라미터 무리는 반드시 리스트로 굳혀서 넣는다.**
    groups = [{"params": list(heads.parameters()), "lr": a.lr_heads}]
    if new_params:
        # ⚠⚠ `proj_head` 는 **`model` 의 하위 모듈**이라 `model.parameters()` 에 이미 있다.
        #   974261 이 여기서 죽었다: "some parameters appear in more than one parameter group".
        #   그래서 몸통 무리에서 **빼고** 따로 넣는다 — 새로 난 머리라 머리 학습률이 맞고,
        #   `--freeze-trunk` 여도 이 머리만은 배워야 한다 (안 그러면 0 에 박혀 `power` 와 같아진다).
        for q in new_params:
            q.requires_grad_(True)
        groups.append({"params": new_params, "lr": a.lr_heads})
    if bghead is not None:
        groups.append({"params": list(bghead.parameters()), "lr": a.lr_heads})
    if not a.freeze_trunk:
        _pid = {id(q) for q in new_params}
        groups.append({"params": [q for q in model.parameters() if id(q) not in _pid],
                       "lr": a.lr})
    # 같은 텐서가 두 무리에 들면 AdamW 가 거부한다 — 조립 직후에 우리가 먼저 잡는다.
    _seen = set()
    for g in groups:
        g["params"] = list(g["params"])          # 제너레이터면 여기서 굳는다 (13.99)
        if not g["params"]:
            raise SystemExit("파라미터 무리가 비었다 — 제너레이터를 이미 소진했다 (13.99)")
        for q in g["params"]:
            if id(q) in _seen:
                raise SystemExit("파라미터가 두 무리에 들어 있다 — 옵티마이저 조립이 틀렸다")
            _seen.add(id(q))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    # 조립 **뒤에** AdamW 가 실제로 쥔 것을 센다. 위 검사는 우리 `groups` 를 보는데
    # 이 줄은 `opt.param_groups` 를 본다 — 둘이 어긋나면 바로 여기서 걸린다.
    _want = sum(len(g["params"]) for g in groups)
    _got = sum(len(g["params"]) for g in opt.param_groups)
    if _got != _want:
        raise SystemExit("AdamW 가 쥔 텐서 %d != 넣은 텐서 %d" % (_got, _want))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epochs * (len(ds) // a.records_per_batch)))

    if a.dry_run:
        # **연기 시험** (13.88). 974261 이 옵티마이저 조립에서 죽었는데, 그때까지
        # 캐시 굽기 4분 + 관문을 다 지나고 나서였다. sbatch 가 캐시 뒤에 이것을 한 번
        # 부르면 같은 부류의 사고를 **몇 초 만에** 잡는다.
        # ⚠ **수를 찍기만 하면 시험이 아니다** (13.99). 13.88 판은 "텐서 56개" 를 찍었고
        #   그 56 이 **몸통만**이었는데 (사슬 머리 10개가 빠졌다) 나는 "통과" 로 읽었다.
        #   그래서 여기서 **기대치를 명시하고 대조한다.**
        _in_opt = {id(q) for g in opt.param_groups for q in g["params"]}
        _miss = {nm for nm, q in heads.named_parameters() if id(q) not in _in_opt}
        if _miss:
            raise SystemExit("사슬 머리가 옵티마이저에 없다: %s" % sorted(_miss))
        if not a.freeze_trunk:
            _mm = {nm for nm, q in model.named_parameters() if id(q) not in _in_opt}
            if _mm:
                raise SystemExit("몸통 일부가 옵티마이저에 없다: %s" % sorted(_mm)[:5])
        n = sum(len(g["params"]) for g in groups)
        print("[seq] 연기 시험 통과 — 무리 %d개 · 텐서 %d개 (머리 %d + 몸통 %d) · 사영 %s(%s)"
              % (len(groups), n, len(list(heads.parameters())),
                 len(list(model.parameters())), a.proj, a.proj_resp), flush=True)
        return 0

    crf_cpu = (not a.no_crf_cpu) and dev != "cpu"
    if crf_cpu:
        print("[seq] CRF 를 **CPU** 에서 센다 (13.84.63 — 값은 같고 4~4.8배 빠르다)", flush=True)
    crit = None
    if not a.no_window_loss:
        #: ── 14.171 — **1단계가 쓴 물리를 체크포인트에서 읽어 온다** ──────────
        #  여기 없던 18개 때문에 2단계가 1단계와 **다른 순방향 모형**으로 경사를 받고
        #  있었다. 살아 있던 갈라짐은 둘이다: `harm_sig_vnorm`(전압 크기 앵커) 과
        #  `cons_deadzone`. 플래그로 받으면 또 잊으므로 **체크포인트가 기준**이다.
        #  관문: `src/run_gate_lossparity.py`.
        _phys = _stage1_physics(a, apps)
        crit = build_loss(apps, dev, weights=LossWeights(
            harm=a.w_harm, cons=a.w_cons, over=a.w_over, z=a.w_z,
            state_power=a.w_state_power),
            power_signatures=a.pow_sig,
            drift_basis=a.drift_proj, drift_k=a.drift_k, **_phys)
        if _phys:
            print("[seq] 1단계 물리를 이어받았다: "
                  + " · ".join("%s=%s" % (k, (v if not hasattr(v, "shape") else "(K,)"))
                               for k, v in sorted(_phys.items())), flush=True)
        print("[seq] 창 단위 손실 조립 · 켜짐 가중 %.2f 를 CRF %.2f 로 갈아 끼운다"
              % (crit.w.on, a.w_crf), flush=True)

    real = None
    ho_dl = None
    if a.no_real:
        ho_ds = SeqChunks(a.cache, ho_i[:a.holdout_records], a.chunk, meta["target_offset"],
                          meta["window_cycles"], grid, fixed_start=0)
        ho_dl = DataLoader(ho_ds, batch_size=a.records_per_batch, shuffle=False,
                           num_workers=min(4, a.workers), collate_fn=collate)
        print("[seq] 실측 채점 **건너뜀** — 합성 홀드아웃 %d기록으로 잰다 (13.84.27)"
              % len(ho_ds), flush=True)
    else:
        real = real_windows(apps, meta["grid_s"], dev)
        print("[seq] 실측 준비 " + " · ".join("%s %d단계" % (s, len(d["t"]))
                                            for s, d in real.items()), flush=True)

    def score_holdout():
        """합성 홀드아웃에서 사슬 대 창별.

        ⚠ **실측 값과 직접 견주지 마라** — 여기는 전 기기·전 단계를 세고, 실측 채점은
        그 파일에 **있는 기기만** 세서 기기별로 평균한다. 규약이 다르다
        ([[match-the-scoring-convention-before-comparing]]). 두 판(A/B) 을 서로 견주는 데만 쓴다.
        """
        model.eval(); heads.eval()
        ca = wa = nn_ = 0.0
        with torch.no_grad():
            for bt in ho_dl:
                B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
                o = model(bt["fine"].to(dev).flatten(0, 1), bt["wide"].to(dev).flatten(0, 1))
                z = o["z"].reshape(B, T, -1)
                gl = o["on_logit"].reshape(B, T, K)
                em, on, off, ini = heads(z, bt["dfeat"].to(dev), gl)
                y = bt["y_on"].to(dev).bool()
                ca += float((viterbi(em, on, off, ini) == y).float().sum())
                wa += float(((gl > 0) == y).float().sum())
                nn_ += y.numel()
        model.train(); heads.train()
        return ca / nn_, wa / nn_, {}

    def score_real():
        model.eval(); heads.eval()
        accs, mp, base = [], {}, []
        with torch.no_grad():
            for stem, d in real.items():
                Z, GL = [], []
                for i in range(0, len(d["t"]), 512):
                    o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                              torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                    Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                z = torch.cat(Z)[None]
                gl = torch.cat(GL)[None]
                em, on, off, ini = heads(z, torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
                p = viterbi(em, on, off, ini)[0].cpu().numpy()
                w = (gl[0] > 0).cpu().numpy()
                y = d["y"].astype(bool)
                for k in np.nonzero(d["present"])[0]:
                    accs.append(float((p[:, k] == y[:, k]).mean()))
                    base.append(float((w[:, k] == y[:, k]).mean()))
                    mp.setdefault(stem, {})[k] = (base[-1], accs[-1])
        model.train(); heads.train()
        return float(np.mean(accs)), float(np.mean(base)), mp

    def save(ep, r, b):
        """⚠ **epoch 마다** 덮어쓴다 (13.84.27). 시간 한도에 걸려 죽어도 거기까지는 건진다 —
        SLURM 이 FIFO 라 짧은 시간을 걸어야 backfill 에 끼는데, 그러면 초과 위험이 생긴다."""
        torch.save({"model": model.state_dict(), "heads": heads.state_dict(),
                    "appliances": apps, "meta": meta, "hidden": a.hidden,
                    "score_norm": a.score_norm, "init": a.init, "ref": a.ref,
                    "vswap_p": a.vswap_p, "no_mask": a.no_mask, "keep_on": a.keep_on,
                    "pow_sig": a.pow_sig, "bg_head": a.bg_head, "w_bg": a.w_bg,
                    # 13.84.68 — 체크포인트가 **자기 설정을 담아야** 나중에 견줄 수 있다
                    "w_state_power": a.w_state_power,
                    "no_state_init": a.no_state_init,
                    # 계획 A (14.3). `run_gate_check.load_model` 이 이 키를 읽는다.
                    "proj": a.proj, "proj_cap": a.proj_cap,
                    "proj_floor": a.proj_floor, "proj_resp": a.proj_resp,
                    # 13.93 — 어텐션 설정도 체크포인트가 들고 있어야 채점기가 짓는다
                    "appl_attn": a.appl_attn, "appl_attn_heads": a.appl_attn_heads,
                    "w_cons": a.w_cons,
                    "bghead": (bghead.state_dict() if bghead is not None else None),
                "epoch": ep, "epochs": a.epochs,
                    "holdout_chain": r, "holdout_window": b,
                    "zdim": zdim, "ddim": N_FEAT + 2}, "results/%s.pt" % a.tag)
        print("     저장 results/%s.pt (epoch %d)" % (a.tag, ep), flush=True)
        # ⚠ 후반 N epoch 은 **따로** 남긴다. 덮어쓰기만 하면 짝지은 epoch 비교가 불가능해진다
        #   (13.84.37 은 작업이 도는 동안 매 epoch 내려받아서 겨우 쟀다).
        #   시간 한도에 걸려 죽어도 그 앞 epoch 들은 이미 파일로 남아 있다.
        if a.keep_last > 0 and ep > a.epochs - a.keep_last:
            shutil.copyfile("results/%s.pt" % a.tag, "results/%s_ep%d.pt" % (a.tag, ep))
            print("     남김 results/%s_ep%d.pt" % (a.tag, ep), flush=True)

    score = score_holdout if a.no_real else score_real
    r, b, _ = score()
    print("  epoch  0 (학습 전)  %s 사슬 %.4f / 창별 %.4f"
          % ("홀드아웃" if a.no_real else "실측", r, b), flush=True)
    t0 = time.time()
    nwin = a.records_per_batch * a.chunk
    for ep in range(1, a.epochs + 1):
        tot, nb = 0.0, 0
        te = time.time()
        for bt in dl:
            B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
            fine = bt["fine"].to(dev, non_blocking=True).flatten(0, 1)
            wide = bt["wide"].to(dev, non_blocking=True).flatten(0, 1)
            vswap(fine, wide, a.vswap_p)          # 13.84.11 — 0 이면 아무것도 안 한다
            o = model(fine, wide)
            z = o["z"].reshape(B, T, -1)
            em, on, off, ini = heads(z, bt["dfeat"].to(dev, non_blocking=True),
                                     o["on_logit"].reshape(B, T, K))
            # ⚠ `y_on` 은 CRF 가 CPU 면 **안 올린다** (올렸다 내리면 동기점이 둘이 된다)
            crf = crf_nll(em, on, off,
                          bt["y_on"].bool() if crf_cpu else bt["y_on"].to(dev).bool(),
                          ini, on_cpu=crf_cpu)
            # 세션 배경 — 조각 하나당 값 **하나**를 내고 그 안의 모든 창에 같이 건다 (13.84.38).
            bg_off = None
            if bghead is not None:
                bg = bghead(z)                                          # (B,H,2)
                bg_off = bg[:, None].expand(B, T, -1, -1).flatten(0, 1)
            if crit is None:
                loss = a.w_crf * crf
            else:
                # 창 단위 항은 그대로 쓰고 **켜짐 BCE 만** CRF 로 갈아 끼운다.
                tg = {"y_power": bt["y_power"].to(dev).flatten(0, 1),
                      "y_on": bt["y_on"].to(dev).float().flatten(0, 1),
                      "y_plugged": bt["y_plugged"].to(dev).float().flatten(0, 1),
                      "y_standby": bt["y_standby"].to(dev).flatten(0, 1),
                      "y_state": bt["y_state"].to(dev).long().flatten(0, 1),
                      "obs_harm": bt["obs_harm"].to(dev).flatten(0, 1),
                      "p_noise": bt["p_noise"].to(dev).flatten(0, 1),
                      "p_observed": bt["p_observed"].to(dev).flatten(0, 1),
                      "harm_offset": bg_off,
                      "log_z": torch.log(bt["z_grid"].to(dev)[..., 0].clamp(min=1e-3)).flatten(0, 1)}
                parts = crit(o, tg)
                # ⚠ `--keep-on` 이 0 이면 켜짐 BCE 를 **통째로** 뺀다 — v37 에서 이어 배울 때는
                #   그 감독을 이미 받은 몸통이라 괜찮지만, **처음부터 배우면 게이트가 한 번도
                #   감독을 못 받는다** (13.84.27). 그러면 방출이 base 와 emit 의 **큰 값 상쇄**로
                #   풀려 실측에서 부서진다 (항 규모 ±0.6 대 v37 판 ±0.05, 잔차는 둘 다 −0.05~−0.09).
                #   일부를 남기면 한 판으로 끝난다 — 2단계 학습을 안 해도 된다.
                loss = (parts["total"] - (1.0 - a.keep_on) * crit.w.on * parts["on"]
                        + a.w_crf * crf)
                if bghead is not None:
                    # ⚠ 크기 벌점이 없으면 자유 항이 되어 기기에서 전류를 빼앗는다.
                    loss = loss + a.w_bg * (bg / crit.harm_scale[None, :, None]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(heads.parameters()) + list(model.parameters())
                + (list(bghead.parameters()) if bghead is not None else []), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        el = time.time() - te
        if ep % a.eval_every == 0 or ep == a.epochs:
            r, b, mp = score()
            print("  epoch %2d  손실 %8.4f · %.0f초/epoch (%.0f창/초) · %s 사슬 %.4f / 창별 %.4f · 남은 %.0f분"
                  % (ep, tot / max(nb, 1), el, nb * nwin / max(el, 1e-6),
                     "홀드아웃" if a.no_real else "실측", r, b,
                     (a.epochs - ep) * el / 60), flush=True)
            save(ep, r, b)
        else:
            print("  epoch %2d  손실 %8.4f · %.0f초/epoch (%.0f창/초) · 남은 %.0f분"
                  % (ep, tot / max(nb, 1), el, nb * nwin / max(el, 1e-6),
                     (a.epochs - ep) * el / 60), flush=True)
    r, b, mp = score()
    km = apps.index("minipc")
    if mp:                                    # --no-real 이면 실측 표가 없다
        print("\n미니PC 시간 정확도 (실측)")
        print("  파일      창별   사슬")
        for stem in FILES:
            if km in mp.get(stem, {}):
                print("  %-8s %6.3f %6.3f" % (stem, mp[stem][km][0], mp[stem][km][1]))
    save(a.epochs, r, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
