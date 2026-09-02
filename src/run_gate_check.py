"""
게이트 진단 — 소프트 게이트가 물리적으로 불가능한 전력을 내고 있는가 (12.9.14절)
==================================================================================
`net.py` 의 출력은 `power = sigmoid(on_logit) * p_raw` 다. 이 형태는 정격 1233W 인
전기포트에서 `sigma=0.381` 로 **469W** 를 내는 것을 허용한다. 그런 물리적 상태는 없다.

12.9.14 가 실측 실패를 이렇게 분해했다:

    전기포트    p_raw 1233W (정격 1290W)  x  sigmoid(on) 0.381  ->  469W
    핫플레이트  p_raw  397W (정격  468W)  x  sigmoid(on) 0.004  ->    2W

**크기는 둘 다 정확히 안다. 어느 쪽인지 결정을 못 한다.** 12.12.3 의 처방(헤지 벌점)은
손실 항이라 적응한 창에서만 듣는다 — 12.14.2 가 그것을 확인했다 (홀드아웃 핫플 F1 은
`w_hedge` 와 무관하게 0.63).

여기서는 **학습하지 않는다.** 저장된 가중치를 그대로 두고 추론에서만 게이트를
이진화해(`1[sigma>0.5] * p_raw`) 무엇이 바뀌는지 본다. 구조를 바꿔 재학습할
가치가 있는지 5분에 판정하기 위한 것이다.

    python -m src.run_gate_check --ckpt results/hedge_0.2.pt

[돌리기 전 예측 - 12.9.5 의 규율]
소프트 게이트로 학습된 모델이므로, 이진화하면 469W 짜리 유령 포트는 사라지지만
**핫플이 그것을 받아가지는 못하고 잔차만 커질** 것이다. 그렇게 나오면 "표현 공간이
오답을 허용한다" 는 진단은 맞고 해법은 재학습이다. 만약 핫플이 실제로 받아가면
재학습 없이 끝난다.
"""
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import (SESSION_MERGE_CYCLES, load_events,
                                        score_absent, score_on_off)
from src.evaluation.sealing import is_sealed
from src.model.inputs import (FINE_CYCLES, LEGACY_FINE_CHANNELS, ZERO_EVEN_HARMONICS,
                              LEGACY_FINE_CYCLES, LEGACY_TARGET_LOOKAHEAD,
                              TARGET_LOOKAHEAD)
from src.model.net import NILMNet, appliance_state_counts
from src.model.realdata import dense_targets, upsample_to_cycles
from src.run_baseline import S_I

# 헤지로 판정한다 - 이 밖이면 "결정했다", 안이면 "망설였다"
HEDGE_LO, HEDGE_HI = 0.05, 0.95


def assert_target_config(ck: dict, ckpt_path: str) -> None:
    """체크포인트가 학습된 타깃 시점 구성이 현재 코드와 같은가 (12.45.3).

    **세밀 채널 수와 성격이 다르다.** 채널은 뒤에 붙는 규약이라 슬라이스로 맞출 수
    있지만, 타깃 시점이 어긋나면 입력이 가리키는 순간과 체크포인트가 배운 순간이
    달라진다. 맞출 방법이 없고 조용히 틀린다 — 그래서 막는다.

    2026-08-24: 이 검사가 없어서 룩어헤드를 9초로 올린 채로 6초 체크포인트를
    채점할 뻔했다. 같은 부류의 결함이 세 번째다 (11.2절, 12.45.3).
    """
    # 짝수차 배제 (12.77). **이것이 어긋나면 조용히 틀린다** — 12.74 에서 지터 없이
    # 학습한 모델의 짝수차를 추론에서만 껐더니 충전기가 0.937 -> 0.868 로 무너졌다.
    # 기본값 False 는 이 필드가 없던 시절의 체크포인트가 전부 짝수차를 쓰기 때문이다.
    ck_ze = bool(ck.get("zero_even_harmonics", False))
    if ck_ze != ZERO_EVEN_HARMONICS:
        raise SystemExit(
            f"체크포인트의 짝수차 구성이 현재 코드와 다릅니다: {ckpt_path}" + chr(10)
            + f"  체크포인트  zero_even_harmonics={ck_ze}" + chr(10)
            + f"  현재 코드    ZERO_EVEN_HARMONICS={ZERO_EVEN_HARMONICS}" + chr(10)
            + "  src/model/inputs.py 의 ZERO_EVEN_HARMONICS 를 체크포인트 값으로"
            + " 맞춘 뒤 다시 실행하십시오 (그 값으로 만든 캐시도 함께 써야 합니다).")

    got = (int(ck.get("target_lookahead", LEGACY_TARGET_LOOKAHEAD)),
           int(ck.get("fine_cycles", LEGACY_FINE_CYCLES)))
    want = (TARGET_LOOKAHEAD, FINE_CYCLES)
    if got != want:
        raise SystemExit(
            f"체크포인트의 타깃 시점 구성이 현재 코드와 다릅니다: {ckpt_path}" + chr(10)
            + f"  체크포인트  TARGET_LOOKAHEAD={got[0]} FINE_CYCLES={got[1]}" + chr(10)
            + f"  현재 코드    TARGET_LOOKAHEAD={want[0]} FINE_CYCLES={want[1]}" + chr(10)
            + "  src/model/inputs.py 를 체크포인트 값으로 맞춘 뒤 다시 실행하십시오."
            + chr(10) + "  (그 값으로 만든 캐시·홀드아웃도 함께 써야 합니다)")


def load_model(ckpt_path: str, dev: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert_target_config(ck, ckpt_path)
    apps = ck["appliances"]
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, apps, ck


#: 짝수 차수(2,4..14)의 Re/Im 채널과 |I2|/|I1|. 12.72 가 계측 인공물로 확정했다.
EVEN_CHANNELS = [1, 3, 5, 7, 9, 11, 13] + [16, 18, 20, 22, 24, 26, 28] + [35]


@torch.no_grad()
def forward_file(model, stem: str, dev: str, stride: int = 30,
                 zero_ch: Optional[List[int]] = None) -> dict:
    """파일 하나를 촘촘히 훑어 게이트와 원시 전력을 그대로 돌려준다."""
    rw = dense_targets(stem, stride=stride)
    G, R, SB, PL, PN, POBS, OH = [], [], [], [], [], [], []
    for i in range(0, len(rw), 512):
        idx = np.arange(i, min(i + 512, len(rw)))
        f, w, pobs, oh, pn = rw.batch(idx)
        if zero_ch:
            f = f.copy()
            f[:, [c for c in zero_ch if c < f.shape[1]]] = 0.0
        ft = torch.from_numpy(np.ascontiguousarray(f)).to(dev)
        wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(ft, wt)
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        R.append(o["power_raw"].float().cpu().numpy())
        SB.append(o["standby"].float().cpu().numpy())
        # 대기 전류 항의 계수. `NILMLoss` 가 L_harm 을 만들 때 쓰는 것과 같은 식이라
        # 채점 쪽에서 그 손실을 재현할 수 있다 (12.139 의 유령 인과 분해).
        PL.append((torch.sigmoid(o["plugged_logit"])
                   * (1.0 - torch.sigmoid(o["on_logit"]))).float().cpu().numpy())
        PN.append(pn); POBS.append(pobs); OH.append(oh)
    return {"gate": np.concatenate(G), "p_raw": np.concatenate(R),
            "standby": np.concatenate(SB), "idle": np.concatenate(PL),
            "p_noise": np.concatenate(PN),
            "p_observed": np.concatenate(POBS), "targets": rw.target_cycle,
            # 단자 전압 (n,). 저항 정합(12.112)이 P = V^2/R 을 푸는 데 쓴다.
            "v_rms": rw.v_observed.astype(np.float64),
            # 관측 고조파 (n,15,2) Re/Im. 12.40 의 전이 스냅이 |I3| 를 쓴다 —
            # 저항 부하는 3차를 거의 안 흘려서 SMPS 계단만 남는다 (12.37.2).
            "obs_harm": np.concatenate(OH),
            # 관측 무효전력 (n,). 12.133 이 **두 번째 판별자**로 확정했다 —
            # SMPS 쌍 d′ 2.31~4.64 로 고조파(0.91~1.85)보다 2.2~2.5배 잘 가른다.
            "q_observed": rw.reactive(np.arange(len(rw))).astype(np.float64)}


def merge_smps(d: dict, d_smps: dict, apps: List[str]) -> dict:
    """SMPS 3종만 두 번째 체크포인트의 값으로 바꾼다 (`run_live --ckpt-smps` 와 동일).

    **운영에서 실제로 도는 것은 두 체크포인트의 조합이다** (12.31.5). 1단계 지표로
    문제를 고르면 안 된다는 것이 12.47·12.50·12.51.3 의 결론이므로, 채점도 운영
    조합으로 해야 한다. 12.52 의 네 조합 표를 만든 스크립트가 임시 폴더에 있다가
    사라져 재현이 끊겼었다 — 그래서 여기에 넣는다.
    """
    from src.run_live import SMPS_GROUP
    six = [apps.index(x) for x in SMPS_GROUP if x in apps]
    out = dict(d)
    for k in ("gate", "p_raw", "standby"):
        out[k] = d[k].copy()
        out[k][:, six] = d_smps[k][:, six]
    return out


def gated(d: dict, hard: bool) -> np.ndarray:
    g = (d["gate"] > 0.5).astype(np.float32) if hard else d["gate"]
    return g * d["p_raw"]


_SIG_CACHE: dict = {}


def _signatures(apps: List[str]):
    """와트당 고조파 지문 / 대기 지문 / 계측계 지문. 한 번만 만든다."""
    key = tuple(apps)
    if key not in _SIG_CACHE:
        from src.model.net import (harmonic_signatures, noise_signature,
                                   standby_signatures)
        from src.synthesis.segment_pool import SegmentPool
        pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        _SIG_CACHE[key] = (harmonic_signatures(pool, apps),
                           standby_signatures(pool, apps), noise_signature(pool))
    return _SIG_CACHE[key]


def score_one(d: dict, P: np.ndarray, stem: str, apps: List[str], ev: dict,
              session_merge=None) -> dict:
    resid = P.sum(1) + d["standby"].sum(1) + d["p_noise"] - d["p_observed"]
    n_cycles = int(ev[stem]["cycles"])
    on_c = upsample_to_cycles(d["gate"] > 0.5, d["targets"], n_cycles)
    ab = score_absent(P, stem, apps, pred_on=d["gate"] > 0.5, s_i=S_I, events=ev)
    f1 = score_on_off(on_c, stem, apps, events=ev, session_merge=session_merge)
    vals = [v["f1"] for v in f1.values() if v["n_true_on"] > 0]
    return {"absent_sum_w": ab["absent_sum_w"], "absent_fa_rel_max":
            max([v["fa_rel"] for v in ab["absent"].values()
                 if v["fa_rel"] == v["fa_rel"]] or [float("nan")]),
            "residual_abs_w": float(np.abs(resid).mean()),
            "on_off_f1_mean": float(np.mean(vals)) if vals else float("nan"),
            # 오귀속을 **기기별로도** 남긴다 (12.90). 합만 저장하던 탓에 12.89 의
            # 유령 회귀(8.67 -> 16.27W)가 어느 기기 때문인지 보려고 채점을 다시
            # 돌려야 했다. 원인은 SMPS 가 아니라 전기포트였다 - 합만 봐서는
            # "SMPS 레시피가 유령을 늘렸다" 로 잘못 읽힌다.
            "absent": ab["absent"],
            "per_app_f1": f1}


def hedge_report(d: dict, apps: List[str]) -> dict:
    """망설인 게이트가 만드는 전력이 얼마나 되는가."""
    g, p = d["gate"], d["p_raw"]
    soft = g * p
    hedging = (g > HEDGE_LO) & (g < HEDGE_HI)
    total = float(soft.sum())
    out = {"hedged_window_share": float(hedging.any(1).mean()),
           "hedged_power_share": float(soft[hedging].sum() / total) if total > 0 else 0.0}
    per = {}
    for j, a in enumerate(apps):
        m = hedging[:, j]
        if m.sum() == 0:
            continue
        per[a] = {"n": int(m.sum()), "share_of_windows": float(m.mean()),
                  "mean_gate": float(g[m, j].mean()),
                  "mean_emitted_w": float(soft[m, j].mean()),
                  "mean_rated_w": float(p[m, j].mean())}
    out["per_appliance"] = per
    return out


def oven_on_breakdown(d: dict, apps: List[str], ev: dict) -> dict:
    """`test_4` 의 오븐 히터 통전 구간에서 핫플/포트가 어떻게 갈리는가.

    12.9.5 가 "남은 실패 - 하나로 좁혀졌다" 고 한 바로 그 구간이다.
    """
    iv = ev["test_4"]["intervals"]
    n = int(ev["test_4"]["cycles"])

    def mask(pairs):
        m = np.zeros(n, bool)
        for s, e in pairs:
            m[int(s * 60):int(e * 60)] = True
        return m

    ov = mask(iv["oven"]["_heater_pulses"])[d["targets"]]
    hp = mask(iv["hotplate"]["on"])[d["targets"]]
    jh, jk, jo = (apps.index("hotplate"), apps.index("electiric_kettle"), apps.index("oven"))
    out = {}
    for name, m in (("오븐ON+핫플통전", ov & hp), ("오븐ON+핫플휴지", ov & ~hp),
                    ("오븐OFF+핫플통전", ~ov & hp)):
        if m.sum() < 5:
            continue
        out[name] = {
            "n": int(m.sum()),
            "hotplate_gate": float(np.median(d["gate"][m, jh])),
            "hotplate_soft_w": float(np.median(d["gate"][m, jh] * d["p_raw"][m, jh])),
            "hotplate_hard_w": float(np.median((d["gate"][m, jh] > 0.5) * d["p_raw"][m, jh])),
            "hotplate_recall": float((d["gate"][m, jh] > 0.5).mean()),
            "kettle_gate": float(np.median(d["gate"][m, jk])),
            "kettle_soft_w": float(np.median(d["gate"][m, jk] * d["p_raw"][m, jk])),
            "kettle_hard_w": float(np.median((d["gate"][m, jk] > 0.5) * d["p_raw"][m, jk])),
            "kettle_fa": float((d["gate"][m, jk] > 0.5).mean()),
            "oven_gate": float(np.median(d["gate"][m, jo])),
            "oven_soft_w": float(np.median(d["gate"][m, jo] * d["p_raw"][m, jo])),
            "p_observed": float(np.median(d["p_observed"][m])),
            "pred_total_w": float(np.median((d["gate"][m] * d["p_raw"][m]).sum(1))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="게이트 진단 — 소프트 vs 하드 (12.9.14절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_v15.pt", "results/hedge_0.2.pt"])
    ap.add_argument("--ckpt-smps", default=None, metavar="PT",
                    help="SMPS 3종(프로젝터/충전기/미니PC)만 이 체크포인트로 채점한다. "
                         "**운영 조합으로 재는 방법이다** — 12.47/12.51.3 참조")
    ap.add_argument("--zero-even", action="store_true",
                    help="SMPS 체크포인트 입력의 짝수차 채널을 0 으로 (12.74절). "
                         "짝수차는 계측 인공물이다 (12.72)")
    ap.add_argument("--zero-ch", default="", metavar="LIST",
                    help="SMPS 체크포인트 입력에서 0 으로 만들 세밀 채널 (쉼표 구분). "
                         "예: --zero-ch 33,34,47 (비율 채널, 12.80.3)")
    ap.add_argument("--postproc", default="off", choices=("off", "on", "sync"),
                    help="물리 전력 상한 후처리 (12.102절). 프로젝터가 상한(55W)을 넘는 "
                         "만큼을 다른 SMPS 로 넘긴다. sync 는 게이트도 맞춘다. "
                         "**2단계 단독에서만 이득이다** (44/59 vs 27/59). 하이브리드는 "
                         "41 -> 36/59 로 나빠지므로 기본은 꺼 둔다")
    ap.add_argument("--resmatch", type=float, default=0.0, metavar="TOL",
                    help="저항 부하 정합 후처리 (12.112절). 관측 전력·전압으로 등가저항을 "
                         "역산해 저항 조합을 **맞바꾼다** (개수는 안 바꾼다). 0.02 권장, 0=끔")
    ap.add_argument("--session-merge", action="store_true",
                    help="주기 부하(오븐)를 **세션 단위**로 잰다 (12.119). "
                         "예측과 정답 양쪽에 90초 공백 병합을 걸어 "
                         "라벨 granularity 차이를 없앤다. 기본 꺼짐")
    ap.add_argument("--rm-snap", action="store_true",
                    help="저항 정합: 조합이 이미 맞을 때도 전력을 V^2/R 로 스냅한다 "
                         "(12.117 의 A). 개수·신원 불변")
    ap.add_argument("--rm-gate-min", type=float, default=0.0, metavar="G",
                    help="저항 정합: 맞바꿈 후보의 최소 게이트 (12.117 의 B). "
                         "이미 켜진 기기는 문턱과 무관하게 남는다. 0 이면 무제한")
    # ── 프로젝터 전력 스냅 (SMPS_PLAN 4.3) ──────────────────────────────
    ap.add_argument("--snap", type=float, default=0.0, metavar="W",
                    help="프로젝터를 이 값으로 **양방향** 스냅한다 (0=끔). "
                         "격리 통전 중앙 47.4W (12.120.1). 상한 55W 의 clip 과 "
                         "다르다 - 낮게 붙은 것도 끌어올린다")
    ap.add_argument("--snap-oneway", action="store_true",
                    help="스냅을 한 방향으로만 (초과분만 넘긴다). 절제용 - "
                         "효과가 어느 쪽에서 오는지 가른다 (규칙 3)")
    ap.add_argument("--snap-share", default="gate", choices=("gate", "harm"),
                    help="넘긴 전력을 나누는 비중. gate=게이트 가중(지금과 같음), "
                         "harm=3차 이상 고조파 코사인(게이트 편향을 안 탄다)")
    ap.add_argument("--snap-no-redist", action="store_true",
                    help="스냅한 초과분을 남에게 안 준다 (총합 보존 포기). "
                         "12.122.4 의 반증이 재배분 탓인지 스냅 탓인지 가른다")
    ap.add_argument("--snap-min-gate", type=float, default=0.5, metavar="G",
                    help="이 게이트 아래면 손대지 않는다. 규칙 18 - 개수는 안 바꾼다")
    ap.add_argument("--squelch", type=float, default=0.0, metavar="TAU",
                    help="게이트 TAU 아래 기기의 전력을 0 으로 (12.149). 유령8 의 82%%가 "
                         "문턱 아래 누설이다. 0.5 는 채점·화면과 같은 문턱이라 "
                         "**자유 파라미터가 아니다** — 소프트 열이 하드 열과 같아진다. "
                         "값 둔감성은 0.02~0.5 쓸기로 본다")
    ap.add_argument("--absorb-norton", default="", metavar="NPZ",
                    help="흡수가 보는 잔차에 계통 임피던스 보정을 넣는다 (12.152). "
                         "손실은 이미 쓰는데 흡수는 안 썼다. 잔차 101.9 -> 60.5 mA")
    ap.add_argument("--absorb-mode", default="cos", choices=("cos", "nnls", "pq"),
                    help="배분 규칙 (12.152). cos=지문별 독립 코사인(현행), "
                         "nnls=셋을 같이 풀어 와트로 낸다, "
                         "pq=거기에 **무효전력 방정식**을 더한다 (12.153)")
    ap.add_argument("--absorb-limit", action="store_true",
                    help="고조파가 지지하는 만큼만 준다 (12.152). 남는 것은 잔차로 둔다")
    ap.add_argument("--absorb-wq", type=float, default=3.0, metavar="W",
                    help="pq 모드에서 무효전력 방정식의 가중 (12.153). "
                         "고조파 항 대비. 값 둔감성 검사용")
    ap.add_argument("--absorb-cap-scale", type=float, default=1.0, metavar="X",
                    help="흡수 천장(`ABSORB_CAP_W`)에 곱할 배수 (12.149.4). "
                         "값 둔감성 검사용. 1.0 = 격리 사이클 최대 그대로")
    ap.add_argument("--absorb", type=float, default=0.0, metavar="FRAC",
                    help="총전력 잔차를 고조파가 닮은 SMPS 에 흡수시킨다 (12.104절). "
                         "0.5 면 잔차 8.88 -> 7.35W. 0 이면 끔")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/gate_check.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    payload: Dict[str, dict] = {}

    zch = list(EVEN_CHANNELS) if a.zero_even else []
    if a.zero_ch:
        zch += [int(x) for x in a.zero_ch.split(",") if x.strip()]
    zch = sorted(set(zch)) or None

    model_smps = None
    if a.ckpt_smps:
        model_smps, apps_smps, ck_smps = load_model(a.ckpt_smps, dev)

    for ckpt in a.ckpt:
        model, apps, ck = load_model(ckpt, dev)
        tag = Path(ckpt).stem
        if model_smps is not None:
            if list(apps_smps) != list(apps):
                raise SystemExit("두 체크포인트의 가전 목록이 다릅니다")
            tag = (f"{tag}+{Path(a.ckpt_smps).stem}" + ("+zeroeven" if a.zero_even else "")
                   + (f"+z{a.zero_ch.replace(',', '_')}" if a.zero_ch else ""))
        if a.postproc != "off":
            tag = f"{tag}+pp{'sync' if a.postproc == 'sync' else ''}"
        if a.resmatch > 0:
            tag = f"{tag}+rm{a.resmatch:g}"
        if a.squelch > 0:
            tag = f"{tag}+sq{a.squelch:g}"
        if a.absorb > 0:
            tag = f"{tag}+ab{a.absorb:g}"
            if a.absorb_cap_scale != 1.0:
                tag = f"{tag}c{a.absorb_cap_scale:g}"
            if a.absorb_norton:
                tag = f"{tag}N"
            if a.absorb_mode != "cos":
                tag = f"{tag}{a.absorb_mode}"
                if a.absorb_mode == "pq" and a.absorb_wq != 3.0:
                    tag = f"{tag}w{a.absorb_wq:g}"
            if a.absorb_limit:
                tag = f"{tag}L"
        if a.snap > 0:
            tag = (f"{tag}+snap{a.snap:g}"
                   + ("1way" if a.snap_oneway else "")
                   + ("h" if a.snap_share == "harm" else "")
                   + ("nr" if a.snap_no_redist else ""))
        print("=" * 88)
        print(f"[{tag}] stage {ck.get('stage', 1)} | {', '.join(stems)}")
        if model_smps is not None:
            from src.run_live import SMPS_GROUP
            print(f"  SMPS 3종은 {a.ckpt_smps} (stage {ck_smps.get('stage', 1)}) "
                  f"에서 가져옵니다 -> {', '.join(SMPS_GROUP)}")
        print("=" * 88)

        per_file, rows = {}, {"soft": [], "hard": []}
        for stem in stems:
            d = forward_file(model, stem, dev, stride=a.stride)
            if model_smps is not None:
                d = merge_smps(d, forward_file(model_smps, stem, dev, stride=a.stride,
                                               zero_ch=zch), apps)
            d_soft, d_hard = d, d
            P_soft, P_hard = gated(d, False), gated(d, True)
            if a.squelch > 0:
                # **후처리 앞이다.** 상한/스냅/저항정합이 문턱 아래 유령을 실체로
                # 오인해 재배분하면 안 된다. 하드 열은 τ<=0.5 에서 항등이다.
                from src.model.postproc import squelch
                P_soft = squelch(P_soft, d["gate"], a.squelch)
                P_hard = squelch(P_hard, d["gate"], a.squelch)
            if a.postproc != "off":
                from src.model.postproc import apply_postproc
                sync = a.postproc == "sync"
                P_soft, g_soft = apply_postproc(P_soft, d["gate"], apps, gate_sync=sync)
                P_hard, g_hard = apply_postproc(P_hard, d["gate"], apps, gate_sync=sync)
                d_soft = dict(d, gate=g_soft)
                d_hard = dict(d, gate=g_hard)
            if a.snap > 0:
                # **상한 뒤, 저항 정합 앞.** 상한이 먼저 극단을 자르고 스냅이
                # 나머지를 격리값으로 맞춘다. 저항 정합은 저항 열만 만지므로
                # 순서가 무관하지만, 잔차 흡수는 스냅 결과를 봐야 한다.
                from src.model.postproc import snap_power
                sn = {"beam_projector": float(a.snap)}
                kw = dict(targets=sn, bidirectional=not a.snap_oneway,
                          share=a.snap_share, min_gate=a.snap_min_gate,
                          redistribute=not a.snap_no_redist)
                if a.snap_share == "harm":
                    kw["obs_harm"] = d["obs_harm"]
                    kw["sig"] = _signatures(apps)[0]
                P_soft, g_soft = snap_power(P_soft, d_soft["gate"], apps, **kw)
                P_hard, g_hard = snap_power(P_hard, d_hard["gate"], apps, **kw)
                d_soft, d_hard = dict(d, gate=g_soft), dict(d, gate=g_hard)
            if a.resmatch > 0:
                from src.model.postproc import resistive_match
                P_soft, g_soft = resistive_match(
                    P_soft, d_soft["gate"], apps, d["p_observed"], d["v_rms"],
                    d["standby"], d["p_noise"], obs_harm=d["obs_harm"], tol=a.resmatch,
                    snap=a.rm_snap, cand_gate_min=a.rm_gate_min)
                P_hard, g_hard = resistive_match(
                    P_hard, d_hard["gate"], apps, d["p_observed"], d["v_rms"],
                    d["standby"], d["p_noise"], obs_harm=d["obs_harm"], tol=a.resmatch,
                    snap=a.rm_snap, cand_gate_min=a.rm_gate_min)
                d_soft, d_hard = dict(d, gate=g_soft), dict(d, gate=g_hard)
            if a.absorb > 0:
                from src.model.postproc import absorb_residual
                sigs = _signatures(apps)
                # 스냅으로 못 박은 기기는 잔차를 안 받는다 (12.149.2).
                exc = ["beam_projector"] if a.snap > 0 else None
                from src.model.postproc import ABSORB_CAP_W
                cw = {k: v * a.absorb_cap_scale for k, v in ABSORB_CAP_W.items()}
                ho = None
                if a.absorb_norton:
                    from src.model.postproc import norton_offset
                    ho = norton_offset(d["obs_harm"], a.absorb_norton)
                if a.absorb_mode == "pq":
                    from src.model.net import noise_reactive, reactive_signatures
                    from src.synthesis.segment_pool import SegmentPool
                    _pl = SegmentPool(npz_dir="processed_data/npz", time_split="train")
                    _qp, _ = reactive_signatures(_pl, apps)
                    kw_pq = dict(qp=np.asarray(_qp, np.float64),
                                 noise_q=float(noise_reactive(_pl)),
                                 q_observed=d["q_observed"], w_q=a.absorb_wq)
                    del _pl
                else:
                    kw_pq = {}
                P_soft = absorb_residual(P_soft, d_soft["gate"], apps, d["standby"],
                                         d["p_noise"], d["p_observed"], d["obs_harm"],
                                         *sigs, frac=a.absorb, exclude=exc, caps=cw,
                                         harm_offset=ho, mode=a.absorb_mode,
                                         limit_by_harm=a.absorb_limit, **kw_pq)
                P_hard = absorb_residual(P_hard, d_hard["gate"], apps, d["standby"],
                                         d["p_noise"], d["p_observed"], d["obs_harm"],
                                         *sigs, frac=a.absorb, exclude=exc, caps=cw,
                                         harm_offset=ho, mode=a.absorb_mode,
                                         limit_by_harm=a.absorb_limit, **kw_pq)
            # ── 기기별 전력 오차 (12.122.6, 2026-09-01) ──────────────────
            # **참값을 아는 넷에 대해서만** 배분 오차를 잰다. 이게 없던 동안
            # 배분을 겨냥한 처방이 부작용(유령·잔차)으로만 평가받았다 —
            # 프로젝터 스냅이 전력 오차를 8.1 -> 0.5W 로 줄이는데도 '반증' 으로
            # 닫힐 뻔했다 (12.122.4 를 12.122.7 이 정정한다).
            from src.evaluation.power_ref import score_power_ref
            n_cyc = int(ev[stem]["cycles"])
            pw_soft = score_power_ref(
                upsample_to_cycles(P_soft, d["targets"], n_cyc),
                upsample_to_cycles(d_soft["gate"] > 0.5, d["targets"], n_cyc),
                stem, apps, events=ev,
                p_observed=upsample_to_cycles(d["p_observed"], d["targets"], n_cyc),
                v_rms=upsample_to_cycles(d["v_rms"], d["targets"], n_cyc))

            sm = SESSION_MERGE_CYCLES if a.session_merge else None
            s_soft = score_one(d_soft, P_soft, stem, apps, ev, session_merge=sm)
            s_soft["power_ref"] = pw_soft
            s_hard = score_one(d_hard, P_hard, stem, apps, ev, session_merge=sm)
            per_file[stem] = {"soft": s_soft, "hard": s_hard, "hedge": hedge_report(d, apps)}
            rows["soft"].append(s_soft); rows["hard"].append(s_hard)
            if stem == "test_4":
                per_file[stem]["oven_on"] = oven_on_breakdown(d, apps, ev)

        K = [("absent_sum_w", "오귀속W"), ("absent_fa_rel_max", "최악FA"),
             ("residual_abs_w", "잔차W"), ("on_off_f1_mean", "on/off F1")]
        print(f"\n  {'':10s}" + "".join(f"{l:>24s}" for _, l in K))
        print(f"  {'':10s}" + "".join(f"{'소프트':>11s}{'하드':>13s}" for _ in K))
        for stem in stems:
            c = per_file[stem]
            print(f"  {stem:10s}" + "".join(
                f"{c['soft'][k]:>11.3f}{c['hard'][k]:>13.3f}" for k, _ in K))
        print(f"  {'평균':10s}" + "".join(
            f"{np.mean([r[k] for r in rows['soft']]):>11.3f}"
            f"{np.mean([r[k] for r in rows['hard']]):>13.3f}" for k, _ in K))

        print("\n  [망설임] 게이트가 (0.05, 0.95) 안에 있는 창")
        for stem in stems:
            h = per_file[stem]["hedge"]
            print(f"    {stem:8s} 창의 {100*h['hedged_window_share']:5.1f}% 에서 발생, "
                  f"예측 전력의 {100*h['hedged_power_share']:5.1f}% 를 만든다")
            for app, v in sorted(h["per_appliance"].items(),
                                 key=lambda x: -x[1]["mean_emitted_w"] * x[1]["share_of_windows"])[:3]:
                print(f"        {app:18s} 창 {100*v['share_of_windows']:5.1f}%  "
                      f"게이트 {v['mean_gate']:.3f}  정격 {v['mean_rated_w']:7.1f}W "
                      f"-> {v['mean_emitted_w']:7.1f}W 를 낸다")

        ob = per_file.get("test_4", {}).get("oven_on")
        if ob:
            print("\n  [test_4 오븐 히터 통전 구간] — 12.9.5 가 지목한 남은 실패")
            print(f"    {'구간':18s}{'n':>6s}{'오븐':>16s}{'핫플':>16s}{'포트':>16s}"
                  f"{'예측합':>10s}{'관측P':>10s}")
            for name, v in ob.items():
                print(f"    {name:18s}{v['n']:>6d}"
                      f"{v['oven_gate']:>7.3f}{v['oven_soft_w']:>8.1f}W"
                      f"{v['hotplate_gate']:>7.3f}{v['hotplate_soft_w']:>8.1f}W"
                      f"{v['kettle_gate']:>7.3f}{v['kettle_soft_w']:>8.1f}W"
                      f"{v['pred_total_w']:>9.1f}W{v['p_observed']:>9.1f}W")
        payload[tag] = per_file
        print()

    # 규칙 33 — 이 표를 만든 명령을 산출물 안에 남긴다. 플래그가 태그에 안 붙는
    # 것들(`--rm-snap` 등)이 있어, 없으면 나중에 이 파일이 무엇인지 증명 못 한다.
    payload["_config"] = {"argv": sys.argv, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
