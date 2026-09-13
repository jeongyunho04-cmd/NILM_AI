# -*- coding: utf-8 -*-
"""사슬을 버리면 무엇을 얻고 잃나 — **같은 자로** 나란히 잰다 (2026-09-13).

사용자: *"이정도면 사슬 전이 구조를 버려야겠는데 혹시 사슬 전이 구조를 도입하기
전으로 롤백할수는 없어?"*

되돌릴 수는 있다 (`results/cnn_v37.pt` 가 사슬 이전 판이다). 그런데 지금까지
두 계열을 **다른 채점 경로**로 재 왔다 — cnn 은 `run_gate_proj` 의 훑기로,
사슬은 `run_score_seq` 로. 규약이 어긋나면 비교가 안 선다
([[match-the-scoring-convention-before-comparing]]).

여기서는 **네 판 전부 `run_baseline_nilmtk.score_arm` 하나로** 잰다.
사슬 판은 Viterbi 경로를, cnn 판은 창별 게이트를 on/off 로 쓰고, 전력은 둘 다
모델이 낸 `power` 그대로다.

    python -X utf8 src/run_diag_rollback.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_baseline_nilmtk import score_arm
from src.run_score_seq import DUTY_APPS
from src.run_train_seq import real_windows

DEFAULT = ["results/cnn_v37.pt", "results/seq_fix_ctl.pt",
           "results/seq_fix_attn.pt", "results/seq_fix_alloc.pt"]


def _avg(acc, duty):
    """기기별 정확도 평균 — `duty=True` 면 듀티 기기만, 아니면 나머지만.

    ⚠ **갈라서 재는 이유**는 `DUTY_APPS` 주석에 있다. 아홉을 뭉치면 둘이 평균을
      끌고, 그 평균으로 판정하면 라벨 규약 차이를 모델 성능으로 읽는다.
    """
    v = [np.mean(x) for k, x in acc.items() if (k in DUTY_APPS) == duty]
    return float(np.mean(v)) if v else float("nan")


def _rp(cf, app):
    """'재현/정밀' 문자열. 듀티 기기는 **재현만 낮고 정밀은 높아야** 정상이다."""
    if app not in cf:
        return "-"
    tp, fp, fn, _ = cf[app]
    if tp + fn == 0 or tp + fp == 0:
        return "-"
    return "%.3f / %.3f" % (tp / (tp + fn), tp / (tp + fp))


def predict(path, cache, dev, no_chain=False):
    """(on (T,K) bool, power (T,K))  파일별. 사슬이면 Viterbi, 아니면 창별 게이트.

    `no_chain=True` 면 **사슬 체크포인트라도 창별 게이트로** 판정한다 — 이것이
    재학습 없는 롤백이다. 몸통은 그 체크포인트가 배운 그대로(v36p 생성기 · 고친
    `S_STATE`)이고, 얹혀 있던 `ChainHeads` 만 안 쓴다.

    ⚠ **사슬 없이 처음부터 배운 판과 같지 않다.** 몸통이 학습 중에 CRF 항을
      받았으므로(`--keep-on 1.0` 이라 켜짐 BCE 도 함께 받았다) 표현이 그쪽으로
      기울어 있을 수 있다. 그래도 "지금 있는 것으로 사슬만 떼면 얼마인가" 는
      정확히 이 값이다.
    """
    from src.run_gate_check import load_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    apps = list(ck["appliances"])
    chain = ("heads" in ck) and not no_chain
    if "heads" in ck:
        from src.model.chain import ChainHeads, viterbi
        from src.model.transition import N_FEAT
        m = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)),
                       proj_from={k: ck[k] for k in
                                  ("proj", "proj_cap", "proj_floor", "proj_resp",
                                   "appl_attn", "appl_attn_heads") if k in ck})[0]
        m.load_state_dict(ck["model"])
        hd = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                        score_norm=int(ck.get("score_norm", 0))).to(dev)
        hd.load_state_dict(ck["heads"]); hd.eval()
        if not chain:
            hd = None
    else:
        m = load_model(path, dev)[0]
        hd = None
    m.eval()
    out = {}
    with torch.no_grad():
        for stem, d in cache.items():
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = m(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                      torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                PW.append(o["power"].float())
            gl = torch.cat(GL)[None]
            pw = torch.cat(PW).cpu().numpy().astype(np.float64)
            if chain:
                em, on_, off_, ini = hd(torch.cat(Z)[None],
                                        torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
                on = viterbi(em, on_, off_, ini)[0].cpu().numpy().astype(bool)
            else:
                on = (gl[0] > 0).cpu().numpy()
            out[stem] = (on, pw)
    return apps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=DEFAULT)
    ap.add_argument("--no-chain", nargs="*", default=[], metavar="CKPT",
                    help="이 체크포인트들은 **사슬을 떼고** 창별로 채점한다 (재학습 없는 롤백)")
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps0 = list(torch.load(a.ckpt[0], map_location="cpu",
                            weights_only=False)["appliances"])
    cache = real_windows(apps0, a.grid_s, dev)

    print("사슬을 버리면 — **같은 채점기**로 나란히 (score_arm)", flush=True)
    print("=" * 78)
    res = {}
    for path in a.ckpt:
        apps, pr = predict(path, cache, dev, no_chain=(path in a.no_chain))
        assert apps == apps0, "기기 순서가 다르다 — 비교가 안 선다"
        acc, idn, pwr, rec, cf = {}, {}, [], {}, {}
        for stem, d in cache.items():
            on, pw = pr[stem]
            ac, id_, p_, rc, cn = score_arm(on, pw, d, apps)
            for k, v in ac.items():
                acc.setdefault(k, []).append(v)
            for k, v in cn.items():                       # 파일을 가로질러 **합산**
                cf[k] = tuple(x + y for x, y in zip(cf.get(k, (0,) * 4), v))
            pwr += [x[0] for x in p_.values()]
            for k, v in id_.items():
                idn.setdefault(k, ([], []))[0].append(v[0])
                idn[k][1].append(v[1])
            for k, v in rc.items():
                rec.setdefault(k, []).append(v)
            if rc:
                r = pw.sum(1) + float(d["p_base"]) - d["p_obs"]
                rec.setdefault("rel", []).append(
                    float(np.median(np.abs(r) / np.maximum(d["p_obs"], 10.0))))
        res[path] = (acc, idn, pwr, rec, cf)
        print("  %s 끝" % path.split("/")[-1], flush=True)

    tags = [(p.split("/")[-1][:-3][:11] + ("-창별" if p in a.no_chain else ""))
            for p in a.ckpt]
    print()
    print("  %-20s%s" % ("", "".join("%15s" % t for t in tags)))
    rows = [
        ("on/off · 듀티 제외 7종", lambda r: _avg(r[0], False), "%15.4f"),
        ("on/off · 듀티 2종 ⚠", lambda r: _avg(r[0], True), "%15.4f"),
        ("  핫플 재현 / 정밀", lambda r: _rp(r[4], "hotplate"), "%15s"),
        ("  오븐 재현 / 정밀", lambda r: _rp(r[4], "oven"), "%15s"),
        ("on/off (옛 규약·9종)", lambda r: np.mean([np.mean(v) for v in r[0].values()]),
         "%15.4f"),
        ("저항 4종 신원", lambda r: np.average(r[1]["저항 무리"][0],
                                           weights=r[1]["저항 무리"][1]), "%15.4f"),
        ("SMPS 3종 신원", lambda r: np.average(r[1]["SMPS 무리"][0],
                                            weights=r[1]["SMPS 무리"][1]), "%15.4f"),
        ("상대 잔차 (중앙)", lambda r: 100 * np.mean(r[3]["rel"]), "%14.1f%%"),
        ("절대 잔차 (W)", lambda r: np.mean(r[3]["abs"]), "%14.1fW"),
        ("혼자켜짐 전력오차", lambda r: 100 * np.mean(r[2]), "%14.1f%%"),
    ]
    for nm, fn, fmt in rows:
        line = "  %-20s" % nm
        for p_ in a.ckpt:
            try:
                line += fmt % fn(res[p_])
            except Exception:
                line += "%15s" % "-"
        print(line)

    print()
    print("⚠ **판정은 첫 줄(듀티 제외 7종)로 한다.**")
    print("⚠⚠ 듀티 2종 줄은 **모델이 아니라 `losses.S_STATE` 의 슬롯 상수가 정한다** (13.98).")
    print("   작은 와트 슬롯이 없으면 '켜짐인데 0W' 를 표현할 수 없어 게이트가 0 으로 간다.")
    print("   핫플 슬롯을 549.6 하나에서 {10.0, 549.6} 으로 **옮기기만** 했더니 재현이")
    print("   0.409 -> 0.990 이 됐고 **전력 출력은 그대로다.** 분해 성능이 아니다.")
    print("   맨 아래 '옛 규약·9종' 은 지난 표와 잇기 위해서만 남긴다.")
    print("⚠ `alloc` 의 잔차는 **정의상** 작다 (사영이 합을 맞춘다). 판정에 쓰지 않는다.")
    print("⚠ cnn_v37 의 on/off 는 **창별 게이트**이고 사슬 판은 **Viterbi 경로**다 —")
    print("   그것이 사슬이 주는 것이고, 나머지 줄이 그 대가다.")


if __name__ == "__main__":
    main()
