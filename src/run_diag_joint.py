# -*- coding: utf-8 -*-
"""형제를 **묶어서** 풀 자리가 있는가 (13.84.45).

지금 `chain.py` 는 기기 축을 **독립**으로 둔다 — 기기마다 2상태 사슬을 따로 돌리고
손실에서만 합친다. 그런데 SMPS 배분은 형제 셋이 같은 전류를 놓고 다투는 문제다
(13.84.31: 셋이 21° 안 · 13.84.32: 유령과 진짜가 같은 차수를 메운다).

결합 구조(형제 무리의 **구성**을 한꺼번에 고르는 사슬)를 만들기 전에, 그 구조가
**메울 자리가 있는지** 먼저 본다. 재는 것 넷:

  ① 구성 혼동   형제 3비트 구성의 참 대 예측. 틀린 것 중 **맞바꿈**(개수는 맞고 주인이 틀림)이
                몇 %인가. 맞바꿈이면 결합 복호로 잡을 수 있다. 개수 오류면 못 잡는다.
  ② 조건부      미니PC 가 틀린 순간에 형제도 **반대 방향으로** 틀리는가 (보상 오류)
  ③ 합 제약     예측 구성이 함의하는 SMPS 전력이 관측을 넘는가 (넘으면 복호 때 쳐낼 수 있다)
  ④ 상한        참 구성을 **개수만** 알려주면 (미니PC 가 몇 대 중 하나인지) 얼마나 오르나

    python -X utf8 src/run_diag_joint.py [results/seq_h38_base.pt]
"""
import sys
from itertools import product

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
SIB = ["laptop_charger", "beam_projector", "minipc"]


def bits(v):
    return "".join("1" if x else "0" for x in v)


def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "results/seq_h38_base.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)
    ks = [apps.index(x) for x in SIB]
    km = apps.index("minipc")

    conf, tot = {}, 0
    nswap = nmiss = nextra = 0
    cond = {"형제도 틀림(보상)": 0, "형제도 틀림(같은방향)": 0, "미니PC만 틀림": 0}
    over = [0, 0]
    lift = {"지금": [], "개수 줌": []}
    with torch.no_grad():
        for stem, d in cache.items():
            if not all(d["present"][k] for k in ks):
                continue
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                PW.append(o["power"].float())
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            path = viterbi(em, on, off, ini)[0].cpu().numpy()
            y = d["y"].astype(bool)
            pw = torch.cat(PW).cpu().numpy()
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            P = np.asarray(r["power_features"])[:, 0]
            ti = np.clip((d["t"] * FS).astype(int), 0, len(P) - 1)

            yt = y[:, ks]
            pt = path[:, ks]
            for a, b in zip(yt, pt):
                tot += 1
                conf[(bits(a), bits(b))] = conf.get((bits(a), bits(b)), 0) + 1
                if (a != b).any():
                    if a.sum() == b.sum():
                        nswap += 1
                    elif b.sum() > a.sum():
                        nextra += 1
                    else:
                        nmiss += 1
            mi = SIB.index("minipc")
            wrong_m = yt[:, mi] != pt[:, mi]
            for j in np.nonzero(wrong_m)[0]:
                sw = yt[j] != pt[j]
                sw[mi] = False
                if not sw.any():
                    cond["미니PC만 틀림"] += 1
                elif yt[j].sum() == pt[j].sum():
                    cond["형제도 틀림(보상)"] += 1
                else:
                    cond["형제도 틀림(같은방향)"] += 1
            # ③ 합 제약 — 예측 SMPS 전력이 관측 총전력을 넘는가
            psm = pw[:, ks].sum(1)
            over[0] += int((psm > P[ti]).sum()); over[1] += len(psm)
            # ④ 상한 — 참 **개수**만 주고 미니PC 를 다시 정한다
            lift["지금"].append(float((pt[:, mi] == yt[:, mi]).mean()))
            g = torch.cat(GL).cpu().numpy()[:, ks]
            fixed = np.zeros_like(pt)
            for j in range(len(yt)):
                n = int(yt[j].sum())
                fixed[j, np.argsort(-g[j])[:n]] = True
            lift["개수 줌"].append(float((fixed[:, mi] == yt[:, mi]).mean()))

    print("① 형제 구성 혼동 (%s) — 전체 %d 단계" % (" · ".join(SIB), tot))
    bad = nswap + nmiss + nextra
    print("   틀린 단계 %d (%.1f%%) 중" % (bad, 100 * bad / tot))
    print("      **맞바꿈** (개수 맞고 주인 틀림) %6d  %5.1f%%   <- 결합 복호로 잡을 수 있다"
          % (nswap, 100 * nswap / max(bad, 1)))
    print("      과다   (개수 더 셈)            %6d  %5.1f%%" % (nextra, 100 * nextra / max(bad, 1)))
    print("      과소   (개수 덜 셈)            %6d  %5.1f%%" % (nmiss, 100 * nmiss / max(bad, 1)))
    print("\n   큰 혼동 칸 (참 -> 예측, %s 순)" % "".join(x[0] for x in SIB))
    for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])[:8]:
        print("      %s -> %s  %6d  %5.1f%%%s" % (a, b, n, 100 * n / tot,
                                                  "" if a == b else "   <- 오류"))

    print("\n② 미니PC 가 틀린 순간 형제는")
    s = sum(cond.values())
    for k, v in cond.items():
        print("   %-22s %6d  %5.1f%%" % (k, v, 100 * v / max(s, 1)))

    print("\n③ 합 제약 — 예측 SMPS 전력이 관측 총전력을 넘는 단계 %d / %d (%.1f%%)"
          % (over[0], over[1], 100 * over[0] / max(over[1], 1)))

    print("\n④ 상한 — 형제 중 **몇 대가 켜졌는지**만 알려주면 (게이트 순위로 배정)")
    print("   미니PC 시간 정확도  지금 %.4f -> 개수 줌 %.4f"
          % (float(np.mean(lift["지금"])), float(np.mean(lift["개수 줌"]))))
    print("   ⚠ 이것은 신탁이다. 방법이 아니라 **결합 구조가 메울 수 있는 자리의 상한**이다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
