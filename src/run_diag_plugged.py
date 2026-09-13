# -*- coding: utf-8 -*-
"""`plugged` 머리는 사슬 밖에 있다 — 대기 항이 얼마나 흔들리나 (13.84.64).

`L_harm` 은 대기를 **쓰고 있다**:

    pred = Σ_k P_k . sig_k  +  Σ_k idle_k . standby_sig_k  +  noise_sig
    idle_k = σ(plugged_k) . (1 − σ(on_k))

그런데 `on` 은 CRF 가 시간으로 매끄럽게 하고 **`plugged` 는 안 한다** — 사슬의 상태가
기기당 둘(켜짐/꺼짐)뿐이라 `plugged` 는 창마다 독립인 CNN 머리다. 물리에서 `plugged` 는
이 문제에서 **가장 오래 유지되는 양**이다 (몇 시간 동안 안 변한다).

그 흔들림의 값이 기기마다 다르다. 대기 페이저가 통전 증분에 견줘:
    충전기   6.1mA 대 148mA   (4%)   -> 흔들려도 안 보인다
    미니PC  26.6mA 대  28mA  (95%)   -> **흔들림이 켜짐/꺼짐과 같은 크기다**

재는 것 (실측 5파일, 참값은 `real_events.json` 의 `appliances_present`):
  ① 수준    파일에 **있는** 기기의 σ(plugged) 가 1 인가. 없는 기기는 0 인가
  ② 흔들림  창끼리 |Δσ(plugged)| — 그리고 그것이 만드는 대기 전류 흔들림 (mA)
  ③ 견줌    그 흔들림이 그 기기의 **통전 증분**의 몇 %인가
  ④ 상관    미니PC 의 plugged 가 참 on 과 얼마나 붙어 있나 (붙어 있으면 대기가 두 번 세진다)

    python -X utf8 src/run_diag_plugged.py [results/seq_h38_base.pt ...]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.model.lossbuild import build_loss
from src.run_gate_check import load_model
from src.run_train_seq import real_windows

WATCH = ("laptop_charger", "minipc", "beam_projector")


def main():
    cks = sys.argv[1:] or ["results/seq_h38_base.pt"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    for ckp in cks:
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        apps = list(ck["appliances"])
        model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                           mask=not bool(ck.get("no_mask", False)))[0]
        model.load_state_dict(ck["model"]); model.eval()
        heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                           score_norm=int(ck.get("score_norm", 0))).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        if not cache:
            cache.update(real_windows(apps, ck["meta"]["grid_s"], dev))

        crit = build_loss(apps, "cpu", verbose=False)
        sbc = crit.standby_sig.numpy()[..., 0] + 1j * crit.standby_sig.numpy()[..., 1]  # (K,H)
        sgc = crit.sig.numpy()[..., 0] + 1j * crit.sig.numpy()[..., 1]

        print("\n=== %s ===" % ckp)
        print("① **수준** — 파일에 있는 기기의 σ(plugged) 중앙 (참값 1) · 없는 기기 (참값 0)")
        print("   %-8s %-18s %7s %8s %8s %8s %8s"
              % ("파일", "기기", "있나", "중앙", "p10", "p90", "on 중앙"))
        rows = {a: {"d": [], "ma": [], "on": [], "pl": []} for a in WATCH}
        grp = {}
        for stem, d in cache.items():
            with torch.no_grad():
                PL, ON, Z, GL = [], [], [], []
                for i in range(0, len(d["t"]), 512):
                    o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                              torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                    PL.append(torch.sigmoid(o["plugged_logit"]).float().cpu())
                    ON.append(torch.sigmoid(o["on_logit"]).float().cpu())
                    Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                em, on, off, ini = heads(torch.cat(Z)[None],
                                         torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                         torch.cat(GL)[None])
                path = viterbi(em, on, off, ini)[0].cpu().numpy()
            pl = torch.cat(PL).numpy(); og = torch.cat(ON).numpy()
            grp[stem] = {a: (bool(d["present"][apps.index(a)]),
                             float(np.median(pl[:, apps.index(a)])),
                             float(d["y"][:, apps.index(a)].max())) for a in WATCH}
            for a in WATCH:
                k = apps.index(a)
                pres = bool(d["present"][k])
                print("   %-8s %-18s %7s %8.3f %8.3f %8.3f %8.3f"
                      % (stem, a, "있다" if pres else "없다",
                         np.median(pl[:, k]), *np.percentile(pl[:, k], [10, 90]),
                         np.median(og[:, k])))
                if not pres:
                    continue
                # `idle` 은 사슬이 정한 on 을 쓴다 — 채점 경로와 같은 규약
                idle = pl[:, k] * (1.0 - path[:, k])
                rows[a]["d"].append(np.abs(np.diff(idle)))
                rows[a]["ma"].append(np.abs(sbc[k, 0]))
                rows[a]["on"].append(d["y"][:, k].astype(float))
                rows[a]["pl"].append(pl[:, k])

        print("\n② **흔들림** — 창끼리 |Δidle| 과 그것이 만드는 h1 대기 전류 (mA)")
        print("   %-18s %9s %9s %9s %10s %10s"
              % ("기기", "|Δidle|중앙", "p90", "최대", "mA 중앙", "mA p90"))
        for a in WATCH:
            if not rows[a]["d"]:
                continue
            dd = np.concatenate(rows[a]["d"]); ma = 1000 * float(np.mean(rows[a]["ma"]))
            print("   %-18s %9.4f %9.4f %9.4f %10.2f %10.2f"
                  % (a, np.median(dd), np.percentile(dd, 90), dd.max(),
                     ma * np.median(dd), ma * np.percentile(dd, 90)))

        print("\n③ **견줌** — 그 흔들림이 통전 **증분**의 몇 %인가 (증분 = P_typ.|sig| − |대기|)")
        print("   %-18s %10s %10s %10s %9s" % ("기기", "대기 mA", "증분 mA", "흔들림 mA(p90)", "몫 %"))
        PTYP = {"laptop_charger": 30.0, "minipc": 12.0, "beam_projector": 45.0}
        for a in WATCH:
            if not rows[a]["d"]:
                continue
            k = apps.index(a)
            sb = 1000 * abs(sbc[k, 0]); dl = 1000 * abs(PTYP[a] * sgc[k, 0] - sbc[k, 0])
            fl = sb * np.percentile(np.concatenate(rows[a]["d"]), 90)
            print("   %-18s %10.1f %10.1f %14.2f %8.1f"
                  % (a, sb, dl, fl, 100 * fl / max(dl, 1e-9)))

        print("\n④ **상관** — plugged 가 참 on 을 따라가나 (따라가면 대기가 켜질 때 두 번 세진다)")
        print("   %-18s %10s %10s %9s" % ("기기", "on 때 중앙", "off 때 중앙", "상관"))
        for a in WATCH:
            if not rows[a]["pl"]:
                continue
            p = np.concatenate(rows[a]["pl"]); y = np.concatenate(rows[a]["on"])
            if y.std() < 1e-6 or p.std() < 1e-6:
                print("   %-18s %10.3f %10.3f %9s"
                      % (a, np.median(p[y > 0.5]) if (y > 0.5).any() else float("nan"),
                         np.median(p[y < 0.5]) if (y < 0.5).any() else float("nan"), "—"))
                continue
            print("   %-18s %10.3f %10.3f %9.2f"
                  % (a, np.median(p[y > 0.5]), np.median(p[y < 0.5]),
                     float(np.corrcoef(p, y)[0, 1])))

        # -- ⑤ 무리 합 ------------------------------------------------------
        # 프로젝터와 미니PC 의 **대기 페이저가 같다** (크기비 1.02 · 3°, 13.84.64).
        # 그러면 둘의 `plugged` 는 따로 못 정하고 **합만** 정해진다. 그 예측을 건다.
        print("\n⑤ **무리 합** — 프로젝터·미니PC 는 대기가 같은 물건이다. 합만 정해져야 한다")
        print("   %-8s %9s %9s %11s %9s   %9s %s"
              % ("파일", "proj σ", "mini σ", "합", "참 개수", "충전기 σ", "그 파일에서 켜지나"))
        for stem, g in grp.items():
            bp, mp, lc = g["beam_projector"], g["minipc"], g["laptop_charger"]
            print("   %-8s %9.3f %9.3f %11.2f %9d   %9.3f %s"
                  % (stem, bp[1], mp[1], bp[1] + mp[1], int(bp[0]) + int(mp[0]), lc[1],
                     " ".join(x[:4] for x in WATCH if g[x][2] > 0.5) or "아무것도"))
        print("   ⚠ 갈림은 대기가 아니라 **`on ⊂ plugged` 함의**(12.164.9)가 만든다 —")
        print("     그 파일에서 켜지는 기기만 plugged 가 오른다. 아무도 안 켜지면 둘 다 0.55 다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
