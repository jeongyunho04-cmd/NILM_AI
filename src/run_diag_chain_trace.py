# -*- coding: utf-8 -*-
"""사슬이 test_2 미니PC 를 왜 못 켜는가 (13.84.25).

13.84.24: 사슬이 실측 평균을 0.852 -> 0.912~0.916 으로 올렸는데 **test_2 만 0.44 -> 0.29** 로
세 시드 모두 나빠졌다. 0.29 는 "거의 내내 꺼짐" 이고, test_2 는 미니PC 참값 ON 이 85% 인 파일이다.
의심: 켜짐 전이를 못 찾으면 사슬이 전환 벌점 때문에 처음부터 끝까지 꺼짐으로 간다.

파일마다 낸다.
  ① 참 켜짐/꺼짐 시각에서 전이 머리가 낸 점수 대 그 파일 전체의 분포 (순위)
  ② 방출 점수의 부호 — 창별이 그 구간에서 켜짐이라 하는가
  ③ 사슬이 고른 열의 전환 횟수와 첫 상태
  ④ 참값이 창 **밖에서 이미 켜져 있던** 경우 — 격자 안에 켜짐 전이가 아예 없는가

    python -X utf8 src/run_diag_chain_trace.py [results/chain_s0.pt]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.run_train_chain import FILES, real_sequences

CK = sys.argv[1] if len(sys.argv) > 1 else "results/chain_s0.pt"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CK, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"])
    heads.eval()
    real, _ = real_sequences(ck["meta"]["ckpt"], dev, ck["meta"]["grid_s"])
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    k = apps.index("minipc")
    print("%s · 전환 벌점(미니PC) %.2f · 방출 배율 %.2f · v37 배율 %.2f"
          % (CK, float(heads.switch_bias[k]), float(heads.emit_scale[k]),
             float(heads.base_scale[k])))
    for stem in FILES:
        d = real[stem]
        if "minipc" not in ev[stem]["appliances_present"]:
            continue
        with torch.no_grad():
            z = torch.from_numpy(d["z"]).float()[None].to(dev)
            df = torch.from_numpy(d["dfeat"]).float()[None].to(dev)
            bs = torch.from_numpy(d["base"]).float()[None].to(dev)
            em, on, off = heads(z, df, bs)
            path = viterbi(em, on, off)[0, :, k].cpu().numpy()
        t = d["t"]
        y = d["y"][:, k].astype(bool)
        e = em[0, :, k].cpu().numpy()
        so = on[0, :, k].cpu().numpy()
        sf = off[0, :, k].cpu().numpy()
        ivs = ev[stem]["intervals"].get("minipc", {}).get("on", [])
        print("\n=== %s  참값 ON %.0f%% · 격자 %.0f~%.0f초" % (stem, 100 * y.mean(), t[0], t[-1]))
        print("    사슬 예측 ON %.0f%% · 전환 %d회 · 첫 상태 %s"
              % (100 * path.mean(), int((np.diff(path.astype(np.int8)) != 0).sum()),
                 "ON" if path[0] else "OFF"))
        print("    방출 점수: 참 ON 구간 중앙 %+.2f · 참 OFF 구간 중앙 %+.2f (>0 이면 창별은 켜짐)"
              % (np.median(e[y]) if y.any() else np.nan,
                 np.median(e[~y]) if (~y).any() else np.nan))
        for t0, t1 in ivs:
            for tt, nm, sc in ((t0, "켜짐", so), (t1, "꺼짐", sf)):
                if tt < t[0] or tt > t[-1]:
                    print("    %s %6.1f초 — **격자 밖** (창 앞/뒤라 사슬이 못 본다)" % (nm, tt))
                    continue
                j = int(np.argmin(np.abs(t - tt)))
                w = slice(max(0, j - 3), min(len(t), j + 4))
                best = float(sc[w].max())
                rank = float((sc > best).mean())
                print("    %s %6.1f초 — 점수 최대 %+7.2f (파일 상위 %.1f%%) · 그 파일 점수 p50 %+.2f p99 %+.2f"
                      % (nm, tt, best, 100 * rank, float(np.median(sc)), float(np.percentile(sc, 99))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
