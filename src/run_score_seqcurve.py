# -*- coding: utf-8 -*-
"""epoch 별 체크포인트를 **한 번에** 실측 채점하고 후반 중앙값을 낸다 (13.84.27).

⚠ 실측 값은 epoch 마다 널뛴다 (13.84.24). 특히 test_2 의 미니PC 는 참값 ON 이 85% 라
사슬이 통째로 꺼지면 0.15, 통째로 켜지면 0.99 로 **중간이 거의 없다** — 그 파일 하나가
5파일 평균을 크게 흔든다. 그래서 한 점이 아니라 **후반 여러 epoch 의 중앙값**으로 읽는다
(13.84.25 가 시드 여섯 판에서 쓴 방식과 같다).

    python -X utf8 src/run_score_seqcurve.py results/hpc          # seq_*_ep*.pt 를 전부
    python -X utf8 src/run_score_seqcurve.py results/hpc --late 4 # 후반 4판으로 중앙값
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_score_seq import score
from src.run_train_seq import FILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["results/hpc"])
    ap.add_argument("--late", type=int, default=4, help="후반 몇 판으로 중앙값을 낼지")
    ap.add_argument("--baseline", default="results/cnn_v37.pt",
                    help="고정 기준선. 창별 헤드만 쓴다 (사슬 머리가 없다)")
    a = ap.parse_args()

    runs = defaultdict(dict)                      # {판이름: {epoch: 경로}}
    for d in a.dirs:
        for p in sorted(Path(d).glob("seq_*_ep*.pt")):
            m = re.match(r"(.+)_ep(\d+)$", p.stem)
            if m:
                runs[m.group(1)][int(m.group(2))] = str(p)
    if not runs:
        raise SystemExit("seq_*_ep*.pt 를 못 찾았습니다: " + ", ".join(a.dirs))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    out = {}                                      # {판: {epoch: (사슬, 창별, 파일별)}}
    for name, eps in sorted(runs.items()):
        out[name] = {}
        for ep in sorted(eps):
            out[name][ep] = score(eps[ep], dev, cache)
            print(".", end="", flush=True)
    print()

    for name in sorted(out):
        eps = sorted(out[name])
        print("\n=== %s  (%d판: epoch %s)" % (name, len(eps), ", ".join(map(str, eps))))
        print("  epoch   사슬    창별  |  " + " ".join("%-7s" % s for s in FILES) + "   (미니PC 사슬)")
        for ep in eps:
            c, w, rows = out[name][ep]
            line = "  %5d  %.4f  %.4f  |  " % (ep, c, w)
            line += " ".join("%-7s" % ("%.3f" % rows[s][1] if s in rows else "–") for s in FILES)
            print(line)
        late = eps[-a.late:]
        if len(late) >= 2:
            print("  후반 %d판 중앙   사슬 %.4f · 창별 %.4f  |  " % (
                len(late),
                float(np.median([out[name][e][0] for e in late])),
                float(np.median([out[name][e][1] for e in late])))
                + " ".join("%-7s" % ("%.3f" % np.median([out[name][e][2][s][1]
                                                        for e in late if s in out[name][e][2]]))
                           for s in FILES))

    # 고정 기준선 — 창별 헤드만. 사슬 머리가 없으므로 사슬 칸은 비운다.
    if Path(a.baseline).exists() and cache:
        from src.run_gate_check import load_model
        model = load_model(a.baseline, dev)[0]
        km = None
        with torch.no_grad():
            accs, rows = [], {}
            for stem, d in cache.items():
                GL = []
                for i in range(0, len(d["t"]), 512):
                    o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                              torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                    GL.append(o["on_logit"].float())
                w = (torch.cat(GL) > 0).cpu().numpy()
                y = d["y"].astype(bool)
                if km is None:
                    km = list(d["apps"]).index("minipc") if "apps" in d else None
                for k in np.nonzero(d["present"])[0]:
                    accs.append(float((w[:, k] == y[:, k]).mean()))
                rows[stem] = w, y
        print("\n=== 고정 기준선 %s (창별만) ===" % a.baseline)
        print("  전체 창별 %.4f" % float(np.mean(accs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
