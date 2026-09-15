# -*- coding: utf-8 -*-
"""**규약을 맞춰서** 다시 채점한다 (14.152) — 전력에 게이트를 어떻게 쓸 것인가 셋.

사용자: *"이 경우에 게이트를 어떤방식으로 써야하지?"*

```
  raw   out["power"] 그대로       기준선은 이미 σ(on) 이 곱해져 있고 분해판은 (1−mix_0) 로 닫혀 있다
  hard  power x 1[on_logit > 0]   판정 마스크
  soft  power x σ(on_logit)       추론에서만 곱을 되살린다 (학습 결합은 여전히 끊김)
```

**결론: `raw` 다.** 네 판 모두에서 가장 낫다. 하드 마스크는 **기준선을 더 크게 망치고**
(pcap SMPS .9807 -> .9333 — 게이트의 연속값이 argmax 에 주던 정보를 문턱이 버린다),
소프트는 분해판에서 **이중 감쇠**다 (이미 닫힌 문에 또 곱한다 — gfp 혼자켜짐 34.2 -> 40.0).
그리고 규약을 바꿔도 **격차의 부호가 안 바뀐다** — 판정은 regime artifact 가 아니다.

    python -X utf8 src/run_diag_gateconv.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa
import torch
from src.run_baseline_nilmtk import score_arm
from src.run_gate_check import load_model
from src.run_train_seq import real_windows
from src.run_score_seq import DUTY_APPS

ARMS = [("cnn_pcap_s%d.pt", "cnn_gfp_s%d.pt"), ("cnn_hv2_s%d.pt", "cnn_gfv2_s%d.pt")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
apps = list(torch.load("results/cnn_pcap_s0.pt", map_location="cpu",
                       weights_only=False)["appliances"])
cache = real_windows(apps, 2.0, dev)


def run(ck, conv):
    m = load_model("results/" + ck, dev)[0]; m.eval()
    acc, idn, pwr, rec, nid = {}, {}, [], {}, {}
    with torch.no_grad():
        for stem, d in cache.items():
            GL, PW = [], []
            for i in range(0, len(d["t"]), 256):
                o = m(torch.from_numpy(np.ascontiguousarray(d["fine"][i:i+256])).to(dev),
                      torch.from_numpy(np.ascontiguousarray(d["wide"][i:i+256])).to(dev))
                GL.append(o["on_logit"].float().cpu().numpy())
                PW.append(o["power"].float().cpu().numpy())
            gl, pw = np.concatenate(GL), np.concatenate(PW).astype(np.float64)
            on = gl > 0
            if conv == "hard":
                pw = pw * on
            elif conv == "soft":
                pw = pw / (1.0 + np.exp(-gl))
            ac, id_, p_, rc, _ = score_arm(on, pw, d, apps)
            for k, v in ac.items():
                acc.setdefault(k, []).append(v)
            pwr += [x[0] for x in p_.values()]
            for k, v in id_.items():
                idn.setdefault(k, ([], []))[0].append(v[0]); idn[k][1].append(v[1])
                nid[k] = nid.get(k, 0) + v[1]
            for k, v in rc.items():
                rec.setdefault(k, []).append(v)
    a7 = np.mean([np.mean(x) for k, x in acc.items() if k not in DUTY_APPS])
    f = lambda k: (np.average(idn[k][0], weights=idn[k][1]) if k in idn else np.nan)
    return (a7, f("저항 무리"), f("SMPS 무리"), np.mean(rec["abs"]),
            100 * np.mean(pwr), nid)


for base, trt in ARMS:
    print("\n" + "=" * 100)
    print("■ %s  대  %s" % (base % 0, trt % 0))
    print("  %-8s %-14s %9s %9s %9s %9s %9s"
          % ("규약", "판", "판정줄7", "저항4", "SMPS3", "절대잔차", "혼자켜짐"))
    for conv in ("raw", "hard", "soft"):
        for tmpl in (base, trt):
            R = [run(tmpl % s, conv) for s in range(3)]
            n = R[0][5]
            print("  %-8s %-14s %9.4f %9.4f %9.4f %8.1fW %8.1f%%   (신원 표본 저항 %d · SMPS %d)"
                  % (conv, tmpl.replace("_s%d.pt", ""),
                     *[np.mean([r[i] for r in R]) for i in range(5)],
                     n.get("저항 무리", 0), n.get("SMPS 무리", 0)))
        print()
