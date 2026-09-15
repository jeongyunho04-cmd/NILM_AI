# -*- coding: utf-8 -*-
"""**통전 기준** 오븐 자 — ★2 는 상태를 안 갈랐다 (14.135).

사용자: *"오븐 오탐을 볼 때 통전으로 오탐 나는 경우만 보고 있는 거 맞지?"*

**아니었다.** ★2 와 오늘 쓴 모든 "헛게이트" 는 `on_logit > 0` 이라 오븐의 두 상태를
한 덩이로 셌다 — `S_STATE["oven"] = {1: 16.8, 2: 1357.1}` 인데 **16.8W 팬조명 헛detect
와 1357W 통전 헛detect 가 같은 무게**로 들어갔다. 갈라 보면:

```
  참 OFF · 큰 저항 부하 켜진 창 1043개 (cnn_pcap 3시드)
    게이트 (지금까지 쓴 것)   **20.2%**
    ├─ 통전 state2 > 0.5      **5.6%**   <- 그림에서 파란 1100W 덩이로 보이는 것
    └─ 팬조명만               **13.6%**  <- 16W. 그래프에 거의 안 보인다
```
**넷 중 하나만 통전이다.** 그리고 거리축에서 둘이 다르게 움직인다 — 게이트는 계단에서
멀수록 9.2 -> 37.7% 로 4배가 되는데 **통전은 4.9 -> 6.7% 로 거의 평평**하다.

⚠ ★1 은 원래부터 `softmax(state)[:, oven, 2]` 라 **통전만** 본다. 안 갈린 것은 ★2 다.
⚠⚠ 못 박아 둔 ★1·★2 는 **바꾸지 않는다** (자를 갈면 지난 판정과 못 잇는다).
   여기는 **통전판 짝**을 따로 두는 도구다.

재는 것
-------
```
  **포트창 오븐통전%**   포트 참ON · 오븐 참OFF 창에서 오븐이 통전이라고 한 비율
                        = "포트를 오븐 히터로 읽는다" (★2 의 통전판)
  **포트창 오븐게이트%**  같은 창에서 상태 무관 게이트 (★2 그대로 — 견주려고 남긴다)
  **전이근처 오븐통전%**  전이 ±20초 안 · 큰 저항 부하 · 오븐 참OFF 창에서 통전
                        = "누가 켜지고 꺼질 때 오븐 히터를 붙인다"
  **전이밖 오븐통전%**    같은 조건, 전이 ±20초 **밖** — 0 이어야 한다
  **오븐 재현**          오븐 참ON 창의 게이트 (억누르면 여기가 떨어진다 — 지킴)
  **오븐 헛전력 W**      참 OFF 인데 오븐에 준 평균 전력 (크기로 본 것)
```

    python -X utf8 src/run_diag_ovencond.py --arm pcap results/cnn_pcap_s{0,1,2}.pt \\
                                            --arm hzwt results/cnn_hzwt_s{0,1,2}.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.evaluation.real_events import load_events  # noqa: E402
from src.model.realdata import dense_targets  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
RES4 = {"oven", "electiric_kettle", "hotplate", "hair_dryer"}
BIG3 = ("electiric_kettle", "hair_dryer", "hotplate")
ON_STATE = 2
NEAR_S = 20.0


def build():
    """(창, 오븐참ON, 포트참ON, 큰부하참ON, 전이근처) 를 파일마다."""
    out = {}
    for s in FILES:
        rw = dense_targets(s, stride=30)
        t = np.asarray(rw.target_cycle, float) / 60.0
        iv = load_events()[s]["intervals"]
        ev = load_events()[s]["events"]

        def mask(app):
            m = np.zeros(len(t), bool)
            for t0, t1 in iv.get(app, {}).get("on", []):
                m |= (t >= t0) & (t <= t1)
            return m

        big = np.zeros(len(t), bool)
        for a in BIG3:
            big |= mask(a)
        xs = np.array(sorted(e["t_s"] for e in ev if e["appliance"] in RES4))
        near = np.zeros(len(t), bool)
        for x in xs:
            near |= np.abs(t - x) <= NEAR_S
        out[s] = (rw, mask("oven"), mask("electiric_kettle"), big, near)
    return out


def probe(ck, D, dev):
    m, apps, _ = load_model(ck, dev)
    j = apps.index("oven")
    S, G, P = [], [], []
    with torch.no_grad():
        for s in FILES:
            rw = D[s][0]
            for i in range(0, len(rw), 256):
                f, w, *_ = rw.batch(np.arange(i, min(i + 256, len(rw))))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                          torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                S.append(torch.softmax(o["state"].float(), -1).cpu().numpy()[:, j])
                G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy()[:, j])
                P.append(o["power"].float().cpu().numpy()[:, j])
    return np.concatenate(S), np.concatenate(G), np.concatenate(P)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", nargs="+", default=[], metavar="NAME CKPT",
                    help="팔 이름과 체크포인트들. 여러 번 줄 수 있다")
    ap.add_argument("--base", default="", help="짝차의 기준이 될 팔 이름")
    a = ap.parse_args()
    if not a.arm:
        raise SystemExit("--arm 을 하나 이상 줘라")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = build()
    cat = lambda k: np.concatenate([D[s][k] for s in FILES])
    ov, ke, big, near = cat(1), cat(2), cat(3), cat(4)
    M2 = ke & ~ov                       # ★2 의 모집단 (포트 참ON · 오븐 참OFF)
    M5 = (~ov) & big & near
    M5o = (~ov) & big & ~near

    print("**통전 기준** 오븐 자 (14.135) — ★2 는 팬조명 16.8W 와 통전 1357W 를 안 갈랐다")
    print("  모집단: ★2c %d창 · ★5 %d창 · ★5밖 %d창 · 오븐 참ON %d창\n"
          % (M2.sum(), M5.sum(), M5o.sum(), ov.sum()))
    #: 이름은 **무엇이 문제인지** 바로 읽히게 쓴다. ★번호는 못 박은 자(★1·★2)에만 쓴다.
    hdr = ("포트창 오븐통전%", "포트창 오븐게이트%", "전이근처 오븐통전%",
           "전이밖 오븐통전%", "오븐 재현", "오븐 헛전력 W")
    print("  %-10s %s" % ("팔", "  ".join("%-17s" % h for h in hdr)))
    res = {}
    for arm in a.arm:
        nm, cks = arm[0], arm[1:]
        rows = []
        for ck in cks:
            S, G, P = probe(ck, D, dev)
            c = S[:, ON_STATE] > 0.5
            rows.append([100 * c[M2].mean(), 100 * (G[M2] > 0.5).mean(),
                         100 * c[M5].mean(), 100 * c[M5o].mean() if M5o.sum() else 0.0,
                         float((G[ov] > 0.5).mean()), float(P[(~ov) & big].mean())])
        R = np.asarray(rows)
        res[nm] = R
        print("  %-10s %s" % (nm, "  ".join(
            "%7.2f ± %-7.2f" % (R[:, k].mean(), R[:, k].std()) for k in range(6))))
    if a.base and a.base in res:
        print("\n  짝차 (기준 %s)" % a.base)
        B = res[a.base]
        for nm, R in res.items():
            if nm == a.base or R.shape != B.shape:
                continue
            d = R - B
            print("  %-10s %s" % (nm, "  ".join(
                "%+7.2f(%d/%d)" % (d[:, k].mean(), int((d[:, k] < 0).sum()), len(d))
                if k < 4 else "%+7.3f(%d/%d)" % (d[:, k].mean(),
                                                 int((d[:, k] > 0).sum()), len(d))
                for k in range(6))))
        print("  (앞 넷은 **내려가야** 좋다 · 재현·헛전력은 괄호가 오른 시드 수)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
