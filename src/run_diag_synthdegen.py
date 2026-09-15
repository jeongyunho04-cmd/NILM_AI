# -*- coding: utf-8 -*-
"""합성에서도 **포트↔오븐 축퇴**가 나는가 (14.140) — 갈림을 하나로 좁히는 시험.

실측에서 잰 것 (오븐 참OFF · 큰 저항부하 참ON · `cnn_pcap` 3시드):
```
  포트만 1.1% · 드라이만 0.0% · 핫플만 0.0% · 포트+핫플 0.0% · 드라이+핫플 0.0%
  **포트+드라이 59.6%** · **셋 다 83.9%**
```
그리고 사전을 보면 둘은 거의 같은 물건이다.
```
  electiric_kettle s1   와트당|I1| **4.387**  h2/h1 0.0003  h3/h1 0.0326   35.8Ω
  oven             s2   와트당|I1| **4.749**  h2/h1 0.0019  h3/h1 0.0051   40.6Ω
```
구별자가 **컨덕턴스 13%** 하나뿐이다. 그럼 갈림은 둘이다.

```
  (가) 합성에서도 혼동이 난다  -> **식별 불가**. 사전·입력을 고쳐야 한다
                                 (구조 손잡이를 아무리 돌려도 안 된다)
  (나) 합성은 0%인데 실측만 60% -> **도메인 간극**. 합성에 그 조합·잔여 분포가
                                 없는 것이고 합성을 고치면 된다
```
합성은 기기별 참값(`y_power`·`y_state`)이 있으므로 실측 표를 **그대로** 복제한다.

    python -X utf8 -m src.run_diag_synthdegen results/cnn_pcap_s{0,1,2}.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.evaluation.holdout import load_holdout  # noqa: E402
from src.model.inputs import build_inputs  # noqa: E402
from src.run_gate_check import load_model, sync_even_median  # noqa: E402

ON_STATE = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32h")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    #: 14.190 — 창을 짓기 **전에** 짝수차 규약을 체크포인트에 맞춘다.
    #  이 도구는 14.140 (EVEN_MEDIAN=0) 때 지었는데 지금 팔들은 `--even-median 5`
    #  로 굽는다. 안 맞추면 `load_model` 의 관문이 멈춘다 (985321 이 그렇게 죽었다).
    sync_even_median(a.ckpt, 0)
    hs = load_holdout(a.holdout)
    apps = list(hs.appliances)
    jo, jk, jd, jh = (apps.index(x) for x in
                      ("oven", "electiric_kettle", "hair_dryer", "hotplate"))
    print("합성 축퇴 시험 — 홀드아웃 %d창 · 장치 %s" % (len(hs.y_on), dev))

    F, W = [], []
    for i in range(0, len(hs.y_on), 512):
        f, w = build_inputs(np.asarray(hs.X[i:i + 512]))
        F.append(f); W.append(w)
    F, W = np.concatenate(F), np.concatenate(W)

    ST, PW, ONL = [], [], []
    for ck in a.ckpt:
        m, mapps, _ = load_model(ck, dev)
        assert mapps == apps, "기기 순서가 다르다"
        s_, p_, o_ = [], [], []
        with torch.no_grad():
            for i in range(0, len(F), 512):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = m(torch.from_numpy(F[i:i + 512]).to(dev),
                          torch.from_numpy(W[i:i + 512]).to(dev))
                s_.append(torch.softmax(o["state"].float(), -1).cpu().numpy())
                p_.append(o["power"].float().cpu().numpy())
                o_.append(o["on_logit"].float().cpu().numpy())
        ST.append(np.concatenate(s_)); PW.append(np.concatenate(p_))
        ONL.append(np.concatenate(o_))
        print("  읽음 %s" % ck)
    ST, PW, ONL = np.stack(ST), np.stack(PW), np.stack(ONL)

    on = hs.y_on.astype(bool)
    yst = np.asarray(hs.y_state)
    ov, ke, hd, hp = on[:, jo], on[:, jk], on[:, jd], on[:, jh]
    ovcond = ov & (yst[:, jo] == ON_STATE)          # 오븐이 **통전**으로 참ON
    POP = (~ovcond) & (ke | hd | hp)                # 통전 참OFF · 큰 저항부하
    cond = (ONL[:, :, jo] > 0) & (ST[:, :, jo, ON_STATE] > 0.5)     # (seeds, N)

    print("\n  참ON: 포트 %d · 드라이 %d · 핫플 %d · 오븐 통전 %d · 오븐 팬조명 %d"
          % (ke.sum(), hd.sum(), hp.sum(), ovcond.sum(),
             (ov & (yst[:, jo] == 1)).sum()))
    print("  모집단(오븐 통전 참OFF · 큰 저항부하 참ON) **%d창**\n" % POP.sum())

    print("★ 조합별 오븐 통전 헛detect — **실측 표와 같은 자**")
    print("  %-22s %8s %24s %12s" % ("켜진 조합", "창", "통전 헛% (시드별)", "오븐 헛W"))
    REAL = {"포트만": 1.1, "드라이만": 0.0, "핫플만": 0.0, "포트+드라이": 59.6,
            "포트+핫플": 0.0, "드라이+핫플": 0.0, "셋 다": 83.9}
    for nm, mk in (("포트만", ke & ~hd & ~hp), ("드라이만", hd & ~ke & ~hp),
                   ("핫플만", hp & ~ke & ~hd), ("포트+드라이", ke & hd & ~hp),
                   ("포트+핫플", ke & ~hd & hp), ("드라이+핫플", ~ke & hd & hp),
                   ("셋 다", ke & hd & hp)):
        q = POP & mk
        if q.sum() == 0:
            print("  %-22s %8d   (창 없음)   실측 %.1f%%" % (nm, 0, REAL[nm]))
            continue
        rs = [100 * cond[s][q].mean() for s in range(len(a.ckpt))]
        print("  %-22s %8d   %s  %10.1f   실측 **%.1f%%**"
              % (nm, int(q.sum()), " ".join("%6.1f%%" % r for r in rs),
                 PW[:, q, jo].mean(), REAL[nm]))

    print("\n★ 포트를 오븐으로 바꿔 읽나 — 포트 참ON 창에서")
    q = POP & ke
    if q.sum():
        print("  포트 참전력 %7.1fW -> 모델 %7.1fW   (게이트 %4.1f%%)"
              % (hs.y_power[q, jk].mean(), PW[:, q, jk].mean(),
                 100 * (ONL[:, :, jk][:, q] > 0).mean()))
        print("  오븐 참전력 %7.1fW -> 모델 %7.1fW   (게이트 %4.1f%%)"
              % (hs.y_power[q, jo].mean(), PW[:, q, jo].mean(),
                 100 * (ONL[:, :, jo][:, q] > 0).mean()))
        print("  실측 같은 자리: 포트 1235W -> **63W**(게이트 1.2%) · "
              "오븐 0W -> **1176W**(게이트 100%)")
    print("\n  판정: 합성에서도 포트+드라이가 크면 **식별 불가**(사전을 고쳐야 한다).")
    print("        합성이 0%면 **도메인 간극**(합성에 그 조합·잔여 분포가 없다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
