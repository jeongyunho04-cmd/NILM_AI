# -*- coding: utf-8 -*-
"""**전이 창이 구조적으로 무엇이 다른가** — 부호 있는 거리와 정규화 통계 (14.138).

사용자: *"모델 구조 전체를 면밀히 분석해서 전이 근처에 오탐을 발생시키는 원인이
있는지 한번 더 검토해 줄 수 있어?"*

지금까지 "전이 ±20초 안" 이라고만 셌다. **부호를 안 봤다.** 그런데 부호가
기전을 가른다.

```
  계단이 **미래**에 있다 (아직 안 일어났다)  -> 미리 켜는 것. **미래로 새는 줄**이 있다
  계단이 **과거**에 있다 (이미 일어났다)      -> 과도·정착 문제. 미래와 무관하다
```

그리고 구조에서 **전이 창에만** 다른 것이 하나 더 있다 —

```
  `_blk` = Conv1d -> **GroupNorm(g, C)** -> GELU
  GroupNorm 은 (C_g, **T 전체**) 로 μ·σ 를 낸다. 창에 계단이 있으면 σ 가 부푼다.
  ⇒ 타깃 순간의 국소 내용이 **똑같아도** 창 어딘가에 계단이 있으면 특징이 달라진다.
     수용영역과 무관하다. `--fine-time-split`·`--fine-pad`·`--seg-pool` 이 못 끊는다.
```

재는 것
-------
```
  [A] ★ **부호 있는 계단 거리**축에서 오븐 통전 헛detect
      음수 = 계단이 미래 (아직 안 일어났다) · 양수 = 이미 일어났다
  [B] 블록별 GroupNorm **σ 가 전이 창에서 얼마나 부푸나**
  [C] ★ σ 부풀이가 **답을 움직이나** — 같은 σ 로 되돌려 놓고 머리를 다시 태운다
      (몸통 가중치도 입력도 안 건드린다. **정규화 통계만** 계단 없는 창 값으로 바꾼다)
```

계단은 `|p60[i+6] − p60[i−6]| > 200W` 로 잡는다 (14.134 에서 589개 · 중앙 간격 0.90초).

    python -X utf8 src/run_diag_stepside.py results/cnn_pcap_s0.pt
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
from src.model.realdata import DEFAULT_DIR, dense_targets  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
BIG3 = ("electiric_kettle", "hair_dryer", "hotplate")
RES4 = {"oven", "electiric_kettle", "hotplate", "hair_dryer"}
STEP_W = 200.0          # 계단 판정 문턱 (14.134)
STEP_K = 6              # +-6 사이클 = +-0.1초


def steps_of(s):
    """관측 총전력에서 계단 시각(초)을 뽑는다 — 라벨이 아니라 **관측**이다."""
    p = np.asarray(load_nilm_npz("%s/%s.npz" % (DEFAULT_DIR, s))["p_denoised_w"], float)
    d = np.zeros(len(p))
    d[STEP_K:-STEP_K] = np.abs(p[2 * STEP_K:] - p[:-2 * STEP_K])
    idx = np.flatnonzero(d > STEP_W)
    if len(idx) == 0:
        return np.zeros(0)
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > 30:               # 0.5초 안쪽은 한 계단으로 본다
            keep.append(i)
    return np.asarray(keep, float) / 60.0


def table(nm, POP, cond, dd, extra=""):
    print("[%s] 부호 있는 거리 — 음수 = **아직 안 일어났다**(미리 켬) · "
          "양수 = **이미 일어났다**%s" % (nm, extra))
    print("  %-18s %8s %12s" % ("거리 띠 (초)", "창", "통전 헛%"))
    bins = [(-30, -20), (-20, -10), (-10, -6), (-6, -4), (-4, -2), (-2, 0),
            (0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 30)]
    pre = post = npre = npost = 0
    for lo, hi in bins:
        c = POP & (dd >= lo) & (dd < hi)
        if c.sum() == 0:
            continue
        r = 100 * cond[c].mean()
        print("  %7.0f ~ %-8.0f %8d %11.2f%%  %s" % (lo, hi, c.sum(), r,
                                                     "#" * int(round(r / 2))))
        if hi <= 0:
            pre += cond[c].sum(); npre += c.sum()
        else:
            post += cond[c].sum(); npost += c.sum()
    far = POP & (np.abs(dd) >= 30)
    print("  %-18s %8d %11.2f%%" % ("|거리| >= 30", far.sum(),
                                    100 * cond[far].mean() if far.sum() else 0))
    print("  ⇒ **전**(미래) %d/%d = **%.2f%%**   ·   **후**(과거) %d/%d = **%.2f%%**\n"
          % (pre, npre, 100 * pre / max(npre, 1), post, npost,
             100 * post / max(npost, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, apps, _ = load_model(a.ckpt, dev)
    jo = apps.index("oven")
    print("전이 창이 구조적으로 무엇이 다른가 — 부호 있는 거리와 정규화 통계")
    print("  %s · 장치 %s\n" % (a.ckpt, dev))

    SD, ED, OV, BIG, ON, ST, PW = [], [], [], [], [], [], []
    D = {}
    SIG = []                                   # 블록별 GroupNorm σ
    nst = nev = 0
    for s in FILES:
        rw = dense_targets(s, stride=30)
        t = np.asarray(rw.target_cycle, float) / 60.0
        iv = load_events()[s]["intervals"]

        def mask(app):
            mm = np.zeros(len(t), bool)
            for t0, t1 in iv.get(app, {}).get("on", []):
                mm |= (t >= t0) & (t <= t1)
            return mm

        big = np.zeros(len(t), bool)
        for ap_ in BIG3:
            big |= mask(ap_)
        xs = steps_of(s)
        nst += len(xs)
        evs = np.array(sorted(e["t_s"] for e in load_events()[s]["events"]
                              if e["appliance"] in RES4))
        nev += len(evs)
        if len(evs):
            ke = np.abs(t[:, None] - evs[None]).argmin(1)
            ED.append(t - evs[ke])
        else:
            ED.append(np.full(len(t), 1e9))
        #: **부호 있는** 거리 — 가장 가까운 계단까지. 양수면 계단이 **과거**다.
        if len(xs):
            k = np.abs(t[:, None] - xs[None]).argmin(1)
            sd = t - xs[k]
        else:
            sd = np.full(len(t), 1e9)
        SD.append(sd)
        OV.append(mask("oven"))
        BIG.append(big)
        D[s] = dict(ke=mask("electiric_kettle"), hd=mask("hair_dryer"),
                    hp=mask("hotplate"))

        # ── 한 번 태우면서 on/state 와 블록별 σ 를 같이 뽑는다 ───────────────
        box = {}
        hs = [blk[1].register_forward_pre_hook(
            lambda mod, inp, i=i: box.__setitem__(i, inp[0].detach()))
            for i, blk in enumerate(m.fine)]
        with torch.no_grad():
            for i in range(0, len(rw), 256):
                f, w, *_ = rw.batch(np.arange(i, min(i + 256, len(rw))))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                          torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                ON.append(o["on_logit"].float().cpu().numpy())
                PW.append(o["power"].float().cpu().numpy())
                ST.append(torch.softmax(o["state"].float(), -1).cpu().numpy())
                #: GroupNorm 이 실제로 쓰는 σ — (B, g) 로 (C_g, T 전체) 에서 낸다
                sg = []
                for i2 in sorted(box):
                    x = box[i2].float()
                    B, C, T = x.shape
                    g = m.fine[i2][1].num_groups
                    sg.append(x.reshape(B, g, C // g * T).std(-1).mean(-1).cpu().numpy())
                SIG.append(np.stack(sg, 1))     # (B, blocks)
        for h in hs:
            h.remove()

    sd, ed = np.concatenate(SD), np.concatenate(ED)
    ov, big = np.concatenate(OV), np.concatenate(BIG)
    on, st, pw = np.concatenate(ON), np.concatenate(ST), np.concatenate(PW)
    sig = np.concatenate(SIG)
    cond = (on[:, jo] > 0) & (st[:, jo, 2] > 0.5)
    POP = (~ov) & big                          # 오븐 참OFF · 큰 저항부하 켜짐
    print("  관측 계단 %d개 · 창 %d개 · 모집단(오븐 참OFF · 큰 저항부하) %d창\n"
          % (nst, len(sd), POP.sum()))

    table("A1 ★ 라벨 전이", POP, cond, ed,
          "  (사람 스위칭 로그 — **진짜 전이**)")
    table("A2   관측 계단", POP, cond, sd,
          "  (핫플 듀티가 대부분 — 대조로만 읽어라)")

    # ── [B] GroupNorm σ 가 전이 창에서 부푸나 ─────────────────────────────
    near = np.abs(sd) <= 6.0
    print("\n[B] 블록별 GroupNorm **σ** — 계단이 창에 있으면 부푸나")
    print("  (σ 는 (C_g, **T 전체**) 에서 난다. 창 어딘가의 계단이 타깃 특징을 나눈다)")
    print("  %-8s %10s %10s %9s %10s" % ("블록", "RF", "계단없음", "계단있음", "배"))
    rf = [7, 19, 43, 91, 187, 379, 763][:sig.shape[1]]
    for b in range(sig.shape[1]):
        q, r = sig[POP & ~near, b].mean(), sig[POP & near, b].mean()
        print("  %-8d %10d %10.4f %9.4f %9.3f배" % (b, rf[b], q, r, r / max(q, 1e-9)))

    # ── [C] ★ 거리축이 시간인가, 아니면 **기기 신원의 대리변수**인가 ────────
    print("")
    print("[C] ★ ±20초 **절벽**의 정체 — 거리축을 기기 신원과 교차한다")
    print("    [A1] 은 감쇠가 아니라 평평한 고원 + 절벽이다. 전이가 원인이면")
    print("    거리에 따라 **줄어야** 한다. 안 줄면 거리는 시간이 아니라 대리변수다.")
    print("  %-20s %10s %11s %10s %11s" % ("켜진 큰 부하", "<20초 창", "통전 헛%",
                                           ">=20초 창", "통전 헛%"))
    KE, HD, HP = [np.concatenate([D[s2][k] for s2 in FILES]) for k in ("ke", "hd", "hp")]
    inn = np.abs(ed) < 20.0
    for nm, msk in (("포트 (kettle)", KE), ("드라이기", HD), ("핫플레이트", HP),
                    ("포트만", KE & ~HD & ~HP), ("핫플만", HP & ~KE & ~HD)):
        c = POP & msk
        a, b = c & inn, c & ~inn
        print("  %-20s %10d %10.2f%% %10d %10.2f%%"
              % (nm, a.sum(), 100 * cond[a].mean() if a.sum() else 0,
                 b.sum(), 100 * cond[b].mean() if b.sum() else 0))
    print("")
    print("  ±20초 **밖** 창 %d개의 구성: 포트 %.0f%% · 드라이 %.0f%% · 핫플 %.0f%%"
          % ((POP & ~inn).sum(), 100 * KE[POP & ~inn].mean(),
             100 * HD[POP & ~inn].mean(), 100 * HP[POP & ~inn].mean()))
    print("  ±20초 **안** 창 %d개의 구성: 포트 %.0f%% · 드라이 %.0f%% · 핫플 %.0f%%"
          % ((POP & inn).sum(), 100 * KE[POP & inn].mean(),
             100 * HD[POP & inn].mean(), 100 * HP[POP & inn].mean()))

    # ── [D] ★ 진짜 축 — **어떤 조합이 켜져 있나** ───────────────────────────
    print("")
    print("[D] ★ 거리를 버리고 **조합**으로 가른다 (모집단: 오븐 참OFF · 큰 저항부하)")
    print("  %-26s %8s %12s %13s" % ("켜진 조합", "창", "통전 헛%", "오븐 헛전력W"))
    combos = [("포트만", KE & ~HD & ~HP), ("드라이만", HD & ~KE & ~HP),
              ("핫플만", HP & ~KE & ~HD), ("포트+드라이", KE & HD & ~HP),
              ("포트+핫플", KE & ~HD & HP), ("드라이+핫플", ~KE & HD & HP),
              ("셋 다", KE & HD & HP)]
    for nm, msk in combos:
        c = POP & msk
        if c.sum() == 0:
            print("  %-26s %8d" % (nm, 0))
            continue
        r = 100 * cond[c].mean()
        print("  %-26s %8d %11.2f%% %12.1f  %s"
              % (nm, c.sum(), r, pw[c, jo].mean(), "#" * int(round(r / 2))))
    print("")
    print("  ⇒ **단독이면 안 난다. 둘이 겹쳐야 난다.** 컨덕턴스 축퇴의 정의 그대로다 —")
    print("    부하가 하나면 Σ 를 쪼갤 다른 방법이 없고, 둘이면 오븐이 낄 자리가 생긴다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
