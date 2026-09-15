# -*- coding: utf-8 -*-
"""**창을 토막이 아니라 훑는다** — 시간 sweep 과 채널x시간 지도 (14.158).

사용자: *"근데 600창을 다 조정해본 거야? 이렇게 빨리 가능한 일이었어?"*

14.157 은 세밀 창 600샘플을 **2초씩 5토막**으로 묶어서 봤다 (후보 16개). 여기서는
**훑는다.**

```
  [A] 시간 sweep   폭 `--width`(기본 0.5초) 를 `--step`(0.25초) 씩 밀며 **창 전체 60초**를
                   덮는다. 그 자리만 앞 창 값으로 되돌리고 응답을 잰다 -> 241자리
  [B] 채널 x 시간   원시 45채널(또는 무리) x 세밀 10초의 0.5초 칸 20개 = **900칸**.
                   어느 채널이 어느 시각에서 답을 정하는지 2차원으로
  [C] 정밀 sweep    [A] 에서 가장 센 자리 주변을 `--fine-step`(0.05초) 로 다시
```

전부 **앞 창의 실제 값**으로 갈아 끼운다 (0 으로 죽이지 않는다). 원시에서 바꾸고
`build_inputs` 를 다시 돌리므로 파생 채널까지 정합이다.

    python -X utf8 -m src.run_diag_finesweep --stem test_2 --at 231.4 --gap 2.5 \
        --app oven --ckpt results/cnn_pcap_s0.pt
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import build_inputs, fine_target_index  # noqa: E402
from src.model.realdata import RealWindows, target_index  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_diag_flipedge import SH, raw_groups, raw_names  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FS = 60
WIN = 3600
FINE = 600


def run_cands(m, base, edits, dev, chunk=96):
    """`edits` = [(채널색인 또는 None, 시작, 끝)] 를 앞 창 값으로 갈아 끼운 창들을 돌린다."""
    G, P = [], []
    for i in range(0, len(edits), chunk):
        blk = []
        for idx, i0, i1, src in edits[i:i + chunk]:
            y = base.copy()
            if idx is None:
                y[:, i0:i1] = src[:, i0:i1]
            else:
                y[np.ix_(idx, np.arange(i0, i1))] = src[np.ix_(idx, np.arange(i0, i1))]
            blk.append(y)
        f, w = build_inputs(np.stack(blk))
        with torch.no_grad():
            o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        P.append(o["power"].float().cpu().numpy())
        del blk, f, w
    return np.concatenate(G), np.concatenate(P)


def spark(v, lo, hi):
    lv = " .:-=+*#@"
    r = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    return "".join(lv[min(int(q * (len(lv) - 1)), len(lv) - 1)] for q in r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_pcap_s0.pt")
    ap.add_argument("--stem", default="test_2")
    ap.add_argument("--at", type=float, default=231.4)
    ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--app", default="oven")
    ap.add_argument("--also", default="electiric_kettle")
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--fine-step", type=float, default=0.05)
    ap.add_argument("--bin", type=float, default=0.5, help="[B] 의 시간 칸")
    ap.add_argument("--per-channel", action="store_true",
                    help="[B] 를 무리가 아니라 **45채널 낱개**로 (900칸)")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 (14.160) — 분포 밖 시험용. "
                         "안 주면 체크포인트가 적어 둔 값을 따라간다")
    a = ap.parse_args()
    from src.run_gate_check import sync_even_median
    sync_even_median(a.ckpt, a.even_median)   #: 창을 짓기 **전에** 전역을 맞춘다

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt, map_location="cpu",
                           weights_only=False)["appliances"])
    ev = json.load(open("processed_data/real_events.json",
                        encoding="utf-8"))["files"][a.stem]
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % a.stem)
    x = RealWindows._to_33ch(raw)
    off = target_index(WIN)
    ti = fine_target_index()
    f0 = WIN - FINE

    def win(t_s):
        c = int(round(t_s * FS))
        return x[:, c - off:c - off + WIN].copy()

    w_lo, w_hi = win(a.at - a.gap), win(a.at + a.gap)
    k = apps.index(a.app)
    k2 = apps.index(a.also) if a.also in apps else None
    m = load_model(a.ckpt, dev)[0]
    m.eval()
    f2, wd2 = build_inputs(np.stack([w_lo, w_hi]))
    with torch.no_grad():
        o = m(torch.from_numpy(f2).to(dev), torch.from_numpy(wd2).to(dev))
    G0 = torch.sigmoid(o["on_logit"]).float().cpu().numpy()
    P0 = o["power"].float().cpu().numpy()
    print("%s · %s · 경계 %.2f초 (앞 %.2f / 뒤 %.2f) · %s"
          % (a.stem, a.ckpt.split("/")[-1].replace(".pt", ""), a.at,
             a.at - a.gap, a.at + a.gap, SH.get(a.app, a.app)))
    print("  기준: 뒤 창 게이트 %.3f · 전력 %.0fW   (앞 창 %.3f · %.0fW)"
          % (G0[1, k], P0[1, k], G0[0, k], P0[0, k]))

    # 창 좌표 -> 타깃 기준 시각
    def tau(i):
        return (i - (f0 + ti)) / FS

    # ── [A] 시간 sweep ──────────────────────────────────────────────────
    wv = int(round(a.width * FS))
    sv = int(round(a.step * FS))
    starts = list(range(0, WIN - wv + 1, sv))
    ed = [(None, s, s + wv, w_lo) for s in starts]
    G, P = run_cands(m, w_hi, ed, dev)
    pw = P[:, k]
    print("\n[A] **시간 sweep** — 폭 %.2f초를 %.2f초씩 밀며 **창 60초 전부** (%d자리)"
          % (a.width, a.step, len(starts)))
    print("  전력이 가장 많이 떨어지는 자리 10")
    print("    %-18s %10s %10s %10s" % ("구간(타깃기준)", "게이트", "전력W", SH.get(a.also, "짝")))
    for j in np.argsort(pw)[:10]:
        s = starts[j]
        print("    %+7.2f ~ %+7.2f초 %10.3f %10.0f %10.0f"
              % (tau(s), tau(s + wv), G[j, k], pw[j],
                 P[j, k2] if k2 is not None else np.nan))
    # 프로필
    tl = np.array([tau(s + wv / 2) for s in starts])
    for lo, hi, ttl in ((-54, -4, "광역 전용 −54~−4초"), (-4, 6, "세밀 −4~+6초")):
        msk = (tl >= lo) & (tl <= hi)
        if msk.sum() < 4:
            continue
        v = pw[msk]
        print("  %s  전력 %.0f~%.0fW" % (ttl, v.min(), v.max()))
        print("    " + spark(v, pw.min(), pw.max()))
    # ── [C] 정밀 sweep ──────────────────────────────────────────────────
    j0 = int(np.argmin(pw))
    c0 = starts[j0]
    wf = max(int(round(a.fine_step * FS)), 1)
    st2 = [s for s in range(max(c0 - 3 * wv, 0), min(c0 + 3 * wv, WIN - wf), wf)]
    ed2 = [(None, s, s + wf, w_lo) for s in st2]
    G2, P2 = run_cands(m, w_hi, ed2, dev)
    print("\n[C] **정밀 sweep** — 가장 센 자리 둘레를 폭 %.2f초 · %.2f초 눈금으로 (%d자리)"
          % (a.fine_step, a.fine_step, len(st2)))
    v = P2[:, k]
    print("  %+.2f ~ %+.2f초 · 전력 %.0f~%.0fW" % (tau(st2[0]), tau(st2[-1] + wf),
                                                 v.min(), v.max()))
    print("    " + spark(v, v.min(), v.max()))
    jb = int(np.argmin(v))
    print("  최저: %+.2f ~ %+.2f초 에서 전력 %.0fW (기준 %.0fW)"
          % (tau(st2[jb]), tau(st2[jb] + wf), v[jb], P0[1, k]))

    # ── [B] 채널 x 시간 지도 ────────────────────────────────────────────
    bv = int(round(a.bin * FS))
    bins = list(range(f0, WIN - bv + 1, bv))
    if a.per_channel:
        items = [("ch%02d %s" % (i, raw_names()[i]), [i]) for i in range(45)]
    else:
        items = list(raw_groups().items())
    ed3 = [([*idx], b, b + bv, w_lo) for _, idx in items for b in bins]
    G3, P3 = run_cands(m, w_hi, ed3, dev)
    M = P3[:, k].reshape(len(items), len(bins))
    print("\n[B] **채널 x 시간 지도** — %d채널/무리 x %d칸(%.2f초) = **%d칸**"
          % (len(items), len(bins), a.bin, M.size))
    print("  값 = 그 채널을 그 칸에서만 되돌렸을 때의 전력(W). 기준 %.0fW" % P0[1, k])
    hdr = "    %-22s " % "채널/무리"
    hdr += "".join("%+.0f" % tau(b + bv / 2) if (b - f0) % (4 * bv) == 0 else " "
                   for b in bins)
    print(hdr + "   최저W")
    lo_, hi_ = M.min(), P0[1, k]
    for i, (nm_, _) in enumerate(items):
        print("    %-22s %s  %6.0f" % (nm_, spark(M[i], lo_, hi_), M[i].min()))
    print("    (진할수록 **많이 떨어진다** = 그 채널이 그 시각에서 답을 정한다)")
    best = np.unravel_index(np.argmin(M), M.shape)
    print("  최저 칸: **%s** @ %+.2f ~ %+.2f초 -> %.0fW"
          % (items[best[0]][0], tau(bins[best[1]]), tau(bins[best[1]] + bv),
             M[best]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
