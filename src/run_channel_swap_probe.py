# -*- coding: utf-8 -*-
"""실측/합성 간극을 **채널군 교환**으로 국소화한다 (12.179).

    python -m src.run_channel_swap_probe --app minipc --site B
    python -m src.run_channel_swap_probe --app hair_dryer --site B --p-range 900,1150

같은 ON 집합·비슷한 전력·정상 상태인 실측 창과 합성 창을 짝짓고, 실측 창의 채널군
하나를 합성 짝의 값으로 바꿔 모델(기본 `cnn_ac`)의 답이 어디서 뒤집히는지 본다.
분포 차(d′)는 **어디가 다른지**를, 교환은 **모델이 어디를 읽는지**를 준다 — 둘은
다르다 (규칙 70).

12.179 에서 얻은 것: 미니PC 는 Re/Im h1~h15 를 통째로 바꾸면 1.2 -> 14.5W 로 돌아오고
h5~h9 는 게이트를, h11~h15 는 p_raw 를 연다. 드라이기 강풍은 위상 φ3~φ9 (0.24 ->
0.83) 와 h11~h15 (0.56) 가 게이트를 연다. 짝수차·P·Q·V·리플·광역은 아무것도 안 바꾼다.
"""
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

RES = ("electiric_kettle", "oven", "hair_dryer", "hotplate", "air_conditioner", "fan")
SMPS = ("minipc", "laptop_charger", "beam_projector")
SITE = {"A": ("test_5", "test_6", "test_7", "test_8", "test_13"),
        "B": ("test_15", "test_16", "test_17", "test_18")}
SHORT = {"hair_dryer": "드라", "electiric_kettle": "포트", "hotplate": "핫플", "minipc": "PC",
         "beam_projector": "프로", "laptop_charger": "충전", "oven": "오븐", "fan": "선풍", "air_conditioner": "AC"}
NAMES = {**{i: f"Re h{i + 1}" for i in range(15)}, **{15 + i: f"Im h{i + 1}" for i in range(15)},
         30: "P", 31: "Q", 32: "V", 33: "|I3|/|I1|", 34: "|I5|/|I1|", 35: "|I2|/|I1|", 36: "P리플0.5s", 37: "P리플2.5s",
         38: "cosφ3", 39: "sinφ3", 40: "cosφ5", 41: "sinφ5", 42: "cosφ7", 43: "sinφ7", 44: "cosφ9", 45: "sinφ9",
         46: "PF", 47: "|I9|/|I3|", 48: "강하3s", 49: "강하5.5s", 50: "반파|I2|-|I4|", 51: "|I2|"}
GROUPS = {"Re/Im h1~h15": list(range(30)), "  h1": [0, 15], "  h3": [2, 17], "  h5~h9": [4, 6, 8, 19, 21, 23],
          "  h11~h15": [10, 12, 14, 25, 27, 29], "  짝수차": [1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 26, 28],
          "P,Q,V": [30, 31, 32], "비율 33-35": [33, 34, 35], "리플 36-37": [36, 37], "위상 φ3~φ9": list(range(38, 46)),
          "PF,|I9|/|I3|": [46, 47], "강하 48-49": [48, 49], "반파 50": [50], "|I2| 51": [51], "광역 갈래": "wide"}
STEADY = 0.08   # 창 안 asinh(P/100) 범위 — 이보다 크면 전이가 들어 있다


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--app", required=True)
    ap.add_argument("--site", default="B", choices=("A", "B"))
    ap.add_argument("--ckpt", default="results/cnn_ac.pt")
    ap.add_argument("--cache", default="cache/train60_ac")
    ap.add_argument("--p-range", default="", metavar="LO,HI", help="관측 전력 범위 (저항 기기용)")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    a = ap.parse_args()
    import torch
    from src.run_gate_check import load_model
    from src.model.realdata import dense_targets
    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, _ = load_model(a.ckpt, dev); j = apps.index(a.app)
    others = tuple(x for x in RES if x != a.app)
    is_smps = a.app in SMPS
    plo, phi = (float(x) for x in a.p_range.split(",")) if a.p_range else (0.0, 1e9)

    # ── 실측 창: app ON, 다른 저항 없음, 정상 상태. 동반 SMPS 집합을 라벨로 남긴다
    RF, RW, RP, RL = [], [], [], []
    for stem in SITE[a.site]:
        if stem not in ev or a.app not in ev[stem]["intervals"]:
            continue
        n = int(ev[stem]["cycles"]); iv = ev[stem]["intervals"]
        on = _mask(iv[a.app].get("on", []), n); big = np.zeros(n, bool)
        for x in others:
            if x in iv:
                big |= _mask(iv[x].get("on", []) + iv[x].get("uncertain", []), n)
        comp = {x: (_mask(iv[x].get("on", []), n) if x in iv else np.zeros(n, bool)) for x in SMPS if x != a.app}
        rw = dense_targets(stem, stride=30); t = rw.target_cycle
        steady = (rw.fine[:, 30].max(1) - rw.fine[:, 30].min(1)) < STEADY
        for i, ti in enumerate(t):
            lo, hi = max(0, ti - 360), ti + 360
            if not (steady[i] and on[lo:hi].all() and not big[lo:hi].any() and plo <= rw.p_observed[i] <= phi):
                continue
            if any(comp[x][lo:hi].any() and not comp[x][lo:hi].all() for x in comp):
                continue                                   # 동반이 창 안에서 바뀐다
            lab = "+".join(SHORT[x] for x in comp if comp[x][lo:hi].all())
            RF.append(rw.fine[i]); RW.append(rw.wide[i]); RP.append(rw.p_observed[i]); RL.append((stem, lab))
    if not RF:
        raise SystemExit("실측 창이 없습니다")
    RF, RW, RP = np.stack(RF), np.stack(RW), np.array(RP)
    labs = np.array(["+".join(filter(None, (s, l))) for s, l in RL])
    print("실측 창:", {l: int((labs == l).sum()) for l in np.unique(labs)})

    # ── 합성 창: 같은 ON 집합, 전력 근접, 정상 상태
    meta = json.load(open(a.cache + "/meta.json", encoding="utf-8")); ca = meta["appliances"]
    yo = np.asarray(np.load(a.cache + "/y_on.npy", mmap_mode="r")); po = np.asarray(np.load(a.cache + "/p_observed.npy", mmap_mode="r"))
    yp = np.asarray(np.load(a.cache + "/y_power.npy", mmap_mode="r"))
    fine_mm = np.load(a.cache + "/fine.npy", mmap_mode="r"); wide_mm = np.load(a.cache + "/wide.npy", mmap_mode="r")
    jr = [ca.index(x) for x in others]; cj = ca.index(a.app)
    rng = np.random.default_rng(0); pairs = []
    for ri in rng.choice(len(RP), size=min(a.pairs, len(RP)), replace=False):
        _, lab = RL[ri]
        sel = (yo[:, cj] > 0) & (yo[:, jr].sum(1) == 0)
        if not is_smps:
            sel &= yp[:, cj] > 0.8 * RP[ri]
        for x in SMPS:
            if x != a.app:
                sel &= ((yo[:, ca.index(x)] > 0) == (SHORT[x] in lab))
        idx = np.where(sel)[0]
        for i in idx[np.argsort(np.abs(po[idx] - RP[ri]))][:40]:
            f = np.asarray(fine_mm[i])
            if (f[30].max() - f[30].min()) < STEADY:
                pairs.append((ri, i)); break
    pairs = np.array(pairs)
    if not len(pairs):
        raise SystemExit("짝을 못 만들었습니다")
    SF = np.stack([np.asarray(fine_mm[i]) for i in pairs[:, 1]]).astype(np.float32)
    SW = np.stack([np.asarray(wide_mm[i]) for i in pairs[:, 1]]).astype(np.float32)
    RFp, RWp = RF[pairs[:, 0]], RW[pairs[:, 0]]
    print(f"짝 {len(pairs)}  관측 전력 실측 {RP[pairs[:, 0]].mean():.1f}W / 합성 {po[pairs[:, 1]].mean():.1f}W"
          f"  합성 {SHORT[a.app]} 전력 {yp[pairs[:, 1], cj].mean():.1f}W")

    @torch.no_grad()
    def run(f, w):
        ft = torch.from_numpy(np.ascontiguousarray(f)).to(dev); wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(ft, wt)
        return torch.sigmoid(o["on_logit"]).float().cpu().numpy(), o["power_raw"].float().cpu().numpy()

    def rep(tag, g, r):
        top = sorted(range(len(apps)), key=lambda k: -(g[:, k] * r[:, k]).mean())[:3]
        print(f"{tag:30s} {SHORT[a.app]} 게이트 {g[:, j].mean():.3f} p_raw {r[:, j].mean():7.2f} 전력 {(g[:, j] * r[:, j]).mean():7.2f}"
              + " | " + " ".join(f"{SHORT[apps[k]]} {(g[:, k] * r[:, k]).mean():.0f}" for k in top))

    g, r = run(RFp, RWp); rep("실측", g, r)
    g, r = run(SF, SW); rep("합성 짝", g, r)
    mr, ms = RFp.mean(2), SF.mean(2)
    print("\n[채널별 창 평균의 분포 차 d′ = (실측−합성)/pooled σ, |d′|>1]")
    d = (mr.mean(0) - ms.mean(0)) / (np.sqrt((mr.var(0) + ms.var(0)) / 2) + 1e-9)
    for c in np.argsort(-np.abs(d)):
        if abs(d[c]) > 1.0:
            print(f"   ch{c:2d} {NAMES[c]:14s} d′ {d[c]:+6.2f}   실측 {mr[:, c].mean():+8.4f}  합성 {ms[:, c].mean():+8.4f}")
    print("\n[실측 창의 채널군을 합성 짝의 값으로 바꾼다]")
    for name, chs in GROUPS.items():
        f, w = RFp.copy(), RWp.copy()
        if chs == "wide":
            w = SW.copy()
        else:
            f[:, chs] = SF[:, chs]
        g, r = run(f, w); rep(f"실측←합성 {name}", g, r)
    print("\n[합성 창의 채널군을 실측 값으로 바꾼다]")
    for name, chs in GROUPS.items():
        f, w = SF.copy(), SW.copy()
        if chs == "wide":
            w = RWp.copy()
        else:
            f[:, chs] = RFp[:, chs]
        g, r = run(f, w); rep(f"합성←실측 {name}", g, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
