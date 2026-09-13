# -*- coding: utf-8 -*-
"""`--bg-head` 가 **진짜 세션 배경을 잡았는지, 아니면 슬랙이 됐는지** 직접 잰다 (13.84.38 ③).

13.84.38 에서 `--bg-head` 가 사슬을 7승 0패(+0.0101)로 올렸다. 그런데 파일별로 보면
대조보다 나쁘거나 같고(test_5 1.000 -> 0.942) 총점만 올랐다 — 13.59 의 함정 모양이다.
총점이 오른 이유가 **배경을 맞게 잡아서**인지 **자유 항이 잔차를 빨아서**인지 가른다.

배경 항이 진짜라면 13.84.35⑧ 이 실측에서 잰 모양을 따라야 한다:
```
전부 OFF 구간의 배경   파일 **안** 0.19~0.66 mA   (한 파일 안에서는 거의 완벽히 일정)
                      파일 **간** 3.6~22.6 mA    (세션마다 다르다)
```
그래서 넷을 잰다.
  ① 크기      0 이면 죽은 항, 상한 50mA 에 붙었으면 슬랙, 3.6~22.6 대역이면 말이 된다
  ② 산포      파일 안이 작고 파일 간이 커야 한다 (실측과 **같은 모양**). 뒤집히면 슬랙이다
  ③ 일치      파일 간 편차가 실측 배경의 파일 간 편차와 같은 방향인가 (차수별 상관·cos)
  ④ 슬랙검정  한 파일 **안에서** 조각 배경이 그 조각의 구성(총전력·켜진 개수)을 따라가는가.
              따라간다면 그것은 배경이 아니라 **기기 전류를 대신 받은 것**이다

⚠ 절대값을 실측 배경과 바로 견주면 안 된다. 손실에는 `noise_sig`(전역 상수)와 대기 항이 따로
  있어서 이 머리가 맡은 것은 **파일 간 나머지**다. 그래서 ③ 은 편차(파일평균 뺀 값)로 잰다.

    python -X utf8 src/run_diag_bghead.py results/seq_h38_bg.pt results/seq_h38_both.pt
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from scipy.stats import trim_mean

from src.model.chain import BgHead
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
#: 홀수차만 본다. 짝수차는 전파 정류라 홀수차의 1% 미만이다 (13.84.31 ①).
ORD = [1, 5, 9, 11, 13, 15]
#: 학습기의 조각 길이. `--chunk 64` 가 기본이고 배경은 조각당 값 하나다.
CHUNK = 64
MARGIN = 5          # 전이에서 띄울 초 (run_diag_ss_real 과 같은 규약)


def mA(v):
    """복소 페이저 배열 -> mA 크기."""
    return 1000.0 * np.abs(v)


def real_background():
    """파일마다 **전부 OFF** 구간의 배경 페이저 (15,) complex, A 단위."""
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    out = {}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        n = len(H)
        spec = ev[stem]
        on_any = np.zeros(n, bool)
        for x in spec["appliances_present"]:
            for t0, t1 in spec["intervals"].get(x, {}).get("on", []):
                on_any[int(t0 * FS):int(min(t1 * FS, n))] = True
        # 전이에서 MARGIN 초 띄운다 — 켜짐 구간 양쪽을 넓혀 배제한다.
        w = MARGIN * FS
        pad = np.convolve(on_any.astype(np.float32), np.ones(2 * w + 1), mode="same") > 0
        off = ~pad
        if off.sum() < FS:
            out[stem] = (None, 0)
            continue
        v = (trim_mean(H[off].real, 0.2, axis=0)
             + 1j * trim_mean(H[off].imag, 0.2, axis=0))
        out[stem] = (v, int(off.sum()))
    return out


def model_background(ck_path, dev, cache):
    """체크포인트의 배경 머리가 실측 파일마다 내는 값.

    반환: {stem: (전체 (15,) complex, 조각별 (n,15) complex, 조각별 총전력 (n,), 조각별 켜진수 (n,))}
    """
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    if ck.get("bghead") is None:
        return None, ck
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"])
    model.eval()
    bh = BgHead(ck["zdim"]).to(dev)
    bh.load_state_dict(ck["bghead"])
    bh.eval()

    if not cache:
        cache.update(real_windows(apps, ck["meta"]["grid_s"], dev))

    out = {}
    with torch.no_grad():
        for stem, d in cache.items():
            Z = []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float())
            z = torch.cat(Z)                                   # (N, Z)
            whole = bh(z[None])[0].cpu().numpy()               # (15, 2)
            n = (len(z) // CHUNK) * CHUNK
            if n >= CHUNK:
                zc = z[:n].reshape(-1, CHUNK, z.shape[1])
                seg = bh(zc).cpu().numpy()                     # (n_chunk, 15, 2)
            else:
                seg = whole[None]
            # 조각마다의 구성 — 총전력과 켜진 기기 수
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            P = np.asarray(r["power_features"])[:, 0]
            ti = np.clip((d["t"] * FS).astype(int), 0, len(P) - 1)
            nk = len(seg)
            pw = np.array([P[ti[i * CHUNK:(i + 1) * CHUNK]].mean() for i in range(nk)])
            non = np.array([d["y"][i * CHUNK:(i + 1) * CHUNK].sum(1).mean() for i in range(nk)])
            out[stem] = (whole[:, 0] + 1j * whole[:, 1],
                         seg[..., 0] + 1j * seg[..., 1], pw, non)
    return out, ck


def report(tag, mb, rb):
    oi = [o - 1 for o in ORD]
    stems = [s for s in FILES if s in mb]

    print("\n" + "=" * 96)
    print("## %s" % tag)

    # ── ① 크기 ─────────────────────────────────────────────────────────────
    print("\n① 크기 — 모델이 낸 파일별 배경 (mA). 상한은 50mA 다")
    print("  %-8s %s" % ("파일", "".join("  h%-6d" % o for o in ORD)))
    M = np.stack([mA(mb[s][0][oi]) for s in stems])            # (F, len(ORD))
    for s, row in zip(stems, M):
        print("  %-8s %s" % (s, "".join("  %7.2f" % v for v in row)))
    print("  %-8s %s" % ("최대비", "".join("  %6.0f%%" % (100 * v / 50.0) for v in M.max(0))))
    if M.max() < 0.05:
        print("  -> **죽은 항이다** (전부 0.05mA 미만). 배웠다고 볼 수 없다")
    elif M.max() > 40:
        print("  -> **상한에 붙었다** (>40mA). 슬랙의 전형이다")

    # ── ② 산포 ─────────────────────────────────────────────────────────────
    print("\n② 산포 — 파일 **안**(조각끼리) 대 파일 **간**. 실측은 0.19~0.66 대 3.6~22.6 mA 다")
    within = np.zeros(len(ORD))
    for j, o in enumerate(ORD):
        w = [np.abs(mb[s][1][:, o - 1] - mb[s][1][:, o - 1].mean()).mean() for s in stems]
        within[j] = 1000.0 * float(np.mean(w))
    across = np.array([1000.0 * float(np.abs(
        np.array([mb[s][0][o - 1] for s in stems])
        - np.mean([mb[s][0][o - 1] for s in stems])).mean()) for o in ORD])
    print("  %-10s %s" % ("", "".join("  h%-6d" % o for o in ORD)))
    print("  %-10s %s" % ("파일 안", "".join("  %7.2f" % v for v in within)))
    print("  %-10s %s" % ("파일 간", "".join("  %7.2f" % v for v in across)))
    rat = np.where(within > 1e-9, across / np.maximum(within, 1e-9), np.nan)
    print("  %-10s %s" % ("간/안", "".join("  %7.1f" % v for v in rat)))
    print("  -> 실측 배경은 이 비가 **10배 이상**이다 (0.19~0.66 대 3.6~22.6).")
    print("     1 근처거나 뒤집히면 이 항은 세션이 아니라 **창의 내용**을 따라간 것이다")

    # ── ③ 실측과의 일치 ────────────────────────────────────────────────────
    print("\n③ 일치 — 파일 간 **편차**가 실측 배경의 편차와 같은 방향인가")
    have = [s for s in stems if rb[s][0] is not None]
    if len(have) < 3:
        print("  전부 OFF 구간이 3파일 미만이라 못 잰다")
    else:
        print("  (전부 OFF 표본: %s)" % " · ".join("%s %d초" % (s, rb[s][1] // FS) for s in have))
        print("  %-10s %s" % ("", "".join("  h%-6d" % o for o in ORD)))
        rm, cs = [], []
        for o in ORD:
            a = np.array([mb[s][0][o - 1] for s in have])
            b = np.array([rb[s][0][o - 1] for s in have])
            a = a - a.mean()
            b = b - b.mean()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            cs.append(float(np.real(np.vdot(b, a)) / den) if den > 0 else np.nan)
            rm.append(1000.0 * float(np.abs(b).mean()))
        print("  %-10s %s" % ("실측 편차", "".join("  %7.2f" % v for v in rm)))
        print("  %-10s %s" % ("cos", "".join("  %7.2f" % v for v in cs)))
        print("  -> cos 가 0 근처면 방향이 안 맞는 것이다. 파일 %d개라 ±0.5 는 우연이다" % len(have))

    # ── ④ 슬랙 검정 ────────────────────────────────────────────────────────
    print("\n④ 슬랙 검정 — 파일 **안에서** 조각 배경이 그 조각의 구성을 따라가는가")
    print("  %-8s %6s %s" % ("파일", "조각", "".join("  h%-6d" % o for o in ORD)))
    allr = []
    for s in stems:
        seg, pw = mb[s][1], mb[s][2]
        if len(seg) < 4:
            print("  %-8s %6d  (조각이 모자라 못 잰다)" % (s, len(seg)))
            continue
        rr = []
        for o in ORD:
            x = np.abs(seg[:, o - 1])
            rr.append(np.corrcoef(x, pw)[0, 1] if x.std() > 1e-12 else np.nan)
        allr.append(rr)
        print("  %-8s %6d %s" % (s, len(seg), "".join("  %7.2f" % v for v in rr)))
    if allr:
        m = np.nanmean(np.abs(np.array(allr)), axis=0)
        print("  %-8s %6s %s" % ("|r| 평균", "", "".join("  %7.2f" % v for v in m)))
        print("  -> |r| 이 크면 배경이 **총전력을 따라간다** = 기기 전류를 대신 받는 슬랙이다.")
        print("     진짜 세션 배경은 그 파일 안에서 상수여야 하므로 r 이 정의되지도 않을 만큼 평평하다")


def main():
    cks = sys.argv[1:] or ["results/seq_h38_bg.pt"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("실측 전부-OFF 배경을 잰다 (전이에서 %d초 띄움)..." % MARGIN, flush=True)
    rb = real_background()
    cache = {}
    for c in cks:
        mb, ck = model_background(c, dev, cache)
        if mb is None:
            print("\n%s 에는 배경 머리가 없다 (bg_head=%s)" % (c, ck.get("bg_head")))
            continue
        report("%s  (epoch %s · w_bg %s)" % (c, ck.get("epoch"), ck.get("w_bg")), mb, rb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
