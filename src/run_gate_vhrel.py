# -*- coding: utf-8 -*-
"""`--harm-vhrel-anchor` 의 **배선** 관문 (14.56).

⚠⚠ **함수를 재지 말고 배선을 재라** (983506 이 45분을 버렸다). 여기서 제일 위험한 줄은
② 다 — `_vhrel_from_fine` 이 `inputs.build_inputs` 의 눈금을 **정확히 되돌리는가**.
되돌리기가 틀리면 손실은 조용히 **엉뚱한 파형**으로 앵커한다.

```
① 녹화 파형 `v_h_rel` 이 자리별로 갈린다 (오븐·핫플 D ~0.7%% · 나머지 E ~3%%)
② ★ **왕복 검정** — 아는 V_h 를 `build_inputs` 에 넣고 되돌려 원값과 같은가
③ frac=0 이면 `_harm_pred_active` 가 **비트 동일**
④ frac>0 이면 델타가 정확히 `power·sig_1·frac·(rel_창 − rel_녹화)` 이고
   **안 걸린 기기(SMPS 다섯)는 정확히 0**
⑤ `rel_창 == rel_녹화` 면 frac 과 무관하게 **정확히 0** (물리 항등)
⑥ `run_train_cnn` 이 실제로 넘긴다
```
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import (RAW_CHANNELS, VOLT_IM0, VOLT_ORDERS,  # noqa: E402
                              VOLT_RE0, build_inputs)
from src.model.losses import NILMLoss  # noqa: E402
from src.model.net import (harmonic_scales, harmonic_signature_vhrel,  # noqa: E402
                           harmonic_signatures, harmonic_signatures_by_state)
from src.run_baseline import S_I  # noqa: E402
from src.run_train_cnn import FINE_TPOS, _vhrel_from_fine, _vnorm_exp  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

OK, NG = "✅", "❌"
FAIL = []


def ck(name, ok, note=""):
    print(("  " + (OK if ok else NG) + " ") + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _cx(a):
    """(..., 2) [Re, Im] -> 복소."""
    a = np.asarray(a, dtype=np.float64)
    return a[..., 0] + 1j * a[..., 1]


def main() -> int:
    print("`sig` 파형 앵커 배선 관문 (14.56)\n")
    apps = sorted(S_I)
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    vh = harmonic_signature_vhrel(pool, apps)                  # (K,15,2)

    # ── ① 녹화 파형이 자리별로 갈린다 ──────────────────────────────────
    c = _cx(vh)
    h3 = {a: 100 * abs(c[j, 2]) for j, a in enumerate(apps)}
    d_site = [h3["oven"], h3["hotplate"]]
    e_site = [h3[a] for a in apps if a not in ("oven", "hotplate")]
    ck("① 녹화 파형이 자리별로 갈린다 (h3: D 무리 아래 1% 아래 E 무리)",
       max(d_site) < 1.0 and min(e_site) > 2.0,
       "오븐 %.2f · 핫플 %.2f | E 무리 %.2f~%.2f (백분율)"
       % (h3["oven"], h3["hotplate"], min(e_site), max(e_site)))
    ck("① h1 은 정의상 1+0j",
       bool(np.allclose(vh[:, 0, 0], 1.0) and np.allclose(vh[:, 0, 1], 0.0)))

    # ── ② ★ 왕복 검정 ─────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    n, W = 4, 3600
    raw = (rng.standard_normal((n, RAW_CHANNELS, W)) * 0.01).astype(np.float32)
    v1 = (210.0 + rng.standard_normal((n, W))).astype(np.float32)
    for s_, h_ in enumerate(VOLT_ORDERS):
        if h_ == 1:
            vr, vi = v1, np.zeros_like(v1)
        else:                                   # 실측 크기대로 (0.5~4V)
            vr = ((0.5 + 3.5 * rng.random((n, 1))) * np.ones_like(v1)).astype(np.float32)
            vi = ((-2.0 + 4.0 * rng.random((n, 1))) * np.ones_like(v1)).astype(np.float32)
        raw[:, VOLT_RE0 + s_] = vr
        raw[:, VOLT_IM0 + s_] = vi
    from src.model.inputs import FINE_CYCLES
    fine, _ = build_inputs(raw)
    got = _vhrel_from_fine(torch.from_numpy(np.asarray(fine)), True).numpy()
    # ⚠⚠ `build_inputs` 는 창의 **뒤 600사이클**만 쓴다 (`seg = x[:, :, -FINE_CYCLES:]`).
    #   그래서 원시 쪽 기준 색인은 `W − 600 + FINE_TPOS` 다. 처음에 `FINE_TPOS` 로 쟀다가
    #   1.36e-4 가 나왔는데, 그것은 왕복 오차가 아니라 **다른 사이클의 V_1 로 나눈 것**이었다
    #   (창 안 V 산포 1V / 210V x rel 0.02 = 1e-4 — 크기까지 정확히 맞았다).
    #   기준을 맞추면 float32 왕복이라 **1e-6 아래**여야 한다.
    T0 = W - FINE_CYCLES + FINE_TPOS
    m1 = np.hypot(raw[:, VOLT_RE0, T0], raw[:, VOLT_IM0, T0])
    err = []
    for s_, h_ in enumerate(VOLT_ORDERS):
        want = (np.ones(n, dtype=np.complex128) if h_ == 1 else
                (raw[:, VOLT_RE0 + s_, T0] + 1j * raw[:, VOLT_IM0 + s_, T0]) / m1)
        err.append(float(np.abs(_cx(got[:, h_ - 1]) - want).max()))
    ck("② ★ 왕복 검정 — `_vhrel_from_fine` 이 `build_inputs` 를 정확히 되돌린다",
       max(err) < 1e-6, "차수별 최대 차 %.2e (허용 1e-6)" % max(err))
    ck("② 관측 안 되는 차수(h13·h15·짝수)는 0 이다",
       bool(np.allclose(got[:, [1, 3, 5, 11, 12, 13, 14], :], 0.0)))

    # ── ③④⑤ 손실 끝단 ──────────────────────────────────────────────────
    sig = harmonic_signatures(pool, apps)
    sst, _ = harmonic_signatures_by_state(pool, apps)
    hs = harmonic_scales(pool, apps)
    ex = np.asarray(_vnorm_exp(apps, "RESISTIVE"), dtype=np.float64)
    on = (ex != 0).astype(np.float32)
    K, S = len(apps), sst.shape[1]
    kw = dict(s_i=torch.tensor([S_I[a] for a in apps]),
              signatures=torch.from_numpy(sig),
              signatures_state=torch.from_numpy(sst),
              harm_scale=torch.from_numpy(hs), harm_sig_vnorm=True,
              harm_vnorm_exp=torch.as_tensor(ex, dtype=torch.float32))
    torch.manual_seed(0)
    B = 3
    mix = torch.softmax(torch.randn(B, K, S), -1)
    stt = torch.rand(B, K, S) * 500 + 50
    praw = (mix * stt).sum(-1)
    out = {"power_raw": praw, "power_mix": mix, "power_states": stt}
    V_WIN = 210.0
    vrel = torch.full((B,), V_WIN / 222.0)
    A = NILMLoss(**kw)

    # ③ frac=0
    C0 = NILMLoss(**kw, harm_vhrel_rec=torch.from_numpy(vh),
                  harm_vhrel_frac=0.0, harm_vhrel_on=torch.from_numpy(on))
    rw_same = torch.from_numpy(vh[apps.index("oven")])[None].repeat(B, 1, 1).contiguous()
    A._vrel = vrel; C0._vrel = vrel
    A._vhrel = None; C0._vhrel = rw_same
    ck("③ frac=0 이면 **비트 동일** (`use_vhrel` 이 False)",
       (not C0.use_vhrel)
       and bool(torch.equal(A._harm_pred_active(out, praw),
                            C0._harm_pred_active(out, praw))))

    # ⑤ rel_창 == rel_녹화 -> 그 기기 기여가 정확히 0 변화
    C1 = NILMLoss(**kw, harm_vhrel_rec=torch.from_numpy(vh),
                  harm_vhrel_frac=1.0, harm_vhrel_on=torch.from_numpy(on))
    ck("④ `use_vhrel` 이 켜졌다", C1.use_vhrel)
    j_ov = apps.index("oven")
    p_ov = torch.zeros(B, K); p_ov[:, j_ov] = 300.0
    o_ov = {"power_raw": p_ov, "power_mix": mix, "power_states": stt}
    A._vrel = vrel; C1._vrel = vrel
    A._vhrel = None; C1._vhrel = rw_same
    ck("⑤ `rel_창 == rel_녹화` 면 frac=1 이어도 **정확히 0** 변화 (물리 항등)",
       bool(torch.equal(A._harm_pred_active(o_ov, p_ov),
                        C1._harm_pred_active(o_ov, p_ov))))

    # ④ 기기별. ⚠ **두 갈래를 나눠 잰다** — 상태 지문이 있으면 손실은 `sgs`(상태별)를
    #   타므로 기기 수준 `sig_1` 으로 쓴 닫힌 식과 안 맞는다 (처음에 그렇게 짜서 5~17%
    #   어긋났다). ⓐ 상태 갈래를 끄고 닫힌 식과 정확히 맞추고, ⓑ 상태 갈래에서는
    #   **frac 에 정확히 비례**하는지로 본다 (닫힌 식 없이도 서는 검정이다).
    FR = 0.5
    kwn = {k: v for k, v in kw.items() if k != "signatures_state"}
    An = NILMLoss(**kwn)
    Cn = NILMLoss(**kwn, harm_vhrel_rec=torch.from_numpy(vh),
                  harm_vhrel_frac=FR, harm_vhrel_on=torch.from_numpy(on))
    rw = torch.from_numpy(vh[apps.index("minipc")])[None].repeat(B, 1, 1).contiguous()
    rwc = _cx(rw[0].numpy())
    print("    ⓐ 상태 갈래 **끔** — 닫힌 식 `power·sig_1·frac·(rel_창−rel_녹화)`")
    print("    %-20s%8s%14s%14s%7s" % ("기기", "걸리나", "예상 Σ|Δ|", "실측 Σ|Δ|", ""))
    bad = 0
    for j2, app in enumerate(apps):
        p1 = torch.zeros(B, K); p1[:, j2] = 300.0
        o1 = {"power_raw": p1}
        An._vrel = vrel; Cn._vrel = vrel
        An._vhrel = None; Cn._vhrel = rw
        got_d = float((Cn._harm_pred_active(o1, p1)
                       - An._harm_pred_active(o1, p1)).abs().sum())
        want_d = 0.0
        if on[j2]:
            sg = _cx(sig[j2]) * (V_WIN / 222.0) ** ex[j2]
            for h_ in VOLT_ORDERS:
                if h_ == 1:
                    continue
                z = sg[0] * ((rwc[h_ - 1] - _cx(vh[j2])[h_ - 1]) * FR) * 300.0
                want_d += (abs(z.real) + abs(z.imag)) * B
        good = abs(got_d - want_d) < max(2e-4, 0.005 * max(want_d, 1e-12))
        if not good:
            bad += 1
        print("    %-20s%8s%14.5f%14.5f%7s" %
              (app[:20], "O" if on[j2] else ".", want_d, got_d, OK if good else NG))
    ck("④ⓐ 상태 갈래 끔 — 델타가 닫힌 식과 맞고 **SMPS 다섯은 정확히 0**", bad == 0)

    # ⓑ 상태 갈래 — frac 에 정확히 비례하는가 (닫힌 식 없이 서는 검정)
    lin, zero = [], []
    for j2, app in enumerate(apps):
        p1 = torch.zeros(B, K); p1[:, j2] = 300.0
        o1 = {"power_raw": p1, "power_mix": mix, "power_states": stt}
        A._vrel = vrel; A._vhrel = None
        base = A._harm_pred_active(o1, p1)
        ds = []
        for f in (0.5, 1.0):
            Cs = NILMLoss(**kw, harm_vhrel_rec=torch.from_numpy(vh),
                          harm_vhrel_frac=f, harm_vhrel_on=torch.from_numpy(on))
            Cs._vrel = vrel; Cs._vhrel = rw
            ds.append(float((Cs._harm_pred_active(o1, p1) - base).abs().sum()))
        if on[j2]:
            lin.append(abs(ds[1] - 2 * ds[0]) <= max(1e-6, 1e-4 * ds[1]) and ds[1] > 1e-6)
        else:
            zero.append(ds[0] == 0.0 and ds[1] == 0.0)
    ck("④ⓑ 상태 갈래 — 저항 넷의 델타가 **frac 에 정확히 비례**한다", all(lin),
       "%d/4" % sum(lin))
    ck("④ⓑ 상태 갈래 — SMPS 다섯은 **정확히 0**", all(zero), "%d/5" % sum(zero))

    # ⑥ 학습기 배선
    src = Path("src/run_train_cnn.py").read_text(encoding="utf-8")
    ck("⑥ `run_train_cnn` 이 실제로 넘긴다 (rec·frac·on)",
       all(k in src for k in ("harm_vhrel_rec=", "harm_vhrel_frac=", "harm_vhrel_on=")))

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 " + OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
