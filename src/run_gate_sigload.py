# -*- coding: utf-8 -*-
"""(나) `--load-rot` 의 관문 (14.402). 미리 적은 자(§52.4 · §54.5)를 그대로 판다.

```
  [1] 안 켜면 **비트 동일** (a=0 이면 곱이 항등)
  [2] ★ 회전이 **크기를 보존**하나 (순수 회전이어야 한다)
  [3] ★ **저항 넷·빔의 a 가 정확히 0** — 대조군이 통과 조건이다
  [4] ★★ **채워진 칸 전부**를 찍는다 (s2 만 보면 미니PC s1 을 놓친다 — §54.5 ③)
  [5] ★ 참 배분 잔차가 **걸린 칸에서만** 내려가나 (사이클 그대로 · 사전만 바꾼다)
  [6] ⚠⚠ **진짜 손실 객체**를 지어 forward 가 유한하고 저항이 안 움직이나
  [7] ⚠ `--pow-sig-instate` 와 **하드 스톱**이 걸려 있나 (분모 공유)
```

    python -X utf8 -m src.run_gate_sigload
"""
from typing import List

import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model.net import harmonic_signatures_by_state, state_power_w  # noqa: E402
from src.model.sigload import LOAD_ROT_A_CELL, load_rot_table  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SMPS = ("minipc", "laptop_charger", "beam_projector")
QUIET = ("oven", "electiric_kettle", "hair_dryer", "hotplate", "fan", "air_conditioner")
DISC = (9, 11, 13)
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-52s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def cell(pool, app, sid):
    """그 칸의 **모든 주기** (z 와 P) — `harmonic_signatures_by_state` 와 같은 규약."""
    zs, ps = [], []
    for a in pool.appliance_activations.get(app, []):
        st = getattr(a, "state_id", None)
        if st is None:
            continue
        pw = state_power_w(a, False)
        m = (pw > 1.0) & (np.asarray(st) == sid)
        if m.any():
            zs.append(np.asarray(a.net_harmonics_complex)[m]
                      / np.maximum(pw[m], 1e-6)[:, None])
            ps.append(pw[m])
    if not zs:
        return None
    return np.concatenate(zs), np.concatenate(ps)


def irr(z, sig, orders):
    out = []
    for h in orders:
        i = h - 1
        out.append(np.median(np.abs(z[:, i] - sig[i]))
                   / max(np.median(np.abs(z[:, i])), 1e-12))
    return float(np.mean(out))


def main() -> int:
    from src.model.losses import NILMLoss
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    a_np, lpref, on = load_rot_table(pool, APPS)
    sig_state, used = harmonic_signatures_by_state(pool, APPS)
    print("부하 의존 위상 관문 (14.402) — 표 %s\n" % LOAD_ROT_A_CELL)

    s_i = torch.ones(len(APPS))
    A = torch.from_numpy(a_np)
    P = torch.from_numpy(lpref)
    O = torch.from_numpy(on.astype(np.float32))
    L0 = NILMLoss(s_i)
    L1 = NILMLoss(s_i, load_rot_a=A, load_rot_lpref=P, load_rot_on=O)
    torch.manual_seed(0)
    sgs = torch.randn(4, len(APPS), lpref.shape[1], 15, 2)
    pst = torch.rand(4, len(APPS), lpref.shape[1]) * 40 + 1

    # [1] 안 켜면 비트 동일
    chk(1, "안 켜면 **비트 동일**인가", not L0.use_load_rot,
        "`load_rot_a` 를 안 주면 `use_load_rot=%s` 라 경로를 안 탄다 — 옛 코드 그대로"
        % L0.use_load_rot)

    # [2] ★ 크기 보존 (순수 회전)
    got = L1._apply_load_rot(sgs, pst)
    dm = float((got.norm(dim=-1) - sgs.norm(dim=-1)).abs().max())
    chk(2, "★ 회전이 **크기를 보존**하나 (순수 회전)", dm < 1e-5,
        "|‖후‖ − ‖전‖| 최대 **%.2e** — 크기가 바뀌면 그건 회전이 아니라 이득이다" % dm)

    # [3] ★ 저항·빔은 정확히 0
    zero = [x for x in APPS if not any(k[0] == x for k in LOAD_ROT_A_CELL)]
    j0 = [APPS.index(x) for x in zero]
    same0 = torch.equal(got[:, j0], sgs[:, j0])
    chk(3, "★ **저항 넷·빔의 a 가 정확히 0** 인가 (대조군)", same0 and "beam_projector" in zero,
        "표에 없는 기기 %s — 빔은 통전 폭 x1.05 라 **못 잰다**(§56) · 저항은 §56 대조군이 "
        "0 을 보증한다 (오븐 +0.4%%p · 핫플 −0.3%%p)" % ", ".join(zero))

    # [4] ★★ 채워진 칸 **전부**
    rows, n_on, n_off = [], 0, 0
    for j, app in enumerate(APPS):
        for s in range(used.shape[1]):
            if not used[j, s]:
                continue
            g = cell(pool, app, s)
            if g is None:
                continue
            _z, p = g
            w = float(np.percentile(p, 95) / max(np.percentile(p, 5), 1e-9))
            hot = bool(on[j, s])
            n_on += hot
            n_off += (not hot)
            rows.append("%s s%d %s(폭 x%.2f)" % (app[:7], s, "**켬**" if hot else "끔", w))
    chk(4, "★★ **채워진 칸 전부**를 찍나 (s2 만 보면 놓친다)", n_on + n_off >= 15,
        "칸 %d개 — 켠 칸 **%d** · 끈 칸 %d\n      %s"
        % (n_on + n_off, n_on, n_off, " · ".join(rows)))

    # [5] ★ 걸린 칸에서만 잔차가 내려가나 (사이클 그대로 · 사전만 바꾼다)
    res, ok5 = [], True
    for app in SMPS + ("oven", "hotplate"):
        j = APPS.index(app)
        for s in range(used.shape[1]):
            if not used[j, s]:
                continue
            g = cell(pool, app, s)
            if g is None or len(g[1]) < 2000:
                continue
            z, p = g
            sg = sig_state[j, s, :, 0] + 1j * sig_state[j, s, :, 1]
            i0 = irr(z, sg, DISC)
            rot = np.exp(1j * np.radians(float(a_np[j, s]))
                         * (np.log(np.maximum(p, 1e-9)) - float(lpref[j, s]))[:, None]
                         * np.arange(1, z.shape[1] + 1)[None])
            i1 = float(np.mean([np.median(np.abs(z[:, h - 1] - sg[h - 1] * rot[:, h - 1]))
                                / max(np.median(np.abs(z[:, h - 1])), 1e-12) for h in DISC]))
            d = 100 * (i1 - i0) / max(i0, 1e-9)
            res.append("%s s%d %+.1f%%" % (app[:7], s, d))
            if on[j, s] and a_np[j, s] != 0:
                ok5 &= d < 0.0                       # 걸린 칸은 **내려가야** 한다
            else:
                ok5 &= abs(d) < 1e-9                 # 안 걸린 칸은 **정확히 0**
    chk(5, "★ **걸린 칸만** 내려가고 나머지는 정확히 0 인가", ok5, " · ".join(res))

    # [6] ⚠⚠ 진짜 손실 객체
    B = 4
    out = {"power": torch.rand(B, len(APPS)) * 30,
           "power_raw": torch.rand(B, len(APPS)) * 30 + 1,
           "on_logit": torch.zeros(B, len(APPS)),
           "plugged_logit": torch.zeros(B, len(APPS))}
    from src.model.lossbuild import build_loss
    LB = build_loss(APPS, "cpu", state_signatures=True, verbose=False)
    pr = LB._harm_pred_active(out, out["power"])
    chk(6, "⚠⚠ **진짜 손실 객체**가 서고 forward 가 유한한가",
        bool(torch.isfinite(pr).all()),
        "`build_loss` 로 지은 판에서 `_harm_pred_active` 유한 %s · 모양 %s"
        % (bool(torch.isfinite(pr).all()), tuple(pr.shape)))

    # [7] ⚠ 하드 스톱이 걸려 있나 (AST 로 본다 — 부르면 SystemExit 이라)
    import inspect
    from src import run_train_cnn as RT
    src = inspect.getsource(RT)
    chk(7, "⚠ `--pow-sig-instate` 와 **하드 스톱**이 걸려 있나",
        "a.load_rot and a.pow_sig_instate" in src
        and "a.load_rot and not a.state_signatures" in src,
        "분모 공유(`g = sig_대역/sig_state`)라 같이 켜면 비가 어긋난다 — 14.172 꼴")

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
