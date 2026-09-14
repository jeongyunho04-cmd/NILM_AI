# -*- coding: utf-8 -*-
"""`L_harm` 이 오븐↔포트를 **가를 수 있는가** — `--w-harm` 을 올리기 전에 재는 자 (14.84).

무엇을 가르려는가
-----------------
14.83 이 남긴 가설은 *"증거는 포트를 가리키는데 모델이 오븐을 고르니 `L_harm`(0.1) 이
`L_power`(1.0) 에 진다"* 였다. 이 한 문장에 **서로 다른 주장 셋**이 겹쳐 있다
([[decompose-before-attributing-a-percentage]]):

```
  ① 감도   L_harm 이 두 답을 **구별하기는 하나**?   (구별 못 하면 가중은 무의미)
  ② 부호   구별한다면 **어느 쪽**을 고르나?          (오븐을 고르면 올릴수록 나빠진다)
  ③ 순위   포트를 고르는데도 지면, **무엇에** 지나?
```

**전력 보존 맞바꿈**으로 재면 셋이 깨끗이 갈린다. 모델이 오븐에 준 통전 와트를
**그대로** 포트로 옮기면 Σ전력이 한 톨도 안 변하므로 `L_cons`·`L_over` 는 **정확히
0 만큼** 움직인다 (숫자로 확인한다). 남는 것은 `L_harm` 뿐이고, 그러면

```
    Δtotal = w_harm · ΔL_harm        Δ = L_harm(오븐답) − L_harm(포트답)
```
이라 **부호가 w_harm 과 무관**하다. Δ > 0 이면 고조파가 포트를 가리키고, Δ < 0 이면
`--w-harm` 을 올릴수록 축퇴가 **심해진다**. 학습 한 판(27분)을 굽기 전에 알 수 있다.

왜 h1 을 따로 보나
------------------
12.156 이 `L_harm` 판별의 **97.6%가 h1** 이라고 쟀다. 그런데 오븐(4.749 mA/W)과
포트(4.387 mA/W)는 h1 와트당 전류가 8%밖에 안 다르다 — **축퇴인 축이 바로 그 h1** 이다.
모양(h3·5·7)이 포트를 가리켜도 h1 이 그것을 덮으면 `L_harm` 전체는 무디다. 그래서
`전부` / `h1 뺀 것` 두 줄을 나란히 찍는다.

자리별로 가르는 이유
--------------------
학습 로그가 찍는 지문은 **녹화 자리의 전압**이다 (14.49 의 적합 전압: 오븐 210.4V,
포트 227.7V). 순저항은 `I_h/I_1 = V_h/V_1` 이라 h3/h1 이 포트 0.0326(자리 E) ·
오븐 0.0051(자리 D) 로 6.4배 갈리는데 **그것은 기기의 성질이 아니다.** 자리 D 에서
켜진 참 포트는 h3/h1 이 0.005 근처로 보이므로 `L_harm` 이 **오븐 쪽에 더 가깝다고
읽는다.** 그 가설이 맞으면 Δ 의 **부호가 자리마다 갈린다.**

    python -X utf8 src/run_diag_harmrank.py --ckpt results/cnn_v32h_off_s0.pt
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_train_seq import real_windows  # noqa: E402

KE, OV = "electiric_kettle", "oven"
S_KE, S_OV = 1, 2          # 통전 상태 — 학습 로그가 찍는 `s1` / `s2` 의 그 번호다
#: 관측 페이저의 차수는 h1..h15 (`inputs` 의 규약). 자리 i = 차수 i+1.
H1 = 0


def _site(stem):
    try:
        from src.preprocessing.file_registry import site_of
        return site_of(stem) or "?"
    except Exception:
        return "?"


def build_crit(apps, dev):
    """`run_train_cnn` 의 조립을 **그 순서 그대로** 옮긴다 (v32h BASE 의 깃발).

    ⚠ 따로 지은 물건을 재면 진짜 경로가 반쪽이어도 통과한다
      ([[the-gate-must-build-the-real-object]]). 그래서 아래 `_pin_assets` 가
      학습 **로그가 찍은 값**과 맞대 본다 — 안 맞으면 세우다 만 것이다.
    """
    from src.model.losses import NILMLoss, LossWeights, build_state_scales
    from src.model.net import (harmonic_scales, harmonic_signatures,
                               harmonic_signatures_by_state, noise_signature,
                               standby_signatures, harmonic_signature_vref,
                               harmonic_signature_vhrel)
    from src.model.companion import standby_operating_signatures
    from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
    from src.synthesis.sp_curves import background_signature
    from src.synthesis.segment_pool import SegmentPool
    from src.run_baseline import S_I
    from src.run_train_cnn import _vnorm_exp

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb_sig = standby_signatures(pool, apps)
    sb_op, _pw, sb_used = standby_operating_signatures(
        pool, apps, only=SESSION_PLUGGED_APPS)                 # --standby-operating session
    for x in sb_used:
        sb_sig[apps.index(x)] = sb_op[apps.index(x)]
    nz_sig = noise_signature(pool) + background_signature()    # 캐시가 배경을 넣었다
    h_scale = harmonic_scales(pool, apps)
    vref, vref_st = harmonic_signature_vref(pool, apps)        # --harm-vnorm-anchor
    sig_state = harmonic_signatures_by_state(pool, apps)[0]    # --state-signatures
    # 14.56 파형 앵커. **배선만 해 두고 `vhrel_frac` 을 실행 중에 0/1 로 돌린다** —
    # 자산을 두 번 짓지 않아야 두 판이 그 한 가지만 다르다.
    vhrel_rec = harmonic_signature_vhrel(pool, apps, source="conducting")
    vhrel_on = (np.asarray(_vnorm_exp(apps, "RESISTIVE"), np.float32) != 0
                ).astype(np.float32)
    del pool

    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        signatures_state=torch.from_numpy(sig_state),
        harm_even_magnitude=True,                              # --harm-even-magnitude
        harm_sig_vnorm=True,
        harm_vnorm_exp=_vnorm_exp(apps, "RESISTIVE"),          # --harm-vnorm-classes
        harm_vnorm_vref=torch.from_numpy(vref),
        harm_vnorm_vref_state=torch.from_numpy(vref_st),
        even_coherent=None,
        harm_vhrel_rec=torch.from_numpy(vhrel_rec),
        harm_vhrel_frac=1.0,
        harm_vhrel_on=torch.from_numpy(vhrel_on),
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0, z=0.3),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)
    return crit, vref


def _pin_assets(crit, apps, vref):
    """학습 로그가 찍은 값과 맞대어 **이 crit 이 그 crit 인지** 못 박는다."""
    want = {(KE, S_KE): (4.387, 0.0326, 227.7), (OV, S_OV): (4.749, 0.0051, 210.4)}
    print("  [0] 세운 물건 검산 — 학습 로그(cnn_v32h_off_s0.log)가 찍은 값과 맞댄다")
    ok = True
    for (app, s), (i1w, h3w, vw) in want.items():
        j = apps.index(app)
        c = crit.sig_state[j, s].cpu().numpy()
        z = c[:, 0] + 1j * c[:, 1]
        i1 = abs(z[0]) * 1e3
        h3 = abs(z[2]) / abs(z[0])
        v = float(vref[j])
        good = abs(i1 - i1w) < 5e-3 and abs(h3 - h3w) < 5e-4 and abs(v - vw) < 0.05
        ok = ok and good
        mark = "OK" if good else "** 다르다 **"
        print("      %-18s s%d  |I1| %.3f (로그 %.3f) · h3/h1 %.4f (로그 %.4f) · "
              "적합전압 %.1fV (로그 %.1fV)  %s"
              % (app, s, i1, i1w, h3, h3w, v, vw, mark))
    if not ok:
        raise SystemExit("지문이 학습 때와 다르다 — 이 자로 잰 값은 뜻이 없다")
    zk = crit.sig_state[apps.index(KE), S_KE].cpu().numpy()
    zo = crit.sig_state[apps.index(OV), S_OV].cpu().numpy()
    zk = zk[:, 0] + 1j * zk[:, 1]
    zo = zo[:, 0] + 1j * zo[:, 1]
    print("      => h1 와트당 전류 비 오븐/포트 = %.4f  (**여기가 축퇴 축이다**) · "
          "h3/h1 비 = %.2f배"
          % (abs(zo[0]) / abs(zk[0]),
             (abs(zo[2]) / abs(zo[0])) / (abs(zk[2]) / abs(zk[0]))))


def _mk_out(pw, plug, onl):
    """`(B,K,S)` 와트 표를 `_harm_pred_active` 가 받는 꼴로 만든다.

    `pw = (power/power_raw) · power_mix · power_states` 이므로 `power_raw = power`
    (게이트 1) 로 두고 `power_mix = pw/power`, `power_states = power` 로 놓으면
    곱이 정확히 `pw` 다. 실제 메서드를 **그대로** 부른다.
    """
    p = pw.sum(-1)                                                   # (B,K)
    mix = pw / p.clamp(min=1e-9)[..., None]
    return dict(power=p, power_raw=p.clamp(min=1e-9),
                power_mix=mix, power_states=p[..., None].expand_as(pw),
                plugged_logit=plug, on_logit=onl)


def harm_of(crit, pw, plug, onl, obs, vrel, vhrel, drop_h1):
    """그 답의 `L_harm` (창별). `drop_h1=True` 면 h1 자리의 가림을 0 으로 둔다."""
    crit._vrel = vrel
    crit._vhrel = vhrel if crit.use_vhrel else None
    out = _mk_out(pw, plug, onl)
    pred = crit._harm_pred_active(out, out["power"])
    idle = torch.sigmoid(plug) * (1.0 - torch.sigmoid(onl))
    pred = pred + torch.einsum("bk,khc->bhc", idle, crit.standby_sig)
    pred = pred + crit.noise_sig[None]
    err = crit._harm_err(pred, obs, crit._coherent_even_w(out["power"]))
    m = crit.harm_mask.clone()
    if drop_h1:
        m[H1] = 0.0
    return ((err * m[None, :, None]).mean(dim=(1, 2))
            / m.mean().clamp(min=1e-6))                               # (B,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--min-oven-w", type=float, default=100.0,
                    help="모델이 오븐에 이만큼 넘게 준 창만 (축퇴가 실제로 난 창)")
    ap.add_argument("--vhrel", type=float, nargs="+", default=[0.0], metavar="FRAC",
                    help="14.56 파형 앵커 `sig_h += sig_1·(rel_창 − rel_녹화)` 를 이 몫으로 "
                         "걸고 **같은 모델 출력에** 다시 잰다. 여럿 주면 나란히 찍는다. "
                         "0 이면 지금 본선과 같다 (항등).")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from src.run_gate_check import load_model
    from src.model.inputs import V_CENTER

    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    j_ke, j_ov = apps.index(KE), apps.index(OV)
    cache = real_windows(apps, a.grid_s, dev)
    crit, vref = build_crit(apps, dev)
    _pin_assets(crit, apps, vref)
    #: 와트당 h1 전류의 비 (오븐/포트). h1 보존 맞바꿈이 포트에 이만큼 더 얹는다.
    _z = crit.sig_state.cpu().numpy()
    r_h1 = float(np.hypot(_z[j_ov, S_OV, 0, 0], _z[j_ov, S_OV, 0, 1])
                 / np.hypot(_z[j_ke, S_KE, 0, 0], _z[j_ke, S_KE, 0, 1]))

    from src.run_train_cnn import _vhrel_from_fine
    from src.model.net import V_CH_FINE
    from src.model.inputs import V_SPAN

    for path in a.ckpt:
        m, apps_m = load_model(path, dev)[:2]
        assert list(apps_m) == apps
        m.eval()
        print("")
        print("=" * 96)
        print(path.split("/")[-1])
        print("=" * 96)

        # ── 몸통은 **한 번만** 돈다. 파형 앵커는 손실 쪽 손잡이라 같은 출력에 다시 건다 ──
        fw = {}
        for stem, d in sorted(cache.items()):
            PW, PL, GL, VR, VH = [], [], [], [], []
            with torch.no_grad():
                for i in range(0, len(d["t"]), 512):
                    fi = torch.from_numpy(d["fine"][i:i + 512]).to(dev)
                    o = m(fi, torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                    g = torch.sigmoid(o["on_logit"].float())
                    PW.append(g[..., None] * o["power_mix"].float()
                              * o["power_states"].float())
                    PL.append(o["plugged_logit"].float())
                    GL.append(o["on_logit"].float())
                    # `vrel` 과 `vhrel` 을 **같은 자리**에서 읽는다 (14.53) — 둘 다 창 평균.
                    _v = fi[:, V_CH_FINE].mean(-1).float() * V_SPAN + V_CENTER
                    VR.append(_v / V_CENTER)
                    VH.append(_vhrel_from_fine(fi, False).float())
            fw[stem] = (torch.cat(PW), torch.cat(PL), torch.cat(GL),
                        torch.cat(VR), torch.cat(VH))

        for _frac in a.vhrel:
            crit.vhrel_frac = float(_frac)
            crit.use_vhrel = bool(_frac != 0.0)
            print("")
            print("  --- 14.56 파형 앵커 %.2f할 %s ---"
                  % (_frac, "(항등 — 지금 본선)" if _frac == 0.0 else ""))

            rows, tot = [], {}
            for stem, d in sorted(cache.items()):
                y = d["y"].astype(bool)
                pw, plug, onl, vr_all, vh_all = fw[stem]

                # 겨냥 창 — **참 포트 ON · 참 오븐 OFF** 인데 모델이 오븐을 세운 창
                base = y[:, j_ke] & ~y[:, j_ov]
                sel = torch.from_numpy(base).to(dev) & (pw[:, j_ov, S_OV] > a.min_oven_w)
                n = int(sel.sum())
                n_base = int(base.sum())
                if n == 0:
                    rows.append((stem, _site(stem), n_base, 0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    continue

                si = sel.cpu().numpy()
                pw_a = pw[sel].clone()
                pw_b = pw_a.clone()
                moved = pw_a[:, j_ov, S_OV].clone()
                pw_b[:, j_ov, S_OV] = 0.0
                pw_b[:, j_ke, S_KE] = pw_b[:, j_ke, S_KE] + moved      # 전력 보존
                # ── 대조: **h1 보존** 맞바꿈 ───────────────────────────────────
                # 오븐은 와트당 h1 전류가 포트의 1.0824배다. 전력을 보존하면 예측 h1 이
                # 8% **줄어드는데**, 모델이 이미 전류를 모자라게 내고 있으면 그것만으로
                # 손실이 는다 — 그것은 **신원이 아니라 수준**이다. 포트 와트를 1.0824배로
                # 얹어 h1 을 맞추면 남는 것은 **모양**뿐이다 (대신 Σ전력이 8% 는다).
                pw_c = pw_a.clone()
                pw_c[:, j_ov, S_OV] = 0.0
                pw_c[:, j_ke, S_KE] = pw_c[:, j_ke, S_KE] + moved * r_h1

                obs = torch.from_numpy(d["obs_harm"][si]).to(dev)
                vrel, vhr = vr_all[sel], vh_all[sel]
                plug_s, onl_s = plug[sel], onl[sel]

                # 합이 안 변하는지 **숫자로** 확인한다 (그래야 Δ 가 L_harm 만의 것이다)
                dsum = float((pw_b.sum((1, 2)) - pw_a.sum((1, 2))).abs().max())

                ha = harm_of(crit, pw_a, plug_s, onl_s, obs, vrel, vhr, False)
                hb = harm_of(crit, pw_b, plug_s, onl_s, obs, vrel, vhr, False)
                hc = harm_of(crit, pw_c, plug_s, onl_s, obs, vrel, vhr, False)
                ha3 = harm_of(crit, pw_a, plug_s, onl_s, obs, vrel, vhr, True)
                hb3 = harm_of(crit, pw_b, plug_s, onl_s, obs, vrel, vhr, True)
                dd = (ha - hb).cpu().numpy()
                dd3 = (ha3 - hb3).cpu().numpy()
                ddc = (ha - hc).cpu().numpy()
                rows.append((stem, _site(stem), n_base, n, float(ha.mean()),
                             float(hb.mean()), float(dd.mean()),
                             float((dd > 0).mean()), float(dd3.mean()),
                             float(ddc.mean()), float((ddc > 0).mean())))
                t = tot.setdefault(_site(stem), [0, [], [], 0.0, [], []])
                t[0] += n
                t[1].append(dd)
                t[2].append(dd3)
                t[3] = max(t[3], dsum)
                t[4].append(ddc)

                # ── 양성 대조 (거꾸로) ────────────────────────────────────────
                # **모델이 맞힌** 창 — 참 포트 ON 이고 오븐을 안 세운 창 — 에서 반대로
                # 포트 와트를 오븐으로 옮긴다. `L_harm` 이 신원을 담고 있으면 여기서는
                # **포트 답이 더 좋아야** 한다 (Δrev < 0). 여기서도 오븐이 이기면
                # `L_harm` 은 참값과 **무관하게** 오븐을 고르는 것이다 — 약한 게 아니라
                # 그 쌍에 대해 **틀린** 항이다.
                ok = torch.from_numpy(base).to(dev) \
                    & (pw[:, j_ov, S_OV] <= a.min_oven_w) \
                    & (pw[:, j_ke, S_KE] > a.min_oven_w)
                if int(ok.sum()) == 0:
                    continue
                oi = ok.cpu().numpy()
                pw_p = pw[ok].clone()
                pw_q = pw_p.clone()
                mv = pw_p[:, j_ke, S_KE].clone()
                pw_q[:, j_ke, S_KE] = 0.0
                pw_q[:, j_ov, S_OV] = pw_q[:, j_ov, S_OV] + mv          # 전력 보존
                obs2 = torch.from_numpy(d["obs_harm"][oi]).to(dev)
                vr2, vh2 = vr_all[ok], vh_all[ok]
                hp = harm_of(crit, pw_p, plug[ok], onl[ok], obs2, vr2, vh2, False)
                hq = harm_of(crit, pw_q, plug[ok], onl[ok], obs2, vr2, vh2, False)
                t[5].append((hp - hq).cpu().numpy())

            print("  [1] 전력 보존 맞바꿈 — 오븐 통전 와트를 **그대로** 포트로 옮긴다")
            print("      Δ = L_harm(오븐답) − L_harm(포트답).  "
                  "**Δ > 0 이면 고조파가 포트를 가리킨다**")
            print("      %-9s%4s%9s%7s%12s%12s%11s%9s%11s%12s%9s"
                  % ("파일", "자리", "참포트ON", "축퇴창", "L_harm오븐",
                     "L_harm포트", "Δ 전부", "포트승률", "Δ h1뺀것",
                     "Δ h1맞춤", "포트승률"))
            for r in rows:
                if r[3] == 0:
                    print("      %-9s%4s%9d%7d        (축퇴 창 없음)" % (r[0], r[1], r[2], 0))
                    continue
                print("      %-9s%4s%9d%7d%12.5f%12.5f%+11.5f%8.0f%%%+11.5f%+12.5f%8.0f%%"
                      % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], 100 * r[7], r[8],
                         r[9], 100 * r[10]))

            print("")
            print("  [2] 자리별 — 지문이 **녹화 자리의 전압**이라는 가설의 예측: 부호가 갈린다")
            for k in ("D", "E"):
                if k not in tot or tot[k][0] == 0:
                    continue
                dd = np.concatenate(tot[k][1])
                d3 = np.concatenate(tot[k][2])
                dc = np.concatenate(tot[k][4])
                print("      자리 %s  축퇴창 %4d   Δ 전부 %+.5f ± %.5f (포트승률 %3.0f%%)   "
                      "Δ h1뺀것 %+.5f (포트승률 %3.0f%%)   Δ h1맞춤 %+.5f (포트승률 %3.0f%%)"
                      % (k, tot[k][0], dd.mean(), dd.std(), 100 * (dd > 0).mean(),
                         d3.mean(), 100 * (d3 > 0).mean(),
                         dc.mean(), 100 * (dc > 0).mean()))
                print("              Σ전력 차 최대 %.3e W -> L_cons·L_over 의 차는 "
                      "**정확히 0**, Δtotal = w_harm·Δ" % tot[k][3])
                if tot[k][5]:
                    rv = np.concatenate(tot[k][5])
                    print("              [양성 대조] 모델이 **맞힌** 포트 창 %d개에서 거꾸로 "
                          "(포트->오븐): Δrev %+.5f · 포트승률 %3.0f%%   %s"
                          % (rv.size, rv.mean(), 100 * (rv < 0).mean(),
                             "OK (여기선 포트를 고른다)" if rv.mean() < 0
                             else "** 참값과 무관하게 오븐을 고른다 **"))

            print("")
            print("  [3] 판정")
            pool_d = [x for v in tot.values() if v[0] for x in v[1]]
            pool_c = [x for v in tot.values() if v[0] for x in v[4]]
            if pool_d:
                alld = np.concatenate(pool_d)
                allc = np.concatenate(pool_c)
                s = float(alld.mean())
                print("      전체 Δ %+.5f · 포트승률 %.0f%% (%d창)"
                      % (s, 100 * (alld > 0).mean(), alld.size))
                print("      h1 을 맞추면 (수준을 빼고 **모양만**) Δ %+.5f · 포트승률 %.0f%%  "
                      "[포트 와트 x%.4f]"
                      % (float(allc.mean()), 100 * (allc > 0).mean(), r_h1))
                if s > 0:
                    print("      => 고조파는 포트를 가리킨다. `L_cons`·`L_over` 가 **무차별**이므로")
                    print("         총손실에서 이 맞바꿈을 막는 항이 **하나도 없다.** 그런데도 모델이")
                    print("         오븐을 고른다면 진 상대는 다른 손실 항이 아니라 **모델이 배운 것**이다.")
                else:
                    print("      => ★ 고조파가 **오븐**을 가리킨다. `--w-harm` 을 올리면 축퇴가 "
                          "**심해진다.**")
                    print("         가설 반증. 한 판(27분)을 안 굽는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
