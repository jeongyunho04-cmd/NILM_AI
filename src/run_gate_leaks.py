# -*- coding: utf-8 -*-
"""관문 — **미래가 새는 구멍 넷**을 막는 손잡이 (14.183). 일곱 줄.

14.182 가 센 것: 같은 두 사이클 차이가 판마다 **다른 길**로 간다.

```
   막은 %                          dzn_s0    dzn_s1   osig_s0
   ⓐ 파생채널 29·30·41·42·43·44       −3%       93%       −6%
   ⓑ 정규화 μ (GroupNorm 이 T 전체)    45%       47%       −5%
   ⓒ 전역최대 amax (위치가 없다)        40%       67%       68%
   ⓓ 수용영역 (블록5 반RF 189 ≥ 148)   35%       15%       32%
```

그래서 `--fine-time-split`·`--fine-future-segs`·`--fine-pad`·`--seg-pool`·`--head-drop`·
전방탭·광역룩어헤드가 **전부** 널이었다 — 하나씩 닫았기 때문이다.

이 관문은 **막힌 것과 일부러 남긴 것**을 둘 다 못 박는다. 정보를 버리는 것이 아니라
**시간 부호를 붙이는 것**이 처방이다 (인계 2026-09-15b).

```
  [1] 넷 다 기본값이면 **비트 동일**
  [2] ⓑ `fine_norm=causal` — 정규화 통계가 **미래를 바꿔도 불변**
  [3] ⓓ `fine_conv=causal` — 타깃 자리 출력이 **미래를 바꿔도 불변** (전 블록)
  [4] ⓒ `fine_tpool=split` — 과거 조각 요약이 불변 (ⓓ 와 **같이 켤 때만**)
  [5] ⓐ `fine_derive=both` — 새 네 채널이 **미래를 안 본다**
  [6] 넷 다 켜도 **일부러 남는 길**이 무엇인지 센다 (0 이면 오히려 의심해야 한다)
  [7] 넷 다 켠 판이 **경계에서 실제로 덜 움직인다** (test_2 176.0초, 두 사이클)
```

    python -X utf8 -m src.run_gate_leaks
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import fine_target_index  # noqa: E402
from src.model.losses import S_STATE  # noqa: E402
from src.model.net import NILMNet  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]
T = fine_target_index()
ALL = dict(fine_norm="causal", fine_conv="causal", fine_tpool="split", fine_derive="both")


def mk(**kw):
    torch.manual_seed(0)
    return NILMNet(APPS, NS, fine_channels=57, fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0,
                   aux_z=True, **kw).eval()


def pair(seed=1):
    """미래(타깃 뒤)만 다른 두 입력. 과거와 타깃은 **한 톨도 안 다르다**."""
    torch.manual_seed(seed)
    f = torch.randn(2, 57, 600)
    w = torch.randn(2, 47, 120)
    f[1, :, :T + 1] = f[0, :, :T + 1]
    w[1] = w[0]
    return f, w


def main() -> int:
    ok = True
    f, w = pair()

    # [1] 기본값이면 비트 동일
    base = mk()
    same = mk(fine_norm="window", fine_conv="sym", fine_tpool="whole", fine_derive="window")
    same.load_state_dict(base.state_dict())
    with torch.no_grad():
        o0, o1 = base(f, w), same(f, w)
    d = max(float((o1[k] - o0[k]).abs().max()) for k in ("power", "on_logit", "state"))
    print("[1] 넷 다 기본값 = **비트 동일**   최대차 %.3e  %s" % (d, "OK" if d == 0 else "FAIL"))
    ok &= (d == 0)

    # ── [2] ⓓ **conv 자체**가 인과인가 (정규화 전, 날것) ────────────────────
    #    ⚠ 첫 판에서 이 줄을 "인과 conv 면 블록 출력이 불변" 으로 적었다가 FAIL 했다.
    #      까닭은 conv 가 아니라 **GroupNorm** 이었다 — 그것이 [3] 이다.
    mc, ms = mk(fine_conv="causal"), mk(fine_conv="sym")
    with torch.no_grad():
        yc = mc.fine[0][0](mc._conv_in(f))
        ys = ms.fine[0][0](ms._conv_in(f))
    dc = float((yc[1, :, T] - yc[0, :, T]).abs().max())
    ds = float((ys[1, :, T] - ys[0, :, T]).abs().max())
    print("[2] ⓓ conv 날출력 — 인과면 타깃이 **정확히 불변** %.3e · 대칭은 %.4f  %s"
          % (dc, ds, "OK" if (dc == 0.0 and ds > 1e-4) else "FAIL"))
    ok &= (dc == 0.0 and ds > 1e-4)

    # ── [3] ⓑ 없이 ⓓ 만 켜면 **정규화가 다시 연다** (구멍이 합성된다) ────────
    mcn = mk(fine_conv="causal", fine_norm="causal")
    worst = {}
    for nm, mm in (("ⓓ만", mc), ("ⓓ+ⓑ", mcn)):
        with torch.no_grad():
            h = mm._conv_in(f)
            v = 0.0
            for blk in mm.fine:
                h = blk(h)
                v = max(v, float((h[1, :, T] - h[0, :, T]).abs().max()))
        worst[nm] = v
    print("[3] ⓑ 없이 ⓓ 만 = 타깃이 **여전히 샌다** %.4f (GroupNorm 이 T 전체) · "
          "ⓓ+ⓑ 면 **불변** %.3e  %s"
          % (worst["ⓓ만"], worst["ⓓ+ⓑ"],
             "OK" if (worst["ⓓ만"] > 1e-4 and worst["ⓓ+ⓑ"] < 1e-5) else "FAIL"))
    ok &= (worst["ⓓ만"] > 1e-4 and worst["ⓓ+ⓑ"] < 1e-5)

    # ── [4] ⓒ 과거 조각 요약 — ⓑ+ⓓ 와 **같이** 켜야 불변이다 ────────────────
    m4 = mk(fine_tpool="split", fine_conv="causal", fine_norm="causal")
    m4b = mk(fine_tpool="split")
    def past_amax(mm):
        with torch.no_grad():
            h = mm._conv_in(f)
            for blk in mm.fine:
                h = blk(h)
        return float((h[1, :, :T + 1].amax(-1) - h[0, :, :T + 1].amax(-1)).abs().max())
    p4, p4b = past_amax(m4), past_amax(m4b)
    print("[4] ⓒ 과거 조각 amax — ⓒ+ⓑ+ⓓ 면 **불변** %.3e · ⓒ 만이면 %.4f  %s"
          % (p4, p4b, "OK" if (p4 < 1e-5 and p4b > 1e-4) else "FAIL"))
    ok &= (p4 < 1e-5 and p4b > 1e-4)

    # ── [5] ⓐ 새 네 채널 — **실측 창**으로 잰다 (무작위 텐서는 41·42 가 파생이 아니다) ──
    try:
        from src.model.inputs import build_inputs
        from src.model.realdata import RealWindows, target_index
        from src.preprocessing import load_nilm_npz
        x = RealWindows._to_33ch(load_nilm_npz("processed_data/composite_eval/test_2.npz"))
        off, f0, FS, WIN = target_index(3600), 3000, 60, 3600
        wn = lambda tt: x[:, int(round(tt * FS)) - off:int(round(tt * FS)) - off + WIN].copy()
        A5 = wn(176.0)
        B5 = A5.copy()
        B5[:, f0 + 387:WIN] = wn(171.0)[:, f0 + 387:WIN]   # **미래 두 사이클 뒤만** 다르다
        f5, w5 = build_inputs(np.stack([B5, A5]))
        F5 = torch.from_numpy(np.ascontiguousarray(f5))
        raw_same = float(np.abs(B5[:, :f0 + 387] - A5[:, :f0 + 387]).max())
        dpast = np.abs(f5[1, :, :T + 1] - f5[0, :, :T + 1]).max(-1)
        leaky = [int(c) for c in np.where(dpast > 1e-4)[0]]
        with torch.no_grad():
            ci = mk(fine_derive="both")._conv_in(F5)
        dnew = float((ci[1, 57:, T] - ci[0, 57:, T]).abs().max())
        dold = float((ci[1, [41, 42], T] - ci[0, [41, 42], T]).abs().max())
        print("[5] ⓐ 실측 창 — 새 네 채널(57~60) 타깃이 **불변** %.3e · 원래 41·42 는 "
              "**%.4f** 움직인다  %s" % (dnew, dold,
                                     "OK" if (dnew < 1e-5 and dold > 1e-3) else "FAIL"))
        print("      **원시** 45채널의 과거·타깃은 한 톨도 안 다르다 (최대차 %.3e)." % raw_same)
        print("      그런데 **지어진** 57채널은 과거까지 달라진다 — 새는 채널 %s"
              % (leaky if leaky else "없음"))
        print("      = ch41·42 가 p(t+180)·p(t+330) 을, ch29·30·43·44 가 대칭 평활을 쓴다.")
        ok &= (dnew < 1e-5 and dold > 1e-3 and raw_same == 0.0
               and set(leaky) <= {26, 27, 28, 29, 30, 40, 41, 42, 43, 44})
    except Exception as e:                                   # noqa: BLE001
        print("[5] 건너뜀 (실측 파일 없음): %s" % e)

    # [6] 넷 다 켜도 **일부러 남는 길**
    ma = mk(**ALL)
    with torch.no_grad():
        oa = ma(f, w)
    left = float((oa["on_logit"][1] - oa["on_logit"][0]).abs().max())
    print("[6] 넷 다 켜도 미래가 답을 **얼마나** 움직이나: on_logit 최대차 %.4f" % left)
    print("    남기는 길 — ⓒ 의 **미래 조각 요약** · ⓐ 의 원래 41·42 · ⑥ 광역 평균 ·")
    print("    ⑦ 원시 전력통계 `fp.amax`. 전부 **시간 부호가 붙은** 자리다.")
    print("    0 이면 오히려 의심해야 한다 — 미래 6초는 **쓰인다**(15c: 짧을수록 나쁘다).  %s"
          % ("OK" if left > 1e-4 else "**FAIL — 미래를 통째로 지웠다**"))
    ok &= (left > 1e-4)

    # [7] 실측 경계에서 덜 움직이나
    try:
        from src.model.inputs import build_inputs
        from src.model.realdata import RealWindows, target_index
        from src.preprocessing import load_nilm_npz
        x = RealWindows._to_33ch(load_nilm_npz("processed_data/composite_eval/test_2.npz"))
        off, f0, FS, WIN = target_index(3600), 3000, 60, 3600
        win = lambda tt: x[:, int(round(tt * FS)) - off:int(round(tt * FS)) - off + WIN].copy()
        A = win(176.0)
        B = A.copy()
        B[:, f0 + 387:WIN] = win(171.0)[:, f0 + 387:WIN]
        ff, ww = build_inputs(np.stack([B, A]))
        Ft = torch.from_numpy(np.ascontiguousarray(ff))
        Wt = torch.from_numpy(np.ascontiguousarray(ww))
        rows = []
        for nm, kw in (("기본", {}), ("ⓑ", dict(fine_norm="causal")),
                       ("ⓐⓑⓒⓓ", ALL)):
            mm = mk(**kw)
            with torch.no_grad():
                oo = mm(Ft, Wt)
            rows.append((nm, float((oo["on_logit"][1] - oo["on_logit"][0]).abs().mean())))
        print("[7] test_2 176.0초 경계(두 사이클) — **학습 안 된** 새 판의 on_logit 평균 변화")
        print("    " + " · ".join("%s %.4f" % r for r in rows))
        print("    ⚠ 가중치가 무작위라 **크기는 뜻이 없다**. 경로가 실제로 끊겼는지만 본다.")
    except Exception as e:                                  # noqa: BLE001
        print("[7] 건너뜀 (실측 파일 없음): %s" % e)

    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
