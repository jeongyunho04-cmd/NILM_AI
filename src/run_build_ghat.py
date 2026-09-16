# -*- coding: utf-8 -*-
"""캐시에 **`g_hat.npy`** 를 얹는다 — 창마다 타깃 순간의 `Ĝ_sum` (mS) (14.346).

조합 머리(`--comb-tau`, 14.347)가 쓸 물건이다. `Ĝ` 는 **입력만의 함수**라 학습 전에
한 번 풀어 두면 되고, **캐시를 다시 굽지 않아도 된다** — 필요한 것이 둘 다 이미 있다:
```
  `obs_harm` (N,15,2)   타깃 순간의 `harmonics_ri` = I_h Re/Im 15차수
  `fine[:, 45:61, t]`   같은 순간의 V_h Re/Im **홀수 8차수** (되돌릴 수 있다)
```
`run_gate_gbudget_cache` 가 이 되짚기를 8/8 로 확인했다 (float16 이어도 |잔차| 0.191 mS).

⚠ **타깃 순간만** 푼다. 창 최대는 안 만든다 — 조합 머리는 **전력**에 붙고 타깃 순간의
  참 전력은 듀티 OFF 면 0 이 맞다. 창 최대가 필요한 것은 **게이트 프라이어** 쪽이고
  (14.335: 순시면 오븐 라벨 ON 의 50.5%와 싸운다) 그건 다른 판이다.

⚠ `fine` 을 통째로 읽으면 300,000창이 **22GB** 다. 창 축으로 토막내 읽는다.

    python -X utf8 -m src.run_build_ghat --cache cache/train60_v49
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from src import env_guard  # noqa: F401

from src.model import gbudget as GB  # noqa: E402
from src.model import inputs as _I  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def raw_from_target(fin, obs):
    """(B,61) 타깃 프레임 + (B,15,2) -> 분해기가 먹는 (B, 49, 1). `run_gate_gbudget_cache` 와 같다."""
    n = len(fin)
    nv = len(_I.VOLT_ORDERS)
    raw = np.zeros((n, 33 + 2 * nv, 1), np.float64)
    raw[:, 0:15, 0] = obs[:, :, 0]
    raw[:, 15:30, 0] = obs[:, :, 1]
    raw[:, 30, 0] = np.sinh(fin[:, 23]) * 100.0
    raw[:, 32, 0] = fin[:, 25] * _I.V_SPAN + _I.V_CENTER
    for s, h in enumerate(_I.VOLT_ORDERS):
        a = fin[:, _I.FINE_VOLT0 + s]
        b = fin[:, _I.FINE_VOLT0 + nv + s]
        if h == 1:
            vr, vi = a * _I.V_SPAN + _I.V_CENTER, b * _I.V_SPAN
        else:
            vr = np.sinh(a) / _I.VOLT_HARM_SCALE
            vi = np.sinh(b) / _I.VOLT_HARM_SCALE
        raw[:, 33 + s, 0] = vr
        raw[:, 33 + nv + s, 0] = vi
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--verify", type=int, default=2000, help="쓴 뒤 다시 풀어 대조할 창 수")
    #: ⚠ 14.346 — **FCM 기준전압 표본이 Ĝ 를 움직인다.** 관문 [9] 가 잡았다:
    #  표본 500/2000/5000 창일 때 전체 기준과의 차 최대 **1.524 / 0.275 / 0.079 mS**.
    #  범인은 |V₃| 다 (5.13 -> 6.34V 로 수렴). 포트 식별 여유가 0.635 mS 니
    #  2,000창 판은 **여유의 43%%** 를 흔든 것이다. 2만으로 키우고 meta 에 적는다.
    ap.add_argument("--vref-windows", type=int, default=20000)
    a = ap.parse_args()
    c = a.cache.rstrip("/")
    f = np.load("%s/fine.npy" % c, mmap_mode="r")
    obs_all = np.load("%s/obs_harm.npy" % c, mmap_mode="r")
    n = len(f)
    two_d = f.ndim == 2                      # 타깃 프레임만 잘라 둔 캐시
    t = _I.fine_target_index()
    if f.shape[1] != _I.FINE_CHANNELS:
        sys.exit("캐시 세밀이 %d채널이다 — 코드는 %d (배치가 다르다)"
                 % (f.shape[1], _I.FINE_CHANNELS))
    print("[ghat] %s · 창 %d · 세밀 %s · 차수 %s · 타깃 프레임 %d%s"
          % (c, n, tuple(f.shape[1:]), tuple(_I.VOLT_ORDERS), t,
             " (이미 잘려 있다)" if two_d else ""))

    #: 배선은 **녹화(캐시) 하나당 한 번** 짓는다 — FCM 표가 기준 전압을 쓴다
    k0 = min(int(a.vref_windows), n)
    head = np.asarray(f[:k0] if two_d else f[:k0, :, t], np.float64)
    nv = len(_I.VOLT_ORDERS)
    v15 = np.zeros(15, complex)
    r0 = raw_from_target(head, np.asarray(obs_all[:len(head)], np.float64))
    for s, h in enumerate(_I.VOLT_ORDERS):
        v15[h - 1] = (np.median(r0[:, 33 + s, 0]) + 1j * np.median(r0[:, 33 + nv + s, 0]))
    print("       기준 전압 (%d창) |V₁| %.2fV · |V₃| %.3fV" % (k0, abs(v15[0]), abs(v15[2])))
    bud = GB.Budget(APPS, v15, volt_re0=33, volt_orders=_I.VOLT_ORDERS)

    g = np.empty(n, np.float32)
    t0 = time.time()
    for i in range(0, n, a.chunk):
        j = min(i + a.chunk, n)
        fin = np.asarray(f[i:j] if two_d else f[i:j, :, t], np.float64)
        raw = raw_from_target(fin, np.asarray(obs_all[i:j], np.float64))
        g[i:j] = bud.g_sum(raw)[:, 0]
        if (i // a.chunk) % 10 == 0:
            print("       %6d/%d · %.0fs" % (j, n, time.time() - t0), flush=True)
    out = "%s/g_hat.npy" % c
    np.save(out, g)
    print("[ghat] 저장 %s · %.0fs · Ĝ 중앙 %.2f mS · p1 %.2f · p99 %.2f"
          % (out, time.time() - t0, np.median(g), *np.percentile(g, [1, 99])))

    # ── 자기 검증 — **다시 풀어** 비트 동일인가 ────────────────────────────
    k = min(a.verify, n)
    idx = np.random.default_rng(0).choice(n, k, replace=False)
    idx.sort()
    fin = np.asarray(f[idx] if two_d else f[idx, :, t], np.float64)
    again = bud.g_sum(raw_from_target(fin, np.asarray(obs_all[idx], np.float64)))[:, 0]
    d = float(np.abs(again - g[idx]).max())
    print("[ghat] 검증 %d창 — 다시 푼 값과 최대차 **%.3e mS** %s"
          % (k, d, "✅" if d < 1e-4 else "❌ **다르다**"))
    #: meta 에 적어 둔다 — 없으면 학습이 옛 `g_hat` 을 조용히 먹는다
    mp = "%s/meta.json" % c
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        #: ★ `v_ref` 를 **반드시 적는다.** 안 적으면 다시 풀 때 표본이 달라져
        #  Ĝ 가 최대 0.275 mS 어긋나고, 그건 포트 식별 여유의 43% 다.
        m["g_hat"] = {"volt_orders": [int(h) for h in _I.VOLT_ORDERS],
                      "target_frame": int(t), "n": int(n),
                      "vref_windows": int(k0),
                      "v_ref": [[float(z.real), float(z.imag)] for z in v15],
                      "median_ms": float(np.median(g))}
        json.dump(m, open(mp, "w", encoding="utf-8"), ensure_ascii=False)
        print("[ghat] meta 에 `g_hat` 규약을 적었다")
    return 0 if d < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
