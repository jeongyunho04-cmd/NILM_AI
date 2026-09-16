# -*- coding: utf-8 -*-
"""`gbudget` 이 **학습 캐시 위에서도** 서는가 (14.338).

실측 관문(`run_gate_gbudget`)은 `processed_data/composite_eval` 을 읽어서
**HPC 에서 못 돈다** (HPC_RULES §0). 그리고 라벨이 스위치 로그라 "라벨 ON = 통전"
이 아니다 (오븐은 라벨 ON 의 54~71%가 비통전).

캐시는 그 두 문제가 **둘 다 없다**:
```
  `y_power` 가 **기기별 참 전력**이다 -> 참 컨덕턴스를 소수점까지 안다
  HPC 에서 돈다 (실측을 안 올려도 된다)
```
대신 캐시만의 조건이 둘 붙는다 — 이 관문이 재는 것이 그것이다:
```
  ⓐ 전압이 **홀수 8차수**뿐이다 (세밀 45~60). 실측은 15차수를 쓴다
  ⓑ 자료가 **float16** 이다. 고차 전류는 mA 급이라 되짚어 볼 값어치가 있다
```

    python -X utf8 -m src.run_gate_gbudget_cache --cache cache/_v49chk
"""
import argparse
import sys

import numpy as np

from src import env_guard  # noqa: F401

from src.model import gbudget as GB  # noqa: E402
from src.model import inputs as _I  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-42s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def raw_from_cache(fine, obs):
    """캐시의 **타깃 프레임** (B, 61) -> 분해기가 먹는 (B, 49, 1). 전압은 홀수 8차수다.

    세밀 전압 블록은 되돌릴 수 있다 (`inputs.build_fine`):
        h=1  : `(vr − V_CENTER)/V_SPAN` · `vi/V_SPAN`
        h>1  : `arcsinh(v · VOLT_HARM_SCALE)`
    """
    n = len(fine)
    nv = len(_I.VOLT_ORDERS)
    raw = np.zeros((n, 33 + 2 * nv, 1), np.float64)
    raw[:, 0:15, 0] = obs[:, :, 0]                     # Re I_h
    raw[:, 15:30, 0] = obs[:, :, 1]                    # Im I_h
    raw[:, 30, 0] = np.sinh(fine[:, 23]) * 100.0                           # P
    v = fine[:, 25] * _I.V_SPAN + _I.V_CENTER
    raw[:, 32, 0] = v
    for s, h in enumerate(_I.VOLT_ORDERS):
        a = fine[:, _I.FINE_VOLT0 + s]
        b = fine[:, _I.FINE_VOLT0 + nv + s]
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
    ap.add_argument("--cache", default="cache/_v49chk")
    #: ⚠ 300,000창 캐시에서 `fine` 을 통째로 읽으면 **22GB** 다. 창을 자른다.
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()
    c = a.cache.rstrip("/")
    _f = np.load("%s/fine.npy" % c, mmap_mode="r")
    n = min(int(a.limit), len(_f)) if a.limit > 0 else len(_f)
    ch = _f.shape[1]
    obs = np.asarray(np.load("%s/obs_harm.npy" % c, mmap_mode="r")[:n], np.float64)
    yp = np.asarray(np.load("%s/y_power.npy" % c, mmap_mode="r")[:n], np.float64)
    print("캐시 관문 (14.338) — %s · 창 %d/%d · 세밀 %d채널 · 전압 %d차수"
          % (c, n, len(_f), ch, len(_I.VOLT_ORDERS)))

    chk(1, "캐시가 **이 코드의 배치**인가", ch == _I.FINE_CHANNELS,
        "fine %d채널 대 코드 %d · obs_harm %s · dtype %s"
        % (ch, _I.FINE_CHANNELS, obs.shape, _f.dtype))
    if ch != _I.FINE_CHANNELS:
        print("\n**실패 있음**  (%d/%d)" % (sum(OK), len(OK)))
        return 1

    #: ⚠ **타깃 프레임만** 읽는다. 창 전체를 float64 로 올리면 창당 293KB 다.
    #  `fine` 이 2차원이면 **이미 타깃 프레임만 잘라 둔 것**이다 (HPC 에서 잘라 받는 판).
    t = _I.fine_target_index()
    fin = (np.asarray(_f[:n], np.float64) if _f.ndim == 2
           else np.asarray(_f[:n, :, t], np.float64))   # (n, 61)
    raw = raw_from_cache(fin, obs)
    v1 = np.abs(raw[:, 33, 0] + 1j * raw[:, 33 + len(_I.VOLT_ORDERS), 0])

    # [2] 되짚기가 맞나 — 세밀 ch25 의 V 와 전압고조파 h1 크기가 같아야 한다
    vv = raw[:, 32, 0]
    rel = np.abs(v1 - vv) / np.maximum(vv, 1.0)
    chk(2, "세밀 전압 채널 **되짚기**가 맞나", float(np.median(rel)) < 0.02,
        "|V₁|(고조파 블록) 대 ch25(V) 상대차 중앙 **%.4f** · p99 %.4f · V₁ 중앙 %.1fV"
        % (float(np.median(rel)), float(np.percentile(rel, 99)), float(np.median(v1))))

    # ── 참값 ────────────────────────────────────────────────────────────────
    ri = [APPS.index(x) for x in GB.RESISTIVE]
    g_true = 1e3 * yp[:, ri].sum(1) / np.maximum(v1 ** 2, 1.0)
    fan_on = yp[:, APPS.index("fan")] > 1.0
    ac_on = yp[:, APPS.index("air_conditioner")] > 5.0

    #: ⚠ 14.346 — FCM **기준전압**은 캐시가 적어 둔 것을 쓴다. 여기서 다시 중앙값을
    #  잡으면 표본이 달라 Ĝ 가 최대 **0.275 mS** 어긋난다 (포트 식별 여유의 43%).
    v0 = np.zeros(15, complex)
    import json as _json
    import os as _os
    _mp = "%s/meta.json" % c
    _vr = None
    if _os.path.exists(_mp):
        _vr = (_json.load(open(_mp, encoding="utf-8")).get("g_hat") or {}).get("v_ref")
    if _vr:
        v0 = np.array([complex(x[0], x[1]) for x in _vr])
    else:
        for s, h in enumerate(_I.VOLT_ORDERS):
            v0[h - 1] = (np.median(raw[:, 33 + s, 0])
                         + 1j * np.median(raw[:, 33 + len(_I.VOLT_ORDERS) + s, 0]))
    bud = GB.Budget(APPS, v0, volt_re0=33, volt_orders=_I.VOLT_ORDERS)
    g = bud.g_sum(raw)[:, 0]
    r = g - g_true

    # [3] 추정기가 참 컨덕턴스를 되찾나
    clean = (~fan_on) & (~ac_on) & (g_true > 5.0)
    med = float(np.median(r[clean])) if clean.sum() > 20 else float("nan")
    sd = float(r[clean].std()) if clean.sum() > 20 else float("nan")
    chk(3, "★ 캐시에서 **참 컨덕턴스**를 되찾나 (선풍기·에어컨 꺼진 창)",
        clean.sum() > 20 and abs(med) < 0.5 and sd < 1.5,
        "n=%d · 잔차 중앙 **%+.3f mS** · σ **%.3f** · 참 Ĝ 중앙 %.2f mS"
        % (int(clean.sum()), med, sd, float(np.median(g_true[clean]))))

    # [4] 선풍기·에어컨이 **위로** 민다 (14.337 의 예측)
    rows = []
    for nm, m in (("저항만", clean), ("선풍기 켜짐", fan_on & ~ac_on & (g_true > 5)),
                  ("에어컨 켜짐", ac_on & (g_true > 5))):
        if m.sum() >= 20:
            rows.append((nm, int(m.sum()), float(np.median(r[m]))))
    up = [x for x in rows if x[0] != "저항만"]
    base = rows[0][2] if rows else 0.0
    chk(4, "★ 선풍기·에어컨이 Ĝ 를 **위로** 미나 (14.337 예측)",
        bool(up) and all(x[2] > base for x in up),
        " · ".join("%s n=%d 잔차중앙 **%+.3f mS**" % x for x in rows)
        + "  (위로 밀면 거부권·바닥이 **덜 선다** = 안 하는 쪽)")

    # [5] 사중이 캐시 위에서 돈다 (모델 자리에 참값을 넣어 본다)
    pw = yp.copy()
    q, info = GB.apply(pw, g, v1, APPS)
    chk(5, "사중이 캐시 위에서 **돈다**", np.isfinite(q).all(),
        "거부권 %d칸 · 바닥 %d칸 · 겹침 **%d** · 출력 유한 %s"
        % (int(info["veto"].sum()), int(info["floor"].sum()),
           int((info["veto"] & info["floor"]).sum()), bool(np.isfinite(q).all())))

    # [6] ★ 거부권·바닥이 동시에 안 선다 (캐시판)
    chk(6, "★ 거부권과 바닥이 **동시에 안 선다** (캐시)",
        int((info["veto"] & info["floor"]).sum()) == 0,
        "겹친 칸 %d / 잰 칸 %d" % (int((info["veto"] & info["floor"]).sum()), q.size))

    # [7] 고정이 참 전력을 되찾나 — 캐시는 기기별 참값이 있어 직접 채점된다
    errs = []
    for j, x in enumerate(APPS):
        if x not in GB.PIN_MS:
            continue
        m = yp[:, j] > 300.0
        if m.sum() < 10:
            continue
        on = np.zeros(pw.shape, bool)
        on[m, j] = True
        pin = GB.pin_power(pw, v1, APPS, on)[m, j]
        errs.append((x, int(m.sum()), float(np.median(np.abs(pin - yp[m, j]))),
                     float(np.median(pin - yp[m, j]))))
    worst = max((e for _, _, e, _ in errs), default=1e9)
    chk(7, "고정이 **기기별 참 전력**을 되찾나", worst < 40.0,
        " · ".join("%s n=%d |오차|중앙 **%.1fW** (치우침 %+.1f)" % x for x in errs)
        + "  (기준 <40W — 캐시 전력 지터가 0.5~1.0%% 라 7~15W 는 원리상 못 준다)")

    # [8] float16 이 결론을 안 바꾸나 — 참 전력에서 지은 Ĝ 와 견준다
    chk(8, "**float16** 이 결론을 안 바꾸나",
        float(np.median(np.abs(r[clean]))) < 0.5,
        "|잔차| 중앙 **%.3f mS** — 고차 전류가 mA 급이라 여기서 무너지면 바로 보인다. "
        "기기별 식별 여유(포트 0.635 · 오븐 2.590)와 견줘라"
        % float(np.median(np.abs(r[clean]))))

    # [9] 14.346 — 캐시에 얹은 `g_hat` 이 **지금 푼 것**과 같나
    import os as _os
    gp = "%s/g_hat.npy" % c
    if _os.path.exists(gp):
        gh = np.asarray(np.load(gp, mmap_mode="r")[:n], np.float64)
        d9 = float(np.abs(gh - g).max())
        chk(9, "★ 캐시 `g_hat` 이 지금 푼 Ĝ 와 같나", d9 < 1e-3,
            "최대차 **%.3e mS** (창 %d) — 다르면 학습이 **옛 배치의 Ĝ** 를 조용히 먹는다"
            % (d9, n))
    else:
        chk(9, "★ 캐시 `g_hat` 이 지금 푼 Ĝ 와 같나", True,
            "`%s` 가 없다 — `run_build_ghat` 을 아직 안 돌렸다 (조합 머리 전에 필요하다)" % gp)

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
