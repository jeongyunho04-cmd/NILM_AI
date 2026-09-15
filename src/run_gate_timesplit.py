# -*- coding: utf-8 -*-
"""관문 — `--fine-time-split` 이 **미래를 현재에서 떼어내는가** (14.116).

무엇을 고치려는 것인가
----------------------
사용자: *"미래를 더 많이 알수록 혼동이 는다는 게 말이 되냐. 이게 말이 된다는 것 자체가
모델 구조에 결함이 있다는 거잖아."* **맞다.** 정보가 늘어서 나빠질 수는 없다 — 모델이
무시하면 되니까. 나빠진다면 그 정보가 **"미래의 것"이라는 표시 없이** 들어온다는 뜻이다.

```
  세밀 창 600 · 타깃 239 · 미래 361사이클 = 6.02초
  깊은 탭 수용영역   wtap 없음 187 (타깃 좌우 ±1.56초)
                    **wtap 켬  763 (타깃 좌우 ±6.36초)**  <- 창 전체를 덮는다
```
`--fine-extra-dilations 32,64` 를 켜면 `h[:, :, 239]` 라는 "지금의 특징" 안에 **미래
6.02초가 이미 섞여** 있다. 실측 개입이 그것을 확인했다 — test_5 260.0초에서 미래를
타깃값으로 덮으면 핫플 헛게이트가 **0.878 -> 0.004** 로 사라진다. 전체로는 오븐
**헛detect 의 22%** 가 미래에 기대는데 **참detect 는 0.8%** 뿐이다.

⚠ `--seg-pool` 과 다르다. 저쪽은 **전역 풀링**을 타깃에서 쪼갠다. 14.42 가 그 처방을
  낼 때 몸통은 RF 187 이라 미래가 **풀링으로만** 샜다 — 그때는 맞았고, 그래서 14.72 가
  `--seg-pool 2` 를 재고 기각한 것(−0.9 ± 2.0)도 그 구조에서의 결과다.
  **wtap 을 켠 지금은 새는 길이 탭 안으로 옮겨갔고, 풀링을 쪼개도 못 막는다.**

무엇을 확인하나
---------------
```
  [1] 끄면 **비트 동일** (같은 씨앗에서 출력이 한 비트도 안 다르다)
  [2] 켜면 달라진다
  [3] 몸통 파라미터가 **안 는다** (가중치 공유) — 머리 입력만 +2·c2
  [4] ★ **타깃 탭이 미래를 안 본다** — 켠 경로의 타깃 특징이 `과거만 넣은 것`과 비트 동일
  [5] ★ 음성 대조 — **끈** 경로에서는 그 둘이 **다르다** (미래가 섞인다는 증거)
  [6] 정보를 **버리지 않았다** — 미래만 바꿔도 출력은 여전히 달라진다 (미래 요약 경로)
  [7] `fine_dim_mask` 가 `trunk_in` 과 맞는다 (갈래 드롭아웃이 안 어긋난다)
  [8] autocast(bfloat16) 에서 돈다
```

    python -X utf8 src/run_gate_timesplit.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import NILMNet  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
WTAP = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4))
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    m = NILMNet(APPS, [3] * len(APPS), **dict(WTAP, **kw))
    m.eval()
    return m


def _trunk_tap(m, fine, past_only: bool):
    """몸통을 태워 **타깃 열의 깊은 특징**을 낸다. `past_only` 면 과거 조각만 넣는다."""
    t = m.target_pos
    h = fine[:, :, :t + 1] if past_only else fine
    with torch.no_grad():
        for blk in m.fine:
            h = blk(h)
    return h[:, :, -1] if past_only else h[:, :, t]


def main() -> int:
    print("`--fine-time-split` — 미래를 현재에서 떼어낸다 (14.116)")
    rf = 1 + 6 * sum((1, 2, 4, 8, 16, 32, 64))
    print("  세밀 창 600 · 타깃 239 · 미래 361사이클(6.02초) · 깊은 탭 RF **%d** (±%.2f초)\n"
          % (rf, rf / 2 / 60))

    f = torch.randn(3, 57, 600)
    w = torch.randn(3, 47, 120)
    off, off2, on = _net(0), _net(0, fine_time_split=False), _net(0, fine_time_split=True)

    with torch.no_grad():
        o0, o1, o2 = off(f, w), off2(f, w), on(f, w)
    same = all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k]))
    ck("[1] 끄면 **비트 동일**", same)
    ck("[2] 켜면 달라진다", not torch.equal(o0["on_logit"], o2["on_logit"]),
       "on_logit 최대차 %.4g" % float((o0["on_logit"] - o2["on_logit"]).abs().max()))

    n_tr_off = sum(p.numel() for p in off.fine.parameters())
    n_tr_on = sum(p.numel() for p in on.fine.parameters())
    c2 = off.trunk[0].in_features
    d_in = on.trunk[0].in_features - c2
    ck("[3] 몸통 파라미터 불변 (가중치 공유)", n_tr_off == n_tr_on,
       "몸통 %d = %d · 머리 입력 %d -> %d (+%d)"
       % (n_tr_off, n_tr_on, c2, on.trunk[0].in_features, d_in))

    # ★ 진짜 검사: **모델의 실제 forward 경로**에서 타깃 탭을 꺼내, 미래만 바꿔 본다.
    #   (앞선 판은 몸통을 따로 돌려 재는 바람에 forward 를 안 탔다 — 검사가 틀렸다.)
    f2 = f.clone()
    f2[:, :, 240:] = torch.randn_like(f2[:, :, 240:])    # 미래만 바꾼다
    with torch.no_grad():
        on(f, w); tap_a = on._tap_now.clone()
        o3 = on(f2, w); tap_b = on._tap_now.clone()
        off(f, w); tap_c = off._tap_now.clone()
        off(f2, w); tap_d = off._tap_now.clone()
    ck("[4] ★ 켠 경로 — 미래만 바꿔도 타깃 탭이 **비트 동일**",
       torch.equal(tap_a, tap_b), "최대차 %.4g" % float((tap_a - tap_b).abs().max()))
    d = float((tap_c - tap_d).abs().max())
    ck("[5] ★ 음성 대조 — **끈** 경로는 타깃 탭이 미래에 따라 **움직인다**", d > 1e-6,
       "최대차 %.4g  (RF 763 이 창 전체를 덮는다)" % d)
    ck("[6] 정보를 안 버렸다 — 미래만 바꿔도 출력이 달라진다",
       not torch.equal(o2["on_logit"], o3["on_logit"]),
       "on_logit 최대차 %.4g" % float((o2["on_logit"] - o3["on_logit"]).abs().max()))

    ck("[7] `fine_dim_mask` 가 머리 입력과 맞는다",
       int(on.fine_dim_mask.numel()) == on.trunk[0].in_features,
       "%d == %d · 세밀 유래 %d"
       % (int(on.fine_dim_mask.numel()), on.trunk[0].in_features, int(on.fine_dim_mask.sum())))

    err = ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(on(f, w)["on_logit"].float().abs().max())
        good8 = np.isfinite(v)
    except Exception as e:                               # noqa: BLE001
        good8, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[8] autocast(bfloat16) 에서 돈다", good8, err or "OK")

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
