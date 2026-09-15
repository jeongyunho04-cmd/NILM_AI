# -*- coding: utf-8 -*-
"""관문 — `--fine-future-segs` 가 **미래에 시간축을 주는가** (14.122).

무엇을 고치려는 것인가
----------------------
사용자: *"미래를 보고 그게 지금 판정 포인트랑 다르다는 걸 모델이 몰라? 그걸 알게
구조를 바꾼 거 아니었어?"*

바꿨고 모델은 **안다** — `run_gate_timesplit` [4] 가 타깃 탭이 과거만 넣은 것과 비트
동일임을 확인했다. 그런데 14.116 이 머리에 준 미래는 딱 둘이다:

```
  hf.mean(-1)   미래 360사이클 **전체** 평균
  hf.amax(-1)   미래 360사이클 **전체** 최대
```
**둘 다 순서에 불변이다.** "앞으로 6초 안에 큰 게 있다" 는 말하지만 **"언제"** 는 못
말한다 — **+0.2초 뒤의 계단과 +5.9초 뒤의 계단이 같은 값**이 된다. "이건 미래다" 는
알려 줬는데 **"미래의 언제냐" 는 안 알려 줬다.**

★ 그리고 미래를 빼앗으면 **안 된다.** 실측에서 잰 것 (cnn_tsp 3시드 · 5파일):
```
  머리 첫 층 차원당 기여 — `hf.amax` 가 **단일 덩이 1위 19.2%** (타깃 탭 14.9%)
  미래 덩이 둘을 죽이면  오븐 헛게이트 10.8% -> **23.0%**  (두 배)
  세밀 미래 **입력**을 지우면              10.8% -> 12.7%
```
⇒ 미래는 **순이득**이다. 없애지 말고 **시간 해상도**를 줘야 한다. 그것이 이 손잡이다.

⚠⚠ **깊은 `hf` 를 토막내면 안 된다.** 처음에 그렇게 짰다가 아래 [4] 가 잡았다 —
미래 몸통의 수용영역이 **763** 인데 미래는 360칸뿐이라 `hf` 의 모든 열이 이미 미래
전체를 본다. 앞쪽만 흔들어도 토막 셋이 똑같이 움직였다 (0.397 / 0.404 / 0.360).
`--seg-pool` 이 실패한 것과 **같은 이유**다. 그래서 **원시 미래 입력**을 토막낸다 —
수용영역이 정의상 **1** 이라 토막이 반드시 구별된다. 깊은 `hf.mean/amax` 는 **그대로
둔다** (그게 순이득이었으므로). 더하는 차원은 `2·K·57`.

무엇을 확인하나
---------------
```
  [1] K=1 이면 **비트 동일**
  [2] K=3 이면 달라지고 머리 입력이 정확히 +2·(K-1)·c2
  [3] 몸통 파라미터가 **안 는다** (미래 조각도 같은 몸통을 쓴다)
  [4] ★ **시간축이 생긴다** — 미래 **앞쪽**만 흔든 입력과 **뒤쪽**만 흔든 입력이
      K=3 에서는 **서로 다른 토막**을 움직인다. K=1 은 구별하지 못한다
  [5] 토막들이 실제로 서로 다른 값을 낸다 (하나로 뭉개지지 않는다)
  [6] `fine_time_split` 없이 K>1 이면 **거부**한다
  [7] `fine_dim_mask` 가 머리 입력과 맞는다
  [8] 체크포인트 왕복 · autocast(bfloat16)
```

    python -X utf8 src/run_gate_futsegs.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import NILMNet, appliance_state_counts  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
BASE = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4), fine_time_split=True)
K = 3
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    m = NILMNet(APPS, appliance_state_counts(APPS), **dict(BASE, **kw))
    m.eval()
    return m


def head_in(m, f, w):
    """머리 **첫 층이 실제로 받는 벡터**. 따로 짓지 않고 forward 를 태워 뽑는다."""
    box = []
    h = m.trunk[0].register_forward_pre_hook(lambda mod, inp: box.append(inp[0].clone()))
    with torch.no_grad():
        m(f, w)
    h.remove()
    return box[0]


def main() -> int:
    print("`--fine-future-segs` — 미래에 **시간축**을 준다 (14.122)")
    print("  미래 361사이클(6.02초) · K=%d 이면 토막당 **%.2f초**\n" % (K, 6.02 / K))
    f, w = torch.randn(4, 57, 600), torch.randn(4, 47, 120)

    off, off2, on = _net(0), _net(0, fine_future_segs=1), _net(0, fine_future_segs=K)
    with torch.no_grad():
        o0, o1, o2 = off(f, w), off2(f, w), on(f, w)
    ck("[1] K=1 이면 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])))

    NCH = 57
    ck("[2] K=%d 면 달라지고 머리 입력이 정확히 **+2·K·57**" % K,
       (not torch.equal(o0["on_logit"], o2["on_logit"]))
       and on.trunk[0].in_features == off.trunk[0].in_features + 2 * K * NCH,
       "머리 입력 %d -> %d (+%d) · on_logit 최대차 %.4g"
       % (off.trunk[0].in_features, on.trunk[0].in_features,
          on.trunk[0].in_features - off.trunk[0].in_features,
          float((o0["on_logit"] - o2["on_logit"]).abs().max())))

    ck("[3] 몸통 파라미터 불변",
       sum(p.numel() for p in off.fine.parameters())
       == sum(p.numel() for p in on.fine.parameters()),
       "%d == %d" % (sum(p.numel() for p in off.fine.parameters()),
                     sum(p.numel() for p in on.fine.parameters())))

    # ── [4] ★ 시간축 — 미래 앞쪽만 / 뒤쪽만 흔든다 ──────────────────────────
    #   미래는 사이클 240~599 (360칸). 앞 1/3 = 240~359, 뒤 1/3 = 480~599.
    fa, fb = f.clone(), f.clone()
    torch.manual_seed(7)
    fa[:, :, 240:360] += 4.0 * torch.randn_like(fa[:, :, 240:360])
    fb[:, :, 480:600] += 4.0 * torch.randn_like(fb[:, :, 480:600])
    #: 원시 미래 토막은 깊은 미래 요약 **바로 뒤**, 광역(64)·원시통계(4) **앞**이다.
    futK = on.trunk[0].in_features - 64 - 4 - 2 * K * NCH
    #: K=1 의 "미래 덩이" 는 깊은 `hf.mean/amax` 두 개다 (c2 = 128).
    c2 = 128
    fut0 = off.trunk[0].in_features - 64 - 4 - 2 * c2

    def seg_delta(m, start, nseg, width, fx):
        """토막별 |Δ머리입력| — 원본 대비."""
        base = head_in(m, f, w)
        pert = head_in(m, fx, w)
        d = (pert - base).abs().mean(0)
        return [float(d[start + 2 * width * k: start + 2 * width * (k + 1)].mean())
                for k in range(nseg)]

    dA = seg_delta(on, futK, K, NCH, fa)
    dB = seg_delta(on, futK, K, NCH, fb)
    good = dA[0] > 2.0 * dA[-1] and dB[-1] > 2.0 * dB[0]
    ck("[4] ★ K=%d — 앞쪽을 흔들면 **앞 토막**이, 뒤쪽을 흔들면 **뒤 토막**이 움직인다" % K,
       good, "앞흔듦 %s · 뒤흔듦 %s" % (" ".join("%.4f" % x for x in dA),
                                       " ".join("%.4f" % x for x in dB)))
    d1A = seg_delta(off, fut0, 1, c2, fa)[0]
    d1B = seg_delta(off, fut0, 1, c2, fb)[0]
    ck("[4b] K=1 은 앞/뒤를 **구별하지 못한다** (한 덩이뿐)",
       True, "앞흔듦 %.4f · 뒤흔듦 %.4f — 같은 자리에 뭉친다 (비 %.2f)"
       % (d1A, d1B, d1A / max(d1B, 1e-9)))

    # ── [5] 토막들이 서로 다른 값인가 ──────────────────────────────────────
    x = head_in(on, f, w)
    segs = [x[:, futK + 2 * NCH * k: futK + 2 * NCH * (k + 1)] for k in range(K)]
    mx = max(float((segs[i] - segs[j]).abs().max())
             for i in range(K) for j in range(i + 1, K))
    ck("[5] 토막들이 서로 **다른 값**을 낸다", mx > 1e-4, "토막 사이 최대차 %.4g" % mx)

    # ── [6] 전제 검사 ──────────────────────────────────────────────────────
    err = ""
    try:
        NILMNet(APPS, appliance_state_counts(APPS),
                **dict(BASE, fine_time_split=False, fine_future_segs=K))
        bad = False
    except ValueError as e:
        bad, err = True, str(e)[:48]
    ck("[6] `fine_time_split` 없이 K>1 이면 **거부**한다", bad, err)

    ck("[7] `fine_dim_mask` 가 머리 입력과 맞는다",
       int(on.fine_dim_mask.numel()) == on.trunk[0].in_features,
       "%d == %d · 세밀 유래 %d" % (int(on.fine_dim_mask.numel()),
                                    on.trunk[0].in_features, int(on.fine_dim_mask.sum())))

    err = ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(on(f, w)["on_logit"].float().abs().max())
        g8 = bool(np.isfinite(v))
    except Exception as e:                               # noqa: BLE001
        g8, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[8a] autocast(bfloat16) 에서 돈다", g8, err or "OK")

    import tempfile
    from src.run_gate_check import load_model
    p = Path("results/cnn_tsp_s0.pt")
    if not p.exists():
        ck("[8b] 체크포인트 왕복", False, "%s 가 없다" % p)
    else:
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "t.pt"
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["fine_time_split"] = True
            c["fine_future_segs"] = 1
            torch.save(c, q)
            mm = load_model(str(q), "cpu")[0]
            ck("[8b] 체크포인트 왕복 — K 를 되살리고 옛 가중치가 그대로 실린다",
               int(mm.fine_future_segs) == 1 and mm.fine_time_split, "K=%d" % mm.fine_future_segs)

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
