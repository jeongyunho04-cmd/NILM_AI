# -*- coding: utf-8 -*-
"""관문 — `--head-drop` 이 **그 덩이만 정확히 빼는가** (14.128).

무엇을 묻는 판인가
------------------
사용자: *"세밀갈래를 처음부터 만드는 건 어때. 너무 많은 정보가 헤드에 덕지덕지 붙어
있는 모양새인데 이게 문제일 수도 있지 않을까?"*

재 봤더니 중복은 실재한다 (실측 2753창, `cnn_pcap_s0`, 머리 입력 765칸):
```
  유효 차원   분산 90%까지 79칸 · 95% 144칸 · 99% 339칸 · 참여비 **20.2**
  덩이별 R^2 (그 덩이를 **나머지 전부**로 맞힌 값)
    과거평균 0.982 · 원시창통계 0.982 · 탭0 0.980 · 탭1 0.976 · 원시타깃 0.968
    과거최대 0.947 · 광역평균 0.929 · 탭4 0.925 · 깊은탭 0.905
```
**어느 덩이를 빼도 나머지로 90~98% 복원된다.**

⚠ 그런데 **불안정의 원인이라는 증거는 없다** — 같은 팔 3시드의 오븐 on_logit 상관이
  0.945~0.960, 판정 일치 0.94~0.95 다. 그래서 이 손잡이는 "고침" 이 아니라
  **"빼도 되나" 를 묻는 실험**이고, 통과 조건도 "좋아진다" 가 아니라 **"안 나빠진다"** 다.

⚠⚠ 추론에서 0 으로 죽여 보는 것으로는 못 묻는다. 14.123 이 그걸로 원시 창통계를
  −3.6%p 로 읽었는데 `asinh(P/100)=0` 은 실측에 없는 입력이라 **분포 이탈**이었다.
  빼고 **다시 학습**해야 한다.

무엇을 확인하나
---------------
```
  [1] 빈 값이면 **비트 동일**
  [2] 차원이 정확히 준다 (rawstat -4 · tap0 -64 · pastmean -128 · 셋 -196)
  [3] ★ **그 덩이만** 빠진다 — 머리 입력이 원본에서 **해당 열만 지운 것과 비트 동일**
      (몸통 가중치는 trunk 보다 먼저 뽑히므로 두 판이 정확히 비교된다)
  [4] `fine_dim_mask` 가 머리 입력과 맞는다
  [5] 몸통 파라미터 불변
  [6] 이름이 틀리면 **거부**한다 (조용히 무시하지 않는다)
  [7] autocast(bfloat16) · 체크포인트 왕복
```

    python -X utf8 src/run_gate_headdrop.py
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
#: ⚠ 14.138 — `prior_kappa` 를 **여기서 켠다**. `NILMNet` 기본은 0.0 인데
#:   트레이너 기본은 **8.0** 이라, 안 켜면 관문이 물리 프라이어 블록을
#:   **한 번도 안 태운다**. 984924 가 그래서 통과하고 학습에서 48초에 죽었다.
BASE = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4), prior_kappa=8.0)
#: 덩이 경계 (ns=wns=1 일 때). 14.128 이 실측 체크포인트에서 확인한 배치다.
SPAN = {"rawtgt": (0, 57), "tap0": (57, 121), "tap1": (121, 185), "tap4": (185, 313),
        "pastmean": (313, 441), "pastmax": (441, 569), "wide": (697, 761),
        "rawstat": (761, 765)}
DROP = "rawstat,tap0,pastmean"
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
    """머리 첫 층이 **실제로 받는** 벡터. 따로 짓지 않고 forward 를 태워 뽑는다."""
    box = []
    h = m.trunk[0].register_forward_pre_hook(lambda mod, inp: box.append(inp[0].clone()))
    with torch.no_grad():
        m(f, w)
    h.remove()
    return box[0]


def main() -> int:
    print("`--head-drop` — 머리 입력에서 덩이를 뺀다 (14.128)")
    print("  중복 측정: 유효 차원 참여비 **20.2** · 모든 덩이 R^2 0.905~0.982\n")
    f, w = torch.randn(4, 57, 600), torch.randn(4, 47, 120)

    off, off2 = _net(0), _net(0, head_drop="")
    with torch.no_grad():
        o0, o1 = off(f, w), off2(f, w)
    ck("[1] 빈 값이면 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])))

    base = off.trunk[0].in_features
    x0 = head_in(off, f, w)
    for nm in ("rawstat", "tap0", "pastmean", DROP):
        m = _net(0, head_drop=nm)
        names = [x.strip() for x in nm.split(",")]
        cut = sum(SPAN[a][1] - SPAN[a][0] for a in names)
        ok_dim = m.trunk[0].in_features == base - cut
        # ★ 그 덩이 **만** 빠졌나 — 원본에서 해당 열을 지운 것과 비트 동일이어야 한다
        keep = np.ones(base, bool)
        for a in names:
            keep[SPAN[a][0]:SPAN[a][1]] = False
        same = torch.equal(head_in(m, f, w), x0[:, torch.from_numpy(np.flatnonzero(keep))])
        ck("[2/3] `%s` — 차원 -%d 이고 **그 열만** 빠진다" % (nm, cut), ok_dim and same,
           "머리 입력 %d -> %d · 열 일치 %s" % (base, m.trunk[0].in_features, same))
        ck("   [4] `fine_dim_mask` 가 맞는다",
           int(m.fine_dim_mask.numel()) == m.trunk[0].in_features,
           "%d == %d" % (int(m.fine_dim_mask.numel()), m.trunk[0].in_features))

    m = _net(0, head_drop=DROP)
    ck("[5] 몸통 파라미터 불변",
       sum(p.numel() for p in off.fine.parameters())
       == sum(p.numel() for p in m.fine.parameters()),
       "%d == %d · 머리 파라미터 %d -> %d"
       % (sum(p.numel() for p in off.fine.parameters()),
          sum(p.numel() for p in m.fine.parameters()),
          off.trunk[0].weight.numel(), m.trunk[0].weight.numel()))

    err = ""
    try:
        _net(0, head_drop="rawstat,오타")
        bad = False
    except ValueError as e:
        bad, err = True, str(e)[:52]
    ck("[6] 이름이 틀리면 **거부**한다", bad, err)

    err = ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(m(f, w)["on_logit"].float().abs().max())
        g = bool(np.isfinite(v))
    except Exception as e:                               # noqa: BLE001
        g, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[7a] autocast(bfloat16) 에서 돈다", g, err or "OK")

    import tempfile
    from src.run_gate_check import load_model
    p = Path("results/cnn_pcap_s0.pt")
    if not p.exists():
        ck("[7b] 체크포인트 왕복", False, "%s 가 없다" % p)
    else:
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "t.pt"
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["head_drop"] = ""
            torch.save(c, q)
            mm = load_model(str(q), "cpu")[0]
            ck("[7b] 체크포인트 왕복 — 빈 값이면 옛 가중치가 그대로 실린다",
               mm.head_drop == () and mm.trunk[0].in_features == base,
               "head_drop=%r · 머리 %d" % (mm.head_drop, mm.trunk[0].in_features))

    print("")
    print("  ⚠ 통과 조건은 **'좋아진다' 가 아니라 '안 나빠진다'** 다 — 중복이 무해한지를")
    print("    묻는 실험이다. 무너지면 그 중복이 일을 하고 있었다는 뜻이고, 그것도 소득이다.")
    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
