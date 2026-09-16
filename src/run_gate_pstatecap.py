# -*- coding: utf-8 -*-
"""관문 — `--p-state-cap` 이 **표류한 슬롯만 자르는가** (14.121).

무엇을 고치려는 것인가
----------------------
`p_raw = Σ_s mix[s]·p_states[s]` 이고 `p_states = softplus(·)` 는 **위로 상한이 없다.**
13.84.68 은 슬롯이 **아래로** 죽는 것을 막았다 (`state_power_init`). 위쪽은 안 막혔고,
실제로 갔다. 실측 창에서 9개 체크포인트를 재 보면:

```
  beam_projector s1  초기  10.0W  ->  **83.6배**   (자리채움 슬롯 — 주석은 "p90 4.5")
  laptop_charger s1  초기  36.4W  ->    4.31배
  minipc         s1  초기  10.7W  ->    2.17배
  그 밖 17개 슬롯                 ->    1.52배 이하 (대부분 0.8~0.9배)
```
기전은 13.84.68 의 거울상이다 — 혼합이 **거의** 안 고르는 슬롯은 기울기가
`mix[s]·∂L/∂p_raw` 로 작지만 **부호가 일정**해서 300에포크 동안 쌓인다. 그 뒤엔
상태 머리가 6%만 그쪽을 골라도 `p_raw` 가 통째로 튄다 — `test_4` 310~371초에서
빔의 `p_raw` 가 **99.2W** 였다 (참 40.6W). 게이트 0.346 이 곱해져 26.6W 로 보였을 뿐이다.

⚠ R 은 **위 측정에서 온다.** 통과시키려고 고르는 값이 아니다
  ([[dont-loosen-a-gate-to-make-it-pass]]). R=3 이면 병적인 둘만 걸리고 나머지
  18개 슬롯은 손도 안 탄다.

무엇을 확인하나
---------------
```
  [1] 끄면 (R=0) **비트 동일**
  [2] 켜도 **상한 밑이면 비트 동일** — 정상 슬롯은 손도 안 댄다
  [3] 상한 위는 실제로 잘린다 (`p_states <= R x S_STATE`)
  [4] ★ **진짜 체크포인트**에서 잰다 — 빔 s1 이 잘리고 `p_raw` 가 내려간다
  [5] 버퍼가 `state_dict` 에 **안 들어간다** (옛 체크포인트가 안 깨진다)
  [6] 체크포인트 왕복 — `load_model` 이 R 을 되살린다
```

    python -X utf8 src/run_gate_pstatecap.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model import inputs as _I  # noqa: E402

import torch  # noqa: E402

from src.model.losses import S_STATE  # noqa: E402
from src.model.net import NILMNet, appliance_state_counts  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
WTAP = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4))
R = 3.0
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    m = NILMNet(APPS, appliance_state_counts(APPS), **dict(WTAP, **kw))
    m.eval()
    return m


def main() -> int:
    print("`--p-state-cap` — 표류한 상태 전력 슬롯만 자른다 (14.121)\n")
    #: 14.331 — 채널 수를 박지 않는다. `VOLT_ORDERS` 를 넓히면 여기가 조용히 틀린다.
    f = torch.randn(4, _I.FINE_CHANNELS, 600)
    w = torch.randn(4, _I.WIDE_CHANNELS, 120)

    off, off2, on = _net(0), _net(0, p_state_cap=0.0), _net(0, p_state_cap=R)
    with torch.no_grad():
        o0, o1 = off(f, w), off2(f, w)
    ck("[1] 끄면 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])))

    # [2] 갓 지은 판은 `state_power_init` 때문에 슬롯이 정확히 S_STATE 다 -> 전부 상한 밑
    with torch.no_grad():
        o2 = on(f, w)
    hi = float((o0["power_states"] / on.p_state_cap_w[None].clamp(max=1e9)).max())
    ck("[2] 켜도 **상한 밑이면 비트 동일** (정상 슬롯은 안 건드린다)",
       torch.equal(o0["power"], o2["power"]),
       "최대 p_states/상한 = %.3f (1 밑이면 아무것도 안 잘린다)" % hi)

    # [3] 슬롯 하나를 **일부러 띄워** 잘리는지 본다 — 빔 s1 을 550W 로 (실측한 표류값)
    drift = _net(0, p_state_cap=R)
    j, sid = APPS.index("beam_projector"), 1
    with torch.no_grad():
        drift.heads[j].bias[sid] = 550.0
    base = _net(0, p_state_cap=0.0)
    with torch.no_grad():
        base.heads[j].bias[sid] = 550.0
        od, ob = drift(f, w), base(f, w)
    cap_w = R * S_STATE["beam_projector"][sid]
    got = float(od["power_states"][:, j, sid].max())
    raw = float(ob["power_states"][:, j, sid].max())
    ck("[3] 상한 위는 잘린다", abs(got - cap_w) < 1e-3 and raw > cap_w * 5,
       "빔 s1 %.1fW -> **%.1fW** (상한 %.1f x %.1f = %.1fW)"
       % (raw, got, R, S_STATE["beam_projector"][sid], cap_w))
    ck("[3b] 그 기기의 `p_raw` 가 실제로 내려간다",
       float(od["power_raw"][:, j].mean()) < float(ob["power_raw"][:, j].mean()),
       "p_raw %.1fW -> %.1fW"
       % (float(ob["power_raw"][:, j].mean()), float(od["power_raw"][:, j].mean())))
    other = [k for k in range(len(APPS)) if k != j]
    ck("[3c] **다른 기기는 비트 동일** (자른 것만 달라진다)",
       torch.equal(od["power_raw"][:, other], ob["power_raw"][:, other]))

    # [4] ★ 진짜 체크포인트 — 따로 지은 물건이 아니라 구운 가중치에서 잰다
    #
    # ⚠ 14.336 — **이름을 박지 않는다.** 옛 판은 `results/cnn_hzwt_s0.pt` 를 박아 뒀는데,
    #   14.331 이 배치를 넓히자 그 체크포인트(광역 47채널)가 `load_model` 의 하드 스톱에
    #   걸려 **985993 이 여섯 토막 전부 12초에 죽었다.** 배치를 바꾸면 "맞는 체크포인트가
    #   하나도 없는" 부트스트랩 구간이 반드시 생긴다 — 그때는 **실패가 아니라 건너뜀**이다.
    #   ⚠ 건너뜀이 굳지 않게 **자동으로 찾는다**: 새 팔이 하나라도 구워지면 다시 돈다.
    from src.run_gate_check import load_model, newest_compatible_ckpt
    _q = newest_compatible_ckpt()
    p = Path(_q) if _q else Path("results/__none__.pt")
    if not p.exists():
        print("  [4] **건너뜀** — 지금 배치(FINE %d · WIDE %d · 차수 %s)와 맞는 체크포인트가"
              " results/ 에 하나도 없다.\n"
              "      배치를 바꾼 직후의 부트스트랩이다. 한 팔이라도 구워지면 이 줄이 다시 돈다."
              % (_I.FINE_CHANNELS, _I.WIDE_CHANNELS, tuple(_I.VOLT_ORDERS)))
    else:
        m_off = load_model(str(p), "cpu")[0]
        m_on = load_model(str(p), "cpu")[0]
        m_on.p_state_cap = R
        cap = torch.full((len(APPS), m_on.n_pow), float("inf"))
        for k, a in enumerate(APPS):
            for sid_, w_ in S_STATE.get(a, {}).items():
                if 0 <= sid_ < m_on.n_pow and w_ > 0:
                    cap[k, sid_] = R * float(w_)
        m_on.register_buffer("p_state_cap_w", cap, persistent=False)
        with torch.no_grad():
            a_, b_ = m_off(f, w), m_on(f, w)
        d = float((a_["power_states"][:, j, sid] - b_["power_states"][:, j, sid]).max())
        ck("[4] ★ 진짜 체크포인트에서 빔 s1 이 잘린다", d > 1.0,
           "%.1fW -> %.1fW (%.1fW 깎였다)"
           % (float(a_["power_states"][:, j, sid].mean()),
              float(b_["power_states"][:, j, sid].mean()), d))

    # [5] 버퍼가 state_dict 에 안 들어간다 (옛 체크포인트 호환)
    ck("[5] `p_state_cap_w` 가 `state_dict` 에 **없다**",
       not any("p_state_cap_w" in k for k in on.state_dict()),
       "키 %d개 · off 와 같은 집합 %s"
       % (len(on.state_dict()), set(on.state_dict()) == set(off.state_dict())))

    # [6] 왕복
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        q = Path(td) / "t.pt"
        base_ck = torch.load(str(p), map_location="cpu",
                             weights_only=False) if p.exists() else None
        if base_ck is None:
            print("  [6] **건너뜀** — [4]와 같은 까닭 (맞는 체크포인트가 없다)")
        else:
            base_ck["p_state_cap"] = R
            torch.save(base_ck, q)
            mm = load_model(str(q), "cpu")[0]
            ck("[6] 체크포인트 왕복 — `load_model` 이 R 을 되살린다",
               float(mm.p_state_cap) == R and hasattr(mm, "p_state_cap_w"),
               "R=%.1f · 버퍼 있음 %s" % (mm.p_state_cap, hasattr(mm, "p_state_cap_w")))

    print("")
    print("  ⚠ R=3 은 **측정에서 온 값**이다 — 9개 체크포인트의 max(p_states)/초기값 이")
    print("    빔 s1 83.6배 · 충전기 s1 4.31배 · 미니PC s1 2.17배 · 나머지 1.52배 이하.")
    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
