# -*- coding: utf-8 -*-
"""관문 — `--fine-pad replicate` 가 **개입과 같은 물건이 되는가** (14.131).

무엇을 고치려는 것인가
----------------------
14.116 은 진단 **개입**과 학습 **처치**를 같은 것으로 취급했다. **아니었다.**

```
  (가) 600칸 그대로                        미래가 섞인다 (v1)
  (나) 240칸으로 잘라서 — **0 패딩**        `--fine-time-split` 이 한 것
  (다) 600칸이되 미래를 **타깃값 복제**      진단 **개입**이 한 것
```
둘 다 *"미래를 안 본다"* 인데 오른쪽 경계값이 **0** 이냐 **마지막 값**이냐가 다르다.
`asinh` 눈금에서 0 은 *"고조파가 0"* 이라 실측 창에 없는 값이다. 타깃은 조각의 **맨
끝**이라 바로 옆이 전부 그 값이 된다. 실측에서 잰 차이 (cnn_pcap_s0, 타깃 위치):

```
  블록 0 (RF **7**)  (나)0패딩 대 (다)복제  cos **0.825**   <- 이미 갈린다
                     (다)복제 대 (가)600    cos **0.994**
  블록 6 (RF 763)    (나) 대 (다)  cos 0.714  ·  (다) 대 (가)  cos 0.816
```
그리고 결과가 갈렸다 — 개입은 고쳤고(포트 0.003 -> **0.938**), 처치는 ★2 를
**악화**시켰다(+7.10 ± 3.09).

패딩 비중도 크다 — block 6(d=64, pad 192)은 창 600 에서 탭의 **18.3%**, 조각 240 에서
**45.7%** 가 패딩이다 (수용영역 763 이 조각의 **3.18배**라 넘치는 만큼이 전부 패딩이다).

무엇을 확인하나
---------------
```
  [1] `zeros` 가 기본이고 **비트 동일**
  [2] `replicate` 면 달라진다 · 파라미터는 **안 는다**
  [3] ★ **conv 가 개입과 비트 동일** — `replicate` 로 240칸을 태운 블록 0 의 **conv
      출력**(GroupNorm 전)이 600칸에 미래를 복제해 태운 것과 정확히 같아야 한다.
      오른쪽 패딩 3칸이 곧 `x[239]` 세 개이므로 **정의상** 같다.
      ⚠ GroupNorm 뒤로는 **결코 같아질 수 없다** — 통계가 240칸/600칸에서 다르다.
        (처음엔 블록 출력으로 검사했다가 cos 0.985 로 떨어졌다. 패딩이 지배하는
         범위는 **conv 까지**다.)
  [4] 0 패딩에서는 conv 도 **다르다** (음성 대조)
  [5] 경계에서 값이 튀지 않는다 — 출력 맨 끝이 이웃과 이어진다
  [6] v1/v2 · time_split 켬/끔 모두에서 돈다 · autocast · 체크포인트 왕복
```

    python -X utf8 src/run_gate_finepad.py
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
BASE = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4))
T = 239
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


def main() -> int:
    print("`--fine-pad replicate` — 조각내 태울 때 창 밖을 **경계값으로** (14.131)")
    print("  block 6(d=64, pad 192) 은 창 600 에서 탭의 18.3%%, 조각 240 에서 **45.7%%** 가 패딩\n")
    f, w = torch.randn(4, 57, 600), torch.randn(4, 47, 120)

    z, z2 = _net(0), _net(0, fine_pad="zeros")
    with torch.no_grad():
        o0, o1 = z(f, w), z2(f, w)
    ck("[1] `zeros` 가 기본이고 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])))

    r = _net(0, fine_pad="replicate")
    with torch.no_grad():
        o2 = r(f, w)
    ck("[2] `replicate` 면 달라지고 파라미터는 **안 는다**",
       (not torch.equal(o0["on_logit"], o2["on_logit"]))
       and sum(p.numel() for p in z.parameters()) == sum(p.numel() for p in r.parameters()),
       "파라미터 %d == %d · on_logit 최대차 %.4g"
       % (sum(p.numel() for p in z.parameters()), sum(p.numel() for p in r.parameters()),
          float((o0["on_logit"] - o2["on_logit"]).abs().max())))

    # ── [3][4] ★ 조각내 태운 것이 **개입**과 같아지나 ───────────────────────
    frep = f.clone()
    frep[:, :, T + 1:] = f[:, :, T:T + 1]          # (다) 개입 — 미래를 타깃값 복제

    for tag, m, want in (("[3] ★ `replicate`", r, True), ("[4] 음성대조 `zeros`", z, False)):
        conv = m.fine[0][0]                      # 블록 0 의 Conv1d (GroupNorm 전)
        with torch.no_grad():
            cB = conv(f[:, :, :T + 1])[:, :, -1]
            cC = conv(frep)[:, :, T]
            #: GroupNorm 뒤 — 결코 같아질 수 없다. 얼마나 벌어지는지만 찍는다.
            hB, hC, cos = f[:, :, :T + 1], frep, []
            for blk in m.fine:
                hB, hC = blk(hB), blk(hC)
                cos.append(float(torch.nn.functional.cosine_similarity(
                    hB[:, :, -1], hC[:, :, T], dim=1).mean()))
        eq = bool(torch.allclose(cB, cC, atol=1e-6))
        ck("%s — 블록0 **conv 출력**이 개입(600 복제)과 %s"
           % (tag, "**비트 동일**" if want else "다르다"), eq == want,
           "최대차 %.3g · (GroupNorm 뒤 층별 cos %s)"
           % (float((cB - cC).abs().max()), " ".join("%.3f" % c for c in cos)))

    # ── [5] 경계에서 값이 튀지 않나 ────────────────────────────────────────
    with torch.no_grad():
        hz, hr = f[:, :, :T + 1], f[:, :, :T + 1]
        for b0, b1 in zip(z.fine, r.fine):
            hz, hr = b0(hz), b1(hr)
    jz = float((hz[:, :, -1] - hz[:, :, -2]).abs().mean() / hz.std().clamp_min(1e-9))
    jr = float((hr[:, :, -1] - hr[:, :, -2]).abs().mean() / hr.std().clamp_min(1e-9))
    ck("[5] 경계에서 값이 **덜 튄다** (마지막 칸과 그 앞칸의 차)", jr < jz,
       "zeros %.4f -> replicate **%.4f** (%.2f배)" % (jz, jr, jr / max(jz, 1e-9)))

    n = 0
    for lay in ("v1", "v2"):
        for ts in (False, True):
            try:
                mm = _net(0, fine_pad="replicate", head_layout=lay, fine_time_split=ts)
                with torch.no_grad():
                    v = float(mm(f, w)["on_logit"].abs().max())
                n += int(np.isfinite(v))
            except Exception:                            # noqa: BLE001
                pass
    ck("[6a] v1/v2 x time_split 켬/끔 넷 모두 돈다", n == 4, "%d/4" % n)

    err = ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(r(f, w)["on_logit"].float().abs().max())
        g = bool(np.isfinite(v))
    except Exception as e:                               # noqa: BLE001
        g, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[6b] autocast(bfloat16) 에서 돈다", g, err or "OK")

    import tempfile
    from src.run_gate_check import load_model
    p = Path("results/cnn_pcap_s0.pt")
    if not p.exists():
        ck("[6c] 체크포인트 왕복", False, "%s 가 없다" % p)
    else:
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "t.pt"
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["fine_pad"] = "zeros"
            torch.save(c, q)
            mm = load_model(str(q), "cpu")[0]
            ck("[6c] 체크포인트 왕복 — zeros 면 옛 가중치가 그대로 실린다",
               mm.fine_pad == "zeros", "fine_pad=%r" % mm.fine_pad)

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
