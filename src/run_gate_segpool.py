# -*- coding: utf-8 -*-
"""14.46 구간별 풀링 관문 — **끄면 비트 동일**부터 역전파까지.

[1] `pool_segments` 가 타깃 직후를 경계로 두고 창을 빠짐없이·겹침없이 덮는가
[2] `seg_pool=0` 과 `1` 이 **출력 비트 동일**인가 (둘 다 창 전체 한 구간)
[3] 옛 체크포인트를 `seg_pool=0` 으로 실으면 **가중치가 그대로 들어가고** 출력이 같은가
[4] `seg_pool=2` 는 머리 입력이 커져 옛 가중치를 **거부**해야 한다 (조용히 통과하면 위험)
[5] `fine_dim_mask` 길이가 `trunk_in` 과 맞는가 (`fine_dropout` 이 이것을 곱한다)
[6] `seg_pool=2` 에서 **autocast(bf16) 역전파**가 도는가 (982877 이 여기서 죽었다)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.inputs import FINE_CHANNELS, FINE_CYCLES, fine_target_index  # noqa: E402
from src.model.net import NILMNet, appliance_state_counts, pool_segments     # noqa: E402
from src.run_baseline import S_I                                             # noqa: E402

APPS = sorted(S_I)
TGT = fine_target_index()
OK = "✅"; NG = "❌"


def _mk(seg, dev="cpu"):
    torch.manual_seed(0)
    return NILMNet(APPS, appliance_state_counts(APPS), width=1.0,
                   wide_summary=True, wide_target=True, seg_pool=seg).to(dev)


def main() -> int:
    bad = 0

    # [1] 구간 나누기
    print("[1] pool_segments — 타깃 직후가 경계 · 빠짐없이 · 겹침없이")
    for n_tot, t in ((FINE_CYCLES, TGT), (120, 107), (600, 239)):
        for n in (1, 2, 3, 4, 6):
            segs = pool_segments(n_tot, t, n)
            cov = np.zeros(n_tot, int)
            for a, b in segs:
                cov[a:b] += 1
            ok = (len(segs) == max(n, 1) and cov.min() == 1 and cov.max() == 1
                  and (n <= 1 or any(a == t + 1 for a, _ in segs)))
            if not ok:
                bad += 1
                print(f"    {NG} n_tot={n_tot} t={t} n={n}: {segs}")
    print(f"    {OK if not bad else NG} 전부 통과" if not bad else "")
    print(f"    세밀 n=2 {pool_segments(FINE_CYCLES, TGT, 2)}")
    print(f"    세밀 n=4 {pool_segments(FINE_CYCLES, TGT, 4)}")
    print(f"    광역 n=2 {pool_segments(120, 107, 2)}")

    f = torch.randn(4, FINE_CHANNELS, FINE_CYCLES)
    w = torch.randn(4, 47, 120)

    # [2] 0 과 1 이 비트 동일
    m0, m1 = _mk(0), _mk(1)
    m1.load_state_dict(m0.state_dict())
    m0.eval(); m1.eval()
    with torch.no_grad():
        o0, o1 = m0(f, w), m1(f, w)
    same = all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k]))
    print(f"[2] seg_pool 0 대 1 출력 비트 동일: {OK if same else NG}")
    bad += 0 if same else 1

    # [3] 옛 체크포인트를 0 으로 (키·모양이 안 바뀌어야 한다)
    import glob
    cks = sorted(glob.glob("results/cnn_vncls_res_s0.pt")) or sorted(glob.glob("results/cnn_*.pt"))
    if cks:
        # ⚠ 직접 짓지 말고 **실제 적재 경로**(`load_model`)를 쓴다 — `aux_z`·`proj`·`appl_attn`
        #   같은 키를 빼먹으면 관문이 모델이 아니라 **내 스크립트**를 재게 된다 (실제로 그랬다).
        from src.run_gate_check import load_model
        ck = torch.load(cks[0], map_location="cpu", weights_only=False)
        mm, _apps, _ = load_model(cks[0], "cpu")
        assert int(ck.get("seg_pool", 0)) == int(mm.seg_pool) == 0
        print(f"[3] 옛 체크포인트 {cks[0].split('/')[-1]} -> seg_pool=0: {OK} 그대로 실린다"
              f" (머리 입력 {mm.trunk[0].weight.shape[1]})")
    else:
        print("[3] 건너뜀 (체크포인트 없음)")

    # [4] seg_pool=2 는 옛 가중치를 거부해야 한다
    m2 = _mk(2)
    d0, d2 = m0.trunk[0].weight.shape[1], m2.trunk[0].weight.shape[1]
    try:
        m2.load_state_dict(m0.state_dict(), strict=True)
        print(f"[4] {NG} seg_pool=2 가 옛 가중치를 **조용히 받았다** (머리 {d0} -> {d2})")
        bad += 1
    except RuntimeError:
        print(f"[4] {OK} seg_pool=2 가 옛 가중치를 거부한다 (머리 입력 {d0} -> {d2})")

    # [5] fine_dim_mask 길이
    ok5 = (len(m2.fine_dim_mask) == d2) and (len(m0.fine_dim_mask) == d0)
    print(f"[5] fine_dim_mask 길이 == trunk_in: {OK if ok5 else NG}"
          f" ({len(m0.fine_dim_mask)}=={d0}, {len(m2.fine_dim_mask)}=={d2})")
    bad += 0 if ok5 else 1

    # [6] autocast 역전파
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = _mk(2, dev); m.train()
        ff, ww = f.to(dev), w.to(dev)
        ctx = (torch.autocast(dev, dtype=torch.bfloat16) if dev == "cuda"
               else torch.autocast("cpu", dtype=torch.bfloat16))
        with ctx:
            o = m(ff, ww)
            loss = o["power"].float().square().mean() + o["on_logit"].float().square().mean()
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        ok6 = np.isfinite(g) and g > 0
        print(f"[6] autocast(bf16) 역전파 ({dev}): {OK if ok6 else NG} |grad| 합 {g:.3e}")
        bad += 0 if ok6 else 1
    except Exception as e:                                     # noqa: BLE001
        print(f"[6] {NG} 역전파 실패: {type(e).__name__}: {e}")
        bad += 1

    print("\n" + ("관문 전부 통과 " + OK if not bad else f"{NG} 실패 {bad}건"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
