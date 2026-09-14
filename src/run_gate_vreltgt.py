# -*- coding: utf-8 -*-
"""`--vrel-target` 의 **배선** 관문 (14.53).

⚠⚠ **함수를 재지 말고 배선을 재라** (983506 이 45분을 버렸다). 여기서 제일 중요한 줄은
③ 이다 — 1단계가 세밀 채널에서 되살린 타깃 전압이 2단계가 쓰는
`RealWindows.v_observed`(= `power_features[targets, 4]`, `realdata.py:199`)와 **같은 값**인가.
그 둘이 어긋나 있었던 것이 이 고침의 이유이고, 색인이 한 칸만 밀려도 여기서 잡힌다.

```
① 끄면 옛 식과 **비트 동일** (10초 평균)
② 켜면 `fine[:, 25, 239]` 그대로 (부동소수 오차도 없이)
③ ★ **두 입구 맞대기** — 실측 창에서 ② 가 `RealWindows.v_observed` 와 같다
④ `vrel == v_rms / V_CENTER` (한쪽만 옮기면 그 자체가 새 불일치다)
⑤ `run_train_cnn` 이 학습 고리에서 실제로 넘긴다
⑥ 차이의 크기가 `run_diag_vrel` 이 잰 것과 맞는다 (고전력 창에서만 뜬다)
```
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import V_CENTER, V_SPAN, fine_target_index  # noqa: E402
from src.model.net import V_CH_FINE  # noqa: E402
from src.run_train_cnn import FINE_TPOS, to_targets  # noqa: E402

OK, NG = "✅", "❌"
FAIL = []


def ck(name, ok, note=""):
    print(("  " + (OK if ok else NG) + " ") + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _fake_batch(n=8, w=600, k=9, seed=0):
    """학습 캐시 배치와 **같은 자리수**의 가짜 배치 (11개)."""
    g = torch.Generator().manual_seed(seed)
    fine = torch.randn(n, 57, w, generator=g)
    # 채널 25 를 실제 전압처럼: 200~230V 가 천천히 움직이고 통전 구간만 뚝 떨어진다
    v = 216.0 + torch.randn(n, w, generator=g) * 0.3
    v[:, 200:300] -= 7.0                       # '통전' 구간
    fine[:, V_CH_FINE] = (v - V_CENTER) / V_SPAN
    return (fine, torch.randn(n, 47, 120, generator=g),
            torch.rand(n, k), torch.zeros(n, k, dtype=torch.int8),
            torch.zeros(n, k, dtype=torch.int8), torch.zeros(n, k),
            torch.zeros(n, k, dtype=torch.int8), torch.randn(n, 15, 2),
            torch.rand(n), torch.rand(n), torch.rand(n, 2) + 1.0), fine, v


def main() -> int:
    print("`--vrel-target` 배선 관문 (14.53)\n")
    tp = fine_target_index()
    ck("색인이 맞다 (FINE_TPOS == fine_target_index())", FINE_TPOS == tp, "= %d" % tp)

    b, fine, v = _fake_batch()
    _, _, t0 = to_targets(b, "cpu", vrel_target=False)
    _, _, t1 = to_targets(b, "cpu", vrel_target=True)

    old = fine[:, V_CH_FINE].mean(-1) * V_SPAN + V_CENTER
    ck("① 끄면 옛 식과 **비트 동일** (10초 평균)",
       bool(torch.equal(t0["v_rms"], old)) and
       bool(torch.equal(t0["vrel"], old / V_CENTER)))
    tgt = fine[:, V_CH_FINE, tp] * V_SPAN + V_CENTER
    ck("② 켜면 `fine[:, 25, %d]` 그대로" % tp,
       bool(torch.equal(t1["v_rms"], tgt)),
       "평균 %.2fV -> 타깃 %.2fV" % (float(old.mean()), float(tgt.mean())))
    ck("④ `vrel == v_rms / V_CENTER` (양쪽 다)",
       bool(torch.equal(t0["vrel"], t0["v_rms"] / V_CENTER)) and
       bool(torch.equal(t1["vrel"], t1["v_rms"] / V_CENTER)))

    # ③ ★ 두 입구 맞대기 — 실측 창에서
    try:
        from src.model.realdata import RealWindows
        rw = RealWindows(stems=["test_1"], stride=600, require_valid=False)
        idx = np.arange(0, min(400, len(rw)))
        f, *_ = rw.batch(idx)
        got = np.asarray(f)[:, V_CH_FINE, tp] * V_SPAN + V_CENTER
        want = np.asarray(rw.voltage(idx), dtype=np.float64)
        d = np.abs(got - want)
        ck("③ ★ **두 입구 맞대기** — 1단계 세밀채널@타깃 == 2단계 `v_observed`",
           float(d.max()) < 0.05,
           "창 %d개 · 최대 차 %.4fV · 중앙 %.4fV" % (len(idx), d.max(), np.median(d)))
        # 음성 대조 — 관문이 관문 노릇을 하는지. ⚠ **한 칸으로는 안 된다**:
        #   계측기가 전압을 **0.5초(30사이클)마다** 갱신해서(`measurement_frame_cycles`)
        #   238 과 239 가 같은 블록이다. 블록을 넘겨야 값이 바뀐다.
        one = np.asarray(f)[:, V_CH_FINE, tp - 1] * V_SPAN + V_CENTER
        blk = np.asarray(f)[:, V_CH_FINE, tp - 60] * V_SPAN + V_CENTER
        ck("③ 전압은 0.5초 블록으로 잡혀 있다 (한 칸 밀어도 같다)",
           float(np.abs(one - want).max()) < 1e-6,
           "한 칸 밀린 최대 차 %.6fV — 그래서 ±1 은 음성 대조가 못 된다" % np.abs(one - want).max())
        ck("③ 색인을 **블록 하나(1초)** 밀면 그 검사가 실제로 깨진다",
           float(np.abs(blk - want).max()) > 0.05,
           "60사이클 밀린 최대 차 %.4fV" % np.abs(blk - want).max())
        # ⑥ 크기 — 고전력 창에서만
        pobs = np.asarray(rw.p_observed[idx], dtype=np.float64)
        mean_v = np.asarray(f)[:, V_CH_FINE].mean(-1) * V_SPAN + V_CENTER
        for lbl, m in (("전체", np.ones(len(idx), bool)), ("P>1500", pobs > 1500)):
            if int(m.sum()) < 5:
                continue
            dp = 100.0 * np.median(mean_v[m] - want[m]) / np.median(want[m])
            print("     ⑥ %-7s 창 %4d · 10초평균 − 타깃 = %+.2f%%" % (lbl, int(m.sum()), dp))
    except Exception as e:                                  # noqa: BLE001
        ck("③ ★ 두 입구 맞대기", False, "실측을 못 읽었다: %s" % e)

    # ⑤ 학습기 배선
    src = open("src/run_train_cnn.py", encoding="utf-8").read()
    ck("⑤ `run_train_cnn` 이 학습 고리에서 실제로 넘긴다",
       "to_targets(batch, dev, vrel_target=a.vrel_target)" in src)

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
