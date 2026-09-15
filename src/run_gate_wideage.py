# -*- coding: utf-8 -*-
"""광역 갈래가 **"계단이 몇 초 전이었나"** 를 담을 수 있나 — 학습 없이 (14.87).

왜 만드나
---------
14.86 이 잰 것: 오븐 통전->팬·조명 계단이 **세밀 창(과거 3.98초)을 떠나는 그 창부터**
세 모델 전부 무너진다 (t=128.0초, 계단은 123.0초). 광역은 `[t−54초, t+6초]` 라 그 계단을
**177초까지** 갖고 있는데, 지워 보니 오븐 통전 확률이 **0.03~0.16** 밖에 안 움직였다 —
세밀 창 안에 있을 때의 **0.805** 와 비교하면 5~25배 약하다.

머리가 광역에서 받는 것은 `hw[:, :, a:b].mean(-1)` — **시간 평균**이다. 그래서 갈림길이 둘:

```
  ⓐ 정보가 담기는데 평균이 죽인다  -> `seg_pool` 을 키우면 된다. **N 도 여기서 정한다.**
  ⓑ 애초에 안 담긴다              -> 머리를 고쳐도 소용없다. 세밀 창의 과거를 늘려야 한다.
```

어떻게 재나 — **복호기가 아니라 개입**
--------------------------------------
계단이 앉은 블록을 **왼쪽으로 민다** (= 더 오래된 것처럼). 나머지는 그대로 두고 꼬리만
가장자리 복제. 그리고 **학습된 진짜 광역 몸통** `model.wide` 를 그대로 돌려, 머리가
실제로 뽑는 방식마다 특징 벡터가 **얼마나 움직이는지** 잰다.

⚠⚠ **처음엔 선형 복호기로 "나이를 맞혀 봐라" 를 했는데 틀린 도구였다.** 양성 대조(원시
입력)가 R² **−0.147 -> 0.000** 으로 두 번 다 실패했고, 그제서야 까닭을 알았다 — 계단의
크기와 부호가 구간마다 제각각(−2W ~ −1436W)이라 **전역 선형 방향 하나로는 원리적으로
못 읽는다.** 정칙화 눈금도 차원마다 달라 고차원 판이 음수 R² 를 냈다. 개입은 그 둘을 다
비켜간다 — 한 창 안에서 **그것 하나만** 바꾸기 때문이다.
([[check-conditioning-before-believing-a-fit]], [[the-estimator-can-flip-the-conclusion]])

⚠ `wtgt` 은 이 개입에서 **과하게 나온다** — 왼쪽으로 밀면 타깃 열에 들어오는 내용 자체가
바뀐다. 계단 위치만의 효과가 아니다. `seg*` 과 `mean` 은 그 편향이 없다.

    python -X utf8 src/run_gate_wideage.py --ckpt results/cnn_wtap_s1.pt
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_train_seq import real_windows  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402
from src.model.net import pool_segments, wide_target_index  # noqa: E402

SPAN, LOOK = 54.0, 6.0          # 광역 창 = [t−54초, t+6초]
KEYS = ("mean", "wtgt", "seg2", "seg4", "seg6", "seg8", "full")


def head_feats(hw, wt, L):
    """머리가 실제로 뽑는 방식들. `hw` 는 (C, L)."""
    f = {"mean": hw.mean(-1), "wtgt": hw[:, wt], "full": hw.reshape(-1)}
    for n in (2, 4, 6, 8):
        f["seg%d" % n] = np.concatenate(
            [hw[:, a:b].mean(-1) for a, b in pool_segments(L, wt, n)])
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_wtap_s1.pt"])
    ap.add_argument("--stem", default="test_5")
    ap.add_argument("--at", type=float, default=128.0, help="이 창에서 개입한다 (초)")
    ap.add_argument("--step", type=float, default=123.0, help="옮길 계단의 시각 (초)")
    ap.add_argument("--app", default="oven")
    ap.add_argument("--state", type=int, default=2, help="볼 상태 번호 (오븐 통전 = 2)")
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    j = apps.index(a.app)
    cache = real_windows(apps, a.grid_s, dev)
    d = cache[a.stem]
    t = d["t"]
    i0 = int(np.argmin(np.abs(t - a.at)))
    L = d["wide"].shape[-1]
    bl = (SPAN + LOOK) / L
    p0 = int(round((a.step - (t[i0] - SPAN)) / bl))
    wt = wide_target_index(L)
    x0 = d["wide"][i0]

    def shift_to(q):
        """계단을 블록 `q` 로 옮긴 입력 — `p0−q` 만큼 왼쪽으로 밀고 꼬리는 가장자리 복제."""
        k = p0 - q
        if k <= 0:
            return x0.copy()
        y = np.empty_like(x0)
        y[:, :L - k] = x0[:, k:]
        y[:, L - k:] = x0[:, -1:]
        return y

    print("%s · t=%.1f초 창 · 광역 %d블록 x %.2f초 · %.1f초 계단은 블록 **%d** (%.1f초 전)"
          % (a.stem, t[i0], L, bl, a.step, p0, t[i0] - a.step))
    print("계단만 왼쪽으로 민다 (= 더 오래된 것처럼). 나머지 그대로, 꼬리는 가장자리 복제.")
    qs = [p0 - k for k in (0, 8, 16, 24, 32)]

    for path in a.ckpt:
        model, apps_m = load_model(path, dev)[:2]
        model.eval()
        assert list(apps_m) == apps
        print("")
        print("=" * 96)
        print("%s   (%s 상태 s%d)" % (path.split("/")[-1], a.app, a.state))
        print("=" * 96)
        print("  %-10s %7s %9s | %s" % ("계단 나이", "블록", "s%d" % a.state,
                                        "머리 특징의 **상대 이동** |Δf| / |f_기준|"))
        print("  %-10s %7s %9s | %s" % ("", "", "", " ".join("%8s" % k for k in KEYS)))
        ref = None
        with torch.no_grad():
            for q in qs:
                xq = torch.from_numpy(shift_to(q)[None]).to(dev)
                hw = model.wide(xq)[0].float().cpu().numpy()
                o = model(torch.from_numpy(d["fine"][i0:i0 + 1]).to(dev), xq)
                s = o["state"][0, j].softmax(-1)[a.state].item()
                f = head_feats(hw, wt, L)
                if ref is None:
                    ref = f
                rel = [np.linalg.norm(f[k] - ref[k]) / (np.linalg.norm(ref[k]) + 1e-9)
                       for k in KEYS]
                print("  %-10.1f %7d %9.3f | %s"
                      % (t[i0] - (a.step - (p0 - q) * bl), q, s,
                         " ".join("%8.4f" % v for v in rel)))
    print("")
    print("  읽는 법")
    print("   · `full` 이 크게 움직여야 개입이 먹힌 것이다 (**양성 대조**).")
    print("   · 그 상태에서 `mean` 이 작으면 시간 평균이 위치를 **거의** 지운다는 뜻이고,")
    print("     `seg*` 가 N 에 따라 커지면 **쪼개면 담긴다** — 그 표에서 N 을 고른다.")
    print("   · 마지막 칸은 **이 모델이 실제로 그걸 쓰는가**다. 특징이 크게 움직였는데")
    print("     상태가 안 움직이면, 통로는 있는데 **모델이 광역을 안 믿는** 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
