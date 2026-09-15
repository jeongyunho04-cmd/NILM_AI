# -*- coding: utf-8 -*-
"""구운 캐시가 **정말로** 처치대로 구워졌나 (14.109).

사용자: *"20분은 너무 빠른 것 같다. 회로 모델이 정상적으로 호출되어 작동했는지,
seg_s 가 제대로 적용되었는지 다시 검사해 달라."*

앞서 한 검사(`train60_v32h` 와 `fine` 이 다르다)는 **이 의심을 못 지운다** — 회로가
통째로 죽어 `texture_delta` 가 None 을 돌려줬어도, 텍스처 **표집 간격**이 60초에서
1.25초로 바뀌었으니 뽑히는 텍스처가 달라져 `fine` 은 어차피 달라진다.

그래서 셋을 따로 잰다.

```
  [A] **창 안 텍스처 계단** — 구운 자료만 읽는다 (합성 안 함)
      세밀 45~56 채널이 곧 `V_h/V_1` 다. 토막이 걸렸으면 600사이클 안에서
      **계단이 8개쯤** 보이고, 안 걸렸으면 **한 덩이**다. v32h(끔)와 나란히 본다
  [B] **회로가 실제로 불렸나** — `SmpsCircuit.current` 를 세면서 창 몇 개를 합성한다
      0번이면 회로가 죽은 채로 구워진 것이다
  [C] **창당 비용** — 같은 깃발로 한 창을 실제로 재서, 22분/토막이 말이 되는지 본다
```

    python -X utf8 src/run_diag_bakecheck.py --cache cache/train60_v32hs3 \
        --ref cache/train60_v32h --runs 8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

FINE_VOLT0, N_VOLT = 45, 12      #: 세밀 45~56 = V_h/V_1 (Re 6 + Im 6), 13.26
FS = 60.0


def jumps_in_window(f: np.ndarray, k: float = 8.0, m: int = 6):
    """(57, 600) 한 창에서 **12채널이 동시에 뛴** 자리 수 (= 토막 경계).

    ⚠ 처음엔 "값이 바뀌는 사이클" 을 셌는데 두 캐시 다 599/599 가 나왔다 — 전압
      채널에는 **사이클마다 잡음·지터**가 얹혀 있어 늘 바뀐다. 토막을 보려면
      *바뀌는가* 가 아니라 **얼마나 크게 바뀌는가**를 봐야 한다.
      토막 경계의 도약은 텍스처가 통째로 갈리는 것이라 잡음보다 훨씬 크다.

    ⚠⚠ 그리고 **채널을 통틀어 OR 하면 안 된다** — 그것도 두 캐시가 272 대 271 로
      똑같았다. 잡음은 채널마다 **따로** 뛰는데 토막 경계는 12채널이 **동시에** 뛴다.
      그래서 `동시에 m개 이상` 을 센다. 이것이 토막을 잡는 자다.
    """
    v = np.asarray(f[FINE_VOLT0:FINE_VOLT0 + N_VOLT], np.float64)     # (12, 600)
    d = np.abs(np.diff(v, axis=1))                                    # (12, 599)
    med = np.median(d, axis=1, keepdims=True)
    hit = d > np.maximum(k * med, 1e-9)
    co = hit.sum(0)                                                   # (599,) 동시 채널 수
    big = float(d.max()) / max(float(np.median(med)), 1e-12)
    return int((co >= m).sum()), big


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train60_v32hs3")
    ap.add_argument("--ref", default="cache/train60_v32h", help="seg 끔 대조")
    ap.add_argument("--windows", type=int, default=300, help="[A] 에서 읽을 창 수")
    ap.add_argument("--runs", type=int, default=8, help="[B][C] 에서 합성할 창 수")
    ap.add_argument("--skip-synth", action="store_true", help="[B][C] 를 건너뛴다")
    ap.add_argument("--shards", type=int, default=8, help="병합 전 토막 수 ([C] 산수용)")
    ap.add_argument("--workers", type=int, default=40, help="굽기에 쓴 워커 수 ([C] 산수용)")
    a = ap.parse_args()

    # ── [A] 구운 자료의 창-안 계단 ───────────────────────────────────────
    print("[A] 창 안 **전압 고조파 계단** (세밀 45~56, 600사이클=10초)")
    print("    %-24s %8s %8s %8s   %s" % ("캐시", "계단중앙", "p10", "p90", "뜻"))
    got = {}
    for tag, path in (("처치", a.cache), ("대조(seg 끔)", a.ref)):
        p = Path(path) / "fine.npy"
        if not p.exists():
            print("    %-24s (없다: %s)" % (tag, p))
            continue
        arr = np.load(p, mmap_mode="r")
        n = min(a.windows, len(arr))
        r = [jumps_in_window(arr[i]) for i in range(0, n)]
        s = np.array([x[0] for x in r]); bg = np.array([x[1] for x in r])
        got[tag] = s
        print("    %-24s %8d %8d %8d   최대도약/잡음 중앙 **%.0f배**"
              % (path, int(np.median(s)), int(np.percentile(s, 10)),
                 int(np.percentile(s, 90)), float(np.median(bg))))
    if "처치" in got and "대조(seg 끔)" in got:
        t, c = np.median(got["처치"]), np.median(got["대조(seg 끔)"])
        ok = t >= 4 and t > 2 * max(c, 0.5)
        print("    => %s   (처치 %d · 대조 %d · 기대: 처치 8 근처, 대조 0)"
              % ("**세밀 창 안에 토막이 실재한다** OK" if ok else "** 기대와 다르다 **",
                 int(t), int(c)))

    if a.skip_synth:
        return 0

    # ── [B][C] 회로 호출 수와 창당 비용 ─────────────────────────────────
    print("")
    print("[B][C] 같은 깃발로 창 %d개를 **실제로 합성**한다" % a.runs)
    import json

    from src.model import traincache as tc
    from src.run_recipe_mix_probe import PRESETS as MIX
    from src.synthesis import coupling as C
    from src.synthesis.augmentor import (FLOAT_FILL_PRESETS, POWER_SCALE_STD_PRESETS,
                                         STATE_MIX_PRESETS, STEADY_CROP_PRESETS)

    cnt = {"current": 0, "simulate_true": 0}
    for nm in list(cnt):
        if not hasattr(C.SmpsCircuit, nm):
            cnt.pop(nm)
            continue
        orig = getattr(C.SmpsCircuit, nm)

        def mk(orig=orig, nm=nm):
            def f(self, *ar, **kw):
                cnt[nm] += 1
                return orig(self, *ar, **kw)
            return f
        setattr(C.SmpsCircuit, nm, mk())

    meta = json.load(open(Path(a.cache) / "meta.json", encoding="utf-8"))
    tc._init("processed_data/npz", 3600, "train", 0,
             recipe_mix_json=json.dumps(MIX["steady2"]),
             power_scale_std_json=json.dumps(POWER_SCALE_STD_PRESETS["measured"]),
             sp_curves=True, sp_per_texture=True, vtail=True,
             state_mix_json=json.dumps(STATE_MIX_PRESETS["minipc_balanced"]),
             carrier_apps=("oven",), couple_ext=True, smps_focus_off_p=0.4,
             float_fill_json=json.dumps(FLOAT_FILL_PRESETS["charger_float"]),
             steady_crop_json=json.dumps(STEADY_CROP_PRESETS["smps_steady"]),
             standby_jitter_cap=95.0,
             harmonic_z=bool(meta.get("harmonic_z")),
             vtex_step_s=float(meta.get("vtex_step_s") or 0.0),
             vtex_seg_s=float(meta.get("vtex_seg_s") or 0.0),
             vtex_coarse_s=float(meta.get("vtex_coarse_s") or 0.0))
    g = tc._GEN
    circ = getattr(g, "synthesizer", g).grid_sim.circuit
    have = [d for d in C.SMPS_DEVICES if circ.has(d)]
    print("    회로 모델이 있는 기기: %s" % (have or "**없다 — 회로가 죽었다**"))

    for k in cnt:
        cnt[k] = 0
    t0 = time.time()
    for i in range(a.runs):
        np.random.seed(10_000 + i)
        getattr(g, "synthesizer", g)      # 접근만 (초기화 고정)
        g._synthesize_window()
    dt = time.time() - t0
    per = dt / max(a.runs, 1)

    print("    [B] 회로 호출 — %s" % " · ".join("%s **%d번** (창당 %.1f)"
                                               % (k, v, v / max(a.runs, 1))
                                               for k, v in cnt.items()))
    print("        => %s" % ("회로가 실제로 돈다 OK" if sum(cnt.values()) > 0
                             else "** 한 번도 안 불렸다 — 회로 없이 구워졌다 **"))
    bs = float(meta.get("build_seconds") or 0)
    nw = int(meta.get("n_windows") or 0)
    #: ⚠ 병합된 meta 의 `build_seconds` 는 **토막 하나**의 시간인데 `n_windows` 는
    #:   **전체**다. 나눠 쓰면 8배 싸 보인다 — `--shards` 로 토막 수를 준다.
    nw_shard = nw / max(a.shards, 1)
    print("    [C] 창당 **%.3f초** (1코어) · 합성 %d창 %.1f초" % (per, a.runs, dt))
    if bs and nw_shard:
        core_s = bs * float(a.workers) / nw_shard
        print("        구운 기록: 토막당 %d창 / %.0f초 x %d워커 -> 창당 **%.3f 코어·초**"
              % (nw_shard, bs, a.workers, core_s))
        print("        지금 잰 값 %.3f초 대비 **%.2f배** (1.0 근처면 앞뒤가 맞는다)"
              % (per, core_s / max(per, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
