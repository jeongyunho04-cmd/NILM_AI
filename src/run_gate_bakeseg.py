# -*- coding: utf-8 -*-
"""**굽기가 실제로 타는 경로**에서 비균일 토막이 서는가 (14.103).

`run_gate_texoffset` 는 `_texture_plan` 을 **따로 불러** 잰다. 그건 함수 검사고,
굽기가 `traincache._init` 로 지은 생성기가 정말 그 계획을 쓰는지는 안 본다
([[the-gate-must-build-the-real-object]] — 오늘 이미 한 번 당했다: `_init` 서명이
밀려서 `harmonic_z` 가 안 걸렸는데 함수 관문은 통과했다).

여기서는 **굽기와 같은 깃발**로 `_init` 을 부르고, 그 생성기가 내놓는 `환경`에서
```
  ① 창 하나의 토막 수 — 균일 20 이 아니라 **12 근처**
  ② 세밀 창(뒤 600사이클) 안의 토막 수 — 균일 3 이 아니라 **9 근처**
  ③ 경계 자리가 창마다 흩어진다 (외울 수 있는 고정 자리가 아니다)
  ④ 세밀 창 **밖**은 성기다 (토막 길이 중앙이 25초 근처)
  ⑤ `--vtex-coarse-s 0` 으로 지으면 **옛 균일 경로**로 돌아간다 (음성 대조)
```
회로를 안 푼다 — `sample_environment` 만 부른다. 로그인 노드에서 몇 초다.

    python -X utf8 src/run_gate_bakeseg.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

N, FS = 3600, 60.0
FINE0 = N - 600
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _gen(coarse):
    """**굽기와 같은 깃발**로 `_init` 을 부른다 (`cache_v32hs3_par.sbatch` 의 COMMON+TREAT)."""
    from src.model import traincache as tc
    from src.run_recipe_mix_probe import PRESETS as MIX_PRESETS
    from src.synthesis.augmentor import (FLOAT_FILL_PRESETS, POWER_SCALE_STD_PRESETS,
                                         STATE_MIX_PRESETS, STEADY_CROP_PRESETS)
    tc._init("processed_data/npz", N, "train", 0,
             recipe_mix_json=json.dumps(MIX_PRESETS["steady2"]),
             power_scale_std_json=json.dumps(POWER_SCALE_STD_PRESETS["measured"]),
             sp_curves=True, sp_per_texture=True, vtail=True,
             state_mix_json=json.dumps(STATE_MIX_PRESETS["minipc_balanced"]),
             carrier_apps=("oven",), couple_ext=True, smps_focus_off_p=0.4,
             float_fill_json=json.dumps(FLOAT_FILL_PRESETS["charger_float"]),
             steady_crop_json=json.dumps(STEADY_CROP_PRESETS["smps_steady"]),
             standby_jitter_cap=95.0, harmonic_z=True,
             vtex_step_s=1.25, vtex_seg_s=1.25, vtex_coarse_s=float(coarse))
    return tc._GEN


def _segs(gen, k=200):
    """창 k개의 토막 경계 — `sample_environment` 가 굽기에 주는 그대로."""
    from src.synthesis.grid_simulator import texture_segments
    gs = getattr(gen, "synthesizer", gen).grid_sim
    out = []
    for i in range(k):
        np.random.seed(1000 + i)
        env = gs.sample_environment(N)
        out.append(texture_segments(env, N))
    return out


def main() -> int:
    print("굽기 경로의 **비균일 토막** 관문 (14.103)")
    print("  창 %d사이클(60초) · 세밀 창 = [%d, %d) · 굽기 깃발 그대로\n" % (N, FINE0, N))

    segs = _segs(_gen(25.0))
    ns = np.array([len(s) for s in segs])
    fine_n = np.array([sum(1 for a, b, _ in s if b > FINE0) for s in segs])
    edges = np.array([a for s in segs for a, _b, _t in s if 0 < a < N])
    fine_e = np.array([a for s in segs for a, _b, _t in s if FINE0 < a < N])
    coarse_len = np.array([b - a for s in segs for a, b, _ in s if b <= FINE0])
    fine_len = np.array([b - a for s in segs for a, b, _ in s if a >= FINE0])

    ck("① 창당 토막 중앙 %d (균일 3초면 20 · 기대 12 근처)" % int(np.median(ns)),
       8 <= np.median(ns) <= 16, "범위 %d~%d · 비용비 20/%.1f = %.2f배"
       % (ns.min(), ns.max(), ns.mean(), 20.0 / ns.mean()))
    ck("② ★ 세밀 창 안 토막 중앙 %d (균일이면 3 · 기대 9)" % int(np.median(fine_n)),
       np.median(fine_n) >= 7,
       "세밀 해상도 %.2f초 (균일 3.33초)" % (10.0 / max(fine_n.mean(), 1)))
    ck("③ 경계가 창마다 흩어진다 — 서로 다른 자리 %d개 (세밀 창 안 %d개)"
       % (len(np.unique(edges)), len(np.unique(fine_e))),
       len(np.unique(fine_e)) >= 100)
    if len(coarse_len) and len(fine_len):
        ck("④ 세밀 밖은 성기고 안은 촘촘하다 — 바깥 중앙 %.1f초 · 안 중앙 %.2f초"
           % (np.median(coarse_len) / FS, np.median(fine_len) / FS),
           np.median(coarse_len) / FS > 8.0 and np.median(fine_len) / FS < 3.0)
    else:
        ck("④ 세밀 안팎이 갈린다", False, "한쪽이 비었다")

    s0 = _segs(_gen(0.0), k=40)
    n0 = np.array([len(s) for s in s0])
    want = [(i * N // 48, (i + 1) * N // 48) for i in range(48)]   # step=seg=1.25s -> 48토막
    uni = all([(a, b) for a, b, _ in s] == want for s in s0[:3]) or bool((n0 == n0[0]).all())
    ck("⑤ 음성 대조 — `--vtex-coarse-s 0` 이면 **균일**로 돌아간다 (토막 %d 고정)" % n0[0],
       uni and n0[0] != int(np.median(ns)),
       "비균일 %d 대 균일 %d" % (int(np.median(ns)), n0[0]))

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
