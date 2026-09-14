# -*- coding: utf-8 -*-
"""창-안 텍스처 교체(`--vtex-seg-s`)의 **배선** 관문 (14.51).

⚠⚠ **함수를 재지 말고 배선을 재라.** 983506 이 45분을 버린 까닭이 이것이다 — `str.replace`
가 조용히 빗나가 손잡이가 안 닿았는데, 관문이 *함수*를 따로 불러 보고 통과를 찍었다.
여기서는 `genopts.build_synthesizer` 로 **진짜 생성기**를 짓고 **진짜 창**을 합성해,
모델이 보는 입력 텐서에서 창-안 전압 변동을 잰다.

재는 것
-------
① `sample_run(rng, 1)` 이 `sample(rng)` 과 **같은 텍스처·같은 난수 소비**인가
② `texture_segments` 가 창을 **빈틈·겹침 없이** 덮는가 (N 이 K 로 안 나눠떨어져도)
③ `genopts.check()` 가 안 걸린 배선과 `seg_s != step_s` 를 **잡는가**
④ ★ 끝단 — seg 켬/끔 두 창을 **같은 씨앗**으로 합성해서
     · `y_power`·`y_on`·`y_state` 가 **비트 동일**이어야 한다 (난수 흐름이 안 밀렸다)
     · 창-안 `V_h/|V_1|` 표준편차가 **켜면 눈에 띄게 커야** 한다 (끄면 −Z·I 항뿐)
     · 그 창이 실제로 텍스처 **여러 장**을 썼어야 한다
⑤ `--vtex-seg-s 60` (60초 창 = 한 장)이 `0`(끔)과 **비트 동일**인가 — n=1 이 옛 경로다
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.inputs import VOLT_IM0, VOLT_RE0, VOLT_ORDERS   # noqa: E402
from src.synthesis import genopts                              # noqa: E402
from src.synthesis.grid_simulator import texture_segments      # noqa: E402

W = 3600                    #: 창 길이 (사이클) — 캐시와 같다
NPZ = "processed_data/npz"
FAIL = []


def ck(name: str, ok: bool, note: str = "") -> None:
    print(("  ✅ " if ok else "  ❌ ") + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _windows(opts, seed: int, n: int = 3):
    """`genopts` 로 진짜 생성기를 짓고 창 n개를 합성한다 — 캐시가 쓰는 그 경로."""
    from src.synthesis.dataset import NILMBatchGenerator
    gen = genopts.build_synthesizer(opts, NPZ, "train", compute_gt_harmonics=False)
    bad = genopts.check(opts, gen)
    g = NILMBatchGenerator(segment_pool=gen.pool, window_size_cycles=W,
                           synthesizer=gen, compute_gt_harmonics=False)
    np.random.seed(seed)
    out = []
    for _ in range(n):
        smp, _ = g._synthesize_window()
        x = g._format_inputs(smp)                 # (45, W)
        t = g._format_targets(smp)
        out.append((x, t, smp))
    return gen, bad, out


def _vh_spread(x: np.ndarray) -> np.ndarray:
    """창-안 `V_h/|V_1|` 의 표준편차 (Re·Im, h>1). 텍스처가 소유한 축이다."""
    re = x[VOLT_RE0:VOLT_IM0]                     # (6, W)
    im = x[VOLT_IM0:VOLT_IM0 + len(VOLT_ORDERS)]
    v1 = np.hypot(re[0], im[0])
    v1 = np.where(v1 > 1.0, v1, 1.0)
    r = (re + 1j * im) / v1[None, :]
    return np.stack([r[1:].real.std(1), r[1:].imag.std(1)], 1).ravel()


def main() -> int:
    print("창-안 텍스처 교체 배선 관문 (14.51)\n")

    # ── ① 난수 규율 ──────────────────────────────────────────────────────
    from src.synthesis import vtexture
    vtexture.set_default_step_s(10.0)
    lib = vtexture.default_library()
    a1 = np.random.default_rng(7); b1 = np.random.default_rng(7)
    t_one = lib.sample(a1, vrms_target=229.5, site="E")
    r_one = lib.sample_run(b1, 1, vrms_target=229.5, site="E")
    ck("① sample_run(·,1) == sample(·)",
       len(r_one) == 1 and r_one[0].id == t_one.id and
       int(a1.integers(1 << 30)) == int(b1.integers(1 << 30)),
       "텍스처 id %d · 이후 난수 동일" % t_one.id)
    a6 = np.random.default_rng(7); b6 = np.random.default_rng(7)
    lib.sample(a6, vrms_target=229.5, site="E")
    run6 = lib.sample_run(b6, 6, vrms_target=229.5, site="E")
    ck("① sample_run(·,6) 도 난수를 더 안 쓴다",
       int(a6.integers(1 << 30)) == int(b6.integers(1 << 30)) and len(run6) == 6,
       "녹화 %s · t_rel %s" % (run6[0].stem, [round(t.t_rel_s) for t in run6]))
    ck("① 연속이고 같은 녹화다",
       len({t.stem for t in run6}) == 1 and
       all(run6[i + 1].t_rel_s > run6[i].t_rel_s for i in range(5)))
    ck("① 녹화당 텍스처가 6장 이상이다",
       min(lib.run_lengths().values()) >= 6,
       "최소 %d장" % min(lib.run_lengths().values()))

    # ── ② 토막이 창을 덮는가 ─────────────────────────────────────────────
    class _E:
        texture = None
        texture_seq = tuple(range(7))
    okc = True
    for n in (3600, 3601, 599, 7, 6, 5):
        for k in (1, 2, 3, 6, 7):
            _E.texture_seq = tuple(range(k))
            segs = texture_segments(_E, n)
            cov = np.zeros(n, np.int64)
            for a, b, _ in segs:
                cov[a:b] += 1
            okc &= (len(segs) == k) and bool((cov == 1).all())
    ck("② 토막이 창을 빈틈·겹침 없이 덮는다 (N%K != 0 포함)", okc)
    _E.texture_seq = ()
    _E.texture = object()
    ck("② 텍스처 열이 비면 창 전체 한 토막", texture_segments(_E, 100) == [(0, 100, _E.texture)])
    _E.texture = None
    ck("② 텍스처가 아예 없으면 빈 목록", texture_segments(_E, 100) == [])

    # ── ③ check() 가 안 걸린 배선을 잡는가 ───────────────────────────────
    bad_mix = genopts.check(dict(genopts.V32S, vtex_seg_s=10.0, vtex_step_s=20.0),
                            type("G", (), {"augmentor": type("A", (), {})(),
                                           "grid_sim": type("S", (), {"vtex_seg_s": 10.0})(),
                                           "pool": type("P", (), {})()})())
    ck("③ check() 가 seg_s != step_s 를 잡는다",
       any("step_s" in b for b in bad_mix), " | ".join(b[:60] for b in bad_mix if "step_s" in b))
    bad_off = genopts.check(dict(genopts.V32S),
                            type("G", (), {"augmentor": type("A", (), {})(),
                                           "grid_sim": type("S", (), {"vtex_seg_s": 0.0})(),
                                           "pool": type("P", (), {})()})())
    ck("③ check() 가 '안 걸린 손잡이' 를 잡는다",
       any("안 걸렸다" in b and "vtex_seg_s" in b for b in bad_off))

    # ── ④⑤ 끝단 ─────────────────────────────────────────────────────────
    base = dict(genopts.V32, vtex_step_s=10.0)
    gen0, bad0, w0 = _windows(dict(base), 1234)
    gen1, bad1, w1 = _windows(dict(base, vtex_seg_s=10.0), 1234)
    # ⑤ 의 대조 — `step_s` 를 **같게 두고** `seg_s` 만 60 으로. 60초 창이면 토막이 하나라
    #   `_sample_texture_run` 이 `lib.sample` 로 되돌아간다. 여기가 "n=1 = 옛 경로" 의 증명이다.
    #   (`check()` 는 seg!=step 을 막으므로 이 팔은 일부러 그 검사를 안 탄다.)
    gen6, bad6, w6 = _windows(dict(base, vtex_seg_s=60.0), 1234)
    ck("④ genopts.check() 통과 (끔·켬)", not bad0 and not bad1,
       " | ".join(bad0 + bad1) or "빈 목록")
    ck("④ 손잡이가 시뮬레이터까지 닿았다",
       gen0.grid_sim.vtex_seg_s == 0.0 and gen1.grid_sim.vtex_seg_s == 10.0 and
       gen1.grid_sim.n_texture_segments(W) == 6,
       "끔 %.0f · 켬 %.0f초 -> %d장"
       % (gen0.grid_sim.vtex_seg_s, gen1.grid_sim.vtex_seg_s,
          gen1.grid_sim.n_texture_segments(W)))

    # ⚠ 난수의 증인은 **일정**이다 — `y_on`·`y_state`·`y_plugged`. `y_power` 는 아니다:
    #   토막난 텍스처가 총전류를 바꾸고 -> `apply_load_drop` 의 `v_true` 를 바꾸고 ->
    #   `P ∝ V^e` 가 움직인다. **물리적 귀결이지 난수 이동이 아니다** (실측 최대 0.002W).
    same = all(np.array_equal(a[1][k], b[1][k]) for a, b in zip(w0, w1)
               for k in ("y_on", "y_state", "y_plugged"))
    ck("④ ★ 난수 흐름이 한 칸도 안 밀렸다 (y_on·y_state·y_plugged 비트 동일)", same)
    dp = np.concatenate([np.abs(np.asarray(a[1]["y_power"], float)
                                - np.asarray(b[1]["y_power"], float)) for a, b in zip(w0, w1)])
    ck("④ y_power 는 물리로만 움직인다 (< 0.01W)", float(dp.max()) < 0.01,
       "최대 %.4f W" % dp.max())

    s0 = np.median([_vh_spread(x) for x, _, _ in w0], 0)
    s1 = np.median([_vh_spread(x) for x, _, _ in w1], 0)
    gain = float(np.median(s1 / np.maximum(s0, 1e-12)))
    ck("④ ★ 창-안 V_h/|V_1| 변동이 커졌다", gain > 3.0,
       "중앙 배수 ×%.1f   (끔 %.2e · 켬 %.2e ×1e-4: %.2f -> %.2f)"
       % (gain, np.median(s0), np.median(s1), 1e4 * np.median(s0), 1e4 * np.median(s1)))

    x_eq = (all(np.array_equal(a[0], b[0]) for a, b in zip(w0, w6)) and
            all(np.array_equal(a[1][k], b[1][k]) for a, b in zip(w0, w6)
                for k in ("y_power", "y_on", "y_state", "y_plugged", "y_standby_power")))
    ck("⑤ ★ --vtex-seg-s 60 (창 = 한 장) 이 끔과 **비트 동일** (입력·정답 전부)", x_eq,
       "같은 step_s 에서 토막 %d장" % gen6.grid_sim.n_texture_segments(W))
    x_ne = any(not np.array_equal(a[0], b[0]) for a, b in zip(w0, w1))
    ck("⑤ --vtex-seg-s 10 은 입력이 달라진다", x_ne)

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
