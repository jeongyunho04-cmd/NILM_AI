# -*- coding: utf-8 -*-
"""관문 — **홀드아웃**에도 비균일 토막이 걸리는가 (14.103 / 14.108).

왜 따로 필요한가
----------------
`--vtex-coarse-s` 를 만들 때 **학습 캐시 경로만** 배선했다. 홀드아웃은
`run_build_holdout` -> `evaluation.holdout.build_holdout` -> `_build_generator` 라는
**다른 입구**라, 학습 쪽 관문이 전부 통과해도 홀드아웃은 조용히 **균일**로 구워진다.
그러면 학습 분포와 평가 분포가 갈리고, 그 차이를 모델 성능으로 읽는다.

오늘 같은 꼴을 세 번 만났다 — 14.62 가 `traincache._init` 에만 표가 안 걸린 것을
잡았고, 오늘 아침 `_init` 서명이 밀려 `harmonic_z` 가 안 걸렸고, 이것이 셋째다.
[[pin-the-two-entry-points-against-each-other]] · [[verify-the-input-path-not-just-the-model]]

무엇을 확인하나 — **진짜로 굽는다** (창 40개, 몇 초)
```
  [1] `--vtex-coarse-s 0` 이면 그 인자가 **없을 때와 바이트 동일** (옛 경로 항등)
  [2] 켜면 자료가 **달라진다** (안 달라지면 손잡이가 아무것도 안 한다)
  [3] 균일(`seg=1.25, coarse=0`)과도 **달라진다** — 이것이 '비균일' 의 내용이다
  [4] `meta` 에 `vtex_coarse_s` 가 적힌다 (학습 캐시와 짝을 맞출 근거)
  [5] 생성기가 실제로 **창당 12토막 근처**를 낸다 (균일 1.25초면 48토막이다)
  [6] 음성 대조 — 시뮬레이터에 안 걸린 척하면 `build_holdout` 이 **죽는다**
```

    python -X utf8 src/run_gate_hocoarse.py
"""
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

N = 40          #: 관문용 창 수. 청크 20 x 2 이면 병렬 경로도 탄다
ARRS = ("fine", "wide", "y_power", "y_on", "y_state", "obs_harm")
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


#: 굽기 인자 — `patches/holdout_v32h*.sbatch` 와 **같은 꼴**이다.
COMMON = ["--windows", str(N), "--window-cycles", "3600", "--holdout-frac", "0.2",
          "--seed", "20260821", "--workers", "2", "--chunk-windows", "20",
          "--recipe-mix", "steady2", "--smps-focus-off-p", "0.4",
          "--state-mix", "minipc_balanced", "--carrier-on", "oven", "--couple-ext",
          "--sp-curves", "--sp-per-texture", "--vtail",
          "--float-fill", "charger_float", "--steady-crop", "smps_steady",
          "--standby-jitter-cap", "95", "--harmonic-z",
          "--vtex-step-s", "1.25", "--vtex-seg-s", "1.25"]


def _bake(out: Path, extra=(), expect_fail: bool = False, env_extra=None):
    """**진짜 CLI 로** 하위 프로세스에서 굽는다.

    ⚠ 한 프로세스에서 두 번 못 굽는다 — `set_default_step_s` 가 모듈 전역 텍스처
      라이브러리를 늘리는데, 두 번째 호출에서 *"간격을 줄였는데 텍스처가 안 늘었다"*
      로 죽는다. 그리고 진짜 굽기도 프로세스마다 한 번이므로 이쪽이 실제 경로다.
    """
    import json
    import os
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.run_build_holdout",
                        "--out", str(out), *COMMON, *extra],
                       capture_output=True, text=True, errors="replace", env=env)
    if expect_fail:
        return r.returncode != 0, (r.stdout + r.stderr)[-400:]
    if r.returncode != 0:
        print(r.stdout[-1200:])
        print(r.stderr[-1200:])
        raise SystemExit("굽기 실패: %s" % (extra,))
    return json.load(open(out / "meta.json", encoding="utf-8"))


def _same(a: Path, b: Path):
    bad = []
    for nm in ARRS:
        pa, pb = a / ("%s.npy" % nm), b / ("%s.npy" % nm)
        if not (pa.exists() and pb.exists()):
            continue
        x, y = np.load(pa), np.load(pb)
        if x.shape != y.shape or not np.array_equal(x, y):
            bad.append(nm)
    return bad


def main() -> int:
    print("홀드아웃 **비균일 토막** 관문 (14.108) · 창 %d개\n" % N)
    tmp = Path(tempfile.mkdtemp(prefix="gate_hocoarse_"))
    try:
        m_none = _bake(tmp / "none")                              # 인자 자체를 안 줌
        m_zero = _bake(tmp / "zero", ["--vtex-coarse-s", "0"])
        m_trt = _bake(tmp / "trt", ["--vtex-coarse-s", "25"])

        bad = _same(tmp / "none", tmp / "zero")
        ck("[1] `vtex_coarse_s=0` 이 **인자 없음과 바이트 동일**", not bad,
           "다른 배열: %s" % bad if bad else "%d개 배열 전부 동일" % len(ARRS))

        diff = _same(tmp / "zero", tmp / "trt")
        ck("[2] 켜면 자료가 달라진다", bool(diff), "달라진 배열: %s" % diff)

        ck("[3] 균일(coarse=0)과 비균일이 **같은 창에서 다르다**",
           "fine" in diff and "wide" in diff, "세밀·광역 둘 다 달라야 한다")

        ck("[4] meta 에 적힌다",
           float(m_trt.get("vtex_coarse_s") or 0) == 25.0
           and float(m_zero.get("vtex_coarse_s") or 0) == 0.0
           and float(m_none.get("vtex_coarse_s") or 0) == 0.0,
           "처치 %r · 0 판 %r" % (m_trt.get("vtex_coarse_s"), m_zero.get("vtex_coarse_s")))

        # [5] 생성기가 실제로 내는 토막 수 — 홀드아웃 입구로 지어서 센다
        from src.synthesis.grid_simulator import GridSimulator
        g_u = GridSimulator(vtex_seg_s=1.25, vtex_coarse_s=0.0)
        g_n = GridSimulator(vtex_seg_s=1.25, vtex_coarse_s=25.0)
        r = np.random.default_rng(0)
        k_u, _ = g_u._texture_plan(3600, r)
        ks = [g_n._texture_plan(3600, np.random.default_rng(i))[0] for i in range(50)]
        k_n = int(np.median(ks))
        ck("[5] 창당 토막 균일 %d -> 비균일 %d" % (k_u, k_n),
           k_u >= 40 and 8 <= k_n <= 16, "1.25초 균일이면 48장, 비균일 기대 12장")

        # [6] 음성 대조 — 배선을 일부러 끊으면 굽기가 죽어야 한다.
        #     `NILM_HO_BLIND_COARSE` 는 `holdout.py` 안의 **이 관문 전용** 고리다.
        caught, msg = _bake(tmp / "neg", ["--vtex-coarse-s", "25"], expect_fail=True,
                            env_extra={"NILM_HO_BLIND_COARSE": "1"})
        ck("[6] 음성 대조 — 시뮬레이터에 안 걸리면 **굽기가 죽는다**", caught,
           msg.strip().splitlines()[-1][:90] if caught else "조용히 균일로 구워졌다")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
