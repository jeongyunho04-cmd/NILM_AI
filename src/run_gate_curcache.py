# -*- coding: utf-8 -*-
"""`current()` 메모이즈가 **수치를 한 비트도 안 바꾼다** (14.99).

왜 이 고침인가
--------------
`vtex_seg_s > 0` 로 창을 토막 내면 `texture_delta` 가 토막마다 회로를 **두 번** 푼다:
```
  a = current(device, pq, rel_env, vq, rq)   # 그 토막의 텍스처  -> 토막마다 다르다
  b = current(device, pq, rel_rec, vq, rq)   # **녹화 자신의 텍스처**
```
`b` 의 입력(`rel_rec`·`p`·`v1`·`R`)은 **토막 내내 완전히 같은데**, `texture_delta` 의 캐시
키가 **쌍**(`env_id`,`rec_id`)이라 토막이 바뀌면 통째로 미스가 난다. 20토막이면 `b` 를
**19번 헛계산**한다. `_compute_coupling` 의 `b`(그 토막의 텍스처)도 `texture_delta` 의 `a`
와 같은 계산이라, `current()` 를 메모이즈하면 **교차 적중**까지 얻는다.

무엇을 확인하나
---------------
```
  [1] 같은 입력에 캐시 켬/끔이 **바이트 동일**한 전류를 낸다 (감시값·이름공간 포함)
  [2] `None`(회로 실패)도 캐시된다 — `.get(key)` 였으면 미스로 새는 자리다
  [3] 텍스처 id 와 녹화 id 가 **같은 정수**여도 안 섞인다 (이름공간)
  [4] ★ 진짜 굽기를 **캐시 켜고/끄고** 구워 모든 배열이 **바이트 동일**
        + 그때의 적중률과 속도를 찍는다 (이득이 얼마인지 여기서 나온다)
```
⚠ [4] 가 이 관문의 전부다. [1]~[3] 은 단위 검사고, **굽기 산출물이 바이트 동일**해야
  기존 캐시(v32·v32h)와 견줄 수 있다 ([[the-gate-must-build-the-real-object]]).

    python -X utf8 src/run_gate_curcache.py
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

COMMON = [
    "--windows", "300000", "--window-cycles", "3600", "--seed", "0",
    "--power-scale-std", "measured", "--recipe-mix", "steady2",
    "--smps-focus-off-p", "0.4", "--state-mix", "minipc_balanced",
    "--carrier-on", "oven", "--couple-ext", "--sp-curves", "--sp-per-texture",
    "--vtail", "--float-fill", "charger_float", "--steady-crop", "smps_steady",
    "--standby-jitter-cap", "95", "--harmonic-z",
    "--vtex-step-s", "3", "--vtex-seg-s", "3",
]
#: 관문 크기 — **분모는 청크 수(1200)를 넘을 수 없다** (`0/3000` 은 "토막에 청크가
#: 없다" 로 죽는다). `0/1200` = **250창이 최소**다 — `chunk=250` 이 `build_cache` 에
#: 박혀 있고 CLI 로 못 바꾼다. 검정 자체는 25창이면 충분한데 그게 한계다.
#: 시간은 `고정 시작부하 115초 x 2판` + `창수 x 한계비용 / 워커 x 2판`. 워커 40이면 ~10분.
#: 워커 수는 산출물을 안 바꾼다
#: (`test_worker_count_does_not_change_the_generated_training_set`).
SHARD = os.environ.get("GATE_SHARD", "0/1200")
WORKERS = os.environ.get("GATE_WORKERS", "40")
ARRS = ("fine", "wide", "y_power", "y_on", "y_plugged", "y_standby",
        "y_state", "obs_harm", "p_noise", "p_observed")


def unit_checks() -> bool:
    from src.synthesis import coupling as C
    ok = True
    circ = C.SmpsCircuit()
    dev = "minipc"
    if not circ.has(dev):
        print("  [1~3] 건너뜀 — 회로 모델이 없다")
        return True
    rng = np.random.default_rng(0)
    rel_a = np.ones(15, complex)
    rel_a[1:] += 0.01 * (rng.standard_normal(14) + 1j * rng.standard_normal(14))
    rel_b = np.ones(15, complex)
    rel_b[1:] += 0.01 * (rng.standard_normal(14) + 1j * rng.standard_normal(14))
    pb, pq, vb, vq, rb, rq = 10, 50.0, 222000, 222.0, -1, None

    raw = circ.current(dev, pq, rel_a, vq, rq)
    c1 = circ.current_id(dev, pb, pq, rel_a, ("env", 7), vb, vq, rb, rq)
    c2 = circ.current_id(dev, pb, pq, rel_a, ("env", 7), vb, vq, rb, rq)   # 적중
    same = (raw is None and c1 is None) or np.array_equal(raw, c1)
    hit = (c1 is None and c2 is None) or np.array_equal(c1, c2)
    ok &= same and hit and circ.cur_hits >= 1
    print("  [1] 캐시 켬/끔 바이트 동일 %s · 두 번째가 적중 %s (hits %d)"
          % ("OK" if same else "** 다르다 **", "OK" if hit else "**X**", circ.cur_hits))

    n0 = circ.cur_misses
    circ.current_id(dev, pb, 0.0, rel_a, ("env", 999), vb, vq, rb, rq)     # p<=0.5 -> None
    circ.current_id(dev, pb, 0.0, rel_a, ("env", 999), vb, vq, rb, rq)     # 적중해야 한다
    got_none_hit = circ.cur_misses == n0 + 1
    ok &= got_none_hit
    print("  [2] None(실패)도 캐시된다 (감시값)   %s"
          % ("OK" if got_none_hit else "** 미스로 샌다 **"))

    x = circ.current_id(dev, pb, pq, rel_a, ("env", 3), vb, vq, rb, rq)
    y = circ.current_id(dev, pb, pq, rel_b, ("rec", 3), vb, vq, rb, rq)
    sep = not (x is not None and y is not None and np.array_equal(x, y))
    ok &= sep
    print("  [3] 텍스처 id 3 과 녹화 id 3 이 안 섞인다   %s"
          % ("OK" if sep else "** 섞였다 **"))
    return ok


def bake(out: Path, off: bool):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if off:
        env["NILM_NO_CUR_CACHE"] = "1"
    else:
        env.pop("NILM_NO_CUR_CACHE", None)
    t = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.run_build_traincache",
                        "--out", str(out), "--shard", SHARD, *COMMON, "--workers", WORKERS],
                       env=env, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("굽기가 실패했다 (캐시 %s)" % ("끔" if off else "켬"))
    return time.time() - t


def main() -> int:
    print("`current()` 메모이즈 관문 (14.99)")
    print("")
    ok = unit_checks()
    print("")
    tmp = Path(tempfile.mkdtemp(prefix="gate_curcache_"))
    try:
        print("  [4] ★ 진짜 굽기(%s · 워커 %s)를 캐시 **끄고** / **켜고** — 바이트 동일 + 속도"
              % (SHARD, WORKERS))
        t_off = bake(tmp / "off", off=True)
        print("      캐시 끔  %6.1f초" % t_off)
        t_on = bake(tmp / "on", off=False)
        print("      캐시 켬  %6.1f초   (**%.2f배**)" % (t_on, t_off / max(t_on, 1e-9)))
        bad = []
        for nm in ARRS:
            a = np.load(tmp / "off" / ("%s.npy" % nm), mmap_mode="r")
            b = np.load(tmp / "on" / ("%s.npy" % nm), mmap_mode="r")
            same = a.shape == b.shape and np.array_equal(np.asarray(a), np.asarray(b))
            if not same:
                bad.append(nm)
            print("      %-12s %5d창  %s" % (nm, len(a), "바이트 동일 OK" if same else "** 다르다 **"))
        ok &= not bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("관문 전부 통과" if ok else "** 관문 실패 — 굽기에 쓰지 마라 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
