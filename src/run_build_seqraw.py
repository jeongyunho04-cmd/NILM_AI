# -*- coding: utf-8 -*-
"""원시 시퀀스 캐시 — 몸통까지 끝까지 학습하는 판 (13.84.26).

13.84.24 의 얼린 몸통 판은 z 만 담아 444MB 였다. 몸통도 배우려면 **원시 45채널**이 있어야 하고,
입력(세밀 57x600 · 광역 47x120)은 **미리 굽지 않는다** — 격자 2초에 세밀 창이 10초라 같은 사이클이
다섯 번 들어가 5.9배가 된다(단계당 155.6KB · 120만 단계면 187GB). 원시로 담고 학습할 때 만든다.

담는 것 (기록 i, 격자 단계 t):
    raw   (45, N)      원시 채널. `build_inputs` 가 이걸 창으로 잘라 먹는다
    y_*   (T, K)       전력·켜짐·플러그·대기·상태 — 창 캐시와 **같은 라벨**
    obs_harm (T,15,2)  관측 고조파 (L_harm 이 쓴다)
    p_noise · p_observed (T,)
    z_grid (T, 2)      선로 임피던스 (보조 감독)
⚠ `y_on` 은 **듀티 구멍을 메운 판**이다 (13.84.24 ②) — 안 메우면 전이 라벨의 89% 가 핫플 릴레이다.

    python -X utf8 src/run_build_seqraw.py --out cache/seqraw_v1 --records 3000
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.model.inputs import VOLT_ORDERS
from src.synthesis.sequence import DUTY_CLOSE_S, _close_gaps, make_record, to_raw45

FS = 60
W_CYC = 3600
LOOK = 360
TGT_OFF = W_CYC - 1 - LOOK
PRE = 13

_G = {}


def _init(npz_dir, split, gen_json):
    from src.synthesis.genopts import build_synthesizer, check, resolve
    opts = resolve(gen_json)
    gen = build_synthesizer(opts, npz_dir, split)
    bad = check(opts, gen)
    if bad:
        raise SystemExit("[seqraw] 생성기 설정이 안 걸렸다: " + " / ".join(bad))
    _G["gen"] = gen
    _G["apps"] = sorted(gen.known_appliances)


def _one(args):
    i, n_cyc, grid, seed = args
    gen, apps = _G["gen"], _G["apps"]
    # ⚠ 생성기는 **전역 난수**를 먹는다 (환경 표집·잡음·회로 되먹임). 일정만 지역 RandomState 로
    #   뽑아서는 재현이 안 된다 — 워커 수와 `imap_unordered` 가 집어 가는 순서에 따라 달라진다.
    #   `traincache._chunk` 와 **같은 규약**으로 기록 번호에서 시드한다 ([[ab-compare-seed-after-construct]]).
    from src.synthesis.dataset import chunk_seed
    sd = chunk_seed(seed, i)
    np.random.seed(sd)
    smp = make_record(gen, n_cyc, np.random.RandomState(sd))
    raw = to_raw45(smp, VOLT_ORDERS)
    g = int(DUTY_CLOSE_S * FS)
    K = len(apps)
    T = len(grid)
    y_on = np.zeros((T, K), np.int8)
    y_pow = np.zeros((T, K), np.float32)
    y_plg = np.zeros((T, K), np.int8)
    y_sb = np.zeros((T, K), np.float32)
    y_st = np.zeros((T, K), np.int16)
    for k, a in enumerate(apps):
        on = _close_gaps(np.asarray(smp.gt_is_on[a]).astype(bool), g)
        y_on[:, k] = on[grid]
        y_pow[:, k] = np.asarray(smp.gt_target_power_w[a])[grid]
        y_plg[:, k] = np.asarray(smp.gt_is_plugged[a])[grid]
        y_sb[:, k] = np.asarray(smp.gt_standby_power_w[a])[grid]
        y_st[:, k] = np.asarray(smp.gt_state_id[a])[grid]
    oh = np.asarray(smp.harmonics_ri)[grid]
    pn = np.asarray(smp.p_noise_w)[grid]
    po = np.asarray(smp.power_features)[grid, 0]
    zg = np.full((T, 2), np.nan, np.float32)
    zg[:, 0] = smp.metadata.get("r_grid_ohm", np.nan)
    zg[:, 1] = smp.metadata.get("x_grid_ohm", np.nan)
    return (i, raw.astype(np.float32), y_on, y_pow, y_plg, y_sb, y_st,
            oh.astype(np.float32), pn.astype(np.float32), po.astype(np.float32), zg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=3000)
    ap.add_argument("--record-s", type=float, default=300.0)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz-dir", default="processed_data/npz")
    ap.add_argument("--split", default="train")
    ap.add_argument("--gen", default="v36",
                    help="생성기 설정. 'v36'(=cache_v36.sbatch 와 같은 값) · 'legacy'(seqraw_v1) · JSON")
    a = ap.parse_args()

    N = int(a.record_s * FS)
    grid = np.arange(TGT_OFF, N - PRE * FS - 1, int(a.grid_s * FS))
    T = len(grid)
    from src.synthesis.genopts import describe, resolve
    gen_opts = resolve(a.gen)
    print("[seqraw] 생성기 '%s': %s" % (a.gen, describe(gen_opts)))
    from src.synthesis.segment_pool import SegmentPool
    apps = sorted(SegmentPool(npz_dir=a.npz_dir, time_split=a.split,
                              carrier_apps=tuple(gen_opts.get("carrier_apps") or ()) or None
                              ).get_appliance_types())
    K = len(apps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = {"raw": (a.records, 45, N), "y_on": (a.records, T, K),
              "y_power": (a.records, T, K), "y_plugged": (a.records, T, K),
              "y_standby": (a.records, T, K), "y_state": (a.records, T, K),
              "obs_harm": (a.records, T, 15, 2), "p_noise": (a.records, T),
              "p_observed": (a.records, T), "z_grid": (a.records, T, 2)}
    dt = {"raw": np.float32, "y_on": np.int8, "y_power": np.float32,
          "y_plugged": np.int8, "y_standby": np.float32, "y_state": np.int16,
          "obs_harm": np.float32, "p_noise": np.float32, "p_observed": np.float32,
          "z_grid": np.float32}
    tot = sum(int(np.prod(s)) * np.dtype(dt[k]).itemsize for k, s in shapes.items())
    print("[seqraw] 기록 %d x %.0f초 · 격자 %.1f초 -> %d단계/기록 · 기기 %d · 예상 %.2f GB"
          % (a.records, a.record_s, a.grid_s, T, K, tot / 1e9))
    mm = {k: np.lib.format.open_memmap(out / (k + ".npy"), mode="w+", dtype=dt[k], shape=s)
          for k, s in shapes.items()}

    tasks = [(i, N, grid, a.seed) for i in range(a.records)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    import json as _json
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.npz_dir, a.split, _json.dumps(gen_opts))) as pool:
        for done, r in enumerate(pool.imap_unordered(_one, tasks, chunksize=4), 1):
            i, raw, y_on, y_pow, y_plg, y_sb, y_st, oh, pn, po, zg = r
            mm["raw"][i] = raw
            mm["y_on"][i] = y_on
            mm["y_power"][i] = y_pow
            mm["y_plugged"][i] = y_plg
            mm["y_standby"][i] = y_sb
            mm["y_state"][i] = y_st
            mm["obs_harm"][i] = oh
            mm["p_noise"][i] = pn
            mm["p_observed"][i] = po
            mm["z_grid"][i] = zg
            if done % 200 == 0:
                el = time.time() - t0
                print("  %d/%d  (%.2f초/기록, 남은 %.0f분)"
                      % (done, a.records, el / done, (a.records - done) * el / done / 60),
                      flush=True)
    meta = dict(records=a.records, record_s=a.record_s, grid_s=a.grid_s, steps=T,
                appliances=apps, seed=a.seed, window_cycles=W_CYC,
                target_lookahead=LOOK, target_offset=TGT_OFF,
                duty_close_s=DUTY_CLOSE_S, gen=a.gen, gen_opts=gen_opts, grid=[int(x) for x in grid[:3]] + ["..."],
                build_seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    Y = np.asarray(mm["y_on"])
    print("[seqraw] 끝 %.1f분 · 기기별 ON 몫 / 전이 수" % ((time.time() - t0) / 60))
    for k, ap_ in enumerate(apps):
        tr = int((np.diff(Y[:, :, k].astype(np.int8), axis=1) != 0).sum())
        print("    %-18s %5.1f%%  %6d" % (ap_, 100 * Y[:, :, k].mean(), tr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
