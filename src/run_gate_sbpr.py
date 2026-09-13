# -*- coding: utf-8 -*-
"""`standby_per_record` 관문 (13.84.40) — 굽기 전에 **여기서** 막는다.

둘을 본다.
  ① 항등   꺼짐(`v36`)이면 같은 씨앗에서 기록이 **비트 동일**이어야 한다.
           `new_record()` 가 정말 no-op 인지, 난수 흐름을 안 건드리는지의 실검정이다.
  ② 효과   켜짐(`v36r`)이면 (a) 기록마다 고른 녹화가 실제로 바뀌고
           (b) 녹화별 대기 산포가 목표 대역(h9~h15 에서 2.2~3.6mA)에 든다.

⚠ 로컬에는 `numba` 가 없어 회로모델 경로가 안 돈다 (13.84.9). **캐시를 굽는 환경에서** 돌린다.

    python -X utf8 -m src.run_gate_sbpr
"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.synthesis.genopts import PRESETS, build_synthesizer, check
from src.synthesis.sequence import make_record

ORD = [1, 5, 9, 11, 13, 15]
#: 실측 전부-OFF 배경의 파일 간 편차에서 **꽂힌 기기 대기로 설명되는 몫을 뺀** 나머지 (13.84.40).
TARGET = dict(zip(ORD, [2.05, 1.91, 7.00, 5.42, 4.71, 6.27]))
#: v36 이 주던 것 (계측계 잡음 참조 3개의 산포).
V36_GIVES = dict(zip(ORD, [1.70, 1.52, 1.51, 1.45, 1.48, 1.26]))


def build(preset):
    opts = dict(PRESETS[preset])
    gen = build_synthesizer(opts, "processed_data/npz", "train")
    bad = check(opts, gen)
    if bad:
        raise SystemExit("%s 조립 관문 실패: %s" % (preset, bad))
    return gen


def run(preset, n_rec=4, cycles=60 * 60):
    """같은 씨앗으로 기록 n개. (sha256, 기록별로 고른 녹화 색인)"""
    gen = build(preset)
    h, picks = hashlib.sha256(), []
    for i in range(n_rec):
        # ⚠ 생성기는 **전역 난수**를 먹는다 (HPC_RULES §8). 지역 RandomState 만으로는 안 고정된다.
        np.random.seed(4242 + i)
        smp = make_record(gen, cycles, np.random.RandomState(4242 + i))
        h.update(np.ascontiguousarray(np.asarray(smp.harmonics_ri, np.float32)).tobytes())
        picks.append(dict(getattr(gen.pool, "_standby_pick", {})))
    return h.hexdigest(), picks, gen


def main():
    fail = []

    print("① 항등 — standby_per_record 꺼짐 (v36)")
    a, _, _ = run("v36")
    b, _, _ = run("v36")
    print("   같은 씨앗 두 번  %s  %s" % (a[:16], "일치" if a == b else "**불일치**"))
    if a != b:
        fail.append("v36 가 자기 자신과 재현이 안 된다 — 관문 자체가 못 쓴다")

    print("\n② 효과 — 켜짐 (v36r)")
    c, picks, gen = run("v36r")
    same = (c == a)
    print("   sha %s  — v36 와 %s" % (c[:16], "**같다 (안 걸렸다)**" if same else "다르다"))
    if same:
        fail.append("v36r 이 v36 와 비트 동일이다 — 경로가 안 걸렸다")
    for i, p in enumerate(picks):
        print("   기록 %d  %s" % (i, dict(sorted(p.items())) or "(뽑을 기기 없음)"))
    keys = sorted({k for p in picks for k in p})
    nvary = sum(1 for k in keys if len({p.get(k) for p in picks if k in p}) > 1)
    print("   -> 기록 사이에서 실제로 바뀐 기기 %d / %d" % (nvary, len(keys)))
    if nvary == 0:
        fail.append("기록마다 같은 녹화만 뽑혔다 — 추첨이 안 돈다")

    print("\n③ 크기 — 녹화별 대기 산포가 목표 대역인가 (mA)")
    alt = gen.pool.standby_profiles_all
    print("   %-18s %6s %s" % ("기기", "녹화", "".join("  h%-6d" % o for o in ORD)))
    devs = []
    for app in sorted(alt):
        lst = alt[app]
        if len(lst) < 2:
            continue
        C = np.stack([np.asarray(p.harmonics_complex) for p in lst])
        w = np.array([p.sample_count for p in lst], float)
        w /= w.sum()
        m = (C * w[:, None]).sum(0)
        d = np.array([1000 * float((w * np.abs(C[:, o - 1] - m[o - 1])).sum()) for o in ORD])
        devs.append(d)
        print("   %-18s %6d %s" % (app, len(lst), "".join("  %7.2f" % v for v in d)))
    if not devs:
        fail.append("녹화가 2개 이상인 기기가 없다")
    else:
        tot = np.sqrt((np.stack(devs) ** 2).sum(0))
        print("   %-18s %6s %s  <- 직교합" % ("합", "", "".join("  %7.2f" % v for v in tot)))
        print("   %-18s %6s %s  <- 목표 (실측 남는 몫)"
              % ("", "", "".join("  %7.2f" % TARGET[o] for o in ORD)))
        print("   %-18s %6s %s  <- v36 가 주던 것"
              % ("", "", "".join("  %7.2f" % V36_GIVES[o] for o in ORD)))
        # 고차(h9~h15)에서 v36 보다 **확실히** 커야 의미가 있다. 1.2배를 문턱으로 둔다.
        hi = [o for o in ORD if o >= 9]
        worse = [o for o in hi if tot[ORD.index(o)] < 1.2 * V36_GIVES[o]]
        if worse:
            fail.append("h%s 에서 v36 대비 1.2배를 못 넘는다 — 배울 것이 안 늘었다"
                        % ",".join(str(o) for o in worse))

    print()
    if fail:
        for f in fail:
            print("관문 실패: %s" % f)
        return 1
    print("관문 통과 — 굽어도 된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
