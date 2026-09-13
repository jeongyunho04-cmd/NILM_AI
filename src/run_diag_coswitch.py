# -*- coding: utf-8 -*-
"""0단계 (i) — 한 사슬 스텝에 **몇 대가 동시에** 상태를 바꾸는가 (재설계 계획 C②).

AFAMAP (Kolter & Jaakkola 2011) 의 핵심 장치 셋 중 하나가 **"한 번에 한 상태만 바뀐다"**
제약이다. 그것이 성립하면 결합 상태공간이 2^9=512 에서 K+1=10 으로 줄어, 9개 사슬을
관측 합으로 **결합**해도 계산이 감당된다. 성립 안 하면 참을 자른다.

    잰다: 실측 5파일의 사람 타임라인에서, 스텝 간격 S 초로 격자를 놓고
          연속한 두 격자점 사이에 **상태가 바뀐 기기 수**의 분포.

⚠ 규칙 14 — 이것은 라벨 통계다. 모델은 안 본다. `real_events.json` 이 정답 원본이고
   `build_on_off_truth` 가 uncertain 구간을 `scorable=False` 로 빼 준다. **그 구간을
   OFF 로 치면 안 된다** (오븐 팬/조명이 대표 사례) — 여기서는 양 끝점이 모두
   scorable 인 스텝만 센다. 뺀 수도 함께 찍는다.

    python -X utf8 src/run_diag_coswitch.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.evaluation.real_events import build_on_off_truth, load_events

FS = 60
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
#: 사슬 스텝 후보 (초). 지금 쓰는 값은 2.0 (`run_build_seqcache.py --grid-s`).
STEPS_S = (1.0, 2.0, 4.0)


def truth(stem, apps, n):
    on, sc = build_on_off_truth(stem, apps, n, events=load_events())
    return np.asarray(on, bool), np.asarray(sc, bool)


def main():
    ev = load_events()
    rows = {s: [] for s in STEPS_S}
    print("0단계 (i) — 사슬 스텝당 동시 전이 기기 수", flush=True)
    print("=" * 72)
    for stem in FILES:
        d = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
        n = int(d["is_on"].shape[0])
        apps = sorted(ev[stem]["intervals"].keys())     # 그 파일에 실제로 있는 기기만
        on, sc = truth(stem, apps, n)
        print()
        print("[%s] %.0f초 · 기기 %d종 (%s)" % (stem, n / FS, len(apps), ", ".join(apps)))
        for s_s in STEPS_S:
            step = int(round(s_s * FS))
            idx = np.arange(0, n, step)
            o, k = on[idx], sc[idx]
            # 양 끝점이 모두 scorable 인 기기만 그 스텝에서 센다
            ok = k[:-1] & k[1:]
            chg = (o[:-1] != o[1:]) & ok
            ncg = chg.sum(1)
            # 스텝 전체를 버리는 기준: 그 스텝에 판정 불가 기기가 하나라도 있으면
            full = ok.all(1)
            nf = ncg[full]
            tot = len(nf)
            if tot == 0:
                print("   S=%.0fs  판정 가능한 스텝이 없다" % s_s)
                continue
            c0 = int((nf == 0).sum()); c1 = int((nf == 1).sum()); c2 = int((nf >= 2).sum())
            mx = int(nf.max())
            # 전이가 **있는** 스텝 중 ≥2 인 비율이 제약의 실제 손해다
            act = c1 + c2
            print("   S=%.0fs  스텝 %4d (제외 %3d)   변화없음 %4d · 1대 %3d · **≥2대 %3d**"
                  "   최대 %d대   전이스텝중 ≥2 = %s"
                  % (s_s, tot, int((~full).sum()), c0, c1, c2, mx,
                     ("%.1f%%" % (100.0 * c2 / act)) if act else "—"))
            rows[s_s].append((c0, c1, c2, int((~full).sum())))

    print()
    print("=" * 72)
    print("전 파일 합계")
    print("  S      스텝      변화없음     1대     **≥2대**    ≥2 비율   전이스텝중 ≥2")
    for s_s in STEPS_S:
        if not rows[s_s]:
            continue
        a = np.array(rows[s_s]).sum(0)
        c0, c1, c2 = int(a[0]), int(a[1]), int(a[2])
        tot = c0 + c1 + c2
        act = c1 + c2
        print("  %.0fs  %6d  %8d  %6d  %10d   %6.2f%%   %s"
              % (s_s, tot, c0, c1, c2, 100.0 * c2 / max(tot, 1),
                 ("%.1f%%" % (100.0 * c2 / act)) if act else "—"))
    print()
    print("판정 기준 (계획 C② — 미리 적는다):")
    print("  전이스텝중 ≥2 가  < 5%  -> 제약이 사실상 공짜. 그대로 넣는다")
    print("                  5~20% -> 제약을 **소프트**로 (동시전이에 벌점, 금지는 아님)")
    print("                  > 20% -> 참을 자른다. C② 는 버리고 C①③ 만 간다")


if __name__ == "__main__":
    main()
