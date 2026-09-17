# -*- coding: utf-8 -*-
"""배포 묶음을 **학습 워크스페이스와 다시 맞춘다** (14.378).

왜 이게 있나 — 2026-09-14 판 묶음은 `nilm_runtime/` 안에 `net.py`·`inputs.py`·
`postproc.py` 를 **손으로 깎은 사본**으로 들고 있었다. 그래서 사흘 만에 이렇게 됐다:

```
  세밀 채널  묶음 50  ·  지금 **61**        광역  묶음 12  ·  지금 **51**
  net.py     묶음 381줄  ·  지금 **2147줄**  (조합 머리가 통째로 없다)
  ⇒ 체크포인트만 갈아 끼우면 **모양이 안 맞아 죽는다**. 조용히 어긋난 게 아니라
    "새 모델을 배포할 수 없는 상태"로 15일을 있었다
```
⇒ 고침은 **사본을 없애는 것**이다. 연구 코드를 `deploy/src/` 로 **바이트 동일**하게
  비추고, 묶음 고유 코드(`predictor`·`receiver`)만 `nilm_runtime/` 에 남긴다.
  그러면 어긋남이 `cmp` 한 번으로 **보인다** ([[verify-the-input-path-not-just-the-model]]).

    python deploy/sync_runtime.py            # 다시 맞춘다 (+ 표를 굽는다)
    python deploy/sync_runtime.py --check    # 어긋났으면 **0 이 아닌 값**으로 죽는다
"""
from pathlib import Path
import argparse
import filecmp
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = Path(__file__).resolve().parent

#: 그대로 비추는 것. **고치지 마라** — 연구 코드와 바이트가 같아야 한다.
MIRROR = [
    "src/model/inputs.py",          # 33ch -> 세밀 61 / 광역 51
    "src/model/net.py",             # NILMNet + 조합 머리
    "src/model/build.py",           # ★ 체크포인트 -> 모델 (연구 채점기와 같은 함수)
    "src/model/postproc.py",        # 상한·저항정합·스켈치·흡수
    "src/model/losses.py",          # S_STATE (p_state_cap 이 읽는다). 손실은 안 쓴다
    "src/model/gbudget.py",         # ★ Ĝ 예산 + PIN_MS + COMB_OVER_OP
    "src/model/physdecomp.py",      # ★ Ĝ 를 푸는 정규화 NNLS
    "src/model/fcmtab.py",          # ★ FCM 기둥 표
    "src/labeling/state_definitions.py",
    "src/synthesis/fcm.py",         # ★ circ12 절임 읽기 (numpy2 shim 포함)
    "src/synthesis/circuit_sim.py",
    #: ⚠ 절임이 `circuit_model.fcm12.FCM` 로 풀린다 — **클래스 코드도 와야 한다.**
    #  이게 빠져서 재생 시험이 `ModuleNotFoundError` 로 죽었다.
    "circuit_model/__init__.py",
    "circuit_model/fcm12.py",
    "circuit_model/circuit12.py",
    "circuit_model/circ12_beam_projector.pkl",
    "circuit_model/circ12_laptop_charger.pkl",
    "circuit_model/circ12_minipc.pkl",
]

#: ⚠ 패키지 표식은 **빈 파일로 둔다.** 원본 `src/synthesis/__init__.py` 는
#  `segment_pool`·`synthesizer` 까지 끌어와서 학습 스택이 통째로 딸려 온다.
STUBS = ["src/__init__.py", "src/model/__init__.py",
         "src/labeling/__init__.py", "src/synthesis/__init__.py"]

STUB_TEXT = ("# -*- coding: utf-8 -*-\n"
             "#: 배포 묶음의 패키지 표식. 원본 `__init__` 은 학습 스택을 끌어온다 —\n"
             "#: 여기서는 **비워 둔다** (`deploy/sync_runtime.py` 가 만든다).\n")


def mirror(check: bool) -> list:
    bad = []
    for rel in MIRROR:
        src, dst = ROOT / rel, DEPLOY / rel
        if not src.exists():
            bad.append("원본 없음: %s" % rel); continue
        if check:
            if not dst.exists():
                bad.append("묶음에 없음: %s" % rel)
            elif not filecmp.cmp(src, dst, shallow=False):
                bad.append("**어긋남**: %s" % rel)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in STUBS:
        dst = DEPLOY / rel
        if check:
            if not dst.exists():
                bad.append("표식 없음: %s" % rel)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(STUB_TEXT, encoding="utf-8")
    return bad


def bake(apps) -> None:
    """풀에서만 나오는 표를 **구워 둔다** — 배포에는 `SegmentPool` 이 없다.

    `signatures.npz`(잔차 흡수용)와 같은 규약이다. 여기 것은 `gbudget.Budget` 이
    에어컨 기둥을 세울 때 쓰는 **상태별 지문**이다.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from src.model.net import (harmonic_signatures_by_state, harmonic_signatures,
                               standby_signatures, noise_signature,
                               reactive_signatures, noise_reactive)
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir=str(ROOT / "processed_data/npz"), time_split="train")
    sig_s, us = harmonic_signatures_by_state(pool, list(apps))
    out = DEPLOY / "nilm_runtime/runtime_tables.npz"
    np.savez_compressed(out, appliances=np.array(list(apps)),
                        sig_state=sig_s, sig_state_used=us)
    print("  구움  %s  (상태별 지문 %s · 맞춘 칸 %d)"
          % (out.name, sig_s.shape, int(us.sum())))
    # 잔차 흡수용 지문도 같은 풀에서 다시 굽는다 (기기 목록이 바뀌면 깨진다)
    #: ⚠ `qp`/`noise_q` 를 빼먹지 마라 — `absorb_mode="pq"`(운영 기본)가 없으면
    #  `ValueError` 로 죽는다. 처음 구울 때 실제로 빠뜨렸고 바로 잡혔다.
    _qp, _qu = reactive_signatures(pool, list(apps))
    _qp, _qu = np.asarray(_qp), np.asarray(_qu, bool)
    z = DEPLOY / "nilm_runtime/signatures.npz"
    np.savez_compressed(z, appliances=np.array(list(apps)),
                        sig=np.asarray(harmonic_signatures(pool, list(apps)), np.float32),
                        standby_sig=np.asarray(standby_signatures(pool, list(apps)), np.float32),
                        noise_sig=np.asarray(noise_signature(pool), np.float32),
                        qp=_qp.astype(np.float32), qp_usable=_qu,
                        noise_q=np.float32(noise_reactive(pool)))
    print("  구움  %s  (qp·noise_q 포함)" % z.name)


def main() -> int:
    ap = argparse.ArgumentParser(description="배포 묶음 재동기화")
    ap.add_argument("--check", action="store_true", help="어긋났으면 죽는다 (관문용)")
    ap.add_argument("--no-bake", action="store_true", help="표 굽기를 건너뛴다")
    ap.add_argument("--apps", default=("air_conditioner,beam_projector,electiric_kettle,"
                                       "fan,hair_dryer,hotplate,laptop_charger,minipc,oven"))
    a = ap.parse_args()
    apps = [s for s in a.apps.split(",") if s]
    bad = mirror(a.check)
    if a.check:
        for b in bad:
            print("  " + b)
        print("\n%s  (%d/%d 일치)"
              % ("묶음이 연구 코드와 **같다**" if not bad else "**어긋났다** — "
                 "`python deploy/sync_runtime.py` 로 다시 맞춰라",
                 len(MIRROR) + len(STUBS) - len(bad), len(MIRROR) + len(STUBS)))
        return 1 if bad else 0
    print("비춘 파일 %d + 표식 %d" % (len(MIRROR), len(STUBS)))
    if bad:
        for b in bad:
            print("  " + b)
        return 1
    if not a.no_bake:
        bake(apps)
    print("\n다 맞췄다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
