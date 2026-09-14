# -*- coding: utf-8 -*-
"""SMPS 기기의 **수준 오차** — 계단 참값과 단독 창 참값으로 (14.74).

왜 따로 만드나
--------------
`run_diag_levelerr` 는 **컨덕턴스 참값**(`P = V^2/R`)을 쓴다. 저항 기기만 된다.
SMPS 는 정전압 제어라 `R` 이 없어서 그 자로는 못 잰다. 그래서 2026-09-14 까지
**SMPS 수준 오차를 한 번도 안 쟀다.**

그런데 재보니 그게 사슬의 머리였다 (14.75):
```
  test_4 · 빔프로젝터 예측 중앙 46.73W · 참값(자기 계단) 37~41W  -> **+6~9W 과대**
  그 결과 충전기가 꺼져 총합이 54W 로 내려가면
    관측 53.70 − 기준선 4.23 − 빔프 46.63 = **남는 자리 2.5W**
    미니PC raw 는 9.96W 로 **맞게** 내는데, 켜면 +7.1W 초과 / 끄면 −2.5W 미달이라
    모델이 게이트를 0.004 로 눌러 버린다 (`power = sigmoid(on_logit) * p_raw`).
```

참값 둘
-------
```
① 계단   — 그 기기 자신의 on/off 이벤트 |ΔP| (`real_events.json` 의 delta_p_w).
           한 구간의 참값 = 양끝 계단의 평균. 한쪽만 있으면 그것.
② 단독 창 — **전 기기 중 그것 하나만** 켜진 창. 참값 = 관측 − 전부-꺼짐 기준선.
           제일 세지만 그런 창이 드물다.
```

⚠ **대조를 같이 낸다.** 저항 기기도 같은 자로 재서 `run_diag_levelerr` 의 컨덕턴스
참값과 견준다. 두 자가 어긋나면 **이 자를 믿지 마라** — 오븐 +24.9 +- 4.5W 가 기준이다.

    python -X utf8 src/run_diag_levelsmps.py --ckpt results/cnn_v32h_on_s0.pt
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_diag_rollback import predict  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402

SMPS = ("beam_projector", "laptop_charger", "minipc")
#: ⚠ **계단 참값이 안 서는 기기.** 라벨 ON 계단이 그 기기의 운전 전력이 아니다.
#:   오븐: 라벨 ON 은 팬·조명(16.8W)이고 히터는 나중에 붙는다 -> 계단 11W 대 운전 1100W.
#:   핫플·드라이: 듀티/다단이라 구간 중앙이 계단보다 낮다.
#:   2026-09-14 에 이 자를 만들고 **대조로 이것을 확인했다** — 저항 줄을 읽지 마라.
STEP_INVALID = ("oven", "hotplate", "hair_dryer", "electiric_kettle")
RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
KO = {"beam_projector": "빔프", "laptop_charger": "충전기", "minipc": "미니PC",
      "electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플",
      "hair_dryer": "드라이", "fan": "선풍기", "air_conditioner": "에어컨"}


#: 전환 앞뒤 가드와 정착 창 (초). 가드는 전이를, 창은 잡음을 다룬다.
STEP_GUARD, STEP_WIN = 2.0, 6.0
#: ⚠⚠ **양쪽이 조용할 때만 쓴다** (W). 오븐·핫플의 듀티 순환은 `events` 에 **안 적혀**
#:   있어서 "다른 기기 전환" 거르개로는 못 거른다. 이걸 안 걸었더니 test_1/3/5 에서
#:   계단이 1,070~1,140W 로 읽혀 빔프 참값이 123W 가 됐다 (2026-09-14).
#:   `run_diag_zharm` 이 `P.std() > 40` 으로 같은 일을 한다.
STEP_QUIET_W = 3.0


def observed_step(ev, P, fs, app):
    """{전환시각: 계단W} — **관측 총전력**의 정착 구간 차.

    ⚠⚠ 2026-09-14: 처음에 `real_events.json` 의 `delta_p_w` 를 참값으로 썼다가
    **빔프로젝터를 39.4W 로 읽어 "+19% 과대예측" 이라는 틀린 결론**을 냈다.
    사용자가 *"복합 녹화도 45~46W 수준이야"* 라고 짚어 다시 재보니 관측 계단이
    **42.8W**(깨끗한 전환 10개 중앙)였다 — 라벨이 3~7W 를 과소로 적는다.
    `delta_p_w` 는 **귀속값**(`_note` 의 "ΔP출처 h3")이지 계단 자체가 아니다.
    ⇒ 참값은 **관측에서** 낸다. 그리고 다른 기기가 그 창에서 전환하면 **버린다**.
    """
    n = len(P)
    t = np.arange(n) / fs
    others = sorted(float(e["t_s"]) for e in ev.get("events", [])
                    if e.get("appliance") != app)
    out = {}
    for e in ev.get("events", []):
        if e.get("appliance") != app:
            continue
        te = float(e["t_s"])
        a0, a1 = te - STEP_GUARD - STEP_WIN, te - STEP_GUARD
        b0, b1 = te + STEP_GUARD, te + STEP_GUARD + STEP_WIN
        if a0 < 0 or b1 > t[-1]:
            continue
        if any(a0 - 1 <= o <= b1 + 1 for o in others):
            continue                                   # 다른 기기 전환이 겹친다
        ma, mb = (t >= a0) & (t < a1), (t >= b0) & (t < b1)
        if P[ma].std() > STEP_QUIET_W or P[mb].std() > STEP_QUIET_W:
            continue                                   # 한쪽이라도 시끄럽다
        pa, pb = float(np.median(P[ma])), float(np.median(P[mb]))
        out[round(te, 2)] = abs(pb - pa)
    return out


def edge_truth(ev, app, P=None, fs=60.0):
    """[(t0, t1, 참W, 계단수)] — 그 기기 자신의 on/off 계단으로 낸 구간별 참값."""
    out = []
    step = observed_step(ev, P, fs, app) if P is not None else {}
    for t0, t1 in ev["intervals"].get(app, {}).get("on", []):
        v = [step.get(round(float(t0), 2)), step.get(round(float(t1), 2))]
        v = [x for x in v if x and x > 0]
        if v:
            out.append((float(t0), float(t1), float(np.mean(v)), len(v)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=0.5)
    ap.add_argument("--guard-s", type=float, default=4.0,
                    help="구간 양끝에서 이만큼 잘라낸다 (전이 창을 빼려고)")
    ap.add_argument("--min-s", type=float, default=12.0, help="이보다 짧은 구간은 버린다")
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    ev_all = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]

    for path in a.ckpt:
        ap_, pr = predict(path, cache, dev, postproc=a.postproc)
        assert ap_ == apps
        print()
        print("=" * 88)
        print("%s   수준 오차 (예측 − 참)  **양수 = 과대예측**" % path.split("/")[-1])
        print("=" * 88)

        for grp, nm in ((SMPS, "SMPS"), (RES, "저항 (대조 — 컨덕턴스 자와 견줘라)")):
            print("  [%s]" % nm)
            print("    %-8s %5s %8s %9s %9s %9s   %s"
                  % ("기기", "구간", "참W(계단)", "예측W", "오차W", "오차%", "파일별"))
            for app in grp:
                if app not in apps:
                    continue
                k = apps.index(app)
                rows, per = [], []
                for stem, d in sorted(cache.items()):
                    ev = ev_all.get(stem)
                    if not ev or app not in ev.get("appliances_present", []):
                        continue
                    t = d["t"]
                    w = pr[stem][1][:, k]
                    from src.preprocessing import load_nilm_npz
                    _r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
                    _P = np.asarray(_r["power_features"])[:, 0]
                    for t0, t1, tw, nedge in edge_truth(ev, app, _P, 60.0):
                        if t1 - t0 < a.min_s:
                            continue
                        m = (t >= t0 + a.guard_s) & (t <= t1 - a.guard_s)
                        if m.sum() < 5:
                            continue
                        pw = float(np.median(w[m]))
                        rows.append((tw, pw))
                        per.append((stem, pw - tw))
                if not rows:
                    print("    %-8s (구간 없음)" % KO.get(app, app))
                    continue
                tw = np.array([x[0] for x in rows]); pw = np.array([x[1] for x in rows])
                e = pw - tw
                byf = {}
                for stem, x in per:
                    byf.setdefault(stem, []).append(x)
                bad = " **계단 무효**" if app in STEP_INVALID else ""
                print("    %-8s %5d %8.1f %9.1f %+9.1f %+8.1f%%   %s%s"
                      % (KO.get(app, app), len(rows), tw.mean(), pw.mean(),
                         e.mean(), 100.0 * e.mean() / max(tw.mean(), 1e-6),
                         " ".join("%s %+.0f" % (s.replace("test_", "t"), np.mean(v))
                                  for s, v in sorted(byf.items())), bad))
            print()

        # ── 단독 창 참값 (제일 센 자) ────────────────────────────────────
        # ⚠ **통전 거르개**가 없으면 오븐의 팬·조명 창이 히터 창과 섞여 참값이 무너진다
        #   (`harmonic_signatures` 와 같은 규칙: 그 기기 정상 전력의 절반 이상).
        #   이것을 넣으니 핫플 +2.6W · 오븐이 `run_diag_levelerr` 의 컨덕턴스 자와 맞는다.
        from src.synthesis.segment_pool import SegmentPool
        _pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        print("  [단독 창 — 그것 하나만 켜지고 **통전**(정상 전력의 절반 이상)인 창]")
        # ⚠ **그 기기 예측**과 **전체 예측합**을 같이 낸다. 둘이 갈리면 수준 오차가 아니라
        #   **다른 기기로 샌 것**이다 (저항 축퇴, 14.71). 그 구분 없이 읽으면 오븐의
        #   −610W 를 "과소예측" 으로 잘못 적게 된다.
        print("    %-8s %6s %9s %9s %9s %9s %9s"
              % ("기기", "창", "참W", "그 기기", "오차W", "전체합", "합 오차"))
        y_any = {stem: d["y"].astype(bool) for stem, d in cache.items()}
        for app in list(SMPS) + list(RES):
            if app not in apps:
                continue
            k = apps.index(app)
            tw, pw, sw = [], [], []
            for stem, d in sorted(cache.items()):
                y = y_any[stem]
                try:
                    thr = 0.5 * float(_pool.get_steady_power_w(app))
                except Exception:
                    thr = 0.0
                solo = ((y.sum(1) == 1) & y[:, k]
                        & (d["p_obs"] - float(d["p_base"]) > thr))
                if solo.sum() < 5:
                    continue
                tw += list(d["p_obs"][solo] - float(d["p_base"]))
                pw += list(pr[stem][1][solo, k])
                sw += list(pr[stem][1][solo].sum(1))
            if len(tw) < 5:
                print("    %-8s (단독 창 부족)" % KO.get(app, app))
                continue
            tw = np.array(tw); pw = np.array(pw); sw = np.array(sw)
            leak = " **샌다**" if abs((pw - tw).mean()) > 3 * abs((sw - tw).mean()) + 20 else ""
            print("    %-8s %6d %9.1f %9.1f %+9.1f %9.1f %+9.1f%s"
                  % (KO.get(app, app), len(tw), tw.mean(), pw.mean(), (pw - tw).mean(),
                     sw.mean(), (sw - tw).mean(), leak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
