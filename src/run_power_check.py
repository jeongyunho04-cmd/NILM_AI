"""
기기별 전력 오차 채점 — 참값을 아는 넷에 대해 (12.122.6, 2026-09-01)
=======================================================================
**이 저장소는 배분을 겨냥한 처방을 원리적으로 판정할 수 없었다.** 재는 것이
잔차(총합)와 유령(없는 기기)뿐이라, 프로젝터 전력을 88W -> 47W 로 참값에
맞춰도 개선은 어느 지표에도 안 잡히고 부작용만 잡혔다 (12.122.4/12.122.6).

그런데 격리 통전 전력이 좁은 기기가 넷 있다 — 전기포트·핫플·오븐·프로젝터.
이 스크립트가 **정답이 ON 인 구간에서 예측 전력이 참값에서 얼마나 벗어나는지**
잰다.

    # 운영점
    python -m src.run_power_check --ckpt results/adapt_ovh.pt --postproc on \
        --resmatch 0.02 --rm-snap

    # 처방 비교 (스냅이 실제로 배분을 고치는가)
    python -m src.run_power_check --ckpt results/adapt_ovh.pt --postproc on \
        --resmatch 0.02 --rm-snap --snap 47.4

    # 참값 표를 다시 만든다 (격리 녹화가 바뀌었을 때)
    python -m src.run_power_check --recompute-ref

[읽는 법]
    중앙|오차|   배분 오차의 크기. **이 지표의 본체다**
    평균오차     **부호가 있다.** 양수면 과대예측 — 처방의 방향을 정한다
    폭안        격리 p5~p95 안에 든 비율. 소수 상태에 둔감하다
    검출률      정답 ON 중 게이트가 켠 비율. **배분이 아니라 검출 지표다** —
               낮으면 그 기기의 배분 오차는 표본이 적어 못 믿는다

[⚠ 모델끼리 견줄 때는 --postproc 를 끄고 돌릴 것]
상한(프로젝터 55W)이 켜져 있으면 예측이 눌려 **모든 모델이 8.1W 로 같아진다**
(55.0 − 46.9). 후처리 처방을 볼 때만 운영 조합을 쓴다.

[⚠ 검출률이 낮으면 배분 오차를 읽지 말 것]
`P̂ = σ(on)·p_raw` 라 게이트가 꺼진 창은 아예 뺀다. 그래서 검출률이 낮은
기기는 **쉬운 창만 남은 표본**이라 배분 오차가 낙관 쪽으로 치우친다.
"""
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import glob
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.evaluation.power_ref import (REFERENCE_W, format_power_ref,
                                      score_power_ref, summarize_power_ref)
from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import upsample_to_cycles
from src.run_gate_check import _signatures, forward_file, gated

WINDOW = 3600


def recompute_reference(npz_dir: str = "processed_data/npz",
                        max_spread_frac: float = 0.10) -> Dict[str, tuple]:
    """격리 녹화에서 참값 표를 다시 만든다.

    **폭이 중앙의 `max_spread_frac` 을 넘으면 표에 안 넣는다** — 상태가 여럿인
    기기(드라이기 강/약, 충전기 고속/트리클, 미니PC 유휴/작업)를 거른다.
    거른 이유를 같이 찍는다. 규칙 14 — 안 잰 것을 측정처럼 쓰지 않는다.
    """
    from src.preprocessing import classify_file, load_nilm_npz
    print("격리 녹화의 60초 창 통전 전력")
    print(f"  {'기기':18s}{'녹화':>5s}{'창':>6s}{'p5':>9s}{'중앙':>9s}{'p95':>9s}"
          f"{'폭':>8s}{'폭/중앙':>9s}   판정")
    out: Dict[str, tuple] = {}

    # **녹화가 아니라 기기 단위로 묶는다.** 스템으로 가르면 `laptop_charger_4`
    # 처럼 한 상태만 잡힌 녹화가 따로 떨어져 나와 "폭이 좁다" 로 잘못 채택된다
    # (2026-09-01 에 실제로 그랬다 — 충전기는 15~76W 인데 0.082 로 통과했다).
    by_app: Dict[str, List[str]] = {}
    for f in sorted(glob.glob(f"{npz_dir}/*.npz")):
        try:
            app = classify_file(f).appliance_type
        except Exception:
            continue
        if app:
            by_app.setdefault(app, []).append(f)

    for app in sorted(by_app):
        V: List[float] = []
        for f in by_app[app]:
            z = load_nilm_npz(f)
            p = np.asarray(z["p_denoised_w"])
            m = np.asarray(z["is_on"]).astype(bool) & np.asarray(z["is_valid"]).astype(bool) & (p > 1.0)
            i = np.flatnonzero(m)
            for k in range(0, len(i) - WINDOW, WINDOW // 4):
                s = i[k:k + WINDOW]
                if s[-1] - s[0] > WINDOW * 1.5:      # 구간을 넘어 이어붙인 창은 버린다
                    continue
                V.append(float(p[s].mean()))
        if len(V) < 3:
            continue
        a = np.array(V)
        lo, mid, hi = np.percentile(a, [5, 50, 95])
        frac = (hi - lo) / max(mid, 1e-9)
        ok = frac <= max_spread_frac
        if ok:
            out[app] = (round(float(mid), 1), round(float(lo), 1), round(float(hi), 1))
        print(f"  {app:18s}{len(by_app[app]):>5d}{len(a):>6d}{lo:9.1f}{mid:9.1f}{hi:9.1f}"
              f"{hi - lo:8.1f}{frac:9.3f}   {'채택' if ok else '**제외 (상태 여럿)**'}")
    print(f"\nREFERENCE_W = {json.dumps(out, ensure_ascii=False)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/adapt_ovh.pt"])
    ap.add_argument("--stems", nargs="+", default=None, help="기본: 봉인 안 된 전부")
    ap.add_argument("--postproc", default="off", choices=("off", "on", "sync"))
    ap.add_argument("--resmatch", type=float, default=0.0)
    ap.add_argument("--rm-snap", action="store_true")
    ap.add_argument("--snap", type=float, default=0.0, help="프로젝터 스냅 (12.122.4)")
    ap.add_argument("--snap-no-redist", action="store_true")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--recompute-ref", action="store_true",
                    help="격리 녹화에서 참값 표를 다시 만들고 끝낸다")
    ap.add_argument("--out", default="results/power_check.json")
    a = ap.parse_args()

    if a.recompute_ref:
        recompute_reference()
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = a.stems or [s for s in sorted(ev) if not is_sealed(s)]
    payload: Dict[str, dict] = {}

    for ck_path in a.ckpt:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        apps = ck["appliances"]
        from src.model.net import NILMNet, appliance_state_counts
        from src.model.inputs import LEGACY_FINE_CHANNELS
        model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                        wide_summary=ck.get("wide_summary", False),
                        periodicity=ck.get("periodicity", False),
                        fine_dropout=ck.get("fine_dropout", 0.0),
                        prior_kappa=ck.get("prior_kappa", 0.0),
                        prior_beta=ck.get("prior_beta", 0.5),
                        fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()

        tag = Path(ck_path).stem
        if a.postproc != "off":
            tag += "+pp"
        if a.resmatch > 0:
            tag += f"+rm{a.resmatch:g}"
        if a.snap > 0:
            tag += f"+snap{a.snap:g}" + ("nr" if a.snap_no_redist else "")

        per_file: Dict[str, dict] = {}
        for stem in stems:
            if stem not in ev:
                continue
            d = forward_file(model, stem, dev, stride=a.stride)
            P = gated(d, False)
            g = d["gate"]
            if a.postproc != "off":
                from src.model.postproc import apply_postproc
                P, g = apply_postproc(P, g, apps, gate_sync=a.postproc == "sync")
            if a.snap > 0:
                from src.model.postproc import snap_power
                P, g = snap_power(P, g, apps, targets={"beam_projector": float(a.snap)},
                                  redistribute=not a.snap_no_redist)
            if a.resmatch > 0:
                from src.model.postproc import resistive_match
                P, g = resistive_match(P, g, apps, d["p_observed"], d["v_rms"],
                                       d["standby"], d["p_noise"], obs_harm=d["obs_harm"],
                                       tol=a.resmatch, snap=a.rm_snap)
            n_cycles = int(ev[stem]["cycles"])
            Pc = upsample_to_cycles(P, d["targets"], n_cycles)
            Gc = upsample_to_cycles(g > 0.5, d["targets"], n_cycles)
            Po = upsample_to_cycles(d["p_observed"], d["targets"], n_cycles)
            Vc = upsample_to_cycles(d["v_rms"], d["targets"], n_cycles)
            per_file[stem] = score_power_ref(Pc, Gc, stem, apps, events=ev,
                                             p_observed=Po, v_rms=Vc)

        summ = summarize_power_ref(per_file)
        payload[tag] = {"per_file": per_file, "summary": summ}
        print("=" * 88)
        print(f"[{tag}]  {len(stems)}파일")
        print("=" * 88)
        print(format_power_ref(summ))
        print()
        for app in sorted(summ):
            rows = [(s, v[app]) for s, v in per_file.items()
                    if app in v and v[app].get("n_detected", 0) > 0]
            if len(rows) < 2:
                continue
            print(f"  [{app}] 파일별 중앙|오차|W")
            print("    " + "  ".join(f"{s}:{v['median_abs_err_w']:.1f}" for s, v in rows))

    # 규칙 33 — 이 표를 만든 명령을 산출물 안에 남긴다. 플래그가 태그에 안 붙는
    # 것들(`--rm-snap` 등)이 있어, 없으면 나중에 이 파일이 무엇인지 증명 못 한다.
    payload["_config"] = {"argv": sys.argv, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
