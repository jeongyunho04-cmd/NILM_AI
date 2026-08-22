"""
실측 정답 감사 — 서로 독립인 세 소스를 맞댄다
================================================
12.25절에서 미니PC 라벨이 신호와 모순되는 것을 찾았다. 나머지도 같은 수준으로
훑으려면 **손으로 읽은 타임라인 말고 다른 근거**가 필요하다. 셋을 나란히 놓는다.

    ① 라벨          `real_events.json` (사람이 신호를 보고 적은 것)
    ② GBM           `baseline_gbm.pkl` — CNN 과 **다른 모델 가족·다른 특징·다른 창**
    ③ 지문 정합기    학습된 모델이 아니라, **개별 녹화에서 잰 기기별 지문**으로
                    신호의 계단을 직접 맞춘다

**②는 정답이 아니다.** GBM 도 같은 합성 데이터로 학습됐으므로 sim-to-real 격차를
그대로 물려받는다. 그것으로 라벨을 "검증" 하면 순환논증이다. 쓸모는 다른 데 있다 —
**CNN 과 GBM 이 갈리는 구간은 신호가 애매하다는 뜻**이라 사람이 볼 자리를 짚어 준다.

**③이 유일하게 모델과 독립이다.** 계단마다 (ΔP, ΔI3) 를 재고, 개별 녹화에서 측정한
기기별 (P, I3) 와 맞춘다. 합성기도 학습도 거치지 않는다.

    python -m src.run_label_audit
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.preprocessing import load_nilm_npz

# 개별 녹화가 있는 기기만 지문을 잴 수 있다
RECORDINGS: Dict[str, List[str]] = {
    "minipc": ["minipc_1", "minipc_2", "minipc_3"],
    "beam_projector": ["beam_projector", "beam_projector_2"],
    "laptop_charger": ["laptop_charger_1", "laptop_charger_2"],
    "fan": ["fan_1", "fan_2", "fan_3"],
    "hotplate": ["hotplate_1", "hotplate_2"],
    "oven": ["oven", "oven_2"],
    "electiric_kettle": ["electiric_kettle"],
    "hair_dryer": ["hair_dryer_1", "hair_dryer_2"],
    "air_conditioner": ["air_conditioner"],
}
KOR = {"oven": "오븐", "hotplate": "핫플레이트", "electiric_kettle": "전기포트",
       "hair_dryer": "헤어드라이기", "minipc": "미니PC", "beam_projector": "빔프로젝터",
       "laptop_charger": "노트북충전기", "fan": "선풍기", "air_conditioner": "에어컨"}


def device_signatures() -> Dict[str, dict]:
    """개별 녹화에서 기기별 (P, I3) 를 잰다. **켜짐 전이가 만드는 변화량**의 기준이다."""
    out = {}
    for app, stems in RECORDINGS.items():
        acc = []
        for st in stems:
            path = Path("processed_data/npz") / f"{st}.npz"
            if not path.exists():
                continue
            r = load_nilm_npz(path)
            pf, hc = r["power_features"], r["harmonics_complex"]
            p = pf[:, 0]
            m = p > max(0.4 * np.percentile(p, 95), 5.0)
            if m.sum() < 300:
                continue
            acc.append((float(np.median(p[m])), float(np.median(np.abs(hc[m])[:, 2]))))
        if acc:
            out[app] = {"P": float(np.median([a[0] for a in acc])),
                        "I3": float(np.median([a[1] for a in acc]))}
    return out


# 이 값을 넘는 계단은 "고부하 저항" 으로 보고 I3 정합을 포기한다 (아래 주석).
BIG_W = 300.0
SMPS_I3 = 0.05      # 이보다 큰 I3 를 가진 기기를 SMPS 무리로 본다


def match_step(dp: float, di3: float, sig: Dict[str, dict],
               tol_p: float = 0.45, tol_i3: float = 0.60) -> List[Tuple[str, float]]:
    """계단 하나를 기기 지문에 맞춘다. 상대오차가 작은 순으로 후보를 돌려준다.

    [ΔI3 의 부호를 지켜야 한다]
    첫 판은 `abs(di3)` 로 맞췄는데 **물리적으로 틀렸다.** 켜짐이면 그 기기의 I3 가
    더해지므로 ΔI3 > 0 이어야 하고, 꺼짐이면 음수여야 한다. 절대값으로 보면
    켜짐/꺼짐이 구분되지 않는다.

    [고부하 저항이 켜지면 ΔI3 가 오히려 음수가 된다]
    1,100W 가 붙으면 계통 임피던스 때문에 전압이 7~10V 떨어지고(0.4절), 그러면
    **이미 켜져 있던 SMPS 들의 전류가 줄어든다.** 실측에서 오븐 ON 의 ΔI3 가
    −0.009 ~ −0.034 로 나오는 이유다. 이 효과의 크기는 그때 켜져 있는 SMPS 양에
    달려 있어 지문만으로는 모델링할 수 없다.

    그래서 |ΔP| > `BIG_W` 이면 **I3 는 무리 판정에만 쓴다** — SMPS 무리 기기를
    후보에서 빼고, 저항 4종은 전력으로만 고른다. 저항 4종끼리는 전력이 유일한
    단서이고 오븐(1146W) vs 포트(1260W)는 9% 차이라 이 정합기로도 안 갈린다.
    **그것이 이 과제의 본질적 난점이고**(12.15.3), 정합기의 결함이 아니다.
    """
    cands = []
    for app, s in sig.items():
        if s["P"] < 1e-6:
            continue
        ep = abs(abs(dp) - s["P"]) / s["P"]
        if ep > tol_p:
            continue
        if abs(dp) > BIG_W:
            # 고부하 구간: SMPS 기기는 애초에 이만한 전력을 못 내므로 P 로 이미
            # 걸러진다. I3 는 "SMPS 가 켜진 것은 아니다" 만 확인한다.
            if s["I3"] >= SMPS_I3 and abs(di3) < s["I3"] * 0.5:
                continue
            cands.append((app, float(ep)))
            continue
        expected = (1.0 if dp > 0 else -1.0) * s["I3"]
        denom = max(s["I3"], 0.02)
        ei = abs(di3 - expected) / denom
        if ei <= tol_i3:
            cands.append((app, float(np.hypot(ep, ei))))
    return sorted(cands, key=lambda x: x[1])


def signal_events(stem: str, sig: Dict[str, dict], thr_w: float = 6.0) -> List[dict]:
    """신호의 계단을 뽑고 각각을 기기에 맞춰 본다 (모델 없음)."""
    r = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
    pf, hc = r["power_features"], r["harmonics_complex"]
    p, i3 = pf[:, 0], np.abs(hc[:, 2])
    nsec = len(p) // 60
    ps = np.array([np.median(p[i * 60:(i + 1) * 60]) for i in range(nsec)])
    i3s = np.array([np.median(i3[i * 60:(i + 1) * 60]) for i in range(nsec)])
    dp, di = np.diff(ps), np.diff(i3s)
    out = []
    for t in range(len(dp)):
        if abs(dp[t]) < thr_w:
            continue
        m = match_step(dp[t], di[t], sig)
        out.append({"t_s": t + 1, "dP": float(dp[t]), "dI3": float(di[t]),
                    "edge": "ON" if dp[t] > 0 else "OFF",
                    "candidates": [{"app": a, "err": round(e, 3)} for a, e in m[:3]]})
    return out


def gbm_predict(stem: str, model, window: int = 600) -> Optional[dict]:
    """GBM 을 실측 파일에 돌린다. 학습 때와 같은 10초 창·같은 타깃 위치를 쓴다."""
    from src.baseline.features import extract
    from src.model.realdata import RealWindows

    r = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
    x = RealWindows._to_33ch(r)                       # (33, N)
    n = x.shape[1]
    ti_in_win = window - 1 - 60                       # 학습 때 target_index=539
    lo, hi = ti_in_win, n - (window - 1 - ti_in_win)
    if hi <= lo:
        return None
    targets = np.arange(lo, hi, 30, dtype=np.int64)
    cut = np.stack([x[:, t - ti_in_win:t - ti_in_win + window] for t in targets])
    feats = extract(cut, target_index=ti_in_win)
    power, prob = model.predict(feats)
    return {"targets": targets, "power": power, "on": prob > 0.5, "apps": model.appliances}


def main() -> int:
    ap = argparse.ArgumentParser(description="실측 정답 감사 — 라벨 / GBM / 지문정합기")
    ap.add_argument("--gbm", default="results/baseline_gbm.pkl")
    ap.add_argument("--cnn", default="results/adapt_v17.pt")
    ap.add_argument("--out", default="results/label_audit.json")
    ap.add_argument("--thr-w", type=float, default=6.0)
    a = ap.parse_args()

    sig = device_signatures()
    print("=" * 92)
    print("[기기 지문] 개별 녹화에서 측정 — 지문 정합기의 기준")
    print("=" * 92)
    for app, s in sorted(sig.items(), key=lambda kv: -kv[1]["P"]):
        print(f"  {KOR.get(app, app):12s} P={s['P']:8.1f}W   I3={s['I3']:.4f}A")

    import torch
    from src.baseline.train import BaselineModel
    from src.model.realdata import upsample_to_cycles
    from src.run_gate_check import forward_file, load_model

    gbm = BaselineModel.load(a.gbm)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cnn, apps, _ = load_model(a.cnn, dev)
    ev = load_events()
    payload: Dict[str, dict] = {"signatures": sig}

    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        n_cycles = int(ev[stem]["cycles"])
        present = set(ev[stem].get("appliances_present", []))
        print("\n" + "=" * 92)
        print(f"[{stem}]  실제 있는 기기: {', '.join(KOR.get(x, x) for x in present)}")
        print("=" * 92)

        evs = signal_events(stem, sig, thr_w=a.thr_w)
        g = gbm_predict(stem, gbm)
        d = forward_file(cnn, stem, dev, stride=30)
        cnn_on = upsample_to_cycles(d["gate"] > 0.5, d["targets"], n_cycles)
        gbm_on = upsample_to_cycles(g["on"], g["targets"], n_cycles) if g else None

        print(f"\n  ── 켜짐 비율 (파일 전체) ─────────────────────────────────")
        print(f"    {'기기':14s}{'라벨':>9s}{'GBM':>9s}{'CNN':>9s}{'일치도':>9s}")
        rows = {}
        for app in [x for x in gbm.appliances]:
            iv = ev[stem]["intervals"].get(app, {})
            lab = np.zeros(n_cycles, bool)
            for s0, s1 in iv.get("on", []):
                lab[int(s0 * 60):int(s1 * 60)] = True
            j = apps.index(app)
            jg = g["apps"].index(app) if g else None
            c = cnn_on[:, j]
            gg = gbm_on[:, jg] if gbm_on is not None else np.zeros(n_cycles, bool)
            if not (lab.any() or c.any() or gg.any()):
                continue
            agree = float((c == gg).mean())
            rows[app] = {"label": float(lab.mean()), "gbm": float(gg.mean()),
                         "cnn": float(c.mean()), "cnn_gbm_agree": agree,
                         "present": app in present}
            mark = "" if app in present else "  <- 없는 기기"
            print(f"    {KOR.get(app, app):14s}{100*lab.mean():>8.1f}%{100*gg.mean():>8.1f}%"
                  f"{100*c.mean():>8.1f}%{100*agree:>8.1f}%{mark}")

        print(f"\n  ── 지문 정합기: 신호 계단 -> 기기 (모델 없음, |ΔP|>{a.thr_w:.0f}W) ──")
        named = [e for e in evs if e["candidates"]]
        print(f"    계단 {len(evs)}개 중 {len(named)}개가 기기 지문과 맞았다")
        for e in named:
            best = e["candidates"][0]
            print(f"      t={e['t_s']:4d}s  {e['edge']:3s}  ΔP {e['dP']:+8.1f}W  ΔI3 {e['dI3']:+.4f}"
                  f"   -> {KOR.get(best['app'], best['app'])} (오차 {best['err']:.2f})"
                  + ("" if best["app"] in present else "   ⚠ 이 파일에 없는 기기"))
        payload[stem] = {"rates": rows, "signal_events": evs}

    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
