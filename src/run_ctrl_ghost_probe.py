"""대조 파일 유령을 가른다 — `가중+지문` 채택 전 필수 (12.139)

무엇이 걸려 있나 (12.137.1 이 남긴 것)
-------------------------------------
`가중 1/h² + in-situ 지문` 이 프로젝터 과대예측을 +30.0 -> **+7.3W** 로 닫았다.
그런데 **대조 파일 유령이 7.09 -> 11.51W** 로 네 판 중 최악이고 초가법이다.

    지표        가중Δ   지문Δ   가법예측   조합Δ   잡음바닥
    유령대조    +1.13  +0.22   +1.36   +4.42   1.63   <- **초가법 (나쁜 쪽)**

규칙 20 으로 읽으면 **겨냥 파일의 유령 증가(+0.98)는 처방에 귀속 못 한다.**
그러나 **대조가 나빠진 것 자체는 실재한다** — 저항 전용 파일에 SMPS 를 붙이고
있다는 뜻이다. 12.134.1 의 `consQ` 참사(대조 52.38)와 같은 자리, 규모는 1/5.

**재학습 없이 갈 수 있는 데까지 간다.** 12.137.1 이 이미 12개 체크포인트를 한
명령으로 채점해 뒀고(`results/gc_wi.json`, 규칙 33), 그 산출물에 기기별 유령이
그대로 들어 있다 (`score_one` 의 `absent`, 12.90 이 넣은 것).

두 부분
-------
**A. 어디인가** — 순전파조차 없이 `gc_*.json` 만으로 판 × 파일 × 기기 로 가른다.
    시드 3개의 범위를 같이 찍는다. 판끼리 범위가 겹치면 잡음이다 (규칙 20).

**B. 언제인가** — A 가 지목한 셀에서 순전파를 돌려 유령 전력을 **어느 저항이
    켜져 있는가**로 층화한다 (규칙 24). 학습은 없다.

    H1 **헤어드라이어 반파**   유령이 헤어드라이어 ON 구간에 붙는다.
        반파 정류는 짝수차와 DC 를 흘려 낮은 차수에서 SMPS 처럼 보인다.
        `inv_h²` 는 L_harm 을 사실상 h1~h3 로 좁히므로 그 구간에서
        판별이 죽고, 지문이 날카롭게 만든 프로젝터 값이 거기로 간다.
    H2 **총전력 압력**        유령이 관측 총전력·겹침 수에 붙는다 (규칙 29).
        보존 제약이 있는 계에서 오차는 사라지지 않고 옮겨 다닌다.
    H3 **재학습 잡음**        시간 구조가 없다. 시드마다 다른 자리에 뜬다.

    ⚠ **반증 조건을 먼저 적는다.**
      - H1 은 유령의 조건부 평균이 헤어드라이어 ON/OFF 에서 **같으면 죽는다.**
      - H2 는 유령과 관측 총전력의 상관이 **약하면 죽는다.**
      - H3 는 시드 간 유령 시계열 상관이 **높으면 죽는다.**
      셋 다 죽으면 여기서 멈추고, 무엇을 더 재야 하는지만 적는다.

쓰는 법
------
    python -X utf8 -m src.run_ctrl_ghost_probe                    # A + B
    python -X utf8 -m src.run_ctrl_ghost_probe --no-forward       # A 만 (즉시)
    python -X utf8 -m src.run_ctrl_ghost_probe --gc results/gc_wi.json
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

# 대조 3파일 — 정답상 SMPS 가 0종이다 (12.137.1 의 표 관례, `run_sig_conditioning`
# 도 같은 목록을 쓴다). 여기 뜨는 SMPS 전력은 **전부** 유령이다.
CTRL_STEMS = ("test_9", "test_11", "test_12")

# 판 = (표시이름, 체크포인트 stem 3개). 12.137.1 의 네 판 그대로.
PANS: Dict[str, Sequence[str]] = {
    "기준선 w4.0": ("adapt_ovh", "adapt_ovh_s1", "adapt_ovh_s2"),
    "가중 1/h²": ("adapt_hw2_s0", "adapt_hw2_s1", "adapt_hw2_s2"),
    "in-situ 지문": ("adapt_isig3_s0", "adapt_isig3_s1", "adapt_isig3_s2"),
    "가중+지문": ("adapt_wi_s0", "adapt_wi_s1", "adapt_wi_s2"),
}

SUFFIX = "+pp+rm0.02"


# ── A. 어디인가 ──────────────────────────────────────────────────────────────
def _cell(gc: dict, pan: Sequence[str], stem: str, app: str | None) -> np.ndarray:
    """판 하나의 시드별 값 (n_seed,). `app` 이 None 이면 파일 합계."""
    out = []
    for t in pan:
        d = gc[t + SUFFIX][stem]["soft"]
        out.append(d["absent_sum_w"] if app is None else d["absent"][app]["mean_w"])
    return np.array(out, float)


def _fmt(v: np.ndarray) -> str:
    return f"{v.mean():7.2f} [{v.min():5.2f},{v.max():5.2f}]"


def part_a(gc: dict) -> dict:
    """판 × 파일 × 기기 로 대조 유령을 가른다. 지목된 셀을 돌려준다."""
    base = next(iter(PANS))
    combo = list(PANS)[-1]
    # `absent` 는 **그 파일에서 꺼져 있던 기기만** 담는다. 파일마다 다르다.
    apps_of = {s: sorted(gc[PANS[base][0] + SUFFIX][s]["soft"]["absent"])
               for s in CTRL_STEMS}

    print("=" * 96)
    print("A. 어디인가 — 순전파 없이 `gc` 산출물만으로 (판 x 파일 x 기기)")
    print("=" * 96)
    print("\n  [A-1] 대조 파일별 합계 W  — 평균 [시드 최소, 최대]\n")
    print(f"  {'판':<14s}" + "".join(f"{s:>24s}" for s in CTRL_STEMS)
          + f"{'대조평균':>10s}")
    print("  " + "-" * 92)
    for name, pan in PANS.items():
        cells = [_cell(gc, pan, s, None) for s in CTRL_STEMS]
        avg = np.mean([c.mean() for c in cells])
        print(f"  {name:<14s}" + "".join(f"{_fmt(c):>24s}" for c in cells)
              + f"{avg:>10.2f}")

    print("\n  [A-2] 조합이 기준선에서 얼마나 움직였나 — 시드 범위가 겹치면 잡음이다"
          " (규칙 20)\n")
    hits: List[dict] = []
    print(f"  {'파일':<9s}{'기기':<17s}{'기준선':>18s}{'가중':>18s}"
          f"{'지문':>18s}{'조합':>18s}{'가법예측':>10s}{'조합Δ':>9s}  판정")
    print("  " + "-" * 118)
    for stem in CTRL_STEMS:
        for app in apps_of[stem]:
            v = {n: _cell(gc, p, stem, app) for n, p in PANS.items()}
            d_comb = v[combo].mean() - v[base].mean()
            pred = ((v[list(PANS)[1]].mean() - v[base].mean())
                    + (v[list(PANS)[2]].mean() - v[base].mean()))
            # 시드 범위가 안 겹치면 재학습 잡음으로 설명되지 않는다
            sep = v[combo].min() > v[base].max() or v[combo].max() < v[base].min()
            if abs(d_comb) < 0.30 and not sep:
                continue
            verdict = "**분리**" if sep else "겹침(보류)"
            print(f"  {stem:<9s}{app:<17s}"
                  + "".join(f"{_fmt(v[n]):>18s}" for n in PANS)
                  + f"{pred:>+10.2f}{d_comb:>+9.2f}  {verdict}")
            # 분리돼도 **크기가 없으면** 지목하지 않는다. 대조 평균은 3파일
            # 나눗셈이라 0.5W 미만인 셀은 최종 지표에서 0.17W 도 못 움직인다.
            if sep and d_comb > 0.5:
                hits.append({"stem": stem, "app": app, "delta": d_comb,
                             "additive": pred})
    print("\n  `가법예측` = 가중Δ + 지문Δ. 조합Δ 가 그보다 크면 초가법이다 (규칙 32).")

    # [A-3] 최종 지표 +4.42W 를 기기별로 분해한다. 대조 평균은 3파일 평균이므로
    # 각 셀의 기여는 조합Δ / 3 이다. 합이 총 Δ 와 맞아야 분해가 닫힌다.
    tot = (np.mean([_cell(gc, PANS[combo], s, None).mean() for s in CTRL_STEMS])
           - np.mean([_cell(gc, PANS[base], s, None).mean() for s in CTRL_STEMS]))
    by_app: Dict[str, float] = {}
    for stem in CTRL_STEMS:
        for app in apps_of[stem]:
            d = (_cell(gc, PANS[combo], stem, app).mean()
                 - _cell(gc, PANS[base], stem, app).mean())
            by_app[app] = by_app.get(app, 0.0) + d / len(CTRL_STEMS)
    print(f"\n  [A-3] 대조 유령 총 Δ {tot:+.2f}W 의 기기별 분해 (셀Δ / 3파일)\n")
    for app, d in sorted(by_app.items(), key=lambda kv: -abs(kv[1])):
        bar = "#" * int(round(abs(d) / max(1e-9, abs(tot)) * 40))
        print(f"     {app:<17s}{d:>+8.2f}W  ({d / tot * 100:>5.1f}%)  {bar}")
    print(f"     {'합':<17s}{sum(by_app.values()):>+8.2f}W")

    if hits:
        hits.sort(key=lambda h: -h["delta"])
        print("\n  ▶ 지목된 셀 (분리 + 나빠짐):")
        for h in hits:
            print(f"      {h['stem']:<9s}{h['app']:<17s}"
                  f"조합Δ {h['delta']:+.2f}W  (가법예측 {h['additive']:+.2f}W)")
    else:
        print("\n  ▶ 분리되는 셀이 없다. 대조 유령은 재학습 잡음으로 설명된다.")
    return {"hits": hits, "apps_of": apps_of}


# ── B. 언제인가 ──────────────────────────────────────────────────────────────
def _windows(model, stem: str, dev: str, stride: int) -> dict:
    from src.run_gate_check import forward_file, gated
    d = forward_file(model, stem, dev, stride=stride)
    return {"P": gated(d, hard=False), "gate": d["gate"],
            "targets": d["targets"], "pobs": d["p_observed"],
            "idle": d["idle"], "obs_harm": d["obs_harm"]}


# ── C. 유령은 L_harm 을 **줄이는가** ─────────────────────────────────────────
def _harm_weights(harm_weight: str, hsc: np.ndarray) -> tuple:
    """손실이 실제로 쓰는 차수별 무게. `NILMLoss.__init__` 과 같은 식."""
    h = len(hsc)
    hh = np.arange(1, h + 1, dtype=np.float64)
    m = {"off": np.ones(h), "inv_h": 1.0 / hh, "inv_h2": 1.0 / (hh * hh)}[harm_weight]
    m = m / m.max()
    return m, m.mean()


def _l_harm(pred: np.ndarray, obs: np.ndarray, m: np.ndarray, mmean: float,
            hsc: np.ndarray) -> np.ndarray:
    """창별 L_harm (`NILMLoss.forward` 의 `parts['harm']` 을 창 단위로)."""
    err = np.abs(pred - obs) / np.maximum(hsc, 1e-9)[None, :, None]
    return (err * m[None, :, None]).mean((1, 2)) / max(mmean, 1e-6)


def part_c(hits: List[dict], stride: int) -> None:
    """유령을 0 으로 되돌리면 L_harm 이 오르는가 내리는가.

    **오르면** 유령은 손실이 시킨 것이다 — 지문이 실측 고조파를 못 만들어서
    모델이 SMPS 로 메우고 있다는 뜻이고, 그러면 고칠 곳은 적응이 아니라 지문이다.
    **내리면** 유령은 손실을 거스르는 것이고, 원인은 다른 항(L_cons·L_power)이다.
    """
    import torch
    from src.evaluation.real_events import load_events
    from src.model.net import harmonic_scales, harmonic_signatures, noise_signature, standby_signatures
    from src.run_gate_check import load_model
    from src.synthesis.segment_pool import SegmentPool

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = sorted({h["stem"] for h in hits})
    apps0 = None

    print()
    print("=" * 96)
    print("C. 유령은 L_harm 을 **줄이는가** — 손실이 시킨 것인가 거스른 것인가")
    print("=" * 96)

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    for stem in stems:
        print(f"\n  ── {stem}")
        print(f"     {'판':<14s}{'유령W':>8s}{'L_harm(현재)':>14s}{'L_harm(유령0)':>15s}"
              f"{'Δ':>10s}{'유령이 손실을':>16s}")
        print("     " + "-" * 80)
        for name, pan in PANS.items():
            hw = "inv_h2" if "가중" in name else "off"
            use_insitu = "지문" in name
            cur, off, gw = [], [], []
            for ck in pan:
                model, apps, _ = load_model(f"results/{ck}.pt", dev)
                if apps0 is None:
                    apps0 = apps
                    sig_s = harmonic_signatures(pool, apps)
                    sb = standby_signatures(pool, apps)
                    nz = noise_signature(pool)
                    hsc = harmonic_scales(pool, apps)
                    sig_i = np.asarray(
                        np.load("results/sig_insitu.npz", allow_pickle=True)["sig"],
                        np.float32)
                sig = sig_i if use_insitu else sig_s
                m, mm = _harm_weights(hw, hsc)
                w = _windows(model, stem, dev, stride)
                pj = apps.index("beam_projector")
                pred = (np.einsum("bk,khc->bhc", w["P"], sig)
                        + np.einsum("bk,khc->bhc", w["idle"], sb) + nz[None])
                p0 = w["P"].copy(); p0[:, pj] = 0.0
                pred0 = (np.einsum("bk,khc->bhc", p0, sig)
                         + np.einsum("bk,khc->bhc", w["idle"], sb) + nz[None])
                cur.append(_l_harm(pred, w["obs_harm"], m, mm, hsc).mean())
                off.append(_l_harm(pred0, w["obs_harm"], m, mm, hsc).mean())
                gw.append(w["P"][:, pj].mean())
                del model
            c, o, g = np.mean(cur), np.mean(off), np.mean(gw)
            verdict = "**줄인다**" if o > c else "늘린다"
            print(f"     {name:<14s}{g:>8.2f}{c:>14.4f}{o:>15.4f}"
                  f"{c - o:>+10.4f}{verdict:>16s}  와트당 {(c - o) / max(g, 1e-9):+.4f}")
        print("     `L_harm(유령0)` 은 프로젝터 전력만 0 으로 두고 같은 식을 다시 푼 값이다.")

        # ── D. 결손은 **몇 차에** 있나. 유령을 뺀 뒤의 부호 있는 잔차를 차수별로.
        #    양수 = 관측이 지문보다 많다 = 지문이 실측 고조파를 못 만든다.
        model, apps, _ = load_model(f"results/{PANS['기준선 w4.0'][0]}.pt", dev)
        w = _windows(model, stem, dev, stride)
        pj = apps.index("beam_projector")
        p0 = w["P"].copy(); p0[:, pj] = 0.0
        pred0 = (np.einsum("bk,khc->bhc", p0, sig_s)
                 + np.einsum("bk,khc->bhc", w["idle"], sb) + nz[None])
        r = np.linalg.norm(w["obs_harm"], axis=2) - np.linalg.norm(pred0, axis=2)
        del model
        from src.evaluation.real_events import build_on_off_truth
        on, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), ev)
        idx = np.clip(w["targets"], 0, len(on) - 1)
        res = {a: on[idx, j] for j, a in enumerate(apps)
               if a in ("hair_dryer", "hotplate", "oven", "electiric_kettle")
               and on[idx, j].any()}
        print(f"\n     [D] 차수별 결손 mA — |관측| − |지문 예측(유령0)|. **양수면 지문이"
              f" 모자란다**")
        print(f"     {'차수':<6s}{'harm_scale':>11s}{'1/h² 무게':>10s}{'전체':>9s}"
              + "".join(f"{a[:9] + ' ON':>14s}" for a in res)
              + f"{'프로젝터 지문':>13s}")
        m2, _ = _harm_weights("inv_h2", hsc)
        for h in range(len(hsc)):
            row = (f"     h{h + 1:<5d}{hsc[h]:>11.4f}{m2[h]:>10.4f}"
                   f"{r[:, h].mean() * 1000:>9.2f}")
            for a, msk in res.items():
                row += f"{r[msk, h].mean() * 1000:>14.2f}"
            print(row + f"{np.linalg.norm(sig_s[pj, h]) * 1000:>13.3f}")
        print("     `프로젝터 지문` 은 A/W 단위 x1000 — 유령 1W 가 그 차수에 넣는 mA 다.")

        # ── E. **어느 차수가 유령 값을 치르나.** ΔL_harm 을 차수로 쪼갠다.
        #    음수 = 그 차수에서 유령이 손실을 줄인다 = 그 차수가 유령을 부른다.
        print(f"\n     [E] 유령의 ΔL_harm 을 차수별로 — 음수인 차수가 유령을 부른다\n")
        print(f"     {'차수':<6s}" + "".join(f"{n:>16s}" for n in PANS))
        rows = {n: None for n in PANS}
        for name, pan in PANS.items():
            hw = "inv_h2" if "가중" in name else "off"
            sg = sig_i if "지문" in name else sig_s
            m, mm = _harm_weights(hw, hsc)
            acc = []
            for ck in pan:
                model, apps, _ = load_model(f"results/{ck}.pt", dev)
                w = _windows(model, stem, dev, stride)
                pj = apps.index("beam_projector")
                base = (np.einsum("bk,khc->bhc", w["idle"], sb) + nz[None])
                pred = np.einsum("bk,khc->bhc", w["P"], sg) + base
                p0 = w["P"].copy(); p0[:, pj] = 0.0
                pred0 = np.einsum("bk,khc->bhc", p0, sg) + base
                e = np.abs(pred - w["obs_harm"]) / np.maximum(hsc, 1e-9)[None, :, None]
                e0 = np.abs(pred0 - w["obs_harm"]) / np.maximum(hsc, 1e-9)[None, :, None]
                acc.append(((e - e0) * m[None, :, None]).mean(2).mean(0)
                           / (len(hsc) * max(mm, 1e-6)))
                del model
            rows[name] = np.mean(acc, 0)
        for h in range(len(hsc)):
            print(f"     h{h + 1:<5d}" + "".join(f"{rows[n][h]:>+16.4f}" for n in PANS))
        print(f"     {'합':<6s}" + "".join(f"{rows[n].sum():>+16.4f}" for n in PANS))


def part_b(hits: List[dict], stride: int, gate_min: float) -> None:
    import torch
    from src.evaluation.real_events import build_on_off_truth, load_events
    from src.run_gate_check import load_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = sorted({h["stem"] for h in hits})
    apps_of_interest = sorted({h["app"] for h in hits})

    print()
    print("=" * 96)
    print(f"B. 언제인가 — 순전파 {sum(len(p) for p in PANS.values())}회 x {len(stems)}"
          f"파일 (학습 없음, dev={dev}, stride={stride})")
    print("=" * 96)

    for stem in stems:
        n_cycles = int(ev[stem]["cycles"])
        present = [a for a in ev[stem]["appliances_present"]]
        print(f"\n  ── {stem} — 실제로 켜진 것: {', '.join(present)}")

        truth_cache: dict = {}
        series: Dict[str, List[np.ndarray]] = {}
        pobs_ref = None
        for name, pan in PANS.items():
            for ck in pan:
                model, apps, _ = load_model(f"results/{ck}.pt", dev)
                w = _windows(model, stem, dev, stride)
                if not truth_cache:
                    on, _ = build_on_off_truth(stem, apps, n_cycles, ev)
                    idx = np.clip(w["targets"], 0, len(on) - 1)
                    truth_cache = {a: on[idx, j] for j, a in enumerate(apps)}
                    pobs_ref = w["pobs"]
                for app in apps_of_interest:
                    j = apps.index(app)
                    # 게이트가 확실히 켜졌다고 본 창만 — 소프트 전력은 망설임까지
                    # 섞여 '언제' 를 흐린다 (12.39 의 hedge 와 같은 이유).
                    series.setdefault(f"{name}|{app}", []).append(
                        w["P"][:, j] * (w["gate"][:, j] > gate_min))
                del model

        res_on = {a: v for a, v in truth_cache.items()
                  if a in ("hair_dryer", "hotplate", "oven", "electiric_kettle")
                  and v.any()}

        for app in apps_of_interest:
            print(f"\n     [{app}] 유령 W — 저항 ON/OFF 로 층화 (규칙 24)")
            head = f"     {'판':<14s}{'전체':>9s}"
            for r in res_on:
                head += f"{r[:9] + ' ON':>14s}{'OFF':>9s}"
            print(head)
            print("     " + "-" * (len(head) - 5))
            for name in PANS:
                arrs = series[f"{name}|{app}"]
                x = np.mean(np.stack(arrs), 0)          # 시드 평균 시계열
                row = f"     {name:<14s}{x.mean():>9.2f}"
                for r, m in res_on.items():
                    row += f"{x[m].mean():>14.2f}{x[~m].mean():>9.2f}"
                print(row)

            # H2 — 관측 총전력과의 상관 / H3 — 시드 간 상관
            print(f"\n     {'판':<14s}{'corr(유령,P관측)':>18s}"
                  f"{'시드간 상관 중앙':>18s}{'유령>1W 창 비율':>18s}")
            for name in PANS:
                arrs = np.stack(series[f"{name}|{app}"])
                x = arrs.mean(0)
                c_obs = (np.corrcoef(x, pobs_ref[:len(x)])[0, 1]
                         if x.std() > 1e-9 else float("nan"))
                cs = [np.corrcoef(arrs[i], arrs[k])[0, 1]
                      for i in range(len(arrs)) for k in range(i + 1, len(arrs))
                      if arrs[i].std() > 1e-9 and arrs[k].std() > 1e-9]
                cmed = float(np.median(cs)) if cs else float("nan")
                print(f"     {name:<14s}{c_obs:>18.3f}{cmed:>18.3f}"
                      f"{float((x > 1.0).mean()):>18.3f}")

        # ── 규칙 6 — 층화 변수가 서로 상관된다. 교차표 없이는 H1/H2 를 못 가른다.
        for app in apps_of_interest:
            x = {n: np.mean(np.stack(series[f"{n}|{app}"]), 0) for n in PANS}
            _crosstab(stem, app, x, res_on, pobs_ref)
            _power_matched(stem, app, x, res_on, pobs_ref)

    print("\n  판정 규칙 (미리 적은 것):")
    print("    H1 헤어드라이어 반파 — ON/OFF 조건부 평균이 같으면 죽는다")
    print("    H2 총전력 압력      — corr(유령, P관측) 이 약하면 죽는다")
    print("    H3 재학습 잡음      — 시드 간 상관이 높으면 죽는다")


def _crosstab(stem: str, app: str, x: Dict[str, np.ndarray],
              res_on: Dict[str, np.ndarray], pobs: np.ndarray) -> None:
    """저항 조합별 교차표 (규칙 6). 어느 저항이 켜졌는지가 층이다."""
    names = list(res_on)
    key = np.stack([res_on[r] for r in names], 1)          # (n, R) bool
    combos, inv = np.unique(key, axis=0, return_inverse=True)
    order = np.argsort([-int((inv == i).sum()) for i in range(len(combos))])
    print(f"\n     [{app}] 교차표 — 저항 조합별 (규칙 6). 창 8개 미만은 생략\n")
    print(f"     {'저항 조합':<34s}{'창':>5s}{'P관측':>9s}"
          + "".join(f"{n:>13s}" for n in PANS))
    print("     " + "-" * (34 + 14 + 13 * len(PANS)))
    for i in order:
        m = inv == i
        if m.sum() < 8:
            continue
        lab = "+".join(n[:6] for n, b in zip(names, combos[i]) if b) or "(전부 OFF)"
        print(f"     {lab:<34s}{int(m.sum()):>5d}{pobs[:len(m)][m].mean():>9.0f}"
              + "".join(f"{x[n][m].mean():>13.2f}" for n in PANS))


def _power_matched(stem: str, app: str, x: Dict[str, np.ndarray],
                   res_on: Dict[str, np.ndarray], pobs: np.ndarray) -> None:
    """총전력을 맞춘 뒤에도 기기별 차이가 남는가 (규칙 5).

    H2 가 참이면 같은 P관측 구간 안에서는 어느 저항이든 유령이 같아야 한다.
    남으면 **기기의 파형**이 원인이고 총전력은 교란변수다.
    """
    n = len(next(iter(x.values())))
    p = pobs[:n]
    edges = np.quantile(p, [0, .25, .5, .75, 1.0])
    combo = list(PANS)[-1]
    print(f"\n     [{app}] 총전력을 맞춘 뒤 (규칙 5) — `{combo}` 유령 W, 사분위 안에서\n")
    print(f"     {'P관측 구간':<20s}{'창':>5s}"
          + "".join(f"{r[:9] + ' ON':>14s}{'OFF':>8s}{'비':>7s}" for r in res_on))
    print("     " + "-" * (25 + 29 * len(res_on)))
    for b in range(4):
        m = (p >= edges[b]) & (p <= edges[b + 1] if b == 3 else p < edges[b + 1])
        if m.sum() < 8:
            continue
        row = f"     {f'{edges[b]:.0f}~{edges[b + 1]:.0f}W':<20s}{int(m.sum()):>5d}"
        for r, rm in res_on.items():
            a, o = m & rm[:n], m & ~rm[:n]
            va = x[combo][a].mean() if a.sum() >= 4 else float("nan")
            vo = x[combo][o].mean() if o.sum() >= 4 else float("nan")
            ratio = va / vo if vo > 1e-6 else float("nan")
            row += f"{va:>14.2f}{vo:>8.2f}{ratio:>7.1f}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc", default="results/gc_wi.json",
                    help="run_gate_check 산출물 (12개 체크포인트 한 명령, 규칙 33)")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--gate-min", type=float, default=0.5)
    ap.add_argument("--no-forward", action="store_true", help="A 만 돌린다")
    a = ap.parse_args()

    gc = json.loads(Path(a.gc).read_text(encoding="utf-8"))
    missing = [t + SUFFIX for p in PANS.values() for t in p if t + SUFFIX not in gc]
    if missing:
        print(f"  ⚠ {a.gc} 에 없는 태그: {missing}")
        return 1
    print(f"  채점 산출물: {a.gc}")
    print(f"  그것을 만든 명령: {' '.join(gc.get('_config', {}).get('argv', ['(없음)'])[1:])}\n")

    res = part_a(gc)
    if not a.no_forward and res["hits"]:
        part_b(res["hits"], a.stride, a.gate_min)
        part_c(res["hits"], a.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
