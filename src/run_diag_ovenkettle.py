# -*- coding: utf-8 -*-
"""저항성 기기 사이의 배분이 **고조파로 갈리는가** — 실측 파일 하나에서 (14.30).

사용자가 `cnn_sigc_vn_s0` 의 test_5 그림을 보고: *"포트가 켜지기 전에 오븐이 포트로
오탐되고 오븐 통전이 꺼질 때는 포트가 오븐으로 오탐되는 걸로 보이는데 원인이 뭐야?"*
— 둘 다 맞았다. 이 도구가 그것을 재는 자다.

네 가지를 찍는다:

  ① **이벤트 경계 구간표** — 참 조합 대 모델 배분. 총전력이 맞는데 배분만 틀리면 축퇴다
  ② **관문 시각** — 참 사건 대비 이르나 늦나. 관문이 정확한데 와트가 틀리면 배분 문제다
  ③ **지문 각도** — `sig = median(I/P)` 사이각. 작으면 `L_harm` 이 못 가른다
  ④ **몫 훑기** — 총전력을 고정하고 두 기기 사이 몫을 0→1 로 옮기며 고조파 잔차.
     **h1 을 빼고** 재는 것이 핵심이다 — h1 은 총전력이라 이 훑기에서 정보가 아니다.
     변화폭이 몇 %면 **모양으로는 못 가른다**는 뜻이다.

⚠ 판정 기준: 지문 표류가 분 단위로 9~11% 다 ([[signature-drifts-minute-to-minute]]).
   훑기 변화폭이 그보다 작으면 **정보가 잡음 밑에 있다** — 손실을 만져도 못 고친다.

    python -X utf8 -m src.run_diag_ovenkettle --ckpt results/cnn_sigc_vn_s0.pt
    python -X utf8 -m src.run_diag_ovenkettle --stem test_5 --pair oven electiric_kettle
"""
import argparse
import sys

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.losses import S_STATE
from src.model.net import harmonic_signatures
from src.run_gate_check import load_model
from src.run_plot_real import KO, SAMPLING_HZ, dense_targets, load_events, predict
from src.synthesis.genopts import build_synthesizer, resolve

WATCH = ("oven", "electiric_kettle", "hotplate", "hair_dryer")


def _ang(u, v) -> float:
    u, v = np.asarray(u).ravel(), np.asarray(v).ravel()
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _on_at(iv, app, tm) -> bool:
    return any(t0 <= tm <= t1 for t0, t1 in iv.get(app, {}).get("on", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_sigc_vn_s0.pt")
    ap.add_argument("--stem", default="test_5")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--pair", nargs=2, default=["oven", "electiric_kettle"],
                    help="몫을 훑을 두 기기")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, _ = load_model(a.ckpt, dev)
    rw = dense_targets(a.stem, stride=a.stride)
    pred, _standby, gate = predict(model, rw, dev)[:3]
    t = rw.target_cycle / SAMPLING_HZ
    oh = rw.obs_harm
    ev = load_events()[a.stem]
    iv = ev["intervals"]
    evs = sorted(ev.get("events", []), key=lambda e: e["t_s"])
    watch = [x for x in WATCH if x in apps]
    ki = [apps.index(x) for x in watch]

    print("[%s] %s · 창 %d개 · %.1f초 간격"
          % (a.stem, a.ckpt, len(t), a.stride / SAMPLING_HZ))

    # ── ① 이벤트 경계 구간표 ────────────────────────────────────────────
    print("\n## ① 이벤트 경계 구간별 — 참 ON 대 모델 배분 (W)")
    print("  %-15s%4s | %-30s | %s"
          % ("구간(초)", "창", "참 ON", "  ".join("%6s" % KO.get(x, x)[:6] for x in watch)))
    bnd = sorted({0.0, float(t.max())} | {e["t_s"] for e in evs})
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        m = (t >= lo + 1) & (t <= hi - 1)
        if m.sum() < 3:
            continue
        mid = 0.5 * (lo + hi)
        truth = " ".join(KO.get(x, x) if _on_at(iv, x, mid) else "·" * len(KO.get(x, x))
                         for x in watch)
        print("  %6.0f-%-8.0f%4d | %-30s | %s   합 %5.0f  관측 %5.0f"
              % (lo, hi, m.sum(), truth,
                 "  ".join("%6.0f" % pred[m, k].mean() for k in ki),
                 pred[m].sum(1).mean(), rw.p_observed[m].mean()))
    print("  ⇒ 총전력이 맞는데 배분만 틀리면 **잔차 문제가 아니라 축퇴**다.")

    # ── ② 관문 시각 ─────────────────────────────────────────────────────
    print("\n## ② 관문 0.5 교차 시각 대 참 사건  (− = 모델이 이르다)")
    for x in watch:
        k = apps.index(x)
        g = gate[:, k] > 0.5
        cr = [(t[i], "on" if g[i] else "off") for i in range(1, len(t)) if g[i] != g[i - 1]]
        for e in [q for q in evs if q["appliance"] == x and q["kind"] in ("on", "off")]:
            same = [c for c in cr if c[1] == e["kind"]]
            if not same:
                continue
            j = int(np.argmin([abs(c[0] - e["t_s"]) for c in same]))
            dp = e.get("delta_p_w")
            print("  %-10s %-4s 참 %6.1fs  모델 %6.1fs   차 %+6.1fs%s"
                  % (KO.get(x, x), e["kind"], e["t_s"], same[j][0], same[j][0] - e["t_s"],
                     ("   (참 사건 %+.0fW)" % dp) if dp is not None else ""))
    print("  ⇒ 참 사건이 수십 W 면 **검출 자체가 불가능**하다 — 관문 탓으로 읽지 마라.")

    # ── ③ 지문 각도와 슬롯 ──────────────────────────────────────────────
    pool = build_synthesizer(resolve("v32"), "processed_data/npz", "train").pool
    sig = harmonic_signatures(pool, apps)
    print("\n## ③ 지문 각도 (작을수록 못 가른다) · 상태 슬롯")
    for i, x in enumerate(watch):
        for y in watch[i + 1:]:
            print("  %-10s 대 %-10s %6.2f°   |sig| %.4f 대 %.4f"
                  % (KO.get(x, x), KO.get(y, y), _ang(sig[apps.index(x)], sig[apps.index(y)]),
                     np.linalg.norm(sig[apps.index(x)]), np.linalg.norm(sig[apps.index(y)])))
    for x in watch:
        s = S_STATE.get(x, {})
        v = [float(q) for q in s.values()] if isinstance(s, dict) else list(s)
        print("  %-10s 슬롯 %s%s"
              % (KO.get(x, x), [round(q, 1) for q in v],
                 ("   **%.0f배**" % (max(v) / max(min(v), 1e-9))) if len(v) > 1 else ""))
    print("  ⇒ 슬롯 폭이 넓은 기기가 **남는 와트의 기본 행선지**가 된다 (값싼 흡수처).")

    # ── ④ 몫 훑기 ───────────────────────────────────────────────────────
    ka, kb = apps.index(a.pair[0]), apps.index(a.pair[1])
    print("\n## ④ %s <-> %s 몫 훑기 — **h1 제외** (총전력 정보를 뺀 모양만)"
          % (KO.get(a.pair[0], a.pair[0]), KO.get(a.pair[1], a.pair[1])))
    print("  %-16s%5s%9s%12s%12s%10s" % ("구간(초)", "창", "합W", "참 A몫", "잔차 변화폭", "최소 위치"))
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        m = (t >= lo + 1) & (t <= hi - 1)
        if m.sum() < 5:
            continue
        P = pred[m].copy()
        tot = float((P[:, ka] + P[:, kb]).mean())
        if tot < 100:
            continue
        rs = []
        for f in np.linspace(0, 1, 11):
            Q = P.copy()
            s_ = Q[:, ka] + Q[:, kb]
            Q[:, ka], Q[:, kb] = f * s_, (1 - f) * s_
            hp = np.einsum("nk,khc->nhc", Q, sig)
            d = (hp - oh[m])[:, 1:, :]                 # ⚠ h1 제외가 핵심
            rs.append(float(np.linalg.norm(d.reshape(int(m.sum()), -1), axis=1).mean()))
        rs = np.asarray(rs)
        mid = 0.5 * (lo + hi)
        want = ("1.0" if _on_at(iv, a.pair[0], mid) and not _on_at(iv, a.pair[1], mid)
                else "0.0" if _on_at(iv, a.pair[1], mid) and not _on_at(iv, a.pair[0], mid)
                else " ? ")
        print("  %6.0f-%-9.0f%5d%9.0f%12s%11.1f%%%10.1f"
              % (lo, hi, m.sum(), tot, want,
                 100 * (rs.max() - rs.min()) / max(rs.mean(), 1e-12),
                 np.argmin(rs) / 10.0))
    print("  ⇒ 변화폭이 **지문 표류 9~11%% 보다 작으면** 모양으로 못 가른다.")
    print("     그러면 손실 가중치로 고칠 수 없다 — 슬롯·전이·듀티 같은 **다른 축**이 필요하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
