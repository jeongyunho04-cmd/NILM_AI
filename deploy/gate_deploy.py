# -*- coding: utf-8 -*-
"""배포 묶음 관문 — **묶음과 연구 갈래가 같은 답을 내나** (14.378).

왜 이게 있나: 이 묶음은 2026-09-02 판 사본을 들고 15일을 있었고 그동안
```
  세밀 채널 50 대 **61** · 링버퍼 33 대 **49** · 조합 머리 **없음** · Ĝ 추정기 **없음**
```
으로 갈렸는데 **아무도 못 봤다.** 사본을 없앴으니 이제 관문이 그것을 지킨다
([[verify-the-input-path-not-just-the-model]] · [[the-gate-must-build-the-real-object]]).

    python -X utf8 deploy/gate_deploy.py [--ckpt deploy/models/xxx.pt]
"""
from pathlib import Path
import argparse
import csv
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-52s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "deploy/models/cnn_comb_s0.pt"))
    ap.add_argument("--stem", default="test_1")
    ap.add_argument("--csv", default=str(ROOT / "data/test_1.csv"))
    a = ap.parse_args()
    print("배포 묶음 관문 (14.378) — %s" % Path(a.ckpt).name)

    # [1] 미러가 연구 코드와 **바이트 동일**한가
    r = subprocess.run([sys.executable, "-X", "utf8",
                        str(ROOT / "deploy/sync_runtime.py"), "--check"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    chk(1, "★ 묶음이 연구 코드와 **바이트 동일**한가", r.returncode == 0,
        (r.stdout or "").strip().splitlines()[-1] if r.stdout else "(출력 없음)")

    # [2] CSV -> 49채널이 **캐시와 같은가**
    import torch
    from src.model.inputs import csv_columns, row_to_channels, RAW_CHANNELS
    from src.model.realdata import RealWindows
    z = np.load(ROOT / ("processed_data/composite_eval/%s.npz" % a.stem), allow_pickle=True)
    ref = RealWindows._to_33ch(z).astype(np.float64)
    kn = np.asarray(z["seq"], np.int64) * 1000 + np.asarray(z["cycle"], np.int64)
    X, ORD = {}, []
    with open(a.csv, encoding="utf-8", newline="") as f:
        rr = csv.reader(f); hdr = next(rr); col = csv_columns(hdr)
        for row in rr:
            x = row_to_channels(row, col)
            if x is None:
                continue
            try:
                k = int(row[col["seq"]]) * 1000 + int(row[col["cycle"]])
            except (ValueError, KeyError, IndexError):
                continue
            X[k] = x
            ORD.append((k, float(row[col["t_s"]])))
    hit = np.array([k in X for k in kn])
    idx = np.where(hit)[0]
    G = np.stack([X[kn[i]] for i in idx]).T.astype(np.float64)
    d = np.abs(G - ref[:, idx])
    chk(2, "★ CSV -> **원시 %d채널**이 학습 캐시와 같은가" % RAW_CHANNELS,
        float(d.max()) < 1e-5 and hit.all(),
        "맞춘 사이클 %d/%d · 전류 %.2e · P/Q/V %.2e · **V_h %.2e** (float32 반올림 자리)"
        % (hit.sum(), len(kn), d[0:30].max(), d[30:33].max(), d[33:RAW_CHANNELS].max()))

    # [3] ★★ 링버퍼가 **순서 뒤바뀜**을 제자리에 꽂나 (사용자 질문, 14.378)
    sys.path.insert(0, str(ROOT / "deploy"))
    from nilm_runtime.predictor import CycleRing, WINDOW_CYCLES, CYCLE_HZ
    ORD.sort(key=lambda t: t[1])
    base = ORD[:WINDOW_CYCLES + 600]
    want = np.stack([X[k] for k, _ in base[-WINDOW_CYCLES:]]).T

    def feed(seq):
        ring = CycleRing(WINDOW_CYCLES, use_time=True)
        for k, t in seq:
            ring.push(X[k], t)
        return ring

    rows = []
    rng = np.random.default_rng(0)
    for tag, seq in (
            ("순서대로", list(base)),
            ("프레임 2장 뒤바꿈 (1초)", None),
            ("프레임 9장 역전 (4.5초)", None),
            ("실측 최악 재현 (59사이클)", None)):
        s = list(base)
        if tag.startswith("프레임 2"):
            for i in range(60, len(s) - 60, 600):
                s[i:i + 30], s[i + 30:i + 60] = s[i + 30:i + 60], s[i:i + 30]
        elif tag.startswith("프레임 9"):
            for i in range(60, len(s) - 300, 900):
                blk = [s[i + j * 30:i + (j + 1) * 30] for j in range(9)]
                rng.shuffle(blk)
                s[i:i + 270] = [x for b in blk for x in b]
        elif tag.startswith("실측"):
            #: ⚠ 처음에 `s[i:i+59], s[i+59:i+60] = ...` 로 적었다가 **행을 58개
            #  지웠다** (왼쪽 59칸에 1칸을 넣는 슬라이스 대입이다). 관문이 4.0W 로
            #  실패했고 원인은 링버퍼가 아니라 **내가 지은 자**였다. 회전으로 고친다 —
            #  길이가 안 변해야 "뒤바뀜"이지 "유실"이 아니다.
            for i in range(120, len(s) - 120, 700):
                s[i:i + 60] = [s[i + 59]] + s[i:i + 59]
        if len(s) != len(base):
            raise AssertionError("섞기가 행 수를 바꿨다: %d -> %d (뒤바뀜이 아니라 유실이다)"
                                 % (len(base), len(s)))
        ring = feed(s)
        got = ring.window()
        dd = float(np.abs(got - want).max())
        rows.append("%s 최대차 %.1e (되꽂음 %d · 늦음 %d)"
                    % (tag, dd, ring.n_back, ring.n_stale))
        if tag == "순서대로":
            ok_base = dd == 0.0
        else:
            ok_base = ok_base and dd == 0.0
    chk(3, "★★ **뒤바뀐 순서를 제자리에 꽂나** (t_s 로 자리 결정)", ok_base,
        " · ".join(rows) + "  — 도착 순서가 아니라 `t_s`(보드 seq) 로 꽂아서 "
        "**창이 바이트까지 같다**")

    # [4] ★★ 묶음이 지은 모델 == 연구 채점기가 지은 모델
    import src.run_gate_check as GC
    from nilm_runtime import NILMPredictor
    GC.sync_even_median(a.ckpt)
    m_r, apps_r, ck = GC.load_model(a.ckpt, "cpu")
    pred = NILMPredictor(a.ckpt, device="cpu")
    s_r, s_d = m_r.state_dict(), pred.model.state_dict()
    bad = [k for k in s_r if k not in s_d or not torch.equal(s_r[k], s_d[k])]
    same_cfg = (float(getattr(m_r, "comb_tau", 0)) == float(pred.comb_tau)
                and float(m_r.comb_over) == float(pred.model.comb_over)
                and list(apps_r) == list(pred.appliances))
    chk(4, "★★ **같은 모델**이 서나 (가중치·조합머리·꺾기)",
        not bad and same_cfg,
        "가중치 다른 키 %s · comb_tau %.2f · **꺾기 %.3f** · 기기 %d개"
        % (bad or "없음", pred.comb_tau, float(pred.model.comb_over), len(pred.appliances)))

    # [5] ★★ Ĝ 가 연구 갈래와 같은가
    from src.model.realdata import dense_targets
    from src.run_plot_real import solve_ghat
    rw = dense_targets(a.stem, stride=30, site_transfer=None)
    g_res, _v = solve_ghat(a.stem, rw, list(apps_r))
    tc = np.asarray(rw.target_cycle, np.int64)
    off = pred.target_in_window
    g_dep, keep = [], []
    for i, t in enumerate(tc):
        lo = t - off
        if lo < 0 or lo + WINDOW_CYCLES > ref.shape[1]:
            continue
        g_dep.append(pred._ghat(ref[None, :, lo:lo + WINDOW_CYCLES]))
        keep.append(i)
        if len(keep) >= 60:
            break
    g_dep = np.asarray(g_dep); g_res_k = np.asarray(g_res)[keep]
    dg = np.abs(g_dep - g_res_k)
    chk(5, "★★ **Ĝ 가 연구 갈래와 같은가** (창 %d개)" % len(keep),
        float(np.median(dg)) < 0.05 and float(dg.max()) < 0.30,
        "차 중앙 **%.4f mS** · p90 %.4f · 최대 %.4f — 기준전압이 *창 중앙값* 대 "
        "*녹화 전체 중앙값*이라 0 이 아니다. 자: Ĝ 잡음 σ 0.265 · 포트 여유 0.635"
        % (np.median(dg), np.percentile(dg, 90), dg.max()))

    # [6] ★★ 끝단 — 같은 창을 넣으면 같은 전력이 나오나 (후처리 **끄고**)
    from src.model.inputs import build_inputs
    t0 = tc[keep[10]]
    win = ref[None, :, t0 - off:t0 - off + WINDOW_CYCLES]
    fine, wide = build_inputs(win)
    gh = torch.full((1,), pred._ghat(win), dtype=torch.float32)
    with torch.no_grad():
        o_r = m_r(torch.from_numpy(fine), torch.from_numpy(wide), g_hat=gh)
        o_d = pred.model(torch.from_numpy(fine), torch.from_numpy(wide), g_hat=gh)
    dp = float((o_r["power"] - o_d["power"]).abs().max())
    chk(6, "★★ **같은 창 -> 같은 전력**인가", dp == 0.0,
        "최대차 **%.3e W** · 총전력 %.1fW · 조합 무게 최대 %.3f"
        % (dp, float(o_r["power"].sum()), float(o_d["comb_w"].max())))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
