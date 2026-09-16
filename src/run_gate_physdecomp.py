# -*- coding: utf-8 -*-
"""① 물리 분해 · ② 정밀화의 관문 — **참값으로** 잰다 (14.318).

이건 **모델에 안 붙은** 물건이다. 붙이기 전에 값이 쓸모 있는지부터 본다.

핵심 질문 하나: **`Ĝ_sum` 이 진짜 저항 컨덕턴스 합을 되찾나.**
합성이라 참값을 안다 — `G_true = Σ_{저항 기기} P_k / |V₁|²`.

    python -X utf8 -m src.run_gate_physdecomp [--cache cache/seqraw_k]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.model.inputs import N_HARM, VOLT_IM0, VOLT_RE0
from src.model.net import harmonic_scales, harmonic_signatures_by_state
from src.model.physdecomp import decompose, refine, resistive_templates

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
#: V_EXP 2.0 인 넷 (`--hcond-classes RESISTIVE` 와 **같은 집합**)
RESISTIVE = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-40s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_k")
    ap.add_argument("--records", type=int, default=60)
    a = ap.parse_args()
    C = Path(a.cache)
    m = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    tgt, wc = int(m["target_offset"]), int(m["window_cycles"])
    grid = np.arange(tgt, int(m["record_s"] * 60) - 13 * 60 - 1, int(m["grid_s"] * 60))
    raw = np.load(C / "raw.npy", mmap_mode="r")
    yp = np.load(C / "y_power.npy", mmap_mode="r")
    R, P = [], []
    for i in range(min(a.records, len(raw))):
        r = np.asarray(raw[i], np.float64)
        for t, c in enumerate(grid[::9]):
            s = r[:, c - tgt:c - tgt + wc]
            if s.shape[1] == wc:
                R.append(s[:, ::30])                 # 2Hz 로 솎는다 (광역 격자와 같다)
                P.append(np.asarray(yp[i, t * 9], np.float64))
    Rw, Pw = np.stack(R), np.stack(P)
    print("창 %d · 프레임 %d (2Hz) · 기기 %d" % (Rw.shape[0], Rw.shape[2], len(APPS)))

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig_s, usable = harmonic_signatures_by_state(pool, APPS)
    hsc = harmonic_scales(pool, APPS)
    T, names = resistive_templates(APPS, sig_s, usable, RESISTIVE)
    print("템플릿 %d개: %s\n" % (len(T), " ".join(n[:9] for n in names)))

    # ── [0] 페이저 규약을 **재서** 확인한다 ─────────────────────────────
    #     P = Re(V₁I₁*) 인가 그 절반인가. 가정하면 G 가 2배 틀린다.
    v1 = Rw[:, VOLT_RE0, :] + 1j * Rw[:, VOLT_IM0, :]
    i1 = Rw[:, 0, :] + 1j * Rw[:, N_HARM, :]
    p_ph = np.real(v1 * np.conj(i1))
    p_raw = Rw[:, 30, :]
    k1 = float(np.median(p_raw / np.maximum(p_ph, 1e-9)))
    chk(0, "페이저 규약 — P = c·Re(V₁I₁*) 의 c", 0.45 < k1 < 1.05,
        "c = **%.4f** (1.0 = RMS 페이저 · 0.5 = 피크). 아래 참 G 를 이 c 로 맞춘다" % k1)

    V2 = k1 * np.abs(v1) ** 2                                 # (B,T) — G 의 분모
    res_i = [APPS.index(x) for x in RESISTIVE]
    g_true = 1e3 * Pw[:, res_i].sum(1)[:, None] / np.maximum(V2, 1e-9)   # (B,T) mS

    th, resid, cond, _ = decompose(Rw, T, hsc)
    g_hat = 1e3 * th[:, :, 0]                                 # mS

    # ── [1] 잔차 ────────────────────────────────────────────────────────
    md = float(np.median(resid))
    chk(1, "설계행렬이 관측 전류를 설명하나", md < 0.35,
        "상대잔차 ‖W(y−Aθ̂)‖/‖Wy‖ 중앙 **%.3f** (p10 %.3f · p90 %.3f)  "
        "⚠ 13.84.23 이 SMPS 템플릿 잔차를 17~27%%로 쟀다"
        % (md, float(np.percentile(resid, 10)), float(np.percentile(resid, 90))))

    # ── [2] 조건수 ──────────────────────────────────────────────────────
    mc = float(np.median(cond))
    chk(2, "정규방정식이 풀 만한가", mc < 1e6,
        "cond(H) 중앙 **%.3g** · p90 %.3g (릿지 뒤). 열 %d개"
        % (mc, float(np.percentile(cond, 90)), th.shape[-1]))

    # ── [3] ★★ 참값 회수 ───────────────────────────────────────────────
    big = g_true > 5.0
    rel = np.abs(g_hat - g_true)[big] / g_true[big]
    hit = float((rel < 0.10).mean())
    chk(3, "★ Ĝ_sum 이 **참 저항 컨덕턴스 합**인가", float(np.median(rel)) < 0.15,
        "저항이 켜진 프레임 %d개 · 상대오차 중앙 **%.1f%%** · 10%% 안 **%.0f%%** · "
        "중앙 추정 %.1f 대 참 %.1f mS"
        % (big.sum(), 100 * float(np.median(rel)), 100 * hit,
           float(np.median(g_hat[big])), float(np.median(g_true[big]))))

    # ── [4] ★ 모터 열이 정말 필요한가 (사용자 지적) ─────────────────────
    T2, _ = resistive_templates(APPS, sig_s, usable,
                                tuple(RESISTIVE) + ("fan", "air_conditioner"))
    th2, _, _, _ = decompose(Rw, T2, hsc)
    rel2 = np.abs(1e3 * th2[:, :, 0] - g_true)[big] / g_true[big]
    worse = float(np.median(rel2)) / max(float(np.median(rel)), 1e-9)
    chk(4, "모터(선풍기·에어컨) 열을 빼면 나빠지나", worse > 1.05,
        "열 있음 %.1f%% -> **없음 %.1f%% (%.2f배)**. 스펙의 A=[V,jV,T_SMPS] 에는 모터가 "
        "갈 곳이 없어 Ĝ_sum 에 흡수된다" % (100 * float(np.median(rel)),
                                            100 * float(np.median(rel2)), worse))

    # ── [5] ② 가 잡음을 줄이나 ──────────────────────────────────────────
    c0 = Rw.shape[2] // 2
    thc, fw = refine(Rw, T, hsc, center=c0)
    gc = 1e3 * thc[:, 0]
    bigc = g_true[:, c0] > 5.0
    r1 = np.abs(g_hat[:, c0] - g_true[:, c0])[bigc] / g_true[bigc, c0]
    r2 = np.abs(gc - g_true[:, c0])[bigc] / g_true[bigc, c0]
    chk(5, "★ ② 정밀화가 한 프레임보다 나은가",
        float(np.median(r2)) < float(np.median(r1)),
        "중앙 프레임 한 칸 **%.2f%%** -> 가중합 **%.2f%%** (%.2f배) · 유효 프레임 중앙 %.1f/%d"
        % (100 * float(np.median(r1)), 100 * float(np.median(r2)),
           float(np.median(r2)) / max(float(np.median(r1)), 1e-9),
           float(np.median(fw.sum(1))), Rw.shape[2]))

    # ── [6] 전이를 실제로 막나 ──────────────────────────────────────────
    dp = np.abs(np.diff(Rw[:, 30, :], axis=-1))
    has = dp.max(1) > 200.0
    far = fw[:, [0, -1]].mean(1)
    chk(6, "전이 있는 창에서 먼 프레임의 무게가 죽나", has.sum() > 10
        and float(np.median(far[has])) < float(np.median(far[~has])),
        "전이창 %d개 — 양끝 무게 중앙 **%.3f** · 전이 없는 창 **%.3f**"
        % (has.sum(), float(np.median(far[has])) if has.any() else -1,
           float(np.median(far[~has])) if (~has).any() else -1))

    # ── [7] 끝프레임(실시간) 판도 도나 ──────────────────────────────────
    thl, _ = refine(Rw, T, hsc, center=None)
    gl = 1e3 * thl[:, 0]
    bl = g_true[:, -1] > 5.0
    rl = np.abs(gl - g_true[:, -1])[bl] / g_true[bl, -1]
    chk(7, "끝프레임(과거만) 판도 서나", float(np.median(rl)) < 0.30,
        "상대오차 중앙 **%.1f%%** (중앙 기준은 %.1f%%). 실시간이면 이쪽이고 **지연이 안 는다**"
        % (100 * float(np.median(rl)), 100 * float(np.median(r2))))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
