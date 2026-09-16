"""
실측에서 어디가 틀어지는가 — 진단 플롯
========================================
합성 지표만 보고 손실 가중치를 만지는 접근은 여러 번 실패했다 (12.9.3절).
2단계 설계 전에 **실측에서 무엇이 어긋나는지** 눈으로 확인한다.

파일마다 네 칸을 그린다.

  1) 관측 총전력 vs 예측 합계      — 합이 맞는가
  2) 잔차 (관측 − 예측)            — 어느 구간에서 벌어지는가
  3) 기기별 예측 전력 (누적)        — 그 와트를 누가 가져갔는가
  4) 알려진 정답 구간              — 그때 실제로 무엇이 켜져 있었나
  5) **모델의 on/off 판정**        — 모델은 무엇이 켜졌다고 봤나 (게이트)

5번은 **모든 기기**를 행으로 놓는다. 정답이 없는 기기에 게이트가 서는 것이 곧
오귀속이라, 정답 있는 기기만 그리면 그게 안 보인다. 정답 구간은 검은 테두리로
겹쳐 두었으므로 **테두리 밖의 색 = 헛detect, 테두리 안의 빈칸 = 놓침**이다.

**실측에는 기기별 정답이 없다** (12.4절). 4번 칸은 `real_events.json` 의
*확실* 구간과 이벤트뿐이고, 정상상태 귀속은 애초에 옮기지 않았다 (4.2절).

    python -m src.run_plot_real --ckpt results/cnn_v11.pt
    python -m src.run_plot_real --ckpt results/adapt_v1.pt --tag adapt
"""
from pathlib import Path
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.inputs import LEGACY_FINE_CHANNELS
from src.model.net import NILMNet, appliance_state_counts
from src.model.realdata import dense_targets
from src.preprocessing import load_nilm_npz

KO = {"air_conditioner": "에어컨", "beam_projector": "빔프로젝터",
      "electiric_kettle": "전기포트", "fan": "선풍기", "hair_dryer": "헤어드라이기",
      "hotplate": "핫플레이트", "laptop_charger": "노트북충전기",
      "minipc": "미니PC", "oven": "오븐"}
COL = plt.get_cmap("tab10")
SAMPLING_HZ = 60.0


def _font():
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_model(ckpt: str, dev: str):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    # ── 시퀀스 체크포인트도 받는다 (13.84.74) ─────────────────────────────────
    # 이 도구는 CNN 시절 것이라 **구조를 체크포인트에서 안 읽는다** (세밀 38채널 고정).
    # 시퀀스 판(13.84.26~)은 몸통만 담고 구조·가림은 `ref` 가 가리키는 판에 있다 —
    # `run_gate_check.load_model` 이 그 규약을 안다. 옛 CNN 판은 `heads` 키가 없으므로
    # 아래 옛 경로를 그대로 탄다 (**비트 단위로 같다**).
    if "heads" in ck:
        from src.run_gate_check import load_model as _lm
        m = _lm(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                mask=not bool(ck.get("no_mask", False)),
                proj_from={k: ck[k] for k in
                           ("proj", "proj_cap", "proj_floor", "proj_resp",
                            "appl_attn", "appl_attn_heads") if k in ck})[0]
        m.load_state_dict(ck["model"]); m.eval()
        return m, apps, ck
    # ⚠⚠ **2026-09-13 고침 — 여기가 `zero_channels` 를 안 읽었다.**
    #   옛 갈래가 `NILMNet` 을 직접 지어서 **가림 훅이 안 붙었다.** 그래서 `cnn_v37`
    #   처럼 가림이 있는 판은 그림에서 **학습 때 0 이던 채널에 실제 값**을 받았다 —
    #   본 적 없는 입력이다 (13.80.10: *"v35 를 안 가리고 채점하면 0.929 -> 0.856"*).
    #   증상: 같은 파일에서 그림 잔차 82.1W 대 채점기 40.7W 로 **2배**가 어긋났다.
    #   v36 은 가림이 없어 멀쩡했고 **v37 에서만** 틀려, 두 판 비교가 통째로 뒤집혔다.
    #   `run_gate_check.load_model` 이 그 규약(구조·가림·site_transfer)을 다 안다.
    from src.run_gate_check import load_model as _lm
    m = _lm(ckpt, dev)[0]
    return m, apps, ck


@torch.no_grad()
def predict(model, rw, dev: str, batch: int = 512, g_hat=None, z_in=None):
    """(전력, 대기, 관문, 관측P, 관측고조파, 계측P) — 후처리가 뒤의 셋을 쓴다.

    ★ 14.349 — `comb_tau>0` 이면 **Ĝ 가 입력이다.** 안 넘기면 `net.forward` 가
      멈춘다 (`run_gate_comb` [7]). 모델과 물리가 한 몸이라는 뜻이다.
    """
    P, S, G, PO, OH, PN = [], [], [], [], [], []
    _cb = float(getattr(model, "comb_tau", 0.0) or 0.0) > 0
    _zi = bool(getattr(model, "z_input", False))
    if _cb and g_hat is None:
        raise ValueError("comb_tau 체크포인트인데 g_hat 이 없다 — solve_ghat 을 먼저 불러라")
    if _zi and z_in is None:
        raise ValueError("z_input 체크포인트인데 z_in 이 없다 — site_z(stem) 을 넘겨라")
    for i in range(0, len(rw), batch):
        f, w, pobs, oh, pn = rw.batch(np.arange(i, min(i + batch, len(rw))))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                      torch.from_numpy(np.ascontiguousarray(w)).to(dev),
                      torch.from_numpy(
                          np.ascontiguousarray(g_hat[i:i + batch], np.float32)).to(dev)
                      if _cb else None,
                      torch.full((len(f),), float(z_in), device=dev) if _zi else None)
        P.append(o["power"].float().cpu().numpy())
        S.append(o["standby"].float().cpu().numpy())
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        PO.append(pobs); OH.append(oh); PN.append(pn)
    return (np.concatenate(P), np.concatenate(S), np.concatenate(G),
            np.concatenate(PO), np.concatenate(OH), np.concatenate(PN))


def run_postproc(mode, apps, pred, standby, gate, pobs, oh, pn, v_rms):
    """`cap` = 물리 상한 넘기기, `full` = 거기에 저항 조합 정합까지 (12.112).

    ⚠ `resistive_match` 는 `min_w=150W` 문턱이라 **SMPS 전용 창(93W)은 안 건드린다.**
      자리 D 배분은 후처리로 못 고친다 — 고치는 것은 저항 창의 조합이다.
    """
    if mode == "off":
        return pred, gate
    from src.model.postproc import apply_postproc, resistive_match
    P, g = apply_postproc(pred, gate, apps)
    if mode == "full":
        P, g = resistive_match(P, g, apps, pobs, v_rms, standby, pn, obs_harm=oh)
    return np.asarray(P, np.float32), np.asarray(g, np.float32)


def solve_ghat(stem, rw, apps):
    """14.341·14.349 — 그 파일의 **타깃 사이클에서** `(Ĝ mS, |V₁|)` 를 푼다.

    창·라벨·Ĝ 가 같은 색인이라야 한다 — 34.7 ⑦ 이 39사이클 어긋난 자리를 재서
    헛것을 냈다. ★ **조합 머리(`comb_tau>0`)의 입력이자 사중의 예산이다** — 둘이
    같은 함수를 써야 그림과 후처리가 **같은 Ĝ** 위에 선다.

    ⚠ 기준전압은 **이 녹화의 중앙값**이다. 학습은 캐시의 것을 썼다 (14.346 이 잰
      차가 최대 0.275 mS = 포트 여유의 43%). 실측은 원리상 이쪽밖에 없다.
    """
    import numpy as _np

    from src.model import gbudget as GB
    from src.model import inputs as _I
    from src.model.realdata import RealWindows
    from src.synthesis.segment_pool import SegmentPool

    z = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
    x = RealWindows._to_33ch(z).astype(_np.float64)
    nv = len(_I.VOLT_ORDERS)
    v15 = _np.zeros(15, complex)
    med = _np.median(x[33:33 + nv], 1) + 1j * _np.median(x[33 + nv:33 + 2 * nv], 1)
    for s_, h in enumerate(_I.VOLT_ORDERS):
        v15[h - 1] = med[s_]
    bud = GB.Budget(apps, v15, pool=SegmentPool(npz_dir="processed_data/npz",
                                                time_split="train"),
                    volt_re0=33, volt_orders=_I.VOLT_ORDERS)
    tc = _np.asarray(rw.target_cycle, _np.int64)
    return (bud.g_sum(x[None][:, :, tc])[0],
            _np.abs(x[33, tc] + 1j * x[33 + nv, tc]))


def site_z(stem: str):
    """★ 14.369 — 그 녹화의 **선로 저항** [Ω]. 모르면 `nan` ("모름" 토큰).

    `--z-input` 판은 `forward` 에 이 값을 넘겨야 돈다 (`g_hat` 과 같은 규약이다).
    출처는 `SITE_SESSIONS[...]["z_ohm"]` — §13.4 가 계단법으로 쟀다
    (E1 13계단 0.424 · D1 5계단 1.214±0.017 · D2 test_3 1.35±0.42 / test_5 1.04±0.15).
    ```
      test_1 -> D1 1.15Ω · test_2 -> E1 0.42Ω · test_3·4·5 -> D2 1.19Ω
    ```
    ⚠ **못 잰 자리는 `nan` 을 넘긴다** — 학습이 가림(`--z-drop`)으로 그 갈래를 배웠다.
      임의의 값을 지어 넣으면 **틀린 값을 자신 있게** 쓰게 된다.
    """
    from src.preprocessing.file_registry import SITE_SESSIONS
    for _k, v in SITE_SESSIONS.items():
        if stem in tuple(v.get("stems", ()) or ()):
            z = v.get("z_ohm")
            return float(z) if z else float("nan")
    return float("nan")


def gbudget_apply(stem, rw, apps, pred, gv=None):
    """14.341 — `gbudget` 사중을 걸고 **어느 창이 움직였는지** 같이 돌려준다."""
    import numpy as _np

    from src.model import gbudget as GB

    g, v1 = solve_ghat(stem, rw, apps) if gv is None else gv
    q, info = GB.apply(_np.asarray(pred, _np.float64), g, v1, apps)
    print("    사중 — Ĝ 중앙 %.2f mS · 거부 %d창 · 바닥 %d창 · 고정으로 바뀐 칸 %d"
          % (float(_np.median(g)), int(info["veto"].sum()), int(info["floor"].sum()),
             int((_np.abs(q - pred) > 1.0).sum())))
    return q.astype(np.float32), info


def plot_file(stem, apps, t_pred, pred, standby, t_obs, obs, spec, path, title,
              gate=None, gate_thr=0.5, mark=None):
    n_ax = 4 if gate is None else 5
    hr = [3, 2, 3, 2.2] if gate is None else [3, 2, 3, 2.2, 2.4]
    fig, ax = plt.subplots(n_ax, 1, figsize=(16, 11 if gate is None else 13.4), sharex=True,
                           gridspec_kw={"height_ratios": hr, "hspace": 0.12})

    total = pred.sum(1) + standby.sum(1)
    obs_at = np.interp(t_pred, t_obs, obs)

    # 1) 관측 vs 예측 합계
    ax[0].plot(t_obs, obs, lw=0.4, color="0.35", label="관측 총전력 (60Hz)")
    ax[0].plot(t_pred, total, lw=1.4, color="crimson", label="예측 합계 (활성+대기)")
    ax[0].set_ylabel("전력 (W)")
    ax[0].legend(loc="upper right", fontsize=9, framealpha=.9)
    ax[0].grid(alpha=.3)
    ax[0].set_title(title, fontsize=13, loc="left")

    # 2) 잔차
    r = obs_at - total
    ax[1].axhline(0, color="0.3", lw=0.8)
    ax[1].fill_between(t_pred, r, 0, where=r >= 0, color="tab:red", alpha=.45,
                       label="과소 예측 (관측 > 예측)")
    ax[1].fill_between(t_pred, r, 0, where=r < 0, color="tab:blue", alpha=.45,
                       label="과대 예측")
    ax[1].set_ylabel("잔차 (W)")
    ax[1].legend(loc="upper right", fontsize=9, framealpha=.9)
    ax[1].grid(alpha=.3)
    ax[1].annotate(f"평균 {r.mean():+.1f}W   절대평균 {np.abs(r).mean():.1f}W",
                   (0.01, 0.06), xycoords="axes fraction", fontsize=10, color="0.2")

    # 3) 기기별 예측 (누적)
    order = np.argsort(-pred.mean(0))
    ax[2].stackplot(t_pred, *[pred[:, j] for j in order],
                    labels=[KO.get(apps[j], apps[j]) for j in order],
                    colors=[COL(i % 10) for i in range(len(order))], alpha=.85)
    ax[2].plot(t_obs, obs, lw=0.4, color="0.15", alpha=.6)
    ax[2].set_ylabel("기기별 예측 (W)")
    ax[2].legend(loc="upper right", fontsize=8, ncol=3, framealpha=.9)
    ax[2].grid(alpha=.3)

    # 4) 알려진 정답 구간
    iv = spec["intervals"]
    rows = [a for a in apps if iv.get(a, {}).get("on") or iv.get(a, {}).get("uncertain")]
    for i, a in enumerate(rows):
        for t0, t1 in iv[a].get("uncertain", []):
            ax[3].barh(i, t1 - t0, left=t0, height=0.55, color="0.75",
                       hatch="///", edgecolor="0.5", linewidth=0.4)
        for t0, t1 in iv[a].get("on", []):
            ax[3].barh(i, t1 - t0, left=t0, height=0.55,
                       color=COL(apps.index(a) % 10), alpha=.9)
    # 이벤트 라벨은 **세로로 쓰고 층을 번갈아 놓는다.** 가로로 쓰면 test_5/test_8
    # 처럼 20~24개가 몰린 파일에서 글자가 서로를 덮어 아무것도 안 읽힌다.
    # 그래도 가까운 것끼리는 겹치므로, 앞 라벨과 최소 간격을 두고 층을 올린다.
    ev_sorted = sorted(spec.get("events", []), key=lambda e: e["t_s"])
    span = max(t_obs[-1] - t_obs[0], 1.0) if len(t_obs) else 1.0
    min_gap = span * 0.035          # 이보다 가까우면 다음 층으로 올린다
    last_t, level, n_level = -1e9, 0, 3
    for e in ev_sorted:
        ax[3].axvline(e["t_s"], color="crimson", ls="--", lw=1.0, alpha=.55)
        level = (level + 1) % n_level if (e["t_s"] - last_t) < min_gap else 0
        last_t = e["t_s"]
        # ΔP 가 없는 이벤트가 있다 (test_6/test_9 는 신호 유추라 `_note` 만 있다).
        dp = e.get("delta_p_w")
        ax[3].annotate(f"{KO.get(e['appliance'], e['appliance'])} {e['kind']}"
                       + (f" {dp:+.0f}W" if dp is not None else ""),
                       (e["t_s"], len(rows) - 0.45 - 0.16 * level),
                       fontsize=6.5, color="crimson", ha="left", va="top",
                       rotation=90, rotation_mode="anchor", alpha=.85)
    ax[3].set_yticks(range(len(rows)))
    ax[3].set_yticklabels([KO.get(a, a) for a in rows], fontsize=9)
    # 세로 라벨이 들어갈 자리를 위쪽에 비워 둔다.
    ax[3].set_ylim(-0.6, len(rows) + 1.1)
    if gate is None:
        ax[3].set_xlabel("시간 (초)")
    ax[3].set_ylabel("알려진 정답")
    ax[3].grid(alpha=.3, axis="x")
    ax[3].annotate("색칠 = 확실히 켜짐   빗금 = uncertain(채점 제외)",
                   (0.01, 0.04), xycoords="axes fraction", fontsize=9, color="0.3")

    # 5) 모델의 on/off 판정 — **모든 기기**를 행으로
    if gate is not None:
        g = np.asarray(gate, float)
        iv = spec["intervals"]
        dt = float(np.median(np.diff(t_pred))) if len(t_pred) > 1 else 1.0
        # ⚠ 창은 파일 처음부터 있지 않다 — 광역 갈래가 60초를 먹으므로 test_4 는
        #   **54초**부터다. 그런데 정답 테두리는 파일 전체에 그려지니, 창이 없는
        #   구간이 "테두리 안 빈칸" = **놓침**으로 읽힌다. 실제로 그렇게 읽을 뻔했다
        #   ([[check-the-denominator-before-reading-a-ratio]]). 덮어서 못 읽게 한다.
        t_lo = float(t_pred[0]) - dt / 2 if len(t_pred) else 0.0
        t_hi = float(t_pred[-1]) + dt / 2 if len(t_pred) else 0.0
        for axk in (ax[3], ax[4]):
            for x0, x1 in ((t_obs[0] if len(t_obs) else 0.0, t_lo),
                           (t_hi, t_obs[-1] if len(t_obs) else t_hi)):
                if x1 > x0:
                    axk.axvspan(x0, x1, color="0.82", alpha=.65, zorder=6, lw=0)
                    axk.annotate("창 없음", ((x0 + x1) / 2, 0.5), xycoords=("data",
                                 "axes fraction"), fontsize=7.5, color="0.25",
                                 ha="center", va="center", rotation=90, zorder=7)
        for i, a in enumerate(apps):
            col = COL(apps.index(a) % 10)
            on = g[:, i] > gate_thr
            # 게이트 세기를 진하기로 (0.5 -> 옅게, 1.0 -> 진하게)
            for k in np.flatnonzero(on):
                ax[4].barh(i, dt, left=t_pred[k], height=0.62, color=col,
                           alpha=float(np.clip((g[k, i] - gate_thr) / (1 - gate_thr), 0.15, 1.0)),
                           linewidth=0)
            # 정답 구간을 테두리로 겹친다 — 테두리 밖의 색이 헛detect 다
            for t0, t1 in iv.get(a, {}).get("on", []):
                ax[4].add_patch(plt.Rectangle((t0, i - 0.34), t1 - t0, 0.68, fill=False,
                                              edgecolor="0.1", linewidth=1.1, zorder=5))
            for t0, t1 in iv.get(a, {}).get("uncertain", []):
                ax[4].add_patch(plt.Rectangle((t0, i - 0.34), t1 - t0, 0.68, fill=False,
                                              edgecolor="0.55", linewidth=0.9, ls=":", zorder=5))
        ax[4].set_yticks(range(len(apps)))
        ax[4].set_yticklabels([KO.get(a, a) for a in apps], fontsize=9)
        ax[4].set_ylim(-0.6, len(apps) - 0.4)
        ax[4].set_xlabel("시간 (초)")
        ax[4].set_ylabel(f"모델 판정 (게이트>{gate_thr:g})")
        ax[4].grid(alpha=.3, axis="x")
        # 기기별로 얼마나 켰다고 봤는지 — 정답과 견줄 수 있게 옆에 적는다
        for i, a in enumerate(apps):
            frac = float((g[:, i] > gate_thr).mean())
            # ⚠ 모델 %는 **창**에서 세고 참 %는 **파일 전체**에서 셌다 — 분모가 달랐다.
            #   test_4 충전기는 16.1초에 켜지는데 창은 54초부터라, 창이 없는 38초가
            #   참 쪽 분자에만 들어가 "89% 참인데 모델은 80%" 로 보였다. 실제로는
            #   그 구간 유지율이 **0.999** 다. 창 구간으로 자른다.
            true_s = sum(max(0.0, min(t1, t_hi) - max(t0, t_lo))
                         for t0, t1 in iv.get(a, {}).get("on", []))
            span = max(t_hi - t_lo, 1.0)
            ax[4].annotate(f"{100*frac:.0f}% (참 {100*true_s/span:.0f}%)",
                           (1.002, i), xycoords=("axes fraction", "data"),
                           fontsize=7.5, va="center", color="0.25")
        #: 14.341 — 사중이 **어느 창을 움직였는지** 게이트 칸에 겹친다.
        #   ✕ 가 찍힌 색 = 거부권이 지운 것 · ▲ = 바닥이 세운 것
        #   ⇒ **✕ 없이 테두리 밖에 남은 색이 "사중이 못 막은 오탐"** 이다
        if mark:
            for i, _a in enumerate(apps):
                for key, sym, c, ms in (("veto", "x", "0.05", 4.2),
                                        ("floor", "^", "#0a7d00", 4.0)):
                    mk = mark.get(key)
                    if mk is None:
                        continue
                    sel = np.asarray(mk, bool)[:, i]
                    if not sel.any():
                        continue
                    ax[4].plot(t_pred[sel], np.full(int(sel.sum()), i), sym, ms=ms,
                               mew=1.2, color=c, zorder=8, ls="none")
        ax[4].annotate("진하기 = 게이트 세기   검은 테두리 = 참 ON   점선 = uncertain   "
                       "**테두리 밖의 색 = 헛detect, 테두리 안 빈칸 = 놓침**"
                       + ("   ✕ 거부권이 지움 · ▲ 바닥이 세움" if mark else ""),
                       (0.01, 0.03), xycoords="axes fraction", fontsize=8.5, color="0.3")

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 {path}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="실측 진단 플롯")
    ap.add_argument("--ckpt", default="results/cnn_v11.pt")
    ap.add_argument("--stride", type=int, default=30, help="예측 간격 (사이클)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--gate-thr", type=float, default=0.5,
                    help="5번 칸에서 '켜졌다' 로 볼 게이트 문턱")
    ap.add_argument("--no-gate", action="store_true", help="5번 칸을 안 그린다 (옛 4칸 그림)")
    #: 14.341 — `gbudget` 사중(거부권·바닥·고정). **끄면 비트 동일**이다.
    ap.add_argument("--gbudget", action="store_true",
                    help="컨덕턴스 예산 사중을 걸고 ✕/▲ 로 표시한다")
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"),
                    help="`cap` 물리 상한 넘기기 · `full` 거기에 저항 조합 정합까지. "
                         "⚠ resistive_match 는 min_w=150W 라 SMPS 전용 창은 안 건드린다")
    a = ap.parse_args()

    _font()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    from src.run_gate_check import sync_even_median
    sync_even_median([a.ckpt])   #: 14.164 — 체크포인트의 짝수차 규약으로 그린다
    model, apps, ck = load_model(a.ckpt, dev)
    ev = load_events()
    name = a.tag or Path(a.ckpt).stem
    print(f"[실측 진단] {a.ckpt} (ep{ck.get('epoch')}) | 장치 {dev}")

    rows = []
    for stem in sorted(ev):
        if is_sealed(stem):
            print(f"  {stem}: 봉인 — 건너뜀 (4.3절)")
            continue
        rw = dense_targets(stem, stride=a.stride)
        #: ★ 14.349 — Ĝ 를 **한 번만** 푼다. 조합 머리의 입력이자 사중의 예산이라
        #  둘이 갈리면 그림과 후처리가 다른 자 위에 선다.
        _gv = (solve_ghat(stem, rw, apps)
               if (a.gbudget or float(getattr(model, "comb_tau", 0.0) or 0.0) > 0)
               else None)
        pred, standby, gate, pobs, oh, pn = predict(
            model, rw, dev, g_hat=None if _gv is None else _gv[0],
            z_in=site_z(stem) if getattr(model, "z_input", False) else None)
        mark = None
        if a.gbudget:
            #: 14.341 — 컨덕턴스 예산 사중. `--postproc` 과 **다른 물건**이다
            #  (저쪽은 12.112 의 `resistive_match`). 같이 켜지 못하게 막는다.
            pred, _gi = gbudget_apply(stem, rw, apps, pred, gv=_gv)
            mark = {"veto": _gi["veto"], "floor": _gi["floor"]}
        if a.postproc != "off":
            pred, gate = run_postproc(a.postproc, apps, pred, standby, gate,
                                      pobs, oh, pn,
                                      np.asarray(rw.v_observed, np.float32))
        raw = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
        obs = np.asarray(raw["power_features"])[:, 0]
        t_obs = np.arange(len(obs)) / SAMPLING_HZ
        t_pred = rw.target_cycle / SAMPLING_HZ
        r = plot_file(stem, apps, t_pred, pred, standby, t_obs, obs, ev[stem],
                      str(outd / f"real_{stem}_{name}.png"),
                      f"{stem}  —  {a.ckpt} 예측  ({len(rw):,}창, {a.stride/60:.1f}초 간격)",
                      gate=None if a.no_gate else gate, gate_thr=a.gate_thr,
                      mark=mark)
        rows.append((stem, r, pred))

    print(f"\n{'파일':10s}{'잔차 평균':>11s}{'절대':>9s}  기기별 평균 예측 (W)")
    for stem, r, pred in rows:
        top = np.argsort(-pred.mean(0))[:4]
        s = "  ".join(f"{KO.get(apps[j], apps[j])} {pred[:, j].mean():.0f}" for j in top)
        print(f"{stem:10s}{r.mean():>+10.1f}W{np.abs(r).mean():>8.1f}W  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
