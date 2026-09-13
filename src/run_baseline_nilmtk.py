# -*- coding: utf-8 -*-
"""이 분야의 **기준 알고리즘**을 우리 데이터로 돌린다 — CO 와 FHMM (14.8 의 (가)).

4.1절이 "baseline 을 먼저 세워라" 라고 적었고 6절이 "CNN 부터 시작" 을 금지 목록에
넣었는데, 실제로 세운 기준선은 GBM **분류기** 하나뿐이었다. `run_hmm_probe.py` 는
후처리 평활이고 **기기별 독립**이라 우리 사슬과 같은 결함을 물려받는다.

**NILMTK 의 두 기준 알고리즘을 한 번도 안 돌렸다.** 그래서 "우리 0.8466 이 좋은
건가" 를 답할 수 없다. 여기서 그 구멍을 메운다.

    CO    (Combinatorial Optimisation)  조합을 전수하고 합이 관측에 가장 가까운 것
    FHMM  (Factorial HMM)               거기에 기기별 마르코프 사슬 + 결합 Viterbi

**두 관측으로 각각 돌린다.**
    -P  유효전력만          문헌의 정식 판본 (1Hz 스마트미터 가정)
    -H  30차원 고조파       우리 데이터의 이점 (Re15+Im15, `harmonic_scales` 정규화)

[FHMM 의 전이를 성기게 두는 근거 — 13.85 [1]]
결합 상태가 19,440개라 조밀 Viterbi 는 한 스텝에 3.8e8 이다. 13.85 [1] 이
실측 101개 전이 **전부**가 혼자 일어남을 쟀으므로 (최소 간격 6.17초, 격자 2초)
**한 스텝에 기기 하나만 바뀐다**로 전이를 제한한다 — AFAMAP 의 그 장치다.
선행자가 20개로 줄어 한 스텝 3.9e5 이 된다.
⚠ 이것은 **통제된 시험 녹화의 성질**이다. 가정집 배포에서는 소프트 벌점이라야 한다.

[무엇이 학습 데이터인가 — 규칙 14]
상태별 전력은 `losses.S_STATE` (격리 녹화 12,000창 측정), 상태별 지문은
`harmonic_signatures_by_state` (격리 녹화), 체류시간은 격리 녹화 활성화 길이다.
**실측 복합 5파일은 채점에만 쓴다.** 관측 잡음 sigma 는 CO 잔차로 자가 보정한다 —
손잡이를 훑지 않는다 (훑으면 기준선이 아니라 튜닝이다).

⚠ **후보는 늘 9종 전부다.** 그 파일에 있는 기기만 후보로 두면 정답을 흘리는 것이고,
  모델은 9종에서 고르므로 비교가 성립하지 않는다.

    python -X utf8 src/run_baseline_nilmtk.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.evaluation.metrics import RESISTIVE_MIN_TRUE_W
from src.model.losses import S_STATE
from src.model.net import harmonic_scales, harmonic_signatures_by_state
from src.run_score_seq import POWER_FLOOR_W, RESISTIVE_ALL, SOLO_MIN, SMPS_TRIO
from src.run_train_seq import FS, FILES, real_windows
from src.synthesis.segment_pool import SegmentPool

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
MAX_STATES = 5


# ────────────────────────────────────────────────────────────────────────────
# 사전 — 상태별 전력과 지문 (전부 격리 녹화에서)
# ────────────────────────────────────────────────────────────────────────────
def build_states(pool):
    """기기별 상태 목록. 상태 0 은 OFF(0W). 반환 (n_states, pw, sig30)."""
    sig_s, _ = harmonic_signatures_by_state(pool, APPS)       # (K,S,15,2) 와트당
    scale = np.maximum(harmonic_scales(pool, APPS), 1e-12)    # (15,)
    n_st, pw, s30 = [], [], []
    for j, a in enumerate(APPS):
        ws = [0.0] + [S_STATE[a][s] for s in sorted(S_STATE[a])]
        ids = [0] + sorted(S_STATE[a])
        n_st.append(len(ws))
        pw.append(np.array(ws, np.float64))
        v = np.zeros((len(ws), 30))
        for i, (sid, w) in enumerate(zip(ids, ws)):
            if sid == 0:
                continue                       # OFF 는 0 — 대기는 기준선에 들어 있다
            c = sig_s[j, sid] * w              # 와트당 x 와트 = 절대 페이저
            v[i] = np.concatenate([c[:, 0] / scale, c[:, 1] / scale])
        s30.append(v)
    return n_st, pw, s30, scale


def enumerate_combos(n_st, pw, s30):
    """모든 결합상태. 반환 (combo (M,K) int8, 총전력 (M,), 총지문 (M,30))."""
    grids = np.meshgrid(*[np.arange(n) for n in n_st], indexing="ij")
    combo = np.stack([g.ravel() for g in grids], 1).astype(np.int8)   # (M,K)
    M, K = combo.shape
    tot_p = np.zeros(M)
    tot_h = np.zeros((M, 30))
    for k in range(K):
        tot_p += pw[k][combo[:, k]]
        tot_h += s30[k][combo[:, k]]
    return combo, tot_p, tot_h


# ────────────────────────────────────────────────────────────────────────────
# 전이 — 격리 녹화 활성화 길이에서 (NILMTK 의 FHMM 도 서브미터 자료에서 배운다)
# ────────────────────────────────────────────────────────────────────────────
def build_transitions(pool, n_st, grid_s, duty=0.25):
    """기기별 (n_k, n_k) 로그 전이. 상태 0 = OFF.

    **ON 체류시간만 자료에서 온다** — 격리 녹화 활성화 길이의 중앙값.

    ⚠ **`duty`(가동률)는 자료에서 못 온다. 명시적 사전확률이다.**
    격리 녹화의 `is_on` 비율은 0.77~0.91 인데, 그건 *그 기기를 돌린 녹화*라서
    그렇지 가정집의 가동률이 아니다. NILMTK 의 FHMM 은 서브미터 **가정집** 자료에서
    이것을 배우는데 우리에겐 그런 자료가 없다 (실측 복합 5파일은 채점 전용).
    그래서 상수로 박고 **민감도를 함께 찍는다** — 고르지 않는다.
    """
    out = []
    for j, a in enumerate(APPS):
        acts = pool.appliance_activations.get(a, [])
        durs = [x.duration_s for x in acts if x.duration_s > 0]
        on_steps = max(2.0, float(np.median(durs)) / grid_s) if durs else 30.0
        n = n_st[j]
        p_off_given_on = 1.0 / on_steps
        # 정상분포 pi_on = duty  =>  p_on_given_off = duty/(1-duty) * p_off_given_on
        p_on_given_off = float(np.clip(duty / max(1.0 - duty, 1e-6) * p_off_given_on,
                                       1e-6, 0.5))
        T = np.zeros((n, n))
        T[0, 0] = 1.0 - p_on_given_off
        if n > 1:
            T[0, 1:] = p_on_given_off / (n - 1)            # 켜질 때 상태는 균등
        for s in range(1, n):
            T[s, 0] = p_off_given_on
            # 켜진 상태들 사이 전환은 OFF 로 가는 것과 같은 정도로 둔다 (드라이기 모드 변경)
            if n > 2:
                T[s, [x for x in range(1, n) if x != s]] = p_off_given_on / (n - 2)
            T[s, s] = 1.0 - T[s].sum()
        out.append(np.log(np.maximum(T, 1e-12)))
    return out


def build_sparse_pred(combo, n_st, logT):
    """한 스텝에 **기기 하나만** 바뀐다 (13.85 [1]). 반환 (pred_idx (M,R), pred_lp (M,R))."""
    M, K = combo.shape
    stride = np.ones(K, np.int64)
    for k in range(K - 2, -1, -1):
        stride[k] = stride[k + 1] * n_st[k + 1]          # meshgrid 'ij' 순서
    # 모두 그대로일 때의 로그확률
    stay = np.zeros(M)
    for k in range(K):
        stay += logT[k][combo[:, k], combo[:, k]]
    idx = [np.arange(M)]
    lp = [stay]
    for k in range(K):
        cur = combo[:, k].astype(np.int64)
        for s in range(n_st[k]):
            m = cur != s
            if not m.any():
                continue
            j = np.arange(M) + (s - cur) * stride[k]      # k 번 기기만 s 였던 자리
            # 그 선행자에서 지금으로: k 는 s->cur, 나머지는 제자리
            d = stay - logT[k][cur, cur] + logT[k][s, cur]
            idx.append(np.where(m, j, np.arange(M)))
            lp.append(np.where(m, d, -1e18))
    return np.stack(idx, 1).astype(np.int64), np.stack(lp, 1)


# ────────────────────────────────────────────────────────────────────────────
# 추론
# ────────────────────────────────────────────────────────────────────────────
def emission_sq(pred, obs):
    """||pred_m - obs_t||^2 -> (T, M). pred (M,D) · obs (T,D)."""
    return (np.einsum("md,md->m", pred, pred)[None, :]
            - 2.0 * obs @ pred.T
            + np.einsum("td,td->t", obs, obs)[:, None])


def viterbi_sparse(logE, pred_idx, pred_lp):
    """(T,M) 로그방출 + 성긴 선행자 -> 최적 경로 (T,)."""
    T, M = logE.shape
    dp = logE[0].copy()
    bp = np.zeros((T, M), np.int32)
    for t in range(1, T):
        cand = dp[pred_idx] + pred_lp               # (M,R)
        r = np.argmax(cand, 1)
        dp = cand[np.arange(M), r] + logE[t]
        bp[t] = pred_idx[np.arange(M), r]
    path = np.zeros(T, np.int64)
    path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


# ────────────────────────────────────────────────────────────────────────────
# 채점 — `run_score_seq` 와 **같은 규약**이어야 한다
# ────────────────────────────────────────────────────────────────────────────
def score_arm(pred_on, pred_pw, d, apps):
    """(기기별 정확도 dict, 신원 dict, 전력 dict, 잔차 dict, 혼동 dict).

    다섯째 `conf[기기] = (tp, fp, fn, tn)` 는 **정확도 하나로는 안 보이는 것**을
    보려고 같이 낸다 — 듀티 기기(`DUTY_APPS`)는 재현율만 낮고 정밀도는 0.99 인데,
    정확도로 뭉개면 그냥 "못 맞힌다" 로 보인다. 그 착시로 결론을 한 번 뒤집었다.
    """
    y = d["y"].astype(bool)
    base = float(d.get("p_base", float("nan")))
    acc, ident, pwr, recon, conf = {}, {}, {}, {}, {}
    for k in np.nonzero(d["present"])[0]:
        acc[apps[k]] = float((pred_on[:, k] == y[:, k]).mean())
        p_, t_ = pred_on[:, k], y[:, k]
        conf[apps[k]] = (int((p_ & t_).sum()), int((p_ & ~t_).sum()),
                         int((~p_ & t_).sum()), int((~p_ & ~t_).sum()))
    if np.isfinite(base):
        for k in np.nonzero(d["present"])[0]:
            other = np.delete(np.arange(y.shape[1]), k)
            solo = y[:, k] & (y[:, other].sum(1) == 0)
            t = d["p_obs"] - base
            solo = solo & (t > POWER_FLOOR_W)
            if solo.sum() < SOLO_MIN:
                continue
            e = np.abs(pred_pw[solo, k] - t[solo]) / np.maximum(t[solo], 1e-6)
            pwr[apps[k]] = (float(np.median(e)), int(solo.sum()))
        r = pred_pw.sum(1) + base - d["p_obs"]
        heavy = np.zeros(len(r), bool)
        for hx in ("oven", "hotplate"):
            if hx in apps and d["present"][apps.index(hx)]:
                heavy |= y[:, apps.index(hx)]
        recon = {"abs": float(np.abs(r).mean()), "bias": float(r.mean()),
                 "abs_heavy": (float(np.abs(r[heavy]).mean()) if heavy.sum() >= 10
                               else float("nan")),
                 "abs_light": (float(np.abs(r[~heavy]).mean()) if (~heavy).sum() >= 10
                               else float("nan"))}
    hi = d["p_obs"] >= RESISTIVE_MIN_TRUE_W
    for gname, grp, gate in (("저항 무리", RESISTIVE_ALL, hi),
                             ("SMPS 무리", SMPS_TRIO, np.ones(len(y), bool))):
        gi = [apps.index(x) for x in grp if x in apps]
        yg = y[:, gi]
        solo = (yg.sum(1) == 1) & gate
        if solo.sum() < 10:
            continue
        hit = np.argmax(pred_pw[solo][:, gi], 1) == np.argmax(yg[solo], 1)
        ident[gname] = (float(hit.mean()), int(solo.sum()))
    return acc, ident, pwr, recon, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--duty", type=float, default=0.25,
                    help="가동률 **사전확률**. 자료에서 오지 않는다 — 민감도용")
    a = ap.parse_args()

    print("NILMTK 기준 알고리즘 — CO · FHMM (14.8 의 (가))", flush=True)
    print("=" * 78)
    pool = SegmentPool()
    n_st, pw, s30, scale = build_states(pool)
    combo, tot_p, tot_h = enumerate_combos(n_st, pw, s30)
    logT = build_transitions(pool, n_st, a.grid_s, a.duty)
    pred_idx, pred_lp = build_sparse_pred(combo, n_st, logT)
    # 자기검사 — 선행자 표가 정말 "기기 하나만 다른" 자리를 가리키는가
    _chk = np.random.default_rng(0).integers(0, combo.shape[0], 200)
    for _m in _chk:
        d_ = (combo[pred_idx[_m]] != combo[_m][None, :]).sum(1)
        ok_ = (d_ <= 1) | (pred_lp[_m] < -1e17)
        assert ok_.all(), "선행자 표가 깨졌다"
    print("상태 수 %s  ->  결합상태 %d개 · 선행자 %d개 (13.85 [1] 로 성기게, 자기검사 통과)"
          % ("/".join(str(x) for x in n_st), combo.shape[0], pred_idx.shape[1]))
    print("가동률 사전확률 duty=%.2f — **자료가 아니라 사전이다** (build_transitions 독스트링)"
          % a.duty)

    cache = real_windows(APPS, a.grid_s, "cpu")
    arms = ("CO-P", "CO-H", "FHMM-P", "FHMM-H")
    res = {k: {"acc": {}, "ident": {}, "pwr": {}, "recon": {}} for k in arms}

    from src.preprocessing import load_nilm_npz
    for stem, d in cache.items():
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        tcyc = np.round(d["t"] * FS).astype(int)
        base_p = float(d["p_base"])
        # 전부-꺼짐 **고조파** 기준선 — p_base 와 같은 방식, 사이클 단위 중앙
        import json
        ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
        tt = np.arange(len(H)) / FS
        off = np.ones(len(H), bool)
        for v in ev[stem]["intervals"].values():
            for t0, t1 in v.get("on", []):
                off &= ~((tt >= t0) & (tt < t1))
        hb = (np.median(H[off], axis=0) if off.sum() >= 60
              else np.zeros(H.shape[1], H.dtype))
        h = H[tcyc] - hb[None, :]
        obs_h = np.concatenate([h.real / scale, h.imag / scale], 1)
        obs_p = (d["p_obs"] - base_p)[:, None]

        for tag, pred, obs in (("P", tot_p[:, None], obs_p), ("H", tot_h, obs_h)):
            sq = emission_sq(pred, obs)                      # (T,M)
            co = np.argmin(sq, 1)
            # 관측 잡음 — CO 잔차로 자가 보정 (손잡이 아님)
            var = max(float(np.mean(sq[np.arange(len(co)), co])) / max(obs.shape[1], 1), 1e-9)
            fh = viterbi_sparse(-0.5 * sq / var, pred_idx, pred_lp)
            for name, path in (("CO-" + tag, co), ("FHMM-" + tag, fh)):
                st = combo[path]                             # (T,K)
                on = st > 0
                p = np.zeros_like(st, dtype=np.float64)
                for k in range(len(APPS)):
                    p[:, k] = pw[k][st[:, k]]
                ac, idn, pr, rc, _cf = score_arm(on, p, d, APPS)
                res[name]["acc"][stem] = ac
                res[name]["ident"][stem] = idn
                res[name]["pwr"][stem] = pr
                res[name]["recon"][stem] = rc
        print("  %s 끝" % stem, flush=True)

    # ── 표 ────────────────────────────────────────────────────────────────
    print()
    print("① 기기별 on/off — 그 파일에 있는 기기만, 기기별 평균 (모델과 같은 규약)")
    print("  %-10s %s" % ("", "".join("  %-9s" % x for x in arms)))
    allapps = sorted({a2 for k in arms for s in res[k]["acc"] for a2 in res[k]["acc"][s]})
    for a2 in allapps:
        line = "  %-10s" % a2[:10]
        for k in arms:
            v = [res[k]["acc"][s][a2] for s in res[k]["acc"] if a2 in res[k]["acc"][s]]
            line += "  %-9s" % ("%.4f" % np.mean(v) if v else "—")
        print(line)
    line = "  %-10s" % "**평균**"
    for k in arms:
        v = [x for s in res[k]["acc"] for x in res[k]["acc"][s].values()]
        line += "  %-9s" % ("%.4f" % np.mean(v) if v else "—")
    print(line)

    print()
    print("② 무리 안 신원 — 혼자 켜진 창에서 누구에게 전력을 가장 많이 줬나")
    for g in ("저항 무리", "SMPS 무리"):
        line = "  %-10s" % g
        for k in arms:
            v = [res[k]["ident"][s][g][0] for s in res[k]["ident"] if g in res[k]["ident"][s]]
            line += "  %-9s" % ("%.4f" % np.mean(v) if v else "—")
        print(line)

    print()
    print("③ 잔차 — |관측 − (예측합 + 기준선)| (W, 모든 창)")
    for lab, key in (("전체", "abs"), ("오븐/핫플 켜짐", "abs_heavy"), ("그 밖", "abs_light")):
        line = "  %-14s" % lab
        for k in arms:
            v = [res[k]["recon"][s][key] for s in res[k]["recon"]
                 if res[k]["recon"][s] and np.isfinite(res[k]["recon"][s].get(key, np.nan))]
            line += "  %-9s" % ("%.1f" % np.mean(v) if v else "—")
        print(line)

    print()
    print("④ 전력 — 혼자 켜진 창의 상대오차 중앙값 (기기별 평균 · 표본 얇음)")
    line = "  %-14s" % "상대오차"
    for k in arms:
        v = [x[0] for s in res[k]["pwr"] for x in res[k]["pwr"][s].values()]
        line += "  %-9s" % ("%.1f%%" % (100 * np.mean(v)) if v else "—")
    print(line)

    print()
    print("견줄 대상 (13.84.74 · 같은 채점기):")
    print("  cnn_v37  창별 0.8466 · 잔차 전체 40.7W · 오븐핫플 59.3W · 그밖 14.4W")
    print("  v36p_mask 사슬 0.8197 · 잔차 전체 69.5W · 오븐핫플 104.6W · 그밖 17.7W")


if __name__ == "__main__":
    main()
