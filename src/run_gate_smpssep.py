# -*- coding: utf-8 -*-
"""**SMPS 셋이 차수마다 갈리나** — 분리기를 짓기 전에 재는 자 (14.381).

사용자 지적에서 나왔다: *"상수를 정해줘서 맞추지 말고 고조파를 맞추도록 자유도를
주는 건 어때"*. 방향은 맞다. 그런데 §46 이 기각한 `--w-harm-smps` 가 실패한 이유가
**차수를 안 재고 h3 부터 다 넣은 것**이라, 같은 실수를 두 번 하지 않으려고 먼저 잰다
([[measure-separability-before-building-a-separator]]).

```
  §46.4 가 잰 것 — 상수 sig 가 못 담는 폭 (사이클별 |I_h|/P 의 사분위폭 / |sig|)
    h3·h5   0.01~0.12  상수가 거의 정확 -> **자유도를 줘도 배울 게 없다**
    h9·h11  0.17~0.56  무너진다          -> 여기가 자유도가 필요한 자리
  이 관문이 묻는 것 — 그 차수에서 **세 SMPS 가 서로 갈리기는 하나?**
    안 갈리면 자유도를 줘 봐야 **맞바꿈 방향**만 하나 더 늘린다 (§46.2 가 잰 그것)
```

판별력은 **d′** 로 잰다 (§12.133 과 같은 규약): 기기 사이 거리를 기기 안 산포로 나눈다.
```
  d′_ij(h) = |sig_i[h] − sig_j[h]| / sqrt( (s_i[h]^2 + s_j[h]^2) / 2 )
  s 를 두 가지로 잰다 — **사이클 안**(낙관) 과 **녹화 사이**(비관).
  실제로 중요한 것은 녹화 사이다: 한 녹화 안에서만 좁은 것은 상수가 아니라
  그 녹화의 성질이다 (`net.reactive_signatures` 가 같은 이유로 cross 검사를 한다)
```

    python -X utf8 -m src.run_gate_smpssep
"""
from typing import Dict, List
import numpy as np

from src import env_guard  # noqa: F401

from src.model.net import harmonic_signatures_by_state  # noqa: E402
from src.model.postproc import SMPS_GROUP  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
RES = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
KO = {"beam_projector": "빔", "laptop_charger": "충전기", "minipc": "미니PC"}
ORD = list(range(1, 16))
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-50s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def per_cycle(pool) -> Dict[str, dict]:
    """기기 -> {상태: (중앙 페이저, 사이클내 산포, 녹화간 산포, n사이클, n녹화)}.

    ⚠ 산포는 **로버스트**로 잡는다 (IQR/1.349). 돌입전류·릴레이 과도가 꼬리를 만든다.
    """
    out: Dict[str, dict] = {}
    for a in APPS:
        by_s: Dict[int, dict] = {}
        for act in pool.appliance_activations.get(a, []):
            hh = np.asarray(act.net_harmonics_ri, np.float64)
            pp = np.asarray(act.target_power_w, np.float64)
            st = np.asarray(act.state_id, np.int64)
            if hh.shape[0] != len(pp) or len(st) != len(pp):
                continue
            c = hh[:, :, 0] + 1j * hh[:, :, 1]
            for sid in np.unique(st):
                m = (st == sid) & (pp > 1.0)
                if m.sum() < 20:
                    continue
                r = c[m] / pp[m, None]                       # (n,15) 와트당
                d = by_s.setdefault(int(sid), {"cyc": [], "rec": []})
                d["cyc"].append(r)
                d["rec"].append(np.median(r, 0))             # 이 녹화의 중앙
        for sid, d in by_s.items():
            C = np.concatenate(d["cyc"])
            R = np.stack(d["rec"])                           # (R,15)
            qr = np.percentile(C.real, [25, 75], axis=0)
            qi = np.percentile(C.imag, [25, 75], axis=0)
            s_cyc = np.sqrt((((qr[1] - qr[0]) / 1.349) ** 2
                             + ((qi[1] - qi[0]) / 1.349) ** 2) / 2)
            if len(R) > 1:
                s_rec = np.sqrt((np.std(R.real, 0) ** 2 + np.std(R.imag, 0) ** 2) / 2)
            else:
                s_rec = np.full(15, np.nan)
            out.setdefault(a, {})[sid] = (np.median(C, 0), s_cyc, s_rec, len(C), len(R))
    return out


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, us = harmonic_signatures_by_state(pool, APPS)
    sc = sig[..., 0] + 1j * sig[..., 1]
    P = per_cycle(pool)
    SM = [a for a in APPS if a in SMPS_GROUP]
    print("SMPS 차수별 판별력 관문 (14.381) — SMPS %s\n" % [KO.get(a, a) for a in SM])

    main_s = {}
    for a in SM:
        cand = [(v[3], s_) for s_, v in P.get(a, {}).items()
                if s_ < sc.shape[1] and us[APPS.index(a), s_]]
        if cand:
            main_s[a] = max(cand)[1]
    if len(main_s) < 3:
        print("**SMPS 주 상태를 못 찾았다** %s" % main_s)
        return 1
    print("  주 상태 — " + " · ".join(
        "%s s%d (%d사이클, 녹화 %d)" % (KO[a], main_s[a], P[a][main_s[a]][3],
                                     P[a][main_s[a]][4]) for a in SM) + "\n")

    # [0] 위상 기준이 기기 사이에서 맞나
    ang = {}
    for a in RES:
        for s_, v in P.get(a, {}).items():
            if s_ < sc.shape[1] and us[APPS.index(a), s_]:
                ang["%s s%d" % (a[:8], s_)] = float(np.degrees(np.angle(v[0][0])))
    chk(0, "⚠ **위상 기준이 기기 사이에서 맞나** (저항 ∠I1 ~ 0)",
        bool(ang) and max(abs(x) for x in ang.values()) < 5.0,
        "저항 ∠I1 = %s — 전압 고정 기준이라 **복소로 견줄 수 있다.** 5도 넘으면 "
        "아래 d′ 가 전부 무의미하다 ([[fix-the-phase-reference-before-comparing-phasors]])"
        % {k: round(v, 2) for k, v in ang.items()})

    # [1] 차수별 d′ 표
    print("\n  차수별 판별력 d′ — 기기 사이 거리 / 기기 안 산포")
    print("  %4s | %-27s| %-27s" % ("차수", " 사이클 안 산포로 (낙관)",
                                    " **녹화 사이 산포로 (비관)**"))
    print("  %4s | %8s %8s %8s | %8s %8s %8s"
          % ("", "빔-충전", "빔-미니", "충전-미니", "빔-충전", "빔-미니", "충전-미니"))
    pairs = [(SM[0], SM[1]), (SM[0], SM[2]), (SM[1], SM[2])]
    D = {"cyc": np.full((15, 3), np.nan), "rec": np.full((15, 3), np.nan)}
    for hi, h in enumerate(ORD):
        row = {"cyc": [], "rec": []}
        for i, (x, y) in enumerate(pairs):
            mx, sxc, sxr = P[x][main_s[x]][0], P[x][main_s[x]][1], P[x][main_s[x]][2]
            my, syc, syr = P[y][main_s[y]][0], P[y][main_s[y]][1], P[y][main_s[y]][2]
            gap = abs(mx[h - 1] - my[h - 1])
            for kk, sx, sy in (("cyc", sxc, syc), ("rec", sxr, syr)):
                den = np.sqrt((sx[h - 1] ** 2 + sy[h - 1] ** 2) / 2)
                v = gap / den if np.isfinite(den) and den > 0 else np.nan
                D[kk][hi, i] = v
                row[kk].append(v)
        print("  h%-3d | %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f"
              % (h, *row["cyc"], *row["rec"]))

    # [2] ★★ **제일 약한 쌍**이 어디서 갈리나 — 거기가 맞바꿈 방향이다
    #: ⚠ 처음에 이 칸을 *"h9·h11 이 h3·h5 보다 나은가"* 로 적었다. **전제가 틀렸다** —
    #  §41.3 의 "판별 차수 h9·h11" 은 **미니PC↔충전기** 쌍의 이야기이고, 제일 약한
    #  쌍은 **빔↔충전기**인데 그 쌍은 차수가 올라갈수록 **나빠진다**. 약한 쌍으로 묻는다.
    #: ⚠⚠ 그리고 두 번째로 틀렸다 — 처음엔 **차수 전부**에서 최소를 찾았더니
    #  `충전기-미니PC` 가 약한 쌍으로 뽑혔다 (h8 에서 0.26). 그런데 **짝수차는 SMPS
    #  신호가 애초에 0** 이라 거기의 d′ 는 잡음/잡음이다. 크기가 없는 자리의 판별력은
    #  뜻이 없다. **홀수차 + 크기 바닥**으로 거른다 (`--pow-rel-floor` 와 같은 생각).
    MAG1 = np.mean([abs(P[a][main_s[a]][0][0]) for a in SM])
    live = [i for i in range(15) if ORD[i] >= 3 and ORD[i] % 2 == 1
            and np.mean([abs(P[a][main_s[a]][0][i]) for a in SM]) > 0.02 * MAG1]
    wk = int(np.nanargmin(np.nanmin(D["rec"][live], 0)))
    wname = "%s-%s" % (KO[pairs[wk][0]], KO[pairs[wk][1]])
    wcol = D["rec"][:, wk]
    w_ok = [ORD[i] for i in live if np.isfinite(wcol[i]) and wcol[i] >= 1.0]
    chk(2, "★★ **제일 약한 쌍**(%s)이 갈리는 차수가 있나" % wname, bool(w_ok),
        "%s d′ — h3 %.2f · h5 %.2f · h7 %.2f · h9 %.2f · **h11 %.2f** · h13 %.2f · "
        "h15 %.2f  ⇒ d′>=1 인 차수 **%s**. ⚠ 이 쌍은 고차로 갈수록 **나빠진다** — "
        "§46 이 잰 빔↔충전기 맞바꿈(−36.5%%/+20.8%%)이 바로 이 방향이다"
        % (wname, wcol[2], wcol[4], wcol[6], wcol[8], wcol[10], wcol[12], wcol[14],
           w_ok or "**없음**"))

    # [3] d′ >= 1 인 차수가 있나
    #: ⚠ `live` 로 거른다 — 짝수차는 크기가 0 이라 d′ 가 잡음/잡음이다. 처음에
    #  안 걸렀더니 h10·h12 가 "갈린다" 고 뽑혔다.
    good = [ORD[i] for i in live if np.isfinite(D["rec"][i]).all()
            and np.nanmin(D["rec"][i]) >= 1.0]
    dead = [ORD[i] for i in live if ORD[i] not in good]
    chk(3, "★★ **세 쌍이 다 갈리는 차수**가 있나 (홀수차 · 녹화 사이 d′>=1)", bool(good),
        "쓸 수 있는 차수 **%s** / 못 쓰는 차수 **%s** — 못 쓰는 쪽이 %s 때문이다. "
        "여기가 자유도를 줘도 맞바꿈이 안 생기는 자리다"
        % (good or "**없음**", dead or "없음", wname))

    # [4] 저항 넷이 h9 에서 영방향인가
    rows, zr = [], True
    for a in RES:
        for s_, v in P.get(a, {}).items():
            if s_ >= sc.shape[1] or not us[APPS.index(a), s_]:
                continue
            r = float(np.abs(v[0][8]) / max(np.abs(v[0][0]), 1e-12))
            rows.append("%s s%d %.4f" % (a[:8], s_, r))
            zr &= (r < 0.02)
    chk(4, "저항 넷이 **h9 에서 영방향**인가 (|I9|/|I1| < 2%)", zr,
        " · ".join(rows) + " — §12.122.11 의 *저항은 h>=3 이 거의 영이라 잔차를 "
        "통째로 흡수한다* 가 지금 자료에서도 맞나")

    # [5] ★ **잡음으로 백색화**한 조건수 — 생 조건수는 잡음을 못 본다
    #: ⚠ 처음에 열만 정규화하고 조건수를 쟀다가 "고른 차수가 더 나쁘다(15.3 대 13.7)"
    #  가 나왔다. **자가 틀렸다** — 조건수는 열이 독립인지만 보고 그 행이 **잡음인지는
    #  안 본다.** 차수를 더 넣으면 행이 늘어 조건수는 좋아지는데, 그 행이 잡음이면
    #  실제로는 맞바꿈 방향만 는다. 행을 **그 차수의 산포로 나눠** 재는 것이 맞다
    #  ([[check-conditioning-before-believing-a-fit]] — 스케일이 조건수를 지배한다).
    def design(hs, whiten):
        hs = list(hs)
        if not hs:
            return float("inf")
        rows, wts = [], []
        for h in hs:
            sd = float(np.mean([P[a][main_s[a]][2][h - 1] for a in SM]))
            w = (1.0 / sd) if (whiten and np.isfinite(sd) and sd > 0) else 1.0
            for part in (0, 1):
                rows.append([(P[a][main_s[a]][0][h - 1].real if part == 0
                              else P[a][main_s[a]][0][h - 1].imag) for a in SM])
                wts.append(w)
        A = np.asarray(rows) * np.asarray(wts)[:, None]
        n = np.linalg.norm(A, axis=0, keepdims=True)
        return float(np.linalg.cond(A / n)) if np.all(n > 0) else float("inf")

    allh = list(range(3, 16))
    c_raw_all, c_raw_sel = design(allh, False), design(good, False)
    c_w_all, c_w_sel = design(allh, True), design(good, True)
    chk(5, "★ **잡음으로 백색화**하면 자르지 않아도 되나", c_w_all <= c_w_sel * 1.05,
        "조건수 — 생 것: h3~h15 **%.1f** · 고른 차수 **%.1f** (고른 쪽이 나쁘다, 행이 "
        "적어서다) / **백색화: h3~h15 %.1f · 고른 차수 %.1f** ⇒ 잡음으로 가중하면 "
        "차수를 **자르지 않아도** 된다. 자르기보다 **1/σ² 가중**이 맞는 처방이다"
        % (c_raw_all, c_raw_sel, c_w_all, c_w_sel))

    # [6] ★★ 판별력이 **크기에 있나 위상에 있나** (사용자 질문, 14.382)
    #: `_harm_err` 는 **홀수차를 복소 그대로** 잰다 (`d = pred - obs`) — 위상이 이미
    #: 손실에 들어 있다. 그러니 묻는 것은 *"넣을 수 있나"* 가 아니라
    #: *"판별력이 실제로 위상에 있나"* 다. d′ 를 두 갈래로 쪼갠다:
    #: ```
    #:   d′_크기 = | |s_i| - |s_j| |                    / sigma
    #:   d′_위상 = |∠s_i - ∠s_j| * (|s_i|+|s_j|)/2      / sigma      <- 각도 차의 호 길이
    #: ```
    print("\n  판별력을 쪼갠다 — 크기 몫 대 **위상 몫** (녹화 사이 산포로)")
    print("  %4s | %-23s| %-23s" % ("차수", " 크기만 d′", " **위상만 d′**"))
    print("  %4s | %7s %7s %7s | %7s %7s %7s"
          % ("", "빔-충전", "빔-미니", "충전-미니", "빔-충전", "빔-미니", "충전-미니"))
    Mg = np.full((15, 3), np.nan)
    Ph = np.full((15, 3), np.nan)
    for i_ in live:
        h = ORD[i_]
        rm, rp = [], []
        for k, (x, y) in enumerate(pairs):
            mx, my = P[x][main_s[x]][0][h - 1], P[y][main_s[y]][0][h - 1]
            sx, sy = P[x][main_s[x]][2][h - 1], P[y][main_s[y]][2][h - 1]
            den = float(np.sqrt((sx ** 2 + sy ** 2) / 2))
            da = abs(np.angle(mx) - np.angle(my))
            da = min(da, 2 * np.pi - da) * (abs(mx) + abs(my)) / 2
            Mg[i_, k] = abs(abs(mx) - abs(my)) / den if den > 0 else np.nan
            Ph[i_, k] = da / den if den > 0 else np.nan
            rm.append(Mg[i_, k])
            rp.append(Ph[i_, k])
        print("  h%-3d | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f" % (h, *rm, *rp))
    n_ph = int(np.nansum(Ph[live] > Mg[live]))
    n_tot = int(np.isfinite(Ph[live]).sum())
    chk(6, "★★ SMPS 판별력이 **위상에 더 많이** 있나 (사용자 질문)",
        n_ph > n_tot / 2,
        "홀수차 쌍 %d 칸 중 **%d 칸(%.0f%%)** 에서 위상 몫이 크기 몫보다 크다. "
        "약한 쌍(%s)만 보면 크기 %s · 위상 %s ⇒ %s"
        % (n_tot, n_ph, 100 * n_ph / max(n_tot, 1), wname,
           np.round(Mg[live, wk], 2).tolist(), np.round(Ph[live, wk], 2).tolist(),
           "**위상이 살린다**" if np.nanmean(Ph[live, wk]) > np.nanmean(Mg[live, wk])
           else "**위상으로도 안 살아난다**"))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
