# -*- coding: utf-8 -*-
"""`src/model/gbudget.py` 의 관문 — 네 칸이 **진짜 자료 위에서** 서는가 (14.335).

스크래치에서 옮긴 코드다. 옮기면서 수치가 바뀌면 그건 옮긴 게 아니라 **다시 짠 것**이다.
그래서 [7][8] 이 실측을 다시 돌려 14.322·14.334 의 수를 되찾는지 본다.

    python -X utf8 -m src.run_gate_gbudget [--stride 30] [--files test_1,…]

⚠ 사용자 지적으로 생긴 관문이 [4] 다 — *"양쪽 다 막아야 해"*. 거부권과 바닥이 같은
  창에서 동시에 서면 두 처치가 서로를 지운다. 구조적으로 배타적이라고 **믿지 말고** 잰다.
"""
import argparse
import glob
import sys

import numpy as np

from src import env_guard  # noqa: F401

from src.model import gbudget as GB  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
VRE0, NV = 33, 15
#: §12.5 — *"유령 띠 자리 **전부 포트 ∩ 드라이 겹침 구간 안**"*. 게이트가 무너지는 자리다.
BAND = {"test_2": [(160.8, 178.8), (231.4, 260.1)], "test_5": [(82.5, 143.5)]}
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-40s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def raw63(d):
    """실측 npz -> 63채널 (1, 63, N). `to_raw45` 와 같은 순서, 전압만 15차수."""
    I = np.asarray(d["harmonics_complex"], complex)
    V = np.asarray(d["voltage_harmonics_complex"], complex)
    pf = np.asarray(d["power_features"], np.float64)
    return np.concatenate([I.real.T, I.imag.T, pf[:, 0:1].T, pf[:, 1:2].T, pf[:, 4:5].T,
                           V.real.T, V.imag.T], 0).astype(np.float64)[None]


def table_from_recordings():
    """표를 **녹화에서 다시 짓는다** — 상수를 상수로 검산하면 아무것도 안 잰다."""
    out = {}
    for a in GB.RESISTIVE:
        d = {}
        for p in sorted(set(glob.glob("processed_data/npz/%s*.npz" % a)
                            + glob.glob("processed_data/npz/%s*.npz"
                                        % a.replace("electiric", "electric")))):
            z = np.load(p, allow_pickle=True)
            on, st = z["is_on"].astype(bool), z["state_id"]
            V = np.abs(np.asarray(z["voltage_harmonics_complex"])[:, 0])
            P = np.asarray(z["power_features"], np.float64)[:, 0]
            for s in sorted(set(st[on].tolist())):
                m = on & (st == s) & (P > 100)
                if m.sum() >= 200:
                    d.setdefault(int(s), []).append(
                        1e3 * float(np.median(P[m] / np.maximum(V[m] ** 2, 1e-9))))
        out[a] = {s: float(np.median(v)) for s, v in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=30, help="0.5초 격자 (실시간 stride)")
    ap.add_argument("--files", default="test_1,test_2,test_3,test_4,test_5")
    a = ap.parse_args()
    files = [x for x in a.files.split(",") if x]
    print("컨덕턴스 예산 관문 (14.335) — 파일 %s · 격자 %d사이클" % (",".join(files), a.stride))

    # ── [1] 셋 다 끄면 비트 동일 ────────────────────────────────────────────
    rng = np.random.default_rng(0)
    P0 = rng.random((256, len(APPS))) * 1500.0
    g0 = rng.random(256) * 60.0
    v0 = 210.0 + rng.random(256) * 20.0
    q, _ = GB.apply(P0, g0, v0, APPS, veto=False, floor=False, pin=False)
    chk(1, "셋 다 끄면 **비트 동일**", np.array_equal(q, P0),
        "최대차 %.3e (0 이어야 한다)" % float(np.abs(q - P0).max()))

    # ── [2] 표가 녹화와 맞는가 ──────────────────────────────────────────────
    live = table_from_recordings()
    bad, lines = [], []
    for app, d in GB.PIN_MS.items():
        for s, g in d.items():
            got = live.get(app, {}).get(s)
            if got is None:
                bad.append("%s s%d 없음" % (app, s))
                continue
            e = 100 * abs(got - g) / g
            lines.append("%s s%d %.3f/%.3f (%.2f%%)" % (app[:6], s, g, got, e))
            if e > 1.0:
                bad.append("%s s%d %.2f%%" % (app, s, e))
    chk(2, "표가 **녹화에서 다시 지은 값**과 맞나", not bad,
        "표/재측 (차%%): %s%s" % (" · ".join(lines), "" if not bad else "  **%s**" % bad))

    # ── [3] 드라이 반파가 갈려 있나 ─────────────────────────────────────────
    hd = dict(GB.app_states("hair_dryer"))
    hp = GB.max_ms("hotplate")
    ok3 = (len(hd) == 2 and abs(min(hd.values()) - 9.45) < 0.5
           and abs(min(hd.values()) - hp) < 1.0 and max(hd.values()) > 15.0)
    chk(3, "드라이 **반파/전파**가 갈려 있나", ok3,
        "드라이 %s · 핫플 %.2f — 반파와 핫플이 %.2f mS 차다 (14.329: 0.50mS 가 최소 간격). "
        "뭉개면 14.327 처럼 전력오차가 146 -> 628W 로 터진다"
        % (" ".join("s%d %.2f" % (s, g) for s, g in sorted(hd.items())), hp,
           abs(min(hd.values()) - hp)))

    # ── [5] 고정이 정말 G·V² 를 내나 ────────────────────────────────────────
    K = APPS.index("electiric_kettle")
    Pp = np.zeros((3, len(APPS)))
    Pp[:, K] = 1400.0
    vv = np.array([210.0, 222.0, 230.0])
    on = np.zeros(Pp.shape, bool)
    on[:, K] = True
    got = GB.pin_power(Pp, vv, APPS, on)[:, K]
    want = GB.PIN_MS["electiric_kettle"][1] * 1e-3 * vv ** 2
    chk(5, "고정이 `G·V²` 를 **소수점까지** 내나", np.allclose(got, want, atol=1e-9),
        "210/222/230V -> %s W (기대 %s)"
        % (" ".join("%.2f" % x for x in got), " ".join("%.2f" % x for x in want)))

    # ── [6] 저항 아닌 기기는 안 바뀐다 ──────────────────────────────────────
    q, _ = GB.apply(P0, g0, v0, APPS)
    nonres = [j for j, x in enumerate(APPS) if x not in GB.PIN_MS]
    chk(6, "저항 아닌 기기는 **한 칸도 안 바뀐다**",
        np.array_equal(q[:, nonres], P0[:, nonres]),
        "안 건드릴 기기 %d개 최대차 %.3e"
        % (len(nonres), float(np.abs(q[:, nonres] - P0[:, nonres]).max())))

    # ── 실측 배선 (진짜 객체를 짓는다) ──────────────────────────────────────
    from src.evaluation.real_events import build_on_off_truth, load_events
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    ev = load_events()
    G, KON, VETO, FLOOR, INB, STEM = [], [], [], [], [], []
    KSOLO, DUTY = [], []
    solo_err = []
    for stem in files:
        d = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
        X = raw63(d)
        n = X.shape[-1]
        apps = sorted(ev[stem]["intervals"].keys())
        on_, sc = build_on_off_truth(stem, apps, n, events=ev)
        on_, sc = np.asarray(on_, bool), np.asarray(sc, bool)
        idx = {x: apps.index(x) for x in apps}
        V0 = (np.median(X[0, VRE0:VRE0 + NV], 1)
              + 1j * np.median(X[0, VRE0 + NV:VRE0 + 2 * NV], 1))
        bud = GB.Budget(APPS, V0, pool=pool, volt_re0=VRE0)
        t = np.arange(0, n, a.stride)
        t = t[sc[t].all(1)]
        g = bud.g_sum(X[:, :, t])[0]
        v1 = np.abs(X[0, VRE0][t] + 1j * X[0, VRE0 + NV][t])
        ptot = X[0, 30][t]

        #: 모델이 없으므로 **참 라벨로 만든 가짜 예측**을 넣는다 — 관문이 재는 것은
        #: 마스크의 **상호배타성**과 **헛세움**이지 모델 성적이 아니다.
        fake = np.zeros((len(t), len(APPS)))
        for x in GB.RESISTIVE:
            if x in idx:
                fake[:, APPS.index(x)] = 1000.0 * on_[t, idx[x]]
        vm = GB.veto_mask(fake, g, APPS)
        #: ⚠ 바닥은 **모델이 껐을 때만** 선다. 라벨이 완벽한 가짜 예측에 걸면 영원히 0 이다
        #:   (첫 판에서 내가 그렇게 재서 회수 0.0% 를 봤다 — 자가 틀린 것이다).
        #:   그래서 §12.3 의 **붕괴**를 흉내낸다: 포트 게이트가 0.002 로 무너진 그 판
        #:   (`leak4_s0`)에서는 포트 예측이 사실상 0 이다.
        collapsed = fake.copy()
        collapsed[:, APPS.index("electiric_kettle")] = 0.0
        fm = GB.floor_mask(collapsed, g, APPS)
        kon = on_[t, idx["electiric_kettle"]] if "electiric_kettle" in idx else np.zeros(len(t), bool)
        #: 포트만 켜진 시각 — [7] 의 모집단이다. 다른 저항이 켜져 있으면 Ĝ 는 **합**이라
        #: 명판보다 위로 치우친다 (첫 판에서 38.07 을 보고 실패로 읽었다 — 자가 틀렸다).
        ksolo = kon.copy()
        for x in GB.RESISTIVE:
            if x != "electiric_kettle" and x in idx:
                ksolo &= ~on_[t, idx[x]]
        KSOLO.append(ksolo)
        #: ★ 라벨이 완벽한데도 거부권이 서는 칸 = **듀티로 꺼진 순간**이다.
        #:   학습에 순시 거부권을 넣으면 이만큼이 `L_on` 과 싸운다 (사용자 물음).
        for x in GB.RESISTIVE:
            if x in idx:
                j2 = APPS.index(x)
                DUTY.append((x, int(on_[t, idx[x]].sum()), int(vm[:, j2].sum())))
        ts = t / 60.0
        inb = np.zeros(len(t), bool)
        for lo, hi in BAND.get(stem, []):
            inb |= (ts >= lo) & (ts <= hi)
        G.append(g); KON.append(kon); VETO.append(vm); FLOOR.append(fm)
        INB.append(inb); STEM += [stem] * len(t)

        #: 한 대만 통전인 시각에서는 관측 전력이 곧 그 기기 전력이다 -> 고정을 채점할 수 있다
        #: ⚠ **다른 저항만 빼면 안 된다.** 첫 판에서 비저항(SMPS·선풍기·에어컨)을 안 빼서
        #:   드라이 150W · 핫플 426W 가 나왔다 — 그건 고정 오차가 아니라 남의 전력이었다.
        #: ⚠ 그리고 문턱을 600W 로 고정하면 **핫플(455W)은 혼자서 못 넘는다** — 통과한 6칸이
        #:   전부 "남이 켜진" 칸이었다. 기기 명목의 절반으로 잡는다.
        for x in GB.RESISTIVE:
            if x not in idx:
                continue
            j = APPS.index(x)
            others = [idx[y] for y in apps if y != x]
            nom = GB.max_ms(x) * 1e-3 * float(np.median(v1)) ** 2
            m = (on_[t, idx[x]] & (~on_[t][:, others].any(1) if others else True)
                 & (ptot > 0.5 * nom))
            if m.sum() < 5:
                continue
            onm = np.zeros(fake.shape, bool)
            onm[m, j] = True
            pinned = GB.pin_power(fake, v1, APPS, onm)[m, j]
            solo_err.append((x, int(m.sum()),
                             float(np.median(np.abs(pinned - ptot[m])))))

    g = np.concatenate(G); kon = np.concatenate(KON)
    vm = np.concatenate(VETO); fm = np.concatenate(FLOOR); inb = np.concatenate(INB)
    ksolo = np.concatenate(KSOLO)

    # ── [4] ★ 거부권과 바닥이 동시에 안 선다 ────────────────────────────────
    both = int((vm & fm).sum())
    chk(4, "★ **거부권과 바닥이 동시에 안 선다**", both == 0,
        "겹친 칸 **%d개** / 잰 칸 %d — 거부권 %d · 바닥 %d. "
        "겹치면 두 처치가 서로를 지운다 (사용자 지적)"
        % (both, vm.size, int(vm.sum()), int(fm.sum())))

    # ── [7] Ĝ 가 실측에서 명판을 되찾나 ─────────────────────────────────────
    gk = GB.PIN_MS["electiric_kettle"][1]
    med = float(np.median(g[ksolo])) if ksolo.sum() > 20 else float("nan")
    chk(7, "`Budget` 이 실측에서 **명판을 되찾나**",
        ksolo.sum() > 20 and abs(med - gk) / gk < 0.05,
        "**포트만** 켜진 %d칸의 Ĝ 중앙 **%.2f mS** (표 %.2f · 차 **%.2f%%**). "
        "다른 저항이 같이 켜진 칸을 넣으면 합이라 38 mS 로 치우친다 — 모집단을 가른다"
        % (int(ksolo.sum()), med, gk, 100 * abs(med - gk) / gk))

    # ── [8] ★ 14.334 의 실측 수치를 되찾나 ──────────────────────────────────
    KJ = APPS.index("electiric_kettle")
    fa = float(fm[~kon, KJ].mean()) if (~kon).any() else 1.0
    rb = float(fm[inb & kon, KJ].mean()) if (inb & kon).any() else 0.0
    chk(8, "★ 바닥이 **14.334 의 실측 수치**를 되찾나", fa <= 0.0010 and rb >= 0.95,
        "헛세움 **%.3f%%** (%d/%d · 기준 <=0.10%%) · §12.5 유령 띠 안 회수 **%.1f%%** "
        "(%d칸 · 기준 >=95%%)"
        % (100 * fa, int(fm[~kon, KJ].sum()), int((~kon).sum()),
           100 * rb, int((inb & kon).sum())))

    # ── [9] 고정이 단독창에서 참 전력을 내나 ────────────────────────────────
    worst = max((e for _, _, e in solo_err), default=1e9)
    chk(9, "고정이 **단독 통전 창**에서 참 전력을 내나", worst < 60.0,
        " · ".join("%s n=%d 중앙오차 **%.1fW**" % (x, n, e) for x, n, e in solo_err)
        + "  (기준 <60W — 14.327 의 포트 65W 와 같은 급)")

    # ── 참고: 순시 거부권이 **라벨과 싸우는 양** (학습에 넣을 때의 대가) ────────
    print()
    print("  참고 — 라벨이 **완벽한데도** 거부권이 서는 칸 = 그 순간 통전이 아닌 칸이다.")
    print("         학습에 **순시** 거부권을 넣으면 이만큼이 `L_on` 과 싸운다 (사용자 물음).")
    agg = {}
    for x, non, nv in DUTY:
        p_, q_ = agg.get(x, (0, 0))
        agg[x] = (p_ + non, q_ + nv)
    for x in GB.RESISTIVE:
        if x not in agg:
            continue
        non, nv = agg[x]
        print("           %-18s 라벨 ON %6d칸 · 거부권 %6d칸 = **%5.1f%%**"
              % (x, non, nv, 100 * nv / max(non, 1)))
    print("         => 12.9.8 이 총전력 프라이어에 **창 최대**를 쓴 것과 같은 이유다.")

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
