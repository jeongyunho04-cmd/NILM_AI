# -*- coding: utf-8 -*-
"""실측 대 합성 격차 점수표 — **같은 구성·같은 자리·정상상태** 끼리 (13.83.25).

    python -X utf8 src/run_diag_gap.py [cnn_v30 ...]

실측 test_1~5 에서 60초 내내 모든 라벨 기기의 상태가 일정한 창을 구성(켜진 기기 집합)별로 모으고,
합성기에 `force_active=구성, full_window_placement=True` 로 같은 구성의 정상상태 창을 만들게 한다
(자리는 기저 전압으로 맞춘다: D < 222V ≤ E). 구성마다 실측 ≥ 5창이 있을 때만 견준다.

내는 것 (구성별):
  전력(창 **평균** — 듀티 부하는 중앙값이 휴지 값에 앉는다)·|I_h| 중앙값의 실측/합성 비 (h1·3·5·7·13) · 절대 각도 차(h3·5·7, °) · 창 안 σ|I3| 비 · σP 비
  그리고 판이 주어지면 **게이트 불일치** — 기기별 |실측 게이트 중앙 − 합성 게이트 중앙| 의 최대와 그 기기.

⚠ 라벨 규약: 핫플·오븐은 듀티 휴지도 ON 이다. 합성 오븐은 캐리어 세션(팬·조명 + 히터)이라 실측과 같은
   규약이다. 합성 핫플의 `gt_is_on` 은 휴지에서 0 으로 내려가는 결함(13.82)이 있어 양 끝만 검사한다.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401

import torch

from src.model.inputs import build_inputs, VOLT_ORDERS
from src.model.realdata import RealWindows, WINDOW_CYCLES
from src.preprocessing import load_nilm_npz
from src.preprocessing.file_registry import site_of
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

MODELS = sys.argv[1:] or ["cnn_v30"]
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
W, STRIDE = WINDOW_CYCLES, 120
MIN_REAL, N_SYN, MAX_TRIES = 5, 30, 1500
DUTY = ("hotplate", "oven")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ── 실측: 구성별 정상상태 창 ────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
real = defaultdict(list)
for stem in FILES:
    spec = ev.get(stem)
    if spec is None:
        continue
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    x = RealWindows._to_33ch(raw)
    iv = np.asarray(raw["is_valid"]).astype(bool)
    n = x.shape[1]
    apps = spec["appliances_present"]

    def mk(app, key="on"):
        m = np.zeros(n, bool)
        for t0, t1 in spec["intervals"].get(app, {}).get(key, []):
            m[int(t0 * 60):int(min(t1 * 60, n))] = True
        return m

    on = {a: mk(a) for a in apps}
    unc = np.zeros(n, bool)
    for a in apps:
        unc |= mk(a, "uncertain")
    for t0 in range(0, n - W + 1, STRIDE):
        sl = slice(t0, t0 + W)
        if not iv[sl].all() or unc[sl].any():
            continue
        if any(on[a][sl].std() != 0 for a in apps):
            continue
        comp = tuple(sorted(a for a in apps if on[a][t0]))
        if not comp:
            continue
        real[(comp, site_of(stem))].append(x[:, sl])
keys = [k for k, v in real.items() if len(v) >= MIN_REAL]
keys.sort(key=lambda k: -len(real[k]))
print("실측 정상상태 창 구성 %d개 (≥%d창): %s"
      % (len(keys), MIN_REAL, ", ".join("%s@%s×%d" % ("+".join(a[:4] for a in c), s, len(real[(c, s)])) for c, s in keys)))

# ── 합성: 같은 구성·자리·정상상태 ────────────────────────────────────────────
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
np.random.seed(0)


def synth(comp, site):
    out, tries = [], 0
    while len(out) < N_SYN and tries < MAX_TRIES:
        tries += 1
        try:
            s = gen.synthesize_random_window(window_size_cycles=W, force_active=list(comp),
                                             full_window_placement=True, compute_gt_harmonics=False,
                                             target_lookahead_cycles=360, sustained_power_limit_w=None)
        except Exception:
            continue
        v = s.metadata.get("base_voltage_v", 999)
        if (site == "D") != (v < 222.0):
            continue
        if set(s.active_appliances) != set(comp):
            continue
        ok = True
        for a in comp:
            g = s.gt_is_on[a]
            ok &= (g[:60].all() and g[-60:].all()) if a in DUTY else bool(g.all())
        if not ok:
            continue
        vh = np.asarray(s.voltage_harmonics_complex)[:, [h - 1 for h in VOLT_ORDERS]]
        out.append(np.concatenate([s.harmonics_ri[:, :, 0].T, s.harmonics_ri[:, :, 1].T,
                                   s.power_features[:, 0:1].T, s.power_features[:, 1:2].T,
                                   s.power_features[:, 4:5].T, vh.real.T, vh.imag.T], 0).astype(np.float32))
    return out, tries


def ph(X):
    return X[:, 0:15] + 1j * X[:, 15:30]


def stats(X):
    c = ph(X)
    mag = 1e3 * np.median(np.abs(c), 2)                       # (B,15) 창 중앙 크기
    ang = np.rad2deg(np.angle(np.median(c.real, 2) + 1j * np.median(c.imag, 2)))
    sig3 = 1e3 * np.std(np.abs(c[:, 2]), 1)
    return dict(P=float(np.median(X[:, 30].mean(1))), mag=np.median(mag, 0), ang=np.median(ang, 0),
                sig3=np.median(sig3), sigP=np.median(X[:, 30].std(1)), n=len(X))


def dang(a, b):
    d = (a - b + 180) % 360 - 180
    return d


models = {}
for nm in MODELS:
    try:
        models[nm] = load_model("results/%s.pt" % nm, dev)
    except Exception as e:
        print("  %s 건너뜀 (%s)" % (nm, str(e)[:60]))


@torch.no_grad()
def gates(model, X):
    f, w = build_inputs(np.stack(X))
    G = []
    for i in range(0, len(f), 32):
        o = model(torch.from_numpy(f[i:i + 32]).to(dev), torch.from_numpy(w[i:i + 32]).to(dev))
        G.append(torch.sigmoid(o["on_logit"]).cpu().numpy())
    return np.median(np.concatenate(G), 0)


print("\n%-34s %2s %4s %4s | %6s | %5s %5s %5s %5s %5s | %5s %5s %5s | %5s %5s |%s"
      % ("구성", "자리", "실측", "합성", "P비", "|I1|", "|I3|", "|I5|", "|I7|", "|I13|", "Δ∠3", "Δ∠5", "Δ∠7", "σI3비", "σP비",
         "".join(" %s 게이트불일치(기기)" % m.replace("cnn_", "") for m in models)))
print("  (비 = 실측/합성 중앙값 · Δ∠ = 실측−합성 절대각도 °)")
rows = []
for comp, site in keys:
    R = real[(comp, site)]
    S, tries = synth(comp, site)
    if len(S) < 5:
        print("%-34s %2s %4d %4d | 합성 창 부족 (시도 %d)" % ("+".join(a[:4] for a in comp), site, len(R), len(S), tries))
        continue
    r, s = stats(np.stack(R)), stats(np.stack(S))
    ratio = r["mag"] / np.maximum(s["mag"], 1e-6)
    da = dang(r["ang"], s["ang"])
    line = ("%-34s %2s %4d %4d | %6.2f | %5.2f %5.2f %5.2f %5.2f %5.2f | %+5.0f %+5.0f %+5.0f | %5.2f %5.2f |"
            % ("+".join(a[:4] for a in comp), site, r["n"], s["n"], r["P"] / max(s["P"], 1e-6),
               ratio[0], ratio[2], ratio[4], ratio[6], ratio[12], da[2], da[4], da[6],
               r["sig3"] / max(s["sig3"], 1e-6), r["sigP"] / max(s["sigP"], 1e-6)))
    for nm, (model, apps, ck) in models.items():
        gr, gs = gates(model, R), gates(model, S)
        d = np.abs(gr - gs)
        j = int(np.argmax(d))
        line += "  %.2f (%s %.2f→%.2f)" % (d[j], apps[j][:4], gs[j], gr[j])
    print(line)
    rows.append(dict(comp=list(comp), site=site, n_real=int(r["n"]), n_syn=int(s["n"]), P_ratio=float(r["P"] / max(s["P"], 1e-6)),
                     mag_ratio=[float(v) for v in ratio], dang=[float(v) for v in da], sig3_ratio=float(r["sig3"] / max(s["sig3"], 1e-6)),
                     sigP_ratio=float(r["sigP"] / max(s["sigP"], 1e-6))))
json.dump(rows, open("results/_diag_gap.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n저장: results/_diag_gap.json")
