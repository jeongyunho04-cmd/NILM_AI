# -*- coding: utf-8 -*-
"""축소 충전기(`--float-fill charger_float`) 창을 캐시 경로 그대로 만들어 판별 자로 쓴다 (13.83.23).

    python -X utf8 src/run_diag_float_syn.py build 600          # 창을 만들어 results/_float_syn.npz 에 저장
    python -X utf8 src/run_diag_float_syn.py score cnn_v29 cnn_v30   # 저장된 창에 판을 대 본다

build: `traincache._init`(v29 캐시 설정 + charger_float) -> `_chunk` 로 N 창을 만들고, 충전기 ON ·
충전기 목표 전력 < 20W · 미니PC OFF 인 창(= 학습에 새로 들어가는 종류)과 대조로 충전기 ON ·
≥ 30W · 미니PC OFF 인 창을 뽑아 fine/wide 와 라벨을 저장한다.
score: 각 판의 미니PC·충전기 게이트 중앙값과 >0.5 몫. 재학습 전(v29)에는 축소 창이 실측 부동
충전기처럼 **미니PC** 로 읽혀야 하고(그래야 같은 구멍이다), 재학습 뒤에는 충전기로 읽혀야 한다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401

OUT = "results/_float_syn.npz"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"

if MODE == "build":
    from src.model import traincache as tc
    from src.run_recipe_mix_probe import PRESETS
    from src.synthesis.augmentor import FLOAT_FILL_PRESETS, POWER_SCALE_STD_PRESETS, STATE_MIX_PRESETS
    from src.synthesis.segment_pool import SegmentPool
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    apps = SegmentPool(npz_dir="processed_data/npz", time_split="train").get_appliance_types()
    J, KC = apps.index("minipc"), apps.index("laptop_charger")
    tc._init(npz_dir="processed_data/npz", window_cycles=3600, time_split="train", seed=0,
             exclude_files_json="", dither_amp=0.0, dither_phase_deg=0.0,
             recipe_mix_json=json.dumps(PRESETS["steady2"]),
             dither_even_amp=0.0, dither_even_phase_deg=0.0,
             power_scale_std_json=json.dumps(POWER_SCALE_STD_PRESETS["measured"]),
             sp_curves=True, sp_per_texture=True, vtail=True, background=False,
             level_scramble=None, state_mix_json=json.dumps(STATE_MIX_PRESETS["minipc_balanced"]),
             carrier_apps=("oven",), dither_min_order=2, couple_ext=True, smps_focus_off_p=0.4,
             float_fill_json=json.dumps(FLOAT_FILL_PRESETS["charger_float"]))
    F, Wd, YO, YP = [], [], [], []
    for i in range(0, N, 100):
        r = tc._chunk((7000 + i // 100, min(100, N - i)))     # 청크 번호를 캐시와 겹치지 않게
        F.append(r["fine"]); Wd.append(r["wide"]); YO.append(r["y_on"]); YP.append(r["y_power"])
    F, Wd, YO, YP = map(np.concatenate, (F, Wd, YO, YP))
    chg_on, mpc_off = YO[:, KC] == 1, YO[:, J] == 0
    lo = chg_on & mpc_off & (YP[:, KC] < 20.0)
    hi = chg_on & mpc_off & (YP[:, KC] >= 30.0)
    print("창 %d · 충전기 ON %d · 축소(<20W, 미니PC OFF) %d · 대조(≥30W, 미니PC OFF) %d"
          % (N, chg_on.sum(), lo.sum(), hi.sum()))
    print("  축소 창 충전기 P p10/50/90 %s W" % np.round(np.percentile(YP[lo, KC], [10, 50, 90]), 1))
    np.savez(OUT, fine_lo=F[lo], wide_lo=Wd[lo], fine_hi=F[hi], wide_hi=Wd[hi],
             p_lo=YP[lo, KC], p_hi=YP[hi, KC], apps=np.array(apps))
    print("저장:", OUT)
else:
    import torch
    from src.run_gate_check import load_model
    d = np.load(OUT)
    apps = list(d["apps"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sys.argv[2:] or ["cnn_v29"]
    print("  %-10s %-22s %5s %9s %9s %8s %8s %7s %7s" % ("판", "창", "n", "게이트mpc", "게이트chg", "mpc>0.5", "chg>0.5", "P mpc", "P chg"))
    for nm in names:
        model, mapps, ck = load_model("results/%s.pt" % nm, dev)
        J, KC = mapps.index("minipc"), mapps.index("laptop_charger")
        for tag, key in (("축소 충전기 <20W", "lo"), ("대조 충전기 ≥30W", "hi")):
            f, w = d["fine_" + key].astype(np.float32), d["wide_" + key].astype(np.float32)
            G, PW = [], []
            with torch.no_grad():
                for i in range(0, len(f), 64):
                    o = model(torch.from_numpy(f[i:i + 64]).to(dev), torch.from_numpy(w[i:i + 64]).to(dev))
                    G.append(torch.sigmoid(o["on_logit"]).cpu().numpy()); PW.append(o["power"].cpu().numpy())
            G, PW = np.concatenate(G), np.concatenate(PW)
            print("  %-10s %-22s %5d %9.3f %9.3f %7.0f%% %7.0f%% %7.1f %7.1f"
                  % (nm, tag, len(f), np.median(G[:, J]), np.median(G[:, KC]),
                     100 * (G[:, J] > 0.5).mean(), 100 * (G[:, KC] > 0.5).mean(),
                     np.median(PW[:, J]), np.median(PW[:, KC])))
