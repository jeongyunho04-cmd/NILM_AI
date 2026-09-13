# -*- coding: utf-8 -*-
"""형제 회전을 켜면 생성기가 **실패 ② 의 창**을 만들 수 있는가 — CPU 용량 검사 (13.84.9).

    python -X utf8 src/run_diag_rotcheck.py cnn_v32 [cnn_v31]

학습 믹스·플래그 그대로인 생성기 둘(회전 off / on, 같은 시드)로 창을 만들고, **충전기와 프로젝터가 ON
이고 미니PC 는 OFF 인 창**(실패 ② 의 구성)만 골라 v32 의 미니PC 게이트를 낸다. 회전 c 를 고정값으로
掃引해 용량-반응 곡선도 낸다. 실측 test_3/4 의 같은 구성 창은 게이트 0.84~0.91 (13.84.3) — 회전이 그 근처로
끌어올리면 "생성기가 이제 그 창을 만든다" 이고, 그 창에 미니PC OFF 라벨이 붙어 학습되면 지름길이 닫힌다.
반대로 안 움직이면 실측 편차는 이 회전이 아니다 ([[check-the-generator-can-make-the-failing-window]]).
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model import traincache
from src.model.inputs import build_inputs
from src.run_gate_check import load_model
from src.synthesis.augmentor import (FLOAT_FILL_PRESETS, POWER_SCALE_STD_PRESETS,
                                     STATE_MIX_PRESETS, STEADY_CROP_PRESETS)
from src.run_recipe_mix_probe import PRESETS as MIX_PRESETS

MODELS = sys.argv[1:] or ["cnn_v32"]
N_WIN, MAX_DRAW = 240, 4000
SIBS = ("laptop_charger", "beam_projector")


def make_gen(rot_cfg):
    traincache._init(
        "processed_data/npz", 3600, "train", 0,
        recipe_mix_json=json.dumps(MIX_PRESETS["steady2"]),
        power_scale_std_json=json.dumps(POWER_SCALE_STD_PRESETS["measured"]),
        sp_curves=True, sp_per_texture=True, vtail=True,
        state_mix_json=json.dumps(STATE_MIX_PRESETS["minipc_balanced"]),
        carrier_apps=("oven",), couple_ext=True, smps_focus_off_p=0.4,
        float_fill_json=json.dumps(FLOAT_FILL_PRESETS["charger_float"]),
        steady_crop_json=json.dumps(STEADY_CROP_PRESETS["smps_steady"]),
        standby_jitter_cap=95.0,
        sibling_rotate_json=json.dumps(rot_cfg) if rot_cfg else "")
    return traincache._GEN


def draw(gen, seed):
    """충전기·프로젝터 ON, 미니PC OFF 인 창만. 시드를 같이 걸어 off/on 이 같은 배경·구성을 밟게 한다."""
    np.random.seed(seed)
    apps = gen.appliance_list
    j, kc, kp = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")
    X, V, P = [], [], []
    for _ in range(MAX_DRAW):
        smp, _ = gen._synthesize_window()
        t = gen._format_targets(smp)
        if t["y_on"][j] > 0.5 or t["y_on"][kc] < 0.5 or t["y_on"][kp] < 0.5:
            continue
        X.append(gen._format_inputs(smp)); V.append(smp.metadata.get("base_voltage_v", np.nan))
        P.append(t["y_power"][kc] + t["y_power"][kp])
        if len(X) >= N_WIN:
            break
    return np.stack(X), np.asarray(V), np.asarray(P)


def gate_of(model, apps, X):
    f, w = build_inputs(X)
    out = []
    for i in range(0, len(f), 128):
        with torch.no_grad():
            o = model(torch.from_numpy(np.ascontiguousarray(f[i:i + 128])),
                      torch.from_numpy(np.ascontiguousarray(w[i:i + 128])))
        out.append(torch.sigmoid(o["on_logit"]).numpy())
    return np.concatenate(out)


def main():
    torch.set_num_threads(6)
    cfgs = [("off", None)]
    for c in (2.5, 5.0, 7.5, 10.0, 15.0):
        cfgs.append(("c=±%g" % c, {a: {"p": 1.0, "c_max": c} for a in SIBS}))
    # 고정 c (掃引): c_max 를 쓰되 U(-c,c) 가 아니라 부호만 무작위인 고정 크기 — 용량-반응이 또렷하다.
    data = {}
    for name, cfg in cfgs:
        gen = make_gen(cfg)
        X, V, P = draw(gen, 20260911)
        data[name] = (X, V, P)
        print("  %-8s 창 %d (자리 D %d · E %d, 형제 전력 중앙 %.0fW)" % (
            name, len(X), (V < 222).sum(), (V >= 222).sum(), np.median(P)), flush=True)
    for m in MODELS:
        model, apps, _ = load_model("results/%s.pt" % m, "cpu")
        j, kc, kp = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")
        print("\n== %s · 충전기+프로젝터 ON, 미니PC OFF 인 합성 창 — 형제 회전에 따른 게이트" % m)
        print("   실측 같은 구성(test_3/4 형제ON·미니PC OFF): 미니PC 게이트 0.842 / 0.910, 충전기 0.99, 프로젝터 0.76~0.90")
        print("   %-8s %5s | %8s %8s %8s | %8s %8s | %6s %6s" % (
            "회전", "창", "미니PC", "유령%", "p90", "충전기", "프로젝터", "자리D", "자리E"))
        for name, (X, V, P) in data.items():
            g = gate_of(model, apps, X)
            d_, e_ = V < 222, V >= 222
            print("   %-8s %5d | %8.3f %8.1f %8.3f | %8.3f %8.3f | %6.3f %6.3f" % (
                name, len(g), g[:, j].mean(), 100 * (g[:, j] > 0.5).mean(), np.percentile(g[:, j], 90),
                g[:, kc].mean(), g[:, kp].mean(),
                g[d_, j].mean() if d_.any() else float("nan"), g[e_, j].mean() if e_.any() else float("nan")))


if __name__ == "__main__":
    main()
