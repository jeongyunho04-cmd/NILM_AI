# -*- coding: utf-8 -*-
"""1단계 모델의 게이트가 **풀의 녹화들 사이에서** 얼마나 일관된가 (13.84 ④).

    python -X utf8 src/run_diag_recordings.py minipc cnn_v32 cnn_v31

한 기기의 녹화를 하나만 남긴 풀(`exclude_activation_files` 로 나머지를 뺀다)로 **학습 믹스·플래그 그대로**
창을 만들고, 그 기기가 타깃 시점에 ON 인 창의 게이트 분포를 녹화별·자리별로 낸다. 몸통 z 로 녹화를
맞히는 선형 probe 도 같이 낸다 — 학습에 쓴 녹화들 사이에서 이미 게이트가 갈리면, 제자리 녹화(실측)는
"다섯 번째 녹화" 일 뿐이다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.model import traincache
from src.model.inputs import build_inputs
from src.run_gate_check import load_model
from src.synthesis.augmentor import (FLOAT_FILL_PRESETS, POWER_SCALE_STD_PRESETS,
                                     STATE_MIX_PRESETS, STEADY_CROP_PRESETS)
from src.run_recipe_mix_probe import PRESETS as MIX_PRESETS

APP = sys.argv[1]
MODELS = sys.argv[2:] or ["cnn_v32"]
N_ON, MAX_DRAW = 320, 2500
SIB = {"minipc": ("laptop_charger", "beam_projector"), "laptop_charger": ("minipc", "beam_projector"),
       "beam_projector": ("minipc", "laptop_charger")}[APP]


def make_gen(exclude):
    traincache._init(
        "processed_data/npz", 3600, "train", 0,
        exclude_files_json=json.dumps({APP: exclude}) if exclude else "",
        recipe_mix_json=json.dumps(MIX_PRESETS["steady2"]),
        power_scale_std_json=json.dumps(POWER_SCALE_STD_PRESETS["measured"]),
        sp_curves=True, sp_per_texture=True, vtail=True,
        state_mix_json=json.dumps(STATE_MIX_PRESETS["minipc_balanced"]),
        carrier_apps=("oven",), couple_ext=True, smps_focus_off_p=0.4,
        float_fill_json=json.dumps(FLOAT_FILL_PRESETS["charger_float"]),
        steady_crop_json=json.dumps(STEADY_CROP_PRESETS["smps_steady"]),
        standby_jitter_cap=95.0)
    return traincache._GEN


def draw(gen, j, sib_idx):
    np.random.seed(20260911)
    X, V, P, S = [], [], [], []
    for _ in range(MAX_DRAW):
        smp, _ = gen._synthesize_window()
        t = gen._format_targets(smp)
        if t["y_on"][j] < 0.5:
            continue
        X.append(gen._format_inputs(smp)); V.append(smp.metadata.get("base_voltage_v", np.nan))
        P.append(t["y_power"][j]); S.append(int(any(t["y_on"][k] > 0.5 for k in sib_idx)))
        if len(X) >= N_ON:
            break
    return np.stack(X), np.asarray(V), np.asarray(P), np.asarray(S)


def main():
    torch.set_num_threads(4)
    from src.synthesis.segment_pool import SegmentPool
    pool0 = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    recs = sorted({a.source_file for a in pool0.appliance_activations[APP]})
    del pool0
    data = {}
    for r in recs:
        gen = make_gen([x for x in recs if x != r])
        j = gen.appliance_list.index(APP)
        sib_idx = [gen.appliance_list.index(s) for s in SIB]
        data[r] = draw(gen, j, sib_idx)
        print("  녹화 %-18s ON 창 %d (자리 D %d · E %d, 형제 ON %.0f%%, 전력 중앙 %.1fW)" % (
            r, len(data[r][0]), (data[r][1] < 222).sum(), (data[r][1] >= 222).sum(),
            100 * data[r][3].mean(), np.median(data[r][2])), flush=True)
    for name in MODELS:
        model, apps, _ = load_model("results/%s.pt" % name, "cpu")
        j = apps.index(APP)
        zs = {}
        hook = model.trunk.register_forward_hook(lambda m, i, o: zs.__setitem__("z", o.detach()))
        print("\n== %s · %s — 녹화별 게이트 (그 기기가 타깃 시점 ON 인 합성 창)" % (name, APP))
        print("   %-18s %5s | %6s %6s | %8s %8s %8s | %7s | %6s %6s" % (
            "녹화", "창", "게이트", "미탐%", "로짓p10", "p50", "p90", "형제ON시", "자리D", "자리E"))
        Z, R = [], []
        for r, (X, V, P, S) in data.items():
            f, w = build_inputs(X)
            outs, zz = [], []
            for i in range(0, len(f), 128):
                with torch.no_grad():
                    o = model(torch.from_numpy(np.ascontiguousarray(f[i:i + 128])),
                              torch.from_numpy(np.ascontiguousarray(w[i:i + 128])))
                outs.append(o["on_logit"][:, j].numpy()); zz.append(zs["z"].numpy())
            lg = np.concatenate(outs); z = np.concatenate(zz)
            g = 1 / (1 + np.exp(-lg))
            d_, e_ = V < 222, V >= 222
            print("   %-18s %5d | %6.3f %6.1f | %8.2f %8.2f %8.2f | %7.3f | %6.3f %6.3f" % (
                r, len(lg), g.mean(), 100 * (g < 0.5).mean(), *np.percentile(lg, [10, 50, 90]),
                g[S == 1].mean() if (S == 1).any() else float("nan"),
                g[d_].mean() if d_.any() else float("nan"), g[e_].mean() if e_.any() else float("nan")))
            Z.append(z); R += [r] * len(z)
        hook.remove()
        Z = np.concatenate(Z); R = np.asarray(R)
        if len(set(R)) > 1:
            acc = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), Z, R, cv=5).mean()
            print("   몸통 z → 녹화 식별 probe 정확도 %.3f (우연 %.3f)" % (acc, 1 / len(set(R))))


if __name__ == "__main__":
    main()
