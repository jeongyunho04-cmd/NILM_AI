# -*- coding: utf-8 -*-
"""1단계 지도학습의 목적함수를 항별로 해부한다 (13.84).

    python -X utf8 src/run_diag_stage1.py --syn <dir_train> <dir_holdout> cnn_v32 cnn_v31

무엇을 재나
-----------
① 손실 항별 **몫** — 캐시 학습 이력(`results/<tag>.json`)에서 가중 합 대비 항별 비율의 궤적.
② 합성 소표본(앞 80% / 뒤 20%)에서 항별 손실·게이트 AUC — 합성 안의 일반화 격차.
③ 최종 모델에서 항별 **파라미터 기울기 노름** — 학습 끝에서 갱신을 실제로 미는 항.
④ 창마다 게이트 로짓에 걸리는 기울기를 항별로 분해 — BCE(라벨) / 전력 / 고조파(라벨 없음).
   합성에서는 셋이 같은 쪽을 밀어야 정상이다 (자 검사).
⑤ **실측** 창(test_1~4, stride 30)에서 같은 분해. 고조파 항은 라벨이 없어도 계산되므로
   "학습 목적함수가 실측에서 무엇을 원하는가" 를 라벨과 대면할 수 있다.
   덧붙여 미니PC 게이트를 0/1 로 못 박았을 때의 L_harm 을 창마다 내어 최소가 어느 쪽인지 본다.

⚠ 손실은 `run_train_cnn` 과 같은 구성으로 만든다 (`--harm-even-magnitude --state-signatures
   --standby-operating session --w-z 0.3 --w-over 0.0`, 상태별 척도). 배경 플래그는 캐시 meta 를 따른다.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.net import (harmonic_scales, harmonic_signatures, harmonic_signatures_by_state,
                           noise_signature, standby_signatures)
from src.model.realdata import dense_targets
from src.model.traincache import CachedWindows
from src.run_baseline import S_I
from src.run_gate_check import load_model
from src.run_train_cnn import to_targets

W_ = dict(power=1.0, state=0.3, on=0.3, plugged=0.1, standby=0.1, harm=0.1, z=0.3)
SMPS = ("minipc", "laptop_charger", "beam_projector")


def build_crit(apps, background=False):
    """`run_train_cnn.main` 의 손실 구성을 그대로 되짚는다."""
    from src.model.companion import standby_operating_signatures
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb_sig = standby_signatures(pool, apps)
    sb_op, sb_pw, sb_used = standby_operating_signatures(pool, apps, only=SESSION_PLUGGED_APPS)
    for x in sb_used:
        sb_sig[apps.index(x)] = sb_op[apps.index(x)]
    nz_sig = noise_signature(pool)
    if background:
        from src.synthesis.sp_curves import background_signature
        nz_sig = nz_sig + background_signature()
    h_scale = harmonic_scales(pool, apps)
    sig_state, _ = harmonic_signatures_by_state(pool, apps)
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig), standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig), harm_scale=torch.from_numpy(h_scale),
        signatures_state=torch.from_numpy(sig_state), harm_even_magnitude=True,
        weights=LossWeights(harm=W_["harm"], cons=0.0, over=0.0, state_power=0.0, z=W_["z"]),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    )
    crit.eval()
    return crit


def harm_per_window(crit, out, obs, on_logit=None):
    """창별 L_harm (B,). `on_logit` 을 주면 그것으로 게이트를 다시 만든다 (기울기용)."""
    ol = out["on_logit"] if on_logit is None else on_logit
    g = torch.sigmoid(ol)
    praw = out["power_raw"]
    power = g * praw
    o2 = dict(out); o2["on_logit"] = ol; o2["power"] = power
    pred = crit._harm_pred_active(o2, power)
    idle = torch.sigmoid(out["plugged_logit"]) * (1.0 - g)
    pred = pred + torch.einsum("bk,khc->bhc", idle, crit.standby_sig) + crit.noise_sig[None]
    err = crit._harm_err(pred, obs, crit._coherent_even_w(power))
    m = crit.harm_mask
    return (err * m[None, :, None]).mean((1, 2)) / m.mean().clamp(min=1e-6)


def power_per_window(crit, out, tgt, on_logit):
    """창별·기기별 L_power (B,K) — `forward` 와 같은 척도."""
    idx = tgt["y_state"].long().clamp(0, crit.s_state.shape[1] - 1)
    s = torch.gather(crit.s_state[None].expand(idx.shape[0], -1, -1), 2, idx[..., None]).squeeze(-1)
    power = torch.sigmoid(on_logit) * out["power_raw"]
    d = power / s - tgt["y_power"] / s
    a = d.abs(); dl = crit.power_delta
    return torch.where(a <= dl, 0.5 * d * d, dl * (a - 0.5 * dl))


def gate_grads(crit, out, obs, y_on=None, tgt=None):
    """창별 게이트 로짓 기울기 (B,K) 를 항별로. 가중치까지 곱한 값이다 (배치 평균의 1/B 는 뺐다).

    양수 = 경사하강이 로짓을 **내린다**(OFF 쪽), 음수 = 올린다(ON 쪽).
    """
    det = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    ol = det["on_logit"].clone().requires_grad_(True)
    lh = harm_per_window(crit, det, obs, on_logit=ol)
    g_h, = torch.autograd.grad(lh.sum(), ol)
    res = {"harm": W_["harm"] * g_h, "L_harm": lh.detach()}
    if y_on is not None:
        res["on"] = W_["on"] * (torch.sigmoid(det["on_logit"]) - y_on)
    if tgt is not None:
        ol2 = det["on_logit"].clone().requires_grad_(True)
        lp = power_per_window(crit, det, tgt, ol2)
        g_p, = torch.autograd.grad(lp.sum(), ol2)
        res["power"] = W_["power"] * g_p
    return res


def harm_sweep(crit, out, obs, j, gates=(0.0, 1.0)):
    """기기 j 의 게이트만 값으로 못 박고 창별 L_harm 을 낸다 → {g: (B,)}."""
    det = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    res = {}
    for g in gates:
        ol = det["on_logit"].clone()
        ol[:, j] = float(np.log(max(g, 1e-6) / max(1 - g, 1e-6)))
        res[g] = harm_per_window(crit, det, obs, on_logit=ol).numpy()
    return res


def fwd(model, fine, wide, bs=256):
    outs = []
    for i in range(0, len(fine), bs):
        o = model(torch.from_numpy(np.ascontiguousarray(fine[i:i + bs])),
                  torch.from_numpy(np.ascontiguousarray(wide[i:i + bs])))
        outs.append({k: v.detach() for k, v in o.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


# ── ① 손실 항별 몫 (학습 이력) ───────────────────────────────────────────────
def section_history(tag):
    try:
        d = json.load(open("results/%s.json" % tag, encoding="utf-8"))
    except FileNotFoundError:
        return
    h = d["history"]
    print("\n① %s 학습 이력 — 가중 합 대비 항별 몫 (epoch: 몫)" % tag)
    keys = ["power", "state", "on", "plugged", "standby", "harm", "z"]
    print("   %5s %7s | " % ("ep", "total") + " ".join("%6s" % k for k in keys) + " | 원시 harm  F1")
    for r in h[:1] + h[4:5] + h[9:10] + h[14:15] + h[-1:]:
        L = r["loss"]; tot = L["total"]
        print("   %5d %7.4f | " % (r["epoch"], tot)
              + " ".join("%6.3f" % (W_[k] * L[k] / tot) for k in keys)
              + " | %7.3f  %.4f" % (L["harm"], r["f1"]))


# ── ②③④ 합성 ────────────────────────────────────────────────────────────────
def section_synthetic(model, crit, apps, dirs, n_grad=1024):
    J = apps.index("minipc")
    rows = {}
    for name, d in dirs.items():
        c = CachedWindows(d)
        idx = np.arange(len(c))
        batch = c.batch(idx)
        fine, wide, tgt = to_targets(tuple(torch.from_numpy(x) for x in batch), "cpu")
        with torch.no_grad():
            out = fwd(model, fine.numpy(), wide.numpy())
            parts = crit(out, tgt)
        auc = {}
        for j, a in enumerate(apps):
            y = tgt["y_on"][:, j].numpy()
            if len(set(y)) == 2:
                auc[a] = roc_auc_score(y, out["on_logit"][:, j].numpy())
        rows[name] = (parts, auc, out, tgt, fine, wide)
    print("\n② 합성 소표본 — 앞 80%%(train) 대 뒤 20%%(holdout), 각 %d창" % len(idx))
    keys = ["power", "state", "on", "plugged", "standby", "harm", "z", "total"]
    print("   %-8s " % "" + " ".join("%7s" % k for k in keys))
    for name, (parts, *_) in rows.items():
        print("   %-8s " % name + " ".join("%7.4f" % float(parts[k]) for k in keys))
    print("   게이트 AUC        " + " ".join("%-15s" % a[:15] for a in apps))
    for name, (_, auc, *_) in rows.items():
        print("   %-8s          " % name + " ".join("%-15.4f" % auc.get(a, float("nan")) for a in apps))

    # ③ 파라미터 기울기 노름 (train 소표본 앞 n_grad 창)
    _, _, _, _, fine, wide = rows["train"]
    fine, wide = fine[:n_grad], wide[:n_grad]
    tgt = {k: (v[:n_grad] if torch.is_tensor(v) else v) for k, v in rows["train"][3].items()}
    model.train(False)
    out = model(fine, wide)
    parts = crit(out, tgt)
    params = [p for p in model.parameters() if p.requires_grad]
    trunk = list(model.trunk.parameters())
    head_on = [model.heads[J].weight]
    print("\n③ 최종 모델에서 항별 기울기 노름 (합성 train %d창, 가중 곱함)" % n_grad)
    print("   %-8s %10s %10s %14s %8s" % ("항", "‖∇전체‖", "‖∇몸통‖", "‖∇미니PC on행‖", "cos(총)"))
    g_tot = torch.autograd.grad(parts["total"], params, retain_graph=True)
    flat_tot = torch.cat([g.reshape(-1) for g in g_tot])
    for k in ["power", "state", "on", "plugged", "standby", "harm", "z"]:
        L = W_[k] * parts[k]
        g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        g = [torch.zeros_like(p) if x is None else x for p, x in zip(params, g)]
        flat = torch.cat([x.reshape(-1) for x in g])
        gt = torch.autograd.grad(L, trunk, retain_graph=True, allow_unused=True)
        gt = [torch.zeros_like(p) if x is None else x for p, x in zip(trunk, gt)]
        gh = torch.autograd.grad(L, head_on, retain_graph=True, allow_unused=True)[0]
        gh_on = torch.zeros(1) if gh is None else gh[model.i_on]
        cos = float(F.cosine_similarity(flat[None], flat_tot[None]))
        print("   %-8s %10.4f %10.4f %14.5f %8.3f" % (
            k, float(flat.norm()), float(torch.cat([x.reshape(-1) for x in gt]).norm()),
            float(gh_on.norm()), cos))

    # ④ 창별 게이트 기울기 분해 (자 검사)
    print("\n④ 합성 창별 게이트 로짓 기울기 — 항별 부호 일치율과 크기 (holdout 소표본)")
    _, _, out, tgt, fine, wide = rows["holdout"]
    gr = gate_grads(crit, out, tgt["obs_harm"], y_on=tgt["y_on"], tgt=tgt)
    y = tgt["y_on"].numpy()
    print("   %-16s %6s | %-27s | %-27s | %s" % ("기기", "창", "harm: ON쪽% (참ON / 참OFF)", "power: ON쪽% (참ON / 참OFF)", "|harm|/|BCE| 중앙"))
    for j, a in enumerate(apps):
        gh, gp, gb = gr["harm"][:, j].numpy(), gr["power"][:, j].numpy(), gr["on"][:, j].numpy()
        on, off = y[:, j] > 0.5, y[:, j] <= 0.5
        if on.sum() < 5:
            continue
        r = np.median(np.abs(gh) / np.maximum(np.abs(gb), 1e-9))
        print("   %-16s %6d | %8.1f%% / %8.1f%%          | %8.1f%% / %8.1f%%          | %.2f" % (
            a, len(y), 100 * (gh[on] < 0).mean(), 100 * (gh[off] < 0).mean(),
            100 * (gp[on] < 0).mean(), 100 * (gp[off] < 0).mean(), r))
    return rows


# ── ⑤ 실측 ───────────────────────────────────────────────────────────────────
def real_labels(stem, apps, t, ev):
    spec = ev[stem]; n = int(t.max()) + 1
    y = np.zeros((len(t), len(apps)), np.float32)
    unc = np.zeros((len(t), len(apps)), bool)

    def mask(app, key="on"):
        m = np.zeros(n + 3600, bool)
        for a, b in spec["intervals"].get(app, {}).get(key, []):
            m[int(a * 60):int(b * 60)] = True
        return m[t]
    for j, a in enumerate(apps):
        if a in spec["appliances_present"]:
            y[:, j] = mask(a); unc[:, j] = mask(a, "uncertain")
    return y, unc, spec


def section_real(model, crit, apps, stems, stride=30):
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    J, KC, KP = [apps.index(x) for x in SMPS]
    print("\n⑤ 실측 창 — 학습 목적함수의 게이트 기울기 (라벨 없는 harm 항) 을 사람 라벨과 대면")
    print("   ON쪽% = 경사하강이 그 기기 로짓을 **올리는** 창의 비율. 참이 ON 이면 100 이, OFF 면 0 이 옳다.")
    print("   L_harm(g=0/1) = 미니PC 게이트만 0 / 1 로 못 박은 창별 L_harm 의 평균 — 작은 쪽이 손실이 원하는 답.")
    hdr = "   %-6s %-26s %5s | %6s %7s %7s | %7s %7s %7s | %8s %8s %6s"
    print(hdr % ("파일", "부분집합", "창", "게이트", "hON%", "bceON%", "chgON%", "prjON%", "|h|/|b|", "Lh(g=0)", "Lh(g=1)", "g1<g0%"))
    for stem in stems:
        rw = dense_targets(stem, stride=stride)
        idx = np.arange(len(rw))
        f, w, pobs, oh, pn = rw.batch(idx)
        out = fwd(model, f, w)
        obs = torch.from_numpy(oh)
        y, unc, spec = real_labels(stem, apps, rw.target_cycle, ev)
        gr = gate_grads(crit, out, obs, y_on=torch.from_numpy(y))
        sw = harm_sweep(crit, out, obs, J)
        gate = torch.sigmoid(out["on_logit"]).numpy()
        gh, gb = gr["harm"].numpy(), gr["on"].numpy()
        sib = np.zeros(len(y), bool)
        for k in (KC, KP):
            sib |= y[:, k] > 0.5
        mp_on, mp_off = y[:, J] > 0.5, (y[:, J] <= 0.5) & ~unc[:, J]
        subsets = [("미니PC ON", mp_on), ("미니PC OFF", mp_off),
                   ("형제ON·미니PC OFF", sib & mp_off), ("형제ON·미니PC ON", sib & mp_on),
                   ("형제OFF·미니PC OFF", ~sib & mp_off)]
        if "minipc" not in spec["appliances_present"]:
            subsets = [("미니PC 없음 (전부 OFF)", mp_off)]
        for name, m in subsets:
            if m.sum() < 5:
                continue
            r = np.median(np.abs(gh[m, J]) / np.maximum(np.abs(gb[m, J]), 1e-9))
            print(hdr % (stem, name, m.sum(),
                         "%.3f" % gate[m, J].mean(), "%.1f" % (100 * (gh[m, J] < 0).mean()),
                         "%.1f" % (100 * (gb[m, J] < 0).mean()),
                         "%.1f" % (100 * (gh[m, KC] < 0).mean()), "%.1f" % (100 * (gh[m, KP] < 0).mean()),
                         "%.2f" % r, "%.3f" % sw[0.0][m].mean(), "%.3f" % sw[1.0][m].mean(),
                         "%.0f" % (100 * (sw[1.0][m] < sw[0.0][m]).mean())))
        # 형제 열도 같은 식으로 — 충전기/프로젝터가 ON 인 창에서 harm 이 그것을 끄려 드는가
        for k, nm in ((KC, "충전기"), (KP, "프로젝터")):
            on = (y[:, k] > 0.5) & ~unc[:, k]; off = (y[:, k] <= 0.5) & ~unc[:, k]
            if on.sum() >= 5:
                print("   %-6s %-26s %5d |   게이트 %.3f · harm ON쪽 %.1f%% (참ON) / %.1f%% (참OFF, n=%d)" % (
                    stem, nm + " 열", on.sum(), gate[on, k].mean(),
                    100 * (gh[on, k] < 0).mean(), 100 * (gh[off, k] < 0).mean() if off.sum() else float("nan"), off.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["cnn_v32"])
    ap.add_argument("--syn", nargs=2, metavar=("TRAIN_DIR", "HOLDOUT_DIR"))
    ap.add_argument("--stems", nargs="*", default=["test_1", "test_2", "test_3", "test_4"])
    a = ap.parse_args()
    torch.set_num_threads(8)
    for tag in a.models:
        section_history(tag)
    for tag in a.models:
        print("\n" + "=" * 100 + "\n== %s" % tag)
        model, apps, ck = load_model("results/%s.pt" % tag, "cpu")
        crit = build_crit(apps)
        if a.syn:
            section_synthetic(model, crit, apps, {"train": a.syn[0], "holdout": a.syn[1]})
        section_real(model, crit, apps, a.stems)


if __name__ == "__main__":
    main()
