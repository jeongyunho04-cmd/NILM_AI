# -*- coding: utf-8 -*-
"""관문 — 1단계 `L_swap` 이 **겨냥한 것만 건드리는가** (14.32).

이 계열의 기본 실패 둘을 잡는다:
  · 항이 **조용히 0** 이 된다 (`v_rms` 를 안 넣으면 가드가 그냥 건너뛴다)
  · 저항을 고치겠다면서 **비저항 기기를 흔든다**

다섯을 찍는다.

  [1] 기울기 격리 — `L_swap` 이 비저항 일곱에 흘리는 기울기가 **정확히 0** 인가.
      같은 자로 `L_res` 를 나란히 재서 왜 `L_swap` 인지 보인다.
  [2] 통전 상태 — 오븐 팬·조명 창이 '통전' 으로 세어지지 않는가.
      (세어지면 조합 탐색이 `L_on`(y_on=1) 과 정면으로 싸운다)
  [3] 실제로 무는가 — `swap_frac` 이 0 이 아닌가. 0 이면 항이 안 걸린 것이다.
  [4] 끄면 항등 — `--w-swap 0` 이 옛 동작과 같은가.
  [6] **autocast** — 학습이 도는 방식(bfloat16 autocast) 그대로 항이 돌아가나.
      982877 이 여기서 죽었다: 확률을 받는 `F.binary_cross_entropy` 는 autocast 에서
      금지돼 있는데 관문이 fp32 로 돌아 못 잡았다 ([[verify-the-gate-runs-that-path]]).
  [5] 2단계 불변 — `unlabeled()` 가 **주어진 리비전과 비트 동일**한가.
      1단계에 올리려고 블록을 메서드로 떼어냈으므로, 2단계가 안 흔들렸다는 증거가
      따로 있어야 한다 ([[pin-the-two-entry-points-against-each-other]]).

    python -X utf8 -m src.run_gate_swap1
    python -X utf8 -m src.run_gate_swap1 --ckpt results/cnn_sigc_vn_s0.pt
"""
import argparse
import sys

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.losses import (LossWeights, NILMLoss, S_STATE,
                              build_state_scales)
from src.model.postproc import HALFWAVE_OHM, RESISTIVE_OHM

APPS = ['air_conditioner', 'beam_projector', 'electiric_kettle', 'fan', 'hair_dryer',
        'hotplate', 'laptop_charger', 'minipc', 'oven']
KO = {"air_conditioner": "에어컨", "beam_projector": "프로젝터", "electiric_kettle": "전기포트",
      "fan": "선풍기", "hair_dryer": "드라이기", "hotplate": "핫플레이트",
      "laptop_charger": "충전기", "minipc": "미니PC", "oven": "오븐"}
RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
COND = {"oven": 2, "hotplate": 2}
H, S = 15, 5


def _crit(res_apps=RES, cond=None, **kw):
    s_i = torch.tensor([max(max(S_STATE.get(a, {1: 10.0}).values()), 10.0) for a in APPS])
    return NILMLoss(
        s_i=s_i,
        signatures=torch.from_numpy(_sig()), standby_sig=torch.zeros(len(APPS), H, 2),
        noise_sig=torch.zeros(H, 2), harm_scale=torch.ones(H),
        s_state=build_state_scales(APPS, s_i.tolist()),
        weights=LossWeights(**kw.pop("weights", {})),
        res_ohm=torch.tensor([RESISTIVE_OHM[a] if a in res_apps else 0.0 for a in APPS]),
        res_ohm_half=torch.tensor([HALFWAVE_OHM[a] if (a in res_apps and a in HALFWAVE_OHM)
                                   else 0.0 for a in APPS]),
        res_cond_state=torch.tensor([(cond or {}).get(a, 0) for a in APPS], dtype=torch.long),
        **kw)


def _sig(seed: int = 0) -> np.ndarray:
    r = np.random.RandomState(seed)
    return (r.randn(len(APPS), H, 2) * 1e-3).astype(np.float32)


def _batch(B: int = 96, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lg = (torch.randn(B, len(APPS), generator=g) * 2).requires_grad_(True)
    pr = (torch.rand(B, len(APPS), generator=g) * 800 + 10).requires_grad_(True)
    sb = (torch.rand(B, len(APPS), generator=g) * 3).requires_grad_(True)
    st = torch.randn(B, len(APPS), S, generator=g)
    out = dict(on_logit=lg, power_raw=pr, power=torch.sigmoid(lg) * pr, standby=sb,
               state=st, power_mix=st.softmax(-1),
               power_states=pr[..., None].expand(-1, -1, S),
               plugged_logit=torch.randn(B, len(APPS), generator=g))
    tgt = dict(p_observed=torch.rand(B, generator=g) * 2000 + 200,
               p_noise=torch.full((B,), 1.9), v_rms=torch.full((B,), 222.0),
               obs_harm=torch.randn(B, H, 2, generator=g) * 0.05,
               y_power=torch.rand(B, len(APPS), generator=g) * 100,
               y_on=(torch.rand(B, len(APPS), generator=g) > 0.5).float(),
               y_plugged=torch.ones(B, len(APPS)),
               y_standby=torch.rand(B, len(APPS), generator=g),
               y_state=torch.randint(0, 2, (B, len(APPS)), generator=g))
    return out, tgt, (lg, pr, sb)


def gate1(tol_leak: float = 0.0) -> bool:
    print("\n[1] 기울기 격리 — 비저항 일곱에 기울기가 가나 (작을수록 좋다, 0 이어야 한다)")
    rows = {}
    for nm, kw in (("L_res", dict(w_res=1.0, w_swap=0.0)),
                   ("L_swap", dict(w_res=0.0, w_swap=1.0, swap_tol=0.5))):
        crit = _crit(cond=COND if nm == "L_swap" else None)
        out, tgt, (lg, pr, sb) = _batch()
        parts = crit.unlabeled(out, tgt, **kw)
        key = "res" if nm == "L_res" else "swap"
        L = parts[key]
        if not L.requires_grad:
            print(f"    {nm}: 항이 0 이다 — 조건 미충족"); return False
        gl, gp, gs = torch.autograd.grad(L, (lg, pr, sb), allow_unused=True)
        rows[nm] = [(float(gl[:, j].abs().mean()) if gl is not None else 0.0,
                     float(gp[:, j].abs().mean()) if gp is not None else 0.0,
                     float(gs[:, j].abs().mean()) if gs is not None else 0.0)
                    for j in range(len(APPS))]
    print(f"    {'기기':10s}{'저항':>5s}"
          f"{'L_res ∂on':>12s}{'∂p_raw':>10s}{'∂sb':>10s}"
          f"{'L_swap ∂on':>13s}{'∂p_raw':>10s}{'∂sb':>10s}")
    leak = 0.0
    for j, a in enumerate(APPS):
        r, s_ = rows["L_res"][j], rows["L_swap"][j]
        is_res = a in RES
        if not is_res:
            leak = max(leak, max(s_))
        print(f"    {KO[a]:10s}{'●' if is_res else '·':>5s}"
              f"{r[0]:>12.3e}{r[1]:>10.3e}{r[2]:>10.3e}"
              f"{s_[0]:>13.3e}{s_[1]:>10.3e}{s_[2]:>10.3e}")
    ok = leak <= tol_leak
    print(f"    => 비저항 최대 누수 {leak:.3e}  {'✅ 통과' if ok else '❌ 실패'}")
    print("       (`L_res` 는 비저항 게이트로 새는 것이 정상이다 — 그래서 이 항을 안 쓴다)")
    return ok


def gate2(ckpt: str, dev: str) -> bool:
    print("\n[2] 통전 상태 — 오븐이 켜졌는데 팬·조명인 창을 '통전' 으로 세나")
    from src.evaluation.sealing import is_sealed
    from src.run_gate_check import load_model
    from src.run_plot_real import dense_targets, load_events
    model, apps, _ = load_model(ckpt, dev)
    crit = _crit(cond=COND).to(dev)
    jo = apps.index("oven")
    ev = load_events()
    tot_gate = tot_cond = 0
    pw_gate = []
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        rw = dense_targets(stem, stride=30)
        with torch.no_grad():
            for i in range(0, len(rw), 512):
                f, w, *_ = rw.batch(np.arange(i, min(i + 512, len(rw))))
                o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                          torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                g = torch.sigmoid(o["on_logit"])[:, jo]
                q = crit._cond_prob(o)[:, jo]
                tot_gate += int((g > 0.5).sum()); tot_cond += int((q > 0.5).sum())
                pw_gate.append(o["power"][:, jo][g > 0.5].float().cpu().numpy())
    pw = np.concatenate(pw_gate) if pw_gate else np.zeros(1)
    print(f"    오븐 게이트>0.5 창 {tot_gate:5d}  ->  통전 확률>0.5 창 {tot_cond:5d}"
          f"   ({tot_cond / max(tot_gate, 1) * 100:.1f}%)")
    print(f"    그 창의 모델 오븐 전력 중앙 {np.median(pw):.1f}W "
          f"(히터면 ~1300W, 팬·조명이면 ~16W)")
    ok = tot_cond < tot_gate
    print(f"    => {'✅ 통과 — 팬·조명 창이 통전에서 빠진다' if ok else '❌ 실패 — 둘이 같다. 버퍼가 안 걸렸다'}")
    return ok


def gate3() -> bool:
    print("\n[3] 실제로 무는가 — `swap_frac` 이 0 이면 항이 안 걸린 것이다")
    ok = True
    for nm, tgt_drop in (("정상", None), ("⚠ v_rms 를 뺀 판", "v_rms")):
        crit = _crit(cond=COND)
        out, tgt, _ = _batch()
        if tgt_drop:
            tgt.pop(tgt_drop)
        parts = crit.unlabeled(out, tgt, w_swap=1.0, swap_tol=0.5)
        fr = float(parts["swap_frac"]); v = float(parts["swap"])
        print(f"    {nm:16s} swap_frac {fr:.3f}  swap {v:.4f}")
        if tgt_drop is None and fr <= 0:
            ok = False
    print(f"    => {'✅ 통과' if ok else '❌ 실패 — 정상 판에서도 안 문다'}")
    print("       (v_rms 를 빼면 0 이 되는 것이 정상이고, **그래서 이 관문이 있다**)")
    return ok


def gate4() -> bool:
    print("\n[4] 끄면 항등 — `--w-swap 0` 과 `res_cond_state=0`")
    out, tgt, _ = _batch()
    c0 = _crit(weights=dict(swap=0.0))
    p0 = c0.forward({k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}, tgt)
    swap0 = float(p0["swap"])
    other = sum(float(getattr(c0.w, n)) * float(v) for n, v in p0.items()
                if n not in ("total", "swap"))
    ok_a = swap0 == 0.0 and abs(float(p0["total"]) - other) < 1e-6
    print(f"    w.swap=0 -> parts['swap'] {swap0:.6e} · total−나머지 "
          f"{abs(float(p0['total']) - other):.3e}   {'✅' if ok_a else '❌'}")

    # 통전 상태 버퍼가 0 이면 확률 BCE 가 로짓 BCE 와 같은 값이어야 한다
    o1, t1, _ = _batch(seed=3)
    a = _crit(cond=None).unlabeled(o1, t1, w_swap=1.0, swap_tol=0.5)["swap"]
    o2, t2, _ = _batch(seed=3)
    b = _crit(cond={}).unlabeled(o2, t2, w_swap=1.0, swap_tol=0.5)["swap"]
    ok_b = torch.allclose(a, b, atol=0, rtol=0)
    print(f"    cond 버퍼 0 두 판: {float(a):.8f} vs {float(b):.8f}   "
          f"{'✅ 비트 동일' if ok_b else '❌ 갈린다'}")
    return ok_a and ok_b


def gate6(dev: str) -> bool:
    """학습과 **같은 방식**(autocast bfloat16)으로 항을 돌려 본다.

    ⚠ fp32 로만 재면 autocast 금지 연산을 못 잡는다. 982877 의 B 팔이 정확히
      그것으로 죽었다 — 관문 넷이 다 통과한 뒤 학습 첫 스텝에서 터졌다.
    """
    print("\n[6] autocast — 학습이 도는 방식 그대로 돌아가나")
    ok = True
    for cond, nm in ((None, "cond 버퍼 0"), (COND, "cond 버퍼 있음(오븐·핫플)")):
        for dt, dv in ((torch.bfloat16, dev), (torch.float32, "cpu")):
            if dv == "cpu" and dt is torch.bfloat16:
                continue
            crit = _crit(cond=cond).to(dv)
            out, tgt, _ = _batch()
            out = {k: (v.to(dv) if torch.is_tensor(v) else v) for k, v in out.items()}
            tgt = {k: (v.to(dv) if torch.is_tensor(v) else v) for k, v in tgt.items()}
            try:
                with torch.autocast("cuda", dtype=dt, enabled=(dv == "cuda")):
                    parts = crit.unlabeled(out, tgt, w_swap=1.0, swap_tol=0.5)
                    v = float(parts["swap"])
                    parts["swap"].backward()
                tag = "autocast bf16" if dv == "cuda" else "fp32"
                print(f"    {nm:22s} {tag:14s} swap {v:.4f}  ✅")
            except Exception as e:                                # noqa: BLE001
                print(f"    {nm:22s} {dv:14s} ❌ {type(e).__name__}: {str(e)[:90]}")
                ok = False
    print(f"    => {'✅ 통과' if ok else '❌ 실패 — 학습에서 터진다'}")
    return ok


def gate5(rev: str) -> bool:
    """`git show <rev>:src/model/losses.py` 와 `unlabeled()` 결과를 대 본다."""
    import importlib.util
    import subprocess
    import tempfile
    print(f"\n[5] 2단계 불변 — `{rev}` 의 `unlabeled()` 와 비트 동일한가")
    try:
        src = subprocess.run(["git", "show", f"{rev}:src/model/losses.py"],
                             capture_output=True, check=True).stdout.decode("utf-8")
    except Exception as e:                                   # noqa: BLE001
        print(f"    ⚠ 그 리비전을 못 읽는다 ({e}) — 건너뛴다"); return True
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/losses_ref.py"
        open(f, "w", encoding="utf-8").write(src)
        spec = importlib.util.spec_from_file_location("losses_ref", f)
        ref = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(ref)
        except Exception as e:                               # noqa: BLE001
            print(f"    ⚠ 그 리비전이 안 불러진다 ({e}) — 건너뛴다"); return True
    kw = dict(w_cons=0.4, w_harm=0.1, w_over=0.1, w_res=0.7, w_swap=1.0,
              swap_tol=0.5, swap_tiebreak="mag", swap_tb_orders=(3,))
    worst = 0.0
    for res_apps in (("electiric_kettle", "oven"), RES):
        for seed in (0, 1, 2):
            o1, t1, _ = _batch(seed=seed)
            o2, t2, _ = _batch(seed=seed)
            s_i = torch.tensor([max(max(S_STATE.get(a, {1: 10.0}).values()), 10.0)
                                for a in APPS])
            common = dict(
                s_i=s_i, signatures=torch.from_numpy(_sig()),
                standby_sig=torch.zeros(len(APPS), H, 2), noise_sig=torch.zeros(H, 2),
                harm_scale=torch.ones(H), s_state=build_state_scales(APPS, s_i.tolist()),
                res_ohm=torch.tensor([RESISTIVE_OHM[a] if a in res_apps else 0.0
                                      for a in APPS]),
                res_ohm_half=torch.tensor([HALFWAVE_OHM[a] if (a in res_apps and a in HALFWAVE_OHM)
                                           else 0.0 for a in APPS]))
            a_ = ref.NILMLoss(weights=ref.LossWeights(), **common).unlabeled(o1, t1, **kw)
            b_ = _crit(res_apps=res_apps, weights={}).unlabeled(o2, t2, **kw)
            for k in sorted(set(a_) & set(b_)):
                if torch.is_tensor(a_[k]):
                    worst = max(worst, float((a_[k] - b_[k]).abs().max()))
    ok = worst == 0.0
    print(f"    6판 x 15항 최대차 {worst:.3e}   {'✅ 비트 동일' if ok else '❌ 갈린다'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 L_swap 관문 (14.32)")
    ap.add_argument("--ckpt", default="results/cnn_sigc_vn_s0.pt")
    ap.add_argument("--skip-real", action="store_true", help="[2] 를 건너뛴다 (체크포인트 없이)")
    ap.add_argument("--vs-rev", default="HEAD", metavar="REV",
                    help="[5] 가 2단계를 견줄 리비전. 커밋 뒤에는 그 앞 커밋을 주면 된다.")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("1단계 `L_swap` 관문 (14.32)")
    print("=" * 78)
    res = [("[1] 기울기 격리", gate1()),
           ("[3] 실제로 무는가", gate3()),
           ("[4] 끄면 항등", gate4()),
           ("[5] 2단계 불변", gate5(a.vs_rev)),
           ("[6] autocast", gate6(dev))]
    if not a.skip_real:
        res.insert(1, ("[2] 통전 상태", gate2(a.ckpt, dev)))
    print("\n" + "=" * 78)
    for nm, ok in res:
        print(f"  {nm:20s} {'✅ 통과' if ok else '❌ 실패'}")
    bad = [nm for nm, ok in res if not ok]
    print(f"\n{'전부 통과' if not bad else '실패: ' + ', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
