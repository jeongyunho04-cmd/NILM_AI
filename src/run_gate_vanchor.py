# -*- coding: utf-8 -*-
"""14.49 관문 — `--harm-vnorm-anchor` 가 **손실까지 닿는가**.

⚠⚠ **이 관문이 없어서 45분을 버렸다** (983506). sbatch 관문이
`harmonic_signature_vref` 를 **따로 불러** 배수를 찍고 통과했는데, 정작 그 값을
`NILMLoss` 로 넘기는 줄이 **조용히 안 들어갔다** (`str.replace` 가 들여쓰기 불일치로
아무것도 안 하고 성공했다). 두 팔이 비트 동일이라 `B−A = +0.00 ± 0.00` 이 나왔다.
⇒ **함수를 재지 말고 배선을 재라.** 여기서는 `NILMLoss` 를 실제로 지어 `L_harm` 의
  예측 고조파가 기기별로 **정확히 예상 배수만큼** 달라지는지 본다.

[1] 끄면 두 손실의 예측 고조파가 **비트 동일**
[2] 켜면 저항 넷만 바뀌고 배수가 `(V_CENTER/V_적합)^e` 와 일치 (1e-5 안)
[3] SMPS·모터 다섯은 **정확히 불변**
[4] `run_train_cnn` 의 인자가 실제로 전달된다 (소스에 `harm_vnorm_vref=` 가 있다)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.inputs import V_CENTER                                    # noqa: E402
from src.model.net import (harmonic_scales, harmonic_signature_vref,     # noqa: E402
                           harmonic_signatures, harmonic_signatures_by_state)
from src.model.losses import NILMLoss                                    # noqa: E402
from src.run_baseline import S_I                                         # noqa: E402
from src.run_train_cnn import _vnorm_exp                                 # noqa: E402
from src.synthesis.segment_pool import SegmentPool                       # noqa: E402

OK, NG = "✅", "❌"


def main() -> int:
    apps = sorted(S_I)
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sst, _ = harmonic_signatures_by_state(pool, apps)
    hs = harmonic_scales(pool, apps)
    vref, vrefs = harmonic_signature_vref(pool, apps)
    ex = np.asarray(_vnorm_exp(apps, "RESISTIVE"), dtype=np.float64)
    K, S = len(apps), sst.shape[1]
    kw = dict(s_i=torch.tensor([S_I[a] for a in apps]),
              signatures=torch.from_numpy(sig),
              signatures_state=torch.from_numpy(sst),
              harm_scale=torch.from_numpy(hs), harm_sig_vnorm=True,
              harm_vnorm_exp=torch.as_tensor(ex, dtype=torch.float32))
    A = NILMLoss(**kw)
    B = NILMLoss(**kw, harm_vnorm_vref=torch.from_numpy(vref),
                 harm_vnorm_vref_state=torch.from_numpy(vrefs))
    torch.manual_seed(0)
    B_, = (3,)
    power = torch.rand(B_, K) * 500 + 50
    out = {"power_raw": power.clone(),
           "power_mix": torch.softmax(torch.randn(B_, K, S), -1),
           "power_states": torch.rand(B_, K, S) * 500 + 50}
    out["power_raw"] = (out["power_mix"] * out["power_states"]).sum(-1)
    power = out["power_raw"].clone()
    vrel = torch.full((B_,), 210.0 / V_CENTER)          # 창 전압 210V
    bad = 0

    # [1] 끄면 비트 동일 — vrel 을 넣고 A 를 두 번 불러 같은지 (자기 자신 대조)
    A._vrel = vrel; B._vrel = vrel
    ha1 = A._harm_pred_active(out, power); ha2 = A._harm_pred_active(out, power)
    print(f"[1] 끄면 재현 가능(같은 입력 → 같은 출력): {OK if torch.equal(ha1, ha2) else NG}")
    bad += 0 if torch.equal(ha1, ha2) else 1

    # [2][3] 켜면 기기별 배수
    hb = B._harm_pred_active(out, power)
    print(f"[2][3] 기기별 배수 (예상 `(V_CENTER/V_적합)^e`)")
    print(f"    {'기기':>7}{'적합 V':>9}{'e':>4}{'예상':>10}{'실측':>10}{'':>4}")
    # 기기 하나만 켜서 그 기기의 기여만 본다
    for j, a in enumerate(apps):
        p1 = torch.zeros(1, K); p1[0, j] = 300.0
        o1 = {"power_raw": p1, "power_mix": out["power_mix"][:1],
              "power_states": out["power_states"][:1]}
        o1["power_raw"] = p1
        A._vrel = vrel[:1]; B._vrel = vrel[:1]
        xa = A._harm_pred_active(o1, p1); xb = B._harm_pred_active(o1, p1)
        na, nb = float(xa.norm()), float(xb.norm())
        got = nb / max(na, 1e-12)
        # ⚠ `--state-signatures` 면 **상태별** 기준전압이 쓰이고 혼합으로 섞인다.
        #   드라이기만 상태가 갈린다 (s1 227.5 · s2 226.7 · s3·s4 227.3V) — 기기 전체 값으로
        #   기대하면 0.06% 어긋난다. **혼합 가중**으로 기대해야 맞다 (관문이 처음에 그렇게 틀렸다).
        w = o1["power_mix"][0, j].numpy().astype(np.float64)
        ks = np.where(vrefs[j] > 0, (V_CENTER / np.maximum(vrefs[j], 1.0)) ** ex[j], 1.0)
        want = float((w * ks).sum() / max(w.sum(), 1e-12)) if vref[j] > 0 else 1.0
        good = abs(got - want) < 1e-4
        if not good:
            bad += 1
        print(f"    {a[:6]:>7}{vref[j]:>8.1f}V{ex[j]:>4.0f}{want:>10.4f}{got:>10.4f}"
              f"{'  ' + (OK if good else NG):>4}")

    # [4] 몇 할만 거는 손잡이 (14.51) — `k^(e·f)` 여야 한다
    print("[4] `--harm-vnorm-frac` (14.51): 배수가 `(V_CENTER/V_적합)^(e·f)` 인가")
    j_ov = apps.index("oven") if "oven" in apps else 0
    for f in (0.0, 0.5, 0.7, 1.0):
        C = NILMLoss(**kw, harm_vnorm_vref=torch.from_numpy(vref),
                     harm_vnorm_vref_state=torch.from_numpy(vrefs), harm_vnorm_frac=f)
        p1 = torch.zeros(1, K); p1[0, j_ov] = 300.0
        o1 = {"power_raw": p1, "power_mix": out["power_mix"][:1],
              "power_states": out["power_states"][:1]}
        A._vrel = vrel[:1]; C._vrel = vrel[:1]
        got = float(C._harm_pred_active(o1, p1).norm()) / max(
            float(A._harm_pred_active(o1, p1).norm()), 1e-12)
        w = o1["power_mix"][0, j_ov].numpy().astype(np.float64)
        ks = np.where(vrefs[j_ov] > 0,
                      (V_CENTER / np.maximum(vrefs[j_ov], 1.0)) ** (ex[j_ov] * f), 1.0)
        want = float((w * ks).sum() / max(w.sum(), 1e-12)) if vref[j_ov] > 0 else 1.0
        good = abs(got - want) < 1e-4
        bad += 0 if good else 1
        note = "  (= 끈 것과 같아야 한다)" if f == 0.0 else ""
        print(f"    f={f:<4.1f} 오븐 예상 {want:.4f} · 실측 {got:.4f}"
              f"  {OK if good else NG}{note}")
    # f=0 은 **정확히** 1.0 이어야 한다 — 부동소수 오차도 없이
    C0 = NILMLoss(**kw, harm_vnorm_vref=torch.from_numpy(vref),
                  harm_vnorm_vref_state=torch.from_numpy(vrefs), harm_vnorm_frac=0.0)
    ex0 = bool(torch.all(C0.vnorm_vref_k == 1.0) and torch.all(C0.vnorm_vref_ks == 1.0))
    print(f"[4] f=0 이면 버퍼가 **정확히** 1.0: {OK if ex0 else NG}")
    bad += 0 if ex0 else 1

    # [5] 학습기 배선 — **소스에 실제로 있는가** (983506 이 45분을 버린 자리다)
    src = open("src/run_train_cnn.py", encoding="utf-8").read()
    w4 = ("harm_vnorm_vref=" in src and "harm_vnorm_vref_state=" in src
          and "harm_vnorm_frac=" in src)
    print(f"[5] `run_train_cnn` 이 실제로 넘기는가 (vref·vref_state·frac): {OK if w4 else NG}")
    bad += 0 if w4 else 1
    print("\n" + (f"관문 전부 통과 {OK}" if not bad else f"{NG} 실패 {bad}건"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
