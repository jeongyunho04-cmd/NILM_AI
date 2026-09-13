# -*- coding: utf-8 -*-
"""채점 시점 자기보정 (EM) — 상태열과 **파일별 형제 지문 보정**을 같이 추정한다 (13.84.46).

13.84.45 가 보인 것: 형제(충전기·프로젝터)는 **항상 맞다** — 미니PC 하나의 유무만 틀린다
(미니PC 오류 201단계 전부에서 형제 정답, 맞바꿈 0%). 그러면 형제를 빼고 남은 것만 보면 된다.

측정된 천장이 있다:
```
13.84.34   형제 빼기 자체              0.648 -> 0.718  (참 전력을 준 천장)
13.84.35⑥ 기록별 지문 보정을 주면      0.747 -> 0.937  (신탁)
13.84.34   그 보정을 전이에서 뽑으면    0.718 -> 0.665  (역효과 — 전이로는 못 뽑는다)
```
안 해본 것: **상태열과 보정을 번갈아** 추정하기.
```
사슬로 푼다 -> 미니PC 가 OFF 인 구간을 고른다 -> 거기서 이 파일의 형제 지문을 맞춘다
            -> 보정된 형제를 빼고 잔차를 미니PC 지문에 사영해 새 방출을 만든다 -> 다시 푼다
```

⚠ **아는 실패 방식** (13.84.31 ⑤): test_2 는 미니PC 가 92% 켜져 있어 OFF 구간이 거의 없다.
  보정이 미니PC 기여를 빨아들여 신호가 23.0 -> 2.8W 로 무너졌다. 그래서 관문을 둔다 —
  OFF 구간이 모자라면 **그 파일은 보정을 안 건다**. 그리고 미니PC 자기 지문은 **절대 보정 대상에
  넣지 않는다** (기울기 x2 로 와트를 반토막 내는 자명한 축퇴).

    python -X utf8 src/run_diag_selfcal.py [results/seq_h38_base.pt]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.model.lossbuild import build_loss
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
SIB = ["laptop_charger", "beam_projector"]     # 보정 대상. 미니PC 는 **넣지 않는다**
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
MIN_OFF = 30            # 보정에 쓸 최소 OFF 단계
MIN_OFF_FRAC = 0.10     # 그리고 파일의 최소 몫
GCAP = (0.6, 1.6)       # 보정 배율 상한 — 13.84.35 ⑦ 이 잰 9~12%@h1, 고차 50~113% 를 넉넉히 덮는다
ITERS = 4


def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "results/seq_h38_base.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = list(ck["appliances"])
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    crit = build_loss(apps, "cpu", verbose=False)
    sig = crit.sig.numpy()                                  # (K, 15, 2) 와트당 페이저
    sigc = sig[..., 0] + 1j * sig[..., 1]
    nz = crit.noise_sig.numpy()
    nzc = nz[:, 0] + 1j * nz[:, 1]
    sbc = crit.standby_sig.numpy()
    sbc = sbc[..., 0] + 1j * sbc[..., 1]                    # (S, 15) 또는 (K,15)
    oi = [o - 1 for o in ORD]
    km = apps.index("minipc")
    ksib = [apps.index(x) for x in SIB]
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)

    print("채점 시점 자기보정 — 보정 대상 %s (미니PC 제외) · 반복 %d" % (SIB, ITERS))
    print("   관문: OFF 단계 >= %d 이고 >= %d%% · 배율 %.1f~%.1f"
          % (MIN_OFF, int(100 * MIN_OFF_FRAC), *GCAP))
    print()
    print("   %-8s %7s %8s | %s" % ("파일", "OFF단계", "관문", "반복별 미니PC 시간 정확도"))

    base_acc, fin_acc = [], []
    with torch.no_grad():
        for stem in FILES:
            d = cache.get(stem)
            if d is None or not d["present"][km]:
                continue
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                PW.append(o["power"].float())
            z = torch.cat(Z)[None]
            gl = torch.cat(GL)[None]
            dfe = torch.from_numpy(d["dfeat"]).float()[None].to(dev)
            em0, on, off, ini = heads(z, dfe, gl)
            pw = torch.cat(PW).cpu().numpy()                # (T,K) 예측 와트
            y = d["y"].astype(bool)

            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            H = np.asarray(r["harmonics_complex"])
            ti = np.clip((d["t"] * FS).astype(int), 0, len(H) - 1)
            obs = H[ti][:, oi]                              # (T, 8) 관측 페이저

            path = viterbi(em0, on, off, ini)[0].cpu().numpy()
            accs = [float((path[:, km] == y[:, km]).mean())]
            base_acc.append(accs[0])

            # 보정 관문 — OFF 구간이 모자라면 아예 안 건다 (13.84.31 ⑤ 의 축퇴)
            noff = int((~path[:, km]).sum())
            okgate = noff >= MIN_OFF and noff >= MIN_OFF_FRAC * len(path)
            g = np.ones((len(ksib), len(ORD)), complex)

            if okgate:
                for _ in range(ITERS):
                    m = ~path[:, km]
                    if m.sum() < MIN_OFF:
                        break
                    # ── M 단계: 형제 배율을 차수마다 최소제곱으로 ─────────────
                    # 관측 = Σ_k g_k · (와트_k · sig_k) + (나머지 기기) + 잡음
                    rest = np.zeros_like(obs)
                    for k in range(len(apps)):
                        if k in ksib or k == km:
                            continue
                        rest += pw[:, k, None] * sigc[k][oi][None, :]
                    rest = rest + nzc[oi][None, :]
                    tgt = obs - rest                          # (T,8)
                    A = np.stack([pw[:, k, None] * sigc[k][oi][None, :] for k in ksib], 0)
                    for j in range(len(ORD)):
                        Aj = A[:, m, j].T                     # (n, n_sib)
                        bj = tgt[m, j]
                        if Aj.shape[0] < 5:
                            continue
                        sol, *_ = np.linalg.lstsq(Aj, bj, rcond=None)
                        mag = np.clip(np.abs(sol), *GCAP)
                        ph = np.angle(sol)
                        g[:, j] = mag * np.exp(1j * ph)
                    # ── E 단계: 보정된 형제를 빼고 잔차를 미니PC 지문에 사영 ──
                    fit = rest.copy()
                    for a_, k in enumerate(ksib):
                        fit += g[a_][None, :] * pw[:, k, None] * sigc[k][oi][None, :]
                    resid = obs - fit                         # (T,8)
                    sm = sigc[km][oi]
                    w_hat = (resid @ np.conj(sm)).real / (np.abs(sm) ** 2).sum()   # 추정 와트
                    # 방출로 쓰려면 눈금이 필요하다. 참값 10~13W 의 절반을 문턱으로 둔다.
                    emn = torch.from_numpy(
                        ((w_hat - 6.0) / 6.0).astype(np.float32)).to(dev)
                    em = em0.clone()
                    em[0, :, km] = emn
                    path = viterbi(em, on, off, ini)[0].cpu().numpy()
                    accs.append(float((path[:, km] == y[:, km]).mean()))
            fin_acc.append(accs[-1])
            print("   %-8s %7d %8s | %s"
                  % (stem, noff, "통과" if okgate else "**막음**",
                     "  ".join("%.3f" % v for v in accs)))
            if okgate:
                print("   %-8s %7s %8s |   보정 배율 |g| h1 %s · h11 %s"
                      % ("", "", "",
                         " ".join("%s %.2f" % (SIB[i][:4], abs(g[i, 0])) for i in range(len(ksib))),
                         " ".join("%s %.2f" % (SIB[i][:4], abs(g[i, ORD.index(11)])) for i in range(len(ksib)))))

    print("\n   미니PC 평균  처음 %.4f -> 마지막 %.4f"
          % (float(np.mean(base_acc)), float(np.mean(fin_acc))))
    print("   ⚠ 신탁 없음 — 라벨은 채점에만 썼다. 보정은 **모델이 OFF 라고 푼 구간**에서만 맞췄다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
