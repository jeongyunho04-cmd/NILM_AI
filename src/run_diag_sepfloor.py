# -*- coding: utf-8 -*-
"""**15차 고조파로 충전기와 미니PC 를 가를 수 있는가** — 전력을 맞추고 제대로 (13.84.48).

사용자: *"15차 고조파 데이터면 충분한 거 아니야? 이 정도 데이터를 줬는데도 구분이 안 된다고?"*

13.84.47 의 결론은 **약한 증거**였다 — 원시 발췌 10개, 그것도 충전기 64.5W 대 미니PC 18.4W 로
전력이 안 맞았다. 겹치는 대역은 **19~28W** 다 (충전기 테이퍼 끝 · 미니PC 고부하).
미니PC 는 30W 이상을 물리적으로 못 쓰므로 그 위에서는 맞출 수가 없다 (사용자 지적).

여기서는 npz 전 녹화(사이클 수만 개)를 쓰고 **전력을 맞춰서** 잰다. 그리고 채점은
**녹화 하나 빼기**(LORO)다 — 같은 녹화 안에서 재면 녹화 정체를 외운다 (13.84.7).

가르는 것 둘:
  (A) 격리 판별   충전기 대 미니PC, 각자 단독 녹화. **지문 자체에 정보가 있는가**
  (B) 특징별      와트당 크기 · h1 대비 비 · 상대 위상 · 도통각 — 무엇이 나르는가

이 값이 높으면 "지문은 충분하고 문제는 **섞인 것을 빼는 데** 있다" 가 되고,
낮으면 "지문 자체가 안 갈린다" 가 된다. 둘은 처방이 완전히 다르다.

    python -X utf8 src/run_diag_sepfloor.py [--lo 19 --hi 28]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

NH = 15
NREC = 256


def cond_angle(hc):
    """15차 페이저 (n,15) complex -> 도통각 (n,). h1 위상을 기준으로 복원해서 잰다."""
    n = len(hc)
    c = np.zeros((n, NREC // 2 + 1), complex)
    ph0 = np.angle(hc[:, 0])
    for k in range(1, NH + 1):
        c[:, k] = hc[:, k - 1] * np.exp(-1j * k * ph0)
    w = np.fft.irfft(c * NREC, NREC, axis=1)
    a = np.abs(w)
    pk = a.max(1, keepdims=True)
    return 2 * np.pi * (a > 0.2 * pk).mean(1)


def load_cycles(app, lo, hi):
    """그 기기의 ON·유효·전력대 안 사이클을 녹화별로. [(stem, P (n,), hc (n,15))]"""
    out = []
    for p in sorted(glob.glob("processed_data/npz/%s_*.npz" % app)):
        stem = os.path.basename(p)[:-4]
        d = np.load(p, allow_pickle=True)
        if "harmonics_complex" not in d.files:
            continue
        hc = np.asarray(d["harmonics_complex"])
        P = np.asarray(d["p_denoised_w"]) if "p_denoised_w" in d.files \
            else np.asarray(d["power_features"])[:, 0]
        m = np.ones(len(P), bool)
        if "state_id" in d.files:
            m &= np.asarray(d["state_id"]) != 0
        elif "is_on" in d.files:
            m &= np.asarray(d["is_on"]) == 1
        for k, want in (("is_valid", 1), ("is_unplugged", 0)):
            if k in d.files:
                m &= np.asarray(d[k]) == want
        m &= (P >= lo) & (P <= hi)
        if m.sum() < 50:
            continue
        out.append((stem, P[m], hc[m]))
    return out


def featurize(P, hc, kind):
    a = np.abs(hc)
    if kind == "와트당 크기":
        return a / P[:, None]
    if kind == "h1 대비 비":
        return a / (a[:, :1] + 1e-12)
    if kind == "상대 위상":
        ph = np.angle(hc) - np.arange(1, NH + 1)[None, :] * np.angle(hc[:, :1])
        return np.c_[np.cos(ph), np.sin(ph)]
    if kind == "도통각":
        return cond_angle(hc)[:, None]
    if kind == "전부":
        ph = np.angle(hc) - np.arange(1, NH + 1)[None, :] * np.angle(hc[:, :1])
        return np.c_[a / P[:, None], a / (a[:, :1] + 1e-12),
                     np.cos(ph), np.sin(ph), cond_angle(hc)[:, None]]
    raise ValueError(kind)


def _auc(y, sc):
    """순위 통계로 AUC. sklearn 없이."""
    y = np.asarray(y, bool)
    r = np.argsort(np.argsort(np.asarray(sc, float))) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_logistic(X, y, iters=300, l2=1e-3):
    """torch 로 로지스틱 회귀 (LBFGS). X 는 표준화해서 넣는다."""
    import torch
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # 클래스 불균형 보정 — 사이클 수가 기기마다 다르다
    pw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z, yt, pos_weight=torch.tensor(pw)) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def loro_auc(A, B, kind, lo, hi, nbin=3):
    """녹화 하나 빼기 AUC — 양쪽에서 하나씩 빼고 그 둘로만 채점한다."""
    aucs = []
    FA = [featurize(P, hc, kind) for _, P, hc in A]
    FB = [featurize(P, hc, kind) for _, P, hc in B]
    for ia in range(len(A)):
        for ib in range(len(B)):
            Xtr = np.vstack([FA[j] for j in range(len(A)) if j != ia]
                            + [FB[j] for j in range(len(B)) if j != ib])
            ytr = np.concatenate(
                [np.zeros(len(FA[j])) for j in range(len(A)) if j != ia]
                + [np.ones(len(FB[j])) for j in range(len(B)) if j != ib])
            Xte = np.vstack([FA[ia], FB[ib]])
            yte = np.concatenate([np.zeros(len(FA[ia])), np.ones(len(FB[ib]))])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            w, b = _fit_logistic((Xtr - mu) / sd, ytr)
            aucs.append(_auc(yte, ((Xte - mu) / sd) @ w + b))
    aucs = [x for x in aucs if np.isfinite(x)]
    return (float(np.mean(aucs)), float(np.std(aucs)), len(aucs)) if aucs else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=19.0)
    ap.add_argument("--hi", type=float, default=28.0)
    a = ap.parse_args()

    print("겹치는 전력대 %.0f~%.0fW 에서 **격리 녹화끼리** 가를 수 있는가" % (a.lo, a.hi))
    A = load_cycles("laptop_charger", a.lo, a.hi)
    B = load_cycles("minipc", a.lo, a.hi)
    for nm, G in (("laptop_charger", A), ("minipc", B)):
        print("   %-16s 녹화 %d개 · 사이클 %d" % (nm, len(G), sum(len(x[1]) for x in G)))
        for s, P, hc in G:
            print("      %-22s %6d 사이클 · P %.1f~%.1fW (중앙 %.1f)"
                  % (s, len(P), P.min(), P.max(), np.median(P)))
    if len(A) < 2 or len(B) < 2:
        print("   녹화가 2개 미만인 쪽이 있어 LORO 를 못 한다"); return 1

    print("\n   녹화 하나 빼기(LORO) AUC — 0.5 는 못 가름, 1.0 은 완벽")
    print("   %-14s %9s %9s %7s" % ("특징", "AUC", "표준편차", "짝"))
    for kind in ("와트당 크기", "h1 대비 비", "상대 위상", "도통각", "전부"):
        m, sd, n = loro_auc(A, B, kind, a.lo, a.hi)
        print("   %-14s %9.3f %9.3f %7d" % (kind, m, sd, n))

    print("\n   ⚠ 이것은 **격리 녹화**끼리다 — 한 기기만 켜진 깨끗한 관측이다.")
    print("     높으면: 지문에 정보는 충분하고 문제는 **섞인 것을 빼는 데** 있다.")
    print("     낮으면: 지문 자체가 안 갈린다 — 표현을 바꿔야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
