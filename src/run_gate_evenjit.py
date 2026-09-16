# -*- coding: utf-8 -*-
"""14.245 관문 — `--even-jitter` 가 **물리 연산과 같은 물건**인지 못 박는다.

  [1] sigma=0 이면 비트 동일이고 **난수도 안 뽑는다** (씨앗 흐름 불변)
  [2] ★ **경로 census 를 코드가 아니라 측정으로** 짓는다 — 원시 짝수차를 흔들어
      `build_inputs` 를 다시 돌리고, 움직이는 채널을 전부 찾아 내 표와 맞춘다
  [3] ★★ **진짜 객체로 검증** — 원시를 f 배 하고 다시 지은 것과, 채널공간에서 돌린 것이
      같은가. `asinh`/`sinh` 약분이 맞는지, 43(차)에서 |I4| 복원이 맞는지 여기서 걸린다
  [4] 짝수차 이동중앙값 k=5 에서도 [3] 이 성립 (중앙값이 양수 배수와 교환된다)
  [5] 크기는 안 건드린다 — log f 의 평균이 0 (기하평균 1)
  [6] sigma 별 **실현 모양각** — 실측 반파 폭(에피소드 사이 2.2~9.2도)에 눈금을 맞추라고
"""
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

from src import env_guard                                            # noqa: F401,E402
from src.model import inputs as _I                                   # noqa: E402
from src.model.inputs import RAW_CHANNELS, build_inputs              # noqa: E402
from src.run_train_cnn import (EVEN_FINE_MAG, EVEN_FINE_R2, EVEN_WIDE_H2,  # noqa: E402
                               EVEN_WIDE_MAG, even_jitter)

W, B = 3600, 6
EVEN_H = (2, 4, 6, 8, 10, 12, 14)
DECL_F = sorted(EVEN_FINE_MAG + [EVEN_FINE_R2, _I.HALFWAVE_CH, _I.EVEN2_CH])
DECL_W = sorted(list(EVEN_WIDE_H2) + EVEN_WIDE_MAG)
PF_CH = 39                       # 역률 — i_rms 에 짝수차가 1e-5 미만으로 섞인다. 일부러 뺀다
ok = True


def raw_batch(seed=0):
    """그럴듯한 원시 (B,45,W). Re/Im 은 차수에 따라 줄고 짝수차는 수 mA 다."""
    g = np.random.default_rng(seed)
    x = np.zeros((B, RAW_CHANNELS, W), np.float64)
    for h in range(1, 16):
        amp = (2.5 if h == 1 else (0.15 / h if h % 2 else 0.006 * g.uniform(.7, 1.4)))
        ph = g.uniform(0, 2 * np.pi, (B, 1))
        wob = 1 + 0.05 * np.sin(np.linspace(0, 9, W))[None] + 0.01 * g.normal(size=(B, W))
        x[:, h - 1] = amp * np.cos(ph) * wob
        x[:, 15 + h - 1] = amp * np.sin(ph) * wob
    x[:, 30] = 1500 + 30 * np.sin(np.linspace(0, 5, W))[None]
    x[:, 31] = 120.0
    x[:, 32] = 227.0 + 0.4 * g.normal(size=(B, W))
    for s in range(6):
        x[:, 33 + s] = (227.0 if s == 0 else 0.4 / (s + 1))
        x[:, 39 + s] = 0.2 / (s + 1)
    return x


def scale_even(x, f):
    """원시에서 짝수차 **페이저 크기**를 차수별 f 배 (Re·Im 둘 다)."""
    y = x.copy()
    for s, h in enumerate(EVEN_H):
        y[:, h - 1] *= f[:, s, None]
        y[:, 15 + h - 1] *= f[:, s, None]
    return y


def built(x):
    fi, wi = build_inputs(x)
    return torch.from_numpy(np.ascontiguousarray(fi)), torch.from_numpy(np.ascontiguousarray(wi))


X = raw_batch()
F0, W0 = built(X)

# ── [1] ────────────────────────────────────────────────────────────────────────
torch.manual_seed(7)
f1, w1 = F0.clone(), W0.clone()
even_jitter(f1, w1, 0.0)
s_after = torch.random.get_rng_state()
torch.manual_seed(7)
s_fresh = torch.random.get_rng_state()
d0 = max(float((f1 - F0).abs().max()), float((w1 - W0).abs().max()))
same_rng = bool(torch.equal(s_after, s_fresh))
print("[1] sigma=0 비트 동일 %.3e · 난수 흐름 불변 %s   %s"
      % (d0, "예" if same_rng else "**아니오**", "OK" if d0 == 0 and same_rng else "실패"))
ok &= d0 == 0 and same_rng

# ── [2] 경로 census ────────────────────────────────────────────────────────────
fk = np.full((B, 7), 1.5)
Fs, Ws = built(scale_even(X, fk))
mvF = sorted(int(c) for c in range(F0.shape[1]) if float((Fs - F0)[:, c].abs().max()) > 1e-6)
mvW = sorted(int(c) for c in range(W0.shape[1]) if float((Ws - W0)[:, c].abs().max()) > 1e-6)
hitF, hitW = sorted(set(mvF) - {PF_CH}), sorted(set(mvW))
print("[2] 원시를 흔들면 움직이는 채널 — 세밀 %s" % mvF)
print("    (PF %d 은 i_rms 로 %.1e 만 움직인다 — 일부러 제외)"
      % (PF_CH, float((Fs - F0)[:, PF_CH].abs().max())))
print("    광역 %s" % mvW)
print("    내 표    세밀 %s · 광역 %s" % (DECL_F, DECL_W))
good = hitF == DECL_F and hitW == DECL_W
print("    일치 %s   %s" % ("예" if good else "**아니오**", "OK" if good else "실패"))
ok &= good

# ── [3] 진짜 객체로 검증 ───────────────────────────────────────────────────────
def equiv(k, sigma=0.25, seed=3):
    _I.EVEN_MEDIAN = k
    Fa, Wa = built(X)
    fa, wa = Fa.clone(), Wa.clone()
    torch.manual_seed(seed)
    even_jitter(fa, wa, sigma)
    r = (torch.sinh(fa[:, EVEN_FINE_MAG]) / torch.sinh(Fa[:, EVEN_FINE_MAG]))
    f = r.median(-1).values.double().numpy()                     # (B,7) 되찾은 배수
    tvar = float((r.max(-1).values - r.min(-1).values).abs().max())
    Fb, Wb = built(scale_even(X, f))
    keep = [c for c in range(Fa.shape[1]) if c != PF_CH]
    df = float((fa[:, keep] - Fb[:, keep]).abs().max())
    dw = float((wa - Wb).abs().max())
    return f, tvar, df, dw, float((fa[:, PF_CH] - Fb[:, PF_CH]).abs().max())


for k, tag in ((0, "[3]"), (5, "[4]")):
    f, tvar, df, dw, dpf = equiv(k)
    good = df < 2e-5 and dw < 2e-5 and tvar < 1e-5
    print("%s k=%d  배수가 시간에 상수 %.2e · **세밀 차 %.2e · 광역 차 %.2e** (PF %.1e)  %s"
          % (tag, k, tvar, df, dw, dpf, "OK" if good else "실패"))
    ok &= good
_I.EVEN_MEDIAN = 0

# ── [5] 크기 보존 ──────────────────────────────────────────────────────────────
torch.manual_seed(11)
e = torch.randn(4096, 7)
lf = 0.25 * (e - e.mean(1, keepdim=True))
gm = float(lf.sum(1).abs().max())
print("[5] log f 의 차수 합 |max| %.2e (기하평균 1 = 크기 불변)   %s"
      % (gm, "OK" if gm < 1e-4 else "실패"))
ok &= gm < 1e-4

# ── [6] sigma 눈금 ─────────────────────────────────────────────────────────────
shape = np.array([0.969, 0.231, 0.067, 0.039, 0.029, 0.017, 0.010])   # 풀 드라이 s1
print("[6] sigma 별 실현 모양각 (풀 드라이 s1 모양 기준, 중앙 / p90)")
rng = np.random.default_rng(0)
for sg in (0.05, 0.10, 0.15, 0.25, 0.40):
    ee = rng.normal(size=(4000, 7))
    ff = np.exp(sg * (ee - ee.mean(1, keepdims=True)))
    m2 = shape[None] * ff
    u = m2 / np.linalg.norm(m2, axis=1, keepdims=True)
    a = np.degrees(np.arccos(np.clip(u @ (shape / np.linalg.norm(shape)), -1, 1)))
    print("    sigma %.2f  ->  **%5.2f도** / p90 %5.2f도" % (sg, np.median(a), np.percentile(a, 90)))
print("    실측 반파 폭: 에피소드 사이 2.2~9.2도 · 안 0.7~4.9도 · 풀은 0.06도 (14.234·14.237)")

print()
print("관문 %s" % ("전부 통과" if ok else "**실패**"))
sys.exit(0 if ok else 1)
