# -*- coding: utf-8 -*-
"""관문 — **홀수차 위상 지터** `--odd-phase-jitter` (14.268). 여덟 줄.

까닭 (14.266): 분절 풀이 담는 **세션 간** 변이를 자유도별로 쟀다 (파일이 여럿인 기기로).

```
  자유도               파일 안   파일 사이   비
  차수별 위상 φ (rms°)   6.33   **18.73**  3.0   <- 절대 크기 1위
  짝수차 크기 모양°       3.63     5.99    1.7   <- even_jitter 가 덮는 자리
  크기 수준 %           0.49     3.24    6.6
```
그런데 **포트 s1 · 드라이 s1 · 드라이 s2 는 녹화가 하나뿐**이라 이 변이가 풀에 **0** 이다 —
실패하는 조합(포트+드라이)의 두 기기만 그렇다.

```
  [1] sigma=0 비트 동일 + 난수 흐름 불변
  [2] ★ 경로 census 를 **측정**으로 — 원시 위상을 돌려 움직이는 채널을 찾아 표와 대조
  [3] ★★ **라벨 안전** — P·Q·|I_h|·역률·모든 크기비가 **정확히 불변**
  [4] ★ 세밀이 진짜 연산과 같은가 (원시 회전 + 다시 짓기 대 채널공간)
  [5] ⚠ 광역 φ 는 블록 **중앙값**이라 회전과 안 교환된다 — 그 오차를 잰다
  [6] 실현 회전각이 sigma 와 맞나
  [7] `even_jitter` 와 **채널이 안 겹친다** (같이 걸어도 서로 안 지운다)
  [8] h=1 은 안 건드린다 (P·Q 의 뿌리)
```

    python -X utf8 -m src.run_gate_phasejit
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model import inputs as _I  # noqa: E402

_I.EVEN_MEDIAN = 0
from src.model.inputs import (ODD_ORDERS, PHI0, RAW_CHANNELS, WIDE_PHI0,  # noqa: E402
                              build_inputs)
from src.run_train_cnn import (EVEN_FINE_MAG, EVEN_FINE_R2, N_ODD, PHASE_H,  # noqa: E402
                               PHASE_RE, even_jitter, odd_phase_jitter)

W, B = 3600, 6
DECL_F = sorted([i for i in PHASE_RE] + [N_ODD + i for i in PHASE_RE]
                + list(range(PHI0, PHI0 + 8)))
SAFE_F = [23, 24, 25, 26, 27, 28, 39, 40, 43, 44] + list(range(16, 23)) + [0, N_ODD]
ok = True


def raw_batch(seed=0):
    g = np.random.default_rng(seed)
    x = np.zeros((B, RAW_CHANNELS, W), np.float64)
    for h in range(1, 16):
        amp = 2.5 if h == 1 else (0.15 / h if h % 2 else 0.006 * g.uniform(.7, 1.4))
        ph = g.uniform(0, 2 * np.pi, (B, 1))
        wob = 1 + 0.05 * np.sin(np.linspace(0, 9, W))[None] + 0.01 * g.normal(size=(B, W))
        x[:, h - 1] = amp * np.cos(ph) * wob
        x[:, 15 + h - 1] = amp * np.sin(ph) * wob
    x[:, 30] = 1500 + 30 * np.sin(np.linspace(0, 5, W))[None]
    x[:, 31] = 120.0
    x[:, 32] = 227.0 + 0.4 * g.normal(size=(B, W))
    for s in range(6):
        x[:, 33 + s] = 227.0 if s == 0 else 0.4 / (s + 1)
        x[:, 39 + s] = 0.2 / (s + 1)
    return x


def rot_raw(x, d):
    """원시에서 h>=3 홀수차를 차수별 d[:, j] 만큼 돌린다."""
    y = x.copy()
    for j, h in enumerate(PHASE_H):
        re, im = x[:, h - 1], x[:, 15 + h - 1]
        c, s = np.cos(d[:, j])[:, None], np.sin(d[:, j])[:, None]
        y[:, h - 1] = re * c - im * s
        y[:, 15 + h - 1] = re * s + im * c
    return y


def built(x):
    fi, wi = build_inputs(x)
    return (torch.from_numpy(np.ascontiguousarray(fi)),
            torch.from_numpy(np.ascontiguousarray(wi)))


X = raw_batch()
F0, W0 = built(X)

torch.manual_seed(7)
f1, w1 = F0.clone(), W0.clone()
odd_phase_jitter(f1, w1, 0.0)
same = torch.equal(torch.random.get_rng_state(),
                   (torch.manual_seed(7), torch.random.get_rng_state())[1])
d0 = max(float((f1 - F0).abs().max()), float((w1 - W0).abs().max()))
print("[1] sigma=0 비트 동일 %.3e · 난수 흐름 불변 %s   %s"
      % (d0, "예" if same else "아니오", "OK" if d0 == 0 and same else "FAIL"))
ok &= d0 == 0 and same

dk = np.full((B, len(PHASE_H)), np.radians(12.0))
Fs, Ws = built(rot_raw(X, dk))
mvF = sorted(int(c) for c in range(F0.shape[1]) if float((Fs - F0)[:, c].abs().max()) > 1e-6)
mvW = sorted(int(c) for c in range(W0.shape[1]) if float((Ws - W0)[:, c].abs().max()) > 1e-6)
good = mvF == DECL_F and mvW == list(range(WIDE_PHI0, WIDE_PHI0 + 8))
print("[2] 원시 위상을 돌리면 — 세밀 %s" % mvF)
print("    광역 %s · 내 표 세밀 %s   %s" % (mvW, DECL_F, "OK" if good else "**FAIL**"))
ok &= good

safe = max(float((Fs - F0)[:, c].abs().max()) for c in SAFE_F)
def _mag(T, i):
    return (torch.sinh(T[:, i]).square() + torch.sinh(T[:, N_ODD + i]).square()).sqrt()


mag = max(float((_mag(Fs, i) - _mag(F0, i)).abs().max()) for i in range(N_ODD))
print("[3] ★★ 라벨 안전 — P·Q·V·크기비·역률·짝수블록 |max차| **%.2e** · |I_h| 불변 %.2e   %s"
      % (safe, mag, "OK" if safe < 1e-6 and mag < 1e-5 else "**FAIL**"))
ok &= safe < 1e-6 and mag < 1e-5

torch.manual_seed(3)
fa, wa = F0.clone(), W0.clone()
odd_phase_jitter(fa, wa, 12.0)
rec = np.zeros((B, len(PHASE_H)))
for j, i in enumerate(PHASE_RE):
    z0 = torch.sinh(F0[:, i]) + 1j * torch.sinh(F0[:, N_ODD + i])
    z1 = torch.sinh(fa[:, i]) + 1j * torch.sinh(fa[:, N_ODD + i])
    rec[:, j] = torch.angle(z1 / z0).median(-1).values.numpy()
Fb, Wb = built(rot_raw(X, rec))
df = float((fa - Fb).abs().max())
dwphi = float((wa[:, WIDE_PHI0:WIDE_PHI0 + 8] - Wb[:, WIDE_PHI0:WIDE_PHI0 + 8]).abs().max())
dwoth = float((wa[:, :WIDE_PHI0] - Wb[:, :WIDE_PHI0]).abs().max())
print("[4] ★ 세밀이 진짜 연산과 같은가 — |max차| **%.2e**   %s"
      % (df, "OK" if df < 2e-5 else "**FAIL**"))
print("[5] ⚠ 광역 φ (블록 중앙값이라 근사) 차 **%.3f** · 광역 나머지 %.2e   %s"
      % (dwphi, dwoth, "OK" if dwoth < 1e-6 else "**FAIL**"))
ok &= df < 2e-5 and dwoth < 1e-6

r = np.degrees(rec)
print("[6] 실현 회전각 rms **%.2f도** (요청 12.00) · 차수별 %s   %s"
      % (float(np.sqrt((r ** 2).mean())), " ".join("%.1f" % v for v in r.std(0)),
         "OK" if 9 < np.sqrt((r ** 2).mean()) < 16 else "**FAIL**"))
ok &= 9 < np.sqrt((r ** 2).mean()) < 16

EJ = set(EVEN_FINE_MAG + [EVEN_FINE_R2, _I.HALFWAVE_CH, _I.EVEN2_CH])
clash = EJ & set(DECL_F)
print("[7] `even_jitter` 채널과 겹침 **%s**   %s"
      % (sorted(clash) if clash else "없다", "OK" if not clash else "**FAIL**"))
ok &= not clash

h1 = max(float((Fs - F0)[:, c].abs().max()) for c in (0, N_ODD))
print("[8] h=1 (ch0 · ch%d) 불변 **%.2e**   %s" % (N_ODD, h1, "OK" if h1 == 0 else "**FAIL**"))
ok &= h1 == 0

print()
print("관문 %s" % ("전부 통과" if ok else "**실패**"))
sys.exit(0 if ok else 1)
