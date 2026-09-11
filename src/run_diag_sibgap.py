# -*- coding: utf-8 -*-
"""형제(충전기+프로젝터) 대 미니PC 의 **차수별 크기**와, 형제 편차 증강이 미니PC 를 지우는 폭 (13.84.16).

형제 크기에 배율 g 를 주면 총전류의 h 차가 |I_h^sib|·(g−1) 만큼 흔들린다. 그 흔들림이
미니PC 자기 기여 |I_h^mpc| 를 넘으면 **증강이 판별자를 지운다** ([[augmentation-can-erase-the-discriminant]]).
차수마다 "미니PC 를 안 지우는 최대 배율 폭" = |I_h^mpc| / |I_h^sib| 을 낸다 (3σ 규칙은 출력에서 따로).

    python -X utf8 src/run_diag_sibgap.py [N]
"""
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

W, TGT = 3600, 600
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80

pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True)
np.random.seed(0)
sib, mpc, tot, pw = [], [], [], []
tries = 0
while len(sib) < N and tries < 8000:
    tries += 1
    try:
        smp = gen.synthesize_random_window(
            window_size_cycles=W, force_active=["laptop_charger", "beam_projector", "minipc"],
            full_window_placement=False, compute_gt_harmonics=True,
            target_lookahead_cycles=360, sustained_power_limit_w=None)
    except Exception:
        continue
    if smp.metadata.get("base_voltage_v", 999) > 222.0:
        continue
    on = {a: np.asarray(smp.gt_is_on[a]).astype(bool) for a in smp.gt_is_on}
    if not all(on[a][-TGT:].all() for a in ("laptop_charger", "beam_projector", "minipc")):
        continue
    P = np.asarray(smp.power_features)[:, 0]
    if not (70.0 <= P[-TGT:].mean() <= 110.0):
        continue
    gc = lambda a: (lambda g: g[:, :, 0] + 1j * g[:, :, 1])(np.asarray(smp.gt_harmonics_ri[a]))
    s = gc("laptop_charger")[-TGT:] + gc("beam_projector")[-TGT:]
    m = gc("minipc")[-TGT:]
    sib.append(np.abs(s.mean(0))); mpc.append(np.abs(m.mean(0)))
    tot.append(np.abs(np.asarray(smp.harmonics_complex)[-TGT:].mean(0)))
    pw.append((float(np.mean(smp.gt_target_power_w["laptop_charger"][-TGT:])),
               float(np.mean(smp.gt_target_power_w["beam_projector"][-TGT:])),
               float(np.mean(smp.gt_target_power_w["minipc"][-TGT:]))))

sib, mpc, tot, pw = np.array(sib), np.array(mpc), np.array(tot), np.array(pw)
print("창 %d개 (시도 %d)  충전기 %.1fW · 프로젝터 %.1fW · 미니PC %.1fW (중앙)"
      % (len(sib), tries, *np.median(pw, axis=0)))
print("\n차수  형제 |I_h|   미니PC |I_h|   미니PC/형제   지우지 않는 최대 배율폭(±)   1σ 로 쓸 값")
for h in range(1, 16):
    s_, m_ = np.median(sib[:, h - 1]) * 1e3, np.median(mpc[:, h - 1]) * 1e3
    r = m_ / (s_ + 1e-9)
    print("  %2d  %9.2f mA %10.3f mA %11.3f   %19.1f%%   %8.1f%%"
          % (h, s_, m_, r, 100 * r, 100 * r / 3.0))
