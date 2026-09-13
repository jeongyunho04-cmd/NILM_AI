# -*- coding: utf-8 -*-
"""사슬이 사는 값이 **틈 메우기 하나**인가 — 재학습 없이 확인한다.

사슬 판은 핫플 재현 0.996 / 정밀 0.994 인데 창별 게이트는 0.409 / 0.995 다.
차이가 전부 **서모스탯이 전력을 끊은 틈**이라면, 예측 켜짐 구간 사이의 짧은 틈을
메우는 것만으로 같은 값이 나와야 한다. 나오면 사슬의 유일한 이득이 후처리로
대체되고, 상대 잔차 4.6% 를 지킨 채 롤백할 수 있다.

손잡이를 고르지 않는다 — **너비를 훑어 민감도를 찍는다** (13.86 의 `--duty` 와 같은 수).

**답 (2026-09-13, 13.96 [4]): 안 된다.** 핫플 재현이 0.409 -> **0.447 에서 포화**한다.
놓친 것이 "켜짐 사이의 틈" 이 아니라 **코일이 안 도는 다수 구간 전체**이기 때문이다 —
test_3 의 참켜짐 구간 107스텝 중 창별 게이트는 28스텝만 켠다. 메울 양쪽 끝이 없다.
그래서 사슬이 버는 값은 후처리로 대체되지 않는다. **이 파일은 그 반증을 남겨 둔다.**

    python -X utf8 src/run_diag_close.py
"""
import sys; sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch
from src import env_guard
from src.run_baseline_nilmtk import score_arm
from src.run_diag_rollback import predict, _avg, _rp
from src.run_score_seq import DUTY_APPS
from src.run_train_seq import real_windows

def close_gaps(on, w):
    """켜짐 구간 사이 **w 스텝 이하 틈**을 메운다 (1차원 닫힘). on (T,) bool."""
    if w <= 0: return on
    out = on.copy(); T = len(on)
    i = 0
    while i < T:
        if out[i]: i += 1; continue
        j = i
        while j < T and not out[j]: j += 1
        if i > 0 and j < T and (j - i) <= w:   # 양쪽이 켜짐인 짧은 틈만
            out[i:j] = True
        i = j
    return out

dev = "cuda" if torch.cuda.is_available() else "cpu"
apps = list(torch.load("results/cnn_v37.pt", map_location="cpu", weights_only=False)["appliances"])
cache = real_windows(apps, 2.0, dev)
duty_i = [apps.index(x) for x in DUTY_APPS if x in apps]

print("틈 메우기 너비 훑기 — cnn_v37 창별 게이트 (격자 2초)")
print("=" * 82)
print("  %-10s%10s%10s%14s%14s%10s" % ("너비", "듀티제외7", "듀티2종", "핫플 재/정", "오븐 재/정", "상대잔차"))
_, pr0 = predict("results/cnn_v37.pt", cache, dev)
for w in (0, 1, 2, 3, 5, 8, 12, 20, 40):
    acc, cf, rel = {}, {}, []
    for stem, d in cache.items():
        on, pw = pr0[stem]
        on2 = on.copy()
        for k in duty_i:                      # **듀티 기기에만** 건다
            on2[:, k] = close_gaps(on[:, k], w)
        ac, _, _, rc, cn = score_arm(on2, pw, d, apps)
        for k, v in ac.items(): acc.setdefault(k, []).append(v)
        for k, v in cn.items(): cf[k] = tuple(x + y for x, y in zip(cf.get(k, (0,)*4), v))
        if rc:
            r = pw.sum(1) + float(d["p_base"]) - d["p_obs"]
            rel.append(float(np.median(np.abs(r) / np.maximum(d["p_obs"], 10.0))))
    print("  %-10s%10.4f%10.4f%14s%14s%9.1f%%" % (
        "%d (%ds)" % (w, w * 2), _avg(acc, False), _avg(acc, True),
        _rp(cf, "hotplate"), _rp(cf, "oven"), 100 * np.mean(rel)))
print()
print("사슬(seq_fix_ctl) 대조:  듀티제외7 0.9141 · 듀티2종 0.9865 · 핫플 0.996/0.994 · 잔차 8.8%")
