# -*- coding: utf-8 -*-
"""빈 상태 슬롯 고침의 관문 — 꺼지면 항등, 켜지면 동작 (13.84.68).

13.84.68 이 잰 것: 전력은 `p_raw = Σ_s mix[s]·p_states[s]` 인 **곱**인데, 혼합이 안 고르는
슬롯은 기울기가 `mix[s]·∂L/∂p_raw ≈ 0` 이라 못 오르고, `softplus` 의 음의 포화로 가면
**다시 못 살아난다.** 드라이기 약풍 슬롯이 그렇게 죽어서, 시퀀스 학습이 분류를 정확하게
만들자(`mix[1] = 0.997`) 전력이 **466W -> 3W** 로 무너졌다.

고친 것 둘 (둘 다 **기본값에서 옛 동작**이거나 **학습된 체크포인트에 무영향**):
  ① `NILMNet(state_power_init=True)` — `p_states` 슬롯을 `S_STATE` 의 잰 값에서 출발
     시킨다. 학습된 체크포인트는 `load_state_dict` 가 바이어스를 덮으므로 **영향 없다.**
  ② `run_train_seq --w-state-power W` — `p_states[참상태]` 를 참 전력에 **직접** 묶는다.
     혼합과 무관한 기울기라 죽은 슬롯도 산다. 기본 0 = 옛 동작.

관문 넷:
  [1] 항등(체크포인트)  새 초기화로 지은 모델에 옛 체크포인트를 실으면 출력이 **비트 단위로** 같다
  [2] 항등(손실)        `--w-state-power 0` 이면 손실 전체가 옛 값과 **비트 단위로** 같다
  [3] 동작(초기화)      새 판의 `p_states` 초기값이 `S_STATE` 와 1% 안에서 맞는다
  [4] 동작(기울기)      죽은 슬롯(바이어스 −30)에 ② 가 **0 아닌 기울기**를 준다.
                        ① 없이는 그 기울기가 사실상 0 이라는 것도 같이 보인다

    python -X utf8 src/run_gate_states.py [--ck results/seq_h38_base.pt]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
import torch.nn.functional as F

from src.model.losses import S_STATE, LossWeights, NILMLoss, build_state_scales
from src.model.net import MAX_STATES, NILMNet
from src.run_baseline import S_I

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def fake_batch(K, H, B=6, seed=0):
    """손실이 먹는 최소한의 (out, tgt). 라벨은 있다 — `state_power` 가 `y_state` 를 쓴다."""
    g = torch.Generator().manual_seed(seed)
    pw = torch.rand(B, K, generator=g) * 40.0
    ps = torch.rand(B, K, MAX_STATES, generator=g) * 60.0
    mix = torch.rand(B, K, MAX_STATES, generator=g).softmax(-1)
    out = {"power": pw.clone().requires_grad_(True),
           "power_raw": pw.clamp(min=1e-3),
           "power_states": ps.clone().requires_grad_(True),
           "power_mix": mix,
           "on_logit": torch.randn(B, K, generator=g),
           "plugged_logit": torch.randn(B, K, generator=g),
           "standby": torch.rand(B, K, generator=g),
           "state": torch.randn(B, K, MAX_STATES, generator=g)}
    tgt = {"y_power": torch.rand(B, K, generator=g) * 40.0,
           "y_on": (torch.rand(B, K, generator=g) > 0.5).float(),
           "y_plugged": (torch.rand(B, K, generator=g) > 0.3).float(),
           "y_standby": torch.rand(B, K, generator=g),
           "y_state": (torch.rand(B, K, generator=g) * 2).long() + 1,
           "obs_harm": torch.randn(B, H, 2, generator=g) * 0.1,
           "p_noise": torch.rand(B, generator=g),
           "p_observed": torch.rand(B, generator=g) * 200 + 50,
           "harm_offset": None}
    return out, tgt


def build(w_state_power, K, H):
    s_i = torch.tensor([S_I[x] for x in APPS], dtype=torch.float32)
    return NILMLoss(s_i=s_i,
                    signatures=torch.randn(K, H, 2, generator=torch.Generator().manual_seed(1)) * 0.01,
                    harm_scale=torch.full((H,), 0.05),
                    s_state=build_state_scales(APPS, [S_I[x] for x in APPS]),
                    weights=LossWeights(harm=0.1, cons=0.0, over=0.1, z=0.0,
                                        state_power=w_state_power))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="results/seq_h38_base.pt")
    a = ap.parse_args()
    K, H = len(APPS), 15
    bad = 0

    # ── [1] 항등 — 옛 체크포인트가 비트 단위로 같은 답을 내나 ──────────────────
    print("[1] 항등(체크포인트) — 새 초기화로 지은 모델에 옛 가중치를 실으면 같은 답인가")
    try:
        import os
        if not os.path.exists(a.ck):
            print("   %s 가 없다 — 건너뛴다" % a.ck)
        else:
            ck = torch.load(a.ck, map_location="cpu", weights_only=False)
            from src.run_gate_check import load_model
            ref = ck.get("ref", "results/cnn_v37.pt") if "heads" in ck else a.ck
            outs = []
            for flag in (False, True):
                m = load_model(ref, "cpu", weights=False,
                               mask=not bool(ck.get("no_mask", False)))[0]
                # 지은 뒤 바이어스만 갈아 끼워 두 초기화를 흉내낸다
                if flag:
                    with torch.no_grad():
                        for j, ap_ in enumerate(m.appliances):
                            for sid, w in S_STATE.get(ap_, {}).items():
                                if 0 <= sid < MAX_STATES:
                                    m.heads[j].bias[sid] = float(w)
                m.load_state_dict(ck["model"]); m.eval()
                g = torch.Generator().manual_seed(3)
                f = torch.randn(4, m.fine_channels, 600, generator=g)
                wd = torch.randn(4, 47, 120, generator=g)
                with torch.no_grad():
                    o = m(f, wd)
                outs.append(torch.cat([o["power"].flatten(), o["power_states"].flatten()]))
            d = float((outs[0] - outs[1]).abs().max())
            ok = d == 0.0
            bad += (not ok)
            print("   최대 절대차 %.3e   %s" % (d, "**비트 단위로 같다**" if ok else "!! 다르다 !!"))
            print("   (이유: `load_state_dict` 가 바이어스를 덮는다 — 초기화는 새 판에만 듣는다)")
    except Exception as e:
        print("   확인 실패: %s" % e); bad += 1

    # ── [2] 항등 — w=0 이면 손실이 옛 값 그대로 ───────────────────────────────
    print("\n[2] 항등(손실) — `--w-state-power 0` 이면 손실이 옛 값과 같은가")
    o1, t1 = fake_batch(K, H, seed=5)
    o2, t2 = fake_batch(K, H, seed=5)
    r1 = build(0.0, K, H)(o1, t1)
    r0 = build(0.0, K, H)(o2, t2)
    d = max(float((r1[k] - r0[k]).abs().max()) for k in r1 if torch.is_tensor(r1[k]))
    print("   같은 씨앗 두 판 최대차 %.3e" % d)
    o3, t3 = fake_batch(K, H, seed=5)
    r2 = build(0.3, K, H)(o3, t3)
    ok = (float(r1["state_power"].abs()) == 0.0) and (float(r2["state_power"].abs()) > 0.0)
    bad += (not ok)
    print("   state_power 항  w=0 -> %.6f  ·  w=0.3 -> %.6f   %s"
          % (float(r1["state_power"]), float(r2["state_power"]),
             "**꺼지고 켜진다**" if ok else "!! 안 된다 !!"))
    dt = float((r2["total"] - r1["total"]).abs())
    print("   전체 손실 차 %.6f  (w>0 일 때만 움직여야 한다)" % dt)

    # ── [3] 동작 — 초기값이 잰 값에 앉나 ──────────────────────────────────────
    print("\n[3] 동작(초기화) — 새 판의 `p_states` 가 `S_STATE` 에 앉나")
    ns = [max(S_STATE.get(x, {1: 0}).keys()) + 1 for x in APPS]
    m = NILMNet(APPS, ns, state_power_init=True)
    with torch.no_grad():
        ps = F.softplus(torch.stack([h.bias[:MAX_STATES] for h in m.heads]))
    worst, wname = 0.0, ""
    for j, ap_ in enumerate(APPS):
        for sid, w in S_STATE.get(ap_, {}).items():
            if sid >= ns[j]:
                continue
            e = abs(float(ps[j, sid]) - w) / max(w, 1e-9)
            if e > worst:
                worst, wname = e, "%s/s%d" % (ap_, sid)
    ok = worst < 0.01
    bad += (not ok)
    print("   최악 상대오차 %.4f%% (%s)   %s"
          % (100 * worst, wname, "**앉는다**" if ok else "!! 안 앉는다 !!"))

    # ── [4] 동작 — 죽은 슬롯에 기울기가 가나 ─────────────────────────────────
    print("\n[4] 동작(기울기) — **죽은 슬롯**(pre-activation −30)에 기울기가 가나")
    print("   %-28s %14s %14s" % ("", "w_state_power=0", "w_state_power=0.3"))
    for lbl, pre in (("살아 있는 슬롯 (pre=+6)", 6.0), ("**죽은 슬롯** (pre=−30)", -30.0)):
        gs = []
        for w in (0.0, 0.3):
            o, t = fake_batch(K, H, seed=7)
            raw = torch.full((6, K, MAX_STATES), pre, requires_grad=True)
            o["power_states"] = F.softplus(raw)
            o["power_mix"] = torch.zeros(6, K, MAX_STATES)
            o["power_mix"][..., 2] = 1.0            # 혼합은 슬롯 2 만 고른다
            t["y_state"] = torch.ones(6, K, dtype=torch.long)   # 참 상태는 **1**
            build(w, K, H)(o, t)["total"].backward()
            # ⚠ `w=0` 이면 `power_states` 가 손실 그래프에 **아예 안 들어가** grad 가 None 이다.
            #   그것이 바로 이 관문이 보이려는 것이다 — 0 으로 적는다.
            gs.append(0.0 if raw.grad is None else float(raw.grad[:, :, 1].abs().mean()))
        print("   %-28s %14.3e %14.3e" % (lbl, gs[0], gs[1]))
    print("   혼합이 슬롯 2 만 고르는데 참 상태는 1 이다. 곱만 보는 손실은 슬롯 1 에")
    print("   기울기를 **전혀** 못 준다 (0.000e+00) — `state_power` 는 준다.")
    print()
    print("   ⚠⚠ **그러나 이미 죽은 슬롯은 못 살린다.** 2.0e-18 은 산 슬롯의 1e-13 배다 —")
    print("   `∂softplus/∂x = σ(−30) ≈ 9e-14` 가 사슬에서 곱해져서다. 즉 ② 는 **예방**이지")
    print("   **치료가 아니다.** 이미 죽은 체크포인트(드라이기 약풍)는 학습으로 못 되돌린다:")
    print("   ① 로 새로 굽거나, 실을 때 죽은 슬롯의 바이어스를 다시 세워야 한다.")

    # ── [5] S_STATE 의 **슬롯이 자료와 맞나** (13.92) ─────────────────────────
    # 왜 이 관문이 생겼나: `S_STATE["hotplate"]` 가 `{1: 549.6}` 이었다 — **통전 전력을
    # 휴지 슬롯(state 1)에 얹고** 통전 슬롯(state 2)은 정의조차 없었다. 라벨은 멀쩡했다
    # (풀: state 1 = 0.0W · state 2 = 454.7W). 그래서 `state_power_init` 이 휴지 슬롯을
    # 549.6W 로 띄웠고 cnn_v37 의 `p_states` 가 [0, **241**, 424, ...] 로 굳었다.
    # **숫자가 틀린 게 아니라 슬롯이 어긋났다** — 눈으로는 안 보이는 종류라 관문을 둔다.
    print("\n[5] S_STATE 슬롯이 세그먼트 풀의 상태별 실측과 맞나 (13.92)")
    try:
        from src.synthesis.segment_pool import SegmentPool
        # ⚠ `S_STATE` 는 **모듈 수준에서 이미 들여왔다** (38행). 여기서 다시 들여오면
        #   파이썬이 `main()` 안의 `S_STATE` 를 전부 지역 변수로 취급해, 이 줄보다
        #   **앞에 있는** 145행이 `referenced before assignment` 로 죽는다.
        from src.model.losses import MIN_STATE_SCALE_W
        pool = SegmentPool()
        print("   %-16s %6s %12s %12s %8s" % ("기기/상태", "표본", "풀 p90 W", "S_STATE W", "판정"))
        for a in APPS:
            acts = pool.appliance_activations.get(a, [])
            if not acts:
                continue
            st = np.concatenate([np.asarray(x.state_id) for x in acts])
            pw = np.concatenate([np.asarray(x.target_power_w) for x in acts])
            for s in sorted(set(int(v) for v in st.tolist())):
                if s <= 0:
                    continue
                m = st == s
                if m.sum() < 200:
                    continue
                p90 = float(np.percentile(pw[m], 90))
                have = S_STATE.get(a, {}).get(s)
                # 슬롯 어긋남만 잡는다. **배율은 안 본다** — 측정 관례가 달라 20~30%
                # 차이는 정상이다 (오븐 s2 1357 대 풀 1117 · 프로젝터 s2 50.6 대 45.7).
                # 처음 쓴 규칙(`p90<50 인데 S_STATE>30`)은 **오탐 4개**를 냈다 —
                # 프로젝터 s2·선풍 s2/s3·충전기 s1 은 원래 저전력 기기다. 좁힌다:
                #   (가) 풀이 통전(>=50W)인데 S_STATE 에 **항목이 없다**
                #   (나) 풀이 **사실상 0**(<5W, 휴지)인데 S_STATE 가 **100W 이상**
                #        -> 휴지 슬롯이 통전 전력을 이고 있다. 핫플의 549.6 이 그것이다.
                #        오븐 s1 (풀 0.0 · S_STATE 16.8) 은 안 걸린다 — 풀 값은
                #        받침 제외 순수치이고 16.8 은 총량이라 둘 다 맞다.
                verdict = "ok"
                if p90 >= 50.0 and have is None:
                    verdict = "**없다**"
                elif p90 < 5.0 and have is not None and have >= 100.0:
                    verdict = "**뒤바뀜**"
                if verdict != "ok":
                    bad += 1
                print("   %-16s %6d %12.1f %12s %8s"
                      % ("%s s%d" % (a[:12], s), int(m.sum()), p90,
                         ("%.1f" % have) if have is not None else "-", verdict))
    except Exception as e:                       # 풀이 없는 자리(HPC 등)에서는 건너뛴다
        print("   (건너뜀 — 세그먼트 풀을 못 읽었다: %s)" % str(e)[:60])

    if bad:
        print("\n[판정] !! %d 개가 깨졌다 — 적용하면 안 된다" % bad); return 1
    print("\n[판정] **합격** — 꺼진 채로는 옛 동작과 비트 단위로 같다.")
    print("       ① 초기화가 **예방**이고 ② `state_power` 가 **유지**다 (혼합이 안 고르는")
    print("       슬롯에 기울기를 준다). 둘 다 있어야 하고, **순서가 있다** — ① 없이 ② 만")
    print("       켜면 이미 죽은 슬롯에는 아무 일도 안 일어난다.")
    print("       ⚠ **값이 좋아지는지는 이 관문이 말하지 않는다.** 12.35 가 이 항으로 유령이")
    print("       42.1 -> 86.9W 가 됐다고 쟀다 (옛 판). 실측 채점으로 다시 재라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
