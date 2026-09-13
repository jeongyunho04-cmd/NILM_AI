# -*- coding: utf-8 -*-
"""시퀀스 체크포인트를 **실측 5파일**에 채점한다 (13.84.27).

HPC 는 공용 계정이라 실측 npz 를 두지 않는다 — 거기서는 `--no-real` 로 합성 홀드아웃만 보고,
체크포인트(2.5MB)를 받아 **여기서** 실측을 잰다. 실측은 원래 학습에 안 들어가므로 결과는 같다.

    python -X utf8 src/run_score_seq.py results/seq_v37.pt results/seq_scratch.pt
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.model.transition import N_FEAT
from src.run_gate_check import load_model
from src.evaluation.metrics import RESISTIVE_MIN_TRUE_W, RESISTIVE_TRIO
from src.run_train_seq import FS, FILES, real_windows

#: 0.2절의 다른 한 무리. 저항 무리와 **같은 물음**을 SMPS 쪽에서 묻는다.
SMPS_TRIO = ("beam_projector", "laptop_charger", "minipc")

#: ⚠ `metrics.RESISTIVE_TRIO` 는 (포트·오븐·드라이기) **셋**인데 **핫플이 빠져 있다.**
#: 핫플도 니크롬이고, 신원을 묻는 자리에서 후보를 빼면 "핫플이 켜졌는데 오븐에 얹었다" 를
#: 못 잡는다. 여기서는 **넷**으로 묻는다. 옛 숫자와 견줄 때는 무리가 다름을 유의할 것
#: ([[match-the-scoring-convention-before-comparing]]).
RESISTIVE_ALL = tuple(RESISTIVE_TRIO) + ("hotplate",)

#: **듀티 기기** — 스위치는 켜져 있는데 서모스탯이 전력을 끊는 기기.
#:
#: 라벨 `is_on` 은 **스위치 상태**를 적는다. 핫플의 `ARMED_IDLE`(2.0W)과 오븐의
#: 서모스탯 휴지 구간이 전부 `is_on=True` 인데, 모델이 보는 것은 **전력**이라
#: 그 구간을 꺼짐으로 읽는다. 그래서 핫플은 정밀도 0.995 인데 재현율 0.409 —
#: 그리고 그 0.409 는 실측 가동률 0.417 과 **거의 같다.** 모델이 틀린 게 아니라
#: 두 규약이 다른 것이다.
#:
#: ⚠ 이걸 모른 채 기기별 on/off 를 평균하면 **아홉 중 둘이 평균의 44%를 끌어내린다.**
#: 2026-09-13 에 그 평균을 보고 사슬이 on/off 를 +0.068 올린다고 적었는데,
#: 핫플을 빼자 +0.010 으로 주저앉았다. 표에서 **항상 갈라 찍는다.**
#:
#: ⚠⚠ **더 나쁜 것 (13.98). 이 줄은 모델이 아니라 `losses.S_STATE` 의 상수가 정한다.**
#: `power = sigmoid(on) * sum_s mix[s] * softplus(p_states[s])` 이므로, 어떤 기기의
#: 상태 슬롯에 **작은 와트가 하나도 없으면** 그 기기는 "켜짐인데 0W" 를 표현할 수가
#: 없다 — on 을 끄는 수밖에. cnn_v37 은 `hotplate: {1: 549.6}` 하나였고 휴지 스텝의
#: 게이트 중앙값이 **0.000**, 지금은 `{1: 10.0, 2: 549.6}` 이고 **0.993** 이다.
#: **상태 혼합은 둘 다 슬롯 1 에 0.99 로 똑같이 몰려 있다.** 바뀐 것은 표의 숫자뿐이고
#: **전력 출력은 두 판이 사실상 같다.** 오븐은 슬롯이 원래 맞아서(`{1: 16.8, ...}`)
#: 재현율이 v37 에서 이미 0.990 이고 안 움직였다 — 그게 대조군이다.
#: 그러므로 **듀티 줄은 분해 성능이 아니라 "스위치 상태 추론" 이라는 딴 과제**이고,
#: 그 점수는 표 상수에 딸려 온다. **판정에 쓰지 마라.**
DUTY_APPS = ("hotplate", "oven")

#: 전력 채점에서 **혼자 켜진 창**으로 인정할 최소 순전력 (관측 − 전부꺼짐 기준선).
#: 이 아래는 전이 순간이거나 기준선 잡음이라 상대오차의 분모가 터진다.
POWER_FLOOR_W = 5.0

#: 전력 채점에 필요한 최소 혼자-켜짐 창 수. 실측에서 미니PC 8~29 · 오븐 11 이 고작이라
#: 더 올리면 표가 통째로 빈다. **얇다는 것을 표에 같이 찍는다.**
SOLO_MIN = 8


def score(ck_path, dev, real_cache, detail=False):
    """(사슬 평균, 창별 평균, 미니PC 파일별)  —  `detail` 이면 **기기별**을 넷째로 더 준다.

    ⚠ 반환 길이를 기본값에서 바꾸지 않는다. `run_score_seqcurve` 가 `c, w, rows = ...` 로
      셋을 푼다 — 말없이 늘리면 거기가 깨진다.

    넷째는 `{기기: {파일: (사슬, 창별)}}` 이고, **그 파일에 있는 기기만** 담는다 (평균과
    같은 규약이다). 오래 미니PC 만 찍어 왔는데 저항성 부하도 평균에 들어가 있다 —
    머릿수가 어디서 오는지 보려면 이 표가 필요하다.
    """
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    # 몸통은 체크포인트가 통째로 담고 있다. 구조·가림은 `--init`/`--ref` 가 가리키던 판에서 온다.
    # ⚠ 학습 때와 **같은 가림**이어야 한다. 체크포인트가 그 사실을 담고 있다.
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)),
                       # 사영 설정은 **이 체크포인트**의 것이다 (ref 의 것이 아니다).
                       proj_from={k: ck[k] for k in
                                  ("proj", "proj_cap", "proj_floor", "proj_resp",
                           "appl_attn", "appl_attn_heads")
                                  if k in ck})[0]
    model.load_state_dict(ck["model"])
    model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"])
    heads.eval()

    if not real_cache:
        real_cache.update(real_windows(apps, ck["meta"]["grid_s"], dev))
    km = apps.index("minipc")
    rows, ca, wa = {}, [], []
    by_app, ident, pwr, recon = {}, {}, {}, {}
    with torch.no_grad():
        for stem, d in real_cache.items():
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                PW.append(o["power"].float())
            z = torch.cat(Z)[None]
            gl = torch.cat(GL)[None]
            em, on, off, ini = heads(z, torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
            path = viterbi(em, on, off, ini)[0].cpu().numpy()
            w = (gl[0] > 0).cpu().numpy()
            y = d["y"].astype(bool)
            # ⚠ `run_train_seq.score_real` 과 **같은 규약**이어야 한다 — 그 파일에 있는 기기만
            #   세고, 기기별 정확도를 평균한다. 전 기기를 세면 없는 기기가 공짜로 맞아
            #   0.929 가 0.961 로 보인다 ([[match-the-scoring-convention-before-comparing]]).
            for k in np.nonzero(d["present"])[0]:
                c1 = float((path[:, k] == y[:, k]).mean())
                w1 = float((w[:, k] == y[:, k]).mean())
                ca.append(c1); wa.append(w1)
                by_app.setdefault(apps[k], {})[stem] = (c1, w1)
            rows[stem] = (float((w[:, km] == y[:, km]).mean()),
                          float((path[:, km] == y[:, km]).mean()))
            # ── 전력 채점 — 실측에서도 잴 수 있다 (13.84.74) ──────────────────
            # 실측 복합에는 기기별 참 전력이 없다. 그러나 **혼자 켜진 창**에서는 만들 수
            # 있다: 그 파일의 **전부-꺼짐 기준선**(배경 + 꽂힌 것들의 대기)을 관측 총전력에서
            # 빼면 남는 것이 그 기기다. 기준선은 라벨로 고른 전부-꺼짐 창의 중앙이고
            # (13.84.64 ③ 이 쓴 그 양), 모델과 무관하게 정해진다.
            pw = torch.cat(PW).cpu().numpy()
            base = float(d.get("p_base", float("nan")))
            if np.isfinite(base):
                for k in np.nonzero(d["present"])[0]:
                    other = np.delete(np.arange(y.shape[1]), k)
                    solo = y[:, k] & (y[:, other].sum(1) == 0)
                    t = d["p_obs"] - base
                    # 기준선 잡음과 전이 순간을 뺀다 — 분모가 0 에 가까우면 상대오차가 터진다
                    solo = solo & (t > max(POWER_FLOOR_W, 0.0))
                    if solo.sum() < SOLO_MIN:
                        continue
                    e = np.abs(pw[solo, k] - t[solo]) / np.maximum(t[solo], 1e-6)
                    pwr.setdefault(apps[k], {})[stem] = (
                        float(np.median(e)), int(solo.sum()), float(np.median(t[solo])),
                        float(np.median(pw[solo, k])))
                # ── 잔차 — **가장 중요한 전력 지표** (13.84.74) ─────────────────
                # 기기별 라벨이 필요 없다. **모든 창**에서 잰다. 사용자 지적:
                # *"F1만 좋아지고 다 개박살났잖아. 잔차도 채점 기준에 넣어."* 맞다.
                #
                # ⚠⚠ **절대 W 만 찍으면 안 된다 (13.91).** 오븐/핫플 창은 관측이
                #   수백~1500W 라 **같은 상대오차가 절대 W 로 10배**가 된다. 그것을
                #   "큰 부하에서만 터진다" 로 읽었고 **틀렸다** — cnn_v37 은 오븐핫플
                #   4.1% · 그밖 4.1% 로 정확히 같았다. 사슬 학습은 둘 다 약 2배로
                #   **고르게** 나빠진다 (9.2% / 8.1%). 그래서 상대도 같이 찍는다.
                #
                # ⚠ 여기서 말하는 단계 이름: `seq_*` 는 **사슬 학습**이지 4.2절의
                #   **실측 준지도 적응(2단계)이 아니다.** 후자는 12.31 이 미니PC
                #   0.574 -> 0.137 붕괴로 되돌렸고 6절이 금지한다. `run_train_seq` 의
                #   `real` 은 채점에만 쓰이고 손실에 안 들어간다.
                r = pw.sum(1) + base - d["p_obs"]
                heavy = np.zeros(len(r), bool)
                for hx in ("oven", "hotplate"):
                    if hx in apps and d["present"][apps.index(hx)]:
                        heavy |= y[:, apps.index(hx)]
                rel_all = np.abs(r) / np.maximum(d["p_obs"], 10.0)
                recon[stem] = {
                    "abs": float(np.abs(r).mean()),                 # W, 절대평균
                    "bias": float(r.mean()),                        # W, 부호 있는 평균
                    "rel": float(np.median(rel_all)),
                    "abs_heavy": (float(np.abs(r[heavy]).mean()) if heavy.sum() >= 10
                                  else float("nan")),               # 오븐/핫플 켜짐 구간
                    "abs_light": (float(np.abs(r[~heavy]).mean()) if (~heavy).sum() >= 10
                                  else float("nan")),
                    # **판정은 이 둘로 한다** (13.91). 절대는 눈금에 끌려간다.
                    "rel_heavy": (float(np.median(rel_all[heavy])) if heavy.sum() >= 10
                                  else float("nan")),
                    "rel_light": (float(np.median(rel_all[~heavy])) if (~heavy).sum() >= 10
                                  else float("nan")),
                    "n": int(len(r))}
            # ── 무리 안 **신원** — 켜짐 맞히기와 다른 물음이다 ─────────────────
            # 저항 3종은 켜짐이 쉽다 (수백 W 가 들어온다). 어려운 것은 **누구냐** 이고
            # 0.2절이 "고조파 지문이 사실상 같은 무리" 라고 부른 그것이다.
            # `metrics.resistive_confusion` 과 같은 물음인데, 실측 파일에는 참 **전력**이
            # 없고 켜짐 라벨만 있으므로 그것으로 고른다: 셋 중 **정확히 하나만 켜진 창**에서
            # 모델이 셋 중 누구에게 전력을 가장 많이 줬나.
            # ⚠ 그 파일에 **없는** 기기도 후보로 둔다 — 없는 기기에 얹으면 그게 유령이다.
            #
            # ⚠ **전력 하한이 있어야 한다.** 오븐·핫플의 켜짐 라벨에는 듀티가 쉬는
            #   구간(팬/조명 ~16W)이 섞여 있다 — 실측에서 오븐 켜짐의 **10~40%** 가
            #   총전력 300W 아래다. 그 창에서는 네 예측이 전부 0 근처라 argmax 가
            #   **난수**이고, 그 잡음이 모델 간 차이로 보인다. `metrics` 가 같은 이유로
            #   `RESISTIVE_MIN_TRUE_W` 를 둔다 (거기서는 참 전력, 여기서는 실측 파일에
            #   기기별 참 전력이 없으니 **관측 총전력**으로 건다 — 보수적이다).
            hi = d["p_obs"] >= RESISTIVE_MIN_TRUE_W
            for gname, grp, gate in (("저항 무리", RESISTIVE_ALL, hi),
                                     ("SMPS 무리", SMPS_TRIO, np.ones(len(y), bool))):
                gi = [apps.index(x) for x in grp if x in apps]
                if len(gi) < 2:
                    continue
                yg = y[:, gi]
                solo = (yg.sum(1) == 1) & gate
                if solo.sum() < 10:
                    continue
                hit = np.argmax(pw[solo][:, gi], 1) == np.argmax(yg[solo], 1)
                ident.setdefault(gname, {})[stem] = (float(hit.mean()), int(solo.sum()))
    if detail:
        return (float(np.mean(ca)), float(np.mean(wa)), rows,
                {"app": by_app, "ident": ident, "power": pwr, "recon": recon})
    return float(np.mean(ca)), float(np.mean(wa)), rows


#: 표시용 무리. 나머지는 자동으로 '기타' 로 간다.
GROUPS = (("SMPS 3종", SMPS_TRIO), ("저항 4종", RESISTIVE_ALL))


def main():
    cks = sys.argv[1:] or ["results/seq_e2e.pt"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = {}, {}
    for c in cks:
        out[c] = score(c, dev, cache, detail=True)
        print("%-28s 실측 사슬 %.4f / 창별 %.4f" % (c, out[c][0], out[c][1]), flush=True)

    # ── 기기별 — 머릿수가 어디서 오나 ─────────────────────────────────────────
    # 오래 미니PC 만 찍어 왔는데 평균에는 저항성 부하도 들어가 있다. 무리마다 따로 본다.
    print("\n기기별 **사슬** 정확도 (그 기기가 있는 파일들의 평균 · 괄호는 파일 수)")
    short = [c.split("/")[-1][:-3][:13] for c in cks]
    print("  %-18s %s" % ("기기", "".join("  %-15s" % s for s in short)))
    seen = set()
    for gname, grp in GROUPS + (("기타", None),):
        names = [a for a in grp] if grp else \
            sorted({a for c in cks for a in out[c][3]["app"]} - seen)
        rows_ = []
        for a in names:
            if not any(a in out[c][3]["app"] for c in cks):
                continue
            seen.add(a)
            line = "  %-18s" % a
            for c in cks:
                v = out[c][3]["app"].get(a, {})
                line += "  %10.4f(%d) " % (np.mean([x[0] for x in v.values()]), len(v)) \
                    if v else "  %15s" % "—"
            rows_.append(line)
        if rows_:
            print("  [%s]" % gname)
            print("\n".join(rows_))
            # 무리 평균
            line = "  %-18s" % "  무리 평균"
            for c in cks:
                v = [x[0] for a in names if a in out[c][3]["app"]
                     for x in out[c][3]["app"][a].values()]
                line += "  %10.4f     " % np.mean(v) if v else "  %15s" % "—"
            print(line)

    # ── 무리 안 신원 — 켜짐이 아니라 **누구냐** ───────────────────────────────
    print("\n무리 안 **신원** — 하나만 켜진 창에서 전력을 맞는 기기에 줬나")
    print("  (저항 무리는 **넷** · 관측 총전력 >= %dW 인 창만 — 오븐/핫플 듀티 쉼을 뺀다)"
          % int(RESISTIVE_MIN_TRUE_W))
    for gname in ("저항 무리", "SMPS 무리"):
        print("  [%s]" % gname)
        for stem in FILES:
            if not any(stem in out[c][3]["ident"].get(gname, {}) for c in cks):
                continue
            line = "  %-18s" % stem
            n = 0
            for c in cks:
                e = out[c][3]["ident"].get(gname, {}).get(stem)
                line += "  %10.3f     " % e[0] if e else "  %15s" % "—"
                n = e[1] if e else n
            print(line + " n=%d" % n)
        line = "  %-18s" % "  가중 평균"
        for c in cks:
            e = out[c][3]["ident"].get(gname, {})
            if e:
                w_ = np.array([v[1] for v in e.values()], float)
                line += "  %10.3f     " % (np.average([v[0] for v in e.values()], weights=w_))
            else:
                line += "  %15s" % "—"
        print(line)

    # ── 잔차 — 사용자가 요구한 채점 기준 (13.84.74) ──────────────────────────
    print("\n**잔차** — |관측 총전력 - (예측합 + 전부꺼짐 기준선)| (W, 모든 창)")
    print("  %-24s %s" % ("", "".join("  %-15s" % t for t in short)))
    for stem in FILES:
        if not any(stem in out[c][3]["recon"] for c in cks):
            continue
        for key, nm in (("abs", "절대평균"), ("abs_heavy", "오븐/핫플 켜짐"),
                        ("abs_light", "그 밖")):
            line = "  %-24s" % ("%s  %s" % (stem, nm) if key == "abs" else "      " + nm)
            for c in cks:
                e = out[c][3]["recon"].get(stem)
                v = e[key] if e else float("nan")
                line += "  %13.1fW " % v if np.isfinite(v) else "  %15s" % "-"
            print(line)
    line = "  %-24s" % "**전 파일 절대평균**"
    for c in cks:
        v = [e["abs"] for e in out[c][3]["recon"].values()]
        line += "  %13.1fW " % np.mean(v) if v else "  %15s" % "-"
    print(line)

    # ── 상대 잔차 — **판정은 이쪽으로 한다** (13.91) ─────────────────────────
    print("\n**상대 잔차** — |잔차| / 관측 총전력 (중앙값). ★ 절대보다 이쪽을 보라")
    print("  %-24s %s" % ("", "".join("  %-15s" % t for t in short)))
    for stem in FILES:
        if not any(stem in out[c][3]["recon"] for c in cks):
            continue
        for key, nm in (("rel", "전체"), ("rel_heavy", "오븐/핫플 켜짐"),
                        ("rel_light", "그 밖")):
            line = "  %-24s" % ("%s  %s" % (stem, nm) if key == "rel" else "      " + nm)
            for c in cks:
                e = out[c][3]["recon"].get(stem)
                v = e.get(key, float("nan")) if e else float("nan")
                line += "  %13.1f%% " % (100 * v) if np.isfinite(v) else "  %15s" % "-"
            print(line)
    for key, nm in (("rel", "**전 파일 상대 중앙**"), ("rel_heavy", "  오븐/핫플"),
                    ("rel_light", "  그 밖")):
        line = "  %-24s" % nm
        for c in cks:
            v = [e[key] for e in out[c][3]["recon"].values()
                 if np.isfinite(e.get(key, float("nan")))]
            line += "  %13.1f%% " % (100 * np.mean(v)) if v else "  %15s" % "-"
        print(line)

    print("\n  ⚠⚠ **절대 W 로 판정하지 마라 (13.91).** 오븐/핫플 창은 관측이 수백~1500W 라")
    print("     **같은 상대오차가 절대 W 로 10배**가 된다. 13.84.74 가 그것을 \"큰 부하에서만")
    print("     터진다\" 로 읽었고 **틀렸다** — cnn_v37 은 오븐핫플 4.1% · 그밖 4.1% 로 같다.")
    print("     사슬 학습은 4.1%->9.2% · 4.1%->8.1% 로 **고르게** 나빠진다. 크기 무관이다.")
    print("  ⚠ 단계 이름: `seq_*` 는 **사슬 학습**(순수 합성)이지 4.2절의 **실측 준지도")
    print("     적응(2단계)이 아니다.** 후자는 12.31 이 미니PC 0.574->0.137 로 되돌렸고")
    print("     6절이 금지한다. `run_train_seq` 의 `real` 은 채점에만 쓰인다.")

    print("\n미니PC 시간 정확도 (실측)")
    # ⚠ 마지막 칸은 v37 기준선이 **아니다** — 첫 체크포인트가 그 시점에 낸 창별 값이다.
    #   몸통이 학습되면서 같이 움직인다. 고정 기준선과 견주려면 cnn_v37 을 따로 채점하라.
    print("  파일     " + "".join("  %-14s" % c.split("/")[-1][:-3] for c in cks)
          + "   %s의창별" % cks[0].split("/")[-1][:-3][:8])
    for stem in FILES:
        if stem not in out[cks[0]][2]:
            continue
        line = "  %-8s" % stem
        for c in cks:
            line += "  %14.3f" % out[c][2][stem][1]
        print(line + "  %6.3f" % out[cks[0]][2][stem][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
