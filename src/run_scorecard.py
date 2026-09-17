"""판 비교 채점표 — 관문·격차·전력 헤드를 한 표에 (13.55)
==============================================================
13.49 가 남긴 규율 때문에 셋을 **같이** 본다. 관문(AUC)만 보다가 전력 헤드가
죽은 것을 두 판 동안 못 봤다.

    ① 실측 AUC   적응 모델의 σ(on) 이 정답 ON 구간을 얼마나 가르나
    ② 합성 AUC   같은 것을 얼린 홀드아웃에서. **격차 = 합성 − 실측**
    ③ 전력 헤드   상태별 `power_states` 와 **그 상태에서의 예측 전력**

⚠ **pst 가 0 이라고 죽은 게 아니다** (13.55.5). 드라이기 s1 은 pst 4e-10 인데
  예측이 419.8W (참 455.9) 다 — 모델이 **혼합 가중을 전력 다이얼로** 쓰는 것이고
  v20b·v23 에서 모두 그렇다. 죽음의 판정은 **예측 전력이 참값보다 크게 모자란 것**
  이다. v22 의 드라이기·오븐이 그랬다 (예측이 정확히 0.0W).

⚠ **AUC 는 관문이 통째로 눌린 것을 못 본다** (13.57.4). 순위 기반이라 그 기기의
  ON 창이 다 같이 눌리면 OFF 창과의 순위는 그대로다. 자리별 배분을 볼 때는
  `run_site_split_check` 를 같이 쓸 것.

⚠ 여기 AUC 는 이 스크립트의 규약(5파일 통합, `scorable` 마스크, 타깃 사이클)이다.
  설계 기록의 published AUC 와 **직접 견주지 말 것** — 규약이 다르다. 판끼리는
  같은 규약이므로 서로 견줄 수 있다.

    python -m src.run_scorecard --bare results/cnn_v23.pt results/cnn_v24b.pt \
                               --adapt results/adapt_v23p_s0.pt
    python -m src.run_scorecard --adapt results/adapt_v24b_z_s*.pt --skip-real
"""
from typing import List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.holdout import load_holdout
from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model
from src.run_train_cnn import prepare_holdout_inputs

STEMS = ["test_1", "test_2", "test_3", "test_4", "test_5"]
HOLDOUT = "processed_data/holdout60_v22"


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """순위 기반 AUC. 동점은 평균 순위로 처리한다."""
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return float("nan")
    r = np.empty(len(score), float)
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    r[o] = np.arange(1, len(score) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = float(pos.sum())
    n0 = float(len(pos) - n1)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@torch.no_grad()
def holdout_pass(model, prep, dev: str, batch: int = 512):
    fa, wa = prep
    G, PS, PW = [], [], []
    for i in range(0, len(fa), batch):
        o = model(torch.from_numpy(fa[i:i + batch]).to(dev),
                  torch.from_numpy(wa[i:i + batch]).to(dev))
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        PS.append(o["power_states"].float().cpu().numpy())
        PW.append(o["power"].float().cpu().numpy())
    return np.concatenate(G), np.concatenate(PS), np.concatenate(PW)


def real_auc(model, apps: List[str], dev: str, stride: int = 30) -> dict:
    """파일별 관문 AUC (한 판). uncertain 구간은 `scorable` 로 뺀다."""
    return real_auc_many([("_", model)], apps, dev, stride)["_"]


def real_auc_many(pairs, apps: List[str], dev: str, stride: int = 30) -> dict:
    """여러 판을 **한 번에** 채점한다 — 파일이 바깥 고리다 (14.400).

    ⚠⚠ 왜 이렇게 바꿨나. 판마다 `forward_file` 을 부르면 그 안의
    `dense_targets`(**7.4초**)와 `solve_ghat`(**5.1초**)이 **판 수만큼 다시 돈다**.
    둘 다 **모델과 무관**하다 — 창은 짝수차 규약과 stride 로만 정해지고 Ĝ 는
    입력만의 함수다. 18판 x 5파일이면 **12.5초 x 90 = 약 19분이 순수 낭비**였다
    (전체 ~47분 중). 파일을 바깥으로 돌리면 파일당 **한 번**만 짓는다.

    ⚠ 메모리도 이 순서가 낫다 — 한 파일의 창만 들고 있으면 된다.
    """
    from src.model.realdata import dense_targets
    ev = load_events()
    per = {k: {a: {"s": [], "y": []} for a in apps} for k, _ in pairs}
    for st in STEMS:
        if is_sealed(st):
            continue
        #: 창과 Ĝ 를 **파일당 한 번**. `forward_file` 과 같은 인자여야 한다.
        #: ⚠⚠ **자리 보정을 단 판은 창이 다르다** — 공유하면 조용히 틀린 입력이 된다
        #:   ([[verify-the-input-path-not-just-the-model]]). 하나라도 있으면 멈춘다.
        _stf = {id(getattr(m, "site_transfer", None)) for _, m in pairs
                if getattr(m, "site_transfer", None) is not None}
        if _stf:
            raise SystemExit(
                "✖ 자리 보정(site_transfer)을 단 판이 섞여 있다 — 창을 공유하면 안 된다. "
                "그 판은 따로 채점해라 (1단계 판끼리만 같이 넣어라)")
        rw = dense_targets(st, stride=stride, site_transfer=None)
        _gh = None
        if any(float(getattr(m, "comb_tau", 0.0) or 0.0) > 0 for _, m in pairs):
            from src.run_plot_real import solve_ghat as _sg
            _gh = _sg(st, rw, list(apps))[0]
        n_cyc = int(ev[st]["cycles"])
        on, sc = build_on_off_truth(st, apps, n_cyc, ev)
        for key, model in pairs:
            d = forward_file(model, st, dev, stride=stride, _rw=rw, _ghat=_gh)
            t = np.asarray(d["targets"], int).clip(0, n_cyc - 1)
            for j, a in enumerate(apps):
                m = sc[t, j]
                if not m.any():
                    continue
                per[key][a]["s"].append(d["gate"][m, j])
                per[key][a]["y"].append(on[t, j][m])
        print("    %s 채점 완료 (판 %d)" % (st, len(pairs)), flush=True)
    out = {}
    for key, _ in pairs:
        o = {}
        for a in apps:
            if not per[key][a]["s"]:
                continue
            sv = np.concatenate(per[key][a]["s"])
            yv = np.concatenate(per[key][a]["y"])
            o[a] = (auc(sv, yv), int(yv.sum()), int(len(yv)))
        out[key] = o
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", nargs="*", default=[], help="1단계 체크포인트")
    ap.add_argument("--adapt", nargs="*", default=[], help="2단계 체크포인트")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--skip-real", action="store_true",
                    help="실측 채점을 건너뛴다 (합성 홀드아웃과 전력 헤드만)")
    #: ★ 14.400 — 홀드아웃 **없이** 실측만 잰다. `run_gate_pintot` 처럼 실측 자는
    #  `composite_eval` 을 읽어 **이 노트북에서만** 도는데, 홀드아웃은 HPC 에 있고
    #  5GB 라 내려받을 것이 아니다. 판정할 때 실측 AUC 만 필요한 경우가 그것이다.
    ap.add_argument("--skip-holdout", action="store_true",
                    help="합성 홀드아웃을 건너뛴다 (실측 AUC 만) — 홀드아웃이 없는 자리에서")
    a = ap.parse_args()
    if not a.bare and not a.adapt:
        raise SystemExit("--bare 또는 --adapt 로 체크포인트를 주십시오")
    if a.skip_holdout and a.skip_real:
        raise SystemExit("--skip-holdout 과 --skip-real 을 같이 주면 잴 것이 없다")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    #: ⚠⚠ 14.400 — **창을 짓기 전에** 짝수차 규약을 체크포인트에 맞춘다.
    #  `inputs.EVEN_MEDIAN` 은 모듈 전역이라 안 맞추면 `forward_file` 이 조용히
    #  분포 밖 입력을 만든다 ([[verify-the-input-path-not-just-the-model]]).
    #  규약이 갈리는 판을 한 번에 주면 `sync_even_median` 이 **멈춘다** — 그게 맞다.
    from src.run_gate_check import sync_even_median
    sync_even_median(a.bare + a.adapt)

    if a.skip_holdout:
        from src.model.build import build_model as _bm
        _ck0 = torch.load((a.bare + a.adapt)[0], map_location="cpu", weights_only=False)
        apps = list(_ck0["appliances"])
        hs = prep = yon = yst = None
    else:
        hs = load_holdout(a.holdout)
        apps = hs.appliances
        prep = prepare_holdout_inputs(hs)
    if hs is not None:
        yon = hs.y_on.astype(bool)
        yst = np.asarray(hs.y_state, int)

    rows = {}
    _models = []
    for p in a.bare + a.adapt:
        m, mapps, ck = load_model(p, dev)
        assert list(mapps) == list(apps), (mapps, apps)
        syn, pst = {}, {}
        if hs is not None:
            g, ps, pw = holdout_pass(m, prep, dev)
            syn = {x: auc(g[:, j], yon[:, j]) for j, x in enumerate(apps)}
            for j, x in enumerate(apps):
                for s in range(ps.shape[2]):
                    k = yon[:, j] & (yst[:, j] == s)
                    if k.sum() >= 20:
                        pst[(x, s)] = (float(np.median(ps[k, j, s])),
                                       float(np.median(pw[k, j])),
                                       float(np.median(hs.y_power[k, j])))
        rows[p] = {"syn": syn, "pst": pst, "real": {},
                   "aux_z": bool(ck.get("aux_z", False))}
        _models.append((p, m))
        print(f"  {p} 적재", flush=True)
    if not a.skip_real:
        #: ★ 14.400 — **파일이 바깥 고리다.** 창·Ĝ 를 파일당 한 번만 짓는다
        for k, v in real_auc_many(_models, apps, dev).items():
            rows[k]["real"] = v
    _models.clear()
    torch.cuda.empty_cache()

    names = list(rows)
    short = [n.replace("\\", "/").split("/")[-1].replace(".pt", "") for n in names]

    if not a.skip_real:
        print("\n" + "=" * 104)
        print("① 실측 AUC (관문) — 5파일 통합, uncertain 제외")
        print("=" * 104)
        print(f"  {'기기':20s}" + "".join(f"{s[:15]:>17s}" for s in short))
        for x in apps:
            if not any(x in rows[n]["real"] for n in names):
                continue
            print(f"  {x:20s}" + "".join(
                f"{rows[n]['real'][x][0]:17.3f}" if x in rows[n]["real"]
                else f"{'—':>17s}" for n in names))
        print(f"  {'평균':20s}" + "".join(
            f"{np.nanmean([v[0] for v in rows[n]['real'].values()]):17.3f}"
            for n in names))

    #: ⚠ 14.400 — `--skip-holdout` 이면 `syn`·`pst` 가 **비어 있다**. 안 막으면 여기서
    #  KeyError 로 죽는다. 실측 표는 **이미 다 찍은 뒤**라 더 고약하다 — 값은 다 나왔는데
    #  종료코드만 1 이 된다 (배경 작업이 "실패" 로 보인다).
    if a.skip_holdout:
        print("\n  (합성 홀드아웃은 건너뛰었다 — --skip-holdout)")
        return
    print("\n" + "=" * 104)
    print("② 합성 홀드아웃 AUC" + ("" if a.skip_real else " 와 격차 (합성 − 실측)"))
    print("=" * 104)
    print(f"  {'기기':20s}" + "".join(f"{s[:15]:>17s}" for s in short))
    for x in apps:
        print(f"  {x:20s}" + "".join(f"{rows[n]['syn'][x]:17.3f}" for n in names))
    if not a.skip_real:
        print(f"\n  {'격차':20s}" + "".join(f"{s[:15]:>17s}" for s in short))
        for x in apps:
            if not any(x in rows[n]["real"] for n in names):
                continue
            print(f"  {x:20s}" + "".join(
                f"{rows[n]['syn'][x] - rows[n]['real'][x][0]:17.3f}"
                if x in rows[n]["real"] else f"{'—':>17s}" for n in names))

    print("\n" + "=" * 104)
    print("③ 상태별 전력 헤드 — 예측W / pst.  **판정은 예측W 다** (13.49·13.55.5)")
    print("=" * 104)
    keys = sorted({k for n in names for k in rows[n]["pst"]})
    print(f"  {'기기 · 상태':20s}{'참W':>8s}" + "".join(f"{s[:16]:>19s}" for s in short))
    for x, s_ in keys:
        vals = [rows[n]["pst"].get((x, s_)) for n in names]
        tw = next((v[2] for v in vals if v is not None), 0.0)
        if tw < 5.0:
            continue
        print(f"  {x + ' s' + str(s_):20s}{tw:8.1f}" + "".join(
            f"{v[1]:11.1f} /{v[0]:7.4g}" if v is not None else f"{'—':>19s}"
            for v in vals))
    dead = [(n, k) for n in names for k, v in rows[n]["pst"].items()
            if v[2] >= 30.0 and v[1] < 0.2 * v[2]]
    print("\n  죽은 헤드 (예측 < 참값 20%):", "없음" if not dead else
          ", ".join(f"{n.replace(chr(92), '/').split('/')[-1]}:{k[0]}s{k[1]}"
                    for n, k in dead))


if __name__ == "__main__":
    main()
