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
    """파일별 관문 AUC. uncertain 구간은 `scorable` 로 뺀다."""
    ev = load_events()
    per = {a: {"s": [], "y": []} for a in apps}
    for st in STEMS:
        if is_sealed(st):
            continue
        d = forward_file(model, st, dev, stride=stride)
        n_cyc = int(ev[st]["cycles"])
        on, sc = build_on_off_truth(st, apps, n_cyc, ev)
        t = np.asarray(d["targets"], int).clip(0, n_cyc - 1)
        for j, a in enumerate(apps):
            m = sc[t, j]
            if not m.any():
                continue
            per[a]["s"].append(d["gate"][m, j])
            per[a]["y"].append(on[t, j][m])
    out = {}
    for a in apps:
        if not per[a]["s"]:
            continue
        s = np.concatenate(per[a]["s"])
        y = np.concatenate(per[a]["y"])
        out[a] = (auc(s, y), int(y.sum()), int(len(y)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", nargs="*", default=[], help="1단계 체크포인트")
    ap.add_argument("--adapt", nargs="*", default=[], help="2단계 체크포인트")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--skip-real", action="store_true",
                    help="실측 채점을 건너뛴다 (합성 홀드아웃과 전력 헤드만)")
    a = ap.parse_args()
    if not a.bare and not a.adapt:
        raise SystemExit("--bare 또는 --adapt 로 체크포인트를 주십시오")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    hs = load_holdout(a.holdout)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    yon = hs.y_on.astype(bool)
    yst = np.asarray(hs.y_state, int)

    rows = {}
    for p in a.bare + a.adapt:
        m, mapps, ck = load_model(p, dev)
        assert list(mapps) == list(apps), (mapps, apps)
        g, ps, pw = holdout_pass(m, prep, dev)
        syn = {x: auc(g[:, j], yon[:, j]) for j, x in enumerate(apps)}
        pst = {}
        for j, x in enumerate(apps):
            for s in range(ps.shape[2]):
                k = yon[:, j] & (yst[:, j] == s)
                if k.sum() >= 20:
                    pst[(x, s)] = (float(np.median(ps[k, j, s])),
                                   float(np.median(pw[k, j])),
                                   float(np.median(hs.y_power[k, j])))
        real = {} if a.skip_real else real_auc(m, mapps, dev)
        rows[p] = {"syn": syn, "pst": pst, "real": real,
                   "aux_z": bool(ck.get("aux_z", False))}
        print(f"  {p} 완료", flush=True)
        del m
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
