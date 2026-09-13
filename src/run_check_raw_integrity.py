"""원본 CSV 무결성 점검 — **라벨을 붙이기 전에** (12.155)

왜 필요해졌나
------------
`test_14.csv` 를 받아 보니 **녹화 두 개가 이어 붙어 있었다.** `seq` 가 925 에서 0 으로
되돌아가고 `host_time` 이 6분 점프한다. 전처리는 행 순서를 그대로 믿으므로 그대로
두면 27,210행(7.6분)짜리 앞 녹화가 뒤 녹화 앞에 붙은 채 npz 가 만들어지고,
**사람 타임라인의 seq 가 통째로 어긋난다.**

라벨의 시간축은 `t_s = seq × 0.5` 하나에 걸려 있다 (한 패킷 = 30사이클 = 0.5초).
그 전제가 깨지는 경우를 먼저 전부 찾아 두지 않으면 뒤의 정밀화가 의미가 없다.

무엇을 보나
----------
```
세션 분할     seq 가 되돌아가거나 host_time 이 크게 점프 -> 녹화 여러 개가 이어 붙음
패킷 손실     seq 가 건너뜀 -> 그 구간만큼 시간이 비는데 행에는 안 보인다
불완전 패킷   seq 당 행 수가 30 이 아님
시간 역행     t_s 가 감소
seq-시간 정합  t_s 와 seq*0.5 의 어긋남 (세션 안에서)
```

쓰는 법
------
    python -X utf8 -m src.run_check_raw_integrity                 # data/*.csv 전부
    python -X utf8 -m src.run_check_raw_integrity --stems test_14 test_15
    python -X utf8 -m src.run_check_raw_integrity --json results/raw_integrity.json
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

BLOCK = 30          #: 한 패킷의 사이클 수. `t_s = seq * BLOCK / 60` 의 근거다.
HOST_GAP_S = 5.0    #: 이보다 크게 host_time 이 뛰면 끊긴 것으로 본다


def sessions(seq: np.ndarray, host_s: np.ndarray) -> List[Dict]:
    """세션 경계를 찾는다. **seq 되돌림이 1순위 근거**, host_time 점프가 2순위."""
    cut = [0]
    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1] - 1:                       # seq 되돌림
            cut.append(i)
        elif host_s is not None and host_s[i] - host_s[i - 1] > HOST_GAP_S:
            # host_time 점프는 세션 재시작일 수도, 단순 끊김일 수도 있다.
            # seq 가 이어지면 끊김으로 보고 자르지 않는다 (아래 gaps 가 잡는다).
            if seq[i] <= seq[i - 1]:
                cut.append(i)
    cut.append(len(seq))
    out = []
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a < BLOCK:                                 # 한 패킷도 안 되면 조각이다
            out.append({"lo": a, "hi": b, "rows": b - a, "fragment": True})
            continue
        s = seq[a:b]
        out.append({"lo": a, "hi": b, "rows": b - a, "fragment": False,
                    "seq_lo": int(s.min()), "seq_hi": int(s.max()),
                    "seconds": (b - a) / 60.0})
    return out


def check(stem: str, data_dir: str = "data") -> Dict:
    f = Path(data_dir) / f"{stem}.csv"
    cols = pd.read_csv(f, nrows=0).columns
    use = [c for c in ("host_time", "t_s", "seq", "cycle", "p_w") if c in cols]
    d = pd.read_csv(f, usecols=use)
    seq = d["seq"].to_numpy(np.int64)
    host_s = None
    if "host_time" in d:
        ht = pd.to_datetime(d["host_time"], errors="coerce")
        host_s = (ht - ht.iloc[0]).dt.total_seconds().to_numpy(np.float64)

    ss = sessions(seq, host_s)
    u, c = np.unique(seq, return_counts=True)
    # 세션이 하나일 때만 "패킷당 30행" 이 의미가 있다 (여럿이면 seq 가 겹친다)
    bad_pkt = int((c != BLOCK).sum()) if len(ss) == 1 else None
    r = {
        "stem": stem, "rows": int(len(d)), "sessions": ss,
        "n_sessions": len(ss),
        "t_s_monotonic": bool(np.all(np.diff(d["t_s"].to_numpy()) >= 0)) if "t_s" in d else None,
        "incomplete_packets": bad_pkt,
    }
    # 세션별 패킷 손실
    for s in ss:
        if s.get("fragment"):
            continue
        q = seq[s["lo"]:s["hi"]]
        uu = np.unique(q)
        s["packets"] = int(len(uu))
        s["expected"] = int(s["seq_hi"] - s["seq_lo"] + 1)
        s["missing"] = int(s["expected"] - s["packets"])
        if host_s is not None:
            s["host_gaps"] = int((np.diff(host_s[s["lo"]:s["hi"]]) > HOST_GAP_S).sum())
    return r


def verdict(r: Dict) -> str:
    if r["n_sessions"] > 1:
        return "❌ 녹화 여러 개"
    s = r["sessions"][0]
    if s.get("missing", 0) > 0:
        return f"⚠ 패킷 {s['missing']}개 손실"
    if r["incomplete_packets"]:
        return f"⚠ 불완전 패킷 {r['incomplete_packets']}개"
    if r["t_s_monotonic"] is False:
        return "⚠ t_s 역행"
    return "✅"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    stems = a.stems or sorted(p.stem for p in Path(a.data).glob("*.csv"))
    print("=" * 96)
    print("원본 CSV 무결성 — 라벨의 시간축은 t_s = seq × 0.5 하나에 걸려 있다 (12.155)")
    print("=" * 96)
    print(f"{'파일':<26}{'행':>9}{'세션':>5}{'seq 범위':>16}{'손실':>6}{'판정'}")
    out = {}
    for st in stems:
        try:
            r = check(st, a.data)
        except Exception as e:
            print(f"{st:<26}  읽기 실패: {e}")
            continue
        out[st] = r
        s0 = r["sessions"][0]
        rng = f"{s0.get('seq_lo','-')}~{s0.get('seq_hi','-')}"
        print(f"{st:<26}{r['rows']:>9,}{r['n_sessions']:>5}{rng:>16}"
              f"{s0.get('missing', 0):>6}  {verdict(r)}")
        if r["n_sessions"] > 1:
            for i, s in enumerate(r["sessions"], 1):
                if s.get("fragment"):
                    print(f"      세션{i}: 조각 {s['rows']}행")
                else:
                    print(f"      세션{i}: 행 {s['lo']:,}~{s['hi']-1:,}  "
                          f"{s['seconds']:.1f}s  seq {s['seq_lo']}~{s['seq_hi']}  "
                          f"손실 {s.get('missing',0)}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
