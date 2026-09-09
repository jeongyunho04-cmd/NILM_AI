# -*- coding: utf-8 -*-
"""전압 **꼬리**(h17~h31) 표를 원시 스냅샷에서 만든다 — back-fill (13.73, 2026-09-09).

왜
--
2Hz 녹화는 전압 고조파를 h15 까지만 준다. 그런데 SMPS 전류의 h11~h15 를 지배하는 것은
**h17 위의 전압**이다. 생성기 입구(`sim_harmonics`)에서 충전기 원시 20구간을 재면
`||sim|−|meas||/|meas|` 중앙값이

```
전압 소스        D h11  D h13  D h15  | E h11  E h13  E h15
h15 절단 (옛것)  0.274  0.449  0.632  | 0.165  0.297  0.223
+h17~23         0.073  0.097  0.100  | 0.032  0.068  0.077
+h17~31         0.064  0.042  0.072  | 0.017  0.031  0.088
```

이고, 이 절단이 곧 `s(p)` 곡선이 자리별 전력 의존을 못 만드는 이유이며 (회로모델이 h15 까지만
보므로 저P/고P h13 모양비가 D·E 둘 다 0.7 로 뭉개진다), 그것이 2단계에서 충전기가 프로젝터
와트를 먹는 사슬의 뿌리다 (설계 13.71). **펌웨어를 안 고치고** 지금 자료로 메우는 길이 이것이다.

무엇을 만드나
-----------
`data/raw_*.csv` 22개는 네 세션(D1·D2·E1·E2)을 다 덮는다. 파일마다 꼬리를 뽑고, 세션으로
귀속시켜 **세션당 복소 중앙 꼬리 하나**를 낸다. `vtexture` 가 그 세션의 텍스처 전부에 같은
꼬리를 붙인다.

  · **복소** 중앙값이다. 크기만 맞추고 위상을 무작위로 주면 **절단보다 나쁘다** (설계 4절).
  · **홀수만**. 계통 전압은 반파 대칭이라 짝수차가 원리적으로 없고, 실측에서도 짝수만 넣으면
    절단과 셋째 자리까지 같다.
  · 단자/개방 두 벌 (`rel`/`rel_open` 과 같은 규약). 원시 스냅샷은 그 자신의 부하가 물린
    상태라 강하 `Z(h)·I_h` 가 꼬리의 2~29% 다 — 원시에 전류가 있으니 공짜로 벗긴다.

⚠ 꼬리는 **세션 고정이 아니다.** 세션 안 산포(0.29~0.59)가 세션 사이(0.56~1.27)보다 작을
   뿐이다. 총량만이 자리의 성질이다 (D 0.43% · E 0.17%). 그래서 세션 중앙은 **평균**이지
   그 창의 값이 아니고, 자리 D 의 남는 오차(0.210 대 상한 0.112)가 그 몫이다.

⚠ **자리를 틀리면 절단보다 나쁘다.** 귀속이 정확해야 한다 — `vrms+vh3+vh5+vh7` 최근접으로
   붙이고, `abs(rfft[1])/(N/2)/sqrt(2)` 가 RMS 다 (√2 를 빼먹으면 216V 가 305V 로 읽혀
   전부 E 로 무너진다).

쓰는 법
------
    python -X utf8 -m src.run_build_vtail                 # processed_data/vtail.npz
    python -X utf8 -m src.run_build_vtail --dry-run       # 표만 보고 안 쓴다
"""
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import glob
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import pandas as pd

from src.preprocessing.file_registry import SITE_SESSIONS
from src.synthesis.vtexture import DEEMBED_L_H, DEFAULT_VTAIL_NPZ, TAIL_ORDERS

F = 60.0
NPC = 256                #: 원시 표본/주기
CHUNK = 10               #: 스펙트럼 하나를 뽑는 주기 수 (0.17초 — 파일 안 잡음 바닥 0.02~0.13)
MIN_CHUNKS = 4
#: 세션 Z 를 모를 때의 자리 대푯값 (`vtexture._site_session` 과 같은 폴백)
SITE_Z_OHM = {"D": 1.15, "E": 0.42}
#: 귀속에 쓰는 축의 저울 — 세션 사이 간격의 대략 1σ. vh7 이 D/E 를 가르는 축이라 가장 촘촘하다.
ATTRIB_SCALE = {"v": 3.0, "vh3": 0.3, "vh5": 0.3, "vh7": 0.2}


def _spectrum(V: np.ndarray, I: Optional[np.ndarray] = None):
    """1주기 파형 -> (V1 RMS, rel[h] = V_h/V_1 h배 관례, I_h 절대 RMS).

    `rel` 은 `Texture.rel` 과 **같은 영역·같은 관례**다 — 원시 `v_v` 도 2Hz 의 vh 도 계측 전압
    (RC τ 를 안 벗긴 것)이라 그대로 이어 붙일 수 있다. 참 전압으로 옮기는 것은 h1~h15 까지
    포함해 통째로 해야 하는 별건이고, 재 보니 꼬리를 넣은 뒤 h13 오차가 0.240 대 0.227 로
    갈리는 2차 항이다 (2026-09-09).
    """
    Xv = np.fft.rfft(V) / (NPC / 2) / np.sqrt(2.0)
    ph = np.angle(Xv[1])
    h = np.arange(len(Xv))
    rel = Xv * np.exp(-1j * h * ph) / max(abs(Xv[1]), 1e-12)
    Ic = None
    if I is not None:
        Xi = np.fft.rfft(I) / (NPC / 2) / np.sqrt(2.0)
        Ic = Xi * np.exp(-1j * h * ph)
    return abs(Xv[1]), rel, Ic


def _cmedian(A: np.ndarray) -> np.ndarray:
    """복소 중앙값 — 실수·허수 따로. 위상 상쇄를 피한다 (`vtexture` 와 같은 방식)."""
    A = np.asarray(A)
    return (np.median(A.real, 0) + 1j * np.median(A.imag, 0)).astype(np.complex128)


def read_file(path: str) -> Optional[dict]:
    """원시 CSV 하나 -> 꼬리(단자·개방 전) + 귀속에 쓸 지문."""
    d = pd.read_csv(path, usecols=["v_v", "i_a", "range"])
    nc = len(d) // NPC
    if nc < CHUNK * MIN_CHUNKS:
        return None
    v = d["v_v"].to_numpy(np.float64)[:nc * NPC].reshape(nc, NPC)
    i = d["i_a"].to_numpy(np.float64)[:nc * NPC].reshape(nc, NPC)
    ok = (d["range"].to_numpy()[:nc * NPC].reshape(nc, NPC) == 0).all(1)
    v, i = v[ok], i[ok]
    if len(v) < CHUNK * MIN_CHUNKS:
        return None
    step = max(CHUNK // 2, (len(v) - CHUNK) // 40)
    v1s, rels, ics = [], [], []
    for a in range(0, len(v) - CHUNK + 1, step):
        a1, rel, ic = _spectrum(v[a:a + CHUNK].mean(0), i[a:a + CHUNK].mean(0))
        v1s.append(a1); rels.append(rel); ics.append(ic)
    rel = _cmedian(np.array(rels))
    ic = _cmedian(np.array(ics))
    idx = np.asarray(TAIL_ORDERS, dtype=np.int64)
    # 파일 안 산포 — 잡음 바닥. 이것이 세션 안 산포와 같아지면 세션 중앙에 값이 없다는 뜻이다
    t = rel[idx]
    scat = float(np.median([np.linalg.norm(np.asarray(r)[idx] - t) / max(np.linalg.norm(t), 1e-12)
                            for r in rels]))
    return {"stem": os.path.basename(path)[:-4], "v1": float(np.median(v1s)),
            "vh3": 100 * abs(rel[3]), "vh5": 100 * abs(rel[5]), "vh7": 100 * abs(rel[7]),
            "tail": t, "i_tail": ic[idx], "i1": ic[1], "scat": scat, "n": len(rels)}


def attribute(r: dict) -> str:
    """세션 귀속 — `vrms + vh3 + vh5 + vh7` 최근접 (설계 5절)."""
    best, bd = "", 1e18
    for k, sp in SITE_SESSIONS.items():
        d = (((r["v1"] - sp["v_rms"]) / ATTRIB_SCALE["v"]) ** 2
             + ((r["vh3"] - sp["vh3_pct"]) / ATTRIB_SCALE["vh3"]) ** 2
             + ((r["vh5"] - sp["vh5_pct"]) / ATTRIB_SCALE["vh5"]) ** 2
             + ((r["vh7"] - sp["vh7_pct"]) / ATTRIB_SCALE["vh7"]) ** 2)
        if d < bd:
            best, bd = k, d
    return best


def deembed(r: dict, session: str) -> np.ndarray:
    """개방 꼬리 — 그 스냅샷 자신의 부하 강하를 벗긴다.  V_open = V_term + Z(h)·I_h.

    `vtexture` 의 `rel_open` 과 **같은 식·같은 L**(`DEEMBED_L_H`)이다.
    ⚠ `DEEMBED_L_H` 는 측정값이 아니다. h15 에서는 R 이 지배했지만 **h31 에서는 |jwL| 2.63Ω 으로
       L 이 지배한다.** 벗기는 양이 꼬리의 10% 안팎이라 L 이 2배 틀려도 꼬리의 ±3% 다.
    """
    sp = SITE_SESSIONS.get(session, {})
    z = sp.get("z_ohm") or SITE_Z_OHM.get(sp.get("site", ""))
    if z is None:
        return r["tail"]
    h = np.asarray(TAIL_ORDERS, dtype=np.float64)
    Z = float(z) + 1j * 2 * np.pi * F * h * DEEMBED_L_H
    Z1 = float(z) + 1j * 2 * np.pi * F * 1.0 * DEEMBED_L_H
    # ⚠ **단위를 맞춰라.** `tail` 은 무차원(V_h/V_1)이고 `i_tail` 은 절대 암페어다.
    #   강하도 V_1 으로 나눠야 더할 수 있다 (안 나누면 볼트를 비에 더한다).
    v1 = max(r["v1"], 1e-9)
    # 분모도 같이 벗긴다 — rel 은 V_h/V_1 이고 그 V_1 도 강하를 먹은 단자 값이다.
    v1_open = 1.0 + Z1 * r["i1"] / v1
    return (r["tail"] + Z * r["i_tail"] / v1) / v1_open


def build(raw_glob: str = "data/raw_*.csv") -> dict:
    files = sorted(glob.glob(raw_glob))
    rows: List[dict] = []
    for f in files:
        r = read_file(f)
        if r is None:
            print(f"  건너뜀 (주기 부족): {os.path.basename(f)}")
            continue
        r["sess"] = attribute(r)
        r["site"] = SITE_SESSIONS.get(r["sess"], {}).get("site", "")
        r["tail_open"] = deembed(r, r["sess"])
        rows.append(r)
    if not rows:
        raise RuntimeError(f"{raw_glob}: 읽을 원시가 없다")

    print("=== 원시 %d파일 — 지문·귀속·꼬리 ===" % len(rows))
    print("%-22s %6s %5s %5s %5s | %-4s %7s %7s %6s" %
          ("파일", "vrms", "vh3%", "vh5%", "vh7%", "세션", "h17+총%", "강하/꼬리", "파일안"))
    for r in rows:
        # 차수마다 |Z_h·I_h/V_1| / |V_h/V_1| 의 중앙값 — 설계 5.1 절 표와 같은 규약
        drop = float(np.median(np.abs(r["tail_open"] - r["tail"]) / np.maximum(np.abs(r["tail"]), 1e-12)))
        print("%-22s %6.1f %5.2f %5.2f %5.2f | %-4s %7.3f %7.3f %6.3f" %
              (r["stem"][4:], r["v1"], r["vh3"], r["vh5"], r["vh7"], r["sess"],
               100 * np.linalg.norm(r["tail"]), drop, r["scat"]))

    print("\n=== 벗기는 양 — 차수별 |강하| / |꼬리| (자리 중앙값) ===")
    print("%-6s %s" % ("자리", "  ".join("h%-5d" % h for h in TAIL_ORDERS)))
    for st in sorted({r["site"] for r in rows}):
        g = [r for r in rows if r["site"] == st]
        d = np.median([np.abs(r["tail_open"] - r["tail"]) / np.maximum(np.abs(r["tail"]), 1e-12)
                       for r in g], 0)
        print("%-6s %s" % (st, "  ".join("%6.3f" % x for x in d)))

    keys: List[str] = []
    tail: List[np.ndarray] = []
    topen: List[np.ndarray] = []
    nfil: List[int] = []
    site: List[str] = []
    print("\n=== 세션·자리별 복소 중앙 꼬리 ===")
    print("%-5s %-4s %3s %8s %8s   %s" % ("키", "자리", "n", "h17+총%", "세션안거리", "비고"))
    groups = [(k, [r for r in rows if r["sess"] == k], SITE_SESSIONS.get(k, {}).get("site", ""))
              for k in sorted({r["sess"] for r in rows})]
    groups += [(s, [r for r in rows if r["site"] == s], s) for s in sorted({r["site"] for r in rows})]
    for k, g, st in groups:
        if not g:
            continue
        m = _cmedian(np.array([r["tail"] for r in g]))
        mo = _cmedian(np.array([r["tail_open"] for r in g]))
        inner = float(np.median([np.linalg.norm(r["tail"] - m) / max(np.linalg.norm(m), 1e-12)
                                 for r in g])) if len(g) > 1 else 0.0
        note = "자리 중앙 (세션 폴백)" if k == st else ("⚠ 파일 %d개뿐" % len(g) if len(g) < 3 else "")
        print("%-5s %-4s %3d %8.3f %8.3f   %s" % (k, st, len(g), 100 * np.linalg.norm(m), inner, note))
        keys.append(k); tail.append(m); topen.append(mo); nfil.append(len(g)); site.append(st)

    return {"keys": np.array(keys), "tail": np.array(tail), "tail_open": np.array(topen),
            "orders": np.asarray(TAIL_ORDERS, dtype=np.int64), "n_files": np.array(nfil),
            "sites": np.array(site),
            "meta_json": json.dumps({"npc": NPC, "chunk": CHUNK, "deembed_l_h": DEEMBED_L_H,
                                     "domain": "measured (2Hz npz 와 같은 영역)",
                                     "convention": "V_h/V_1, vhdeg_h = arg(V_h) - h*arg(V_1)",
                                     "source": sorted(r["stem"] for r in rows)}, ensure_ascii=False)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw_*.csv")
    ap.add_argument("--out", default=DEFAULT_VTAIL_NPZ)
    ap.add_argument("--dry-run", action="store_true", help="표만 보고 파일을 안 쓴다")
    a = ap.parse_args()
    d = build(a.raw)
    if a.dry_run:
        print("\n--dry-run: 안 썼다")
        return 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    print("\n%s 에 썼다 — 키 %d개 (%s)" % (a.out, len(d["keys"]), ", ".join(map(str, d["keys"]))))
    print("⚠ 이제부터 캐시를 새로 구워야 한다. 옛 캐시·체크포인트는 이 꼬리를 모른다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
