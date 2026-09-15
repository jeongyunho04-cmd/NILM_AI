# -*- coding: utf-8 -*-
"""차수별 `Z_h` 표의 **굽기 경로** 관문 (14.62).

왜 또 만드나 — `run_gate_zharm` 이 있는데
--------------------------------------
`run_gate_zharm` 은 Z 를 쓰는 **네 소비처의 배수가 같은가**를 잰다. 그건 맞았다.
그런데 2026-09-14 에 굽기를 걸려다 보니 **`traincache._init` 에만
`grid_sim.harmonic_z_table` 대입이 없었다** — `holdout._w_init` 과
`genopts.build_synthesizer` 에는 있었다.

    vtexture 기본 표 (_DEFAULT_HZ)  : 걸림      <- 텍스처를 **새 Z 로 벗긴다**
    grid_sim.harmonic_z_table       : 안 걸림   <- 단자 전압은 **옛 Z 로 다시 입힌다**

벗긴 Z 와 입힌 Z 가 다르면 **서로 안 지워진다.** 자리 D h3 은 `Z_h/Z_1` 이 6.71 대 1.02 라
`(Z_새 − Z_옛)·I_3` 이 통째로 창에 남는다. 30만 창을 그렇게 굽고 학습까지 돌린 뒤에야 안다.

⚠⚠ **그리고 굽기 전 관문이 통과했다.** `cache_v32s_par.sbatch` 가 부르는
`run_gate_vtexseg` 는 `genopts` 로 **따로 지은** 생성기를 잰다 — 굽기가 쓰는 물건이 아니다.
[[verify-the-gate-runs-that-path]] 가 말하는 바로 그 모양이다: *함수를 재지 말고 배선을 재라*,
그리고 **재는 배선이 진짜 타는 배선이어야 한다.**

    python -X utf8 src/run_gate_hzwire.py

```
① 세 입구를 **진짜 생성 함수로** 지어 양쪽 표를 본다 (traincache · holdout · genopts)
② 끄면 **비트 동일** — 표가 없으면 옛 경로와 창이 한 바이트도 안 다르다
③ 켜면 자료가 달라진다 (안 달라지면 손잡이가 아무것도 안 한다)
④ ⚠ **반쪽 배선을 구별하는가** — 일부러 반쪽으로 지어 온전한 것과 견준다.
   이 관문의 존재 이유다. 반쪽이 온전과 같아 보이면 ①은 아무것도 안 지키는 것이다
⑤ 앞으로 생길 **네 번째 입구**를 막는다 — `set_default_harmonic_z` 를 부르는 파일은
   `harmonic_z_table` 도 반드시 대입한다 (정적 검사)
```
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

OK, NG = "✅", "❌"
FAIL = []
NPZ, WCY = "processed_data/npz", 3600


def ck(name, ok, note=""):
    print("  " + (OK if ok else NG) + " " + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _sides(gen):
    """(텍스처 쪽이 걸렸나, 시뮬레이터 쪽이 걸렸나)."""
    from src.synthesis import vtexture as vt
    syn = getattr(gen, "synthesizer", gen)
    return (getattr(vt, "_DEFAULT_HZ", None) is not None,
            getattr(syn.grid_sim, "harmonic_z_table", None) is not None)


def _build(entry, hz):
    """세 입구를 **그 입구의 진짜 함수로** 짓는다. 대역 생성기를 쓰지 않는다."""
    if entry == "traincache":
        from src.model import traincache as tc
        # ⚠ **이름으로 넘긴다.** 2026-09-15 에 `_init` 서명에 `vtex_coarse_s` 가
        #   끼어들면서 위치로 넘기던 `bool(hz)` 가 `vtex_seg_s` 자리로 들어갔다 —
        #   `harmonic_z` 는 기본값(끔)이 되어 관문 ①이 **굽기 판에서** 실패했다.
        #   `initargs` 만 위치여야 하고 (Pool 이 그렇게 준다), 나머지는 전부 이름이다.
        tc._init(NPZ, WCY, "train", 0, sp_curves=True, sp_per_texture=True,
                 vtail=True, vtex_step_s=10.0, vtex_seg_s=10.0, harmonic_z=bool(hz))
        return tc._GEN
    if entry == "genopts":
        from src.synthesis.genopts import V32S, build_synthesizer
        return build_synthesizer(dict(V32S, harmonic_z=bool(hz)), NPZ, "train")
    raise SystemExit(entry)


def _win(g, k=3):
    """`_chunk` **그대로** 창 k개를 만들어 모델이 보는 `(fine, wide)` 를 낸다.

    원시 채널이 아니라 `build_inputs` 를 지난 것을 재는 까닭: 세밀 갈래는 창의 **뒤 600
    사이클**만 보고 광역은 2Hz 로 줄인다. 배선 차이가 거기까지 살아남는지가 우리가 알고
    싶은 것이다 ([[verify-the-input-path-not-just-the-model]])."""
    from src.model.traincache import RAW_CHANNELS, build_inputs
    from src.synthesis.dataset import chunk_seed
    xs = np.empty((k, RAW_CHANNELS, g.window_size), np.float32)
    for i in range(k):
        np.random.seed(chunk_seed(0, i))
        smp, _ = g._synthesize_window()
        xs[i] = g._format_inputs(smp)
    f, w = build_inputs(xs)
    return np.asarray(f, np.float64), np.asarray(w, np.float64)


def main() -> int:
    print("차수별 `Z_h` — **굽기 경로** 배선 관문 (14.62)\n")
    from src.synthesis import vtexture as vt

    # ── ① 세 입구를 진짜 함수로 짓는다 ──────────────────────────────────
    print("① 세 입구 · 표 켬 — 양쪽 다 걸려야 한다")
    gens = {}
    for e in ("traincache", "genopts"):
        vt.set_default_harmonic_z(None)
        g = _build(e, True)
        gens[e] = g
        a, b = _sides(g)
        ck("① %-10s 텍스처 de-embed 쪽" % e, a)
        ck("① %-10s 시뮬레이터 re-embed 쪽" % e, b,
           "" if b else "**여기가 2026-09-14 에 비어 있었다**")
    # ── holdout: **글자 검사로는 부족했다** (2026-09-14, 984064) ──────────
    #   두 줄이 다 있는데도 `NameError: harmonic_z` 로 죽었다 — `_build_generator(o)` 가
    #   `o['harmonic_z']` 를 안 꺼냈기 때문이다 (14.51 에서 함수로 뽑을 때 안 따라왔다).
    #   ⚠ 아래 키 검사는 **그 버그를 못 잡는다** — 버그는 *없는 키를 읽은 것*이 아니라
    #     **이름을 아예 안 묶은 것**(NameError)이라 `need - have` 가 빈다. 처음에 이걸로
    #     닫았다고 적었다가 되짚어보고 거뒀다. 키 검사는 *다른* 부류(옵션을 opts 에 안 넣고
    #     읽는 꼴)를 막으니 남겨 둔다.
    #   ⇒ **진짜 닫는 것은 `run_gate_fmt.py` 의 '안 묶인 이름' 검사다** (14.67).
    #     거기서 고침 전 `holdout.py` 를 넣으면 `_build_generator -> harmonic_z` 가 잡힌다.
    import ast as _a
    ho = Path("src/evaluation/holdout.py").read_text(encoding="utf-8")
    _t = _a.parse(ho)
    _fn = next(n for n in _a.walk(_t)
               if isinstance(n, _a.FunctionDef) and n.name == "_build_generator")
    need = {n.slice.value for n in _a.walk(_fn)
            if isinstance(n, _a.Subscript) and isinstance(n.value, _a.Name)
            and n.value.id == "o" and isinstance(n.slice, _a.Constant)
            and isinstance(n.slice.value, str)}
    have = {n.elts[0].value for n in _a.walk(_t)
            if isinstance(n, _a.Tuple) and len(n.elts) == 2
            and isinstance(n.elts[0], _a.Constant) and isinstance(n.elts[0].value, str)}
    miss = sorted(need - have)
    ck("① holdout    `_build_generator` 가 읽는 키가 opts 에 다 있다", not miss,
       ("**빠진 키: %s**" % miss) if miss else "키 %d개 전부" % len(need))
    ck("① holdout    두 줄이 다 있다",
       "set_default_harmonic_z(_hz)" in ho and "grid_sim.harmonic_z_table = _hz" in ho)
    ck("① holdout    `harmonic_z` 를 `o` 에서 꺼낸다", "harmonic_z" in need)

    # ── ②③④ 자료로 본다 ───────────────────────────────────────────────
    #  ⚠ **순서가 중요하다.** 텍스처 라이브러리는 모듈 전역 캐시라, 배선마다 `_build` 를
    #    다시 부른 **직후에** 재야 한다. 미리 지어 두면 나중 `_build` 가 전역을 갈아
    #    치워 내가 재려던 것과 다른 조합이 된다 (이 관문이 잡으려는 바로 그 병이다).
    print()
    print("②③④ 같은 시드 창 3개를 네 배선으로 합성해 견준다")
    off = _win(_build("traincache", False))
    off2 = _win(_build("traincache", False))
    ck("② 끄면 **되풀이해도 비트 동일** (이 자 자체의 결정성)",
       all(np.array_equal(x, y) for x, y in zip(off, off2)))

    full = _win(_build("traincache", True))              # 온전 — 양쪽 다 새 Z
    _g = _build("traincache", True)                      # 반쪽 — 텍스처만 새 Z
    _g.synthesizer.grid_sim.harmonic_z_table = None      #   (2026-09-14 굽기 경로 그대로)
    half = _win(_g)

    def rms(x, y):
        return float(np.sqrt(np.mean((x - y) ** 2)))

    for bi, nm in ((0, "세밀"), (1, "광역")):
        d_of, d_hf = rms(off[bi], full[bi]), rms(half[bi], full[bi])
        ck("③ %s — 켜면 자료가 달라진다 (끔 대 온전)" % nm, d_of > 0,
           "RMS 차 %.4g" % d_of)
        ck("④ ⚠ %s — **반쪽 배선이 온전과 다르다**" % nm, d_hf > 0,
           "반쪽↔온전 %.4g · 끔↔온전 %.4g · **반쪽 오차 = 손잡이 효과의 %.0f%%**"
           % (d_hf, d_of, 100.0 * d_hf / max(d_of, 1e-30)))

    # ── ⑤ 네 번째 입구를 막는다 ────────────────────────────────────────
    print("\n⑤ 앞으로 생길 입구 — `set_default_harmonic_z` 를 부르면 표도 대입해야 한다")
    bad = []
    for p in sorted(Path("src").rglob("*.py")):
        if p.name.startswith("run_gate_") or p.parts[-1] == "vtexture.py":
            continue
        t = p.read_text(encoding="utf-8")
        if "set_default_harmonic_z(" in t and "harmonic_z_table" not in t:
            bad.append(str(p).replace("\\", "/"))
    ck("⑤ 한쪽만 거는 파일이 없다", not bad,
       "반쪽인 파일: " + " ".join(bad) if bad else "검사한 입구 3곳 전부 양쪽")

    # ── ⑥ `_init` 을 **위치로** 부르는 자리를 막는다 (2026-09-15) ──────────
    #   오늘 `_init` 서명에 `vtex_coarse_s` 가 끼어들자, 위치 16개를 넘기던 이 관문
    #   자신의 호출에서 `bool(hz)` 가 `vtex_seg_s` 자리로 밀려 **굽기 판에서** ①이
    #   실패했다. `initargs` 만 위치여야 한다 — 그건 `Pool` 이 그렇게 주기 때문이고,
    #   거기는 개수 대조가 따로 있다. 나머지 호출은 전부 이름이어야 한다.
    #   앞의 넷(npz_dir·window_cycles·time_split·seed)까지는 관용으로 허용한다.
    import ast as _ast
    POS_OK = 4
    loose = []
    for p in sorted(Path("src").rglob("*.py")):
        try:
            tree = _ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for nd in _ast.walk(tree):
            if (isinstance(nd, _ast.Call) and isinstance(nd.func, _ast.Attribute)
                    and nd.func.attr == "_init" and len(nd.args) > POS_OK):
                loose.append("%s:%d (위치 %d개)"
                             % (str(p).replace("\\", "/"), nd.lineno, len(nd.args)))
    ck("⑥ `_init` 을 위치 %d개 넘게 부르는 자리가 없다" % POS_OK, not loose,
       " / ".join(loose) if loose else "검사한 호출 전부 이름 인자")

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 " + OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
