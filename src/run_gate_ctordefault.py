# -*- coding: utf-8 -*-
"""관문 — **생성자 기본값이 트레이너와 다른 자리** (14.174). 네 줄.

14.138 이 이 함정에 물렸다: *"관문 넷이 못 잡은 까닭은 `NILMNet` 의 `prior_kappa` 기본이
**0.0** 이라(트레이너는 **8.0**) 관문이 지은 물건에서 그 블록이 아예 안 돌았기 때문이다"*.
그때 **한 곳만 고쳤다.** 오늘 재 보니 다섯이 그대로 눈이 멀어 있었다 —
`net.py` 의 프라이어 블록에 `raise` 를 심어도 통과한다:

```
  잡는다   run_gate_pstatecap · run_gate_states        (체크포인트에서 kappa 를 받는다)
  눈멂     run_gate_finerf · run_gate_segpool · run_gate_timesplit ·
           run_gate_wextra · run_gate_wsegpool          (NILMNet 기본 0.0 을 그대로 쓴다)
```

이 관문은 **값이 아니라 구조**를 본다 — "트레이너 기본값과 생성자 기본값이 다른데, 그
차이가 `if 값 > 0:` 같은 **분기를 여닫는** 인자인가". 그런 인자는 안 넘기면 그 코드가
**실행조차 안 된다** ([[the-gate-must-build-the-real-object]]).

```
  [1] argparse 기본값 ≠ 생성자 기본값 인 인자를 전부 찾는다
  [2] 그중 **분기를 여닫는 것**(`if self.X > 0` / `if self.X:`)을 가른다  <- 위험한 부류
  [3] 그 인자가 **체크포인트에 적히나** (안 적히면 채점도 못 되살린다)
  [4] `NILMNet`/`NILMLoss` 를 짓는 도구 중 **그 인자를 안 넘기는 것**을 센다
```

    python -X utf8 -m src.run_gate_ctordefault
"""
import ast
import glob
import inspect
import io
import re
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

TRAINER = "src/run_train_cnn.py"

#: 기본값이 달라도 **위험하지 않은** 인자 -> 이유. 분기를 안 여닫거나 학습에만 쓰인다.
ALLOW = {
    "swap_tiebreak": "`--w-swap` 이 0 이라 분기가 안 열린다. 14.174 에서 체크포인트에 적었다",
    "harm_sig_vnorm": "⚠ **빚** — 기본값만 뒤집으면 짝인 `vnorm_exp` 가 틀린다 "
                      "(전 기기 −1, 트레이너는 [0,0,−1,0,−1,−1,0,0,−1]). 고치는 길은 "
                      "`lossbuild.stage1_physics(ck, apps)` 로 **둘을 같이** 가져오는 것이다 "
                      "(14.171). 지금 13개 도구가 이 블록을 한 줄도 안 돌린다.",
}


def argparse_defaults(path):
    t = ast.parse(io.open(path, encoding="utf-8").read())
    out = {}
    for n in ast.walk(t):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            continue
        nm = None
        d = None
        act = None
        for a in n.args:
            if isinstance(a, ast.Constant) and str(a.value).startswith("--"):
                nm = a.value
        for k in n.keywords:
            if k.arg == "default":
                try:
                    d = ast.literal_eval(k.value)
                except Exception:
                    d = "<식>"
            if k.arg == "action":
                act = getattr(k.value, "value", None)
        if nm:
            v = False if act == "store_true" else (True if act == "store_false" else d)
            out[nm[2:].replace("-", "_")] = (nm, v)
    return out


def saved_keys(path):
    t = ast.parse(io.open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.add(k.value)
    return out


def main() -> int:
    from src.model.losses import NILMLoss
    from src.model.net import NILMNet
    ap = argparse_defaults(TRAINER)
    sv = saved_keys(TRAINER)
    ctor = {}
    for C in (NILMNet, NILMLoss):
        for p in inspect.signature(C.__init__).parameters.values():
            if p.name == "self" or p.default is inspect._empty:
                continue
            ctor.setdefault(p.name, (C, p.default))

    src = io.open("src/model/net.py", encoding="utf-8").read() + \
        io.open("src/model/losses.py", encoding="utf-8").read()

    rows = []
    for name, (flag, ad) in sorted(ap.items()):
        if name not in ctor:
            continue
        C, cd = ctor[name]
        if not (isinstance(ad, (bool, int, float, str))
                and isinstance(cd, (bool, int, float, str))):
            continue
        if ad == cd:
            continue
        gate = bool(re.search(r"if self\.%s\b" % re.escape(name), src))
        rows.append((flag, name, ad, cd, C.__name__, name in sv, gate))

    print("[1] argparse 기본값 ≠ 생성자 기본값 — **%d개**" % len(rows))
    print("  %-22s %10s %10s %-9s %-7s %s"
          % ("손잡이", "트레이너", "생성자", "어디", "ckpt", "분기를 여나"))
    bad = []
    for flag, nm, ad, cd, cn, s_, g in rows:
        print("  %-22s %10s %10s %-9s %-7s %s"
              % (flag, ad, cd, cn, "적힘" if s_ else "**안 적힘**",
                 "**연다**" if g else "아니오"))
        if g and nm not in ALLOW:
            bad.append((nm, cn))
        if (not s_) and nm not in ALLOW:
            bad.append((nm, cn))

    print("\n[2] 그중 **분기를 여닫는** 인자 = 안 넘기면 그 코드가 **실행조차 안 된다**")
    gates = [r for r in rows if r[6]]
    for flag, nm, ad, cd, cn, s_, g in gates:
        print("    %-22s `if self.%s` 가 트레이너에서는 참, 기본값으로는 **거짓**" % (flag, nm))
    if not gates:
        print("    (없음)")

    print("\n[4] 그 인자를 **안 넘기고** 모델을 짓는 도구")
    blind_any = False
    for flag, nm, ad, cd, cn, s_, g in gates:
        blind = []
        for f in sorted(glob.glob("src/*.py")):
            t = io.open(f, encoding="utf-8", errors="replace").read()
            if ("%s(" % cn) not in t:
                continue
            if nm in t:
                continue
            blind.append(f.split("/")[-1])
        if blind:
            blind_any = True
            print("    %s (%s) — **%d개**: %s" % (nm, cn, len(blind), " · ".join(blind)))
    if not blind_any:
        print("    (없음)")

    print("\n⚠ 이 목록은 **후보**다. 진짜로 눈이 멀었는지는 그 블록에 `raise` 를 심어 보면 안다")
    print("   (오늘 그렇게 재서 5/7 이 눈멀었음을 확인했다 — 14.174).")
    ok = not [x for x in bad if x[0] not in ALLOW]
    print("\n%s" % ("전부 통과" if ok else
                    "**실패 — 위 인자를 트레이너 기본값과 맞추거나 체크포인트에 적어라.**\n"
                    "  분기를 여는 인자는 특히 위험하다 — 관문이 그 코드를 **한 줄도 안 돌린다**."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
