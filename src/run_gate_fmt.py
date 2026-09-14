# -*- coding: utf-8 -*-
"""`%` 형식 문자열의 **자리 수와 값 개수**가 맞는지 정적으로 본다 (14.66).

왜 만드나
---------
2026-09-14 에 학습 세 판(`983771`)이 **1분 만에** 죽었다. 내가 넣은 줄이 이랬다:

```python
print("  ** ... %.2f할 · 기준 " + a.harm_vhrel_src + " · 기기 %s **"
      % (a.harm_vhrel_frac, " ".join(...)))
```

`%` 가 `+` 보다 **강하게 묶인다.** 그래서 실제로는 `" · 기기 %s **" % (값1, 값2)` 가 되어
자리 하나에 값 둘이 간다 -> `TypeError: not all arguments converted during string formatting`.
`run_gate_help` 는 **argparse help 만** 보고 런타임 `print` 는 안 본다. 그래서 통과했다.

무엇을 잡나
-----------
`"...literal..." % (a, b)` 꼴에서 **왼쪽 문자열의 자리 수**와 **오른쪽 튜플의 길이**가
다르면 그것은 실행하면 `TypeError` 다. 문자열이 **상수**이고 오른쪽이 **튜플 리터럴**일
때만 본다 (그 밖은 정적으로 못 센다 — 오탐을 안 내는 쪽으로 판단한다).

세는 법: `%%` 는 건너뛰고, 이름 있는 형태(`%(k)s`)와 별표 너비(`%*d`)가 있으면 **그 문자열은
통째로 건너뛴다** (정적으로 개수를 못 정한다).

    python -X utf8 src/run_gate_fmt.py
"""
from pathlib import Path
import ast
import builtins
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, NG = "OK", "NG"
FAIL = []

#: `%` 변환 하나. `%%` 는 따로 걸러낸다.
_SPEC = re.compile(r"%[-+ #0]*[0-9*]*[.]?[0-9*]*[hlL]?[diouxXeEfFgGcrsa]")
_NAMED = re.compile(r"%[(]")


def _count(s):
    """형식 자리 수. 정적으로 못 세면 `None`."""
    if _NAMED.search(s) or "*" in s:
        return None
    return len(_SPEC.findall(s.replace("%%", "")))


def bad_formats(src):
    """[(줄, 자리수, 값개수)] — 실행하면 TypeError 가 날 `%` 형식."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)):
            continue
        L, R = n.left, n.right
        if not (isinstance(L, ast.Constant) and isinstance(L.value, str)):
            continue
        if not isinstance(R, ast.Tuple):
            continue                      # 변수 하나면 정적으로 못 센다
        if any(isinstance(e, ast.Starred) for e in R.elts):
            continue
        k = _count(L.value)
        if k is None or k == len(R.elts):
            continue
        out.append((n.lineno, k, len(R.elts)))
    return out


# ── 14.67: 함수 안에서 Load 되는데 **어디에도 안 묶인 이름** ────────────────────
#   2026-09-14 에 홀드아웃(984064)이 `NameError: harmonic_z` 로 죽었다. 14.51 에서
#   `_build_generator` 를 함수로 뽑을 때 `harmonic_z = o['harmonic_z']` 가 안 따라왔다.
#   `run_gate_hzwire` (1) 은 그 파일에 배선 두 줄이 **있는지**만 보는 글자 검사라 못 잡았고,
#   "`o[...]` 키가 opts 에 다 있나" 로도 **못 잡는다** — 버그는 없는 키를 읽은 게 아니라
#   **이름을 아예 안 묶은** 것이기 때문이다. 방향을 바로잡아 이렇게 잡는다.
#   ⚠ 한쪽으로만 틀리게 짠다: `bound` 를 **넉넉히** 모아 오탐을 안 낸다 (놓칠지언정).
def _bound(fn):
    """`fn` 안에서 묶이는 이름 ∪ 인자. Load 만 되는 이름은 빼고 돌려준다."""
    b, loads = set(), set()

    def args_of(a):
        out = [x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
        if a.vararg:
            out.append(a.vararg.arg)
        if a.kwarg:
            out.append(a.kwarg.arg)
        return out

    b.update(args_of(fn.args))
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            (b if isinstance(n.ctx, (ast.Store, ast.Del)) else loads).add(n.id)
        elif isinstance(n, ast.Lambda):
            b.update(args_of(n.args))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n is not fn:
                b.add(n.name)
                if getattr(n, "args", None):
                    b.update(args_of(n.args))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                b.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            b.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            b.update(n.names)
    return loads - b


def _module_names(tree):
    """모듈 수준에서 묶이는 이름 ∪ 빌트인. 함수/클래스 **안**은 안 본다."""
    g = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__"}
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            g.add(st.name)
            continue
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                g.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    g.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                g.add(n.name)
    return g


def free_names(src):
    """[(줄, 함수, 이름)] — 실행하면 `NameError` 가 날 이름."""
    t = ast.parse(src)
    g = _module_names(t)
    out = []

    def visit(node):
        # ⚠ **중첩 함수로는 내려가지 않는다.** 중첩 함수는 바깥 함수의 지역을 닫아 읽으므로
        #   따로 재면 바깥 지역이 전부 '안 묶인 이름' 으로 보인다 (`losses.py` 의 `_k` 가
        #   `__init__` 의 `_VC` 를 읽는 꼴). 바깥 함수를 잴 때 `_bound` 가 하위 트리를 통째로
        #   훑으므로 중첩 함수의 Load 도 거기서 같이 검사된다.
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for nm in sorted(_bound(ch) - g):
                    out.append((ch.lineno, ch.name, nm))
            else:
                visit(ch)

    visit(t)
    return out


# ── 14.70: sbatch 의 `$PY -c "..."` 블록 안 **백틱** ──────────────────────────
#   큰따옴표 안의 백틱은 **셸이 명령으로 실행한다.** 2026-09-14 에 파이썬 주석에 적은
#   마크다운 백틱이 그대로 실행돼 `train60_v32: command not found` 가 찍혔다. 이번엔
#   빈 문자열이 되어 무해했지만, 백틱 안이 진짜 명령이면 **굽기 노드에서 실행된다.**
def backticks_in_py(src):
    """[(줄, 개수)] — `-c "` 로 여는 파이썬 블록 안에 백틱이 있는 자리."""
    out = []
    pat = '-c "' + chr(10) + '(.*?)' + chr(10) + '"' + chr(10)
    for m in re.finditer(pat, src, re.S):
        k = m.group(1).count(chr(96))
        if k:
            out.append((src[:m.start(1)].count(chr(10)) + 1, k))
    return out


def ck(name, ok, note=""):
    print("  [" + (OK if ok else NG) + "] " + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


#: 음성 대조 — **일부러 깨뜨린 본**과 **멀쩡한 본**.
_BROKEN = 'print("a %s" + v + " b %s" % (x, y))'
_GOOD = ('print("a %s b %s" % (x, y))' + chr(10)
         + 'print("%d%% 늘었다 %s" % (n, s))' + chr(10)
         + 'print("%(k)s" % d)' + chr(10)
         + 'print("%*d" % (w, v))' + chr(10)
         + 'print("%s" % v)')


def main() -> int:
    print("정적 관문 — `%` 자리 수 (14.66) · 안 묶인 이름 (14.67) · sbatch 백틱 (14.70)")
    print()
    ck("음성 대조 — 2026-09-14 에 학습을 죽인 그 꼴을 잡는다",
       len(bad_formats(_BROKEN)) == 1, str(bad_formats(_BROKEN)))
    ck("음성 대조 — 멀쩡한 꼴은 안 잡는다 (%%, 이름 있는 형태, 별표 너비 포함)",
       not bad_formats(_GOOD), str(bad_formats(_GOOD)))

    _NBROKEN = ("def f(o):" + chr(10) + "    x = o['x']" + chr(10)
                + "    if flag:" + chr(10) + "        return x" + chr(10))
    _NGOOD = ("import os" + chr(10) + "G = 1" + chr(10)
              + "def f(o, flag=0):" + chr(10)
              + "    g = lambda n: n + G" + chr(10)
              + "    import re as _r" + chr(10)
              + "    try:" + chr(10) + "        pass" + chr(10)
              + "    except OSError as e:" + chr(10) + "        print(e, _r, os)" + chr(10)
              + "    return [y for y in o if flag], g" + chr(10)
              + "def h():" + chr(10) + "    return f" + chr(10))
    ck("음성 대조 — 안 묶인 이름을 잡는다 (984064 의 그 꼴)",
       [x[2] for x in free_names(_NBROKEN)] == ["flag"], str(free_names(_NBROKEN)))
    ck("음성 대조 — 멀쩡한 꼴은 안 잡는다 (lambda·내포·except as·모듈 전역·늦은 def)",
       not free_names(_NGOOD), str(free_names(_NGOOD)))

    bad = []
    skipped = []
    nbad = []
    for f in sorted(Path("src").rglob("*.py")):
        try:
            t = f.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            hits = bad_formats(t)
        except SyntaxError:
            skipped.append(str(f).replace(chr(92), "/"))
            continue
        for ln, k, m in hits:
            bad.append("%s:%d 자리 %d 대 값 %d" % (str(f).replace(chr(92), "/"), ln, k, m))
        try:
            for ln, fn, nm in free_names(t):
                nbad.append("%s:%d %s -> %s" % (str(f).replace(chr(92), "/"), ln, fn, nm))
        except SyntaxError:
            pass
    ck("src 전체 — 안 묶인 이름이 없다 (NameError 예약)", not nbad,
       ("  ".join(nbad[:6]) if nbad else "검사 통과"))
    ck("src 전체 — 자리 수가 안 맞는 `%` 가 없다", not bad,
       ("  ".join(bad[:6]) if bad else "검사 통과"))

    _TBROKEN = '$PY -c "' + chr(10) + 'print(1)  # ' + chr(96) + 'ls' + chr(96) + chr(10) + '"' + chr(10)
    _TGOOD = '$PY -c "' + chr(10) + "print(1)  # 'ls'" + chr(10) + '"' + chr(10)
    ck("음성 대조 — 파이썬 블록 안 백틱을 잡는다", len(backticks_in_py(_TBROKEN)) == 1,
       str(backticks_in_py(_TBROKEN)))
    ck("음성 대조 — 백틱 없는 블록은 안 잡는다", not backticks_in_py(_TGOOD))
    tick = []
    for f in sorted(Path("patches").glob("*.sbatch")):
        for ln, k in backticks_in_py(f.read_text(encoding="utf-8")):
            tick.append("%s:%d 백틱 %d개" % (f.as_posix(), ln, k))
    ck("patches 전체 — `$PY -c \"...\"` 안에 백틱이 없다 (셸이 실행해 버린다)", not tick,
       ("  ".join(tick[:4]) if tick else "sbatch 검사 통과"))
    if skipped:
        print("  (문법이 깨져 건너뛴 파일 %d개: %s)" % (len(skipped), " ".join(skipped)))

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
