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
    print("`%` 형식 자리 수 관문 (14.66)")
    print()
    ck("음성 대조 — 2026-09-14 에 학습을 죽인 그 꼴을 잡는다",
       len(bad_formats(_BROKEN)) == 1, str(bad_formats(_BROKEN)))
    ck("음성 대조 — 멀쩡한 꼴은 안 잡는다 (%%, 이름 있는 형태, 별표 너비 포함)",
       not bad_formats(_GOOD), str(bad_formats(_GOOD)))

    bad = []
    skipped = []
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
    ck("src 전체 — 자리 수가 안 맞는 `%` 가 없다", not bad,
       ("  ".join(bad[:6]) if bad else "검사 통과"))
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
