# -*- coding: utf-8 -*-
"""argparse `help=` 문자열의 **`%` 서식**을 정적으로 검사한다.

⚠⚠ 2026-09-14 **하루에 두 번** 당했다. argparse 는 `help` 를 `help % params` 로 펼치므로
**홑 `%` 하나가 `--help` 를 통째로 죽인다.** 그리고 그것이 HPC 관문
(`$PY --help | grep -q -- "--플래그"`)을 실패시켜 **작업 배열 전체가 4초 만에 FAILED** 된다
(983493). 한글 뒤의 `%` 는 다음 글자가 멀티바이트라 `unsupported format character` 로 터진다.

    "오븐은 5.5% 다"   -> ❌   "오븐은 5.5%% 다"  -> ✅
    "폭 1.5~2.4%%"     -> ✅   "%(default)s"     -> ✅ (argparse 가 채운다)

**실행 없이** 소스만 읽어 검사하므로 330개 모듈을 다 띄우지 않아도 된다.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: argparse 가 실제로 채우는 이름들. 이것만 `%(name)s` 로 살려 둔다.
PARAMS = {"prog": "", "default": "", "type": "", "choices": "", "metavar": "",
          "const": "", "dest": "", "required": "", "nargs": ""}


def check(path: Path):
    bad = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as e:                                     # noqa: BLE001
        return [(0, "(파싱 실패)", f"{type(e).__name__}: {e}")]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "add_argument":
            continue
        opt = next((a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)), "?")
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                txt = ast.literal_eval(kw.value)
            except Exception:                                  # noqa: BLE001
                continue                                       # 변수·f-string 은 못 본다
            if not isinstance(txt, str):
                continue
            try:
                txt % PARAMS
            except Exception as e:                             # noqa: BLE001
                bad.append((getattr(kw.value, "lineno", node.lineno), opt,
                            f"{type(e).__name__}: {e}"))
    return bad


def main() -> int:
    roots = [Path(p) for p in (sys.argv[1:] or ["src"])]
    files = sorted({f for r in roots for f in
                    ([r] if r.is_file() else r.rglob("*.py"))})
    bad_total = 0
    for f in files:
        for line, opt, msg in check(f):
            print(f"❌ {f}:{line}  {opt}  {msg}")
            bad_total += 1
    print(f"\n검사 {len(files)}파일 · 문제 {bad_total}건"
          + ("  ✅" if not bad_total else "  ⚠ `%` 를 `%%` 로 바꿔라"))
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
