# -*- coding: utf-8 -*-
"""관문 — **두 학습기가 같은 순방향 모형을 쓰나** (14.171). 다섯 줄.

`src/model/lossbuild.py` 머리말이 *"두 학습기가 같은 순방향 모형을 쓰게 하려고 함수로
뺀다 — 한쪽만 바뀌면 두 판을 못 견준다"* 라고 약속한다. **그 약속이 깨져 있었다.**

```
  1단계 `run_train_cnn`  -> NILMLoss(...)   인자 34개
  2단계 `lossbuild.build_loss` -> NILMLoss(...)   인자 **19개**
  => 18개가 어긋났고 그중 둘은 **지금 조리법이 켜는 것**이었다
       harm_sig_vnorm   argparse 기본 **True**  대  NILMLoss 기본 **False**
       cons_deadzone    1단계 100W            대  2단계 안 넘김(= 옛 절대 와트 식)
  `run_train_seq` 가 그것으로 `loss.backward()` 를 한다 — **1단계가 배운 물리와
  다른 물리로 미세조정**하고 있었다.
```

이 관문은 **값이 아니라 인자 면(surface)** 을 본다. 값은 부르는 쪽 몫이고, 면이 갈리면
부르는 쪽이 **표현할 방법 자체가 없다.**

```
  [1] 1단계가 넘기는 인자가 `build_loss` 의 호출에 **전부 있다**
  [2] `build_loss` 의 **서명**에도 그 이름이 다 있다 (호출만 있고 못 받으면 소용없다)
  [3] 거꾸로 build_loss 만 넘기는 것은 **허용 목록**에 이유와 함께 있어야 한다
  [4] 허용 목록이 **낡지 않았다** (이미 해소된 항목이 남아 있으면 실패)
  [5] `LossWeights` 의 필드도 같은 검사를 한다
```

    python -X utf8 -m src.run_gate_lossparity
"""
import ast
import io
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGE1 = "src/run_train_cnn.py"
SHARED = "src/model/lossbuild.py"

#: `build_loss` 만 넘기는 인자 -> **왜 1단계에 없어도 되나**.
#: ⚠ 여기에 이름을 넣는 것은 "두 학습기가 다른 물리를 쓴다" 를 **명시적으로 허락**하는
#: 일이다. 이유를 못 적겠으면 넣지 말고 1단계에 뚫어라.
ALLOW_SHARED_ONLY = {
    "drift_proj": "14.x 표류 사영 — 1단계는 `--drift-proj` 를 안 받는다 (2단계 전용 축)",
    "power_gain": "13.84.38 전력대별 보정비 — 1단계는 `--pow-sig` 가 없다",
    "power_edges": "위와 한 쌍",
}
#: 1단계만 넘겨도 되는 인자 (지금은 없어야 한다). 넣으려면 이유를 적어라.
ALLOW_STAGE1_ONLY: dict = {}

#: `build_loss` 가 **풀에서 직접 계산**하는 것 — 인자로 받을 이유가 없다.
#: 이것이 이 함수의 존재 이유다 (지문·척도를 한 곳에서 만든다).
DERIVED = {
    "s_i": "S_I 표에서",
    "s_state": "build_state_scales",
    "signatures": "harmonic_signatures(pool)",
    "signatures_state": "harmonic_signatures_by_state(pool)",
    "standby_sig": "standby_signatures(pool) + 동작중휴지",
    "noise_sig": "noise_signature(pool) + 상시배경",
    "harm_scale": "harmonic_scales(pool)",
    "smps_group": "apps 순서에서",
    "even_coherent": "PHASE_COHERENT_EVEN 표에서",
}


def call_kwargs(path: str, func: str):
    t = ast.parse(io.open(path, encoding="utf-8").read())
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == func:
            return {k.arg for k in n.keywords if k.arg}
    return None


def func_params(path: str, name: str):
    t = ast.parse(io.open(path, encoding="utf-8").read())
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            a = n.args
            return ({x.arg for x in a.args} | {x.arg for x in a.kwonlyargs}
                    | {x.arg for x in a.posonlyargs})
    return None


def show(tag, names, why=None):
    if not names:
        print("      (없음)")
        return
    for x in sorted(names):
        print("      %-26s %s" % (x, (why or {}).get(x, "")))


def main() -> int:
    ok = True
    A = call_kwargs(STAGE1, "NILMLoss")
    B = call_kwargs(SHARED, "NILMLoss")
    P = func_params(SHARED, "build_loss")
    if A is None or B is None or P is None:
        print("✖ 호출/서명을 못 찾았다 — 파일 구조가 바뀌었나 (A=%s B=%s P=%s)"
              % (A is not None, B is not None, P is not None))
        return 1
    print("1단계 `NILMLoss(...)` 인자 %d개 · `build_loss` 의 호출 %d개 · 서명 %d개"
          % (len(A), len(B), len(P)))

    # [1] 1단계 것이 전부 있나
    gap = A - B - set(ALLOW_STAGE1_ONLY)
    print("\n[1] 1단계가 넘기는 인자가 `build_loss` 호출에 전부 있나  %s"
          % ("OK" if not gap else "**FAIL — %d개가 없다**" % len(gap)))
    show("gap", gap)
    ok &= not gap

    # [2] 서명에도 있나 — 호출만 있고 못 받으면 소용없다.
    #     단 `DERIVED` 는 풀에서 **직접 계산**하므로 인자일 필요가 없다.
    miss = (A & B) - P - set(DERIVED)
    print("[2] 흘려보내는 이름을 `build_loss` **서명**이 받나  %s"
          % ("OK (계산하는 %d개 제외)" % len(DERIVED) if not miss
             else "**FAIL — %d개를 못 받는다**" % len(miss)))
    show("miss", miss)
    ok &= not miss

    # [3] 거꾸로 build_loss 만 넘기는 것은 허용 목록에 있나
    extra = B - A
    bad = extra - set(ALLOW_SHARED_ONLY)
    print("[3] build_loss 만 넘기는 %d개가 전부 허용 목록에 있나  %s"
          % (len(extra), "OK" if not bad else "**FAIL — 이유가 안 적힌 것 %d개**" % len(bad)))
    show("extra", extra, ALLOW_SHARED_ONLY)
    ok &= not bad

    # [4] 허용 목록이 낡지 않았나
    stale = ((set(ALLOW_SHARED_ONLY) - extra) | (set(ALLOW_STAGE1_ONLY) - (A - B))
             | (set(DERIVED) & P))          # 계산한다더니 인자로도 받으면 목록이 낡은 것
    print("[4] 허용 목록에 **이미 해소된 항목**이 남아 있나  %s"
          % ("OK (없다)" if not stale else "**FAIL — 지워라: %s**" % sorted(stale)))
    ok &= not stale

    # [5] LossWeights 필드
    WA = call_kwargs(STAGE1, "LossWeights") or set()
    WB = func_params("src/model/losses.py", "LossWeights")
    if WB is None:
        import dataclasses
        from src.model.losses import LossWeights as _LW
        WB = {f.name for f in dataclasses.fields(_LW)}
    wgap = WA - WB
    print("[5] 1단계가 쓰는 `LossWeights` 필드가 전부 정의돼 있나  %s%s"
          % ("OK" if not wgap else "**FAIL**", "" if not wgap else "  " + str(sorted(wgap))))
    print("      1단계가 쓰는 것: " + " · ".join(sorted(WA)))
    ok &= not wgap

    print("\n%s" % ("전부 통과" if ok else
                    "**실패 — 두 학습기가 다른 순방향 모형을 쓴다.**\n"
                    "  고치는 법: 빠진 인자를 `build_loss` 의 서명과 호출에 뚫어라.\n"
                    "  기본값은 **지금 동작과 비트 동일**하게 두면 옛 판과 계속 비교된다."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
