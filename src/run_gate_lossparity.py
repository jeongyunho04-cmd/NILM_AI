# -*- coding: utf-8 -*-
"""관문 — **두 학습기가 같은 순방향 모형을 쓰나** (14.171). 여섯 줄.

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
ADAPT = "src/run_adapt.py"

#: `build_loss` 만 넘기는 인자 -> **왜 1단계에 없어도 되나**.
#: ⚠ 여기에 이름을 넣는 것은 "두 학습기가 다른 물리를 쓴다" 를 **명시적으로 허락**하는
#: 일이다. 이유를 못 적겠으면 넣지 말고 1단계에 뚫어라.
ALLOW_SHARED_ONLY = {
    "drift_proj": "14.x 표류 사영 — 1단계는 `--drift-proj` 를 안 받는다 (2단계 전용 축)",
    "hcond_cols": "14.299 — `build_loss` 가 `hcond_classes`(사람이 읽는 이름)를 "
                  "`get_load_class` 로 풀어 **지어내는 열 목록**이다. 두 학습기가 다른 "
                  "물리를 쓰는 게 아니라 **같은 변환을 한 곳에서** 한다 — 양쪽에 복붙하면 "
                  "조용히 갈라진다([[pin-the-two-entry-points-against-each-other]]). "
                  "짝인 `hcond_classes` 는 ALLOW_STAGE1_ONLY 에 있다",
}
#: 1단계만 넘겨도 되는 인자. 넣으려면 이유를 적어라.
ALLOW_STAGE1_ONLY: dict = {
    "hcond_classes": "14.299 — `build_loss` 가 **소비해서** `hcond_cols` 로 바꾸므로 "
                     "`NILMLoss` 까지 안 간다. 위 `hcond_cols` 와 한 쌍이고, 둘을 같이 "
                     "봐야 경로가 끊기지 않은 것이 보인다. ⚠ `run_adapt` 쪽 배선은 "
                     "아직이라 [5] 의 빚 목록에 따로 올라 있다",
}

#: `build_loss` 가 **풀에서 직접 계산**하는 것 — 인자로 받을 이유가 없다.
#: 이것이 이 함수의 존재 이유다 (지문·척도를 한 곳에서 만든다).
#: `run_adapt`(2단계 적응) 만 아는 인자 -> **왜 1단계가 못 켜도 되나**.
#: ⚠ 여기 든 이름은 "그 기구를 1단계에서는 **쓸 수 없다**" 는 뜻이다. 12.156 의
#: `companion_sig` 가 그 함정이었다 — 오븐 팬·조명을 메우려고 만들어 adapt 에만 달았고,
#: 정작 오븐 신원을 배우는 **1단계에는 켤 방법이 없어** 버퍼가 내내 0 이었다 (14.167).
ALLOW_ADAPT_ONLY = {
    "companion_sig": "⚠ **미해결** — 12.156. 1단계에 뚫어야 한다 (14.167 이 짚은 그것)",
    "companion_w": "위와 한 쌍. ⚠ 미해결",
    "power_ref": "12.144 `--w-pref` — 2단계 전용 기준 전력",
    "standby_w": "2단계 전용 대기 전력 목표",
    "reactive_qp": "무효전력 재구성 — 1단계 손실에 Q 항이 없다",
    "noise_q": "위와 한 쌍",
    "sig_real": "13.x 실측 지문 — 2단계가 실측에 맞출 때만",
    "signatures_site": "13.59 자리별 지문 — 2단계 전용",
    "signatures_state_site": "위와 한 쌍",
    "harm_deadzone": "2단계 고조파 죽은구역",
    "harm_max_order": "2단계 차수 상한",
    "harm_weight": "2단계 차수 가중",
}

#: **빚** — 아직 자리마다 갈라져 있지만 지금은 무력한 것. 이유와 **언제 살아나는지**를
#: 적는다. 관문은 이것을 **소리 내어 찍되 통과**시킨다 — 숨기면 또 잊는다.
KNOWN_DEBT = {
    "harm_vnorm_vref": "풀에서 계산하는 텐서라 체크포인트에 없다. `--harm-vnorm-anchor` 를 "
                       "2단계에서 켜려면 먼저 저장하거나 풀에서 다시 계산해야 한다",
    "harm_vnorm_vref_state": "위와 한 쌍",
    "harm_vhrel_rec": "위와 같음. **`--harm-vhrel-frac > 0` 이면 곧 살아난다** (985188)",
    "head_conductance": "14.284 — 1단계 전용. 2단계(`run_adapt`)는 이미 학습된 판을 "
                        "미세조정하므로 체크포인트의 `head_conductance` 를 물려받아야 "
                        "맞는데, 그 배선은 아직이다. **이 팔이 채택되면 즉시 갚아야 한다** "
                        "— 안 갚으면 adapt 이 옛 와트 목표로 되돌려 이중 계산이 부활한다",
    "hcond_scale": "14.299 — `head_conductance` 와 **같은 빚**이다. 셋 다 전도도 목표의 "
                   "모양을 정하므로 그 팔이 채택되면 넷을 **한꺼번에** 갚아야 한다. "
                   "특히 `watt` 로 구운 판을 adapt 이 `log` 기본값으로 미세조정하면 "
                   "목표가 도중에 바뀐다 — 채택 전에 갚는 게 안전하다",
    "hcond_on_w": "위와 한 쌍 (14.299). 기본 5.0 이라 안 적으면 adapt 이 저항 쪽 96창"
                  "(오븐 전이 35 · 핫플 61)을 다시 전도도 갈래로 넣는다. "
                  "⚠ 14.301 정정 — 옛 이유에 적었던 '오븐 s1 17W · 4,626배' 는 `S_STATE`"
                  "(척도표)를 목표로 착각하고 분모도 `s_i` 로 쓴 값이라 둘 다 틀렸다",
    "hcond_classes": "위와 한 쌍 (14.299). 기본 빈 값이라 안 적으면 adapt 이 SMPS·모터에도 "
                     "전도도 목표를 걸어 범주 오류가 되살아난다",
    "harm_vhrel_frac": "위와 한 쌍 — vhr 판이 채택되면 **즉시 빚을 갚아야 한다**",
    "harm_vhrel_on": "위와 한 쌍",
    "swap_slack": "`--w-swap` 이 0 이라 무력. 켜는 순간 살아난다",
    "swap_tiebreak": "위와 같음",
    "swap_tb_orders": "위와 같음",
    "power_gain": "14.172 — 1단계에 `--pow-sig` 를 뚫었다. adapt 은 아직. 풀에서 지으면 된다",
    "power_edges": "위와 한 쌍",
    "power_tau": "위와 한 쌍",
}

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
    "power_gain": "harmonic_signatures_by_power(pool) — 14.172 이후 **양쪽 다** 짓는다",
    "power_edges": "위와 한 쌍",
}


def splat_credit(path: str, func: str):
    """`**stage1_physics(...)` 로 뚫은 자리를 **상수 `STAGE1_PHYSICS_KEYS`** 로 인정한다.

    AST 는 `**dict` 안을 못 본다. 그래서 공용 헬퍼가 낼 수 있는 이름을 상수로 **선언**하고,
    그 splat 이 있는 호출은 그 이름들을 명시 인자처럼 가진 것으로 센다.
    ⚠ 상수와 함수가 어긋나면 이 인정이 거짓이 된다 — [4] 가 그것도 본다.
    """
    t = ast.parse(io.open(path, encoding="utf-8").read())
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == func:
            for k in n.keywords:
                if k.arg is None and isinstance(k.value, ast.Call):
                    nm = getattr(k.value.func, "id", "") or getattr(k.value.func, "attr", "")
                    if "stage1_physics" in nm or nm in ("_s1p", "_phys"):
                        from src.model.lossbuild import STAGE1_PHYSICS_KEYS
                        return set(STAGE1_PHYSICS_KEYS)
                if k.arg is None and isinstance(k.value, ast.Name) and k.value.id == "_phys":
                    from src.model.lossbuild import STAGE1_PHYSICS_KEYS
                    return set(STAGE1_PHYSICS_KEYS)
    return set()


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

    C = (call_kwargs(ADAPT, "NILMLoss") or set()) | splat_credit(ADAPT, "NILMLoss")

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
             | (set(DERIVED) & P) | (set(KNOWN_DEBT) - (A - C)))          # 계산한다더니 인자로도 받으면 목록이 낡은 것
    #: `STAGE1_PHYSICS_KEYS` 가 `stage1_physics` 가 실제로 낼 수 있는 것과 맞나 —
    #  어긋나면 [5] 의 인정이 **거짓**이 된다.
    import inspect as _ins
    from src.model.lossbuild import STAGE1_PHYSICS_KEYS as _SK, stage1_physics as _sf
    _src = _ins.getsource(_sf)
    _leak = [k for k in _SK if ('"%s"' % k) not in _src]
    if _leak:
        stale = set(stale) | set(_leak)
        print("      ⚠ 상수에만 있고 함수가 안 내는 이름: %s" % sorted(_leak))
    print("[4] 허용 목록에 **이미 해소된 항목**이 남아 있나  %s"
          % ("OK (없다)" if not stale else "**FAIL — 지워라: %s**" % sorted(stale)))
    ok &= not stale

    # [5] 2단계 적응(`run_adapt`) 도 같은 면을 갖나 — 세 번째 자리다
    a_only = C - A - B - set(ALLOW_ADAPT_ONLY)
    cant = A - C - set(ALLOW_ADAPT_ONLY) - set(KNOWN_DEBT)
    debt = sorted((A - C) & set(KNOWN_DEBT))
    print("[5] `run_adapt` 인자 %d개 — adapt 전용 %d개 · **adapt 이 못 쓰는 1단계 인자 %d개**"
          % (len(C), len(C - A - B), len(A - C)))
    print("      adapt 전용 중 이유가 안 적힌 것  %s" % ("(없음) OK" if not a_only else "**FAIL**"))
    show("a_only", a_only)
    print("      adapt 이 못 쓰는 1단계 인자  %s"
          % ("(없음) OK" if not cant else "**FAIL — %d개**" % len(cant)))
    show("cant", cant)
    if debt:
        print("      ⚠ **빚 %d개** (지금은 무력하지만 갈라져 있다):" % len(debt))
        for x in debt:
            print("         %-24s %s" % (x, KNOWN_DEBT[x]))
    ok &= (not a_only and not cant)

    # [6] LossWeights 필드
    WA = call_kwargs(STAGE1, "LossWeights") or set()
    WB = func_params("src/model/losses.py", "LossWeights")
    if WB is None:
        import dataclasses
        from src.model.losses import LossWeights as _LW
        WB = {f.name for f in dataclasses.fields(_LW)}
    wgap = WA - WB
    print("[6] 1단계가 쓰는 `LossWeights` 필드가 전부 정의돼 있나  %s%s"
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
