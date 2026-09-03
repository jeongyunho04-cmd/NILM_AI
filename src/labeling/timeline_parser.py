"""사람이 적은 타임라인을 기계가 읽는다 (12.155)

`TEST_DATASET_TIMELINE_ANALYSIS.txt` 는 스위치를 누른 사람이 **로그가 올라오는 것을
보면서** 적은 것이다. 그래서:

* 시각은 `seq` 기준이고 ±3초쯤 어긋난다 (기존 라벨에서 +2.2 ~ +2.7초로 확인됨)
* 오타가 있다 (`포트 zu짐`, `미닢피씨`)
* 빠뜨린 것이 있다 (포트는 끓으면 저 혼자 꺼진다)
* on/off 가 아닌 것이 섞여 있다 — `드라이기 강` 은 **단계 변경**, `미니피씨 작업시작`
  은 부하 변동이지 전원이 아니다

**여기서는 해석하지 않는다.** 읽고, 정규화하고, 못 읽은 줄을 그대로 보고한다.
정체를 정하는 것은 신호의 몫이다 (`run_refine_labels`).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import unicodedata

#: 사람이 쓴 이름 -> 기기. 긴 것부터 본다 ('노트북충전기' 가 '노트북' 보다 먼저).
NAMES: List[Tuple[str, str]] = [
    ("노트북충전기", "laptop_charger"), ("랩탑충전기", "laptop_charger"),
    ("충전기", "laptop_charger"), ("노트북", "laptop_charger"),
    ("빔프로젝터", "beam_projector"), ("빔프", "beam_projector"),
    ("프로젝터", "beam_projector"),
    ("미니피씨", "minipc"), ("미닢피씨", "minipc"), ("미니pc", "minipc"),
    ("미니PC", "minipc"), ("미니컴", "minipc"),
    ("전기포트", "electiric_kettle"), ("포트", "electiric_kettle"),
    ("주전자", "electiric_kettle"),
    ("핫플레이트", "hotplate"), ("핫플", "hotplate"),
    ("드라이기", "hair_dryer"), ("헤어드라이기", "hair_dryer"),
    ("선풍기", "fan"), ("에어컨", "air_conditioner"), ("오븐", "oven"),
]
#: 동작. `강/약/중` 은 **단계**이고 그 자체로 on 을 뜻하지 않는다.
ON_WORDS = ("켜짐", "켜진", "겨짐", "zu짐", "on", "ON", "켜기", "켬")
OFF_WORDS = ("꺼짐", "꺼진", "off", "OFF", "끄기", "끔", "종료", "끝")
PRESENT_WORDS = ("켜져있음", "켜져잇음", "켜져 있음", "이미")
MODE_WORDS = {"강": "high", "약": "low", "중": "mid", "강풍": "high",
              "약풍": "low", "중풍": "mid"}
#: 미니PC 의 '작업' 은 전원이 아니라 부하다 (12.155). on/off 로 치면 안 된다.
WORK_WORDS = ("작업시작", "작업 시작", "작업종료", "작업 종료", "작업끝", "작업 끝")

HEAD = re.compile(r"^(test[._0-9]*)\.csv", re.IGNORECASE)
LINE = re.compile(r"^\s*(\d+)\s*[:：]?\s*(.+?)\s*$")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "")


def parse_line(text: str) -> Optional[Dict]:
    """한 줄을 (기기, 동작, 단계) 로. 못 읽으면 None."""
    t = _norm(text)
    app = None
    for k, v in NAMES:
        if _norm(k) in t:
            app = v
            t_wo = t.replace(_norm(k), "", 1)
            break
    if app is None:
        return None
    mode = None
    for k, v in MODE_WORDS.items():
        if k in t_wo:
            mode = v
            break
    work = any(_norm(w) in t for w in WORK_WORDS)
    present = any(_norm(w) in t for w in PRESENT_WORDS)
    on = any(w in t_wo for w in ON_WORDS)
    off = any(w in t_wo for w in OFF_WORDS)
    if work:
        kind = "work_end" if ("종료" in t or "끝" in t) else "work_start"
    elif present:
        kind = "already_on"
    elif off:
        kind = "off"
    elif on:
        kind = "on"
    elif mode:
        kind = "mode"          # '드라이기 강' — 단계만 적힌 줄
    else:
        return None
    return {"appliance": app, "kind": kind, "mode": mode, "raw": text.strip()}


def parse(path: str = "TEST_DATASET_TIMELINE_ANALYSIS.txt") -> Tuple[Dict[str, List[Dict]], List[str]]:
    """파일 전체 -> {stem: [{seq, appliance, kind, mode, raw}]}, 못 읽은 줄."""
    out: Dict[str, List[Dict]] = {}
    bad: List[str] = []
    cur = None
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        h = HEAD.match(ln.strip())
        if h:
            cur = h.group(1)
            out.setdefault(cur, [])
            continue
        if cur is None or not ln.strip():
            continue
        m = LINE.match(ln)
        if not m:
            continue
        seq, body = int(m.group(1)), m.group(2)
        p = parse_line(body)
        if p is None:
            bad.append(f"{cur}  seq {seq}: {body}")
            continue
        p["seq"] = seq
        out[cur].append(p)
    for k in out:
        out[k].sort(key=lambda x: x["seq"])
        _resolve_modes(out[k])
    return out, bad


def _resolve_modes(ev: List[Dict]) -> None:
    """이미 켜져 있는 기기에 단계가 붙으면 그것은 **단계 변경**이지 켜짐이 아니다.

    `test_17` 이 그 예다 — `50:드라이기 강 켜짐` 다음 `240:드라이기 약 켜짐` 인데
    신호는 그 자리에서 **−444W** 다. 강(1000W)에서 약(556W)으로 내린 것이고,
    '켜짐'을 그대로 믿으면 부호가 반대인 라벨이 된다 (12.155).

    `꺼짐` 에 단계가 붙은 것도 마찬가지로 그 단계를 끈 것이라 전원 off 로 본다.
    """
    on = {}
    for e in ev:
        a = e["appliance"]
        if e["kind"] in ("on", "already_on"):
            if on.get(a) and e["mode"] and e["kind"] == "on":
                e["kind"] = "mode"          # 이미 켜져 있다 -> 단계 변경
            else:
                on[a] = True
        elif e["kind"] == "mode":
            if not on.get(a):
                e["kind"] = "on"            # 꺼져 있는데 단계만 적혔다 -> 켜짐
                on[a] = True
        elif e["kind"] == "off":
            on[a] = False
