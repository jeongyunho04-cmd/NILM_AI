"""
실시간 추론 — 수신기 CSV 를 따라가며 분해하고, 예측과 실제를 함께 남긴다
==========================================================================
5.3절의 시연 동선이자, **다음 실측 라벨을 만드는 도구**다.

    python -m src.run_live --csv data/live.csv
    python -m src.run_live --replay data/test3.csv --speed 20     # 보드 없이 검증

[왜 수신기에 넣지 않고 CSV 를 따라가는가]
`nilm_receiver.py` 의 수신 루프는 ACK 타이밍에 묶여 있다. 펌웨어는 ACK 가 2.5초
안에 안 오면 같은 프레임을 다시 보내고, 5번 실패하면 큐가 밀린다(`nilm_link.h`).
그 루프 안에서 GPU 추론을 돌리면 ACK 가 늦어져 **프레임 유실을 만든다** —
측정 자체를 망가뜨리는 위험이다. 별도 프로세스로 CSV 를 따라가면 추론이 아무리
느려도 수신에 영향이 없다.

[예측과 실제를 함께 남기는 이유 — 이 파일의 진짜 목적]
지금 실측 라벨은 신호를 사람이 읽어 쓴 것이고, 12.25 에서 미니PC 구간이 틀린 것이
드러났다. 켜고 끈 시각을 손으로 적는 것도 방법이지만, **모델이 먼저 답하고 사람이
고치는 쪽이 배우는 게 많다**:

  - 사람은 "지금 뭐가 켜져 있나" 를 처음부터 쓰지 않고 **틀린 것만 고친다**
  - 고칠 거리가 생기는 지점이 곧 모델이 헷갈리는 지점이다 (능동 학습)
  - 예측 전체가 자동으로 기록되므로, 사람이 못 본 오답도 나중에 찾을 수 있다

**다만 사람이 알아챈 것만 기록하면 표본이 편향된다.** 모델이 확신하고 틀린
구간은 눈에 안 띈다 (12.15.1 이 그런 실패였다). 그래서 예측 스트림은 **전부**
남기고, 사람의 정정은 그 위에 얹는다.

    실행 중 키:  1~9 해당 기기 토글   0 전부 끔   space 현재 상태 확정
                 u 마지막 정정 취소   q 종료
"""
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.model.inputs import build_inputs, target_index
from src.run_gate_check import load_model

WINDOW_CYCLES = 3600           # 60초 (12.8절에서 확정)
HARMONICS = 15
KOR = {"oven": "오븐", "hotplate": "핫플", "electiric_kettle": "포트",
       "hair_dryer": "드라이기", "minipc": "미니PC", "beam_projector": "프로젝터",
       "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨"}


def csv_columns(header: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(header)}


def row_to_channels(row: List[str], col: Dict[str, int]) -> Optional[np.ndarray]:
    """CSV 한 행 -> 33채널 한 사이클.

    전처리(`feature_extractor` + `numpy_exporter`)와 **같은 식이어야 한다.**
        Re/Im  = ih_rms * (cos, sin)(radians(ih_deg))
        S      = vrms * irms
        Q      = sign(phase) * sqrt(max(0, S^2 - P^2))
    """
    try:
        irms = float(row[col["irms"]])
        p = float(row[col["p_w"]])
        v = float(row[col["vrms"]])
        phase = float(row[col["phase_deg"]])
        mag = np.array([float(row[col[f"ih{k}"]]) for k in range(1, HARMONICS + 1)], np.float32)
        deg = np.array([float(row[col[f"ihdeg{k}"]]) for k in range(1, HARMONICS + 1)], np.float32)
    except (ValueError, IndexError, KeyError):
        return None
    rad = np.radians(deg)
    s = v * irms
    q = np.sign(phase) * np.sqrt(max(0.0, s * s - p * p))
    x = np.empty(33, np.float32)
    x[0:15] = mag * np.cos(rad)
    x[15:30] = mag * np.sin(rad)
    x[30], x[31], x[32] = p, q, v
    return x


def tail_rows(path: Path, replay: bool, speed: float, poll: float = 0.2):
    """CSV 를 행 단위로 흘려보낸다. `replay` 면 파일 끝에서 멈춘다."""
    import csv as _csv
    while not path.exists():
        if replay:
            raise FileNotFoundError(path)
        print(f"  {path} 를 기다리는 중…", flush=True)
        time.sleep(1.0)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        col = csv_columns(header)
        yield ("header", col)
        while True:
            row = next(reader, None)
            if row is None:
                if replay:
                    return
                time.sleep(poll)          # 수신기가 더 쓸 때까지 기다린다
                continue
            yield ("row", row)
            if replay and speed > 0:
                time.sleep(1.0 / (60.0 * speed))


def render(apps: List[str], gate: np.ndarray, power: np.ndarray, p_obs: float,
           actual: Dict[str, bool], t_s: float) -> str:
    order = np.argsort(-power)
    parts = []
    for j in order:
        on = gate[j] > 0.5
        a = actual.get(apps[j])
        # 사람이 확정한 상태와 어긋나면 표시한다
        mark = "" if a is None else ("=" if a == on else "≠")
        if on or (a is True):
            parts.append(f"{KOR.get(apps[j], apps[j])}{mark} {power[j]:5.0f}W")
    body = "  ".join(parts) if parts else "(전부 꺼짐)"
    return f"t={t_s:7.1f}s  관측 {p_obs:7.1f}W | 예측 {power.sum():7.1f}W | {body}"


def main() -> int:
    ap = argparse.ArgumentParser(description="실시간 추론 + 예측/실제 기록")
    ap.add_argument("--csv", default="data/live.csv", help="수신기가 쓰는 CSV")
    ap.add_argument("--replay", default=None, help="기존 CSV 를 재생해 검증한다")
    ap.add_argument("--speed", type=float, default=20.0, help="재생 배속 (0=최대)")
    ap.add_argument("--ckpt", default="results/adapt_v17.pt")
    ap.add_argument("--every", type=int, default=30, help="추론 간격 (사이클). 30=0.5초")
    ap.add_argument("--log", default="results/live_log.jsonl")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    path = Path(a.replay or a.csv)
    replay = a.replay is not None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, ck = load_model(a.ckpt, dev)
    ti = target_index(WINDOW_CYCLES)
    print("=" * 92)
    print(f"[실시간 추론] {a.ckpt} (stage {ck.get('stage', 1)}) | {path}"
          + ("  [재생]" if replay else "  [수신 대기]"))
    print("  키: " + "  ".join(f"{i+1}={KOR.get(x, x)}" for i, x in enumerate(apps[:9]))
          + "   0=전부끔  space=확정  u=취소  q=종료")
    print("=" * 92)

    ring: deque = deque(maxlen=WINDOW_CYCLES)
    actual: Dict[str, bool] = {}
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    logf = log.open("a", encoding="utf-8")
    n_seen = n_infer = 0
    t_s = 0.0
    col: Dict[str, int] = {}
    t_start = time.time()

    getch = None
    if sys.platform == "win32" and not replay:
        import msvcrt

        def getch():
            return msvcrt.getwch() if msvcrt.kbhit() else None

    try:
        for kind, item in tail_rows(path, replay, a.speed):
            if kind == "header":
                col = item
                continue
            x = row_to_channels(item, col)
            if x is None:
                continue
            ring.append(x)
            n_seen += 1
            try:
                t_s = float(item[col["t_s"]])
            except (ValueError, IndexError):
                pass

            if getch is not None:
                c = getch()
                if c == "q":
                    break
                if c is not None:
                    if c.isdigit():
                        d = int(c)
                        if d == 0:
                            actual = {k: False for k in apps}
                        elif 1 <= d <= len(apps):
                            k = apps[d - 1]
                            actual[k] = not actual.get(k, False)
                    elif c == "u":
                        actual = {}
                    elif c == " ":
                        rec = {"t_s": t_s, "type": "actual", "state": dict(actual)}
                        logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
                        print(f"  [확정] t={t_s:.1f}s  "
                              + ", ".join(KOR.get(k, k) for k, v in actual.items() if v))

            if len(ring) < WINDOW_CYCLES or n_seen % a.every:
                continue

            win = np.stack(ring, axis=1)[None]            # (1, 33, 3600)
            fine, wide = build_inputs(win)
            with torch.no_grad():
                o = model(torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev))
            gate = torch.sigmoid(o["on_logit"])[0].float().cpu().numpy()
            power = o["power"][0].float().cpu().numpy()
            standby = float(o["standby"][0].sum())
            p_obs = float(win[0, 30, ti])
            n_infer += 1

            rec = {"t_s": round(t_s, 3), "type": "pred", "p_observed": round(p_obs, 2),
                   "pred_total": round(float(power.sum()) + standby, 2),
                   "gate": {k: round(float(g), 4) for k, g in zip(apps, gate)},
                   "power_w": {k: round(float(p), 2) for k, p in zip(apps, power)}}
            if actual:
                rec["actual"] = dict(actual)
                dis = [k for k in apps if k in actual and actual[k] != (gate[apps.index(k)] > 0.5)]
                if dis:
                    rec["disagree"] = dis
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not a.quiet:
                print("  " + render(apps, gate, power, p_obs, actual, t_s), flush=True)
    except KeyboardInterrupt:
        print("\n  중단")
    finally:
        logf.close()

    el = time.time() - t_start
    print("=" * 92)
    print(f"  사이클 {n_seen:,}개 ({n_seen/60:.1f}초 분량)  추론 {n_infer:,}회"
          f"  경과 {el:.1f}초  ->  {n_infer/max(el, 1e-9):.1f} 추론/초")
    print(f"  기록: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
