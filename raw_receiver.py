#!/usr/bin/env python3
"""
NILM 원시 파형 수신기 (노트북 쪽) - 유선 ST-Link VCP, 프로토콜 v2
=================================================================
보드에 "40주기를 뽑아 달라"고 요청하고, 돌아온 원시 ADC 파형을 CSV로 저장한다.
펌웨어 NILM_ECE_IF/Core/Inc/nilm_raw.h 의 프레임 형식과 1:1로 맞춰져 있다.

WiFi 로 오는 2Hz 요약(nilm_receiver.py)과는 완전히 별개의 경로다. 이쪽은
USB 케이블 하나만 꽂으면 되고, 요약이 아니라 가공 전 ADC 코드 그대로를 준다.
둘을 동시에 돌려도 서로 간섭하지 않는다 - 보드에서 UART 가 서로 다르다.

명령 (한 글자, 개행 불필요):
    R   40주기 캡처 + 전송        S   중단        ?   상태 한 줄

프레임 (전부 리틀엔디언, 주기 1개 = 프레임 1개, 총 2071바이트):

    [A5 5B] [ver=02] [len=2064]   페이로드 2064B   [CRC16]
    └─ 헤더 5B ────────────────┘                   2B

    CRC16-CCITT(0x1021, init 0xFFFF), ver 바이트부터 페이로드 끝까지.
    매직이 A5 5B 라 WiFi 프레임(A5 5A)과 절대 헷갈리지 않는다.

샘플 1개 = 8바이트, u16 4개. ADC 듀얼 동시 모드의 원본 그대로다:
    low   ADC_low  (x51.7 고이득, 랭크1)  v     ADC_V (low 와 동시)
    high  ADC_high (x3.5 저이득, 랭크2)   bias  1.65V 바이어스 탭 (high 와 동시)
전부 14비트 오프셋 바이너리(0..16383). v2 부터 단일 종단이라 코드가 곧 그 핀의
절대 전압이다: 8192 = 1.65V, 1카운트 = 201.4uV (v1 의 402.8uV 에서 절반).

v1 과 달리 바이어스를 소프트웨어로 뺀다. 그리고 ADC_high 는 신호가 INN 핀에
물려 있던 보드라 부호를 뒤집어야 LOW 와 규약이 맞고, HIGH 샘플의 전압은
랭크2 시각으로 보간해야 한다 - 셋 다 convert() 가 펌웨어와 똑같이 한다.

프레임에는 그 순간 펌웨어가 쓰던 추적 DC 오프셋 3개도 실려 온다. 샘플에는
반영돼 있지 않으므로, 이 스크립트가 그 값으로 펌웨어의 변환과 레인지 판정을
그대로 재현해 CSV 에 같이 넣는다(range / i_a / v_v 열). 즉 "원본"과 "펌웨어가
그 원본을 어떻게 읽었는지"를 나란히 놓고 볼 수 있다.

사용법:
    python raw_receiver.py                  # 포트 자동탐지, 1회 캡처
    python raw_receiver.py --port COM7      # 포트 지정
    python raw_receiver.py --repeat         # Enter 칠 때마다 반복 캡처
    python raw_receiver.py --status         # 보드 상태만 물어보고 끝
    python raw_receiver.py --npz            # CSV 대신 npz (10배 작고 빠름)

준비:
    1. USB 케이블로 보드의 ST-LINK 포트를 노트북에 연결
    2. 다른 프로그램이 그 COM 포트를 잡고 있으면 안 된다
       (STM32CubeIDE 의 시리얼 콘솔, PuTTY 등을 먼저 닫을 것)
"""
import argparse
import csv
import os
import struct
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial 이 필요합니다:  pip install pyserial")
    sys.exit(1)

# ── 프로토콜 상수 (펌웨어 nilm_raw.h 와 반드시 일치) ─────────────────────────
MAGIC = b"\xa5\x5b"
PROTO_VER = 2
SAMPLES = 256
META = 16                                  # seq..rsv
HDR = 5 + META                             # 샘플 시작 오프셋 = 21
SAMPLE_BYTES = SAMPLES * 8                 # = 2048
PAYLOAD = META + SAMPLE_BYTES              # = 2064
FRAME_SIZE = 5 + PAYLOAD + 2               # = 2071

assert HDR == 21 and PAYLOAD == 2064 and FRAME_SIZE == 2071

# ── 보정 상수 (펌웨어 nilm_acq.c 와 반드시 일치) ────────────────────────────
ADC_MID = 8192
VREF = 3.3
LSB = VREF / 16384.0                       # 201.416 uV
CT_RATIO = 2000.0
R_BURDEN = 27.0
GAIN_HIGH = 1.0 + 6800.0 / 2700.0          # 3.5185
GAIN_LOW_EXTRA = 75000.0 / 5100.0          # 14.706
SENS_HIGH = R_BURDEN * GAIN_HIGH / CT_RATIO        # 0.0475 V/A
# LOW 경로 실측 트림 (펌웨어 nilm_acq.c 의 NILM_LOW_TRIM 과 반드시 일치).
# 2026-09-05 이전 캡처를 읽을 때는 1.0 으로 두어야 그때 눈금이 재현된다.
LOW_TRIM = 1.00580
SENS_LOW = SENS_HIGH * GAIN_LOW_EXTRA * LOW_TRIM   # 0.70258 V/A
I_SAT_A = 1.65 / SENS_HIGH
VOLT_SCALE = 309.34      # 2026-09-05 무부하 트림 (그 전 캡처는 307.9)
CLIP_LIMIT = int(1.55 / VREF * 16384.0)            # 7695

# 랭크1 -> 랭크2 시간차 / 샘플 주기 = 9.472us / 65.104us (nilm_acq.c 와 동일)
V_INTERP_F = 0.14549

RANGE_LOW, RANGE_HIGH, RANGE_OVER = 0, 1, 2

# ST-LINK 의 USB 벤더 ID (STMicroelectronics)
ST_VID = 0x0483

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── CRC16-CCITT (0x1021, init 0xFFFF) ───────────────────────────────────────
def _make_crc_table():
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
        table.append(crc)
    return table


_CRC_TABLE = _make_crc_table()


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ byte]
    return crc


# ── 포트 찾기 ───────────────────────────────────────────────────────────────
def find_port():
    """ST-LINK VCP 를 찾는다. 여러 개면 첫 번째를 쓰고 전부 보여 준다."""
    cands = [p for p in list_ports.comports() if p.vid == ST_VID]
    if not cands:
        print("[-] ST-LINK VCP 를 못 찾았습니다. 보이는 포트:")
        for p in list_ports.comports():
            vid = f"{p.vid:04X}:{p.pid:04X}" if p.vid else "?"
            print(f"      {p.device}  [{vid}]  {p.description}")
        print("    USB 케이블이 ST-LINK 쪽에 꽂혔는지, 드라이버가 잡혔는지"
              " 확인하고, 안 되면 --port 로 직접 지정하세요.")
        return None
    if len(cands) > 1:
        print("[!] ST-LINK 포트가 여러 개입니다. 첫 번째를 씁니다:")
        for p in cands:
            print(f"      {p.device}  {p.description}")
    return cands[0].device


# ── 프레임 파싱 ─────────────────────────────────────────────────────────────
def parse_frame(fr: bytes) -> dict:
    """CRC 까지 통과한 프레임 1개 -> dict. 샘플은 아직 원본 코드 그대로."""
    seq, arr, idx, n, oh, ol, ov, flags, _rsv = struct.unpack_from(
        "<IHBBhhhBB", fr, 5)
    # 256샘플 x (low, v, high, bias)
    s = struct.unpack_from(f"<{SAMPLES * 4}H", fr, HDR)
    return {
        "seq": seq, "arr": arr, "idx": idx, "n": n,
        # 1/16 카운트 단위로 실려 온다
        "off_high": oh / 16.0, "off_low": ol / 16.0, "off_volt": ov / 16.0,
        "gap_before": bool(flags & 0x01),
        "low": s[0::4], "v": s[1::4], "high": s[2::4], "bias": s[3::4],
    }


def frame_stream(buf: bytearray, stats: dict):
    """버퍼에서 온전한 프레임을 뽑아 yield. 프레임이 아닌 바이트는 보드가
    보낸 텍스트(예: "[raw] burst done: ...")로 보고 stats['text'] 에 모은다."""
    while True:
        i = buf.find(MAGIC)
        if i < 0:
            if len(buf) > 1:
                stats["text"] += bytes(buf[:-1])
                del buf[:-1]
            return
        if i > 0:
            stats["text"] += bytes(buf[:i])
            del buf[:i]
        if len(buf) < 5:
            return
        ver = buf[2]
        length = buf[3] | (buf[4] << 8)
        if ver != PROTO_VER or length != PAYLOAD:
            stats["resync"] += 2       # 데이터 안의 우연한 A5 5B
            del buf[:2]
            continue
        if len(buf) < FRAME_SIZE:
            return
        crc_rx = buf[FRAME_SIZE - 2] | (buf[FRAME_SIZE - 1] << 8)
        if crc16_ccitt(bytes(buf[2:FRAME_SIZE - 2])) != crc_rx:
            stats["crc_err"] += 1
            del buf[:2]                # 매직만 넘기고 다시 찾는다
            continue
        fr = bytes(buf[:FRAME_SIZE])
        del buf[:FRAME_SIZE]
        yield parse_frame(fr)


# ── 캡처 ────────────────────────────────────────────────────────────────────
def capture(ser, timeout=6.0, verbose=True):
    """'R' 을 보내고 한 버스트를 받는다. (frames, stats) 반환."""
    ser.reset_input_buffer()
    ser.write(b"R")
    ser.flush()

    buf = bytearray()
    stats = {"crc_err": 0, "resync": 0, "text": b""}
    frames = {}
    n_total = None
    t0 = time.time()
    t_last = t0

    while time.time() - t0 < timeout:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            t_last = time.time()
            for f in frame_stream(buf, stats):
                frames[f["idx"]] = f
                n_total = f["n"]
                if verbose:
                    print(f"\r  수신 {len(frames):3d}/{n_total}"
                          f"  (idx {f['idx']:2d}, seq {f['seq']})", end="")
            if n_total is not None and len(frames) >= n_total:
                break
        elif time.time() - t_last > 1.5:
            break                       # 1.5초간 조용하면 끝난 것으로 본다
    if verbose:
        print()
    return [frames[k] for k in sorted(frames)], stats, n_total


# ── 물리량 환산 (펌웨어 nilm_acq.c 의 판정을 그대로 재현) ───────────────────
def convert(f):
    """프레임 1개의 256샘플을 펌웨어와 똑같은 규칙으로 해석한다.

    v2(단일 종단)부터 세 가지가 달라졌다:
      1. 바이어스 코드를 빼야 예전의 차동값이 복원된다 (주기 평균으로)
      2. ADC_high 는 신호가 INN 핀에 물려 있던 보드라 부호를 뒤집어야
         LOW 와 부호 규약이 맞는다 (안 뒤집으면 레인지 혼재 주기가 깨진다)
      3. HIGH 샘플은 랭크2 시각이라 전압을 그쪽으로 선형 보간한다

    클리핑 판정은 바이어스를 빼기 전의 raw 코드로, 추적 오프셋이 아니라
    고정 8192 로 한다. 펌웨어가 그렇게 하는 이유(추적 오프셋을 쓰면 검출기가
    자기 문턱을 만들어 폭주한다)는 nilm_acq.c 의 NILM_CLIP_LIMIT_CNT 주석에
    있다. 여기서도 똑같이 해야 재현이 된다.
    """
    off_h = ADC_MID + f["off_high"]
    off_l = ADC_MID + f["off_low"]
    off_v = ADC_MID + f["off_volt"]
    low, volt, high, bias = f["low"], f["v"], f["high"], f["bias"]
    # 바이어스는 샘플별이 아니라 주기 평균으로 뺀다 (펌웨어와 동일.
    # 근거는 nilm_acq.c 의 bias_sum 주석: 바이어스 채널 AC 의 87%가
    # 우리 자신의 읽기 잡음이라, 샘플별로 빼면 HIGH 경로 잡음만 늘었다)
    bi = sum(bias) / float(SAMPLES)
    out = []
    for k in range(SAMPLES):
        lo, vv, hi = low[k], volt[k], high[k]
        d_lo = lo - ADC_MID
        d_hi = hi - ADC_MID
        clip_lo = abs(d_lo) >= CLIP_LIMIT
        clip_hi = abs(d_hi) >= CLIP_LIMIT

        c_lo = ADC_MID + (lo - bi)
        c_hi = ADC_MID - (hi - bi)          # 부호 반전
        c_v = ADC_MID + (vv - bi)

        if not clip_lo:
            rng = RANGE_LOW
            i_a = (c_lo - off_l) * LSB / SENS_LOW
            v_v = (c_v - off_v) * LSB * VOLT_SCALE      # 같은 랭크(1)
        else:
            if not clip_hi:
                rng = RANGE_HIGH
                i_a = (c_hi - off_h) * LSB / SENS_HIGH
            else:
                rng = RANGE_OVER
                # c_hi 가 부호 반전이라 위쪽 레일이 전류의 음의 최대다
                i_a = I_SAT_A if d_hi <= 0 else -I_SAT_A
            # 랭크2 시각으로 전압 보간 (마지막 샘플은 직전 기울기로 외삽)
            v_next = volt[k + 1] if k + 1 < SAMPLES else 2 * vv - volt[k - 1]
            v_b = vv + V_INTERP_F * (v_next - vv)
            v_v = ((ADC_MID + (v_b - bi)) - off_v) * LSB * VOLT_SCALE
        out.append((rng, i_a, v_v,
                    (c_lo - off_l) * LSB / SENS_LOW,    # 두 경로를 각각
                    (c_hi - off_h) * LSB / SENS_HIGH))  # 보고 싶을 때가 있다
    return out


CSV_HEADER = ["t_s", "cyc", "seq", "n",
              "low", "v", "high", "bias",               # 원본 ADC 코드
              "range", "i_a", "v_v",                    # 펌웨어 재현
              "i_low_a", "i_high_a",                    # 두 경로 각각
              "fs_hz", "off_high", "off_low", "off_volt", "gap_before"]


def write_csv(path, frames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = 0.0
    prev_idx = None
    fs_last = 15360.0
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(CSV_HEADER)
        for f in frames:
            fs = 240e6 / (f["arr"] + 1)
            # 빠진 주기가 있으면 그만큼 시간축을 건너뛴다(직전 fs 로 추정)
            if prev_idx is not None and f["idx"] > prev_idx + 1:
                t += (f["idx"] - prev_idx - 1) * SAMPLES / fs_last
            prev_idx, fs_last = f["idx"], fs
            conv = convert(f)
            dt = 1.0 / fs
            for k in range(SAMPLES):
                rng, i_a, v_v, i_lo, i_hi = conv[k]
                w.writerow([
                    f"{t + k * dt:.7f}", f["idx"], f["seq"], k,
                    f["low"][k], f["v"][k], f["high"][k], f["bias"][k],
                    rng, f"{i_a:.6f}", f"{v_v:.4f}",
                    f"{i_lo:.6f}", f"{i_hi:.6f}",
                    f"{fs:.2f}", f"{f['off_high']:.4f}",
                    f"{f['off_low']:.4f}", f"{f['off_volt']:.4f}",
                    int(f["gap_before"]),
                ])
            t += SAMPLES * dt


def write_npz(path, frames):
    try:
        import numpy as np
    except ImportError:
        print("[-] --npz 에는 numpy 가 필요합니다. CSV 로 저장하세요.")
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        low=np.array([f["low"] for f in frames], dtype=np.uint16),
        v=np.array([f["v"] for f in frames], dtype=np.uint16),
        high=np.array([f["high"] for f in frames], dtype=np.uint16),
        bias=np.array([f["bias"] for f in frames], dtype=np.uint16),
        seq=np.array([f["seq"] for f in frames], dtype=np.uint32),
        idx=np.array([f["idx"] for f in frames], dtype=np.uint8),
        arr=np.array([f["arr"] for f in frames], dtype=np.uint16),
        off_high=np.array([f["off_high"] for f in frames]),
        off_low=np.array([f["off_low"] for f in frames]),
        off_volt=np.array([f["off_volt"] for f in frames]),
    )
    return True


def report(frames, stats, n_total):
    if not frames:
        print("[-] 프레임을 하나도 못 받았습니다.")
        txt = stats["text"].decode("utf-8", "replace").strip()
        if txt:
            print(f"    보드가 보낸 말: {txt}")
        else:
            print("    보드가 조용합니다. 펌웨어에 nilm_raw 가 올라갔는지,")
            print("    --baud 가 펌웨어의 NILM_RAW_BAUD 와 같은지 확인하세요.")
        return
    got = {f["idx"] for f in frames}
    n = n_total or max(got) + 1
    missing = sorted(set(range(n)) - got)
    seqs = [f["seq"] for f in frames]
    contiguous = all(b - a == 1 for a, b in zip(seqs, seqs[1:]))
    fs = 240e6 / (frames[0]["arr"] + 1)

    print(f"  받은 주기 {len(frames)}/{n}"
          f"   샘플 {len(frames) * SAMPLES:,}개"
          f"   {len(frames) * SAMPLES / fs:.3f}초분")
    print(f"  fs {fs:,.1f} Hz  ->  계통 {fs / SAMPLES:.4f} Hz"
          f"   seq {seqs[0]}..{seqs[-1]}"
          f" ({'연속' if contiguous else '불연속!'})")
    print(f"  추적 오프셋 [카운트]  high {frames[0]['off_high']:+.2f}"
          f"   low {frames[0]['off_low']:+.2f}"
          f"   volt {frames[0]['off_volt']:+.2f}")
    if missing:
        print(f"  ! 빠진 주기 {len(missing)}개: {missing}")
        print("    호스트가 잠깐 멎어 보드 링이 찼다는 뜻입니다."
              " 다른 프로그램을 닫고 다시 해 보세요.")
    if stats["crc_err"] or stats["resync"]:
        print(f"  ! CRC오류 {stats['crc_err']}  재동기 {stats['resync']}바이트")
    txt = stats["text"].decode("utf-8", "replace").strip()
    if txt:
        for line in txt.splitlines():
            if line.strip():
                print(f"  보드> {line.strip()}")


def main():
    ap = argparse.ArgumentParser(
        description="NILM 원시 파형 수신기 (유선 ST-Link VCP)")
    ap.add_argument("--port", help="COM 포트 (생략하면 ST-LINK 자동탐지)")
    ap.add_argument("--baud", type=int, default=2000000,
                    help="보레이트. 펌웨어의 NILM_RAW_BAUD 와 같아야 한다"
                         " (기본 2000000)")
    ap.add_argument("--out", metavar="FILE",
                    help="저장 파일명. 이름만 주면 data/ 안에 넣는다"
                         " (기본: data/raw_<날짜시각>.csv)")
    ap.add_argument("--npz", action="store_true", help="CSV 대신 npz 로 저장")
    ap.add_argument("--repeat", action="store_true",
                    help="Enter 칠 때마다 반복 캡처 (q + Enter 로 종료)")
    ap.add_argument("--status", action="store_true",
                    help="보드 상태만 물어보고 끝낸다")
    args = ap.parse_args()

    port = args.port or find_port()
    if port is None:
        return 1

    try:
        ser = serial.Serial(port, args.baud, timeout=0.2, write_timeout=2.0)
    except serial.SerialException as e:
        print(f"[-] {port} 를 열 수 없습니다: {e}")
        print("    다른 프로그램(CubeIDE 콘솔, PuTTY 등)이 잡고 있지 않은지"
              " 확인하세요.")
        return 1

    print(f"[+] {port} @ {args.baud:,} baud")
    time.sleep(0.2)

    if args.status:
        ser.reset_input_buffer()
        ser.write(b"?")
        ser.flush()
        time.sleep(0.4)
        txt = ser.read(4096).decode("utf-8", "replace").strip()
        print(txt if txt else "[-] 응답이 없습니다. --baud 를 확인하세요.")
        ser.close()
        return 0

    n_run = 0
    try:
        while True:
            n_run += 1
            print(f"[>] 캡처 요청 (R) ...")
            t0 = time.time()
            frames, stats, n_total = capture(ser)
            print(f"[=] {time.time() - t0:.2f}초")
            report(frames, stats, n_total)

            if frames:
                base = args.out or time.strftime("raw_%Y%m%d_%H%M%S")
                if n_run > 1 and args.out:
                    root, ext = os.path.splitext(base)
                    base = f"{root}_{n_run}{ext}"
                if not os.path.dirname(base):
                    base = os.path.join(DATA_DIR, base)
                root, ext = os.path.splitext(base)
                if args.npz:
                    if write_npz(root + ".npz", frames):
                        print(f"저장됨: {root}.npz")
                else:
                    path = base if ext else root + ".csv"
                    write_csv(path, frames)
                    print(f"저장됨: {path}  ({len(frames) * SAMPLES:,}행)")

            if not args.repeat:
                break
            try:
                if input("\nEnter=다시 캡처, q+Enter=종료 > ").strip().lower() == "q":
                    break
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\n중단합니다.")
    finally:
        try:
            ser.write(b"S")          # 혹시 캡처가 걸려 있으면 풀어 준다
            ser.flush()
        except Exception:
            pass
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
