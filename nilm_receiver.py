#!/usr/bin/env python3
"""
NILM 수신기 (노트북 쪽) - 프로토콜 v3
=====================================
STM32 -> ESP-01S 가 TCP 클라이언트로 접속해 오는 것을 받아 CSV로 저장한다.
펌웨어 NILM_ECE_IF/Core/Inc/nilm_link.h 의 프레임 형식과 1:1로 맞춰져 있다.

프레임 (전부 리틀엔디언, 총 3213바이트):

    [A5 5A] [ver=03] [len=3206]   공통부 86B   주기별 104B x 30 = 3120B  [CRC16]
    └─ 헤더 5B ─────────────────┘              └─ cycles[0] .. cycles[29] ┘  2B

    CRC16-CCITT(다항식 0x1021, init 0xFFFF), ver 바이트부터 페이로드 끝까지.

  - 공통부(SLOW) : 0.5초 평균. 주파수 / Vrms / 전압 THD·고조파 / 품질 플래그.
                   over_cycle_map 은 측정범위 초과(두 전류 경로 모두 클리핑)가
                   발생한 주기를 표시하는 30비트 비트맵.
  - 주기별(CYCLE): 계통 1주기(1/60초)마다의 전류 고조파 15차 rms/위상, P,
                   Irms, V-I 위상차, 범위초과 샘플 수. 평균하지 않은 원본.

CSV는 "주기 1개 = 1행"으로 쓴다(초당 60행). 공통부 값은 그 주기가 속한
0.5초 창의 값으로 각 행에 반복해 넣는다 - pandas로 바로 읽어 쓰기 위함.

CSV는 이 스크립트 옆의 data/ 폴더에 쌓인다(없으면 만든다). 실행 디렉터리와
무관하게 항상 같은 곳이라 학습 데이터를 한 군데에서 찾을 수 있다.

사용법:
    python nilm_receiver.py                     # data/nilm_<날짜시각>.csv
    python nilm_receiver.py --csv run1.csv      # data/run1.csv (있으면 이어쓰기)
    python nilm_receiver.py --csv D:\\tmp\\x.csv  # 경로를 주면 그대로
    python nilm_receiver.py --no-csv            # 콘솔만
    python nilm_receiver.py --port 5000         # 포트 변경 (기본 5000)
    python nilm_receiver.py --quiet             # 콘솔 요약 끄기

준비 (Windows 설정 -> 네트워크 및 인터넷 -> 모바일 핫스팟):
    1. 핫스팟 이름/암호를 nilm_link.h 의 NILM_WIFI_SSID / NILM_WIFI_PASS 와 일치
    2. 네트워크 대역 2.4GHz (ESP8266은 5GHz를 못 잡는다)
    3. "전원 절약" 끄기, 측정 중 노트북 절전 금지
    4. 방화벽에서 이 포트의 인바운드 TCP 허용 (아래 명령 참고)
       netsh advfirewall firewall add rule name="NILM 5000" ^
             dir=in action=allow protocol=TCP localport=5000
실행 순서: 핫스팟 ON -> 이 스크립트 실행 -> 보드 리셋
"""
import argparse
import csv
import os
import socket
import struct
import sys
import time

# ── 프로토콜 상수 (펌웨어 nilm_link.h 와 반드시 일치) ────────────────────────
MAGIC = b"\xa5\x5a"
PROTO_VER = 3
ACK = b"\x06"          # 프레임 정상 수신 시 보드로 되돌리는 생존 신호
HARMONICS = 15
CYCLES = 30
CYCLE_HZ = 60.0        # 주기별 데이터의 시간 해상도

# NILM_WireSlow: seq, freq, vrms, thd_v, vh[15], over_range, clip_volt,
#                range, flags, over_map
SLOW_FMT = "<I3f15fHHBBI"
SLOW_SIZE = struct.calcsize(SLOW_FMT)            # = 86

# NILM_WireCycle: irms, p_w, phase_cdeg, range, over_count,
#                 ih[15], ih_cdeg[15], reserved
CYC_FMT = "<ffhBB15f15hH"
CYC_SIZE = struct.calcsize(CYC_FMT)              # = 104

PAYLOAD_SIZE = SLOW_SIZE + CYCLES * CYC_SIZE     # = 3206
FRAME_SIZE = 5 + PAYLOAD_SIZE + 2                # = 3213

assert SLOW_SIZE == 86, f"SLOW_SIZE={SLOW_SIZE}, 펌웨어와 어긋남"
assert CYC_SIZE == 104, f"CYC_SIZE={CYC_SIZE}, 펌웨어와 어긋남"

# 이 시간 동안 무수신이면 죽은 연결로 보고 끊는다.
# 펌웨어는 ACK가 안 오면 같은 프레임을 2.5초 간격으로 최대 5번까지 다시
# 보낸다(nilm_link.h). 그동안 WiFi가 정말 막혀 있으면 이쪽엔 아무것도 안
# 들어오는데, 이 값이 짧으면 곧 되살아날 연결을 먼저 끊어 버린다.
# 실제로 5초로 두었더니 9초짜리 정체에서 연결이 끊기고 ESP 재접속까지
# 겹쳐 공백이 더 커졌다. 펌웨어가 한 프레임에 쓰는 12.5초보다 길게 잡는다.
STALE_TIMEOUT = 20.0


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


# ── 페이로드 -> 파이썬 dict ─────────────────────────────────────────────────
def parse_payload(payload: bytes) -> dict:
    s = struct.unpack(SLOW_FMT, payload[:SLOW_SIZE])
    d = {
        "seq": s[0],
        "freq_hz": s[1],
        "vrms": s[2],
        "thd_v": s[3],
        "vh_rms": s[4:4 + HARMONICS],
        "over_range": s[4 + HARMONICS],
        "clip_volt": s[5 + HARMONICS],
        "range": s[6 + HARMONICS],
        "flags": s[7 + HARMONICS],
        "over_map": s[8 + HARMONICS],
    }
    d["pll_locked"] = bool(d["flags"] & 0x01)
    d["cal_applied"] = bool(d["flags"] & 0x02)
    # 비트맵 -> 측정범위 초과가 발생한 주기 인덱스 목록
    d["over_cycles"] = [i for i in range(CYCLES) if d["over_map"] & (1 << i)]

    cycles = []
    off = SLOW_SIZE
    for _ in range(CYCLES):
        c = struct.unpack(CYC_FMT, payload[off:off + CYC_SIZE])
        cycles.append({
            "irms": c[0],
            "p_w": c[1],
            "phase_deg": c[2] / 100.0,          # 0.01도 -> 도
            "range": c[3],
            "over": c[4],
            "ih_rms": c[5:5 + HARMONICS],
            "ih_deg": [x / 100.0 for x in c[5 + HARMONICS:5 + 2 * HARMONICS]],
        })
        off += CYC_SIZE
    d["cycles"] = cycles
    return d


# ── 소켓 스트림 -> 프레임 ───────────────────────────────────────────────────
def frame_stream(conn: socket.socket, stats: dict, seen: set):
    """바이트 스트림에서 CRC까지 통과한 프레임의 payload만 뽑아 yield.

    seen 은 이미 받은 seq 집합이다. 펌웨어는 ACK가 제때 안 오면 같은
    프레임을 다시 보내는데, 첫 장이 무사히 갔고 ACK만 늦었던 경우엔 같은
    seq가 두 번 도착한다. 그때 ACK를 두 번 돌려주면 펌웨어가 그 두 번째를
    "다음 프레임에 대한 ACK"로 오해해서, 아직 도착 확인도 안 된 프레임을
    큐에서 빼 버린다(그 장이 조용히 사라진다).
    그래서 중복은 ACK 없이 버린다 - 유일한 프레임마다 ACK 하나가 된다.  """
    buf = b""
    conn.settimeout(1.0)
    t_last = time.time()
    while True:
        try:
            chunk = conn.recv(16384)
        except socket.timeout:
            # 핫스팟이 끊기면 이쪽 소켓은 FIN/RST를 못 받아 살아 있는 것처럼
            # 남는다. 그동안 보드는 이미 새 연결을 걸어 두므로, 낡은 소켓을
            # 제때 버려야 재접속을 받아들일 수 있다.
            if time.time() - t_last > STALE_TIMEOUT:
                print(f"[-] {STALE_TIMEOUT:.0f}초간 수신 없음 - 연결을 끊습니다.")
                return
            continue
        except OSError as e:
            print(f"[-] 소켓 오류: {e}")
            return
        if not chunk:
            return                          # 상대가 연결을 닫음
        t_last = time.time()
        buf += chunk

        while True:
            idx = buf.find(MAGIC)
            if idx < 0:
                # 매직이 없음: 매직이 경계에 걸쳤을 수 있으니 1바이트만 남긴다
                if len(buf) > 1:
                    stats["resync"] += len(buf) - 1
                buf = buf[-1:]
                break
            if idx > 0:
                stats["resync"] += idx
                buf = buf[idx:]             # 매직 앞 쓰레기 제거
            if len(buf) < 5:
                break                       # 헤더가 아직 다 안 옴

            ver = buf[2]
            length = buf[3] | (buf[4] << 8)
            if ver != PROTO_VER or length != PAYLOAD_SIZE:
                # 형식 불일치 = 데이터 안의 우연한 A5 5A. 다음 매직부터 재탐색
                stats["resync"] += 2
                buf = buf[2:]
                continue

            total = 5 + length + 2
            if len(buf) < total:
                break                       # 본문이 아직 다 안 옴

            frame, buf = buf[:total], buf[total:]
            crc_rx = frame[-2] | (frame[-1] << 8)
            if crc16_ccitt(frame[2:-2]) != crc_rx:
                stats["crc_err"] += 1
                continue                    # 조용히 버리고 다음 매직부터

            seq = struct.unpack_from("<I", frame, 5)[0]
            if seq in seen:                 # 재전송된 중복 - ACK 없이 버린다
                stats["dup"] += 1
                continue
            seen.add(seq)
            if len(seen) > 4096:            # 오래된 것부터 정리
                seen.difference_update({s for s in seen if s < seq - 2048})

            # ACK 회신: 펌웨어가 이걸 흐름제어로 쓴다. 이 응답을 받아야
            # 다음 프레임을 내보내므로 절대 빼면 안 된다.
            try:
                conn.sendall(ACK)
            except OSError:
                return
            yield frame[5:-2]


# ── CSV ─────────────────────────────────────────────────────────────────────
CSV_HEADER = (
    ["host_time", "t_s", "seq", "cycle",
     # --- 그 주기의 값 (평균 없음) ---
     "irms", "p_w", "phase_deg", "range", "over_count", "over_range",
     # --- 그 주기가 속한 0.5초 창의 공통 값 ---
     "freq_hz", "vrms", "thd_v", "pll_locked", "cal_applied",
     "win_range", "win_over_range_count", "win_clip_volt_count"]
    + [f"ih{h}" for h in range(1, HARMONICS + 1)]
    + [f"ihdeg{h}" for h in range(1, HARMONICS + 1)]
    + [f"vh{h}" for h in range(1, HARMONICS + 1)]
)


# CSV는 전부 이 폴더 아래에 모은다. 스크립트 위치 기준이라 어느 디렉터리에서
# 실행하든 같은 곳에 쌓이고, 학습 데이터를 한 곳에서 찾을 수 있다.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def resolve_csv_path(arg):
    """--csv 인자를 실제 경로로.

    이름만 주면(예: run1.csv) data/ 안에 넣고, 경로를 포함해 주면
    (예: ../other/x.csv, D:\\tmp\\x.csv) 그 뜻을 그대로 존중한다.
    """
    if arg is None:
        return os.path.join(DATA_DIR, time.strftime("nilm_%Y%m%d_%H%M%S.csv"))
    if os.path.dirname(arg):
        return arg
    return os.path.join(DATA_DIR, arg)


def open_csv(path: str):
    """이어쓰기로 열고, 새 파일이면 헤더를 넣는다."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(CSV_HEADER)
    return f, w


def write_frame(writer, d: dict, t_recv: float, t0_dev):
    """프레임 1개 -> CSV 30행.

    t_s 는 보드 기준 시각이다. seq 는 0.5초마다 1씩 오르므로
        t_s = (seq - seq0) * 0.5 + cycle / 60
    이 되고, 호스트 스케줄링 지터와 무관한 균일 시간축이 된다.
    (호스트 벽시계는 host_time 열에 따로 남긴다)
    """
    host = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_recv)) + \
        f".{int((t_recv % 1) * 1000):03d}"
    base = (d["seq"] - t0_dev) * 0.5
    over_set = set(d["over_cycles"])
    vh = [f"{x:.4f}" for x in d["vh_rms"]]

    for i, c in enumerate(d["cycles"]):
        writer.writerow(
            [host, f"{base + i / CYCLE_HZ:.6f}", d["seq"], i,
             f"{c['irms']:.6f}", f"{c['p_w']:.4f}", f"{c['phase_deg']:.2f}",
             c["range"], c["over"], int(i in over_set),
             f"{d['freq_hz']:.4f}", f"{d['vrms']:.3f}", f"{d['thd_v']:.5f}",
             int(d["pll_locked"]), int(d["cal_applied"]),
             d["range"], d["over_range"], d["clip_volt"]]
            + [f"{x:.6f}" for x in c["ih_rms"]]
            + [f"{x:.2f}" for x in c["ih_deg"]]
            + vh)


# ── 콘솔 요약 ───────────────────────────────────────────────────────────────
def summary_line(d: dict) -> str:
    cyc = d["cycles"]
    p = [c["p_w"] for c in cyc]
    last = cyc[-1]
    msg = (f"#{d['seq']:06d} "
           f"f={d['freq_hz']:7.3f}{'L' if d['pll_locked'] else 'u'} "
           f"V={d['vrms']:6.1f} "
           f"P={sum(p) / len(p):8.2f}W (min {min(p):7.1f} / max {max(p):7.1f}) "
           f"I1={last['ih_rms'][0]:8.4f}A "
           f"ph={last['phase_deg']:7.2f}d "
           f"rng={'HIGH' if d['range'] else 'LOW '}"
           f"{'*' if d['cal_applied'] else ' '}")
    if d["over_cycles"]:
        idx = ",".join(str(i) for i in d["over_cycles"])
        msg += f"  !! OVER-RANGE @cycle[{idx}]"
    return msg


def loss_str(got: int, lost: int) -> str:
    """수신/유실을 '몇 개 중 몇 개(몇 %)' 형태로.

    분모는 seq로 셈한 '와야 했던 프레임 수'(수신 + 공백)다. 보드가 리셋되어
    seq가 되돌아간 구간과 연결이 끊겨 있던 시간은 세지 않는다 - 그 동안의
    공백은 링크 유실이 아니라 그냥 측정이 없던 시간이기 때문이다.
    따라서 이 값은 "붙어 있는 동안 얼마나 흘렸나"를 뜻한다.               """
    expect = got + lost
    if expect == 0:
        return "수신 없음"
    return f"{got}/{expect}프레임, 유실 {lost} ({lost / expect * 100:.2f}%)"


def print_summary(frames: int, lost: int, stats: dict, dur: float, csv_path):
    """종료 시 최종 집계. 유실률이 이 스크립트의 핵심 품질 지표다."""
    expect = frames + lost
    rate = (lost / expect * 100.0) if expect else 0.0

    print(f"\n종료: {dur:.0f}초 동안 {frames}프레임 수신 "
          f"({frames * CYCLES}주기, "
          f"{frames * CYCLES / max(dur, 1e-9):.1f}주기/초)")
    print(f"  유실률 : {rate:.2f}%  (유실 {lost} / 기대 {expect}프레임"
          f" = 주기 {lost * CYCLES}개 분량)")
    print(f"  CRC오류: {stats['crc_err']}프레임"
          f"   중복(재전송): {stats.get('dup', 0)}프레임"
          f"   재동기: {stats['resync']}바이트")
    if rate > 1.0:
        # 유실은 대부분 ESP 버퍼 넘침이다. 보레이트 상향(460800)이 안 됐거나
        # 청크 간격이 짧으면 여기가 가장 먼저 티가 난다.
        print("  ! 유실률이 1%를 넘습니다. 펌웨어 로그에서 baud가 460800으로"
              " 올라갔는지, NILM_LINK_TX_GAP_MS를 늘릴 여지가 있는지"
              " 확인하세요.")
    if csv_path:
        print(f"저장됨: {csv_path}")


def local_ips():
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        return sorted({i[4][0] for i in infos})
    except OSError:
        return []


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="NILM WiFi 수신기 (프로토콜 v3) - 주기별 데이터를 CSV로 저장")
    ap.add_argument("--port", type=int, default=5000, help="TCP 포트 (기본 5000)")
    ap.add_argument("--csv", metavar="FILE",
                    help="CSV 파일명. 이름만 주면 data/ 안에 저장한다"
                         " (기본: data/nilm_<날짜시각>.csv, 있으면 이어쓰기)")
    ap.add_argument("--no-csv", action="store_true", help="CSV 저장 안 함")
    ap.add_argument("--quiet", action="store_true", help="콘솔 요약 끄기")
    args = ap.parse_args()

    fcsv = writer = None
    csv_path = None
    if not args.no_csv:
        csv_path = resolve_csv_path(args.csv)
        fcsv, writer = open_csv(csv_path)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"포트 {args.port} 를 열 수 없습니다: {e}")
        return 1
    srv.listen(1)
    # accept() 를 블로킹으로 두면 Windows에서 Ctrl+C 가 먹지 않는다. 파이썬은
    # 바이트코드 사이에서만 시그널을 처리하는데 블로킹 소켓 호출은 C 안에서
    # 멈춰 있기 때문이다. 접속 대기 중에 종료하면 최종 요약도 못 찍는다.
    # 1초마다 풀어 주면 그 틈에 KeyboardInterrupt 가 올라온다.
    srv.settimeout(1.0)

    ips = local_ips()
    print(f"listening on 0.0.0.0:{args.port}   (Ctrl+C 로 종료)")
    print(f"이 PC의 IPv4: {', '.join(ips) if ips else '(확인 실패)'}")
    if "192.168.137.1" not in ips:
        print("  주의: 192.168.137.1 이 없습니다. 모바일 핫스팟이 꺼져 있거나,"
              " 펌웨어의 NILM_HOST_IP 를 위 주소 중 하나로 바꿔야 합니다.")
    if csv_path:
        print(f"CSV: {csv_path}  (주기 1개 = 1행, 초당 60행)")
    else:
        print("CSV: 저장 안 함")

    frames = 0
    lost = 0
    stats = {"crc_err": 0, "resync": 0, "dup": 0}
    t_wall0 = time.time()

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue                # 시그널 처리 틈 - 정상 경로
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[+] connected from {addr[0]}:{addr[1]}")
            seq_prev = None
            seq_base = None
            seq_seen = set()    # 중복 판정용 (펌웨어가 재전송한다)
            c_frames = 0        # 이 연결에서만의 수신/유실
            c_lost = 0
            try:
                for payload in frame_stream(conn, stats, seq_seen):
                    t_recv = time.time()
                    d = parse_payload(payload)
                    frames += 1
                    c_frames += 1

                    # seq 공백 = 유실. 보드가 리셋되면 seq가 되돌아가므로
                    # 그때는 시간축 기준(seq_base)을 다시 잡는다.
                    if seq_base is None or d["seq"] < (seq_prev or 0):
                        seq_base = d["seq"]
                        if seq_prev is not None:
                            print("[!] seq 가 되돌아감 - 보드가 리셋된 것 같습니다."
                                  " 시간축을 다시 잡습니다.")
                    elif seq_prev is not None and d["seq"] != seq_prev + 1:
                        gap = d["seq"] - seq_prev - 1
                        lost += gap
                        c_lost += gap
                        print(f"[!] 프레임 {gap}개 유실 "
                              f"(seq {seq_prev + 1}..{d['seq'] - 1})")
                    seq_prev = d["seq"]

                    if writer is not None:
                        write_frame(writer, d, t_recv, seq_base)
                        fcsv.flush()        # 중간에 죽어도 데이터는 남는다

                    if not args.quiet:
                        print(summary_line(d))
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[-] 연결 끊김: {e}")
            finally:
                conn.close()
                print(f"[-] disconnected ({loss_str(c_frames, c_lost)}), "
                      f"재접속 대기...")
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if fcsv is not None:
            fcsv.close()
        print_summary(frames, lost, stats, time.time() - t_wall0, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
