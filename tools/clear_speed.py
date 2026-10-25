"""
Reset goal speed (addr 46) and goal time (addr 44) to 0 on all servos.
My test tools leave a speed limit in servo RAM; the runtime/policy needs 0 (= no limit).
Usage: python clear_speed.py [/dev/ttyACM0]
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)


def wr(s, sid, addr, data):
    body = [sid, len(data) + 3, 0x03, addr, *data]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
    s.flush()
    time.sleep(0.01)
    s.read(32)


def rd(s, sid, addr, n):
    body = [sid, 4, 0x02, addr, n]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
    s.flush()
    time.sleep(0.01)
    r = s.read(32)
    i = r.find(bytes([0xFF, 0xFF, sid]))
    return list(r[i + 5 : i + 5 + n]) if i >= 0 and len(r) >= i + 5 + n else None


with serial.Serial(PORT, 1000000, timeout=0.03) as s:
    for sid in IDS:
        wr(s, sid, 46, [0, 0])
        wr(s, sid, 44, [0, 0])
    bad = [sid for sid in IDS if rd(s, sid, 46, 2) != [0, 0]]
    print("speed/time cleared on all servos" if not bad else f"NOT cleared: {bad}")
