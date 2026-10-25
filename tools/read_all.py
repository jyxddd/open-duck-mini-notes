"""
Read-only status of all 14 servos (no torque change).
Usage: python read_all.py [/dev/ttyACM0]
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)


def rd(s, sid, addr, n):
    body = [sid, 4, 0x02, addr, n]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
    s.flush()
    time.sleep(0.01)
    r = s.read(32)
    i = r.find(bytes([0xFF, 0xFF, sid]))
    if i < 0 or len(r) < i + 5 + n:
        return None
    return list(r[i + 5 : i + 5 + n])


with serial.Serial(PORT, 1000000, timeout=0.03) as s:
    for sid in IDS:
        p = rd(s, sid, 56, 2)
        if p is None:
            print(f"ID {sid}: NOT FOUND")
            continue
        tq = rd(s, sid, 40, 1)
        v = rd(s, sid, 62, 1)
        mv = rd(s, sid, 14, 1)
        pos = p[0] | p[1] << 8
        d = pos - 2048
        tq_s = tq[0] if tq else "?"
        v_s = v[0] / 10 if v else "?"
        mv_s = mv[0] if mv else "?"
        print(f"ID {sid}: pos {pos:5d}  diff {d:+6d} ({d * 360 / 4096:+7.1f} deg)  torque={tq_s}  volt={v_s}V  maxV={mv_s}")
