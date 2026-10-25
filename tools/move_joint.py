"""
Direction test: slowly move one servo by +rad from 2048, hold, then go back to 2048.
Same convention as the runtime (rustypot): rad = (pos - 2048) * 2*pi / 4096, so +rad = +steps.

Usage (FD must be Closed):
  python move_joint.py --port COM4 --id 12 --rad 0.3
"""
import argparse
import math
import time

import serial

ADDR_TORQUE = 40
ADDR_GOAL = 42
ADDR_POS = 56


def checksum(body):
    return (~sum(body)) & 0xFF


def write(s, sid, addr, data):
    body = [sid, len(data) + 3, 0x03, addr, *data]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, checksum(body)]))
    s.flush()
    time.sleep(0.02)
    s.read(64)


def read_pos(s, sid):
    body = [sid, 4, 0x02, ADDR_POS, 2]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, checksum(body)]))
    s.flush()
    time.sleep(0.02)
    r = s.read(64)
    i = r.find(bytes([0xFF, 0xFF, sid]))
    if i < 0 or len(r) < i + 8:
        return None
    return r[i + 5] | (r[i + 6] << 8)


def goto(s, sid, pos, speed=200):
    write(s, sid, ADDR_GOAL, [pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8])
    write(s, sid, ADDR_TORQUE, [1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM4")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--rad", type=float, default=0.3)
    p.add_argument("--hold_s", type=float, default=3.0)
    a = p.parse_args()
    a.rad = max(-0.5, min(0.5, a.rad))  # safety: at most ~29 degrees
    target = 2048 + round(a.rad * 4096 / (2 * math.pi))

    with serial.Serial(a.port, 1000000, timeout=0.05) as s:
        start = read_pos(s, a.id)
        if start is None:
            print(f"ID {a.id} not found")
            return
        goto(s, a.id, target)
        time.sleep(a.hold_s)
        mid = read_pos(s, a.id)
        goto(s, a.id, 2048)
        time.sleep(1.5)
        end = read_pos(s, a.id)
        print(f"ID {a.id}: {start} -> {mid} (target {target}, {math.degrees(a.rad):+.0f} deg) -> back {end}")


if __name__ == "__main__":
    main()
