"""
Control test: each servo in turn moves a few degrees from WHERE IT IS NOW and back,
then torque is released again. No servo is sent to 2048, so nothing makes a big jump.

Usage: python wiggle_all.py [/dev/ttyACM0] [--deg 5]
"""
import argparse
import time

import serial

IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)
ADDR_TORQUE, ADDR_GOAL, ADDR_POS = 40, 42, 56


def checksum(body):
    return (~sum(body)) & 0xFF


def write(s, sid, addr, data):
    body = [sid, len(data) + 3, 0x03, addr, *data]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, checksum(body)]))
    s.flush()
    time.sleep(0.01)
    s.read(32)


def read_pos(s, sid):
    body = [sid, 4, 0x02, ADDR_POS, 2]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, checksum(body)]))
    s.flush()
    time.sleep(0.01)
    r = s.read(32)
    i = r.find(bytes([0xFF, 0xFF, sid]))
    if i < 0 or len(r) < i + 8:
        return None
    return r[i + 5] | (r[i + 6] << 8)


def goal(s, sid, pos, speed=300):
    pos = max(0, min(4095, pos))
    write(s, sid, ADDR_GOAL, [pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port", nargs="?", default="/dev/ttyACM0")
    p.add_argument("--deg", type=float, default=5.0)
    p.add_argument("--ids", type=int, nargs="*", default=list(IDS))
    a = p.parse_args()
    step = round(a.deg * 4096 / 360)

    with serial.Serial(a.port, 1000000, timeout=0.03) as s:
        for sid in a.ids:
            start = read_pos(s, sid)
            if start is None:
                print(f"ID {sid}: NOT FOUND")
                continue
            goal(s, sid, start)  # hold where it is, no jump
            write(s, sid, ADDR_TORQUE, [1])
            time.sleep(0.3)
            goal(s, sid, start + step)
            time.sleep(0.8)
            moved = read_pos(s, sid)
            goal(s, sid, start)
            time.sleep(0.8)
            back = read_pos(s, sid)
            write(s, sid, ADDR_TORQUE, [0])
            ok = moved is not None and back is not None and abs(moved - (start + step)) < 25 and abs(back - start) < 25
            print(f"ID {sid}: {start} -> {moved} (target {start + step}) -> {back}   {'OK' if ok else 'CHECK'}")
            time.sleep(0.3)


if __name__ == "__main__":
    main()
