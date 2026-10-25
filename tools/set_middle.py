"""
Make a servo's CURRENT shaft position its new middle (2048).
= FD "中位校准" button (Feetech datasheet 7-14: write 128 to address 40).

Usage (FD must be Closed):
  python set_middle.py --port COM4 --id 10 --release     # torque off, prints position
  (hold the joint in its zero pose)
  python set_middle.py --port COM4 --id 10 --calibrate   # current position -> 2048
  python set_middle.py --port COM4 --id 10 --hold        # go to 2048 and hold (check)
  python set_middle.py --port COM4 --id 10               # read position only
"""
import argparse
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM4")
    p.add_argument("--id", type=int, required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--release", action="store_true")
    g.add_argument("--calibrate", action="store_true")
    g.add_argument("--hold", action="store_true")
    a = p.parse_args()

    with serial.Serial(a.port, 1000000, timeout=0.05) as s:
        pos = read_pos(s, a.id)
        if pos is None:
            print(f"ID {a.id} not found on {a.port}")
            return
        if a.release:
            write(s, a.id, ADDR_TORQUE, [0])
            print(f"ID {a.id}: torque OFF, position {read_pos(s, a.id)}")
        elif a.calibrate:
            write(s, a.id, ADDR_TORQUE, [128])
            time.sleep(0.3)
            after = read_pos(s, a.id)
            ok = after is not None and abs(after - 2048) <= 3
            print(f"ID {a.id}: calibrated, {pos} -> {after}  {'OK' if ok else 'CHECK'}")
        elif a.hold:
            write(s, a.id, ADDR_GOAL, [2048 & 0xFF, 2048 >> 8, 0, 0, 0x2C, 0x01])  # goal 2048, time 0, speed 300
            write(s, a.id, ADDR_TORQUE, [1])
            time.sleep(1.0)
            print(f"ID {a.id}: holding 2048, position {read_pos(s, a.id)}")
        else:
            print(f"ID {a.id}: position {pos}")


if __name__ == "__main__":
    main()
