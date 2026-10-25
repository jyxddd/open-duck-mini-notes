"""
Read-only servo bus error test (no motion, no torque change).
Reads present position / velocity like the runtime does and counts errors per ID.
Usage: python bus_test.py [--n 500] [--port /dev/ttyACM0]
"""
import argparse
import os
import sys
import time

import rustypot

IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14]  # runtime joint order

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500)
p.add_argument("--port", default="/dev/ttyACM0")
a = p.parse_args()

io = rustypot.feetech(a.port, 1000000)

# rustypot prints checksum errors itself -> silence fd 1 during the loop
devnull = os.open(os.devnull, os.O_WRONLY)
saved = os.dup(1)

err_group = {"pos": 0, "vel": 0}
err_single = {i: 0 for i in IDS}
t0 = time.time()
os.dup2(devnull, 1)
try:
    for _ in range(a.n):
        for name, fn in (("pos", io.read_present_position), ("vel", io.read_present_velocity)):
            try:
                fn(IDS)
            except Exception:
                err_group[name] += 1
    t_group = time.time() - t0
    for _ in range(a.n // 5):
        for i in IDS:
            try:
                io.read_present_position([i])
            except Exception:
                err_single[i] += 1
finally:
    sys.stdout.flush()
    os.dup2(saved, 1)

n = a.n
print(f"group reads (all 14 ids): {n} pos + {n} vel in {t_group:.1f}s "
      f"({t_group / n * 1000:.1f} ms per pos+vel)")
print(f"  errors: pos {err_group['pos']} ({100 * err_group['pos'] / n:.1f}%), "
      f"vel {err_group['vel']} ({100 * err_group['vel'] / n:.1f}%)")
print(f"single-id reads ({n // 5} each), errors per id:")
print("  " + "  ".join(f"{i}:{err_single[i]}" for i in IDS))
