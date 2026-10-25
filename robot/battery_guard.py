"""
Battery guard for the duck (runs as a systemd service on the Pi).

Every 30 s, when no robot program is using the servo port, it reads the battery voltage
from the servos (median of a few) and writes it to /home/duck/battery_status.txt.
  < WARN_V : warning in the status file + Pi green LED blinks fast
  < STOP_V for 3 reads in a row : release all servos, then shut the Pi down
    (protects the battery and the SD card)

`python battery_guard.py --once` prints the current battery level and exits.
"""
import argparse
import glob
import os
import statistics
import subprocess
import time

import serial

PORT = "/dev/ttyACM0"
IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)
WARN_V = 7.2
STOP_V = 6.8
STATUS = "/home/duck/battery_status.txt"
LED = "/sys/class/leds/ACT"

# 2S Li-ion, voltage per cell (under light load) -> percent
CURVE = [(3.30, 0), (3.50, 8), (3.60, 15), (3.70, 30), (3.75, 40), (3.80, 50),
         (3.85, 58), (3.90, 66), (3.95, 74), (4.00, 80), (4.10, 92), (4.20, 100)]


def percent(pack_v):
    c = pack_v / 2
    if c <= CURVE[0][0]:
        return 0
    for (v0, p0), (v1, p1) in zip(CURVE, CURVE[1:]):
        if c <= v1:
            return round(p0 + (p1 - p0) * (c - v0) / (v1 - v0))
    return 100


def port_busy():
    """True if another process has the servo port open (robot program running)."""
    me = os.getpid()
    target = os.path.realpath(PORT)
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if int(fd.split("/")[2]) != me and os.path.realpath(fd) == target:
                return True
        except OSError:
            continue
    return False


def _pkt(sid, instr, params):
    body = [sid, len(params) + 2, instr, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def read_voltage():
    vals = []
    with serial.Serial(PORT, 1000000, timeout=0.03) as s:
        for sid in IDS[::3]:
            s.reset_input_buffer()
            s.write(_pkt(sid, 0x02, [62, 1]))
            s.flush()
            time.sleep(0.01)
            r = s.read(32)
            i = r.find(bytes([0xFF, 0xFF, sid]))
            if i >= 0 and len(r) >= i + 6:
                vals.append(r[i + 5] / 10)
    return statistics.median(vals) if vals else None


def release_servos():
    with serial.Serial(PORT, 1000000, timeout=0.03) as s:
        for sid in IDS:
            s.write(_pkt(sid, 0x03, [40, 0]))
            s.flush()
            time.sleep(0.01)


def led(mode):
    try:
        with open(f"{LED}/trigger", "w") as f:
            f.write("timer" if mode == "warn" else "mmc0")
        if mode == "warn":
            with open(f"{LED}/delay_on", "w") as f:
                f.write("100")
            with open(f"{LED}/delay_off", "w") as f:
                f.write("100")
    except OSError:
        pass


def write_status(text):
    with open(STATUS, "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--period", type=float, default=30)
    a = ap.parse_args()

    if a.once:
        if port_busy():
            print("servo port busy (robot program running); last status:")
            print(open(STATUS).read().strip() if os.path.exists(STATUS) else "none yet")
            return
        v = read_voltage()
        print("no servo answered" if v is None else f"battery {v:.1f} V  ~{percent(v)} %")
        return

    low = 0
    while True:
        if not port_busy():
            v = read_voltage()
            if v is not None:
                stamp = time.strftime("%H:%M:%S")
                p = percent(v)
                if v < STOP_V:
                    low += 1
                    write_status(f"{stamp} battery {v:.1f} V ~{p}% CRITICAL ({low}/3)")
                    if low >= 3:
                        write_status(f"{stamp} battery {v:.1f} V CRITICAL -> servos released, shutting down")
                        release_servos()
                        subprocess.run(["shutdown", "-h", "now"])
                        return
                else:
                    low = 0
                    state = "LOW - please charge" if v < WARN_V else "ok"
                    write_status(f"{stamp} battery {v:.1f} V ~{p}% {state}")
                    led("warn" if v < WARN_V else "normal")
        time.sleep(a.period)


if __name__ == "__main__":
    main()
