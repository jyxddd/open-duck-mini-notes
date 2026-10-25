"""
Read / back up / change STS3215 tuning registers on all 14 servos.

  python servo_regs.py                      # read only: table of the registers below
  python servo_regs.py --backup regs.json   # read + save to json
  python servo_regs.py --set_amax 254       # write EEPROM reg 85 (max acceleration) on all ids
  python servo_regs.py --restore regs.json  # write back reg 85 from a backup

Nothing else is written. Run with no other program holding /dev/ttyACM0
(walk scripts stopped; battery-guard skips its read while the port is busy).

Registers (Feetech STS memory table):
  21/22/23  P / D / I gain          (runtime sets P itself: legs 30, head 8)
  41        acceleration (RAM)      0 = no ramp
  84        max speed   (EEPROM, "Vmax")
  85        max accel   (EEPROM, "Amax") <- factory value makes motion sluggish
            (@devemin: 85=254 -> "very smooth"; LeRobot sets it to max;
             Discord DropInQuarters used 85=0)
  55        EEPROM lock (0 = unlocked, 1 = locked)
"""
import argparse
import json
import sys
import time

import serial

IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)
REGS = {"P": 21, "D": 22, "I": 23, "acc41": 41, "vmax84": 84, "amax85": 85, "lock55": 55}
REG_LOCK = 55
REG_AMAX = 85


def checksum(body):
    return (~sum(body)) & 0xFF


def transact(s, sid, instr, params, n_reply):
    body = [sid, len(params) + 2, instr, *params]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, checksum(body)]))
    s.flush()
    time.sleep(0.005)
    r = s.read(6 + n_reply + 8)
    i = r.find(bytes([0xFF, 0xFF, sid]))
    if i < 0 or len(r) < i + 6 + n_reply:
        return None
    pkt = r[i : i + 6 + n_reply]
    if checksum(list(pkt[2:-1])) != pkt[-1]:
        return None
    return list(pkt[5 : 5 + n_reply])


def read_u8(s, sid, addr, tries=3):
    for _ in range(tries):
        v = transact(s, sid, 0x02, [addr, 1], 1)
        if v is not None:
            return v[0]
    return None


def write_u8(s, sid, addr, value):
    return transact(s, sid, 0x03, [addr, value], 0) is not None


def read_all(s):
    out = {}
    for sid in IDS:
        out[str(sid)] = {name: read_u8(s, sid, addr) for name, addr in REGS.items()}
    return out


def print_table(regs):
    names = list(REGS)
    print("ID  " + " ".join(f"{n:>7}" for n in names))
    for sid, v in regs.items():
        print(f"{sid:>2}  " + " ".join(f"{str(v[n]):>7}" for n in names))


def set_amax(s, values):
    """values: {id: amax}. Unlock -> write 85 -> verify -> lock, per servo."""
    ok = True
    for sid, val in values.items():
        write_u8(s, sid, REG_LOCK, 0)
        time.sleep(0.02)
        write_u8(s, sid, REG_AMAX, val)
        time.sleep(0.02)
        back = read_u8(s, sid, REG_AMAX)
        write_u8(s, sid, REG_LOCK, 1)
        time.sleep(0.02)
        lock = read_u8(s, sid, REG_LOCK)
        status = "OK" if back == val and lock == 1 else "FAILED"
        ok &= status == "OK"
        print(f"ID {sid}: reg85 -> {back} (wanted {val}), lock {lock}  {status}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--backup", help="save the current registers to this json file")
    p.add_argument("--set_amax", type=int, help="write reg 85 = this value (0..254) on all ids")
    p.add_argument("--restore", help="write reg 85 back from a backup json")
    a = p.parse_args()

    with serial.Serial(a.port, 1000000, timeout=0.02) as s:
        regs = read_all(s)
        print_table(regs)
        missing = [sid for sid, v in regs.items() if v["amax85"] is None]
        if missing:
            print("NOT READ:", missing, "-> nothing written")
            sys.exit(1)

        if a.backup:
            with open(a.backup, "w") as f:
                json.dump(regs, f, indent=1)
            print("saved", a.backup)

        if a.set_amax is not None:
            if not 0 <= a.set_amax <= 254:
                sys.exit("set_amax must be 0..254")
            if not a.backup:
                sys.exit("refusing to write without --backup (keep the factory values)")
            ok = set_amax(s, {int(sid): a.set_amax for sid in regs})
            print("ALL OK" if ok else "SOME WRITES FAILED")
        elif a.restore:
            old = json.load(open(a.restore))
            ok = set_amax(s, {int(sid): v["amax85"] for sid, v in old.items()})
            print("RESTORED" if ok else "SOME WRITES FAILED")


if __name__ == "__main__":
    main()
