"""
Make every servo flash its red LED when the battery drops below MIN_V.
Writes "min input voltage" (EEPROM addr 15, unit 0.1 V). LED alarm on voltage
(addr 20 bit0) is already on from the factory; torque-unload on voltage (addr 19 bit0)
is off and is left off, so the duck never goes limp by itself.

Usage: python set_low_voltage_alarm.py [/dev/ttyACM0] [70]
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
MIN_V = int(sys.argv[2]) if len(sys.argv) > 2 else 70  # 7.0 V
IDS = (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33)
ADDR_MIN_V, ADDR_UNLOAD, ADDR_LED, ADDR_LOCK = 15, 19, 20, 55


def wr(s, sid, addr, data):
    body = [sid, len(data) + 3, 0x03, addr, *data]
    s.reset_input_buffer()
    s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
    s.flush()
    time.sleep(0.02)
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
        wr(s, sid, ADDR_LOCK, [0])
        wr(s, sid, ADDR_MIN_V, [MIN_V])
        wr(s, sid, ADDR_LOCK, [1])
        mv, unload, led = rd(s, sid, ADDR_MIN_V, 1), rd(s, sid, ADDR_UNLOAD, 1), rd(s, sid, ADDR_LED, 1)
        ok = mv == [MIN_V] and led and led[0] & 1 and unload and not unload[0] & 1
        print(f"ID {sid}: minV={mv} ledAlarm={led[0]:08b} unload={unload[0]:08b}  {'OK' if ok else 'CHECK'}")
