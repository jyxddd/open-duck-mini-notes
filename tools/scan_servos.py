"""
Scan a Feetech STS bus for servos (independent of FD).
Usage: python scan_servos.py [COM3]
Tries common baud rates; pings IDs 0..253 with the PING instruction.
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
BAUDS = [1000000, 500000, 250000, 128000, 115200, 76800, 57600, 38400]


def ping_packet(sid):
    length, instr = 2, 0x01
    chk = (~(sid + length + instr)) & 0xFF
    return bytes([0xFF, 0xFF, sid, length, instr, chk])


def read_pos_packet(sid):
    # READ (0x02) address 56, 2 bytes = present position
    params = [56, 2]
    length = len(params) + 2
    chk = (~(sid + length + 0x02 + sum(params))) & 0xFF
    return bytes([0xFF, 0xFF, sid, length, 0x02, *params, chk])


def scan(baud):
    found = []
    with serial.Serial(PORT, baud, timeout=0.01) as s:
        for sid in range(0, 254):
            s.reset_input_buffer()
            s.write(ping_packet(sid))
            s.flush()
            time.sleep(0.004)
            resp = s.read(64)
            if b"\xff\xff" + bytes([sid]) in resp:
                pos = None
                s.reset_input_buffer()
                s.write(read_pos_packet(sid))
                s.flush()
                time.sleep(0.004)
                r = s.read(64)
                i = r.find(b"\xff\xff" + bytes([sid]))
                if i >= 0 and len(r) >= i + 8:
                    pos = r[i + 5] | (r[i + 6] << 8)
                found.append((sid, pos))
    return found


def main():
    print(f"Port {PORT}")
    try:
        serial.Serial(PORT).close()
    except serial.SerialException as e:
        print(f"Cannot open {PORT}: {e}\n-> In FD click Close first, then run this again.")
        return
    any_found = False
    for baud in BAUDS:
        found = scan(baud)
        if found:
            any_found = True
            for sid, pos in found:
                print(f"  baud {baud}: ID {sid}  position {pos}")
        else:
            print(f"  baud {baud}: nothing")
        if baud == 1000000 and found:
            break
    if not any_found:
        print("No servo answered at any baud rate -> check servo cable / board port / jumper / power.")


if __name__ == "__main__":
    main()
