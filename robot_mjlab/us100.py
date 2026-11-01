"""
US-100 ultrasonic range finder in UART mode (jumper ON), on the Pi's GPIO UART.

Wiring: VCC -> 3.3 V (pin 17), Trig/Tx -> GPIO15 RXD (pin 10),
Echo/Rx -> GPIO14 TXD (pin 8), GND -> pin 25. Powered at 3.3 V its outputs
are 3.3 V, so no divider is needed. Needs enable_uart=1 and no serial console.

Protocol (9600 8N1): send 0x55 -> 2 bytes, distance in mm (big endian);
send 0x50 -> 1 byte, temperature + 45 (deg C).

  python us100.py            # print the distance 10 times a second
"""
import statistics
import threading
import time

import serial

PORT = "/dev/serial0"
VALID_MM = (20, 4500)


class US100:
    def __init__(self, port=PORT, rate_hz=10):
        self.ser = serial.Serial(port, 9600, timeout=0.1)
        self.rate_hz = rate_hz
        self.distance_m = None      # median of the last 3 good readings
        self.last_ok = 0.0          # time of the last good reading
        self._recent = []
        self._raw = []              # (time, m) of the good readings of the last second
        self.temp_c = self.read_temp_c()  # before the thread owns the port
        threading.Thread(target=self._loop, daemon=True).start()

    def read_mm(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"\x55")
        r = self.ser.read(2)
        if len(r) != 2:
            return None
        mm = r[0] * 256 + r[1]
        return mm if VALID_MM[0] <= mm <= VALID_MM[1] else None

    def read_temp_c(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"\x50")
        r = self.ser.read(1)
        return r[0] - 45 if len(r) == 1 else None

    def _loop(self):
        while True:
            t = time.time()
            try:
                mm = self.read_mm()
            except serial.SerialException:
                mm = None
            if mm is not None:
                self._recent = (self._recent + [mm / 1000])[-3:]
                self._raw = [r for r in self._raw if t - r[0] < 1.0] + [(t, mm / 1000)]
                self.distance_m = statistics.median(self._recent)
                self.last_ok = t
            time.sleep(max(0.0, 1 / self.rate_hz - (time.time() - t)))

    def nearest(self, window_s=0.8):
        """Second smallest reading of the last window_s, or None. Close up the
        real echo comes and goes between phantom 20-26 cm readings that win the
        median (floor test and board test 10-30); two low readings are needed
        so that one stray echo does not stop the duck."""
        now = time.time()
        close = sorted(m for t, m in self._raw if now - t < window_s)
        return close[1] if len(close) > 1 else None

    def fresh(self, max_age_s=0.5):
        """Distance in m if a good reading arrived recently, else None."""
        return self.distance_m if time.time() - self.last_ok < max_age_s else None


if __name__ == "__main__":
    us = US100()
    print("temperature:", us.temp_c, "C")
    while True:
        d = us.fresh()
        print("no reading" if d is None else f"{d * 100:6.1f} cm", flush=True)
        time.sleep(0.1)
