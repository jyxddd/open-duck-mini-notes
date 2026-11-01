"""
Fall detection from the accelerometer part of the policy observation.

Tilt = angle between the (low-passed) accelerometer vector and the trunk z axis.
Walking shakes the raw signal a lot, so it is smoothed first and the tilt must
stay above the limit for `hold_s` before a fall is declared.

    guard = FallGuard(tilt_deg=40)
    if guard.update(obs[3:6]):   # obs = gyro(3), accelero(3), ...
        hwi.turn_off()           # release all servos
"""

import math

import numpy as np


class FallGuard:
    def __init__(self, tilt_deg=40.0, hold_s=0.1, freq=50.0, alpha=0.3):
        self.tilt_limit = math.radians(tilt_deg)
        self.hold_steps = max(1, int(round(hold_s * freq)))
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.acc = None
        self.over = 0
        self.tilt = 0.0

    def update(self, accelero):
        """Feed one accelerometer sample (m/s^2, trunk frame). Returns True on a fall."""
        a = np.asarray(accelero, dtype=float)
        norm = np.linalg.norm(a)
        # skip bad reads / free fall (tiny) and I2C glitches or impacts (huge):
        # only the direction of gravity matters here
        if not np.isfinite(norm) or not 2.0 < norm < 30.0:
            return False
        self.acc = a if self.acc is None else self.alpha * a + (1 - self.alpha) * self.acc
        n = np.linalg.norm(self.acc)
        self.tilt = math.acos(float(np.clip(self.acc[2] / n, -1.0, 1.0)))
        self.over = self.over + 1 if self.tilt > self.tilt_limit else 0
        return self.over >= self.hold_steps

    @property
    def tilt_deg(self):
        return math.degrees(self.tilt)
