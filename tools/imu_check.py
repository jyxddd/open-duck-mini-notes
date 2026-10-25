"""
Print the IMU exactly as the runtime sees it (mini_bdx_runtime.raw_imu.Imu, with the
imu_upside_down setting from ~/duck_config.json), averaged over ~1 s.

Simulator convention (imu frame = trunk frame: x forward, y left, z up):
  upright          -> accel ~ ( 0,  0, +9.8)
  nose tilted down -> accel x < 0
  right side down  -> accel y > 0
  turning left     -> gyro z > 0

Usage (from ~/Open_Duck_Mini_Runtime/scripts): python ~/imu_check.py [seconds]
"""
import sys
import time

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.raw_imu import Imu

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
cfg = DuckConfig()
imu = Imu(sampling_freq=50, upside_down=cfg.imu_upside_down)
time.sleep(0.5)
acc, gyr = [], []
t0 = time.time()
while time.time() - t0 < secs:
    d = imu.get_data()
    a, g = np.array(d["accelero"], float), np.array(d["gyro"], float)
    if np.linalg.norm(a) > 1 and np.linalg.norm(a) < 20:  # drop glitches / startup zeros
        acc.append(a)
        gyr.append(g)
    time.sleep(0.02)
acc, gyr = np.array(acc), np.array(gyr)
print(f"upside_down={cfg.imu_upside_down}  samples={len(acc)}")
print("accel mean x,y,z:", np.round(acc.mean(0), 2), " |a|=", round(float(np.linalg.norm(acc.mean(0))), 2))
print("gyro  mean x,y,z:", np.round(gyr.mean(0), 3), " gyro max abs:", np.round(np.abs(gyr).max(0), 2))
