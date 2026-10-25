"""
Measure how often the BNO055 returns corrupted samples (first byte of a burst read flipped).
Reads raw acceleration and gyro for N samples and reports the spike rate on each axis.
Usage: python imu_glitch_test.py [samples]
"""
import sys
import time

import adafruit_bno055
import board
import busio
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
imu = adafruit_bno055.BNO055_I2C(busio.I2C(board.SCL, board.SDA))
time.sleep(0.5)
A, G = [], []
t0 = time.time()
for _ in range(N):
    a, g = imu.acceleration, imu.gyro
    if a and None not in a and g and None not in g:
        A.append(a)
        G.append(g)
    time.sleep(0.005)
dt = time.time() - t0
A, G = np.array(A), np.array(G)
for name, X, thr in (("accel", A, 0.6), ("gyro", G, 0.07)):
    med = np.median(X, axis=0)
    spikes = (np.abs(X - med) > thr).mean(axis=0) * 100
    print(f"{name}: median {med.round(3)}  spike rate % per axis {spikes.round(1)}")
print(f"{len(A)} samples in {dt:.1f} s ({len(A) / dt:.0f} Hz)")
