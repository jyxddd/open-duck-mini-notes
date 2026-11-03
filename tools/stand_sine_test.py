"""
No-load leg check, duck on the stand with the legs hanging free (duck-server stopped).
Each leg joint swings around HOME, left and right mirrored at the same time:
a sine at 1 Hz and 2.5 Hz, then a step. Prints per joint lag, gain and step rise time,
left next to right, and saves the raw data to ~/walk_logs/stand_sine_<time>.pkl.

    python stand_sine_test.py [--kp 20]
"""
import argparse
import os
import pickle
import time

import numpy as np
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

DT = 0.02
NAMES = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
MIRROR = [-1, -1, -1, 1, 1]          # right-leg sign for the same motion as the left
AMP = [0.15, 0.15, 0.25, 0.25, 0.25]  # rad; yaw/roll small so the feet don't touch
STEP = [0.1, 0.1, 0.2, 0.2, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=int, default=20)
    args = ap.parse_args()
    hwi = HWI(DuckConfig(config_json_path=os.path.expanduser("~/duck_config.json")))
    names = list(hwi.joints.keys())
    home = np.array([hwi.init_pos[n] for n in names])
    kps = [args.kp] * 14
    kps[5:9] = [8, 8, 8, 8]  # head, as in the walk runtime
    hwi.kps = np.array(kps)
    rows = []
    try:
        hwi.turn_on()
        time.sleep(1.0)

        def run(targets_fn, sec, tag):
            t0 = time.time()
            while time.time() - t0 < sec:
                t = time.time() - t0
                tgt = targets_fn(t)
                hwi.set_position_all(dict(zip(names, tgt)))
                pos = hwi.get_present_positions()
                rows.append((tag, t, tgt.copy(), None if pos is None else pos.copy()))
                time.sleep(max(0.0, DT - (time.time() - t0 - t)))

        for j in range(5):
            li, ri = j, j + 9
            for f in (1.0, 2.5):
                def fn(t, f=f):
                    tgt = home.copy()
                    s = AMP[j] * np.sin(2 * np.pi * f * t)
                    tgt[li] += s
                    tgt[ri] += MIRROR[j] * s
                    return tgt
                run(fn, 3.0, (j, f))
            def step(t):
                tgt = home.copy()
                if t > 0.5:
                    tgt[li] += STEP[j]
                    tgt[ri] += MIRROR[j] * STEP[j]
                return tgt
            run(step, 1.5, (j, "step"))
            run(lambda t: home.copy(), 0.5, (j, "home"))
    finally:
        try:
            hwi.set_position_all(dict(zip(names, home)))
            time.sleep(0.5)
        finally:
            hwi.turn_off()
            print("torque off")

    out = os.path.expanduser(time.strftime("~/walk_logs/stand_sine_%Y%m%d_%H%M%S.pkl"))
    pickle.dump({"rows": rows, "names": names, "home": home, "kp": args.kp}, open(out, "wb"))
    print("saved", out)

    for j in range(5):
        line = []
        for side, idx, sg in (("L", j, 1), ("R", j + 9, MIRROR[j])):
            parts = []
            for f in (1.0, 2.5):
                r = [(t, tg[idx], p[idx]) for tag, t, tg, p in rows if tag == (j, f) and p is not None and t > 0.5]
                t_, x, y = map(np.array, zip(*r))
                best = max(range(12), key=lambda k: np.corrcoef(x[:len(x) - k], y[k:])[0, 1])
                k = best
                g = np.polyfit(x[:len(x) - k] - x.mean(), y[k:] - y.mean(), 1)[0]
                parts.append(f"{f}Hz lag {k * 20:3d}ms gain {g:4.2f}")
            r = [(t, p[idx]) for tag, t, tg, p in rows if tag == (j, "step") and p is not None]
            t_, y = map(np.array, zip(*r))
            y = sg * (y - y[t_ < 0.5].mean())
            final = y[t_ > 1.2].mean()
            rise = t_[np.argmax(y >= 0.9 * STEP[j])] - 0.5 if (y >= 0.9 * STEP[j]).any() else float("nan")
            parts.append(f"step end {np.degrees(final):4.1f}/{np.degrees(STEP[j]):4.1f}deg 90% in {rise * 1000:4.0f}ms")
            line.append(f"{side}: " + ", ".join(parts))
        print(f"{NAMES[j]:9s} " + " | ".join(line))


if __name__ == "__main__":
    main()
