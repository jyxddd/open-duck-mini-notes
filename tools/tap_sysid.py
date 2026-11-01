"""
Sim replay of a tap_test.py ankle step: how late is the real IMU, and does
the sim body ring like the real one?

The targets the Pi sent (tap log "target", sent just before that row's
reading) are played open-loop into the replay_sysid sim after a settle in the
init pose. The real gyro is then compared with the sim gyro shifted by 0..8
control steps; the best shift is the IMU lag. Joint positions are compared the
same way (they should match at lag 0).

    python tap_sysid.py --log tap_20261030_204115.pkl
    python tap_sysid.py --log ... --kp_legs 16 --backlash 1     # sim knobs as in replay_sysid
Run from ~/microduck_rl (bam + the Open Duck XML).
"""

import argparse
import math
import pickle

import numpy as np

from replay_sysid import simulate

ANKLES = (4, 13)


def best_lag(real, sim, max_lag=8):
    """rms error of real[i] vs sim[i - lag] for each lag (real later = positive lag)."""
    out = []
    for lag in range(max_lag + 1):
        r = real[lag:]
        s = sim[: len(sim) - lag]
        out.append(math.sqrt(((r - s) ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--start", type=float, default=9.5, help="s, window start (step is at --step_at in the log args)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--vin", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--kp_legs", type=float, default=20.0)
    ap.add_argument("--kp_head", type=float, default=8.0)
    ap.add_argument("--backlash", type=float, default=0.0)
    ap.add_argument("--hidden_play", action="store_true")
    ap.add_argument("--flex", type=float, default=0.0)
    ap.add_argument("--foot_solref", default="")
    ap.add_argument("--friction", type=float)
    ap.add_argument("--joint_damping", type=float, default=0.0)
    ap.add_argument("--table", action="store_true", help="print real vs sim pitch gyro per step")
    a = ap.parse_args()

    d = pickle.load(open(a.log, "rb"))
    t = d["t"]
    i0 = int(np.searchsorted(t, a.start))
    n = int(a.seconds * 50)
    gyro, pos, target = d["gyro"][i0:i0 + n], d["pos"][i0:i0 + n], d["target"][i0:i0 + n]
    n = len(target)
    sim_g, sim_q = simulate(a, target, n)

    g_err = best_lag(gyro[:, 1], sim_g[:, 1])
    q_err = best_lag(pos[:, list(ANKLES)].mean(1), sim_q[:, list(ANKLES)].mean(1))
    lag = int(np.argmin(g_err))
    print("lag (steps of 20 ms):  " + " ".join(f"{k:6d}" for k in range(len(g_err))))
    print("pitch gyro rms err:    " + " ".join(f"{e:6.3f}" for e in g_err) + f"   -> best {lag} ({lag * 20} ms)")
    print("ankle pos rms err deg: " + " ".join(f"{math.degrees(e):6.2f}" for e in q_err) + f"   -> best {int(np.argmin(q_err))}")
    print(f"peak |pitch gyro|  real {np.abs(gyro[:, 1]).max():.2f}  sim {np.abs(sim_g[:, 1]).max():.2f} rad/s")
    if a.table:
        print("  t    tgt  | real ankle sim ankle | real gyro  sim gyro (shifted by best lag)")
        for i in range(n):
            s = sim_g[i - lag, 1] if i >= lag else float("nan")
            print(f"{t[i0 + i]:5.2f} {math.degrees(target[i, 4] - target[0, 4]):+4.1f} | "
                  f"{math.degrees(pos[i, 4] - target[0, 4]):+6.1f} {math.degrees(sim_q[i, 4] - target[0, 4]):+6.1f} | "
                  f"{gyro[i, 1]:+.3f} {s:+.3f}")


if __name__ == "__main__":
    main()
