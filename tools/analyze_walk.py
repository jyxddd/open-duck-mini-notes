"""
Analyse a walk log written by v2_rl_walk_keyboard.py --log.

  python analyze_walk.py walk_20261025_150000.pkl [--plot out.png]

Log = {"onnx": ..., "steps": [(t, obs101), ...]}, one entry per policy step
while RUNNING. obs layout (v2_rl_walk_mujoco.get_obs):
  gyro 0:3 | accel 3:6 | commands 6:13 | dof_pos-init 13:27 | dof_vel*0.05 27:41 |
  last_action 41:55 | last_last 55:69 | last_last_last 69:83 |
  motor_targets 83:97 (sent on the previous step) | feet 97:99 | phase 99:101
"""
import argparse
import math
import pickle

import numpy as np

JOINTS = [  # runtime order = hwi.joints order
    ("L_hip_yaw", 20), ("L_hip_roll", 21), ("L_hip_pitch", 22), ("L_knee", 23), ("L_ankle", 24),
    ("neck_pitch", 30), ("head_pitch", 31), ("head_yaw", 32), ("head_roll", 33),
    ("R_hip_yaw", 10), ("R_hip_roll", 11), ("R_hip_pitch", 12), ("R_knee", 13), ("R_ankle", 14),
]
INIT_POS = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0, 0, 0, 0,
                     -0.003, -0.065, 0.635, 1.379, -0.796])
DT = 0.02


def load(path):
    d = pickle.load(open(path, "rb"))
    t = np.array([s[0] for s in d["steps"]])
    obs = np.array([s[1] for s in d["steps"]], dtype=float)
    return d.get("onnx"), t, obs


def segments(t, gap=0.5):
    """Split into continuous RUNNING segments (pauses leave time gaps)."""
    cut = np.where(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), cut)


def tilt(acc):
    """Pitch (+ = nose down / forward) and roll (+ = right side down), degrees.
    IMU frame: x forward, y left, z up (checked on the robot 2026-10-24)."""
    pitch = np.degrees(np.arctan2(-acc[:, 0], acc[:, 2]))
    roll = np.degrees(np.arctan2(acc[:, 1], acc[:, 2]))
    return pitch, roll


def smooth(x, n=5):
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def best_lag(target, pos, max_lag=10):
    """Steps by which pos trails target (max correlation of the motion)."""
    tgt = target - target.mean()
    p = pos - pos.mean()
    if tgt.std() < 1e-3 or p.std() < 1e-3:
        return None, None
    scores = []
    for k in range(max_lag + 1):
        a, b = tgt[: len(tgt) - k], p[k:]
        scores.append(np.corrcoef(a, b)[0, 1])
    k = int(np.argmax(scores))
    return k, scores[k]


def step_freq(contact):
    """Touch-downs per second from a 0/1 contact signal."""
    c = (contact > 0.5).astype(int)
    downs = np.sum(np.diff(c) == 1)
    return downs / (len(c) * DT) if len(c) > 1 else 0.0


def report(onnx, t, obs):
    print(f"model: {onnx}")
    dt = np.diff(t)
    print(f"steps: {len(t)}   running time {len(t) * DT:.1f} s   "
          f"segments: {len(segments(t))}")
    inner = dt[dt < 0.5]
    if len(inner):
        late = np.sum(inner > 0.03)
        print(f"loop: mean {inner.mean() * 1000:.1f} ms, max {inner.max() * 1000:.0f} ms, "
              f"{late} steps > 30 ms ({100 * late / len(inner):.1f}%)  <- dropped/late frames")

    acc = obs[:, 3:6]
    norm = np.linalg.norm(acc, axis=1)
    ok = (norm > 2) & (norm < 30)
    pitch, roll = tilt(acc[ok])
    ps, rs = smooth(pitch), smooth(roll)
    print(f"\ntilt (smoothed): pitch mean {ps.mean():+.1f}° range [{ps.min():+.1f}, {ps.max():+.1f}]  "
          f"(+ = forward)")
    print(f"                 roll  mean {rs.mean():+.1f}° range [{rs.min():+.1f}, {rs.max():+.1f}]  "
          f"(+ = right side down)")
    gyro = obs[:, 0:3]
    print(f"gyro rms (rad/s): roll {np.sqrt(np.mean(gyro[:, 0] ** 2)):.2f}  "
          f"pitch {np.sqrt(np.mean(gyro[:, 1] ** 2)):.2f}  yaw {np.sqrt(np.mean(gyro[:, 2] ** 2)):.2f}")
    if np.any(~ok):
        print(f"bad accel samples skipped: {np.sum(~ok)}")

    cmds = obs[:, 6:9]
    uniq = np.unique(np.round(cmds, 3), axis=0)
    print("\ncommands used (fwd, side, turn):", [tuple(u) for u in uniq])

    feet = obs[:, 97:99]
    both = np.mean((feet[:, 0] > 0.5) & (feet[:, 1] > 0.5))
    none = np.mean((feet[:, 0] < 0.5) & (feet[:, 1] < 0.5))
    print(f"\nfeet: both down {100 * both:.0f}%  left only {100 * np.mean((feet[:, 0] > .5) & (feet[:, 1] < .5)):.0f}%  "
          f"right only {100 * np.mean((feet[:, 0] < .5) & (feet[:, 1] > .5)):.0f}%  none {100 * none:.0f}%")
    print(f"      touch-downs/s: left {step_freq(feet[:, 0]):.2f}  right {step_freq(feet[:, 1]):.2f}")

    pos = obs[:, 13:27]
    target = obs[:, 83:97] - INIT_POS
    print("\njoint tracking  (target = what the policy asked, pos = what the servo did)")
    print(f"{'joint':<12}{'id':>3} {'rms err°':>9} {'max err°':>9} {'lag ms':>7} {'corr':>5} {'pos range°':>11}")
    lags = []
    for j, (name, sid) in enumerate(JOINTS):
        # obs at step i: target sent at step i-1, pos read at step i -> compare directly
        err = np.degrees(target[:, j] - pos[:, j])
        k, c = best_lag(target[:, j], pos[:, j])
        lag = "-" if k is None else f"{(k + 1) * DT * 1000:.0f}"
        if k is not None and j not in (5, 6, 7, 8):
            lags.append((k + 1) * DT * 1000)
        rng = np.degrees(pos[:, j].max() - pos[:, j].min())
        cs = "-" if c is None else f"{c:.2f}"
        print(f"{name:<12}{sid:>3} {np.sqrt(np.mean(err ** 2)):9.1f} {np.abs(err).max():9.1f} "
              f"{lag:>7} {cs:>5} {rng:11.1f}")
    if lags:
        print(f"legs: median lag {np.median(lags):.0f} ms (1 step = 20 ms is the minimum possible here)")


def plot(t, obs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(t)) * DT
    acc = obs[:, 3:6]
    pitch, roll = tilt(acc)
    pos = np.degrees(obs[:, 13:27])
    target = np.degrees(obs[:, 83:97] - INIT_POS)
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(x, smooth(pitch), label="pitch (+fwd)")
    ax[0].plot(x, smooth(roll), label="roll (+right down)")
    ax[0].set_ylabel("deg"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, obs[:, 6], label="fwd cmd"); ax[1].plot(x, obs[:, 7], label="side cmd")
    ax[1].plot(x, obs[:, 8], label="turn cmd"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(x, obs[:, 97] + 0.05, label="left foot"); ax[2].plot(x, obs[:, 98], label="right foot")
    ax[2].legend(); ax[2].grid(alpha=.3)
    for a, js, title in ((ax[3], (2, 3, 4), "left leg"), (ax[4], (11, 12, 13), "right leg")):
        for j in js:
            l, = a.plot(x, pos[:, j], label=f"{JOINTS[j][0]} pos")
            a.plot(x, target[:, j], "--", color=l.get_color(), alpha=.6, label=f"{JOINTS[j][0]} target")
        a.set_ylabel(f"{title} deg (rel. init)"); a.legend(fontsize=7, ncol=3); a.grid(alpha=.3)
    ax[-1].set_xlabel("running time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("plot saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--plot")
    a = p.parse_args()
    onnx, t, obs = load(a.log)
    report(onnx, t, obs)
    if a.plot:
        plot(t, obs, a.plot)
