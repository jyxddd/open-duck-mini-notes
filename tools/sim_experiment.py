"""
Headless MuJoCo rehearsal of the real runtime, to compare with robot logs.

Runs BEST_WALK_ONNX_2 (or any walk onnx) in the Playground scene with the SAME
observation as the Pi runtime (no +1.3 accel offset, no speed clip, no filter)
and an optional IMU pitch mounting error, then writes a log in the
v2_rl_walk_keyboard --log format so tools/analyze_walk.py can read it.

Run in WSL from ~/Open_Duck_Playground:
  .venv/bin/python /mnt/d/Claude/duck/tools/sim_experiment.py --imu_pitch_bias 4

imu_pitch_bias > 0: the IMU reports the trunk leaning BACK by that many extra
degrees (what a backward-tilted mount would do).
"""
import argparse
import math
import pickle

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer

# (seconds, forward, side, turn); --schedule "5:0,10:0.1,5:0:0.1,5:-0.1"
SCHEDULE = [(5, 0.0, 0.0, 0.0), (10, 0.05, 0.0, 0.0), (10, 0.10, 0.0, 0.0), (5, 0.0, 0.0, 0.0)]


def parse_schedule(text):
    out = []
    for part in text.split(","):
        v = [float(x) for x in part.split(":")]
        out.append(tuple(v + [0.0] * (4 - len(v))))
    return out


def rot_y(v, deg):
    """Rotate a body-frame vector about y: makes a backward lean look bigger."""
    b = math.radians(deg)
    x, y, z = v
    return np.array([x * math.cos(b) + z * math.sin(b), y, -x * math.sin(b) + z * math.cos(b)])


def trunk_pitch_deg(qpos_base):
    """True trunk pitch from the floating-base quaternion (+ = leaning back)."""
    w, x, y, z = qpos_base[3:7]
    # rotate world up into body frame, same convention as the accelerometer
    up_body = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return math.degrees(math.atan2(up_body[0], up_body[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="BEST_WALK_ONNX_2.onnx")
    p.add_argument("--scene", default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml")
    p.add_argument("--ref", default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    p.add_argument("--imu_pitch_bias", type=float, default=0.0)
    p.add_argument("--out", default="/tmp/sim_walk.pkl")
    p.add_argument("--schedule", help='"sec:fwd:side:turn,..." (default: stand, 0.05, 0.1, stand)')
    p.add_argument("--keep_going", action="store_true", help="don't stop at the first fall")
    a = p.parse_args()
    schedule = parse_schedule(a.schedule) if a.schedule else SCHEDULE

    m = MjInfer(a.scene, a.ref, a.onnx, standing=False)
    steps, true_pitch = [], []
    fell_at = None
    t = 0.0
    sim_steps = 0
    for dur, fwd, side, turn in schedule:
        m.commands = [fwd, side, turn, 0.0, 0.0, 0.0, 0.0]
        for _ in range(int(dur / (m.sim_dt * m.decimation))):
            # policy step (same order as the Pi runtime: obs -> action -> ctrl)
            m.imitation_i = (m.imitation_i + 1.0) % m.PRM.nb_steps_in_period
            ph = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
            m.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            gyro = rot_y(m.get_gyro(m.data), a.imu_pitch_bias)
            acc = rot_y(m.get_accelerometer(m.data), a.imu_pitch_bias)
            obs = np.concatenate([
                gyro, acc, m.commands,
                m.get_actuator_joints_qpos(m.data.qpos) - m.default_actuator,
                m.get_actuator_joints_qvel(m.data.qvel) * m.dof_vel_scale,
                m.last_action, m.last_last_action, m.last_last_last_action,
                m.motor_targets, m.get_feet_contacts(m.data), m.imitation_phase,
            ])
            steps.append((t, obs.copy()))
            base = m.data.qpos[m._floating_base_qpos_addr:m._floating_base_qpos_addr + 7]
            true_pitch.append(trunk_pitch_deg(base))
            action = m.policy.infer(obs)
            m.last_last_last_action = m.last_last_action.copy()
            m.last_last_action = m.last_action.copy()
            m.last_action = action.copy()
            m.motor_targets = m.default_actuator + action * m.action_scale
            m.data.ctrl = m.motor_targets.copy()
            for _ in range(m.decimation):
                mujoco.mj_step(m.model, m.data)
                sim_steps += 1
            t += m.sim_dt * m.decimation
            if fell_at is None and (abs(true_pitch[-1]) > 45 or base[2] < 0.08):
                fell_at = t
        if fell_at is not None and not a.keep_going:
            break

    pickle.dump({"onnx": a.onnx, "steps": steps, "true_pitch": true_pitch}, open(a.out, "wb"))
    tp = np.array(true_pitch)
    stand = tp[: int(5 / 0.02)]
    print(f"bias {a.imu_pitch_bias:+.0f}°: fell at {fell_at if fell_at else '-'} s | "
          f"true pitch (+back) standing mean {stand[50:].mean():+.1f}°, whole run mean {tp.mean():+.1f}° "
          f"range [{tp.min():+.1f}, {tp.max():+.1f}] | log {a.out}")


if __name__ == "__main__":
    main()
