"""
RLWalk that also runs walk policies trained with mjlab (microduck_rl, task
Mjlab-Velocity-Flat-OpenDuck), and can swap between the two kinds at runtime
(duck_server's model buttons). The kind is read from the ONNX input size.

Playground policies (101-D input): exactly RLWalk's step (gait phase,
target = HOME + 0.25 * action, head command added to the head targets).

mjlab policies (61-D input, normalizer baked in):
  obs = gyro (3, rad/s) | joint pos - HOME (14) | joint vel (14, rad/s)
        | last action (14) | twist cmd fwd, side, turn (3)
        | gravity direction = -accel / |accel| (3)
        | head pose cmd, deltas from HOME (4) | body pose cmd (6, zeros)
  target = HOME + action; the head command goes INTO the observation (the
  policy moves the head itself and balances for it).

Sit/stand policies (61-D too, "sit" in the file name) have the gravity right after
the gyro, and cmd[0] is the sit flag (0 stand, 1 squat):
  obs = gyro | gravity | joint pos - HOME | joint vel | last action | cmd (3)
        | head pose cmd (4) | body pose cmd (6)

get_obs() always returns the 101-D layout, so duck_server's fall guard,
heading hold, logs and tools/analyze_walk.py work for both kinds.
"""
import json
import os
import time

import numpy as np
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.rl_utils import make_action_dict

from v2_rl_walk_mujoco import RLWalk

PLAYGROUND_ACTION_SCALE = 0.25
TRIM_FILE = os.path.expanduser("~/joint_trim.json")
# mjlab position targets are clipped exactly like training (od3+:
# openduck_constants.JOINT_TARGET_LIMITS, a box around HOME clear of every
# joint stop). od1 had no clip and pressed the sim's knee stop with +2..+10 rad
# actions; on the duck the knees folded.
TARGET_LO = np.array([-0.35, -0.3, -1.1, 0.6, -1.45, -0.3, -0.7, -1.2, -0.4,
                      -0.35, -0.3, -0.3, 0.6, -1.45])
TARGET_HI = np.array([0.35, 0.3, 0.3, 1.75, 0.3, 1.0, 0.7, 1.2, 0.4,
                      0.35, 0.3, 1.1, 1.75, 0.3])
# Sit/stand policies squat deeper (openduck_sitstand_env_cfg.SIT_TARGET_LIMITS).
SIT_TARGET_LO = TARGET_LO.copy()
SIT_TARGET_HI = TARGET_HI.copy()
SIT_TARGET_LO[2], SIT_TARGET_HI[11] = -1.35, 1.35  # hip pitch
SIT_TARGET_HI[3], SIT_TARGET_HI[12] = 1.7, 1.7  # knees, the real stop is +1.75
SERVO_PORT = "/dev/ttyACM0"
# 11-02: one garbled reply killed the policy thread mid-walk and left the servos stiff.
# Now a failed step is skipped (the servos hold the last target for 20 ms) and the
# handle reopened; this many in a row pauses and calls on_bus_error.
BUS_ERRORS_MAX = 3
LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]  # hwi.joints order, the head is 5-8


def load_joint_trim():
    """Software zero offsets in rad from ~/joint_trim.json, {"13": -3} in degrees by joint
    index: read as pos + trim, sent as target - trim. Read when a model is chosen."""
    trim = np.zeros(14)
    try:
        with open(TRIM_FILE) as f:
            for k, v in json.load(f).items():
                trim[int(k)] = np.radians(float(v))
    except (OSError, ValueError):
        pass
    return trim


class DuckWalk(RLWalk):
    def __init__(self, onnx_model_path, **kwargs):
        super().__init__(onnx_model_path, **kwargs)
        self.policy_obs = None
        self.on_bus_error = None  # duck_server: release the servos and say so
        self.set_policy(onnx_model_path)

    def set_policy(self, onnx_model_path):
        policy = OnnxInfer(onnx_model_path, input_name="obs", awd=True)
        dim = policy.ort_session.get_inputs()[0].shape[-1]
        self.mjlab = dim == 61
        self.sit_policy = "sit" in os.path.basename(onnx_model_path)
        self.joint_trim = load_joint_trim()
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.policy = policy
        self.onnx_model_path = onnx_model_path

    def get_obs(self):
        if not self.mjlab:
            return super().get_obs()
        imu = self.imu.get_data()
        ignore = ["left_antenna", "right_antenna"]
        dof_pos = self.hwi.get_present_positions(ignore=ignore)
        dof_vel = self.hwi.get_present_velocities(ignore=ignore)
        if dof_pos is None or dof_vel is None or len(dof_pos) != 14 or len(dof_vel) != 14:
            return None
        gyro = np.asarray(imu["gyro"], dtype=np.float64)
        acc = np.asarray(imu["accelero"], dtype=np.float64)
        n = np.linalg.norm(acc)
        gravity = -acc / n if n > 0.1 else np.array([0.0, 0.0, -1.0])
        cmds = np.asarray(self.last_commands, dtype=np.float64)
        pos_rel = dof_pos + self.joint_trim - self.init_pos
        if self.sit_policy:
            parts = [gyro, gravity, pos_rel, dof_vel, self.last_action, cmds[0:3], cmds[3:7], np.zeros(6)]
        else:
            parts = [gyro, pos_rel, dof_vel, self.last_action, cmds[0:3], gravity, cmds[3:7], np.zeros(6)]
        self.policy_obs = np.concatenate(parts).astype(np.float32)
        return np.concatenate([
            gyro, acc, cmds, pos_rel, dof_vel * 0.05, self.last_action,
            self.last_last_action, self.last_last_last_action, self.motor_targets,
            np.zeros(2), np.zeros(2),
        ])

    def _step(self):
        obs = self.get_obs()  # duck_server wraps this (fall guard, commands)
        if obs is None:
            return
        if self.mjlab:
            if self.policy_obs is None:
                return
            action = np.asarray(self.policy.infer(self.policy_obs), dtype=np.float64)
            lo, hi = (SIT_TARGET_LO, SIT_TARGET_HI) if self.sit_policy else (TARGET_LO, TARGET_HI)
            targets = np.clip(np.asarray(self.init_pos) + action, lo, hi)
        else:
            self.imitation_i = (self.imitation_i + 1) % self.PRM.nb_steps_in_period
            ph = self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
            self.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            action = np.asarray(self.policy.infer(obs), dtype=np.float64)
            targets = np.asarray(self.init_pos) + action * PLAYGROUND_ACTION_SCALE
            targets[5:9] = np.asarray(self.last_commands[3:]) + targets[5:9]
        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()
        self.motor_targets = targets
        self.prev_motor_targets = targets.copy()
        sent = targets - self.joint_trim if self.mjlab else targets
        self.hwi.set_position_all(make_action_dict(sent, list(self.hwi.joints.keys())))

    def set_leg_kp(self, kp):
        """Firmware P of the 10 leg servos (the head keeps its own). hwi.turn_on reuses
        hwi.kps, so it also holds after a relax and power_on."""
        kps = np.array(self.hwi.kps, dtype=float)
        kps[LEG_JOINTS] = kp
        self.hwi.set_kps(list(kps))

    def reopen_bus(self):
        """A garbled servo reply makes rustypot panic and poisons its handle (every later
        call panics too): drop it so the port closes, then open a fresh one."""
        import gc

        import rustypot

        self.hwi.io = None
        gc.collect()
        self.hwi.io = rustypot.feetech(SERVO_PORT, 1000000)  # gains stay in the servos

    def run(self):
        dt = 1 / self.control_freq
        errors = 0
        while True:
            t = time.time()
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                self._step()
                errors = 0
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # pyo3 PanicException is not an Exception
                errors += 1
                print(f"[bus] step failed ({errors}): {e}", flush=True)
                try:
                    self.reopen_bus()
                except BaseException as e2:
                    print("[bus] reopen failed:", e2, flush=True)
                if errors >= BUS_ERRORS_MAX:
                    self.paused = True
                    errors = 0
                    if self.on_bus_error is not None:
                        self.on_bus_error(str(e))
            time.sleep(max(0.0, dt - (time.time() - t)))
