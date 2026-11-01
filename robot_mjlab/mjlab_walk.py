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
import os
import time

import numpy as np
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.rl_utils import make_action_dict

from v2_rl_walk_mujoco import RLWalk

PLAYGROUND_ACTION_SCALE = 0.25
# mjlab position targets are clipped exactly like training (od3+:
# openduck_constants.JOINT_TARGET_LIMITS, a box around HOME clear of every
# joint stop). od1 had no clip and pressed the sim's knee stop with +2..+10 rad
# actions; on the duck the knees folded.
TARGET_LO = np.array([-0.35, -0.3, -1.1, 0.6, -1.45, -0.3, -0.7, -1.2, -0.4,
                      -0.35, -0.3, -0.3, 0.6, -1.45])
TARGET_HI = np.array([0.35, 0.3, 0.3, 1.75, 0.3, 1.0, 0.7, 1.2, 0.4,
                      0.35, 0.3, 1.1, 1.75, 0.3])


class DuckWalk(RLWalk):
    def __init__(self, onnx_model_path, **kwargs):
        super().__init__(onnx_model_path, **kwargs)
        self.policy_obs = None
        self.set_policy(onnx_model_path)

    def set_policy(self, onnx_model_path):
        policy = OnnxInfer(onnx_model_path, input_name="obs", awd=True)
        dim = policy.ort_session.get_inputs()[0].shape[-1]
        self.mjlab = dim == 61
        self.sit_policy = "sit" in os.path.basename(onnx_model_path)
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
        pos_rel = dof_pos - self.init_pos
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
            targets = np.clip(np.asarray(self.init_pos) + action, TARGET_LO, TARGET_HI)
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
        self.hwi.set_position_all(make_action_dict(targets, list(self.hwi.joints.keys())))

    def run(self):
        dt = 1 / self.control_freq
        while True:
            t = time.time()
            if self.paused:
                time.sleep(0.1)
                continue
            self._step()
            time.sleep(max(0.0, dt - (time.time() - t)))
