"""
Run a Standing policy (Open_Duck_Playground --env standing) on the real robot.

Differences from v2_rl_walk_mujoco.py, matching playground/open_duck_mini_v2/standing.py:
  - obs has 85 dims: no motor_targets and no imitation_phase
  - lin vel commands are always 0, only head commands (neck_pitch, head_pitch, head_yaw, head_roll)
  - the policy tracks the head commands itself, so they are NOT added to the head motor targets

Usage (from ~/Open_Duck_Mini_Runtime/scripts):
  python v2_rl_stand.py --onnx_model_path standing.onnx            # real robot
  python v2_rl_stand.py --onnx_model_path standing.onnx --fake_hw  # no motors/IMU, test the loop only
  python v2_rl_stand.py --onnx_model_path standing.onnx --dry_run  # read sensors, never move motors
"""

import argparse
import os
import pickle
import time

import numpy as np

from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter

HOME_DIR = os.path.expanduser("~")

OBS_SIZE = 85
NUM_DOFS = 14
HEAD_SLICE = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll

# Same order and values as the "home" keyframe in the Playground xml
INIT_POS = np.array(
    [
        0.002, 0.053, -0.63, 1.368, -0.784,  # left leg
        0.0, 0.0, 0.0, 0.0,  # head
        -0.003, -0.065, 0.635, 1.379, -0.796,  # right leg
    ]
)

# Head command ranges seen during training (standing.py default_config)
HEAD_CMD_LOW = np.array([-0.34, -0.78, -2.7, -0.5])
HEAD_CMD_HIGH = np.array([1.1, 0.78, 2.7, 0.5])


class RLStand:
    def __init__(
        self,
        onnx_model_path,
        duck_config_path=f"{HOME_DIR}/duck_config.json",
        serial_port="/dev/ttyACM0",
        control_freq=50,
        kp=30,
        head_kp=8,
        action_scale=0.25,
        use_controller=True,
        fake_hw=False,
        dry_run=False,
        no_turn_on=False,
        ramp_s=0.0,
        no_feet=False,
        save_obs=False,
        cutoff_frequency=None,
    ):
        self.control_freq = control_freq
        self.ramp_s = ramp_s  # fade policy actions in over the first ramp_s seconds
        self.action_scale = action_scale
        self.fake_hw = fake_hw
        self.dry_run = dry_run or fake_hw
        self.no_feet = no_feet or fake_hw
        self.save_obs = save_obs
        self.saved_obs = []

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        in_shape = self.policy.ort_session.get_inputs()[0].shape
        if in_shape[-1] != OBS_SIZE:
            raise SystemExit(
                f"Model expects obs of size {in_shape[-1]}, standing obs is {OBS_SIZE}. "
                "Use v2_rl_walk_mujoco.py for walk models (101)."
            )

        self.duck_config = DuckConfig(config_json_path=duck_config_path)
        self.init_pos = INIT_POS.copy()

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(control_freq, cutoff_frequency)

        self.hwi = None
        if not fake_hw:
            from mini_bdx_runtime.rustypot_position_hwi import HWI

            self.hwi = HWI(self.duck_config, serial_port)
            assert list(self.hwi.init_pos.values()) == list(self.init_pos), (
                "HWI init_pos differs from the training home pose"
            )
            if not self.dry_run:
                kps = [kp] * NUM_DOFS
                kps[HEAD_SLICE] = [head_kp] * 4
                self.hwi.set_kds([0] * NUM_DOFS)
                if no_turn_on:
                    # Robot already standing on the ground in the init pose:
                    # skip the low-kp phase of turn_on (it would let the legs fold)
                    self.hwi.set_position_all(dict(zip(self.hwi.joints, self.init_pos)))
                    self.hwi.set_kps(kps)
                else:
                    self.hwi.set_kps(kps)
                    self.hwi.turn_on()
                    time.sleep(2)

        self.imu = None
        if not fake_hw:
            from mini_bdx_runtime.raw_imu import Imu

            self.imu = Imu(
                sampling_freq=int(control_freq),
                upside_down=self.duck_config.imu_upside_down,
            )

        self.feet_contacts = None
        if not self.no_feet:
            from mini_bdx_runtime.feet_contacts import FeetContacts

            self.feet_contacts = FeetContacts()

        self.xbox = None
        if use_controller:
            try:
                from mini_bdx_runtime.xbox_controller import XBoxController

                self.xbox = XBoxController(20, only_head_control=True)
            except Exception as e:
                print(f"No Xbox controller ({e}), head commands stay at 0")

        self.commands = np.zeros(7)
        self.last_action = np.zeros(NUM_DOFS)
        self.last_last_action = np.zeros(NUM_DOFS)
        self.last_last_last_action = np.zeros(NUM_DOFS)
        self.motor_targets = self.init_pos.copy()
        self.paused = self.duck_config.start_paused

    def read_state(self):
        if self.fake_hw:
            return (
                np.zeros(3),
                np.array([0.0, 0.0, 9.81]),
                self.motor_targets.copy(),
                np.zeros(NUM_DOFS),
            )

        imu_data = self.imu.get_data()
        dof_pos = self.hwi.get_present_positions(ignore=["left_antenna", "right_antenna"])
        dof_vel = self.hwi.get_present_velocities(ignore=["left_antenna", "right_antenna"])
        if dof_pos is None or dof_vel is None:
            return None
        if len(dof_pos) != NUM_DOFS or len(dof_vel) != NUM_DOFS:
            print(f"ERROR expected {NUM_DOFS} joints, got {len(dof_pos)} pos / {len(dof_vel)} vel")
            return None
        return imu_data["gyro"], imu_data["accelero"], dof_pos, dof_vel

    def get_obs(self):
        state = self.read_state()
        if state is None:
            return None
        gyro, accelero, dof_pos, dof_vel = state

        feet = [1.0, 1.0] if self.no_feet else self.feet_contacts.get()

        return np.concatenate(
            [
                gyro,  # 3
                accelero,  # 3
                self.commands,  # 7
                dof_pos - self.init_pos,  # 14
                dof_vel * 0.05,  # 14
                self.last_action,  # 14
                self.last_last_action,  # 14
                self.last_last_last_action,  # 14
                feet,  # 2
            ]
        ).astype(np.float32)

    def update_commands(self):
        if self.xbox is None:
            return
        cmds, buttons, _, _ = self.xbox.get_last_command()
        self.commands[:3] = 0.0
        self.commands[3:] = np.clip(cmds[3:], HEAD_CMD_LOW, HEAD_CMD_HIGH)
        if buttons.A.triggered:
            self.paused = not self.paused
            print("PAUSE" if self.paused else "UNPAUSE")

    def run(self, max_seconds=None):
        mode = "FAKE HW" if self.fake_hw else ("DRY RUN (motors not driven)" if self.dry_run else "REAL")
        print(f"Starting standing policy [{mode}] at {self.control_freq} Hz, Ctrl+C to stop")
        start_t = time.time()
        loop_times = []
        i = 0
        try:
            while max_seconds is None or time.time() - start_t < max_seconds:
                t = time.time()
                self.update_commands()

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue
                if self.save_obs:
                    self.saved_obs.append(obs)

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # Soft start: scale the applied action from 0 to 1 over ramp_s seconds.
                # The policy still sees its raw action as last_action, like in training.
                gain = 1.0
                if self.ramp_s > 0:
                    gain = min(1.0, (time.time() - start_t) / self.ramp_s)
                self.motor_targets = self.init_pos + gain * action * self.action_scale

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered = self.action_filter.get_filtered_action()
                    if time.time() - start_t > 1:  # give time to the filter to stabilize
                        self.motor_targets = filtered

                if not self.dry_run:
                    self.hwi.set_position_all(
                        make_action_dict(self.motor_targets, list(self.hwi.joints.keys()))
                    )

                took = time.time() - t
                loop_times.append(took)
                if took > 1 / self.control_freq:
                    print(f"Policy control budget exceeded by {took - 1 / self.control_freq:.3f}s")
                if self.dry_run and i % self.control_freq == 0:
                    print(
                        f"t={time.time() - start_t:5.1f}s  loop={took * 1000:5.1f}ms  "
                        f"targets={np.around(self.motor_targets, 2)}"
                    )
                i += 1
                time.sleep(max(0, 1 / self.control_freq - took))
        except KeyboardInterrupt:
            pass
        finally:
            if self.feet_contacts is not None:
                self.feet_contacts.stop()
            if loop_times:
                lt = np.array(loop_times) * 1000
                print(f"Loop time: mean {lt.mean():.1f}ms, max {lt.max():.1f}ms over {len(lt)} steps")
            if self.save_obs:
                pickle.dump(self.saved_obs, open("robot_saved_obs_standing.pkl", "wb"))
                print("Saved robot_saved_obs_standing.pkl")
            print("STOPPED (motors keep their last target; run turn_off.py to release torque)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("--duck_config_path", type=str, default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", "--kp", type=int, default=30)
    parser.add_argument("--head_kp", type=int, default=8)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--no_controller", action="store_true", help="head commands stay at 0")
    parser.add_argument("--no_feet", action="store_true", help="feet switches not wired, report both feet on the ground")
    parser.add_argument("--dry_run", action="store_true", help="read sensors and run the policy, never move motors")
    parser.add_argument("--fake_hw", action="store_true", help="no motors, IMU or feet switches; tests the loop only")
    parser.add_argument("--ramp_s", type=float, default=0.0, help="soft start: fade actions in over this many seconds")
    parser.add_argument("--no_turn_on", action="store_true", help="skip the low-torque turn_on (robot already in init pose on the ground)")
    parser.add_argument("--max_seconds", type=float, default=None)
    parser.add_argument("--save_obs", action="store_true")
    args = parser.parse_args()

    RLStand(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        control_freq=args.control_freq,
        kp=args.kp,
        head_kp=args.head_kp,
        action_scale=args.action_scale,
        use_controller=not args.no_controller,
        fake_hw=args.fake_hw,
        dry_run=args.dry_run,
        no_turn_on=args.no_turn_on,
        ramp_s=args.ramp_s,
        no_feet=args.no_feet,
        save_obs=args.save_obs,
        cutoff_frequency=args.cutoff_frequency,
    ).run(max_seconds=args.max_seconds)
