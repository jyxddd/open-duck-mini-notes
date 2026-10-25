"""
Drive the walk policy from the keyboard over SSH (no Xbox controller needed).

Same idea as scripts/fc_test.py: RLWalk runs with commands=False and we write
rl_walk.last_commands ourselves.

Usage (from ~/Open_Duck_Mini_Runtime/scripts, in an SSH terminal):
  python v2_rl_walk_keyboard.py --onnx_model_path BEST_WALK_ONNX_2.onnx
  python v2_rl_walk_keyboard.py --fake      # no robot: only try the keys
  (or just ~/walk.sh)
Every policy step is recorded to ~/walk_logs/walk_<time>.pkl on quit
(--no_log to skip). Defaults = the settings that walk best on our duck
(2026-10-25): --kp 20, --yaw_trim 0.1, gyro heading hold on; servo
register 85 must be 0 (tools/servo_regs.py).

Fall guard (on by default, --fall_deg 40, --no_fall_guard to disable): when the
trunk tilts more than fall_deg the policy stops and ALL servos are released
(torque off), so they don't stall against the floor. Then: pick the duck up,
put it on the stand, press r (torque back on, init pose), put it on the floor
and press p.

Keys (lowercase):
  p        start / pause (starts PAUSED)
  w / s    forward / backward      (+-0.05 m/s, max 0.15)
  a / d    step left / right       (+-0.05 m/s, max 0.2)
  q / e    turn left / right       (+-0.25 rad/s, max 1.0)
  z / c    yaw trim left / right   (+-0.05 rad/s; only added while walking,
           cancels a drift; start value: --yaw_trim)
  g        heading hold on / off   (on by default: while walking straight the
           gyro keeps the duck on the heading it had when it started walking)
  space    stop walking (all speeds to 0, keeps balancing)
  i / k    head up / down
  j / l    head left / right
  h        head back to center
  r        after a fall: torque on again + init pose (duck on the stand!)
  x        quit (motors keep their last target; run turn_off.py to release)
"""

import argparse
import os
import pickle
import select
import sys
import termios
import time
import tty
from threading import Thread

import numpy as np

from fall_guard import FallGuard

HOME_DIR = os.path.expanduser("~")

# Same ranges as mini_bdx_runtime/xbox_controller.py
X_RANGE = (-0.15, 0.15)
Y_RANGE = (-0.2, 0.2)
YAW_RANGE = (-1.0, 1.0)
HEAD_PITCH_RANGE = (-0.78, 0.3)
HEAD_YAW_RANGE = (-0.5, 0.5)

STEP_LIN = 0.05
STEP_YAW = 0.25
STEP_HEAD = 0.1

# key -> (command index, step, range)
MOVES = {
    "w": (0, STEP_LIN, X_RANGE),
    "s": (0, -STEP_LIN, X_RANGE),
    "a": (1, STEP_LIN, Y_RANGE),
    "d": (1, -STEP_LIN, Y_RANGE),
    "q": (2, STEP_YAW, YAW_RANGE),
    "e": (2, -STEP_YAW, YAW_RANGE),
    "i": (4, -STEP_HEAD, HEAD_PITCH_RANGE),
    "k": (4, STEP_HEAD, HEAD_PITCH_RANGE),
    "j": (5, STEP_HEAD, HEAD_YAW_RANGE),
    "l": (5, -STEP_HEAD, HEAD_YAW_RANGE),
}


def apply_key(key, commands, paused):
    """Returns (commands, paused, quit). commands is a list of 7 floats."""
    commands = list(commands)
    key = key.lower()
    if key in MOVES:
        idx, step, (lo, hi) = MOVES[key]
        commands[idx] = float(np.clip(round(commands[idx] + step, 3), lo, hi))
    elif key == " ":
        commands[0:3] = [0.0, 0.0, 0.0]
    elif key == "h":
        commands[3:7] = [0.0, 0.0, 0.0, 0.0]
    elif key == "p":
        paused = not paused
    elif key in ("x", "\x03"):  # x or Ctrl+C
        return commands, paused, True
    return commands, paused, False


TRIM_STEP = 0.05
TRIM_RANGE = (-0.5, 0.5)


def apply_trim(commands, trim):
    """Commands actually sent: the yaw trim is added only while walking,
    so a standing duck doesn't start turning on the spot."""
    sent = list(commands)
    if commands[0] != 0.0 or commands[1] != 0.0:
        sent[2] = float(np.clip(commands[2] + trim, *YAW_RANGE))
    return sent


class HeadingHold:
    """Keeps the walking direction with the gyro (runs in the policy thread).

    Heading = integrated gyro z (+ = turned left). While walking with no turn
    key pressed, the turn command becomes trim + k * (target - heading); the
    target is the heading at the moment straight walking started, so pressing
    q/e and letting go simply makes the new direction the target.
    The policy turns ~1.2-1.8 rad/s per rad/s of command (measured), so
    k = 0.6 gives a ~1 s correction without fighting the gait wobble.
    The integral term learns the drift (a self-tuning trim) and removes the
    heading offset a pure P term would leave; it is kept between walks.
    """

    def __init__(self, k=0.6, ki=0.25, limit=0.4, i_limit=0.3):
        self.k = k
        self.ki = ki
        self.limit = limit
        self.i_limit = i_limit
        self.heading = 0.0
        self.target = None
        self.integral = 0.0  # rad*s, ki * integral = learned trim (rad/s)
        self.last_t = None
        self.dt = 0.02

    def update(self, gyro_z, t):
        if self.last_t is not None:
            self.dt = min(max(t - self.last_t, 0.0), 0.05)
            self.heading += gyro_z * self.dt
        self.last_t = t

    def command(self, commands, trim, enabled):
        sent = list(commands)
        walking = commands[0] != 0.0 or commands[1] != 0.0
        if not walking or commands[2] != 0.0 or not enabled:
            self.target = None
            return apply_trim(commands, trim)
        if self.target is None:
            self.target = self.heading
        err = self.target - self.heading
        lim = self.i_limit / self.ki
        self.integral = float(np.clip(self.integral + err * self.dt, -lim, lim))
        turn = trim + self.k * err + self.ki * self.integral
        sent[2] = float(np.clip(turn, -self.limit, self.limit))
        return sent


def status_line(commands, paused, fallen=False, trim=0.0, hold=False):
    state = "FALLEN " if fallen else "PAUSED " if paused else "RUNNING"
    return (
        f"\r[{state}] fwd {commands[0]:+.2f}  side {commands[1]:+.2f}  "
        f"turn {commands[2]:+.2f}  trim {trim:+.2f}  hold {'ON ' if hold else 'OFF'}  "
        f"head pitch {commands[4]:+.1f} yaw {commands[5]:+.1f}   "
    )


class FakeWalk:
    def __init__(self):
        self.last_commands = [0.0] * 7
        self.paused = True

    def run(self):
        while True:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, default="BEST_WALK_ONNX_2.onnx")
    parser.add_argument("--duck_config_path", type=str, default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--fake", action="store_true", help="no robot, only test the keys")
    parser.add_argument("--log", action="store_true", default=True,
                        help="record obs/targets of every step (default on)")
    parser.add_argument("--no_log", dest="log", action="store_false")
    parser.add_argument("--fall_deg", type=float, default=40, help="tilt that counts as a fall")
    parser.add_argument("--no_fall_guard", action="store_true")
    # 2026-10-25: with servo reg 85 = 0 (no accel limit) the runtime's kp 30
    # made a 5 Hz shiver on the floor; 20 is stable
    parser.add_argument("--kp", type=int, default=20, help="servo P for the legs (head stays 8)")
    parser.add_argument("--kd", type=int, default=0, help="servo D for all joints (damping)")
    parser.add_argument("--yaw_trim", type=float, default=0.1,
                        help="rad/s added to the turn command while walking (+ = left)")
    parser.add_argument("--no_heading_hold", action="store_true",
                        help="start with the gyro heading hold off (key g toggles)")
    args = parser.parse_args()

    if args.fake:
        rl_walk = FakeWalk()
    else:
        from v2_rl_walk_mujoco import RLWalk

        rl_walk = RLWalk(
            args.onnx_model_path,
            duck_config_path=args.duck_config_path,
            commands=False,
            cutoff_frequency=args.cutoff_frequency,
            pid=[args.kp, 0, args.kd],
        )
    rl_walk.paused = True
    rl_walk.last_commands = [0.0] * 7

    log = []
    guard = None if args.no_fall_guard else FallGuard(tilt_deg=args.fall_deg)
    fall = {"fallen": False, "tilt": 0.0, "t": 0.0}
    # shared with the policy thread: what the keys asked for
    ctl = {"cmd": [0.0] * 7, "trim": args.yaw_trim, "hold": not args.no_heading_hold}
    hold = HeadingHold()
    if not args.fake:
        # wrap get_obs (runs in the policy thread): logging + fall guard.
        # RLWalk's own save_obs only dumps when run() returns, which never
        # happens in the daemon thread.
        get_obs = rl_walk.get_obs

        def guarded_get_obs():
            obs = get_obs()
            if obs is None:
                return None
            if args.log:
                # motor_targets here are the ones sent on the previous step
                log.append((time.time(), obs.copy()))
            if guard is not None and guard.update(obs[3:6]):
                # same thread as set_position_all -> no bus collision
                rl_walk.paused = True
                rl_walk.last_commands = [0.0] * 7
                rl_walk.hwi.turn_off()
                ctl["cmd"] = [0.0] * 7
                fall.update(fallen=True, tilt=guard.tilt_deg, t=time.time())
                return None  # run() skips this step, then sees paused
            hold.update(obs[2], time.time())
            # used from the next step on (this obs already holds the old commands)
            rl_walk.last_commands = hold.command(ctl["cmd"], ctl["trim"], ctl["hold"])
            return obs

        rl_walk.get_obs = guarded_get_obs

    def recover():
        """Torque on again and back to the init pose after a fall."""
        hwi = rl_walk.hwi
        ids = list(hwi.joints.values())
        hwi.io.set_kps(ids, hwi.low_torque_kps)  # soft, so nothing snaps
        if hasattr(hwi.io, "enable_torque"):
            hwi.io.enable_torque(ids)
        hwi.turn_on()  # low kp -> init pose -> normal kp (~2 s)
        # forget what the policy did before the fall
        for name in ("last_action", "last_last_action", "last_last_last_action"):
            setattr(rl_walk, name, np.zeros(rl_walk.num_dofs))
        rl_walk.motor_targets = np.array(rl_walk.init_pos)
        rl_walk.prev_motor_targets = np.array(rl_walk.init_pos)
        if guard is not None:
            guard.reset()
        fall["fallen"] = False

    Thread(target=rl_walk.run, daemon=True).start()

    print(__doc__.split("Keys (lowercase):")[1])
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        def show():
            sys.stdout.write(status_line(ctl["cmd"], rl_walk.paused, fall["fallen"],
                                         ctl["trim"], ctl["hold"]))
            sys.stdout.flush()

        show()
        shown_fall = 0.0
        while True:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if fall["fallen"] and fall["t"] != shown_fall:
                shown_fall = fall["t"]
                print(
                    f"\n*** 摔倒 (倾斜 {fall['tilt']:.0f}°)：策略已停，所有电机已卸力 ***"
                    "\n    扶起鸭子放到支架上 -> 按 r 重新上力 -> 放到地上 -> 按 p"
                )
                show()
            if not ready:
                continue
            # os.read, not sys.stdin.read: the buffered reader would swallow
            # queued keys and select() would never report them
            key = os.read(fd, 1).decode(errors="ignore")
            if not key:
                break
            if fall["fallen"] and key.lower() == "r":
                print("\n重新上力中 (约 2 秒) ...")
                recover()
                print("已回到初始姿态。放到地上后按 p 开始。")
                show()
                continue
            if fall["fallen"] and key.lower() == "p":
                print("\n摔倒后要先按 r 重新上力（鸭子先放在支架上）")
                continue
            if key.lower() in ("z", "c"):
                step = TRIM_STEP if key.lower() == "z" else -TRIM_STEP
                ctl["trim"] = float(np.clip(round(ctl["trim"] + step, 3), *TRIM_RANGE))
            elif key.lower() == "g":
                ctl["hold"] = not ctl["hold"]
            else:
                cmd, paused, quit_ = apply_key(key, ctl["cmd"], rl_walk.paused)
                if quit_:
                    break
                ctl["cmd"] = cmd
                rl_walk.paused = paused
            # send now too (the policy thread refines it every step while running)
            rl_walk.last_commands = apply_trim(ctl["cmd"], ctl["trim"])
            show()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        rl_walk.last_commands = [0.0] * 7
        rl_walk.paused = True
        if log:
            time.sleep(0.2)
            log_dir = os.path.join(HOME_DIR, "walk_logs")
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, time.strftime("walk_%Y%m%d_%H%M%S.pkl"))
            with open(path, "wb") as f:
                pickle.dump({"onnx": args.onnx_model_path, "steps": log}, f)
            print(f"\nsaved {len(log)} steps to {path}")
        print("\nSTOPPED (motors keep their last target; run turn_off.py to release torque)")


if __name__ == "__main__":
    main()
