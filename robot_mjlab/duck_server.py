"""
Duck command service: keeps the duck balancing and runs simple, safe actions
sent over the network (phone page, voice bridge, PC, AI agent).

Runs at boot (systemd duck-server.service) and waits WITHOUT moving anything.
Phone: http://<pi-ip>:8765  ->  上力 (duck on the stand: legs to the init pose)
-> put it on the floor -> 开始 -> walk / turn / ... -> 关机 when done.

  python duck_server.py              # real robot, port 8765
  python duck_server.py --fake       # no robot: simulated gyro, to test the logic

HTTP API (JSON):
  GET  /          phone remote page
  GET  /status    {"state", "action", "result", "battery_v", "heading_deg", ...}
  POST /cmd       {"action": ...}
     power_on                    torque on + init pose (duck on the stand!); also after a fall
     start                       policy on (duck on the floor)
     stop                        stop moving, keep balancing
     pause                       policy off, motors hold their last position
     relax  on_stand=true        servos limp (duck on the stand); refused without on_stand
     walk   seconds=2 speed=0.1  straight walk (speed 0.05..0.12; backward only with --allow_backward)
     turn   degrees=90           + = left, - = right; done while walking, ends on the gyro angle
     drive  fwd=0.1 turn=0       joystick style; must be repeated within 0.5 s or the duck stops
     head   pitch=-0.3 yaw=0.2   only while standing still (rad, small ranges)
     save_log                    write the last 10 minutes to ~/walk_logs/
     model  name=v6_robust_29M   switch the walk policy (not while running)
     avoid  on=true              obstacle stop (US-100 ultrasonic) on / off
     abtest                      26 s comparison run (AB_PHASES), saved as ~/walk_logs/auto_ab_*.pkl
     shutdown                    servos limp, save the log, power the Pi off

States: off (servos not used, port free) -> paused (holding the init pose)
-> running (balancing / walking) ; fallen ; relaxed.
Safety: fall (> 40 deg) -> all servos released until power_on; every action
has a time limit; one action at a time (a new one replaces the old); low
battery -> no more walking.
"""
import argparse
import collections
import json
import os
import pickle
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from fall_guard import FallGuard
from v2_rl_walk_keyboard import HeadingHold

HOME = os.path.expanduser("~")
PORT_DEV = "/dev/ttyACM0"
DT = 0.02
WALK_SPEED = 0.1          # m/s, below ~0.08 the official policy does not step
SPEED_RANGE = (0.05, 0.15)
# Forward speed used for straight walking (walk / drive up). Our own policies stand
# still at 0.1 on the real duck (they step only when also turning), so they get
# 0.15, the top of their training range. Turning keeps WALK_SPEED (works for all).
STRAIGHT_SPEED = {"BEST_WALK_ONNX_2": 0.1}
STRAIGHT_SPEED_DEFAULT = 0.15
MAX_WALK_S = 10.0
BACK_MAX_S = 1.5
TURN_RATE = 0.4           # rad/s actually sent while turning, same both ways (right -0.4: ~31 deg/s;
                          # left +0.6 turned 43 deg/s and the duck fell when it stopped)
SETTLE_S = 0.5            # walk straight this long after a turn instead of braking at once
DRIVE_DEADMAN_S = 0.5
# Head: the policy never saw head commands in training and the Heima head is
# heavy. A 0.4 rad head-up step tipped the duck over backwards in 2 s, so the
# head moves slowly and only a little (up is the dangerous side: CoM goes back).
HEAD_PITCH_LIMITS = (-0.15, 0.2)    # rad, + = up, - = down (user-checked); -0.4 (down) tipped it over
HEAD_YAW_LIMIT = 0.15             # +0.35 (left) made it turn by itself and fall, twice
HEAD_RATE = 0.12                    # rad/s
NECK_SHARE = 0.7                    # of the pitch to neck_pitch and to head_pitch each
LOW_V, CRITICAL_V = 7.2, 6.9
# same phases as auto_test.py --test ab: (name, seconds, fwd, side, turn); turn sent
# raw (no trim compensation) so the numbers match the PC-run comparison
AB_PHASES = [
    ("stand", 3, 0.0, 0.0, 0.0),
    ("straight", 8, 0.1, 0.0, 0.0),
    ("walk + right -0.25", 6, 0.1, 0.0, -0.25),
    ("walk + left +0.25", 6, 0.1, 0.0, 0.25),
    ("stand", 3, 0.0, 0.0, 0.0),
]
# Models offered on the phone page (the other .onnx files stay on the Pi, unlisted).
# mjlab policies (microduck_rl Open Duck task) are named mj_*; they get no yaw trim:
# the trim compensated the Playground policies' drift, heading hold learns the rest.
MODELS = ["mj_od11_13000", "mj_sit2_1000", "mj_od8_9999", "mj_od8_9000", "mj_od7_8000", "ctrl_206M"]
# Measured on the floor 10-30: od8 at turn 0 veers right ~40 deg/s, about 95 deg/s
# per rad/s of command. Cause not found yet, this only lets it walk straight.
# The default heading-hold limit (0.4) would cut the trim off, so it goes up with it.
YAW_TRIM = {"mj_od8_9000": 0.45, "mj_od8_9999": 0.45}
TRIMMED_HOLD_LIMIT = 0.8
# Models whose walking turns are closed on the gyro. od11's open-loop turn rate
# moved by ~15 deg/s between sessions on the same floor (10-31 afternoon +0.4 ->
# +20 deg/s, evening +0.35 -> -4), so no fixed trim fits. The command is the wanted
# yaw rate plus an integral of its error, kept between turns. Not in place: there
# +0.2 does not turn at all and -0.6 stomps (4 g).
TURN_LOOP_MODELS = {"mj_od11_13000"}
TURN_LOOP_KI = 0.8        # 1/s
TURN_LOOP_LIMIT = 0.5     # rad/s of correction
YAW_RATE_TAU_S = 0.5      # low-pass over the ~3 Hz gait wobble
PAUSE_SETTLE_S = 2.5  # a pause locks the legs where they are: mid-step it tips over
MODELS = [m for m in MODELS if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), m + ".onnx"))]
# The BNO055 can drop off the bus at a hard landing and the stock reader then
# hands out its last sample for ever: the policy walks blind and the fall guard
# is dead, or worse: a last sample taken mid-stumble reads as a fall and the
# guard released a standing duck (floor test 10-30). Repeated samples therefore
# never reach the guard. Live readings never repeat this long.
IMU_FROZEN_S = 0.2
# Obstacle stop: forward walking ends when the US-100 sees something closer
# than this for OBSTACLE_HOLD_S (turning stays allowed, to turn away from it).
# A sighting counts for OBSTACLE_LATCH_S: one missed or far reading must not let
# a held up button walk on (floor test 10-27: a refresh got through at 10 cm).
OBSTACLE_STOP_M = 0.10
OBSTACLE_HOLD_S = 0.2
OBSTACLE_LATCH_S = 0.5
OBSTACLE_CLEAR_M = 0.15   # after a stop, forward is allowed again only beyond this
# Slow-down zone before the stop line: forward speed falls linearly from full at
# OBSTACLE_SLOW_M to OBSTACLE_SLOW_MIN of it at OBSTACLE_STOP_M. Our policies
# barely step below ~0.1 m/s, so the low end mostly means marching in place.
OBSTACLE_SLOW_M = 0.60
OBSTACLE_SLOW_MIN = 0.6
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "duck_remote.html")


def summarize(steps, phases):
    """steps: [(t, obs, phase)] -> one row per phase (same numbers as auto_test.report)."""
    rows = []
    for k, (name, *_rest) in enumerate(phases):
        o = np.array([ob for _, ob, ph in steps if ph == k])
        if len(o) < 10:
            continue
        r = o[min(25, len(o) // 4):]  # skip the first 0.5 s of each phase
        acc = r[:, 3:6]
        pitch = np.degrees(np.arctan2(-acc[:, 0], acc[:, 2]))
        roll = np.degrees(np.arctan2(acc[:, 1], acc[:, 2]))
        rows.append({"phase": name, "sec": round(len(o) * DT, 1),
                     "turned": round(float(np.degrees(o[:, 2]).sum() * DT)),
                     "l_up": round(100 * float(np.mean(r[:, 97] < 0.5))),
                     "r_up": round(100 * float(np.mean(r[:, 98] < 0.5))),
                     "pitch_sd": round(float(pitch.std()), 1), "roll_sd": round(float(roll.std()), 1)})
    return rows


def read_voltage_raw():
    """Servo 13 input voltage (reg 62) with pyserial; the port must be free."""
    import serial

    with serial.Serial(PORT_DEV, 1000000, timeout=0.02) as s:
        body = [13, 4, 2, 62, 1]
        s.reset_input_buffer()
        s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
        s.flush()
        r = s.read(7)
    i = r.find(bytes([0xFF, 0xFF, 13]))
    return r[i + 5] / 10 if i >= 0 and len(r) >= i + 7 else None


class Server:
    def __init__(self, args, make_rl, us=None):
        self.args = args
        self.us = us                      # US100 range finder or None
        self.avoid = us is not None
        self.obstacle_since = None
        self.obstacle_seen = None         # (time, distance m) of the last close reading
        self.make_rl = make_rl            # creates the walker (torque on + init pose)
        self.rl = None
        self.lock = threading.Lock()      # protects action / state
        self.bus = threading.Lock()       # serial bus outside the policy loop
        self.guard = FallGuard(tilt_deg=args.fall_deg)
        self.hold = HeadingHold()
        self.hold_limit = self.hold.limit
        self.state = "off"                # off | paused | running | fallen | relaxed
        self.busy = False                 # power_on in progress
        self.action = {"kind": "idle"}
        self.result = "ready: duck on the stand, press power_on"
        self.action_id = 0
        self.head = [0.0, 0.0]            # head pitch, head yaw targets
        self.head_now = [0.0, 0.0]        # what is sent, ramped towards self.head
        self.battery_v = None
        self.low_count = 0
        self.pause_at = None              # time a pause asked for while walking takes effect
        self.yaw_rate = 0.0               # low-passed gyro z
        self.sit_flag = 0.0               # sit/stand models: 1 = squat
        self.turn_i = 0.0                 # learned turn correction
        self.last_moving = 0.0
        self.battery_request = False
        self.log = collections.deque(maxlen=int(600 / DT))  # last 10 minutes
        self.us_log = collections.deque(maxlen=int(600 / DT))  # (t, distance m, age s) per step
        self.tilt = 0.0
        self.imu_last, self.imu_same = None, 0
        self.model = os.path.splitext(os.path.basename(args.onnx_model_path))[0]
        if self.model not in MODELS:  # the start-up model is always selectable
            MODELS.append(self.model)
        self.test_steps = []              # [(t, obs, phase)] of the running abtest
        self.tests = []                   # finished abtest results, newest last

    # ---------- policy thread (50 Hz while running) ----------
    def _imu_frozen(self, obs):
        """True once gyro + accel have repeated exactly for IMU_FROZEN_S."""
        if self.imu_last is None or not np.array_equal(obs[:6], self.imu_last):
            self.imu_last, self.imu_same = obs[:6].copy(), 0
        else:
            self.imu_same += 1
        return not self.args.fake and self.imu_same * DT >= IMU_FROZEN_S

    def _step(self, obs):
        if obs is None:
            return None
        now = time.time()
        self.log.append((now, obs.copy()))
        if self.us is not None:  # age tells "no echo for a while" from "far away"
            self.us_log.append((now, self.us.distance_m, now - self.us.last_ok))
        if self.action["kind"] == "test":
            self.test_steps.append((now, obs.copy(), self.action["phase"]))
        if self._imu_frozen(obs):
            self.rl.paused = True
            self.rl.last_commands = [0.0] * 7
            with self.lock:
                self.state = "paused"
                if self.action["kind"] == "test":
                    self._end_test("IMU frozen")
                self._finish("IMU frozen, walking stopped: check its wires, put the duck on the stand, restart the service")
            print("[imu] frozen, policy paused", flush=True)
            return None
        if self.imu_same == 0 and self.guard.update(obs[3:6]):
            self.rl.paused = True
            self.rl.last_commands = [0.0] * 7
            self.rl.hwi.turn_off()
            with self.lock:
                self.state = "fallen"
                self.tilt = self.guard.tilt_deg
                if self.action["kind"] == "test":
                    self._end_test(f"fell at {now - self.action['t0']:.1f} s ({self.guard.tilt_deg:.0f} deg)")
                self._finish(f"fell ({self.guard.tilt_deg:.0f} deg), servos released")
            return None
        self.hold.update(obs[2], now)
        self.yaw_rate += DT / YAW_RATE_TAU_S * (obs[2] - self.yaw_rate)
        if self.battery_request:
            self.battery_request = False
            self._read_battery()
        with self.lock:
            if self.action["kind"] != "idle":
                self.last_moving = now
            if self.pause_at is not None and now >= self.pause_at:
                self.pause_at = None
                self.rl.paused = True
                self.rl.last_commands = [0.0] * 7
                self.state = "paused"
                self._finish("paused")
                return None
            cmd = self._action_command(now)
            for k in (0, 1):
                step = float(np.clip(self.head[k] - self.head_now[k], -HEAD_RATE * DT, HEAD_RATE * DT))
                self.head_now[k] += step
        if self._sit_model():  # the posture flag sits in the fwd slot, no heading hold
            self.rl.last_commands = [self.sit_flag] + [0.0] * 6
            return obs
        if self.model.startswith("mj_"):  # both pitch servos share the nod, same sign, 3 was left at 0
            cmd[3], cmd[4] = NECK_SHARE * self.head_now[0], NECK_SHARE * self.head_now[0]
        else:
            cmd[4] = self.head_now[0]
        cmd[5] = self.head_now[1]
        trimmed = self.model in YAW_TRIM or self.model in TURN_LOOP_MODELS
        self.hold.limit = TRIMMED_HOLD_LIMIT if trimmed else self.hold_limit
        self.rl.last_commands = self.hold.command(cmd, self._yaw_trim(), True)
        return obs

    def _action_command(self, now):
        """Turn the current action into [fwd, side, turn, ...] (lock held)."""
        a = self.action
        cmd = [0.0] * 7
        if self._obstacle_stop(now):
            return cmd
        if a["kind"] == "walk":
            if now >= a["until"]:
                self._finish(a.get("done_msg", "done"))
            else:
                cmd[0] = a["speed"]
        elif a["kind"] == "turn":
            turned = np.degrees(self.hold.heading - a["yaw0"])
            remaining = a["degrees"] - turned
            lead = np.degrees(TURN_RATE) * 0.15  # stop a little early: it keeps turning a bit
            if abs(remaining) <= lead or np.sign(remaining) != np.sign(a["degrees"]):
                self._settle(now, f"done, turned {turned:.0f} deg")
                cmd[0] = WALK_SPEED
            elif now >= a["until"]:
                self._settle(now, f"timeout, turned {turned:.0f} deg")
                cmd[0] = WALK_SPEED
            else:
                cmd[0] = WALK_SPEED
                cmd[2] = self._turn_cmd(TURN_RATE * np.sign(a["degrees"]))
        elif a["kind"] == "test":
            t = now - a["t0"]
            k, end = 0, 0.0
            for k, ph in enumerate(AB_PHASES):
                end += ph[1]
                if t < end:
                    break
            if t >= end:
                self._end_test("completed")
                self._finish(f"test done: {self.model}")
            else:
                a["phase"] = k
                cmd[0], cmd[1], cmd[2] = AB_PHASES[k][2:5]
                if cmd[2] == 0 and cmd[0] > 0:  # straight: the speed this model actually steps at
                    cmd[0] = self._straight_speed()
        elif a["kind"] == "drive":
            if now - a["last"] > DRIVE_DEADMAN_S:
                self._finish("drive stopped (no update for 0.5 s)")
            else:
                cmd[0], cmd[2] = a["fwd"], self._turn_cmd(a["turn"], a["fwd"] != 0)
        if cmd[0] > 0 and a["kind"] in ("walk", "drive", "test") and a.get("turn", 0) == 0:
            cmd[0] *= self._slow_factor()
        return cmd

    def _slow_factor(self):
        """Forward speed scale in the slow-down zone (1 outside it)."""
        d = self.us.fresh() if self.us is not None and self.avoid else None
        if d is None or d >= OBSTACLE_SLOW_M:
            return 1.0
        f = float(np.clip((d - OBSTACLE_STOP_M) / (OBSTACLE_SLOW_M - OBSTACLE_STOP_M), 0, 1))
        return OBSTACLE_SLOW_MIN + (1 - OBSTACLE_SLOW_MIN) * f

    def _obstacle_ahead(self):
        """Distance (m) of the obstacle ahead, held OBSTACLE_LATCH_S after the
        last close reading, or None."""
        if self.us is None or not self.avoid:
            return None
        now = time.time()
        d = self.us.nearest()
        # once seen, readings up to OBSTACLE_CLEAR_M keep it: at 9-11 cm the
        # reading flickers around the stop line and a held up button walked on
        limit = OBSTACLE_STOP_M if self.obstacle_seen is None else OBSTACLE_CLEAR_M
        if d is not None and d < limit:
            self.obstacle_seen = (now, d)
        elif self.obstacle_seen is not None and now - self.obstacle_seen[0] > OBSTACLE_LATCH_S:
            self.obstacle_seen = None
        return self.obstacle_seen[1] if self.obstacle_seen is not None else None

    def _obstacle_stop(self, now):
        """Lock held. Ends forward motion (walk / drive / test) at an obstacle."""
        a = self.action
        turning = a["kind"] == "drive" and a.get("turn", 0) != 0
        d = None if a["kind"] not in ("walk", "drive", "test") or turning else self._obstacle_ahead()
        if d is None:
            self.obstacle_since = None
            return False
        if self.obstacle_since is None:
            self.obstacle_since = now
        if now - self.obstacle_since < OBSTACLE_HOLD_S:
            return False
        msg = f"obstacle {d * 100:.0f} cm ahead: stopped"
        if self.action["kind"] == "test":
            self._end_test(msg)
        self._finish(msg)
        self.obstacle_since = None
        return True

    def _turn_cmd(self, want, walking=True):
        """HeadingHold adds the yaw trim to any turn while walking; take it back
        out so left and right turns get the same command."""
        if want == 0:
            return 0.0
        if walking and self.model in TURN_LOOP_MODELS:
            step = TURN_LOOP_KI * (want - self.yaw_rate) * DT
            self.turn_i = float(np.clip(self.turn_i + step, -TURN_LOOP_LIMIT, TURN_LOOP_LIMIT))
            return want - self._yaw_trim() + self.turn_i
        return want - self._yaw_trim()

    def _yaw_trim(self):
        if self.model in YAW_TRIM:
            return YAW_TRIM[self.model]
        return 0.0 if self.model.startswith("mj_") else self.args.yaw_trim

    def _end_test(self, outcome):
        """Lock held. Saves the run in the auto_test.py format and keeps a summary."""
        steps, self.test_steps = self.test_steps, []
        model = self.model
        self.tests.append({"model": model, "outcome": outcome, "time": time.strftime("%H:%M:%S"),
                           "rows": summarize(steps, AB_PHASES)})

        def save():
            os.makedirs(os.path.join(HOME, "walk_logs"), exist_ok=True)
            path = os.path.join(HOME, "walk_logs", time.strftime(f"auto_ab_{model}_%Y%m%d_%H%M%S.pkl"))
            with open(path, "wb") as f:
                pickle.dump({"onnx": model + ".onnx", "test": "ab", "phases": AB_PHASES,
                             "outcome": outcome, "steps": [(t, o) for t, o, _ in steps],
                             "phase_of_step": [p for _, _, p in steps]}, f)
            print("[abtest]", model, outcome, path, flush=True)

        threading.Thread(target=save, daemon=True).start()

    def _set_model(self, name):
        if name not in MODELS:
            return {"ok": False, "error": f"unknown model {name!r}"}
        with self.lock:
            if self.state == "running" or self.busy:
                return {"ok": False, "error": "press pause first"}
            if self.rl is not None and not self.args.fake:
                self.rl.set_policy(os.path.join(SCRIPTS, name + ".onnx"))
            self.model = name
        return self._ok(f"model {name}")

    def _sit_model(self):
        return "sit" in self.model

    def _straight_speed(self):
        return STRAIGHT_SPEED.get(self.model, STRAIGHT_SPEED_DEFAULT)

    def _settle(self, now, msg):
        """After a turn: a short straight walk, then stop (braking a fast turn tipped it over)."""
        self.action = {"kind": "walk", "speed": WALK_SPEED, "until": now + SETTLE_S, "done_msg": msg}
        self.result = msg + " (settling)"

    def _finish(self, result):
        self.action = {"kind": "idle"}
        self.result = result

    # ---------- battery ----------
    def _read_battery(self):
        """Run where nothing else uses the bus: policy thread between steps, or
        while paused / off. rustypot opens the port exclusively, so its handle
        is dropped, the voltage read with pyserial, and the port reopened (~11 ms)."""
        if self.args.fake:
            self.battery_v = 8.0
            return
        try:
            with self.bus:
                if self.rl is None:
                    v = read_voltage_raw()
                else:
                    import gc

                    import rustypot

                    hwi = self.rl.hwi
                    hwi.io = None
                    gc.collect()  # drop the rustypot handle -> port closed
                    try:
                        v = read_voltage_raw()
                    finally:
                        hwi.io = rustypot.feetech(PORT_DEV, 1000000)  # gains stay in the servos
        except Exception as e:  # never let the monitor break the walk
            print("[battery]", e, flush=True)
            return
        if v is None:
            return
        self.battery_v = v
        self.low_count = self.low_count + 1 if v < CRITICAL_V else 0
        if self.low_count >= 3:
            with self.lock:
                self._finish(f"battery critical {v:.1f} V: stopped, charge now")

    def paused_fall_loop(self):
        """The fall guard in _step only runs while the policy runs. While paused
        (motors locked) a fall would leave the servos stalling against the floor,
        so watch the IMU here too and release them."""
        last_acc = None
        while True:
            time.sleep(0.05)
            rl = self.rl
            if rl is None or not rl.paused or self.state != "paused" or self.busy:
                continue
            try:
                acc = np.asarray(rl.imu.get_data()["accelero"], dtype=float) if not self.args.fake else None
            except Exception:
                continue
            repeated = acc is not None and last_acc is not None and np.array_equal(acc, last_acc)
            last_acc = acc
            if acc is None or repeated or not self.guard.update(acc):
                continue
            with self.bus:
                if self.rl.paused and self.state == "paused":
                    self.rl.hwi.turn_off()
            with self.lock:
                self.state = "fallen"
                self.tilt = self.guard.tilt_deg
                self._finish(f"fell while paused ({self.guard.tilt_deg:.0f} deg), servos released")

    def battery_loop(self):
        while True:
            if self.rl is None or self.rl.paused:
                self._read_battery()
            else:
                self.battery_request = True  # done by the policy thread between steps
            time.sleep(30)

    # ---------- commands (HTTP threads) ----------
    def command(self, c):
        kind = c.get("action")
        if kind == "save_log":  # always allowed, a fall is when the log matters most
            return self._ok(self._save_log())
        if kind == "shutdown":
            return self._shutdown()
        if kind in ("power_on", "recover"):
            return self._power_on()
        if kind == "relax":
            if c.get("on_stand") is not True:  # the page asks first: limp servos drop the duck
                return {"ok": False, "error": "put the duck on the stand first (send on_stand: true)"}
            return self._relax()
        if kind == "model":
            return self._set_model(str(c.get("name", "")))
        if kind == "avoid":
            if self.us is None:
                return {"ok": False, "error": "no ultrasonic sensor"}
            self.avoid = bool(c.get("on", True))
            return self._ok(f"obstacle stop {'on' if self.avoid else 'off'}")
        with self.lock:
            if self.busy:
                return {"ok": False, "error": "powering on, wait a few seconds"}
            if self.state in ("off", "relaxed"):
                return {"ok": False, "error": "servos are off: duck on the stand, press power_on"}
            if self.state == "fallen":
                return {"ok": False, "error": "duck has fallen: put it on the stand and press power_on"}
            if kind == "start":
                if self.low_count >= 3:
                    return {"ok": False, "error": "battery critical"}
                self.state = "running"
                self.pause_at = None
                self.sit_flag = 0.0
                self.rl.paused = False
                return self._ok("started")
            if kind == "pause":
                if self.action["kind"] == "test":
                    self._end_test("stopped by hand")
                if self.state == "running" and time.time() - self.last_moving < PAUSE_SETTLE_S:
                    if self.pause_at is None:  # stops walking now, _step pauses once it has settled
                        self.pause_at = time.time() + PAUSE_SETTLE_S
                    self._finish("stopping, then pause")
                    return self._ok("stopping, then pause")
                self.rl.paused = True
                self.rl.last_commands = [0.0] * 7
                self.state = "paused"
                self._finish("paused")
                return self._ok("paused")
            if kind == "stop":
                if self.action["kind"] == "test":
                    self._end_test("stopped by hand")
                self._finish("stopped")
                return self._ok("stopped")
            if self.state != "running":
                return {"ok": False, "error": "not started: put the duck on the floor and press start"}
            if self.pause_at is not None:
                return {"ok": False, "error": "stopping, then pause"}
            if kind in ("walk", "turn", "drive") and self.low_count >= 3:
                return {"ok": False, "error": "battery critical"}
            now = time.time()
            if kind == "posture":
                if not self._sit_model():
                    return {"ok": False, "error": "choose the squat model first"}
                self.sit_flag = 1.0 if c.get("sit") else 0.0
                self.last_moving = now
                return self._ok("squat" if self.sit_flag else "stand up")
            if self._sit_model():
                return {"ok": False, "error": "squat model: only squat and stand"}
            if self.action["kind"] == "test":
                return {"ok": False, "error": "test running: press stop to end it"}
            if kind == "abtest":
                if self.low_count >= 3:
                    return {"ok": False, "error": "battery critical"}
                self.test_steps = []
                self.action = {"kind": "test", "t0": now, "phase": 0}
                return self._ok(f"test started: {self.model}, {sum(p[1] for p in AB_PHASES)} s")
            if kind in ("walk", "drive", "abtest") and float(c.get("speed", c.get("fwd", 1))) > 0                     and float(c.get("turn", 0)) == 0 and self._obstacle_ahead() is not None:
                return {"ok": False, "error": f"obstacle {self._obstacle_ahead() * 100:.0f} cm ahead"}
            if kind == "walk":
                speed = float(c.get("speed", self._straight_speed()))
                sec = float(c.get("seconds", 2))
                if speed < 0:
                    if not self.args.allow_backward:
                        return {"ok": False, "error": "backward is disabled (the duck falls)"}
                    speed, sec = max(speed, -SPEED_RANGE[1]), min(sec, BACK_MAX_S)
                else:
                    speed = float(np.clip(speed, *SPEED_RANGE))
                sec = float(np.clip(sec, 0.2, MAX_WALK_S))
                self.action = {"kind": "walk", "speed": speed, "until": now + sec}
                return self._ok(f"walk {speed:+.2f} m/s for {sec:.1f} s")
            if kind == "turn":
                deg = float(np.clip(float(c.get("degrees", 90)), -360, 360))
                if abs(deg) < 5:
                    return {"ok": False, "error": "angle too small"}
                self.action = {"kind": "turn", "degrees": deg, "yaw0": self.hold.heading,
                               "until": now + abs(deg) / 15 + 3}
                return self._ok(f"turn {deg:+.0f} deg")
            if kind == "drive":
                lo = -0.1 if self.args.allow_backward else 0.0
                fwd = float(np.clip(float(c.get("fwd", 0)), lo, SPEED_RANGE[1]))
                if fwd > 0 and float(c.get("turn", 0)) == 0:  # page sends 0.1 for "up"
                    fwd = max(fwd, self._straight_speed())
                turn = float(np.clip(float(c.get("turn", 0)), -TURN_RATE, TURN_RATE))
                self.action = {"kind": "drive", "fwd": fwd, "turn": turn, "last": now}
                return {"ok": True, "id": self.action_id}  # no new id per refresh
            if kind == "head":
                if self.action["kind"] != "idle":
                    return {"ok": False, "error": "head moves only while standing still"}
                self.head = [float(np.clip(float(c.get("pitch", 0)), *HEAD_PITCH_LIMITS)),
                             float(np.clip(float(c.get("yaw", 0)), -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT))]
                return self._ok(f"head pitch {self.head[0]:+.2f} yaw {self.head[1]:+.2f}")
        return {"ok": False, "error": f"unknown action {kind!r}"}

    def _ok(self, msg):
        self.action_id += 1
        self.result = msg
        return {"ok": True, "id": self.action_id, "message": msg}

    def _power_on(self):
        """Torque on + init pose. First time: create the walker (loads the
        policy, IMU, ~5 s). Later (after relax or a fall): re-enable torque."""
        with self.lock:
            if self.busy:
                return {"ok": False, "error": "already powering on"}
            if self.state == "running":
                return {"ok": False, "error": "running: press pause first"}
            self.busy = True
            self.result = "powering on..."
        try:
            with self.bus:
                if self.rl is None:
                    rl = self.make_rl()  # RLWalk.__init__ does turn_on (low kp -> init pose -> kp)
                    if not self.args.fake and self.model != os.path.splitext(os.path.basename(self.args.onnx_model_path))[0]:
                        rl.set_policy(os.path.join(SCRIPTS, self.model + ".onnx"))  # picked before power_on
                    rl.paused = True
                    rl.last_commands = [0.0] * 7
                    get_obs = rl.get_obs
                    rl.get_obs = lambda: self._step(get_obs())
                    self.rl = rl
                    threading.Thread(target=rl.run, daemon=True).start()
                else:
                    rl = self.rl
                    rl.paused = True
                    if not self.args.fake:
                        hwi = rl.hwi
                        ids = list(hwi.joints.values())
                        hwi.io.set_kps(ids, hwi.low_torque_kps)  # soft, so nothing snaps
                        hwi.io.enable_torque(ids)
                        hwi.turn_on()
            for name in ("last_action", "last_last_action", "last_last_last_action"):
                setattr(self.rl, name, np.zeros(self.rl.num_dofs))
            self.rl.motor_targets = np.array(self.rl.init_pos)
            self.rl.prev_motor_targets = np.array(self.rl.init_pos)
            self.guard.reset()
            with self.lock:
                self.state = "paused"
                self.head = [0.0, 0.0]
                self.head_now = [0.0, 0.0]
                self._finish("legs in the init pose: put the duck on the floor, then start")
            return self._ok("powered on")
        except Exception as e:
            with self.lock:
                self.result = f"power on failed: {e}"
            return {"ok": False, "error": f"power on failed: {e}"}
        finally:
            self.busy = False

    def _relax(self):
        with self.lock:
            if self.rl is None:
                return {"ok": False, "error": "servos are already off"}
            if self.state == "running":
                return {"ok": False, "error": "running: press pause first (the duck would drop)"}
            self.rl.paused = True
            self.rl.last_commands = [0.0] * 7
        with self.bus:
            if not self.args.fake:
                self.rl.hwi.turn_off()
        with self.lock:
            self.state = "relaxed"
            self._finish("servos limp")
        return self._ok("relaxed")

    def _shutdown(self):
        if self.rl is not None:
            self.rl.paused = True
            self.rl.last_commands = [0.0] * 7
            if not self.args.fake:
                try:
                    with self.bus:
                        self.rl.hwi.turn_off()
                except Exception as e:
                    print("[shutdown] turn_off:", e, flush=True)
        saved = self._save_log() if self.log else "no log"
        with self.lock:
            self.state = "off"
            self._finish("shutting down: switch the power off when the green LED is off")
        if not self.args.fake:
            threading.Timer(1.0, lambda: subprocess.run(["sudo", "shutdown", "-h", "now"])).start()
        return self._ok(f"shutting down ({saved})")

    def status(self):
        with self.lock:
            a = dict(self.action)
            for k in ("until", "last", "yaw0", "done_msg", "t0"):
                a.pop(k, None)
            return {"state": self.state, "busy": self.busy, "action": a, "result": self.result,
                    "id": self.action_id, "battery_v": self.battery_v,
                    "battery_low": bool(self.battery_v and self.battery_v < LOW_V),
                    "heading_deg": round(float(np.degrees(self.hold.heading)), 1),
                    "head": self.head, "fall_tilt": round(self.tilt, 1),
                    "distance_cm": (round(self.us.fresh() * 100) if self.us is not None and self.us.fresh() is not None else None),
                    "avoid": self.avoid, "has_us": self.us is not None,
                    "avoid_cm": [round(OBSTACLE_STOP_M * 100), round(OBSTACLE_SLOW_M * 100)],
                    "model": self.model, "models": MODELS, "tests": self.tests[-10:],
                    "test_elapsed": round(time.time() - self.action["t0"], 1)
                    if self.action["kind"] == "test" else None}

    def _save_log(self):
        os.makedirs(os.path.join(HOME, "walk_logs"), exist_ok=True)
        path = os.path.join(HOME, "walk_logs", time.strftime("server_%Y%m%d_%H%M%S.pkl"))
        with open(path, "wb") as f:
            pickle.dump({"onnx": self.model + ".onnx", "steps": list(self.log),
                         "us": list(self.us_log)}, f)
        return f"saved {len(self.log)} steps to {path}"


class FakeWalk:
    """Stands in for RLWalk: 50 Hz loop, yaw rate follows the turn command."""

    num_dofs = 14
    init_pos = [0.0] * 14

    def __init__(self, drift=-0.2, gain=1.2):
        self.paused, self.last_commands = True, [0.0] * 7
        self.drift, self.gain = drift, gain
        self.hwi = type("H", (), {"turn_off": staticmethod(lambda: None)})()

    def get_obs(self):
        o = np.zeros(101)
        o[2] = self.gain * self.last_commands[2] + self.drift * (self.last_commands[0] != 0)
        o[5] = 9.81
        o[6:13] = self.last_commands
        o[97:99] = 1.0
        return o

    def run(self):
        while True:
            if not self.paused:
                self.get_obs()
            time.sleep(DT)


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            tok = srv.args.token
            return not tok or self.headers.get("X-Duck-Token") == tok

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    with open(PAGE, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    self._send(404, {"error": "duck_remote.html missing"})
            elif self.path == "/status":
                self._send(200, srv.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/cmd":
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(403, {"ok": False, "error": "bad token"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                c = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, srv.command(c))
            except (ValueError, TypeError) as e:
                self._send(400, {"ok": False, "error": str(e)})

        def log_message(self, fmt, *a):  # quiet: drive sends 5 requests/s
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_model_path", default="BEST_WALK_ONNX_2.onnx")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--kp", type=int, default=20)
    ap.add_argument("--yaw_trim", type=float, default=0.1)
    ap.add_argument("--fall_deg", type=float, default=40)
    ap.add_argument("--allow_backward", action="store_true")
    ap.add_argument("--token", default="", help="if set, POST needs header X-Duck-Token")
    ap.add_argument("--fake", action="store_true", help="no robot (logic test)")
    ap.add_argument("--no_us100", action="store_true", help="don't use the US-100 ultrasonic sensor")
    ap.add_argument("--fresh_imu", action="store_true", help="newest IMU sample instead of the stock 1-slot queue (fresh_imu.py)")
    args = ap.parse_args()
    if args.fresh_imu and not args.fake:
        import fresh_imu

        fresh_imu.enable()

    if args.fake:
        make_rl = FakeWalk
    else:
        def make_rl():
            from mjlab_walk import DuckWalk

            return DuckWalk(args.onnx_model_path, commands=False, pid=[args.kp, 0, 0])

    def on_term(*_):  # systemctl stop -> same clean exit as Ctrl+C (log saved)
        raise KeyboardInterrupt

    import signal

    signal.signal(signal.SIGTERM, on_term)
    us = None
    if not args.fake and not args.no_us100:
        try:
            from us100 import US100

            us = US100()
            print(f"US-100 on {us.ser.port} (temperature {us.temp_c} C)", flush=True)
        except Exception as e:  # no UART / no sensor: run without obstacle stop
            print("US-100 not used:", e, flush=True)
    srv = Server(args, make_rl, us)
    threading.Thread(target=srv.battery_loop, daemon=True).start()
    threading.Thread(target=srv.paused_fall_loop, daemon=True).start()
    http = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(srv))
    print(f"duck server on port {args.port} ({'FAKE' if args.fake else 'robot'}), servos untouched "
          f"until power_on; phone: http://<pi-ip>:{args.port}/", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if srv.rl is not None:
            srv.rl.paused = True
            srv.rl.last_commands = [0.0] * 7
        print(srv._save_log() if srv.log else "no steps to save", flush=True)


if __name__ == "__main__":
    main()
