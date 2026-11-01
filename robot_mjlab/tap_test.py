"""
Tap / step test: how springy is the standing duck? (no policy running)

The servos hold the init pose with the walking gains (legs kp 20, head 8),
the duck stands on the floor, and the IMU + joints are logged at 50 Hz while
you tap its back gently (and, optionally, the script makes one small
ankle step itself, like the policy's first action). The log is what the
simulator has to reproduce before we trust it again.

    ~/run_exclusive.sh tap_test.py                     # 30 s, taps by hand
    ~/run_exclusive.sh tap_test.py --ankle_step_deg 4  # + both ankles +4 deg at 10 s, back at 11 s
    ~/run_exclusive.sh tap_test.py --ankle_step_deg 4 --fresh_imu   # same with the fresher IMU reader

Steps (the script waits for Enter at each one):
  1. duck on the stand            -> servos on, init pose
  2. duck on the floor, hands off -> recording
  3. duck back on the stand       -> servos released

Fall guard: tilt > --fall_deg -> all servos released at once, recording stops.
Log: ~/walk_logs/tap_<time>.pkl  dict(t, gyro, acc, pos, vel, target, args, fell)
"""

import argparse
import os
import pickle
import time

import numpy as np

from fall_guard import FallGuard

HOME_DIR = os.path.expanduser("~")
NUM_DOFS = 14
HEAD = slice(5, 9)
ANKLES = (4, 13)  # left_ankle, right_ankle
INIT_POS = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0.0, 0.0, 0.0, 0.0,
                     -0.003, -0.065, 0.635, 1.379, -0.796])


class FakeHW:
    """--fake: no servos / IMU, for checking the script itself."""

    joints = {f"j{i}": i for i in range(NUM_DOFS)}

    def __init__(self):
        self.q = INIT_POS.copy()
        self.t0 = time.time()

    def set_kds(self, kds): pass
    def set_kps(self, kps): pass
    def turn_on(self): print("[fake] turn_on")
    def turn_off(self): print("[fake] turn_off")

    def set_position_all(self, d):
        self.q = np.array(list(d.values()))

    def get_present_positions(self, ignore=()):
        return self.q + np.random.uniform(-0.002, 0.002, NUM_DOFS)

    def get_present_velocities(self, ignore=()):
        return np.random.uniform(-0.05, 0.05, NUM_DOFS)

    def get_data(self):
        w = 0.3 * np.sin(2 * np.pi * 4.5 * (time.time() - self.t0))
        return {"gyro": np.array([0.0, w, 0.0]), "accelero": np.array([0.6, 0.3, 9.8])}


def ask(msg):
    input(f"\n>>> {msg}  [Enter] ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="recording length")
    ap.add_argument("--kp", type=int, default=20, help="leg servo P (walking: 20)")
    ap.add_argument("--head_kp", type=int, default=8)
    ap.add_argument("--freq", type=float, default=50.0)
    ap.add_argument("--fall_deg", type=float, default=40.0)
    ap.add_argument("--ankle_step_deg", type=float, default=0.0, help="both ankles step by this (+ = toes up), 0 = off")
    ap.add_argument("--step_at", type=float, default=10.0, help="s after recording starts")
    ap.add_argument("--step_len", type=float, default=1.0, help="s before the ankles go back")
    ap.add_argument("--fresh_imu", action="store_true", help="newest IMU sample instead of the stock 1-slot queue (fresh_imu.py)")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    if a.fake:
        hwi = imu = FakeHW()
    else:
        from mini_bdx_runtime.duck_config import DuckConfig
        from mini_bdx_runtime.raw_imu import Imu
        from mini_bdx_runtime.rustypot_position_hwi import HWI

        if a.fresh_imu:
            import fresh_imu

            fresh_imu.enable()

        cfg = DuckConfig(config_json_path=f"{HOME_DIR}/duck_config.json")
        hwi = HWI(cfg, "/dev/ttyACM0")
        assert np.allclose(list(hwi.init_pos.values()), INIT_POS), "HWI init_pos differs from the training home pose"
        imu = Imu(sampling_freq=int(a.freq), upside_down=cfg.imu_upside_down)

    names = list(hwi.joints)
    send = lambda q: hwi.set_position_all(dict(zip(names, q)))

    ask("1/3 鸭子放在支架上，按回车上力（舵机锁到初始姿势）")
    kps = [a.kp] * NUM_DOFS
    kps[HEAD] = [a.head_kp] * 4
    hwi.set_kds([0] * NUM_DOFS)
    hwi.set_kps(kps)
    hwi.turn_on()
    time.sleep(2)

    ask(f"2/3 把鸭子放到地上，放稳、手完全松开，按回车开始记录 {a.seconds:.0f} 秒")
    tip = "期间轻轻拍几下背（前后方向），每拍一下等 3 秒"
    if a.ankle_step_deg:
        tip += f"；第 {a.step_at:.0f} 秒程序会自己动一下脚踝 {a.ankle_step_deg:+.0f}°，那几秒别碰它"
    print(tip)

    guard = FallGuard(tilt_deg=a.fall_deg, freq=a.freq)
    rec = {k: [] for k in ("t", "gyro", "acc", "pos", "vel", "target")}
    fell = False
    target = INIT_POS.copy()
    period = 1.0 / a.freq
    t0 = time.time()
    next_t = t0
    last_print = -1
    try:
        while True:
            now = time.time()
            t = now - t0
            if t >= a.seconds:
                break
            want = INIT_POS.copy()
            if a.ankle_step_deg and a.step_at <= t < a.step_at + a.step_len:
                for i in ANKLES:
                    want[i] += np.radians(a.ankle_step_deg)
            if not np.array_equal(want, target):
                target = want
                send(target)
            d = imu.get_data()
            pos = hwi.get_present_positions(ignore=["left_antenna", "right_antenna"])
            vel = hwi.get_present_velocities(ignore=["left_antenna", "right_antenna"])
            if d is not None and pos is not None and vel is not None:
                for k, v in (("t", t), ("gyro", d["gyro"]), ("acc", d["accelero"]),
                             ("pos", pos), ("vel", vel), ("target", target)):
                    rec[k].append(np.array(v, dtype=float))
                if guard.update(d["accelero"]):
                    hwi.turn_off()
                    fell = True
                    print(f"\n摔倒（倾斜 {np.degrees(guard.tilt):.0f}°），舵机已全部卸力，停止记录")
                    break
            if int(t) != last_print:
                last_print = int(t)
                g = rec["gyro"][-1] if rec["gyro"] else np.zeros(3)
                print(f"\r{t:4.0f} s  陀螺仪 {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}", end="", flush=True)
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        print("\nCtrl+C：停止记录")

    if not fell:  # a goal write after turn_off puts the torque back on
        send(INIT_POS)
    os.makedirs(f"{HOME_DIR}/walk_logs", exist_ok=True)
    path = f"{HOME_DIR}/walk_logs/tap_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    log = {k: np.array(v) for k, v in rec.items()}
    log.update(args=vars(a), fell=fell, joints=names)
    with open(path, "wb") as f:
        pickle.dump(log, f)
    n = len(log["t"])
    rate = (n - 1) / (log["t"][-1] - log["t"][0]) if n > 1 else 0
    print(f"\n已保存 {path}（{n} 帧，{rate:.0f} Hz）")

    if not fell:
        ask("3/3 把鸭子放回支架上，按回车卸力")
        hwi.turn_off()
    print("完成")


if __name__ == "__main__":
    main()
