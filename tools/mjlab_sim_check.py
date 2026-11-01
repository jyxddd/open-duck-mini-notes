"""
CPU MuJoCo rehearsal of an mjlab Open Duck walk policy, with the SAME
observation the Pi builds (pi/mjlab_walk.py DuckWalk) and BAM STS3215
actuators (legs kp 20, head kp 8) like training.

Run in WSL from ~/microduck_rl:
  .venv/bin/python /mnt/d/Claude/duck/tools/mjlab_sim_check.py --onnx mj_od1.onnx \
      --schedule "3:0,6:0.15,6:0.1:0:-0.4,6:0.1:0:0.4,3:0"

schedule: "sec:fwd:side:turn[:neck:head_pitch:head_yaw:head_roll],..."
Prints per segment: body-frame speed, yaw rate, touch-downs/s, tilt; and
whether / when it fell. --out saves a log in the walk_*.pkl format.
"""
import argparse
import math
import pickle

import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model

XML = "src/mjlab_microduck/robot/openduck/robot_walk.xml"
JSON = None  # resolved from bam package
# Target clip of od3+ policies (openduck_constants.JOINT_TARGET_LIMITS = pi/mjlab_walk.py)
TARGET_LO = np.array([-0.35, -0.3, -1.1, 0.6, -1.45, -0.3, -0.7, -1.2, -0.4,
                      -0.35, -0.3, -0.3, 0.6, -1.45])
TARGET_HI = np.array([0.35, 0.3, 0.3, 1.75, 0.3, 1.0, 0.7, 1.2, 0.4,
                      0.35, 0.3, 1.1, 1.75, 0.3])
HOME = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0, 0, 0, 0, -0.003, -0.065, 0.635, 1.379, -0.796])
STIFF_SOLREF = (-5.0e4, -2.0e2)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)
DT, DECIM = 0.005, 4


# robot/openduck_constants.py GROUND_MESHES
GROUND_MESHES = ("trunk_bottom", "body_front", "body_back", "left_cache", "right_cache",
                 "left_knee_to_ankle_left_sheet", "left_knee_to_ankle_right_sheet",
                 "foot_side", "foot_top", "left_roll_to_pitch", "right_roll_to_pitch",
                 "head_bot_sheet")


def parse_schedule(text):
    out = []
    for part in text.split(","):
        v = [float(x) for x in part.split(":")]
        out.append(v + [0.0] * (8 - len(v)))
    return out


UPPER_BODIES = ("trunk_base", "neck_pitch_assembly", "head_pitch_to_yaw", "neck_yaw_assembly", "head_assembly")


def build(vin, json_path, knee_range=None, kp_legs=20.0, kp_head=8.0, backlash_deg=0.0, hidden=False, flex=0.0, ground=False):
    from bam.mujoco import MujocoController

    class EncoderController(MujocoController):
        """Firmware PD closes on the output-side encoder: servo joint + its
        passive backlash joint (microduck_rl BacklashEncoderBamActuator)."""

        def update(self):
            if self.bl_q is None:
                return super().update()
            qpos, qvel = self.mujoco_data.qpos, self.mujoco_data.qvel
            keep_q, keep_v = qpos[self.qpos_indexes].copy(), qvel[self.dof_indexes].copy()
            qpos[self.qpos_indexes] += qpos[self.bl_q]
            qvel[self.dof_indexes] += qvel[self.bl_v]
            super().update()
            qpos[self.qpos_indexes], qvel[self.dof_indexes] = keep_q, keep_v

    spec = mujoco.MjSpec.from_file(XML)
    if ground:  # same floor contacts as the sit/stand training model
        for g in spec.geoms:
            if g.classname.name == "visual" and g.meshname in GROUND_MESHES:
                g.contype, g.conaffinity, g.group = 1, 0, 3
    if flex:  # structural compliance: a hidden spring hinge after each servo joint
        hidden, backlash_deg = True, 1.0
    if backlash_deg:  # a passive hinge with +-backlash play after each servo joint
        servo_joints = {act.target for act in spec.actuators}
        for j in list(spec.joints):
            if j.name in servo_joints:
                p = j.parent.add_joint()
                p.name = "passive_" + j.name + "_backlash"
                p.type = mujoco.mjtJoint.mjJNT_HINGE
                p.axis, p.pos = j.axis.copy(), j.pos.copy()
                p.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                p.range = [-math.radians(backlash_deg), math.radians(backlash_deg)]
                p.armature = 0.001  # microduck_rl add_backlash.py defaults
                p.damping = np.full((3, 1), 0.01)
                p.solref_limit = (0.01, 1.0)
                p.solimp_limit = (0.95, 0.999, 0.0001, 0.5, 2.0)
                if flex:
                    p.limited = mujoco.mjtLimited.mjLIMITED_FALSE
                    p.stiffness = np.array([flex, 0, 0]).reshape(3, 1) if np.ndim(p.stiffness) else flex
                    p.damping = np.full((3, 1), 0.02 * flex ** 0.5)
    if knee_range:  # the real knee bends further than the XML's +-1.57 stop
        for j in spec.joints:
            if j.name.endswith("_knee"):
                j.range = [-knee_range, knee_range]
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.name = "floor"
    models = {"legs": load_model(json_path), "head": load_model(json_path)}
    models["legs"].actuator.kp, models["head"].actuator.kp = kp_legs, kp_head
    force_limit = vin * models["legs"].kt.value / models["legs"].R.value
    names = {"legs": [], "head": []}
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names["head" if ("neck" in act.name or "head" in act.name) else "legs"].append(act.name)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE and not j.name.startswith("passive_"):
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = STIFF_SOLREF
            j.solimp_friction = STIFF_SOLIMP
    m = spec.compile()
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    ctrls = []
    for k in ("legs", "head"):
        models[k].actuator.vin = vin
        c = EncoderController(models[k], names[k], m, d)
        c.bl_q = c.bl_v = None
        if backlash_deg and not hidden:
            bl = [m.joint("passive_" + m.joint(j).name + "_backlash").id for j in c.joint_indexes]
            c.bl_q, c.bl_v = m.jnt_qposadr[bl], m.jnt_dofadr[bl]
        ctrls.append(c)
    return m, d, ctrls


def main():
    import bam

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--schedule", default="3:0,6:0.15,6:0.1:0:-0.4,6:0.1:0:0.4,3:0")
    ap.add_argument("--vin", type=float, default=7.6)
    ap.add_argument("--out")
    ap.add_argument("--knee_range", type=float, help="widen the knee joint range to +-this (rad)")
    ap.add_argument("--clip", action="store_true", help="clip targets to TARGET_LO/HI (like the Pi)")
    ap.add_argument("--act_delay", type=int, default=0, help="extra control steps (20 ms) before a target reaches the servos")
    ap.add_argument("--imu_pitch_deg", type=float, default=0, help="IMU mounted tilted: + = reads nose-up (real duck: ~4)")
    ap.add_argument("--gyro_noise", type=float, default=0, help="uniform +- noise on gyro (rad/s)")
    ap.add_argument("--acc_noise", type=float, default=0, help="uniform +- noise on accel (m/s^2)")
    ap.add_argument("--imu_roll_deg", type=float, default=0, help="IMU mounted rolled: + = accel y reads positive when level (real duck: ~2.5)")
    ap.add_argument("--friction", type=float, help="foot-floor sliding friction (XML default otherwise)")
    ap.add_argument("--head_mass", type=float, default=0, help="extra kg on head_assembly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp_legs", type=float, default=20.0)
    ap.add_argument("--kp_head", type=float, default=8.0)
    ap.add_argument("--hidden_play", action="store_true", help="the play is outside the encoder (loose horn / flexing part): servo and obs see the motor side only")
    ap.add_argument("--flex", type=float, default=0.0, help="hidden spring per servo joint, Nm/rad (structural compliance)")
    ap.add_argument("--backlash", type=float, default=0.0, help="gear play per servo, +- deg (encoder on the output side; microduck_rl uses 1)")
    ap.add_argument("--zero_err", default="", help="servo zero errors, deg per joint index: '4:4,13:4' (encoder reads true+err)")
    ap.add_argument("--kick", default="", help="angular velocity kick 't:wx:wy:wz' (s, rad/s, body frame), e.g. the real start jerk '0.1:0:0.5:0'")
    ap.add_argument("--head_add", default="", help="extra mass on the head 'grams:forward_mm' (voice board)")
    ap.add_argument("--chest_add", default="", help="extra mass in the trunk 'grams:forward_mm' (speaker)")
    ap.add_argument("--ground", action="store_true", help="shins, trunk and head touch the floor (sit/stand models); fwd in the schedule = sit flag")
    ap.add_argument("--hang", action="store_true", help="trunk held still in the air, as on the stand")
    ap.add_argument("--com_x", type=float, default=0.0, help="move the whole-robot CoM forward by this (mm, negative = back; upper body bodies shifted)")
    ap.add_argument("--imu_delay", type=int, default=0, help="control steps (20 ms) the policy's gyro/accel lag behind the body")
    ap.add_argument("--foot_solref", default="", help="soft/bouncy foot contact 'timeconst:dampratio' (MuJoCo default 0.02:1), e.g. 0.04:0.2")
    ap.add_argument("--freeze_head", action="store_true", help="hold the 4 head joints at HOME whatever the policy asks")
    ap.add_argument("--head_log", action="store_true", help="print neck/head targets and pitch gyro for the first 1.5 s")
    ap.add_argument("--imu_yaw180", action="store_true", help="IMU turned 180 deg about z: gyro/accel x,y negated")
    a = ap.parse_args()
    json_path = f"{bam.__path__[0]}/params/feetech_sts3215_7_4V/m1.json"
    m, d, ctrls = build(a.vin, json_path, a.knee_range, a.kp_legs, a.kp_head, a.backlash, a.hidden_play, a.flex, a.ground)
    for spec_add, body in ((a.head_add, "head_assembly"), (a.chest_add, "trunk_base")):
        if spec_add:
            g, dx = (float(v) for v in spec_add.split(":"))
            b, add = m.body(body).id, g / 1000
            old = m.body_mass[b]
            m.body_ipos[b, 0] += add * dx / 1000 / (old + add)
            m.body_inertia[b] *= (old + add) / old
            m.body_mass[b] = old + add
    if a.com_x:
        upper = [m.body(n).id for n in UPPER_BODIES]
        m.body_ipos[upper, 0] += a.com_x / 1000 * m.body_mass.sum() / m.body_mass[upper].sum()
    rng = np.random.default_rng(a.seed)
    zerr = np.zeros(14)
    for kv in filter(None, a.zero_err.split(",")):
        k, v = kv.split(":")
        zerr[int(k)] = math.radians(float(v))
    if a.friction is not None:
        m.geom_friction[:, 0] = a.friction
    if a.foot_solref:
        for g in ("left_foot_collision", "right_foot_collision"):
            m.geom_solref[m.geom(g).id] = [float(x) for x in a.foot_solref.split(":")]
    if a.head_mass:
        b = m.body("head_assembly").id
        m.body_mass[b] += a.head_mass
        mujoco.mj_setConst(m, d)
    tilt = math.radians(a.imu_pitch_deg)
    imu_rot = np.array([[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0], [-math.sin(tilt), 0, math.cos(tilt)]])
    roll = math.radians(a.imu_roll_deg)
    imu_rot = np.array([[1, 0, 0], [0, math.cos(roll), math.sin(roll)], [0, -math.sin(roll), math.cos(roll)]]) @ imu_rot
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])

    jids = [m.actuator(i).trnid[0] for i in range(m.nu)]
    qadr, vadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    if a.backlash and not a.hidden_play and not a.flex:
        bl = [m.joint("passive_" + m.joint(j).name + "_backlash").id for j in jids]
        bl_q, bl_v = m.jnt_qposadr[bl], m.jnt_dofadr[bl]
    else:
        bl_q = bl_v = None
    enc_q = lambda: d.qpos[qadr] + (d.qpos[bl_q] if bl_q is not None else 0)
    enc_v = lambda: d.qvel[vadr] + (d.qvel[bl_v] if bl_v is not None else 0)
    d.qpos[2] = 0.155
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[qadr] = HOME - zerr
    mujoco.mj_forward(m, d)
    for c in ctrls:
        c.reset(d.qpos)
        c.model.actuator.q_target_smooth = d.qpos[c.qpos_indexes].copy()  # STS3215 slew state

    sid = {n: m.sensor_adr[m.sensor(n).id] for n in ("imu_ang_vel", "imu_accel")}
    floor = m.geom("floor").id
    feet = [m.geom("left_foot_collision").id, m.geom("right_foot_collision").id]
    kick = np.array([float(x) for x in a.kick.split(":")]) if a.kick else None
    imu_buf = []
    last_action = np.zeros(14)
    action_steps = []
    pending = [HOME.copy()] * a.act_delay
    lim_lo, lim_hi = m.jnt_range[jids].T
    n_ctrl = n_clip = n_stop = 0  # control steps / asking beyond the clip box / a joint on its stop
    steps, rows = [], []
    t, fell_at = 0.0, None
    for seg, (sec, fwd, side, turn, *head) in enumerate(parse_schedule(a.schedule)):
        cmds = np.array([fwd, side, turn, *head])
        prev_contact = None
        seg_rows = []
        for _ in range(int(round(sec / (DT * DECIM)))):
            gyro = d.sensordata[sid["imu_ang_vel"]:sid["imu_ang_vel"] + 3].copy()
            acc = d.sensordata[sid["imu_accel"]:sid["imu_accel"] + 3].copy()
            imu_buf.append((gyro, acc))
            gyro, acc = imu_buf.pop(0) if len(imu_buf) > a.imu_delay else imu_buf[0]
            gyro = imu_rot @ gyro + rng.uniform(-a.gyro_noise, a.gyro_noise, 3)
            acc = imu_rot @ acc + rng.uniform(-a.acc_noise, a.acc_noise, 3)
            if a.imu_yaw180:
                gyro[:2] *= -1
                acc[:2] *= -1
            grav = -acc / max(np.linalg.norm(acc), 1e-6)
            pos_rel = enc_q() + zerr - HOME
            vel = enc_v().copy()
            if a.ground:  # the sit/stand task has gravity right after the gyro
                obs = np.concatenate([gyro, grav, pos_rel, vel, last_action, cmds[:3], cmds[3:7], np.zeros(6)]).astype(np.float32)
            else:
                obs = np.concatenate([gyro, pos_rel, vel, last_action, cmds[:3], grav, cmds[3:7], np.zeros(6)]).astype(np.float32)
            action = sess.run(None, {"obs": obs[None]})[0][0].astype(np.float64)
            action_steps.append(np.degrees(action - last_action))
            last_action = action
            target = HOME + action
            if a.head_log and t < 1.5:
                print(f"{t:4.2f} gyro {gyro[0]:+.2f} {gyro[1]:+.2f} {gyro[2]:+.2f} head deg", np.degrees(action[5:9]).round(0), "legs", np.degrees(action[[0,1,2,3,4,9,10,11,12,13]]).round(0))
            if a.freeze_head:
                target[5:9] = HOME[5:9]
            n_ctrl += 1
            n_clip += bool(np.any((target < TARGET_LO - 0.02) | (target > TARGET_HI + 0.02)))
            q = d.qpos[qadr]
            n_stop += bool(np.any((q < lim_lo + 0.02) | (q > lim_hi - 0.02)))
            if a.clip:
                target = np.clip(target, TARGET_LO, TARGET_HI)
            target = target - zerr  # the servo holds what its encoder calls target
            if a.act_delay:
                pending.append(target)
                target = pending.pop(0)
            for c in ctrls:
                c.q_target = target[[list(m.actuator(n).name for n in range(m.nu)).index(x) for x in c.actuator]]
            if kick is not None and abs(t - kick[0]) < DT * DECIM / 2:
                d.qvel[3:6] += kick[1:]
            for _ in range(DECIM):
                for c in ctrls:
                    c.update()
                mujoco.mj_step(m, d)
                if a.hang:
                    d.qpos[:7] = [0, 0, 0.3, 1, 0, 0, 0]
                    d.qvel[:6] = 0
            t += DT * DECIM
            contact = [False, False]
            for i in range(d.ncon):
                g = {d.contact[i].geom1, d.contact[i].geom2}
                if floor in g:
                    for k in (0, 1):
                        contact[k] |= feet[k] in g
            touch = 0 if prev_contact is None else sum(int(c and not p) for c, p in zip(contact, prev_contact))
            prev_contact = contact
            quat = d.qpos[3:7]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            v_body = rot.T @ d.qvel[0:3]
            tilt = math.degrees(math.acos(max(-1, min(1, rot[2, 2]))))
            seg_rows.append((v_body[0], v_body[1], d.qvel[5], touch, tilt, d.qpos[2]))
            legacy = np.concatenate([gyro, acc, np.r_[cmds, np.zeros(7 - len(cmds))], pos_rel, vel * 0.05,
                                     last_action, np.zeros(42), contact, np.zeros(2)])
            steps.append((t, legacy))
            if fell_at is None and (tilt > 45 or (d.qpos[2] < 0.08 and not a.ground)):
                fell_at = t
                break
        r = np.array(seg_rows)
        if a.ground:
            print(f"seg {seg} cmd {fwd:.0f}: root z start {r[0, 5]:.3f} end {r[-5:, 5].mean():.3f}, tilt end {r[-5:, 4].mean():.1f} max {r[:, 4].max():.1f}, drop speed max {np.abs(np.diff(r[:, 5])).max() / (DT * DECIM):.2f} m/s")
        skip = min(25, len(r) // 3)
        rr = r[skip:] if len(r) > skip + 5 else r
        rows.append((seg, sec, fwd, side, turn, rr[:, 0].mean(), rr[:, 1].mean(), math.degrees(rr[:, 2].mean()),
                     rr[:, 3].sum() / (len(rr) * DT * DECIM), rr[:, 4].mean(), rr[:, 4].max()))
        if fell_at is not None:
            break

    print(f"{'seg':>3} {'cmd fwd':>7} {'side':>5} {'turn':>5} | {'vx':>6} {'vy':>6} {'yaw°/s':>7} {'steps/s':>7} {'tilt':>5} {'max':>5}")
    for seg, sec, fwd, side, turn, vx, vy, wz, tps, tm, tx in rows:
        print(f"{seg:3d} {fwd:7.2f} {side:5.2f} {turn:5.2f} | {vx:6.3f} {vy:6.3f} {wz:7.1f} {tps:7.2f} {tm:5.1f} {tx:5.1f}")
    print(f"fell at {fell_at:.1f} s" if fell_at else "no fall")
    print(f"beyond clip {100 * n_clip / max(n_ctrl, 1):.0f}% of steps, joint on its stop {100 * n_stop / max(n_ctrl, 1):.0f}%")
    legs = np.array(action_steps)[100:][:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]
    print(f"action step rms {math.sqrt((legs ** 2).mean()):.2f} deg (legs, after the first 2 s)")
    if a.out:
        pickle.dump({"onnx": a.onnx, "steps": steps}, open(a.out, "wb"))


if __name__ == "__main__":
    main()
