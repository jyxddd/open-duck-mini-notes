"""
CPU MuJoCo rehearsal of an mjlab Open Duck walk policy, with the SAME
observation the Pi builds (pi/mjlab_walk.py DuckWalk) and BAM STS3215
actuators (legs kp 20, head kp 8) like training.

Run in WSL from ~/microduck_rl:
  .venv/bin/python /mnt/d/Claude/duck/tools/mjlab_sim_check.py --onnx mj_od1.onnx \
      --schedule "3:0,6:0.15,6:0.1:0:-0.4,6:0.1:0:0.4,3:0"

schedule: "sec:fwd:side:turn[:neck:head_pitch:head_yaw:head_roll[:policy]],..."
policy 1 runs --sit_onnx for that segment (fwd = sit flag), 0 the walk --onnx;
last_action carries over the switch, as in Microduck's infer_policy.py.
Prints per segment: body-frame speed, yaw rate, touch-downs/s, tilt; and
whether / when it fell. --out saves a log in the walk_*.pkl format.
"""
import argparse
import math
import sys
import pickle

import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model

import os
XML = os.environ.get("DUCK_XML", "src/mjlab_microduck/robot/openduck/robot_walk.xml")  # 11-10: DUCK_XML selects a model variant
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


# tasks/openduck_sitstand_env_cfg.py SIT_TARGET_LIMITS
SIT_LO, SIT_HI = TARGET_LO.copy(), TARGET_HI.copy()
SIT_LO[2], SIT_HI[11], SIT_HI[3], SIT_HI[12] = -1.22, 1.22, 1.7, 1.7  # 11-10: hips 1.35 -> 1.22 like the Pi (sim stop)

# robot/openduck_constants.py GROUND_MESHES
GROUND_MESHES = ("trunk_bottom", "body_front", "body_back", "left_cache", "right_cache",
                 "left_knee_to_ankle_left_sheet", "left_knee_to_ankle_right_sheet",
                 "foot_side", "foot_top", "left_roll_to_pitch", "right_roll_to_pitch",
                 "head_bot_sheet")


def parse_schedule(text):
    out = []
    for part in text.split(","):
        v = [float(x) for x in part.split(":")]
        out.append(v + [0.0] * (9 - len(v)))
    return out


UPPER_BODIES = ("trunk_base", "neck_pitch_assembly", "head_pitch_to_yaw", "neck_yaw_assembly", "head_assembly")


def build(vin, json_path, knee_range=None, kp_legs=20.0, kp_head=8.0, backlash_deg=0.0, hidden=False, flex=0.0, ground=False, knee_torque=1.0, kp_knee=None, knee_side="both"):
    from bam.mujoco import MujocoController

    class EncoderController(MujocoController):
        """Firmware PD closes on the output-side encoder: servo joint + its
        passive backlash joint (microduck_rl BacklashEncoderBamActuator)."""

        def update(self):
            if self.bl_q is None:
                super().update()
            else:
                qpos, qvel = self.mujoco_data.qpos, self.mujoco_data.qvel
                keep_q, keep_v = qpos[self.qpos_indexes].copy(), qvel[self.dof_indexes].copy()
                qpos[self.qpos_indexes] += qpos[self.bl_q]
                qvel[self.dof_indexes] += qvel[self.bl_v]
                super().update()
                qpos[self.qpos_indexes], qvel[self.dof_indexes] = keep_q, keep_v
            if knee_torque != 1.0:  # stronger knee servo what-if (STS3046 ~x2.05)
                for i, name in zip(self.act_indexes, self.actuator):
                    if name.endswith("_knee") and (knee_side == "both" or name.startswith(knee_side)):
                        self.mujoco_data.ctrl[i] *= knee_torque

    spec = mujoco.MjSpec.from_file(XML)
    if ground:  # same floor contacts as the sit/stand training model
        for g in spec.geoms:
            if g.classname.name == "visual" and g.meshname in GROUND_MESHES:
                g.contype, g.conaffinity, g.group = 1, 0, 3
    if os.environ.get("DUCK_FEET"):  # od34: bigger soles "in_mm:fwd_mm[:add_g]" (mjlab_openduck/feet/openduck_duck_feet.py)
        sys.path.insert(0, "/mnt/d/Claude/duck/mjlab_openduck/feet")
        from openduck_duck_feet import duck_feet
        v = [float(x) for x in os.environ["DUCK_FEET"].split(":")] + [20.0]
        print("duck feet box mm", {k: np.round(x, 1).tolist() for k, x in duck_feet(spec, v[0] / 1000, v[1] / 1000, v[2] / 1000).items()})
    if os.environ.get("DUCK_PLAY"):  # 11-11: knee/ankle roll play at the horn side, "deg:stiff:same|mirror[:knee+ankle]" (tools/knee_play_sim.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from knee_play_sim import add_play
        v = os.environ["DUCK_PLAY"].split(":") + ["knee"]
        print("play", add_play(spec, float(v[0]), float(v[1]), v[2], tuple(v[3].split("+"))))
    if os.environ.get("DUCK_KNEE_PRIM"):  # sit21b: primitive knee/body contacts (tools/knee_prim_sim.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from knee_prim_sim import add_knee_prim
        add_knee_prim(spec)
    if os.environ.get("DUCK_KNEE_V1C"):  # 11-10: STS3046 knee parts + body/knee contacts (tools/knee_v1c_sim.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from knee_v1c_sim import add_knee_v1c
        add_knee_v1c(spec)
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
    models = {"legs": load_model(json_path), "knee": load_model(json_path), "head": load_model(json_path)}
    models["legs"].actuator.kp, models["head"].actuator.kp = kp_legs, kp_head
    models["knee"].actuator.kp = kp_legs if kp_knee is None else kp_knee  # own P for a different knee servo
    force_limit = vin * models["legs"].kt.value / models["legs"].R.value
    names = {"legs": [], "knee": [], "head": []}
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        k = knee_torque if act.name.endswith("_knee") and (knee_side == "both" or act.name.startswith(knee_side)) else 1.0
        act.forcerange = (-force_limit * k, force_limit * k)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names["head" if ("neck" in act.name or "head" in act.name) else "knee" if act.name.endswith("_knee") else "legs"].append(act.name)
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
    for k in ("legs", "knee", "head"):
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
    ap.add_argument("--joint_stops", action="store_true", help="print stop/clip % per joint index")
    ap.add_argument("--clip", action="store_true", help="clip targets to TARGET_LO/HI (like the Pi)")
    ap.add_argument("--act_delay", type=int, default=0, help="extra control steps (20 ms) before a target reaches the servos")
    ap.add_argument("--imu_pitch_deg", type=float, default=0, help="IMU mounted tilted: + = reads nose-up (real duck: ~4)")
    ap.add_argument("--gyro_noise", type=float, default=0, help="uniform +- noise on gyro (rad/s)")
    ap.add_argument("--switch_n", type=float, default=None, help="od35: switch closes above this force (N); default 0.7 per lever switch (1 bit/foot = any of 3, or 3 per foot)")
    ap.add_argument("--switch_delay", type=int, default=0, help="od35: control steps the foot switches lag")
    ap.add_argument("--switch_fault", default="", help="od35 foot switch fault: stuck0L / stuck1L / stuck0R / stuck1R / flip:P")
    ap.add_argument("--acc_noise", type=float, default=0, help="uniform +- noise on accel (m/s^2)")
    ap.add_argument("--imu_roll_deg", type=float, default=0, help="IMU mounted rolled: + = accel y reads positive when level (real duck: ~2.5)")
    ap.add_argument("--friction", type=float, help="foot-floor sliding friction (XML default otherwise)")
    ap.add_argument("--head_mass", type=float, default=0, help="extra kg on head_assembly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp_legs", type=float, default=20.0)
    ap.add_argument("--knee_side", default="both", choices=("both", "left", "right"), help="which knee gets --knee_torque (11-08: only the left one is an STS3046)")
    ap.add_argument("--knee_torque", type=float, default=1.0, help="knee servo torque multiplier (stronger servo what-if, STS3046 ~2.05)")
    ap.add_argument("--kp_head", type=float, default=8.0)
    ap.add_argument("--kp_knee", type=float, help="knee firmware P (default = --kp_legs); STS3046 knees: half of kp_legs")
    ap.add_argument("--hidden_play", action="store_true", help="the play is outside the encoder (loose horn / flexing part): servo and obs see the motor side only")
    ap.add_argument("--flex", type=float, default=0.0, help="hidden spring per servo joint, Nm/rad (structural compliance)")
    ap.add_argument("--backlash", type=float, default=0.0, help="gear play per servo, +- deg (encoder on the output side; microduck_rl uses 1)")
    ap.add_argument("--zero_err", default="", help="servo zero errors, deg per joint index: '4:4,13:4' (encoder reads true+err)")
    ap.add_argument("--kick", default="", help="angular velocity kick 't:wx:wy:wz' (s, rad/s, body frame), e.g. the real start jerk '0.1:0:0.5:0'")
    ap.add_argument("--head_add", default="", help="extra mass on the head 'grams:forward_mm' (voice board)")
    ap.add_argument("--chest_add", default="", help="extra mass in the trunk 'grams:forward_mm' (speaker)")
    ap.add_argument("--ground", action="store_true", help="shins, trunk and head touch the floor (sit/stand models); fwd in the schedule = sit flag")
    ap.add_argument("--sit_onnx", help="sit/stand model for segments with policy 1 (switch without a pause)")
    ap.add_argument("--hang", action="store_true", help="trunk held still in the air, as on the stand")
    ap.add_argument("--com_x", type=float, default=0.0, help="move the whole-robot CoM forward by this (mm, negative = back; upper body bodies shifted)")
    ap.add_argument("--imu_delay", type=int, default=0, help="control steps (20 ms) the policy's gyro/accel lag behind the body")
    ap.add_argument("--foot_solref", default="", help="soft/bouncy foot contact 'timeconst:dampratio' (MuJoCo default 0.02:1), e.g. 0.04:0.2")
    ap.add_argument("--freeze_head", action="store_true", help="hold the 4 head joints at HOME whatever the policy asks")
    ap.add_argument("--head_log", action="store_true", help="print neck/head targets and pitch gyro for the first 1.5 s")
    ap.add_argument("--imu_yaw180", action="store_true", help="IMU turned 180 deg about z: gyro/accel x,y negated")
    ap.add_argument("--spec", action="store_true", help="11-11: measure the walk against the written gait spec G1-G11 (tools/gait_spec_meter.py)")
    ap.add_argument("--spec_json", help="also write the spec summary + fall time as JSON here")
    ap.add_argument("--gait", action="store_true", help="per segment: feet gap (heading frame), double support %, step length per foot, which foot leads")
    ap.add_argument("--new_head", action="store_true", help="real 11-07 masses: head 247 g (v2, no bottom plate), chest +58 g")
    ap.add_argument("--start_z", type=float, default=0.155, help="root height at t=0 (m)")
    ap.add_argument("--knee_clear", action="store_true", help="per segment (ground): lowest point of the shin meshes above the floor, end and min (mm)")
    a = ap.parse_args()
    json_path = f"{bam.__path__[0]}/params/feetech_sts3215_7_4V/m1.json"
    m, d, ctrls = build(a.vin, json_path, a.knee_range, a.kp_legs, a.kp_head, a.backlash, a.hidden_play, a.flex, a.ground or bool(a.sit_onnx), a.knee_torque, a.kp_knee, a.knee_side)
    for spec_add, body in ((a.head_add, "head_assembly"), (a.chest_add, "trunk_base")):
        if spec_add:
            g, dx = (float(v) for v in spec_add.split(":"))
            b, add = m.body(body).id, g / 1000
            old = m.body_mass[b]
            m.body_ipos[b, 0] += add * dx / 1000 / (old + add)
            m.body_inertia[b] *= (old + add) / old
            m.body_mass[b] = old + add
    if a.new_head:  # od27 (11-07): head v2 247 g whole (top cover only) + chest voice parts +58 g
        hb, tb = m.body("head_assembly").id, m.body("trunk_base").id
        m.body_inertia[hb] *= 0.247 / m.body_mass[hb]; m.body_mass[hb] = 0.247; m.body_ipos[hb] = [0.0163, 0.0, -0.0048]
        mt = m.body_mass[tb]; m.body_ipos[tb] = (mt * m.body_ipos[tb] + 0.058 * np.array([0.03, 0.0, 0.03])) / (mt + 0.058); m.body_mass[tb] = mt + 0.058
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
    walk_sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    walk_dim = walk_sess.get_inputs()[0].shape[1]
    n_switch = walk_dim - 61 if isinstance(walk_dim, int) and walk_dim > 61 else 0   # od35: 2 (1 bit / foot) or 6
    if a.switch_n is None:
        a.switch_n = 0.7
    sit_sess = ort.InferenceSession(a.sit_onnx, providers=["CPUExecutionProvider"]) if a.sit_onnx else None

    jids = [m.actuator(i).trnid[0] for i in range(m.nu)]
    qadr, vadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    if a.backlash and not a.hidden_play and not a.flex:
        bl = [m.joint("passive_" + m.joint(j).name + "_backlash").id for j in jids]
        bl_q, bl_v = m.jnt_qposadr[bl], m.jnt_dofadr[bl]
    else:
        bl_q = bl_v = None
    enc_q = lambda: d.qpos[qadr] + (d.qpos[bl_q] if bl_q is not None else 0)
    enc_v = lambda: d.qvel[vadr] + (d.qvel[bl_v] if bl_v is not None else 0)
    d.qpos[2] = a.start_z  # sit/stand training resets standing at 0.162-0.172
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
    n_stop_j, n_clip_j = np.zeros(14), np.zeros(14)  # the same per joint
    steps, rows = [], []
    gait_rows = []
    foot_sites = [m.site("left_foot").id, m.site("right_foot").id] if a.gait else None
    shin = []  # (geom id, mesh vertices) of the shin sheets, for --knee_clear
    if a.knee_clear:
        for gi in range(m.ngeom):
            mid = m.geom_dataid[gi]
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0 and m.mesh(mid).name.startswith("left_knee_to_ankle_"):
                shin.append((gi, m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))
    knee_j = [m.joint(n).id for n in ("left_knee", "right_knee")] if shin else []

    def shin_z():  # lowest shin-sheet point within 30 mm of a knee axis = the knee cap
        if not shin:
            return float("nan")
        best = 1.0
        for g, v in shin:
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            near = np.min([np.linalg.norm(w - d.xanchor[j], axis=1) for j in knee_j], axis=0)
            if np.any(near < 0.03):
                best = min(best, w[near < 0.03, 2].min())
        return best
    sw_buf = []
    _cf = np.zeros(6)

    _d0 = mujoco.MjData(m); _d0.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, _d0)
    _fwd = []
    for fg in feet:  # forward axis of each sole in its geom frame (robot faces +x at q0)
        _R = _d0.geom_xmat[fg].reshape(3, 3); _i = int(np.argmax(np.abs(_R[0]))); _e = np.zeros(3); _e[_i] = np.sign(_R[0, _i]); _fwd.append(_e)
    SW_D, SW_LINE = 0.024, -0.00865  # = mjlab_openduck/switch/openduck_foot_switch.py

    def foot_switch_bits():
        """od35 foot switches. n_switch 2: per foot, net sole force > switch_n. n_switch 6: 3 lever switches per
        foot (front, mid, rear) under a rigid plate, each closes above switch_n (same model as the training)."""
        F = np.zeros(2); M = np.zeros(2)
        for i in range(d.ncon):
            c = d.contact[i]
            g = {c.geom1, c.geom2}
            if floor not in g:
                continue
            for k, fg in enumerate(feet):
                if fg in g:
                    mujoco.mj_contactForce(m, d, i, _cf); fn = abs(_cf[0])
                    fw = d.geom_xmat[fg].reshape(3, 3) @ _fwd[k]
                    F[k] += fn; M[k] += fn * ((c.pos - d.geom_xpos[fg]) @ fw - SW_LINE)
        lever = F[:, None] / 3 + M[:, None] * np.array([SW_D, 0.0, -SW_D]) / (2 * SW_D ** 2)
        if n_switch == 6:
            b = (lever > a.switch_n).astype(np.float64).ravel()
        else:  # od36: the real board = 3 switches in parallel -> 1 bit per foot (any switch closed)
            b = (lever > a.switch_n).any(axis=1).astype(np.float64)
        f = a.switch_fault
        h = len(b) // 2
        if f.startswith("stuck"):
            sl = slice(0, h) if f.endswith("L") else slice(h, 2 * h)
            b[sl] = float(f[5])
        elif f.startswith("flip:"):
            b = np.where(rng.random(len(b)) < float(f[5:]), 1 - b, b)
        return b
    t, fell_at = 0.0, None
    for seg, (sec, fwd, side, turn, *head, pol) in enumerate(parse_schedule(a.schedule)):
        cmds = np.array([fwd, side, turn, *head])
        ground = a.ground or (pol == 1 and sit_sess is not None)
        sess = sit_sess if pol == 1 and sit_sess is not None else walk_sess
        prev_contact = None
        seg_rows = []
        seg_knee = []
        g_gap, g_ds, g_steps, g_lift, g_n = [], 0, [[], []], [None, None], 0  # gait stats for this segment
        g_order = []  # foot index of each lift-off, in order
        if a.spec:
            if seg == 0:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from gait_spec_meter import GaitSpecMeter, TARGETS
                meter = GaitSpecMeter(m, DT * DECIM)
            meter.new_segment(cmds, seg)
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
            if ground:  # the sit/stand task has gravity right after the gyro
                obs = np.concatenate([gyro, grav, pos_rel, vel, last_action, cmds[:3], cmds[3:7], np.zeros(6)]).astype(np.float32)
            else:
                obs = np.concatenate([gyro, pos_rel, vel, last_action, cmds[:3], grav, cmds[3:7], np.zeros(6)]).astype(np.float32)
                if n_switch and sess is walk_sess:
                    sw_buf.append(foot_switch_bits())
                    sw = sw_buf.pop(0) if len(sw_buf) > a.switch_delay else sw_buf[0]
                    obs = np.concatenate([obs, sw]).astype(np.float32)
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
            on_stop = (q < lim_lo + 0.02) | (q > lim_hi - 0.02)
            n_stop += bool(np.any(on_stop))
            n_stop_j += on_stop
            n_clip_j += (target < TARGET_LO - 0.02) | (target > TARGET_HI + 0.02)
            if a.clip:
                target = np.clip(target, SIT_LO, SIT_HI) if ground else np.clip(target, TARGET_LO, TARGET_HI)
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
            if a.spec:
                meter.step(d, contact)
            if a.gait:
                yaw = math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
                hrot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])  # world -> heading
                fp = [d.site_xpos[s][:2].copy() for s in foot_sites]
                g_gap.append((hrot @ (fp[0] - fp[1]))[1])  # left minus right, sideways; < 0 = crossed
                g_n += 1
                g_ds += int(contact[0] and contact[1])
                if prev_contact is not None:
                    for k in (0, 1):
                        if prev_contact[k] and not contact[k]:
                            g_lift[k] = fp[k]
                            g_order.append(k)
                        elif contact[k] and not prev_contact[k] and g_lift[k] is not None:
                            g_steps[k].append(hrot @ (fp[k] - g_lift[k]))
                            g_lift[k] = None
            prev_contact = contact
            quat = d.qpos[3:7]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            v_body = rot.T @ d.qvel[0:3]
            tilt = math.degrees(math.acos(max(-1, min(1, rot[2, 2]))))
            seg_rows.append((v_body[0], v_body[1], d.qvel[5], touch, tilt, d.qpos[2]))
            if shin and len(seg_rows) % 5 == 0:
                seg_knee.append(shin_z())
            legacy = np.concatenate([gyro, acc, np.r_[cmds, np.zeros(7 - len(cmds))], pos_rel, vel * 0.05,
                                     last_action, np.zeros(28), target, contact, np.zeros(2)])  # 83:97 = servo targets, as in the Pi logs
            steps.append((t, legacy))
            if fell_at is None and (tilt > 45 or (d.qpos[2] < 0.08 and not ground)):
                fell_at = t
                break
        r = np.array(seg_rows)
        if ground:
            print(f"seg {seg} cmd {fwd:.0f}: root z start {r[0, 5]:.3f} end {r[-5:, 5].mean():.3f}, tilt end {r[-5:, 4].mean():.1f} max {r[:, 4].max():.1f}, drop speed max {np.abs(np.diff(r[:, 5])).max() / (DT * DECIM):.2f} m/s"
                  + (f", knee clear end {shin_z() * 1000:.1f} min {min(seg_knee + [shin_z()]) * 1000:.1f} mm" if shin else ""))
        if a.spec:
            meter.end_segment()
        skip = min(25, len(r) // 3)
        rr = r[skip:] if len(r) > skip + 5 else r
        rows.append((seg, sec, fwd, side, turn, rr[:, 0].mean(), rr[:, 1].mean(), math.degrees(rr[:, 2].mean()),
                     rr[:, 3].sum() / (len(rr) * DT * DECIM), rr[:, 4].mean(), rr[:, 4].max()))
        if a.gait:
            gg = np.array(g_gap[skip:] if len(g_gap) > skip + 5 else g_gap) * 1000
            st = [np.array(s) * 1000 if s else np.zeros((0, 2)) for s in g_steps]
            # steps whose sideways move goes the commanded way: outside foot = left for side>0
            gait_rows.append((seg, fwd, side, turn, gg.min() if len(gg) else 0, gg.mean() if len(gg) else 0,
                              100 * g_ds / max(g_n, 1), *[(len(s), s[:, 0].mean() if len(s) else 0, s[:, 1].mean() if len(s) else 0) for s in st],
                              "".join("LR"[k] for k in g_order[:12])))
        if fell_at is not None:
            break

    print(f"{'seg':>3} {'cmd fwd':>7} {'side':>5} {'turn':>5} | {'vx':>6} {'vy':>6} {'yaw°/s':>7} {'steps/s':>7} {'tilt':>5} {'max':>5}")
    for seg, sec, fwd, side, turn, vx, vy, wz, tps, tm, tx in rows:
        print(f"{seg:3d} {fwd:7.2f} {side:5.2f} {turn:5.2f} | {vx:6.3f} {vy:6.3f} {wz:7.1f} {tps:7.2f} {tm:5.1f} {tx:5.1f}")
    if a.spec:
        summ = meter.summary()
        print("gait spec | target | measured")
        for k, v in summ.items():
            print(f"  {k:52s} | {TARGETS[k]:>10s} | " + (f"{v:.3g}" if isinstance(v, float) else str(v)))
        if a.spec_json:
            import json
            json.dump({"onnx": a.onnx, "fell_at": fell_at, "spec": summ}, open(a.spec_json, "w"))
    if a.gait:
        print("gait | feet gap mm (min/mean, <0 crossed) | double support % | steps: L n dx dy  R n dx dy (mm, heading frame) | lift-off order")
        for seg, fwd, side, turn, gmin, gmean, ds, sl, sr, order in gait_rows:
            print(f"{seg:3d} {fwd:5.2f} {side:5.2f} {turn:5.2f} | {gmin:6.1f} {gmean:6.1f} | {ds:5.1f} | "
                  f"L {sl[0]:2d} {sl[1]:6.1f} {sl[2]:6.1f}  R {sr[0]:2d} {sr[1]:6.1f} {sr[2]:6.1f} | {order}")
    print(f"fell at {fell_at:.1f} s" if fell_at else "no fall")
    print(f"beyond clip {100 * n_clip / max(n_ctrl, 1):.0f}% of steps, joint on its stop {100 * n_stop / max(n_ctrl, 1):.0f}%")
    if a.joint_stops:
        print("per joint % (stop/clip):", " ".join(
            f"{j}:{100 * s / max(n_ctrl, 1):.0f}/{100 * c / max(n_ctrl, 1):.0f}"
            for j, (s, c) in enumerate(zip(n_stop_j, n_clip_j)) if s or c))
    legs = np.array(action_steps)[100:][:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]
    print(f"action step rms {math.sqrt((legs ** 2).mean()):.2f} deg (legs, after the first 2 s)")
    if a.out:
        pickle.dump({"onnx": a.onnx, "steps": steps}, open(a.out, "wb"))


if __name__ == "__main__":
    main()
