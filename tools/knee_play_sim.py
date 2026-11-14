"""11-11 user: 左右膝盖的动力盘应该调整为对称，而不是现在的顺拐. The official legs are identical copies (not mirrored):
knee and ankle horns are on the OUTER side of the left leg and the INNER side of the right leg (hip pitch is already
mirrored). Rigid sim: the two legs have the same mass and CoM within 1.5 mm (tools/leg_symmetry.py), so the layout
only matters through stiffness: the horn side is bolted (4 screws), the idler side rides on a bearing with play.
Model: a passive roll hinge (about the fore-aft axis) on the shin (knee) and/or foot (ankle) body, pivoting at the
HORN-side sheet, +-deg play with a weak spring inside. Layout "same" = as built now (left pivot outer, right pivot
inner); "mirror" = symmetric (both horns outer: the right pivot moves to the other sheet, 38.5 mm along the axis).
Hook: DUCK_PLAY="deg:stiff_Nm_per_rad:same|mirror[:knee+ankle]" in tools/mjlab_sim_check.py."""
import math

import mujoco
import numpy as np

SHEET_GAP = 0.0385  # m between the two knee sheets along the joint axis (sim meshes: y 70.6 / 109.1)


def add_play(spec, deg=1.0, stiff=0.3, layout="same", joints=("knee",)):
    m = spec.compile(); d = mujoco.MjData(m); d.qpos[3] = 1; mujoco.mj_kinematics(m, d)
    out, todo = [], []
    for jn in [f"{s}_{k}" for k in joints for s in ("left", "right")]:   # read everything from the compiled model first
        jid = m.joint(jn).id; bid = m.jnt_bodyid[jid]                    # (spec handles go stale after add_joint)
        R = d.xmat[bid].reshape(3, 3)
        ax = m.jnt_axis[jid] / np.linalg.norm(m.jnt_axis[jid])
        pos = m.jnt_pos[jid].copy()
        if layout == "mirror" and jn.startswith("right"):
            pos = pos + SHEET_GAP * ax          # horn moved to the outer sheet
        todo.append((jn, m.body(bid).name, R.T @ np.array([1.0, 0, 0]), pos))
        out.append(f"{jn}: pivot y {1000 * (d.xpos[bid] + R @ pos)[1]:+.1f} mm")
    for jn, bname, axis, pos in todo:
        p = spec.body(bname).add_joint()
        p.name = f"passive_{jn}_play"
        p.type = mujoco.mjtJoint.mjJNT_HINGE
        p.axis = axis                            # roll: about the world fore-aft axis (at qpos 0)
        p.pos = pos
        p.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        p.range = [-math.radians(deg), math.radians(deg)]
        p.armature = 0.001
        p.stiffness = np.array([stiff, 0, 0]).reshape(3, 1) if np.ndim(p.stiffness) else stiff
        p.damping = np.full((3, 1), 0.02)
        p.solref_limit = (0.01, 1.0)
    return out
