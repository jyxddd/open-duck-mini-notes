"""Symmetric right knee: target right cradle (world) = mirror of the left cradle across the robot's mid plane.
Express that mirror in the right cradle's own STL frame (knee frame, mm): A = T_R^-1 M_y T_L. Also check that the
right thigh sheets sit in the right cradle frame where the left ones sit in the left frame."""
import sys
import mujoco, numpy as np
sys.path.insert(0, "/mnt/d/Claude/duck/tools")
from knee_v1c_sim import XML, add_knee_v1c, copy_assets
import os
spec = mujoco.MjSpec.from_file(XML); copy_assets(os.path.join(os.path.dirname(XML), spec.meshdir or "assets"))
add_knee_v1c(spec, pad=0)
m = spec.compile(); d = mujoco.MjData(m); d.qpos[3] = 1; mujoco.mj_kinematics(m, d)
T = {}
for leg in ("left", "right"):
    g = m.geom(f"{leg}_kv1c_cradle").id; mid = m.geom_dataid[g]
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, m.mesh_quat[mid]); Rm = Rm.reshape(3, 3)
    # STL point p (mm): geom-local = Rm^T (0.001 p - mesh_pos)  ->  world = xpos + xmat Rm^T (0.001 p - mesh_pos)
    Rg = d.geom_xmat[g].reshape(3, 3) @ Rm.T
    T[leg] = (1000 * (d.geom_xpos[g] - Rg @ m.mesh_pos[mid]), Rg)
M = np.diag([1, -1, 1])
(xL, RL), (xR, RR) = T["left"], T["right"]
A = RR.T @ M @ RL; t = RR.T @ (M @ xL - xR)
np.set_printoptions(suppress=True, precision=3)
print("A_lin =\n", A, "\ndet", np.linalg.det(A), "\nt =", t)
# leg mid planes (world y of the thigh sheet meshes) to see the L/R offset
for leg, b in (("left", "knee_and_ankle_assembly"), ("right", "knee_and_ankle_assembly_3")):
    ys = []
    for gi in range(m.ngeom):
        if m.body(m.geom_bodyid[gi]).name == b and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH and "sheet" in m.mesh(m.geom_dataid[gi]).name:
            ys.append(1000 * d.geom_xpos[gi][1])
    print(leg, "thigh sheet geom y", np.round(ys, 1))
# the cradle frame z of the knee-frame thigh sheets: where are the right thigh sheets in the right cradle frame?
for leg, b in (("left", "knee_and_ankle_assembly"), ("right", "knee_and_ankle_assembly_3")):
    x0, R0 = T[leg]
    for gi in range(m.ngeom):
        if m.body(m.geom_bodyid[gi]).name == b and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH and "sheet" in m.mesh(m.geom_dataid[gi]).name:
            mid = m.geom_dataid[gi]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = 1000 * d.geom_xpos[gi] + 1000 * v @ d.geom_xmat[gi].reshape(3, 3).T
            loc = (w - x0) @ R0   # STL (knee) frame of the cradle
            print(f"  {leg} {m.mesh(mid).name:32s} in cradle frame: x {loc[:,0].min():6.1f}..{loc[:,0].max():6.1f} y {loc[:,1].min():6.1f}..{loc[:,1].max():6.1f} z {loc[:,2].min():6.1f}..{loc[:,2].max():6.1f}")
