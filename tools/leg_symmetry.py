"""Left vs right leg in the sim model: per body mass, CoM (in world at qpos 0, right mirrored y -> -y), joint axes,
and where the knee servo / horn meshes sit. Identical (translated, not mirrored) legs show up as CoM y offsets
that do NOT mirror."""
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/openduck/robot_walk.xml"); d = mujoco.MjData(m)
d.qpos[3] = 1; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
names = [m.body(i).name for i in range(m.nbody)]
print("bodies:", names)
L = [n for n in names if n.startswith("left") or "_left" in n or n.endswith("_l")]
for n in names:
    if "left" not in n: continue
    r = n.replace("left", "right")
    if r not in names: continue
    a, b = m.body(n), m.body(r)
    ca, cb = d.xipos[a.id], d.xipos[b.id]
    pa, pb = d.xpos[a.id], d.xpos[b.id]
    print(f"{n:28s} mass L {1000*a.mass[0]:6.1f} R {1000*b.mass[0]:6.1f} g | CoM L y {1000*ca[1]:7.1f} R(-y) {-1000*cb[1]:7.1f}  x L {1000*ca[0]:6.1f} R {1000*cb[0]:6.1f}  z L {1000*ca[2]:6.1f} R {1000*cb[2]:6.1f} | frame y L {1000*pa[1]:6.1f} R(-y) {-1000*pb[1]:6.1f}")
for j in range(m.njnt):
    n = m.joint(j).name
    if "left" in n:
        r = n.replace("left", "right"); jr = m.joint(r).id
        print(f"joint {n:18s} axis(world) L {np.round(d.xaxis[j],3)} R {np.round(d.xaxis[jr],3)}  anchor y L {1000*d.xanchor[j][1]:6.1f} R {1000*d.xanchor[jr][1]:6.1f}")
for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mn = m.mesh(m.geom_dataid[g]).name; bn = m.body(m.geom_bodyid[g]).name
        if any(k in mn.lower() for k in ("sts", "horn", "servo", "knee", "idler", "drive")) and ("knee" in bn or "tibia" in bn or "shin" in bn or "thigh" in bn or "femur" in bn or True):
            c = d.geom_xpos[g]
            print(f"  mesh {mn:30s} body {bn:22s} at x {1000*c[0]:6.1f} y {1000*c[1]:7.1f} z {1000*c[2]:6.1f}")
print("--- unnamed leg bodies (left, right)")
for a, b in (("hip_roll_assembly", "hip_roll_assembly_2"), ("left_roll_to_pitch_assembly", "right_roll_to_pitch_assembly"),
             ("knee_and_ankle_assembly", "knee_and_ankle_assembly_3"), ("knee_and_ankle_assembly_2", "knee_and_ankle_assembly_4"),
             ("foot_assembly", "foot_assembly_2")):
    A, B = m.body(a), m.body(b); ca, cb = d.xipos[A.id], d.xipos[B.id]
    # lateral CoM relative to the leg centre line (mid of the two knee sheets ~ y 90.2 / -91.0)
    print(f"{a:28s} vs {b:30s} mass {1000*A.mass[0]:6.1f}/{1000*B.mass[0]:6.1f} g  CoM y L {1000*ca[1]:6.1f} R(-y) {-1000*cb[1]:6.1f}  x {1000*ca[0]:6.1f}/{1000*cb[0]:6.1f} z {1000*ca[2]:6.1f}/{1000*cb[2]:6.1f}  inertia diag L {np.round(1e6*m.body_inertia[A.id],1)} R {np.round(1e6*m.body_inertia[B.id],1)}")
tot = lambda ids: (sum(m.body_mass[i] for i in ids), sum(m.body_mass[i] * d.xipos[i] for i in ids) / sum(m.body_mass[i] for i in ids))
lm, lc = tot([m.body(n).id for n in ("hip_roll_assembly", "left_roll_to_pitch_assembly", "knee_and_ankle_assembly", "knee_and_ankle_assembly_2", "foot_assembly")])
rm, rc = tot([m.body(n).id for n in ("hip_roll_assembly_2", "right_roll_to_pitch_assembly", "knee_and_ankle_assembly_3", "knee_and_ankle_assembly_4", "foot_assembly_2")])
print(f"whole leg: L {1000*lm:.1f} g CoM y {1000*lc[1]:.1f} | R {1000*rm:.1f} g CoM -y {-1000*rc[1]:.1f}; robot total {1000*m.body_mass.sum():.0f} g, CoM y {1000*d.subtree_com[1][1]:.2f} mm")
