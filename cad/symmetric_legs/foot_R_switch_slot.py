"""Right foot with the Heima switch-board slot (11-12). User: the printed right foot (official foot_top/foot_side
mirrored) is fine except it lacks the slot for the green "OpenDuck 脚底板" switch board (3 lever microswitches whose
levers rest on the floating yellow sole plate). Heima foot measured by the user: slot 10.5 mm wide, ~77 mm long
(= the length of the yellow-plate recess underneath), 4 board screw holes 20.5 mm (across) x 54 mm (along).
Official floor plate over the recess: z 8.5..16.2 above the bottom (tools probe), recess x about -45..32 in the
official LEFT-foot frame, centre line y 90 (the sole screw line). The official parts have two small windows
(x -35..-29 and 15..21, y 80..100) exactly where the board screws go -> filled first, then slot + holes.
Right foot = mirror y -> 180 - y (as tools/mirror_feet_parts.py). Run in WSL: ~/cad/bin/python foot_R_switch_slot.py
"""
import numpy as np
from build123d import *

R = "/mnt/d/Claude/duck/cad/foot_ref/"
OUT = "/mnt/d/Claude/duck/cad/print_feet/"
Z0 = -253.625                       # official bottom of the foot parts
FLOOR = (8.5, 16.2)                 # floor plate above the recess (z above Z0)
SLOT_W, SLOT_L, SLOT_XC, YC = 10.5, 66.0, -6.8, 90.0   # 66: must stay between the two joint screws (x -43.3 / 29.7, z 12)
HOLE_DX, HOLE_DY, HOLE_D = 54.0, 20.5, 1.8   # M2 self-tapping pilot


def solid(n):
    T = np.load(R + n + "_tris.npy")
    return Solid(Shell([Face(Wire.make_polygon([Vector(*p) for p in t], close=True)) for t in T]))


def zbox(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, Z0 + (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


ft, fs = solid("foot_top"), solid("foot_side")
for x0, x1 in ((-36.0, -28.0), (14.0, 22.0)):              # fill the two official windows
    ft = ft + zbox(x0, x1, 79.5, 100.5, *FLOOR)
cut = zbox(SLOT_XC - SLOT_L / 2, SLOT_XC + SLOT_L / 2, YC - SLOT_W / 2, YC + SLOT_W / 2, FLOOR[0] - 0.5, FLOOR[1] + 0.5)
for sx in (-1, 1):
    for sy in (-1, 1):
        cut = cut + Pos(SLOT_XC + sx * HOLE_DX / 2, YC + sy * HOLE_DY / 2, Z0 + sum(FLOOR) / 2) * Cylinder(HOLE_D / 2, FLOOR[1] - FLOOR[0] + 1)
ft, fs = ft - cut, fs - cut
# foot_side <-> foot_top joint screws (user: 统一): official foot_top has a Ø4.0 x 4.5 heat-set-insert pocket then Ø2.0;
# no inserts here -> fill both and make a plain M3 self-tapping pilot Ø2.6 x 18 deep from the joint face (y 79.2).
# foot_side keeps its Ø3.5 clearance + Ø6.5 head counterbore.
SEAM_Y = 79.2
for jx in (-43.3, 29.7):
    ycyl = lambda y0, y1, d: Pos(jx, (y0 + y1) / 2, Z0 + 12.0) * Rot(90, 0, 0) * Cylinder(d / 2, y1 - y0)
    ft = ft + ycyl(SEAM_Y, SEAM_Y + 22.0, 4.3)
    ft = ft - ycyl(SEAM_Y - 0.5, SEAM_Y + 18.0, 2.6)
M = Plane(origin=(0, YC, 0), z_dir=(0, 1, 0))               # y -> 180 - y
for name, p in (("foot_top", ft), ("foot_side", fs)):
    p = p.solids()[0] if len(p.solids()) == 1 else p
    r = mirror(p, about=M)
    r = Pos(0, 0, -r.bounding_box().min.Z) * r                # drop to z 0 like the earlier *_R_mirror.stl
    print(name, "solids", len(r.solids()), "valid", r.is_valid, "vol %.1f cm3" % (r.volume / 1000))
    export_stl(r, OUT + name + "_R_mirror_slot.stl", tolerance=0.02, angular_tolerance=0.2)
print("written")
