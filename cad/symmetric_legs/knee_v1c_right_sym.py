"""RIGHT knee cradle for the SYMMETRIC legs (11-12, user: 膝盖和脚踝动力盘改对称，一起改).
Before: the right knee module was the left one turned 180 deg (identical parts) -> knee horn on the INNER side of the
right leg (顺拐). Target: the right leg = mirror of the left leg (horns outer on both legs).
Frames (tools/knee_mirror_frame.py, from the sim): in the cradle's own knee frame the right thigh sheets sit exactly
where the left ones sit (outer z -4..4.45, inner z 33.6..40.9), and "mirror to the right leg" = z -> 36.9 - z.
The shin parts and the cradle are strictly fore-aft symmetric (tools/knee_fore_aft_check.py, max 0.5 mm = noise), so
the mirrored shin_out / shin_in / cap = the SAME parts turned 180 deg about the leg axis (reuse them).
The cradle is fixed to the thigh sheets, which are NOT mirrored, so it is rebuilt: the left cradle body (servo
pocket, ears, inner plate, M4 boss) mirrored z -> 36.9 - z, then the real right thigh sheets cut out and the M3
holes put where the right thigh sheets have them (outer sheet y 29.0, inner sheet y 32.75, same as the left).
Same construction code as knee_v1.py (cradle section). Run in WSL: ~/cad/bin/python knee_v1c_right_sym.py
"""
import numpy as np
from build123d import *

R = "/mnt/d/Claude/duck/cad/knee_ref/"
OUT = "/mnt/d/Claude/duck/cad/print_knee/"
HORN_Z, BACK_Z = -9.85, 39.15
EAR_Z0, EAR_Z1 = 3.85, 6.15
EAR_HW, EAR_Y1 = 9.25, 17.225
CASE_HW = 12.4
ZM = 36.9 / 2                                   # mirror plane (mid plane between the thigh sheets' outer faces)


def official(name, clip):
    T = np.load(R + name.replace("_sheet", "") + "_tris.npy")
    sol = Solid(Shell([Face(Wire.make_polygon([Vector(*p) for p in t], close=True)) for t in T]))
    return sol & clip


def zbox(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def zcyl(x, y, z0, z1, d):
    return Pos(x, y, (z0 + z1) / 2) * Cylinder(d / 2, z1 - z0)


TCLIP = zbox(-40, 40, 0, 42, -10, 50)
thigh_out, thigh_in = official("thigh_outer_sheet", TCLIP), official("thigh_inner_sheet", TCLIP)

# --- left cradle body, servo-side features only (knee_v1.py), WITHOUT the sheet pockets / thigh-sheet holes
cr = zbox(-CASE_HW, CASE_HW, 10.6, 35.1, 1.9, 37.7)
cr -= zbox(-EAR_HW - 0.35, EAR_HW + 0.35, 10.0, EAR_Y1 + 0.4, 1.0, EAR_Z1)
cr += zbox(-CASE_HW, CASE_HW, 10.6, 18.6, 37.0, BACK_Z)
cr += zbox(-CASE_HW, CASE_HW, -32.25, 18.6, BACK_Z, BACK_Z + 4.0)
for s in (1, -1):
    cr += zbox(min(s * 10.25, s * CASE_HW), max(s * 10.25, s * CASE_HW), -32.25, 10.6, BACK_Z - 4.15, BACK_Z)
cr += zbox(-CASE_HW, CASE_HW, -32.25, -30.25, BACK_Z - 4.15, BACK_Z)
B_SEAT = 45.3
cr += zcyl(0, 0, BACK_Z + 4.0, B_SEAT, 5.6)
cr -= zcyl(0, 0, BACK_Z + 0.8, B_SEAT + 0.1, 3.4)
for s in (1, -1):
    cr -= zcyl(s * 5, 13.9, EAR_Z1 - 0.1, EAR_Z1 + 10, 2.7)

# --- mirror to the right leg (z -> 36.9 - z)
crR = mirror(cr, about=Plane(origin=(0, 0, ZM), z_dir=(0, 0, 1)))
# beside the thigh sheets (y >= 19) reach into the outer sheet only as deep as the left cradle does (z >= 1.9),
# otherwise the mirrored block leaves loose slivers in the sheet's openings
crR -= zbox(-20, 20, 19.0, 50, -20, 1.9)
# --- thigh-sheet M3 holes where the RIGHT thigh sheets have them (same as the left: outer y 29.0, inner y 32.75)
for s in (1, -1):
    crR -= zcyl(s * 10.25, 29.0, 1.8, 11.0, 2.7)
    crR -= zcyl(s * 10.25, 32.75, 28.5, 37.8, 2.7)
CLR = 0.15
for sh in (thigh_out, thigh_in):
    for d in ((0, 0, 0), (CLR, 0, 0), (-CLR, 0, 0), (0, CLR, 0), (0, -CLR, 0), (0, 0, CLR), (0, 0, -CLR)):
        crR = crR - Pos(*d) * sh
assert len(crR.solids()) == 1, [round(x.volume) for x in crR.solids()]
crR = crR.solids()[0]
print("right cradle %.1f cm3 valid %s solids %d bbox %s" % (crR.volume / 1000, crR.is_valid, len(crR.solids()), crR.bounding_box()))
export_stl(crR, OUT + "knee_cradle_R_sym_assembly.stl", tolerance=0.02, angular_tolerance=0.2)
export_stl(Rot(0, -90, 0) * crR, OUT + "knee_cradle_R_sym.stl", tolerance=0.02, angular_tolerance=0.2)
print("written")

if __name__ == "__main__":
    for i, s in enumerate(crR.solids()):
        b = s.bounding_box()
        print(f"solid {i}: {s.volume:.0f} mm3  x {b.min.X:.1f}..{b.max.X:.1f} y {b.min.Y:.1f}..{b.max.Y:.1f} z {b.min.Z:.1f}..{b.max.Z:.1f}")
    pass
