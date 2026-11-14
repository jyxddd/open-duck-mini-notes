"""Symmetric legs (11-12): the right foot = mirror of the left foot. foot_top and foot_side are not mirror-symmetric
(tools/mirror_parts_check.py) -> mirrored copies (about the hole line y = 90 of the official parts; foot_bottom_pla /
foot_bottom_tpu are symmetric and reused). Output cad/print_feet/*_R_mirror.stl, in the official (assembly) frame
mirrored, then dropped to z = 0 - orient them in the slicer like the original left ones."""
import numpy as np, trimesh
P = "/home/zfhm/microduck_rl/Open_Duck_Mini/print/"; O = "/mnt/d/Claude/duck/cad/print_feet/"
for n in ("foot_top", "foot_side"):
    m = trimesh.load(P + n + ".stl")
    M = np.eye(4); M[1, 1] = -1; M[1, 3] = 180.0          # y -> 180 - y
    r = m.copy(); r.apply_transform(M)                      # trimesh fixes the winding of a reflection itself
    r.apply_translation([0, 0, -r.bounds[0][2]])
    r.export(O + n + "_R_mirror.stl")
    print(n, "watertight", r.is_watertight, "volume %.1f cm3 (orig %.1f)" % (r.volume / 1000, m.volume / 1000), "size", np.round(r.extents, 1))
