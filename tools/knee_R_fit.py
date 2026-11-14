"""Fit check of the symmetric right cradle (knee frame mm): STS3046 body (mirrored z -> 36.9 - z) and the real thigh
sheets must not overlap the cradle; same numbers for the left cradle with the unmirrored servo for reference.
Overlap = cradle surface sample points inside the other solid (and vice versa), deeper than 0.2 mm."""
import numpy as np, trimesh
K = "/mnt/d/Claude/duck/cad/knee_ref/"; P = "/mnt/d/Claude/duck/cad/print_knee/"
ZM = 18.45
def mz(m):
    m = m.copy(); m.apply_transform(np.diag([1, 1, -1, 1])); m.apply_translation([0, 0, 2 * ZM]); m.fix_normals(); return m
srv = trimesh.load(K + "sts3046_kf.stl")
for nm in ("sts3046_kf", "thigh_outer_sheet", "thigh_inner_sheet"):
    print(nm, "faces", len(trimesh.load(K + nm + ".stl").faces)); tout = trimesh.load(K + "thigh_outer_sheet.stl"); tin = trimesh.load(K + "thigh_inner_sheet.stl")
crL = trimesh.load(P + "knee_cradle_assembly.stl"); crR = trimesh.load(P + "knee_cradle_R_sym_assembly.stl")
so = trimesh.load(P + "knee_shin_outer_assembly.stl"); si = trimesh.load(P + "knee_shin_inner_assembly.stl")
def winding(pts, m, chunk=400):
    """generalised winding number of closed mesh m at pts (no rtree needed): ~1 inside, ~0 outside"""
    T = m.triangles; out = np.zeros(len(pts))
    for i in range(0, len(pts), chunk):
        a = T[None, :, 0, :] - pts[i:i + chunk, None, :]; b = T[None, :, 1, :] - pts[i:i + chunk, None, :]; c = T[None, :, 2, :] - pts[i:i + chunk, None, :]
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("pij,pij->pi", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("pij,pij->pi", a, b) * lc + np.einsum("pij,pij->pi", b, c) * la + np.einsum("pij,pij->pi", c, a) * lb
        out[i:i + chunk] = (2 * np.arctan2(det, den)).sum(1) / (4 * np.pi)
    return np.abs(out)
def overlap(a, b, n=6000, depth=0.2):
    """surface points of a pushed 'depth' mm into a that lie inside b"""
    pa, fi = trimesh.sample.sample_surface(a, n, seed=1)
    pa = pa - depth * a.face_normals[fi]
    lo, hi = b.bounds; near = np.all((pa > lo - 1) & (pa < hi + 1), axis=1)
    if not near.any(): return 0
    return int((winding(pa[near], b) > 0.5).sum())
clip = lambda m: m.slice_plane([0, 19.0, 0], [0, 1, 0])   # thigh sheets above y 19 only (as in the CAD clip)
print("sheet y ranges", tout.bounds[:, 1], tin.bounds[:, 1])
for name, cr, s in (("LEFT  cradle / servo", crL, srv), ("RIGHT cradle / servo mirrored", crR, mz(srv))):
    print(f"{name:32s}: cradle pts in servo {overlap(cr, s)}, servo pts in cradle {overlap(s, cr)}", flush=True)
for name, cr in (("LEFT  cradle", crL), ("RIGHT cradle", crR)):
    print(f"{name} vs thigh sheets: in outer {overlap(cr, tout)}, in inner {overlap(cr, tin)}", flush=True)
# shin parts (turned = mirrored, fore-aft symmetric) vs the thigh sheets and the cradle at knee 0 and bent
for name, a, b, c in (("LEFT ", so, si, crL), ("RIGHT", mz(so), mz(si), crR)):
    for deg in (0, 45, 90):
        R = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])
        A, B = a.copy(), b.copy(); A.apply_transform(R); B.apply_transform(R)
        print(f"{name} knee {deg:2d} deg: shin_out in sheets {overlap(A, tout) + overlap(A, tin)}, shin_in in sheets {overlap(B, tout) + overlap(B, tin)}, "
              f"shins in cradle {overlap(A, c) + overlap(B, c)}", flush=True)
