"""Strict fore-aft symmetry (mirror x -> -x in the knee frame) of the knee parts as assembled: if exact, the mirrored
(symmetric) right-leg version of the part = the same part turned 180 deg about the leg axis -> reuse it."""
import numpy as np, trimesh
from scipy.spatial import cKDTree
for n in ("knee_shin_outer", "knee_shin_inner", "knee_cradle"):
    m = trimesh.load(f"/mnt/d/Claude/duck/cad/print_knee/{n}_assembly.stl")
    p, _ = trimesh.sample.sample_surface(m, 200000, seed=1)
    q, _ = trimesh.sample.sample_surface(m, 200000, seed=2)
    t = cKDTree(q)
    base = t.query(p)[0]; mir = t.query(p * [-1, 1, 1])[0]
    bad = p[mir > 0.6]
    print(f"{n:16s} noise 99.9% {np.percentile(base, 99.9):.2f} max {base.max():.2f} | x-mirror 99.9% {np.percentile(mir, 99.9):.2f} max {mir.max():.2f} mm; points off > 0.6 mm: {len(bad)}"
          + (f" around y {bad[:,1].min():.0f}..{bad[:,1].max():.0f} z {bad[:,2].min():.0f}..{bad[:,2].max():.0f} x {bad[:,0].min():.0f}..{bad[:,0].max():.0f}" if len(bad) else ""))
