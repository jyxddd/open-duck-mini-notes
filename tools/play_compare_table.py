"""Summarise ~/play_cmp: per condition, mean over frictions x seeds of the left/right asymmetry measures."""
import glob, os, re, collections
import numpy as np
R = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.expanduser("~/play_cmp/*.txt"))):
    cond = os.path.basename(f).split("_fr")[0]
    fr = re.search(r"_fr([\d.]+)_", f).group(1)
    t = open(f).read()
    seg = {int(a): [float(x) for x in rest.split()] for a, rest in re.findall(r"^\s+(\d)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+ \|(.*)$", t, re.M) if "|" not in rest}
    gait = re.findall(r"^\s+(\d)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+ \|\s+([-\d.]+)\s+([-\d.]+) \|\s+([-\d.]+) \| L\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+R\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)", t, re.M)
    g = {int(x[0]): [float(v) for v in x[1:]] for x in gait}
    fell = "no fall" not in t
    if 1 not in seg or 1 not in g:
        R[cond].append(dict(fell=True, fr=fr)); continue
    s1, s2, s3 = seg[1], seg[2], seg[3]   # vx vy yaw steps tilt max
    R[cond].append(dict(fell=fell, fr=fr, yaw=s1[2], vy=s1[1] * 1000, tilt=s1[4], tmax=max(s1[5], s2[5], s3[5]),
                        turn_asym=abs(s2[2]) - abs(s3[2]), left=s2[2], right=s3[2],
                        dx_lr=g[1][4] - g[1][7], n_lr=g[1][3] - g[1][6], dy_l=g[1][5], dy_r=g[1][8]))
print("每种条件 18 次（摩擦 0.5/0.8/1.2 × 6 种子），直行 0.15 20 s + 边走边左转/右转各 10 s。均值 ± 标准差")
print(f"{'条件':14s} {'摔':>3s} | {'直行偏航°/s':>12s} {'直行侧移mm/s':>13s} {'步长差L-R mm':>13s} | {'左转°/s':>8s} {'右转°/s':>8s} {'左右转差':>9s} | {'倾斜°':>6s} {'最大':>5s}")
for cond in ("rigid", "same_k1.5", "mirror_k1.5", "same_ka1.5", "mirror_ka1.5", "same_ka3", "mirror_ka3"):
    rs = [r for r in R[cond] if "yaw" in r]
    nf = sum(r["fell"] for r in R[cond])
    f = lambda k: f"{np.mean([r[k] for r in rs]):+6.2f}±{np.std([r[k] for r in rs]):4.2f}"
    print(f"{cond:14s} {nf:3d} | {f('yaw'):>12s} {f('vy'):>13s} {f('dx_lr'):>13s} | {np.mean([r['left'] for r in rs]):8.1f} {np.mean([r['right'] for r in rs]):8.1f} {f('turn_asym'):>9s} | {np.mean([r['tilt'] for r in rs]):6.2f} {np.mean([r['tmax'] for r in rs]):5.1f}")
