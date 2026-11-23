"""
age_error.py - how much does the output drift as the cache gets older?
Compares each policy's predicted masks (.npy) with No Cache, frame by frame, and groups by cache age.
For Every-N, refresh frames are 1, 1+N, 1+2N, ... so age = (frame - 1) % N.

Run from the implicit-depth repo root:
  python age_error.py out_p2_nocache out_p2_every5_r1:5 out_p2_every10:10 out_p2_every20:20 > age_error.txt
Metrics per frame: mean |mask difference| and % of pixels whose visible/hidden decision (0.5) flips.
"""
import os, sys, csv
import numpy as np

def frames(d):
    p = os.path.join(d, "render", "garden_chair")
    return p, sorted(f for f in os.listdir(p) if f.endswith(".npy"))

ref_dir, names = frames(sys.argv[1])
rows = []
for arg in sys.argv[2:]:
    d, n = arg.split(":"); n = int(n)
    pdir, _ = frames(d)
    by = {}
    for idx, f in enumerate(names):           # idx 0 = first frame (frame 1)
        a = np.load(os.path.join(ref_dir, f)); b = np.load(os.path.join(pdir, f))
        mad = float(np.abs(a - b).mean()); flip = float(((a > 0.5) != (b > 0.5)).mean() * 100)
        age = idx % n
        by.setdefault(age, []).append((mad, flip))
        rows.append([d, n, idx + 1, age, f"{mad:.6f}", f"{flip:.4f}"])
    print(f"\n{d} (Every-{n})")
    print(" age  frames  mean|diff|   %pixels flipped")
    for age in sorted(by):
        v = np.array(by[age]); print(f" {age:3d}  {len(v):6d}  {v[:,0].mean():10.5f}   {v[:,1].mean():8.3f}")
with open("age_error_per_frame.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "N", "frame", "age", "mean_abs_diff", "pct_pixels_flipped"]); w.writerows(rows)
print("\nwrote age_error_per_frame.csv")
