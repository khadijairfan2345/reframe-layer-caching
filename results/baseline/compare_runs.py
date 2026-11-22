import numpy as np, os, sys
a, b = sys.argv[1], sys.argv[2]
fs = sorted(os.listdir(a)); d = []
for f in fs:
    d.append(np.abs(np.load(os.path.join(a, f)) - np.load(os.path.join(b, f))).max())
print(len(fs), "frames; max abs diff", max(d), "mean of per-frame max", float(np.mean(d)))
