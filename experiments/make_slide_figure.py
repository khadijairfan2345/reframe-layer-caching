"""
make_slide_figure.py - build a clean slide figure from an existing results/step_times.csv
(periodic refresh vs. the strips setting with the lowest p95). No re-run needed.

Run from the folder that contains "results/":
    python experiments/make_slide_figure.py
    python experiments/make_slide_figure.py --csv results/step_times.csv --steps 40
"""
import argparse, csv, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = math.floor(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/step_times.csv")
ap.add_argument("--steps", type=int, default=40, help="how many steps to show")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
cols = {k[:-3]: [float(r[k]) for r in rows] for k in rows[0] if k.endswith("_ms")}
strips = [k for k in cols if k.startswith("strips")]
best = min(strips, key=lambda k: pct(cols[k][1:], 95))
g = best[6:].split("_")[0]
per, rot = cols["periodic"], cols[best]

n = min(a.steps, len(per))
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(1, n), per[1:n], color="#d1495b", lw=2, marker="o", ms=3, label="Periodic refresh (ReFrame)")
ax.plot(range(1, n), rot[1:n], color="#2e86ab", lw=2, marker="o", ms=3, label=f"Rotating refresh ({g} strips)")
ax.set_ylim(0, 1.3 * max(pct(per[1:], 99), pct(rot[1:], 99)))
ax.set_xlabel("step"); ax.set_ylabel("step latency (ms)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
out = a.out or os.path.join(os.path.dirname(a.csv), "slide_figure.png")
fig.savefig(out, dpi=200)
print(f"saved {out}")
print(f"periodic        p95 = {pct(per[1:], 95):.0f} ms   mean = {sum(per[1:])/len(per[1:]):.0f} ms")
print(f"{g} strips       p95 = {pct(rot[1:], 95):.0f} ms   mean = {sum(rot[1:])/len(rot[1:]):.0f} ms")