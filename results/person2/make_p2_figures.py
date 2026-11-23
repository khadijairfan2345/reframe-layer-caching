"""make_p2_figures.py - Person 2 slide figures from the recorded CSVs (no GPU).
Run from results/person2/:  python make_p2_figures.py"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else "."
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
tab = {r["policy"]: r for r in csv.DictReader(open(os.path.join(D, "p2_summary_table.csv")))}
g = lambda p, k: float(tab[p][k])

# 1. trade-off: worst frame vs accuracy
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ev = ["Every-5 (ReFrame)", "Every-10", "Every-20"]; pp = ["pipe3d-3", "pipe3d-5", "pipe3d-10"]
ax.plot([g(p, "max_ms") for p in ev], [g(p, "mean_pct_px_flipped") for p in ev], "o", ms=9, color=BLUE,
        mec="#fcfcfb", mew=2, label="Every-N refresh (with or without skipping the waste)")
ax.plot([g(p, "max_ms") for p in pp], [g(p, "mean_pct_px_flipped") for p in pp], "-o", lw=2, ms=9, color=ORANGE,
        mec="#fcfcfb", mew=2, label="pipe3d-G (ours: spread refresh over G frames)")
ax.plot([g("No Cache", "max_ms")], [0], "o", ms=9, color=GRAY, mec="#fcfcfb", mew=2, label="No Cache")
for p, lab, dx, dy in [("Every-5 (ReFrame)", "Every-5", -6, 0), ("Every-10", "Every-10", -6, 0), ("Every-20", "Every-20", -6, 0),
                       ("pipe3d-3", "G=3", 2.5, -0.05), ("pipe3d-5", "G=5", 2.5, -0.05), ("pipe3d-10", "G=10", 2.5, -0.05),
                       ("No Cache", "No Cache", -6, 0.05)]:
    x, y = g(p, "max_ms"), (0 if p == "No Cache" else g(p, "mean_pct_px_flipped"))
    ax.annotate(lab, (x + dx, y + dy), ha="right" if dx < 0 else "left", va="center", color=INK, fontsize=10)
ax.set_xlim(0, 120); ax.set_ylim(-0.2, 4)
ax.set_xlabel("slowest frame (ms)  -  lower is smoother"); ax.set_ylabel("pixels different from No Cache (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(D, "p2_tradeoff.png"), dpi=200); plt.close(fig)

# 2. per-frame latency trace
def tot(f): return [float(r["TOTAL"]) for r in csv.DictReader(open(os.path.join(D, f))) if int(r["frame"]) > 1]
series = [("Every-5 (ReFrame as written)", "20261122_1716/every5_r1_part_times.csv", BLUE),
          ("Every-5 + skip unused 3D work", "20261122_1814_3d/every5_skip_part_times.csv", AQUA),
          ("pipe3d-5 (ours)", "20261122_1814_3d/pipe3d5_r1_part_times.csv", ORANGE)]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.grid(axis="y", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
n = 40
for lab, f, c in series:
    v = tot(f)[:n]; ax.plot(range(2, 2 + len(v)), v, lw=2, color=c, label=lab)
ax.set_ylim(0, 125); ax.set_xlabel("frame"); ax.set_ylabel("frame time (ms)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(); fig.savefig(os.path.join(D, "p2_latency_trace.png"), dpi=200); plt.close(fig)
print("saved p2_tradeoff.png, p2_latency_trace.png")


# ---- slide-box versions (same shapes as the image boxes in the deck) ----
fig, ax = plt.subplots(figsize=(8.4, 3.1))
ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.plot([g(p, "max_ms") for p in ev], [g(p, "mean_pct_px_flipped") for p in ev], "o", ms=9, color=BLUE, mec="#fcfcfb", mew=2, label="Every-N (ReFrame)")
ax.plot([g(p, "max_ms") for p in pp], [g(p, "mean_pct_px_flipped") for p in pp], "-o", lw=2, ms=9, color=ORANGE, mec="#fcfcfb", mew=2, label="pipe3d-G (ours)")
ax.plot([g("No Cache", "max_ms")], [0], "o", ms=9, color=GRAY, mec="#fcfcfb", mew=2, label="No Cache")
for p, lab, dx, dy in [("Every-5 (ReFrame)", "N=5", -3, 0), ("Every-10", "N=10", -3, 0), ("Every-20", "N=20", -3, 0),
                       ("pipe3d-3", "G=3", 2.5, -0.1), ("pipe3d-5", "G=5", 2.5, -0.1), ("pipe3d-10", "G=10", 2.5, -0.1), ("No Cache", "No Cache", -3, 0.1)]:
    x, y = g(p, "max_ms"), (0 if p == "No Cache" else g(p, "mean_pct_px_flipped"))
    ax.annotate(lab, (x + dx, y + dy), ha="right" if dx < 0 else "left", va="center", color=INK, fontsize=10)
ax.set_xlim(0, 120); ax.set_ylim(-0.3, 4)
ax.set_xlabel("slowest frame (ms)"); ax.set_ylabel("pixels changed (%)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 0.03), ncol=1, fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(D, "slide3_tradeoff.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.9, 3.1))
ax.grid(axis="y", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
for lab, f, c in series[:2]:
    v = tot(f)[:30]; ax.plot(range(2, 2 + len(v)), v, lw=2, color=c, label=lab)
ax.set_ylim(0, 130); ax.set_xlabel("frame"); ax.set_ylabel("frame time (ms)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(os.path.join(D, "slide2_spikes.png"), dpi=200); plt.close(fig)
print("saved slide3_tradeoff.png, slide2_spikes.png")
