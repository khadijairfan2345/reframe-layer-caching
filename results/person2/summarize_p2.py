"""summarize_p2.py - one table of latency (ms per frame) and accuracy (vs No Cache) for every policy Person 2 ran."""
import csv, glob, os, numpy as np
A, B = "20261122_1716", "20261122_1814_3d"
runs = [("No Cache", f"{A}/nocache_part_times.csv", None),
        ("Every-5 (ReFrame)", f"{A}/every5_r1_part_times.csv", ("out_p2_every5_r1", f"{A}/age_error_per_frame.csv")),
        ("Every-10", f"{A}/every10_part_times.csv", ("out_p2_every10", f"{B}/every10_accuracy_per_frame.csv")),
        ("Every-20", f"{A}/every20_part_times.csv", ("out_p2_every20", f"{B}/every20_accuracy_per_frame.csv")),
        ("Delta-L", f"{A}/deltaL_part_times.csv", None),
        ("pipe-5 (blocks, keeps 3D waste)", f"{A}/pipe5_r1_part_times.csv", ("out_p2_pipe5_r1", f"{B}/pipe5_accuracy_per_frame.csv")),
        ("Every-5 + skip waste", f"{B}/every5_skip_part_times.csv", ("out_p2_every5_r1", f"{A}/age_error_per_frame.csv")),
        ("pipe3d-3", f"{B}/pipe3d3_r1_part_times.csv", ("out_p2_pipe3d3_r1", f"{B}/pipe3d3_accuracy_per_frame.csv")),
        ("pipe3d-5", f"{B}/pipe3d5_r1_part_times.csv", ("out_p2_pipe3d5_r1", f"{B}/pipe3d5_accuracy_per_frame.csv")),
        ("pipe3d-10", f"{B}/pipe3d10_r1_part_times.csv", ("out_p2_pipe3d10_r1", f"{B}/pipe3d10_accuracy_per_frame.csv"))]
rows = []
print(f"{'policy':34s} {'mean':>6s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'max':>6s}  {'%px flipped':>11s} {'worst':>6s}")
for name, pt, acc in runs:
    t = np.array([float(r["TOTAL"]) for r in csv.DictReader(open(pt)) if int(r["frame"]) > 1])
    fl = wf = float("nan")
    if acc:
        v = [float(r["pct_pixels_flipped"]) for r in csv.DictReader(open(acc[1])) if r["run"] == acc[0] and int(r["frame"]) > 1]
        fl, wf = np.mean(v), np.max(v)
    if name == "No Cache": fl = wf = 0.0
    s = [t.mean(), *np.percentile(t, [50, 95, 99]), t.max()]
    rows.append([name] + [f"{x:.1f}" for x in s] + [f"{fl:.2f}", f"{wf:.2f}"])
    print(f"{name:34s} " + " ".join(f"{x:6.1f}" for x in s) + f"  {fl:11.2f} {wf:6.2f}")
with open("p2_summary_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_pct_px_flipped", "worst_frame_pct_px_flipped"]); w.writerows(rows)
print("wrote p2_summary_table.csv")
