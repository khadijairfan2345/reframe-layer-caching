import flip_evaluator, os, csv, numpy as np, cv2

def to_png(policy):
    src = f"example_data/composited/{policy}"
    dst = f"example_data/composited_png/{policy}"
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".jpg"):
            out = f"{dst}/{f[:-4]}.png"
            if not os.path.exists(out):
                cv2.imwrite(out, cv2.imread(f"{src}/{f}"))
    return dst

ref = to_png("nocache")
fs = sorted(f for f in os.listdir(ref) if f.endswith(".png"))
for p in ["every5_fixed", "deltaL"]:
    d = to_png(p)
    scores, worst = [], (-1, None, None)
    for f in fs:
        hm, s, _ = flip_evaluator.evaluate(f"{ref}/{f}", f"{d}/{f}", "LDR")
        scores.append(s)
        if s > worst[0]:
            worst = (s, hm, f)
    with open(f"flip_{p}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "flip"]); w.writerows(zip(fs, scores))
    hm = np.asarray(worst[1])
    if hm.ndim == 3 and hm.shape[0] in (1, 3):
        hm = np.moveaxis(hm, 0, -1)
    cv2.imwrite(f"flip_heatmap_{p}.png", (np.clip(hm, 0, 1) * 255).astype("uint8")[..., ::-1] if hm.ndim == 3 else (np.clip(hm, 0, 1) * 255).astype("uint8"))
    print(f"{p}: mean {np.mean(scores):.4f}  worst {worst[0]:.4f} ({worst[2]})  zeros {sum(s == 0 for s in scores)}/{len(scores)}")
