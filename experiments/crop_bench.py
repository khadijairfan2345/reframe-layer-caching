"""
crop_bench.py - how long do the cacheable parts take on a CROP of the frame (one strip + border)?
Gives the T(h) curve the strip equations need, so strips can be predicted later without a GPU.

Run from the implicit-depth repo root:
  python crop_bench.py weights/implicit_depth_temporal_hypersim.ckpt out_crop_bench.csv
Times (median of 30 runs, CUDA events, random inputs of the real shapes):
  encoder_full : the image encoder, all 5 levels, on an image crop of height 2*rows0
  decoder_full : the whole U-Net++ decoder grid on feature crops (level 0 height rows0)
rows0 = strip height at the 192x256 level = 192/G + 2*border, rounded up to a multiple of 16.
"""
import csv, math, sys, statistics
import torch
from experiment_modules.bd_model import BDModel

ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/implicit_depth_temporal_hypersim.ckpt"
out = sys.argv[2] if len(sys.argv) > 2 else "out_crop_bench.csv"
torch.set_grad_enabled(False)
model = BDModel.load_from_checkpoint(ckpt, args=None).cuda().eval()
H, W = model.run_opts.image_height, model.run_opts.image_width
H0, W0 = H // 2, W // 2                     # level-0 feature size (192 x 256)
dec = model.depth_decoder; dec.use_reframe = False
nch = list(dec.num_ch_enc)
print(f"image {H}x{W}, level-0 features {H0}x{W0}, decoder input channels {nch}")

def bench(fn, reps=30, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

rows_set = []
for G in [1, 2, 4, 8, 16]:
    for b in [0, 8, 16, 32]:
        r0 = min(H0, int(math.ceil((H0 / G + 2 * b) / 16.0) * 16))
        rows_set.append((G, b, r0))
rows = []
cache = {}
for G, b, r0 in rows_set:
    if r0 not in cache:
        img = torch.randn(1, 3, 2 * r0, W, device="cuda")
        feats = [torch.randn(1, nch[i], r0 >> i, W0 >> i, device="cuda") for i in range(5)]
        te = bench(lambda: model.encoder(img))
        td = bench(lambda: dec(feats, refresh_cache=True))
        cache[r0] = (te, td)
    te, td = cache[r0]
    rows.append([G, b, r0, f"{te:.3f}", f"{td:.3f}"])
    print(f"G={G:2d} border={b:2d} rows0={r0:3d}  encoder_full {te:7.2f} ms  decoder_full {td:7.2f} ms")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["G", "border_rows", "rows0", "encoder_full_ms", "decoder_full_ms"]); w.writerows(rows)
print("wrote", out)
