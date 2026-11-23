"""
time_parts.py - time each part of the Implicit Depth model, per frame, on cached vs. refresh frames.

Run from the implicit-depth repo root, with the SAME arguments as run_policy.sh, e.g.:
  python time_parts.py --config configs/models/implicit_depth.yaml \
      --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt \
      --data_config example_data/scans/config/config.yaml \
      --rendered_depth_map_load_dir example_data/renders \
      --dataset_path example_data/scans --output_base_path out_parts_every5 \
      --use_reframe --reframe_cache_policy N-5

Writes <output_base_path>/part_times.csv (one row per frame) and prints mean ms per part,
split into refresh frames and cached frames (frame 1 = warmup, dropped).
"""
import csv, os, sys
from collections import defaultdict
import torch
from torch import nn

import options
import inference.inference as inf
from experiment_modules.bd_model import BDModel

PARTS = ["encoder", "matching_model", "cost_volume", "cost_volume_net", "depth_decoder", "binary_mlp"]
# finer split: image-encoder levels and decoder rows (row 0 = 192x256 ... row 4 = 12x16)
FINE = ["enc_L0", "enc_L1", "enc_L2", "enc_L3", "enc_L4", "dec_row0", "dec_row1", "dec_row2", "dec_row3", "dec_row4"]
events = []          # (frame, part, start_event, end_event)
frames = []          # (frame, kind)
state = {"frame": 0, "skipped_before": 0}

def part_hooks(name):
    def pre(mod, inp):
        e = torch.cuda.Event(enable_timing=True); e.record(); mod._t_start = e
    def post(mod, inp, out):
        e = torch.cuda.Event(enable_timing=True); e.record()
        events.append((state["frame"], name, mod._t_start, e))
    return pre, post

def span_hooks(name, first, last):
    """time from the start of module `first` to the end of module `last` (an encoder level)"""
    holder = {}
    def pre(mod, inp):
        e = torch.cuda.Event(enable_timing=True); e.record(); holder["s"] = e
    def post(mod, inp, out):
        e = torch.cuda.Event(enable_timing=True); e.record()
        events.append((state["frame"], name, holder["s"], e))
    first.register_forward_pre_hook(pre); last.register_forward_hook(post)

def model_pre(mod, inp):
    state["frame"] += 1
    state["skipped_before"] = getattr(mod, "reframe_skipped", 0)
    e = torch.cuda.Event(enable_timing=True); e.record(); mod._t_start = e

def model_post(mod, inp, out):
    e = torch.cuda.Event(enable_timing=True); e.record()
    events.append((state["frame"], "TOTAL", mod._t_start, e))
    cached = getattr(mod, "reframe_skipped", 0) > state["skipped_before"]
    k = getattr(mod, "pipe_k", None)
    kind = ("pipe_k%d" % k) if (cached and k is not None) else ("cached" if cached else "refresh")
    frames.append((state["frame"], kind))

_orig_load = BDModel.load_from_checkpoint
def load_with_hooks(*a, **k):
    model = _orig_load(*a, **k)
    model.register_forward_pre_hook(model_pre)
    model.register_forward_hook(model_post)
    for name in PARTS:
        sub = getattr(model, name, None)
        if isinstance(sub, nn.Module):
            pre, post = part_hooks(name)
            sub.register_forward_pre_hook(pre); sub.register_forward_hook(post)
        else:
            print(f"[time_parts] {name} is not an nn.Module, not timed separately")
    # encoder levels (EfficientNetV2-S stages): L0 = stem + blocks[0], L1 = blocks[1], L2 = blocks[2],
    # L3 = blocks[3..4], L4 = blocks[5]  (channels 24, 48, 64, 160, 256)
    enc = model.encoder
    try:
        span_hooks("enc_L0", enc.conv_stem, enc.blocks[0])
        span_hooks("enc_L1", enc.blocks[1], enc.blocks[1])
        span_hooks("enc_L2", enc.blocks[2], enc.blocks[2])
        span_hooks("enc_L3", enc.blocks[3], enc.blocks[4])
        span_hooks("enc_L4", enc.blocks[5], enc.blocks[5])
    except Exception as ex:
        print("[time_parts] could not hook encoder levels:", ex)
    # decoder rows: every conv in the U-Net++ grid, grouped by the row it runs at
    for key, mod in model.depth_decoder.convs.items():
        row = key.split("_")[-1][0]
        pre, post = part_hooks(f"dec_row{row}")
        mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
    print("[time_parts] hooks installed")
    return model
BDModel.load_from_checkpoint = load_with_hooks

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    handler = options.OptionsHandler()
    handler.parse_and_merge_options()
    opts = handler.options
    inf.main(opts)
    torch.cuda.synchronize()

    per = defaultdict(lambda: defaultdict(float))     # frame -> part -> ms (summed if called several times)
    for f, part, s, e in events:
        per[f][part] += s.elapsed_time(e)
    kind = dict(frames)
    cols = ["TOTAL"] + PARTS + FINE
    out = os.path.join(opts.output_base_path, "part_times.csv")
    os.makedirs(opts.output_base_path, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "kind"] + cols + ["other"])
        for f in sorted(per):
            row = [per[f].get(c, 0.0) for c in cols]
            w.writerow([f, kind.get(f, "?")] + [f"{x:.3f}" for x in row] + [f"{row[0] - sum(row[1:1 + len(PARTS)]):.3f}"])
    print(f"\n[time_parts] wrote {out}")

    for k in sorted(set(kind.values())):
        fs = [f for f in sorted(per) if f > 1 and kind.get(f) == k]
        if not fs: continue
        print(f"\n{k.upper()} frames: {len(fs)}  (mean ms per part)")
        for c in cols:
            if c == FINE[0]: print("  -- finer split (inside encoder / depth_decoder) --")
            print(f"  {c:16s} {sum(per[f].get(c, 0.0) for f in fs) / len(fs):8.2f}")
        tot = sum(per[f]["TOTAL"] for f in fs) / len(fs)
        parts = sum(sum(per[f].get(c, 0.0) for c in PARTS) for f in fs) / len(fs)
        print(f"  {'other':16s} {tot - parts:8.2f}")
