"""
split_refresh_test.py  -  Does splitting the cache refresh remove latency spikes?

Builds on ReFrame's toy U-Net (samples/unet/unet.py in
github.com/ubc-aamodt-group/reframe-layer-caching). Same layers, same cache point (u1).

Compares three ways of running the same U-Net over a sequence of video-like frames:
  1. none      : no caching, full network every step
  2. periodic  : ReFrame "N-5" policy - full refresh every N steps, cheap cached steps in between
  3. split     : the deep path is cut into 5 pieces; one piece runs per step,
                 so a new cache is ready every N steps, with the SAME total refresh work
                 as periodic, but no single step does all of it.
  4. stripsG   : keep ONE cached array; each step recompute 1/G of it (a horizontal strip)
                 from the current frame and use the full mixed array. No full refreshes.

Outputs (in ./results/):
  step_times.csv      per-step latency (ms) for every policy
  summary.txt         mean / p50 / p95 / p99 / max latency + quality error
  timeline.png        latency of every step (the spike plot)
  piece_costs.png     how long each of the 5 pieces takes (are they equal?)

Run:   pip install torch matplotlib
       python split_refresh_test.py                 # auto picks cuda / mps / cpu
       python split_refresh_test.py --size 512 --steps 300
NOTE: weights are random (like ReFrame's demo), so "quality" here = how far each
cached output drifts from the no-cache output. It is a sanity check, not a real image metric.
"""
import argparse, os, time, csv, math
import torch, torch.nn as nn, torch.nn.init as init

# ----------------------------------------------------------------------------- model
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    """Same architecture as ReFrame's samples/unet/unet.py."""
    def __init__(self, cin, cout):
        super().__init__()
        self.down1, self.pool1 = DoubleConv(cin, 64), nn.MaxPool2d(2)
        self.down2, self.pool2 = DoubleConv(64, 128), nn.MaxPool2d(2)
        self.down3, self.pool3 = DoubleConv(128, 256), nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3, self.up_conv3 = nn.ConvTranspose2d(512, 256, 2, 2), DoubleConv(512, 256)
        self.up2, self.up_conv2 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128)
        self.up1, self.up_conv1 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, cout, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                init.zeros_(m.bias)

    # --- the 5 pieces of the deep path (what a "refresh" recomputes) ---
    def piece1(self, d1):  d2 = self.down2(self.pool1(d1)); return d2, self.down3(self.pool2(d2))
    def piece2(self, d3):  return self.bottleneck(self.pool3(d3))
    def piece3(self, bn, d3): return self.up_conv3(torch.cat([self.up3(bn), d3], 1))
    def piece4(self, u3, d2): return self.up_conv2(torch.cat([self.up2(u3), d2], 1))
    def piece5(self, u2):  return self.up1(u2)                      # -> new cached u1

    def head(self, u1, d1):  # the cheap part that runs EVERY step
        return self.out_conv(self.up_conv1(torch.cat([u1, d1], 1)))

    def full(self, x):
        d1 = self.down1(x); d2, d3 = self.piece1(d1); bn = self.piece2(d3)
        u3 = self.piece3(bn, d3); u2 = self.piece4(u3, d2); u1 = self.piece5(u2)
        return self.head(u1, d1), u1

    def deep_u1(self, x):
        """Whole deep path -> cached array u1 (used to recompute one strip from a crop)."""
        d1 = self.down1(x); d2, d3 = self.piece1(d1); bn = self.piece2(d3)
        return self.piece5(self.piece4(self.piece3(bn, d3), d2))

# ----------------------------------------------------------------------------- policies
class Runner:
    def __init__(self, model, policy, N):
        self.m, self.policy, self.N, self.t = model, policy, N, 0
        self.u1 = None; self.buf = {}

    @torch.no_grad()
    def step(self, x):
        m, t = self.m, self.t; self.t += 1
        if self.policy == "none":
            return m.full(x)[0]
        if self.policy == "periodic":
            if t % self.N == 0:
                out, self.u1 = m.full(x); return out
            return m.head(self.u1, m.down1(x))
        if self.policy == "split":
            d1 = m.down1(x)
            if t == 0:                                   # first step must build the cache
                out, self.u1 = m.full(x); return out
            # spread the 5 pieces evenly over each N-step cycle
            phase = (t - 1) % self.N
            every = self.N // 5
            if phase % every == 0:
                k = phase // every                       # which piece: 0..4
                b = self.buf
                if k == 0: b["d2"], b["d3"] = m.piece1(d1)          # uses fresh d1
                elif k == 1: b["bn"] = m.piece2(b["d3"])
                elif k == 2: b["u3"] = m.piece3(b["bn"], b["d3"])
                elif k == 3: b["u2"] = m.piece4(b["u3"], b["d2"])
                elif k == 4: self.u1 = m.piece5(b["u2"])            # swap in new cache
            return m.head(self.u1, d1)
        if self.policy.startswith("strips"):
            # Keep ONE cached array u1. Every step, recompute one horizontal strip of it
            # from the CURRENT frame (crop + margin so the strip sees its neighbours),
            # paste it in, and use the full mixed array. No full-refresh steps.
            d1 = m.down1(x)
            if t == 0:
                out, self.u1 = m.full(x); return out
            H = x.shape[-2]; G = self.G; k = (t - 1) % G
            s, e = k * H // G, (k + 1) * H // G
            a = max(0, (s - self.margin) // 8 * 8)          # align crop to 8 (3 poolings)
            b = min(H, -(-(e + self.margin) // 8) * 8)
            u1_crop = m.deep_u1(x[..., a:b, :])
            self.u1 = self.u1.clone(); self.u1[..., s:e, :] = u1_crop[..., s - a:e - a, :]
            return m.head(self.u1, d1)
        raise ValueError(self.policy)

# ----------------------------------------------------------------------------- helpers
def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()

def make_frames(n, size, ch, dev, seed=0):
    """Smooth 'video': drifting blobs + gentle noise, so consecutive frames are similar."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    centers = torch.rand(6, 2, generator=g) * 2 - 1
    vel = (torch.rand(6, 2, generator=g) - 0.5) * 0.04
    frames = []
    for t in range(n):
        c = centers + vel * t
        img = sum(torch.exp(-((xx - c[i, 0]) ** 2 + (yy - c[i, 1]) ** 2) / 0.05) for i in range(6))
        img = img.expand(ch, -1, -1) + 0.02 * torch.randn(ch, size, size, generator=g)
        frames.append(img.unsqueeze(0).to(dev))
    return frames

def timed(fn, dev):
    if dev.type == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); out = fn(); e.record(); torch.cuda.synchronize(); return out, s.elapsed_time(e)
    sync(dev); t0 = time.perf_counter(); out = fn(); sync(dev)
    return out, (time.perf_counter() - t0) * 1000

def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = math.floor(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", type=int, default=None, help="image size (default 512 on GPU, 256 on CPU)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--N", type=int, default=5, help="refresh period; must be a multiple of 5")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--strips", type=int, nargs="*", default=[4, 8],
                    help="strip counts to test for rotating strip refresh (e.g. --strips 4 8 16)")
    ap.add_argument("--margin", type=int, default=16,
                    help="extra rows of context around each strip (bigger = more accurate, slower)")
    a = ap.parse_args()
    assert a.N % 5 == 0, "N must be a multiple of 5 (5 pieces)"
    dev = pick_device(a.device)
    size = a.size or (512 if dev.type in ("cuda", "mps") else 256)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = UNet(3, 3).to(dev).eval()
    frames = make_frames(a.steps, size, 3, dev)
    print(f"device={dev}  size={size}x{size}  steps={a.steps}  N={a.N}")

    # warmup
    with torch.no_grad():
        for _ in range(a.warmup): model.full(frames[0])
    sync(dev)

    # ---- 1. cost of each piece (is splitting by layer = splitting evenly by time?)
    with torch.no_grad():
        x = frames[0]; d1 = model.down1(x)
        d2, d3 = model.piece1(d1); bn = model.piece2(d3); u3 = model.piece3(bn, d3); u2 = model.piece4(u3, d2); u1 = model.piece5(u2)
        fns = {"head (every step)": lambda: model.head(u1, model.down1(x)),
               "piece1 down2+down3": lambda: model.piece1(d1),
               "piece2 bottleneck": lambda: model.piece2(d3),
               "piece3 up3": lambda: model.piece3(bn, d3),
               "piece4 up2": lambda: model.piece4(u3, d2),
               "piece5 up1": lambda: model.piece5(u2)}
        piece_ms = {k: sorted(timed(f, dev)[1] for _ in range(30))[15] for k, f in fns.items()}  # median of 30

    # ---- 2. run each policy step by step
    times, errs, outs_ref = {}, {}, []
    policies = ["none", "periodic", "split"] + [f"strips{g}_m{a.margin}" for g in a.strips]
    for pol in policies:
        def new_runner():
            r = Runner(model, pol, a.N)
            if pol.startswith("strips"):
                r.G = int(pol[6:].split("_")[0]); r.margin = a.margin
            return r
        # per-policy warmup: run a couple of full cycles untimed, so one-time costs
        # (e.g. GPU kernels compiled the first time each new strip shape appears) don't count
        r = new_runner()
        for x in frames[:min(len(frames), 2 * max(a.N, max(a.strips or [1])) + 2)]: r.step(x)
        sync(dev)
        r = new_runner(); ts, es = [], []
        for t, x in enumerate(frames):
            out, ms = timed(lambda: r.step(x), dev)
            ts.append(ms)
            if pol == "none": outs_ref.append(out.detach().float().cpu())
            else:
                ref = outs_ref[t]; es.append(((out.float().cpu() - ref).abs().mean() / ref.abs().mean()).item())
        times[pol] = ts
        if pol != "none": errs[pol] = es
        print(f"  ran {pol}")

    # ---- 3. summary (skip step 0, which always builds the cache)
    lines = [f"device={dev}  size={size}  steps={a.steps}  N={a.N}", "",
             "Cost of each piece (median ms):"]
    for k, v in piece_ms.items(): lines.append(f"  {k:22s} {v:8.3f}")
    p = [v for k, v in piece_ms.items() if k.startswith("piece")]
    lines.append(f"  largest piece / smallest piece = {max(p)/min(p):.1f}x   (1.0x would be perfectly even)")
    lines += ["", f"{'policy':14s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s} {'max/mean':>9s}   rel.error(mean / max)"]
    for pol, ts in times.items():
        v = ts[1:]; m = sum(v) / len(v)
        e = f"{sum(errs[pol][1:])/len(errs[pol][1:]):.4f} / {max(errs[pol]):.4f}" if pol in errs else "  (reference)"
        lines.append(f"{pol:14s} {m:8.3f} {pct(v,50):8.3f} {pct(v,95):8.3f} {pct(v,99):8.3f} {max(v):8.3f} {max(v)/m:9.2f}   {e}")
    txt = "\n".join(lines); print("\n" + txt)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.out, "step_times.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + [f"{k}_ms" for k in times] + [f"{k}_relerr" for k in errs])
        for t in range(a.steps):
            w.writerow([t] + [f"{times[k][t]:.4f}" for k in times] + [f"{errs[k][t]:.5f}" for k in errs])

    # ---- 4. plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pal = ["#888888", "#d1495b", "#2e86ab", "#3a9d5d", "#e69f00", "#7b5ea7", "#555555"]
        col = {p: pal[i % len(pal)] for i, p in enumerate(times)}
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        show = min(60, a.steps)
        for pol in times: ax[0].plot(range(1, show), times[pol][1:show], label=pol, color=col[pol], lw=1.6, marker="o", ms=3)
        ymax = 1.3 * max(pct(times[p][1:], 99) for p in times)       # keep rare hiccups from squashing the plot
        ax[0].set_ylim(0, ymax)
        ax[0].set_ylabel("step latency (ms)"); ax[0].set_title("Latency of every step (first %d steps)" % show); ax[0].legend()
        for pol in errs: ax[1].plot(range(1, show), errs[pol][1:show], label=pol, color=col[pol], lw=1.6)
        ax[1].set_ylabel("drift from no-cache output"); ax[1].set_xlabel("step"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "timeline.png"), dpi=150)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh(list(piece_ms)[::-1], list(piece_ms.values())[::-1], color="#2e86ab")
        ax.set_xlabel("ms (median)"); ax.set_title("Cost of each piece")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "piece_costs.png"), dpi=150)
        # slide-ready figure: periodic vs. the strips setting with the lowest p95
        strip_pols = [p for p in times if p.startswith("strips")]
        if strip_pols:
            best = min(strip_pols, key=lambda p: pct(times[p][1:], 95))
            g = best[6:].split("_")[0]
            fig, ax = plt.subplots(figsize=(8, 3))
            show2 = min(40, a.steps)
            ax.plot(range(1, show2), times["periodic"][1:show2], color="#d1495b", lw=2, marker="o", ms=3, label="Periodic refresh (ReFrame)")
            ax.plot(range(1, show2), times[best][1:show2], color="#2e86ab", lw=2, marker="o", ms=3, label=f"Rotating refresh ({g} strips)")
            ax.set_ylim(0, 1.3 * max(pct(times[p][1:], 99) for p in ("periodic", best)))
            ax.set_xlabel("step"); ax.set_ylabel("step latency (ms)")
            ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper right")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "slide_figure.png"), dpi=200)
        print(f"\nSaved plots + CSV to ./{a.out}/")
    except ImportError:
        print("matplotlib not installed - skipped plots")

if __name__ == "__main__":
    main()