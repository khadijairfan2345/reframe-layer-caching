"""
patch_pipe3d.py - policy "pipe3d-G": remove the wasted 3D work AND spread the whole refresh over G frames.

What a refresh costs (measured, Every-5 refresh frame minus cached frame, ms):
    matching encoder 9.0 | cost volume 57.4 (64 depth bins, ~0.9 ms each) | cost-volume encoder 3.5
    image encoder levels 1-4: 16.4 (L1+L2 ~3, L3 7.4, L4 6.6) | decoder rows 1-4: 9.2
pipe3d-G takes a snapshot of frame k=0 of each G-frame cycle and builds a fresh cache from it in pieces:
    matching -> cost volume in depth-bin chunks -> encoder L1-2 / L3 / L4 -> CV encoder -> decoder columns 1..4
Each frame gets about 1/G of the ~96 ms (a balanced plan is printed at start). Every other frame does only
encoder L0 + decoder row 0 + MLP (~10 ms), i.e. the unused 3D work of ReFrame cached frames is skipped.

Trade-off to report: the cache finishes at the end of a cycle, from the cycle's first frame, so it is
G-1 .. 2G-2 frames old instead of Every-G's 0 .. G-1.

P3_SELFTEST=1 checks, once per cycle, that the pieces equal a one-shot refresh of the same snapshot
(cost volume, encoder levels, and the 7 decoder cache tensors). Expect max abs diff 0 (or ~1e-6).

Run from the implicit-depth repo root AFTER patch_pipeline.py and patch_skipunused.py:
    python patch_pipe3d.py
Then:  --use_reframe --reframe_cache_policy pipe3d-5   (or pipe3d-10, ...)
"""
import sys

def patch(path, pairs, tag):
    s = open(path).read()
    if tag in s:
        print(f"{path}: already patched"); return
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"{path}: expected 1 match, found {n}:\n{old}")
        s = s.replace(old, new, 1)
    open(path, "w").write(s); print(f"{path}: patched")

# ---------------------------------------------------------------- cost volume: compute a range of depth bins
cv = open("modules/cost_volume.py").read()
if "PIPE3D" not in cv:
    a = cv.index("class FeatureVolumeManager")
    b = cv.index("class FastFeatureVolumeManager")
    old = "        for depth_id in range(self.num_depth_bins):\n"
    assert cv[a:b].count(old) == 1, "cost_volume.py: loop not found"
    new = ("        for depth_id in range(*(getattr(self, 'depth_range', None) or (0, self.num_depth_bins))):  # PIPE3D\n")
    cv = cv[:a] + cv[a:b].replace(old, new) + cv[b:]
    open("modules/cost_volume.py", "w").write(cv); print("modules/cost_volume.py: patched")
else:
    print("modules/cost_volume.py: already patched")

# ---------------------------------------------------------------- decoder: run the queued column steps
patch("modules/networks.py", [(
"            if self.reframe_cache is not None and getattr(self, \"pipe_k\", None) is not None:  # PIPE\n",
"            if self.reframe_cache is not None and getattr(self, \"p3_steps\", None):  # PIPE3D\n"
"                for j, col in self.p3_steps:\n"
"                    self.pipe_k = j\n"
"                    self._pipe_step(col)\n"
"                    if j == 4 and getattr(self, \"p3_ref\", None) is not None:  # self-test\n"
"                        d = max((self.reframe_cache[k][0] - v).abs().max().item() for k, v in self.p3_ref.items())\n"
"                        print(f\"[pipe3d selftest] decoder cache (7 tensors) vs one-shot refresh: max abs diff {d:.3g}\")\n"
"                        self.p3_ref = None\n"
"                self.pipe_k = None\n"
"                self.p3_steps = None\n"
"            if self.reframe_cache is not None and getattr(self, \"pipe_k\", None) is not None:  # PIPE\n")],
"PIPE3D")

# ---------------------------------------------------------------- model
METHODS = r'''
    # > PIPE3D start ------------------------------------------------------------------------------
    # estimated ms per unit of refresh work (from the Every-5 timing run); only used to balance the plan
    P3_COST = {"M": 9.0, "B": 57.4 / 64, "E1": 3.0, "E2": 7.4, "E3": 6.6, "CVN": 3.5,
               "D1": 4.7, "D2": 2.8, "D3": 1.4, "D4": 0.3}

    def _p3_cost(self, u):
        if u[0] == "B":
            return self.P3_COST["B"] * (u[2] - u[1])
        return self.P3_COST[u[0]]

    def _pipe3d_plan(self, G):
        nb = self.cost_volume.num_depth_bins
        units = ([("M",)] + [("B", d, d + 1) for d in range(nb // 2)] + [("E1",), ("E2",)]
                 + [("B", d, d + 1) for d in range(nb // 2, nb)]
                 + [("E3",), ("CVN",), ("D1",), ("D2",), ("D3",), ("D4",)])
        rem = sum(self._p3_cost(u) for u in units)
        plan, i = [], 0
        while len(plan) < G:
            left = G - len(plan); target = rem / left; cur, t = [], 0.0
            while i < len(units):
                x = self._p3_cost(units[i])
                if left > 1 and cur and t + x > target and (t + x - target) > (target - t):
                    break
                cur.append(units[i]); t += x; i += 1
            rem -= t
            merged = []                       # merge neighbouring depth bins into one chunk
            for u in cur:
                if u[0] == "B" and merged and merged[-1][0] == "B" and merged[-1][2] == u[1]:
                    merged[-1] = ("B", merged[-1][1], u[2])
                else:
                    merged.append(u)
            plan.append(merged)
        return plan

    def _p3_blocks(self, x, lo, hi, feats):
        for i in range(lo, hi):
            x = self.encoder.blocks[i](x)
            if i in self.p3_idx:
                feats.append(x)
        return x

    def _pipe3d_work(self, k, cur_image, src_image, src_cam_T_cur_cam, cur_cam_T_src_cam, src_K, cur_invK,
                     unbatched):
        G = int(self.reframe_cache_policy.split("-")[1])
        selftest = os.environ.get("P3_SELFTEST") == "1"
        if getattr(self, "p3_plan", None) is None or len(self.p3_plan) != G:
            self.p3_plan = self._pipe3d_plan(G)
            try:
                self.p3_idx = [int(n.split(".")[1]) for n in self.encoder.feature_info.module_name()]
                assert len(self.p3_idx) == 5 and self.p3_idx[0] == 0 and self.p3_idx == sorted(self.p3_idx)
            except Exception:
                self.p3_idx = [0, 1, 2, 4, 5]
            print(f"[pipe3d] encoder feature blocks {self.p3_idx}; plan for G={G}:")
            for f, units in enumerate(self.p3_plan):
                names = " ".join(("B%d-%d" % (u[1], u[2] - 1)) if u[0] == "B" else u[0] for u in units)
                print(f"   frame k={f}: ~{sum(self._p3_cost(u) for u in units):5.1f} ms extra  [{names}]")
        if k == 0:   # snapshot of this frame; the whole cycle builds the cache from it
            self.p3 = dict(img=cur_image, src=src_image, ext=src_cam_T_cur_cam, poses=cur_cam_T_src_cam,
                           K=src_K, invK=cur_invK, unb=unbatched, chunks=[], enc=[], x=None)
        p = getattr(self, "p3", None)
        if p is None:
            return
        idx = self.p3_idx
        steps = []
        for u in self.p3_plan[k]:
            if u[0] == "M":
                p["mc"], p["ms"] = self.compute_matching_feats(p["img"], p["src"], p["unb"])
            elif u[0] == "B":
                self.cost_volume.depth_range = (u[1], u[2])
                try:
                    vol = self.cost_volume(
                        cur_feats=p["mc"], src_feats=p["ms"], src_extrinsics=p["ext"], src_poses=p["poses"],
                        src_Ks=p["K"], cur_invK=p["invK"],
                        min_depth=torch.tensor(self.run_opts.min_matching_depth).type_as(p["K"]).view(1, 1, 1, 1),
                        max_depth=torch.tensor(self.run_opts.max_matching_depth).type_as(p["K"]).view(1, 1, 1, 1),
                        return_mask=False)[0]
                finally:
                    self.cost_volume.depth_range = None
                p["chunks"].append(vol)
            elif u[0] == "E1":    # stem + blocks up to level 2
                x = self.encoder.bn1(self.encoder.conv_stem(p["img"]))
                p["x"] = self._p3_blocks(x, 0, idx[2] + 1, p["enc"])
            elif u[0] == "E2":    # level 3
                p["x"] = self._p3_blocks(p["x"], idx[2] + 1, idx[3] + 1, p["enc"])
            elif u[0] == "E3":    # level 4
                p["x"] = self._p3_blocks(p["x"], idx[3] + 1, idx[4] + 1, p["enc"])
            elif u[0] == "CVN":
                vol = torch.cat(p["chunks"], dim=1)
                ms = self.run_opts.matching_scale
                full = p["enc"][:ms] + self.cost_volume_net(vol, p["enc"][ms:])
                if selftest:
                    self._pipe3d_selftest(p, vol, full)
                self.reframe_cache["encoder"] = list(p["enc"])
                steps.append((0, full))
            else:                 # D1..D4: decoder column j
                steps.append((int(u[0][1]), None))
        if steps:
            self.depth_decoder.p3_steps = steps

    def _pipe3d_selftest(self, p, vol, full):
        ref_vol = self.cost_volume(
            cur_feats=p["mc"], src_feats=p["ms"], src_extrinsics=p["ext"], src_poses=p["poses"],
            src_Ks=p["K"], cur_invK=p["invK"],
            min_depth=torch.tensor(self.run_opts.min_matching_depth).type_as(p["K"]).view(1, 1, 1, 1),
            max_depth=torch.tensor(self.run_opts.max_matching_depth).type_as(p["K"]).view(1, 1, 1, 1),
            return_mask=False)[0]
        ref_enc = self.encoder(p["img"])
        d_vol = (vol - ref_vol).abs().max().item()
        d_enc = max((a - b).abs().max().item() for a, b in zip(p["enc"], ref_enc))
        print(f"[pipe3d selftest] cost volume {tuple(vol.shape)} vs one-shot: max abs diff {d_vol:.3g}; "
              f"encoder levels ({len(p['enc'])}) vs encoder(): {d_enc:.3g}")
        dec = self.depth_decoder
        saved_cache, saved_col = dec.reframe_cache, getattr(dec, "pipe_col", None)
        dec.pipe_k = None
        dec(full, refresh_cache=True)
        dec.p3_ref = {key: dec.reframe_cache[key][0].clone()
                      for key in ["1_a", "2_a", "3_a", "4_a", "1_b", "2_b", "3_b"]}
        dec.reframe_cache, dec.pipe_col = saved_cache, saved_col
    # > PIPE3D end --------------------------------------------------------------------------------

    def compute_matching_feats(
'''

patch("experiment_modules/bd_model.py", [
    ("        self.pipe_k = None  # PIPE\n",
     "        self.pipe_k = None  # PIPE\n        self.pipe3d = False  # PIPE3D\n"),
    ("                elif self.reframe_cache_policy.startswith(\"pipe\"):  # PIPE\n",
     "                elif self.reframe_cache_policy.startswith(\"pipe3d\"):  # PIPE3D\n"
     "                    self.pipe3d = True\n"
     "                    self.pipe_k = self.frame_counter % int(self.reframe_cache_policy.split(\"-\")[1])\n"
     "                    refresh_cache = False\n\n"
     "                elif self.reframe_cache_policy.startswith(\"pipe\"):  # PIPE\n"),
    ("            elif self.pipe_k == 0:  # PIPE chunk 0",
     "            elif self.pipe_k == 0 and not self.pipe3d:  # PIPE chunk 0"),
    ("                       and self.pipe_k != 0)\n",
     "                       and (self.pipe_k != 0 or self.pipe3d))\n"),
    ("        skip_unused = (os.environ.get(\"REFRAME_SKIP_UNUSED\") == \"1\" and self.use_reframe\n",
     "        skip_unused = ((os.environ.get(\"REFRAME_SKIP_UNUSED\") == \"1\" or self.pipe3d) and self.use_reframe\n"),
    ("        # Decode into depth at multiple resolutions.\n",
     "        if self.pipe3d and not refresh_cache:  # PIPE3D: this frame's share of the refresh\n"
     "            self._pipe3d_work(self.pipe_k, cur_image, src_image, src_cam_T_cur_cam, cur_cam_T_src_cam,\n"
     "                              src_K, cur_invK, unbatched_matching_encoder_forward)\n\n"
     "        # Decode into depth at multiple resolutions.\n"),
    ("        self.depth_decoder.pipe_k = self.pipe_k if self.use_reframe else None  # PIPE\n",
     "        self.depth_decoder.pipe_k = self.pipe_k if (self.use_reframe and not self.pipe3d) else None  # PIPE\n"),
    ("\n    def compute_matching_feats(\n", METHODS),
], "PIPE3D")
