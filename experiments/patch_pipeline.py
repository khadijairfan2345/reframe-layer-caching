"""
patch_pipeline.py - add a rough BLOCK-SPLIT refresh policy "pipe-G" (Person 2, first version).

Instead of refreshing everything every 5th frame (Every-5), each frame does ONE chunk of the deep work:
  k=0: recompute image-encoder levels 1-4 on this frame (new features enter the pipeline)
  k=1: decoder column 1 (rows 1-3) -> updates cached tensors 1_a, 1_b
  k=2: decoder column 2 (rows 1-2) -> updates 2_a, 2_b
  k=3: decoder column 3 (row 1)    -> updates 3_a, 3_b
  k=4: updates 4_a
The top row (row 0) runs every frame as in ReFrame's cached frames. Chunks are NOT yet equal in time;
Person 3 fine-tunes the split. Other policies (N-5, delta-x, no cache) are unchanged.

Run from the implicit-depth repo root AFTER Annika's patches:   python patch_pipeline.py
Then: bash run_policy.sh OUTDIR pipe-5
"""
def patch(p, old, new, n=1):
    s = open(p).read()
    if new in s:
        print(f"{p}: already patched, skipping one edit"); return
    assert s.count(old) == n, f"{p}: expected {n} of {old!r}, found {s.count(old)}"
    open(p, "w").write(s.replace(old, new))

B = "experiment_modules/bd_model.py"
N = "modules/networks.py"

# --- bd_model.py: state, policy, encoder chunk, pass k to the decoder
patch(B, "        self.reframe_skipped = 0\n",
         "        self.reframe_skipped = 0\n        self.pipe_k = None  # PIPE\n")
patch(B, '                elif "N" in self.reframe_cache_policy:\n',
         '                elif self.reframe_cache_policy.startswith("pipe"):  # PIPE\n'
         '                    pipe_g = int(self.reframe_cache_policy.split("-")[1])\n'
         '                    self.pipe_k = self.frame_counter % pipe_g\n'
         '                    refresh_cache = False\n\n'
         '                elif "N" in self.reframe_cache_policy:\n')
patch(B, '            else:\n                cur_feats = self.reframe_cache["encoder"]\n',
         '            elif self.pipe_k == 0:  # PIPE chunk 0: refresh encoder levels 1-4 on this frame\n'
         '                cur_feats = self.encoder(cur_image)\n'
         '                self.reframe_cache["encoder"] = cur_feats\n'
         '            else:\n                cur_feats = self.reframe_cache["encoder"]\n')
patch(B, "        feature_outputs = self.depth_decoder(cur_feats, refresh_cache=refresh_cache)\n",
         "        self.depth_decoder.pipe_k = self.pipe_k if self.use_reframe else None  # PIPE\n"
         "        feature_outputs = self.depth_decoder(cur_feats, refresh_cache=refresh_cache)\n")

# --- networks.py: one decoder chunk per frame
patch(N, "    def forward(self, input_features, refresh_cache=True):\n",
'''    def _pipe_step(self, col0):  # PIPE: one chunk of the lower decoder per frame
        k = self.pipe_k
        if k == 0:
            self.pipe_col = list(col0)      # this frame's column 0 (fresh encoder levels)
            return
        prev = getattr(self, "pipe_col", None)
        if prev is None:
            return
        j = k
        if j <= 4:
            self.reframe_cache[f"{j}_a"] = [upsample(self.convs[f"diag_conv_1{j - 1}"](prev[1]))]
        if j <= 3:
            outs = {}
            max_i = 4 - j
            for i in range(max_i, 0, -1):
                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev[i + 1]))]
                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outs[i + 1]))]
                outs[i] = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
            self.reframe_cache[f"{j}_b"] = [upsample(self.convs[f"up_conv_1{j}"](outs[1]))]
            self.pipe_col = [None] + [outs[i] for i in range(1, max_i + 1)]

    def forward(self, input_features, refresh_cache=True):
''')
patch(N, "        if self.use_reframe:\n            if self.reframe_cache is not None:\n                j = 1\n",
         "        if self.use_reframe:\n"
         "            if self.reframe_cache is not None and getattr(self, \"pipe_k\", None) is not None:  # PIPE\n"
         "                self._pipe_step(prev_outputs)\n"
         "            if self.reframe_cache is not None:\n                j = 1\n")
patch(N, "            else:\n                self.reframe_cache = {}\n",
         "            else:\n                self.reframe_cache = {}\n                self.pipe_col = list(input_features)  # PIPE\n")
print("patched")
