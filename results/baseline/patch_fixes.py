def patch(p, old, new, n=1):
    s = open(p).read()
    assert s.count(old) == n, f"{p}: expected {n} of {old!r}, found {s.count(old)}"
    open(p, "w").write(s.replace(old, new))

# Bug 1: turn caching on for the model and the decoder
patch("inference/inference.py",
      "    model.use_cache = opts.use_reframe\n",
      "    model.use_reframe = opts.use_reframe\n    model.depth_decoder.use_reframe = opts.use_reframe\n")
# Bug 2: Delta used the wrong attribute name (3 occurrences on lines 250-251 plus FIX0 shift)
patch("experiment_modules/bd_model.py",
      "self.cached_image", "self.reframe_cached_image", n=3)
# Bug 3: frame counter never incremented
patch("experiment_modules/bd_model.py",
      "                self.reframe_skipped += 1\n",
      "                self.reframe_skipped += 1\n            self.frame_counter += 1\n")
print("patched")
