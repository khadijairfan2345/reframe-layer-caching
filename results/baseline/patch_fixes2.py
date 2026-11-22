def patch(p, old, new, n=1):
    s = open(p).read()
    assert s.count(old) == n, f"{p}: expected {n} of {old!r}, found {s.count(old)}"
    open(p, "w").write(s.replace(old, new))

# Bug 2: Delta used the wrong attribute name (lines 250-251, two occurrences)
patch("experiment_modules/bd_model.py", "self.cached_image", "self.reframe_cached_image", n=2)
# Bug 3: frame counter never incremented
patch("experiment_modules/bd_model.py",
      "                self.reframe_skipped += 1\n",
      "                self.reframe_skipped += 1\n            self.frame_counter += 1\n")
print("patched")
