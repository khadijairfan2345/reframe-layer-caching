p = "experiment_modules/bd_model.py"
s = open(p).read()
old = "        # > REFRAME start\n        if self.use_reframe:\n            refresh_cache = True\n"
new = "        refresh_cache = True  # FIX0: must exist even when use_reframe is False\n" + old
assert s.count(old) == 1, "not found exactly once"
open(p, "w").write(s.replace(old, new))
print("patched")
