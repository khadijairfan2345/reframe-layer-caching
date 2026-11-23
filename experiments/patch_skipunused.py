"""
patch_skipunused.py - test Annika's hypothesis: on cached frames, the matching encoder, cost volume and
cost-volume encoder run but their output is never used (the cached decoder only reads encoder level 0
and its 7 cached tensors). With REFRAME_SKIP_UNUSED=1 those steps are skipped on frames that don't need
them. Without the variable nothing changes.

If the hypothesis is right: cached frames drop from ~80 ms to ~10 ms AND the masks stay bit-identical
(max abs diff 0.0 vs the same policy without skipping).

Run from the implicit-depth repo root, after patch_pipeline.py:   python patch_skipunused.py
"""
p = "experiment_modules/bd_model.py"
s = open(p).read()
if "REFRAME_SKIP_UNUSED" in s:
    print("already patched"); raise SystemExit
start = s.index("        # Compute matching features\n")
end = s.index("        # Decode into depth at multiple resolutions.\n")
block = s[start:end]
indented = "".join(("    " + l if l.strip() else l) for l in block.splitlines(True))
new = ('        # SKIPUNUSED: on cached frames the cached decoder never reads these outputs\n'
       '        skip_unused = (os.environ.get("REFRAME_SKIP_UNUSED") == "1" and self.use_reframe\n'
       '                       and self.depth_decoder.reframe_cache is not None and not refresh_cache\n'
       '                       and self.pipe_k != 0)\n'
       '        if skip_unused:\n'
       '            lowest_cost, overall_mask_bhw = None, None\n'
       '        else:\n' + indented)
s = s[:start] + new + s[end:]
assert s.count("import types\n") == 1
s = s.replace("import types\n", "import types\nimport os  # SKIPUNUSED\n", 1)
open(p, "w").write(s)
print("patched")
