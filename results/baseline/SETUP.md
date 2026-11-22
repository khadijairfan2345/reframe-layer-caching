# SETUP.md: Implicit Depth + ReFrame on Cannon (Person 1 notes)

Everything here was run on Cannon (Harvard FASRC) on 21 Nov 2026. Commands are in the order they were run, including the errors we hit and how they were fixed. Person 3 can reuse this to reproduce the setup.

## 0. Machine and session

- GPU: NVIDIA A100-SXM4-40GB in **MIG mode, profile 3g.20gb** (about 20 GB, 42 SMs). Nodes seen: holygpu7c26305, holygpu7c26304. Driver 610.57.04, CUDA 13.3 reported by the driver.
- Timings depend on which MIG slice you get. Compare ratios between policies, not raw milliseconds, and record the slice in every run record.
- Interactive session (compute nodes can reach GitHub, Google storage and PyPI):

```bash
salloc -p gpu_test --gres=gpu:1 -t 0-02:00 --mem=32G -c 4
module load Mambaforge/23.11.0-fasrc01
mamba activate binarydepth
cd ~/implicit-depth
```

- Sessions expire. Files in the home directory persist, so after expiry just run `salloc` again and repeat the lines above. Check time left with `squeue -u $USER -o "%i %M %L"`.
- Pasting long multi-line commands into the terminal sometimes garbles them. Put commands in a file with a heredoc (`cat > x.sh <<'EOF' ... EOF`) and run it with `bash x.sh`.

## 1. Repos and data

```bash
cd ~
git clone https://github.com/khadijairfan2345/reframe-layer-caching      # team repo (has samples/IC with ReFrame's 4 files)
git clone https://github.com/nianticlabs/implicit-depth.git              # base repo, HTTPS (README's SSH URL needs a key)
cd implicit-depth
wget -P weights https://storage.googleapis.com/niantic-lon-static/research/implicit-depth/models/implicit_depth_temporal_hypersim.ckpt   # 566 MB
wget https://storage.googleapis.com/niantic-lon-static/research/implicit-depth/example_data.zip   # 270 MB
unzip -q example_data.zip
```

About 1.8 GB total (weights 710 MB incl. helper script, example_data 1.1 GB). Pushing to the team repo over HTTPS needs a GitHub personal access token as the password and write access to the repo. Run `git pull` before pushing, since others push too.

## 2. Environment

The authors' file is `binarydepth_env.yml` (Python 3.9.7, PyTorch 1.10.0, CUDA toolkit 11.3, pytorch3d 0.7.3). The pinned versions run fine on the A100.

**Problem 1: the `default` channel.** The file lists a channel named `default`. Changing it to `defaults` made conda try `repo.anaconda.com`, which is unreachable from the node (4+ minute timeout). Fix: remove the line.

```bash
grep -v "^  - default" binarydepth_env.yml > binarydepth_env_fixed.yml
mamba env create -f binarydepth_env_fixed.yml      # ~10 to 15 min, everything including pytorch3d and open3d installed fine
```

The `Could not open lockfile .../pkgs/cache/cache.lock` messages are harmless (the shared package cache is read-only).

**Problem 2: `import torch` failed** with `undefined symbol: iJIT_NotifyEvent`. Cause: conda installed MKL 2024.2.2, which is too new for PyTorch 1.10. Fix:

```bash
mamba install -n binarydepth "mkl=2024.0" -c conda-forge -y
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 1.10.0 11.3 True NVIDIA A100-SXM4-40GB MIG 3g.20gb
python -c "import pytorch_lightning, kornia, timm, cv2, pytorch3d; print('imports ok')"
pip freeze > env_freeze.txt      # exact versions, committed in results/baseline/
pip install flip_evaluator       # needed later for FLIP
```

If rebuilding from scratch, add `mkl<2024.1` to the env file. Also note that numpy 1.26.4 got installed; it works with torch 1.10 here, so no change was needed.

## 3. Dataset preparation (authors' steps)

```bash
python -m inference.make_config_and_txt_file --input-sequence-dir example_data/scans/garden_chair --save-dir example_data/scans/config
python ./data_scripts/generate_test_tuples.py --data_config example_data/scans/config/config.yaml --num_workers 4
```

I used `--num_workers 4` instead of the README's 16 because the session has 4 CPUs. Result: 296 samples from 297 poses. The `SyntaxWarning: "is not" with a literal` from `arkit_dataset.py` is harmless.

## 4. Baseline (original model, before touching ReFrame)

The authors' README names a config (`sr_bd_high_res_sup_pretrained_7525.yaml`) that does not exist in the repo. Available configs: `implicit_depth.yaml`, `implicit_depth_temporal_hypersim.yaml`, and others.

- `implicit_depth_temporal_hypersim.yaml` + `--use_prior` and `implicit_depth.yaml` without it give nearly identical output (mean correlation 0.994).
- **Decision: baseline = `configs/models/implicit_depth.yaml`, real renders (`--rendered_depth_map_load_dir example_data/renders`), no `--use_prior`.** This matches ReFrame's README. Used for all policies.
- The first run downloads two small backbone weights to `~/.cache/torch/hub`, and takes about 1:47 for 296 frames. Later runs take about 50 s.

Composite check:

```bash
python -m inference.composite --predicted-masks-dir example_data/predictions/render/garden_chair/ \
    --renders-dir example_data/renders --vdr-dir example_data/scans/garden_chair --out-dir example_data/composited/baseline_mine
```

**The authors' reference masks do not match runs made with real renders** (corr about 0.15 to 0.20). The reference was made *without* `--rendered_depth_map_load_dir` (fixed 2 m plane input). A plane-input run matches the reference at mean corr 0.984, 98% agreement at 0.5, so the pipeline reproduces the authors' output. Predictions are saved under `<output>/render/garden_chair/` for real renders and `<output>/plane_2.0/garden_chair/` for plane input.

Open decision for the team: real renders (the AR use case) or plane input as the project baseline. Currently using real renders.

## 5. ReFrame files and fixes

Copy ReFrame's four files from the team repo into the Implicit Depth repo (originals saved in `orig_backup/`):

```bash
R=~/reframe-layer-caching/samples/IC
cp $R/options.py options.py
cp $R/bd_model.py experiment_modules/bd_model.py
cp $R/networks.py modules/networks.py
cp $R/inference.py inference/inference.py
```

All edits were made with exact-match patch scripts (saved in `results/baseline/`). Order applied:

| Step | File | Change |
|---|---|---|
| Timing (`patch_timing.py`) | `inference/inference.py` | `torch.cuda.synchronize()` before `start_time` and `end_time`; per-frame times written to `<output>/frame_times.csv` (all frames; frame 0 includes warmup and is dropped in analysis) |
| Fix 0 (`patch_fix0.py`), not in the original email | `experiment_modules/bd_model.py` | `refresh_cache = True` before the first `# > REFRAME start` block. Without it, **No Cache crashes** (`UnboundLocalError: refresh_cache`) because it is only assigned inside `if self.use_reframe:` |
| Bug 1 (`patch_fixes.py`) | `inference/inference.py` | `model.use_cache = ...` replaced by `model.use_reframe = opts.use_reframe` and `model.depth_decoder.use_reframe = opts.use_reframe` |
| Bug 2 (`patch_fixes2.py`) | `experiment_modules/bd_model.py` | `self.cached_image` replaced by `self.reframe_cached_image` (2 occurrences, in the Delta branch) |
| Bug 3 (`patch_fixes2.py`) | `experiment_modules/bd_model.py` | `self.frame_counter += 1` after `self.reframe_skipped += 1`, inside `if self.use_reframe:` |

Note: `patch_fixes.py` failed partway (my expected count for Bug 2 was 3, actual 2). Bug 1 had already been applied by then, and `patch_fixes2.py` applied Bugs 2 and 3. Read them together.

Validation: with Fix 0 only, No Cache output is bit-identical to the plain-model run (max abs diff 0.0 over 296 frames).

## 6. Running policies

```bash
bash run_policy.sh OUTDIR            # No Cache
bash run_policy.sh OUTDIR N-5        # Every-5
bash run_policy.sh OUTDIR delta-0.25 # Delta-L (tau 0.25)
bash run_policy.sh OUTDIR delta-0.20 # Delta-H (tau 0.20), not run yet
```

`run_policy.sh` uses `implicit_depth.yaml`, the HyperSim checkpoint, real renders and no `--use_prior`. Each run writes predictions and `frame_times.csv` to `OUTDIR`. Compare two runs with `python compare_runs.py DIR_A/render/garden_chair DIR_B/render/garden_chair`.

## 7. Quality (FLIP)

`inference.composite` saves one JPG per frame (296) next to `composited.mp4`. FLIP only reads PNG/BMP/TGA/EXR, so `flip_eval.py` converts JPG to PNG first, then scores each policy against the No Cache composite per frame. The sample script in `samples/flip/compare_images.py` is hardcoded for Sun Temple, so we did not use it. FLIP numbers come from JPEG composites, so there is a small noise floor shared by all policies (identical frames still score exactly 0).

## 8. Results (A100 MIG 3g.20gb, Garden Chair, 295 frames after dropping frame 0)

| Policy | Skipped | Mean ms | p95 | p99 | Max |
|---|---|---|---|---|---|
| No Cache | 0 | 103.3 | 104.2 | 104.8 | 105.8 |
| Every-5 (fixed) | 236 | 84.8 | 104.9 | 105.7 | 106.4 |
| Delta tau=0.25 | 39 | 100.5 | 105.1 | 105.8 | 108.6 |
| Every-5 (unfixed, for reference) | 0 | 104 | n/a | n/a | n/a |

- Every-5 refresh frames average 104.7 ms (n=59), cached frames 79.8 ms (n=236), so T_deep is about 24.9 ms. Check: 79.8 + 24.9/5 = 84.8 ms, matching the measured mean. These are end-to-end times, including the parts that are never cached.
- FLIP vs No Cache: Every-5 mean 0.0150, worst 0.0538 (frame 14, one frame before a refresh; exactly 60 frames at 0). Delta tau=0.25 mean 0.0006, worst 0.0093 (257 frames at 0).
- Before the fixes, Every-5 output was identical to No Cache with 0 frames skipped.

## 9. Cached tensors (probe on frame 0, Every-5; full output in `cache_shapes.txt`)

Decoder cache: `1_a`, `2_a`, `3_a`, `4_a`, `1_b`, `2_b`, `3_b`, each (1, 64, 192, 256) float32, about 12.6 MB each. Encoder cache (`cur_feats`): (1,24,192,256), (1,48,96,128), (1,64,48,64), (1,160,24,32), (1,256,12,16). On cached frames only encoder level 0 is recomputed. Matches the paper's Table 9 (seven 64x192x256 tensors).

## 10. Findings and open questions

- **Delta skip rate differs from the paper.** At tau=0.25 only 13% of frames are skipped (paper: about 79% for Delta-L on Garden Chair). SMAPE is computed on the ImageNet-normalized input (range -2.12 to 2.64, mean about 0.17), where the |a|+|b| denominator is small, so SMAPE is inflated (median about 0.35 to 0.38). Policy code behaves as written. Not yet checked: whether the paper used the same normalized input, and which tau gives about 79% skipped (sweep 0.35, 0.45, 0.55).
- **Unverified hypothesis:** on cached frames the matching encoders, cost volume and cost-volume net appear to still run, but the cached decoder path uses only `prev_outputs[0]` and the cache, so their output looks unused. That would explain why T_head (79.8 ms) is large. This is from reading the code only; per-layer profiling should confirm it.
- The spike is small end to end (about 25 ms on an 80 ms base), much less dramatic than the paper's Fig. 7, which plots decoder blocks only.
- Not run yet: Delta-H, the threshold sweep, 3 repeats, memory measurement.

## 11. Files in `results/baseline/`

`run_policy.sh`, `compare_runs.py`, `flip_eval.py`, `patch_timing.py`, `patch_fix0.py`, `patch_fixes.py`, `patch_fixes2.py`, `env_freeze.txt`, `latency_plot.png`, `times_*.csv` (per-frame times), `flip_*.csv`, `flip_heatmap_every5_fixed.png`, `cache_shapes.txt`.
