#!/bin/bash
# record_all.sh - Person 2: record every GPU measurement needed for the runtime equations, in one go.
# Run on Cannon, inside a GPU session, from ~/implicit-depth, with binarydepth active:
#     bash record_all.sh
# Needs time_parts.py, crop_bench.py, patch_pipeline.py, age_error.py next to it. Takes about 25-30 minutes.
# Everything lands in ~/p2_results/<date-time>/ ; a failed step is logged and the script moves on.

STAMP=$(date +%Y%m%d_%H%M)
R=~/p2_results/$STAMP; mkdir -p $R
ARGS="--config configs/models/implicit_depth.yaml --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt --data_config example_data/scans/config/config.yaml --rendered_depth_map_load_dir example_data/renders --dataset_path example_data/scans"
export CUDA_VISIBLE_DEVICES=0

step () {   # step NAME OUTDIR [extra args]
  name=$1; out=$2; shift 2
  echo "=== $name  ($(date +%H:%M:%S))" | tee -a $R/progress.log
  python time_parts.py $ARGS --output_base_path $out "$@" > $R/$name.log 2>&1 || echo "    FAILED (see $name.log)" | tee -a $R/progress.log
  cp $out/part_times.csv $R/${name}_part_times.csv 2>/dev/null
  f=$(find $out -name frame_times.csv | head -1); [ -n "$f" ] && cp $f $R/${name}_frame_times.csv
  grep -A40 "frames:" $R/$name.log | head -80 > $R/${name}_summary.txt
}

# 0. machine and software record
{ date; hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  (cd ~/reframe-layer-caching && git rev-parse HEAD); } > $R/machine.txt 2>&1

# 1-2. per-part timing: No Cache, Every-5 three times, Delta
step nocache      out_p2_nocache
step every5_r1    out_p2_every5_r1 --use_reframe --reframe_cache_policy N-5
step every5_r2    out_p2_every5_r2 --use_reframe --reframe_cache_policy N-5
step every5_r3    out_p2_every5_r3 --use_reframe --reframe_cache_policy N-5
step deltaL       out_p2_deltaL    --use_reframe --reframe_cache_policy delta-0.25

# 2b. longer cache ages: Every-10 and Every-20 (for the age-vs-error curve)
step every10      out_p2_every10   --use_reframe --reframe_cache_policy N-10
step every20      out_p2_every20   --use_reframe --reframe_cache_policy N-20

# 3. crop benchmark: cost of the cacheable parts on one strip + border (for strip predictions)
echo "=== crop_bench  ($(date +%H:%M:%S))" | tee -a $R/progress.log
python crop_bench.py weights/implicit_depth_temporal_hypersim.ckpt $R/crop_bench.csv > $R/crop_bench.log 2>&1 || echo "    FAILED (see crop_bench.log)" | tee -a $R/progress.log

# 4. first block split (pipe-5): patch once, then time it twice
echo "=== patch_pipeline  ($(date +%H:%M:%S))" | tee -a $R/progress.log
python patch_pipeline.py > $R/patch_pipeline.log 2>&1 || echo "    FAILED (see patch_pipeline.log)" | tee -a $R/progress.log
step pipe5_r1     out_p2_pipe5_r1  --use_reframe --reframe_cache_policy pipe-5
step pipe5_r2     out_p2_pipe5_r2  --use_reframe --reframe_cache_policy pipe-5

# 5. output difference vs No Cache (masks), for a first look at accuracy
{ for d in every5_r1 deltaL pipe5_r1; do
    echo "$d vs nocache:"; python ~/reframe-layer-caching/results/baseline/compare_runs.py out_p2_nocache/render/garden_chair out_p2_$d/render/garden_chair; done; } > $R/mask_diff.txt 2>&1

# 6. error vs cache age (masks vs No Cache), up to age 19
python age_error.py out_p2_nocache out_p2_every5_r1:5 out_p2_every10:10 out_p2_every20:20 > $R/age_error.txt 2>&1 && mv age_error_per_frame.csv $R/ 2>/dev/null

echo "=== done  ($(date +%H:%M:%S)) -> $R" | tee -a $R/progress.log
ls $R
