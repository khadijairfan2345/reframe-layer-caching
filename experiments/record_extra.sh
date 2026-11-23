#!/bin/bash
# record_extra.sh - Person 2, second GPU batch (~8 minutes). Run from ~/implicit-depth after record_all.sh.
#  1. tests the "unused work" hypothesis (REFRAME_SKIP_UNUSED=1) on Every-5 and pipe-5
#  2. checks the masks are identical with and without skipping
#  3. accuracy of pipe-5 by position in its 5-frame cycle
STAMP=$(date +%Y%m%d_%H%M); R=~/p2_results/${STAMP}_extra; mkdir -p $R
ARGS="--config configs/models/implicit_depth.yaml --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt --data_config example_data/scans/config/config.yaml --rendered_depth_map_load_dir example_data/renders --dataset_path example_data/scans"
export CUDA_VISIBLE_DEVICES=0
step () { name=$1; out=$2; shift 2
  echo "=== $name ($(date +%H:%M:%S))" | tee -a $R/progress.log
  python time_parts.py $ARGS --output_base_path $out "$@" > $R/$name.log 2>&1 || echo "    FAILED" | tee -a $R/progress.log
  cp $out/part_times.csv $R/${name}_part_times.csv 2>/dev/null
  f=$(find $out -name frame_times.csv | head -1); [ -n "$f" ] && cp $f $R/${name}_frame_times.csv
  grep -A40 "frames:" $R/$name.log | head -120 > $R/${name}_summary.txt; }
python patch_skipunused.py > $R/patch_skipunused.log 2>&1
REFRAME_SKIP_UNUSED=1 step every5_skip out_p2_every5_skip --use_reframe --reframe_cache_policy N-5
REFRAME_SKIP_UNUSED=1 step pipe5_skip  out_p2_pipe5_skip  --use_reframe --reframe_cache_policy pipe-5
C=~/reframe-layer-caching/results/baseline/compare_runs.py
{ echo "every5_skip vs every5_r1 (should be 0.0):"; python $C out_p2_every5_r1/render/garden_chair out_p2_every5_skip/render/garden_chair
  echo "pipe5_skip vs pipe5_r1 (should be 0.0):";   python $C out_p2_pipe5_r1/render/garden_chair out_p2_pipe5_skip/render/garden_chair; } > $R/skip_identical.txt 2>&1
python age_error.py out_p2_nocache out_p2_pipe5_r1:5 > $R/pipe5_accuracy_by_cycle.txt 2>&1 && mv age_error_per_frame.csv $R/pipe5_accuracy_per_frame.csv
echo "=== done ($(date +%H:%M:%S)) -> $R" | tee -a $R/progress.log; cat $R/skip_identical.txt
