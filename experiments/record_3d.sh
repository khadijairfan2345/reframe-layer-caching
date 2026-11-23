#!/bin/bash
# record_3d.sh - Person 2, third GPU batch (~12 minutes). Run from ~/implicit-depth (after record_all.sh).
# Needs patch_pipe3d.py, time_parts.py, age_error.py in ~/implicit-depth. Replaces record_extra.sh.
#  1. skip-unused check: Every-5 with the unused 3D work removed (masks must be identical to Every-5)
#  2. pipe3d self-test: pieces == one-shot refresh (expect max abs diff 0)
#  3. timing: pipe3d-5 (x2), pipe3d-10, pipe3d-3
#  4. accuracy vs No Cache for pipe3d-5 / pipe3d-10 by position in cycle
STAMP=$(date +%Y%m%d_%H%M); R=~/p2_results/${STAMP}_3d; mkdir -p $R
ARGS="--config configs/models/implicit_depth.yaml --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt --data_config example_data/scans/config/config.yaml --rendered_depth_map_load_dir example_data/renders --dataset_path example_data/scans"
export CUDA_VISIBLE_DEVICES=0
step () { name=$1; out=$2; shift 2
  echo "=== $name ($(date +%H:%M:%S))" | tee -a $R/progress.log
  python time_parts.py $ARGS --output_base_path $out "$@" > $R/$name.log 2>&1 || echo "    FAILED" | tee -a $R/progress.log
  cp $out/part_times.csv $R/${name}_part_times.csv 2>/dev/null
  f=$(find $out -name frame_times.csv | head -1); [ -n "$f" ] && cp $f $R/${name}_frame_times.csv
  grep -A40 "frames:" $R/$name.log | head -400 > $R/${name}_summary.txt; }
{ python patch_skipunused.py; python patch_pipe3d.py; } > $R/patch.log 2>&1; cat $R/patch.log
REFRAME_SKIP_UNUSED=1 step every5_skip out_p2_every5_skip --use_reframe --reframe_cache_policy N-5
P3_SELFTEST=1 step pipe3d10_selftest out_p2_pipe3d10_selftest --use_reframe --reframe_cache_policy pipe3d-10
grep "pipe3d" $R/pipe3d10_selftest.log | head -20 > $R/pipe3d_selftest.txt
step pipe3d5_r1  out_p2_pipe3d5_r1  --use_reframe --reframe_cache_policy pipe3d-5
step pipe3d5_r2  out_p2_pipe3d5_r2  --use_reframe --reframe_cache_policy pipe3d-5
step pipe3d10_r1 out_p2_pipe3d10_r1 --use_reframe --reframe_cache_policy pipe3d-10
step pipe3d3_r1  out_p2_pipe3d3_r1  --use_reframe --reframe_cache_policy pipe3d-3
C=~/reframe-layer-caching/results/baseline/compare_runs.py
{ echo "every5_skip vs every5_r1 (should be 0.0):"; python $C out_p2_every5_r1/render/garden_chair out_p2_every5_skip/render/garden_chair; } > $R/skip_identical.txt 2>&1
python age_error.py out_p2_nocache out_p2_pipe3d5_r1:5  > $R/pipe3d5_accuracy.txt 2>&1  && mv age_error_per_frame.csv $R/pipe3d5_accuracy_per_frame.csv
python age_error.py out_p2_nocache out_p2_pipe3d10_r1:10 > $R/pipe3d10_accuracy.txt 2>&1 && mv age_error_per_frame.csv $R/pipe3d10_accuracy_per_frame.csv
python age_error.py out_p2_nocache out_p2_pipe3d3_r1:3  > $R/pipe3d3_accuracy.txt 2>&1  && mv age_error_per_frame.csv $R/pipe3d3_accuracy_per_frame.csv
# leftovers from record_extra.sh / equal-age baselines (no GPU needed, uses saved masks)
python age_error.py out_p2_nocache out_p2_pipe5_r1:5   > $R/pipe5_accuracy.txt 2>&1  && mv age_error_per_frame.csv $R/pipe5_accuracy_per_frame.csv
python age_error.py out_p2_nocache out_p2_every10:10   > $R/every10_accuracy.txt 2>&1 && mv age_error_per_frame.csv $R/every10_accuracy_per_frame.csv
python age_error.py out_p2_nocache out_p2_every20:20   > $R/every20_accuracy.txt 2>&1 && mv age_error_per_frame.csv $R/every20_accuracy_per_frame.csv
# hand-off: the output masks themselves, so Person 3 can run FLIP / any metric without a GPU
tar czf $R/masks.tgz $(for d in nocache every5_r1 every5_skip every10 every20 deltaL pipe5_r1 pipe3d3_r1 pipe3d5_r1 pipe3d10_r1; do [ -d out_p2_$d/render ] && echo out_p2_$d/render; done) 2>> $R/progress.log
du -h $R/masks.tgz | tee -a $R/progress.log
echo "=== done ($(date +%H:%M:%S)) -> $R" | tee -a $R/progress.log
cat $R/skip_identical.txt $R/pipe3d_selftest.txt
