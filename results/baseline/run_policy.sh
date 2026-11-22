#!/bin/bash
# usage: bash run_policy.sh OUTDIR [POLICY]    (no POLICY = No Cache)
OUT=$1; POL=$2; EXTRA=""
if [ -n "$POL" ]; then EXTRA="--use_reframe --reframe_cache_policy $POL"; fi
CUDA_VISIBLE_DEVICES=0 python -m inference.inference \
    --config configs/models/implicit_depth.yaml \
    --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt \
    --data_config example_data/scans/config/config.yaml \
    --rendered_depth_map_load_dir example_data/renders \
    --output_base_path $OUT \
    --dataset_path example_data/scans $EXTRA
