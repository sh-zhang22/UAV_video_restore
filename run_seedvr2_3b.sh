#!/usr/bin/env bash
set -e
source /home/zsh/anaconda3/etc/profile.d/conda.sh
conda activate seedvr
cd /home/zsh/UAV_video_repair/third_party/SeedVR
echo "CWD=$(pwd)"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec torchrun --nproc-per-node=1 --master_port=29501 \
  projects/inference_seedvr2_3b.py \
  --video_path test_videos --output_dir results \
  --seed 666 --res_h "${RES_H:-720}" --res_w "${RES_W:-960}" --sp_size 1

