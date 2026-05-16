#!/usr/bin/env bash
set -e
source /home/zsh/anaconda3/etc/profile.d/conda.sh
conda activate seedvr

export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export MAX_JOBS=8

cd /home/zsh/UAV_video_repair/third_party/apex_src

LOG=/home/zsh/UAV_video_repair/build_apex.log
: > "$LOG"
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  ./ >>"$LOG" 2>&1
echo "----- tail -----"
tail -n 60 "$LOG"
