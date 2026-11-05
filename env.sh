#!/usr/bin/env bash
# Source before project commands. All experiment caches stay on the data disk.
export JAKA_VLA_ROOT=/data/jaka-vla
export HF_HOME="$JAKA_VLA_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$JAKA_VLA_ROOT/data/lerobot"
export UV_CACHE_DIR="$JAKA_VLA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$JAKA_VLA_ROOT/cache/python"
export TORCH_HOME="$JAKA_VLA_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$JAKA_VLA_ROOT/cache/torch_extensions"
export XDG_CACHE_HOME="$JAKA_VLA_ROOT/cache/xdg"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
