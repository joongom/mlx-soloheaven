#!/bin/bash
# Start SoloHeaven with 3 models (Qwen3.5-122B + Qwen3-Coder-Next + Qwen3.5-9B)
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven \
  --models \
    "$HOME/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-bf16" \
    "$HOME/.lmstudio/models/mlx-community/Qwen3-Coder-Next-8bit:no_think_tag" \
    "$HOME/.lmstudio/models/mlx-community/Qwen3.5-9B-bf16" \
  --memory-budget-gb 200 --gpu-keepalive --verbose "$@"
