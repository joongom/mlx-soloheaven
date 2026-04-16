#!/bin/bash
# Start SoloHeaven with small models (Qwen3.5-4B + Gemma4-E4B + GLM-4.7-Flash)
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven \
  --models \
    "$HOME/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-bf16" \
    "$HOME/.lmstudio/models/mlx-community/gemma-4-e4b-it-mxfp8" \
    "$HOME/.lmstudio/models/lmstudio-community/GLM-4.7-Flash-MLX-8bit" \
  --memory-budget-gb 100 --gpu-keepalive --verbose "$@"
