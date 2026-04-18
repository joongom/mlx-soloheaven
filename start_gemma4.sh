#!/bin/bash
# Start SoloHeaven with Gemma 4 models (31B Dense + 26B MoE)
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven \
  --models \
    "$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit" \
    "$HOME/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-mxfp8" \
  --memory-budget-gb 100 --gpu-keepalive --verbose "$@"
