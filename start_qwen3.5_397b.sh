#!/bin/bash
# Start SoloHeaven with Qwen3.5-397B single model
# Set MODEL_PATH to your local model directory
# no generation_config sampling available; engine uses built-in fallback defaults
MODEL_PATH="${SOLOHEAVEN_397B_PATH:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-397B-A17B-MLX-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven --model "$MODEL_PATH" --memory-budget-gb 50 --gpu-keepalive --verbose "$@"
