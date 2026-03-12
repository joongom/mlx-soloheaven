#!/bin/bash
# Start SoloHeaven with Qwen3.5-397B single model
# Set MODEL_PATH to your local model directory
MODEL_PATH="${SOLOHEAVEN_397B_PATH:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-397B-A17B-MLX-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven --model "$MODEL_PATH" --gpu-keepalive --verbose "$@"
