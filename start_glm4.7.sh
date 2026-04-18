#!/bin/bash
# Start SoloHeaven with GLM-4.7-8bit
MODEL_PATH="$HOME/.lmstudio/models/mlx-community/GLM-4.7-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven --model "$MODEL_PATH" --memory-budget-gb 50 --gpu-keepalive --verbose "$@"
