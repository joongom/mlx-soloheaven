#!/bin/bash
# Start SoloHeaven with GLM-5.1-MXFP4-Q8 (no-thinking + prefill 8192 + PLD)
MODEL_PATH="$HOME/.lmstudio/models/mlx-community/GLM-5.1-MXFP4-Q8"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven --model "$MODEL_PATH" --memory-budget-gb 50 --gpu-keepalive --no-thinking --prefill-step-size 8192 --pld --verbose "$@"
