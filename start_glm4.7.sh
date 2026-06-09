#!/bin/bash
# Start SoloHeaven with GLM-4.7-8bit
# no generation_config sampling available; engine uses built-in fallback defaults
MODEL_PATH="$HOME/.lmstudio/models/mlx-community/GLM-4.7-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven --model "$MODEL_PATH" --memory-budget-gb 50 --gpu-keepalive --verbose --thinking-budget 4096 --repetition-penalty 1.1 "$@"
