#!/bin/bash
# Start SoloHeaven with Qwen3.6-35B-A3B-mxfp8 (MoE, 3B active)
# Note: Qwen3.5/3.6 MoE uses DeltaNet (ArraysCache) — PLD incompatible
MODEL_PATH="$HOME/.lmstudio/models/unsloth/Qwen3.6-27B-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 1024 \
  --verbose \
  "$@"
