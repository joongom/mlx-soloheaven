#!/bin/bash
# Start SoloHeaven with Gemma 4 31B (MLX 8-bit), single model, no MTP drafter.
# NOTE: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=64 / top_p=0.95). The engine now
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --temperature 1.0 \
  --top-k 64 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
