#!/bin/bash
# Start SoloHeaven with Gemma 4 31B 8-bit target; MTP drafter line commented out by default.
# Drafter slug: mlx-community/gemma-4-31B-it-assistant-bf16 (expected at $HOME/.lmstudio/models/<slug>).
# Caveat: 8bit target x bf16 drafter is exploratory (mlx-vlm 0.5.0 MTP is upstream-validated for bf16xbf16); on shape-mismatch or rollback_speculative_cache errors, fall back by re-commenting the --draft-model line below.
# Uncomment the --draft-model line below to enable MTP.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""

# DRAFT_ARGS=()
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 3)

mlx-soloheaven \
  --model "$MODEL_PATH" \
  "${DRAFT_ARGS[@]}" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  "$@"
