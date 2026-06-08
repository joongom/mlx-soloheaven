#!/bin/bash
# Start SoloHeaven with Gemma 4 31B bf16 target + bf16 MTP drafter (upstream-validated combo).
# Drafter slug: mlx-community/gemma-4-31B-it-assistant-bf16
# Target slug:  mlx-community/gemma-4-31b-it-bf16
# This is mlx-vlm 0.5.0's reference MTP combination (bf16 x bf16) — use this when comparing
# acceptance rate against the 8bit-target variant to isolate quant-noise contribution.
# NOTE: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=64 / top_p=0.95). The engine now
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
MODEL_PATH="$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-bf16"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""

# DRAFT_ARGS=()
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 3)

mlx-soloheaven \
  --model "$MODEL_PATH" \
  "${DRAFT_ARGS[@]}" \
  --memory-budget-gb 60 \
  --gpu-keepalive \
  --temperature 1.0 \
  --top-k 64 \
  --top-p 0.95 \
  "$@"
