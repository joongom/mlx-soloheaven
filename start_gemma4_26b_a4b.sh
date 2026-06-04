#!/bin/bash
# Start SoloHeaven with Gemma 4 26B-A4B (MoE, ~4B active) 8-bit target + bf16 MTP drafter.
# Target slug:  lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit
# Drafter slug: mlx-community/gemma-4-26B-A4B-it-assistant-bf16 (HTTP-verified 2026-05-13)
# Caveat: 8bit target x bf16 drafter is exploratory upstream; on shape-mismatch errors, comment the active DRAFT_ARGS line and re-run.
# To disable MTP, comment the active DRAFT_ARGS line below and uncomment the empty one.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""


mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  "$@"
