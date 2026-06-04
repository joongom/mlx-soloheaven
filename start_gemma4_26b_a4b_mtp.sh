#!/bin/bash
# Start SoloHeaven with Gemma 4 26B-A4B (MoE, ~4B active) 8-bit target + 8-bit MTP drafter.
# Target slug:  lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit
# Drafter slug: guardiangate1775/gemma-4-26B-A4B-it-assistant-8bit (8-bit, matches the
#   8-bit target's quantization; verified: loads, output byte-identical to plain greedy,
#   pre-wrap acceptance ~2.6, slightly faster than the bf16 drafter). Override with
#   DRAFT_PATH=... to use the bf16 drafter (mlx-community/gemma-4-26B-A4B-it-assistant-bf16).
# Post-wrap behaviour: the engine auto-disables the drafter once the 1024 sliding-window
#   ring wraps (acceptance collapses there) and falls back to plain decode — so throughput
#   floors at plain-decode speed instead of going net-negative. See _rotating_wrapped().
# To disable MTP entirely, comment the active DRAFT_ARGS line below and uncomment the empty one.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/guardiangate1775/gemma-4-26B-A4B-it-assistant-8bit}"

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
