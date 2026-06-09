#!/bin/bash
# Start SoloHeaven with Gemma 4 31B 8-bit target; MTP drafter line commented out by default.
# Drafter slug: mlx-community/gemma-4-31B-it-assistant-bf16 (expected at $HOME/.lmstudio/models/<slug>).
# Caveat: 8bit target x bf16 drafter is exploratory (mlx-vlm 0.5.0 MTP is upstream-validated for bf16xbf16); on shape-mismatch or rollback_speculative_cache errors, fall back by re-commenting the --draft-model line below.
# Uncomment the --draft-model line below to enable MTP.
# NOTE: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=64 / top_p=0.95). The engine now
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""

# DRAFT_ARGS=()
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 3)

# --backend mlx-vlm: after the mlx-lm-default migration, MTP is an explicit
#   mlx-vlm opt-in. Without this flag the new backend gate would load this
#   (text) model via mlx-lm and the --draft-model guard would raise.
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --backend mlx-vlm \
  "${DRAFT_ARGS[@]}" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --temperature 1.0 \
  --top-k 64 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
