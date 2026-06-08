#!/bin/bash
# Start SoloHeaven with Qwen3.6-27B (unsloth MLX 8-bit), single model.
# Note: Qwen3.5/3.6 MoE uses DeltaNet (ArraysCache) — PLD incompatible
# NOTE: the --temperature/--top-k/--top-p flags below MIRROR the Qwen3.6-27B
#   generation_config.json (temp=1.0 / top_k=20 / top_p=0.95, verified from the
#   lmstudio-community Qwen3.6-27B quant). The engine also AUTO-APPLIES the
#   model's own generation_config values as defaults; these flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
MODEL_PATH="$HOME/.lmstudio/models/unsloth/Qwen3.6-27B-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 1024 \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --verbose \
  "$@"
