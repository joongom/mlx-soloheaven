#!/bin/bash
# Start SoloHeaven with Qwen3.6-27B (MLX 8-bit) — Qwen3_5 architecture.
#
# Architecture (verified from config.json): model_type=qwen3_5, 64 layers =
# 48 linear-attention (Gated DeltaNet, ArraysCache recurrent state) +
# 16 full-attention (KVCache, every 4th, full_attention_interval=4).
# sliding_window = None.
#   => NO RotatingKVCache. None of the gemma4 1024-token sliding-window
#      "wrap -> cold-fill full prompt" TTFT problem applies here: KV-cache
#      prefix reuse stays valid across long chats, so TTFT stays low even at
#      multi-thousand-token context. ~256K context, vision-capable, reasoning.
#
# Do NOT add (not applicable to this arch):
#   --draft-model / MTP : mlx-vlm MTP speculative decoding is gemma4-only.
#   --pld               : DeltaNet ArraysCache is PLD-incompatible.
# Recommended sampling (set per-request by the client / OpenCode config):
#   temperature 0.6, top_k 20, top_p 0.95.
#   Add --no-thinking below if you want faster, non-reasoning replies.
#
# NOTE: the LM Studio hub path (~/.lmstudio/hub/models/qwen/qwen3.6-27b) is a
# catalog stub with no weights; the runnable MLX weights are the path below.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  "$@"
