#!/bin/bash
# Start SoloHeaven with Qwen3.6-35B-A3B (MLX 8-bit) — Qwen3_5-MoE architecture.
#
# Architecture (verified from config.json): model_type=qwen3_5_moe, 40 layers =
# 30 linear-attention (Gated DeltaNet, ArraysCache recurrent state) +
# 10 full-attention (KVCache, every 4th, full_attention_interval=4).
# MoE: 256 experts, 8 active/token (moe_intermediate_size 512, hidden 2048)
# => ~A3B (~3B active) per decode token. sliding_window = None.
#   => NO RotatingKVCache: the gemma4 1024-token sliding-window "wrap ->
#      cold-fill full prompt" TTFT problem does NOT apply. KV-cache prefix
#      reuse stays valid across long chats -> low TTFT at large context.
#   => MoE (~3B active) -> fast decode, much faster than the 27B dense-hybrid.
#      Best of both: fast TTFT (no cold-fill) + fast generation (MoE).
#      ~256K context, vision-capable, reasoning.
#
# Do NOT add (not applicable to this arch):
#   --draft-model / MTP : mlx-vlm MTP speculative decoding is gemma4-only.
#   --pld               : DeltaNet ArraysCache is PLD-incompatible.
# Sampling: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=20 / top_p=0.95). The engine now
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
#   Add --no-thinking below if you want faster, non-reasoning replies.
#
# NOTE: the LM Studio hub path (~/.lmstudio/hub/models/qwen/qwen3.6-35b-a3b) is
# a catalog stub with no weights; the runnable MLX weights are the path below.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
