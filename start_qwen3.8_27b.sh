#!/bin/bash
# Start SoloHeaven with Qwen3.8-27B (MLX 8-bit) — Qwen3_5 architecture.
#
# Architecture (verified from config.json, IDENTICAL to Qwen3.6-27B):
# model_type=qwen3_5, Qwen3_5ForConditionalGeneration, 64 layers =
# 48 linear-attention (Gated DeltaNet, ArraysCache recurrent state) +
# 16 full-attention (KVCache, every 4th, full_attention_interval=4).
# hidden=5120, 24 heads / 4 KV, head_dim=256, vocab=248320,
# 262144 context, 8-bit affine (group_size 64), vision-capable.
# sliding_window = None.
#   => NO RotatingKVCache. None of the gemma4 1024-token sliding-window
#      "wrap -> cold-fill full prompt" TTFT problem applies here: KV-cache
#      prefix reuse stays valid across long chats, so TTFT stays low even at
#      multi-thousand-token context.
#
# MTP: this is the PLAIN (no speculative decoding) variant. A matching
#   qwen3_5_mtp head EXISTS for this target (mlx-community/Qwen3.8-27B-MTP-8bit,
#   ~459MB) — use start_qwen3.8_27b_mtp.sh to run with it. The head is a
#   SEPARATE checkpoint passed via --draft-model; it is not part of these
#   weights (this config has no MTP head of its own).
# Do NOT add (not applicable to this arch):
#   --pld               : DeltaNet ArraysCache is PLD-incompatible (and MTP
#                         takes precedence over PLD anyway).
#   --backend mlx-vlm   : mlx-vlm has no qwen3_5_mtp drafter; that backend's
#                         MTP path is for gemma4_assistant heads only.
# Sampling: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=20 / top_p=0.95). The engine
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
#   Add --no-thinking below if you want faster, non-reasoning replies.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit"

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
