#!/bin/bash
# Start SoloHeaven with Qwen3.5-122B-A10B (MLX mxfp4) — Qwen3_5-MoE architecture.
#
# Architecture (config.json): model_type=qwen3_5_moe — same family as
# Qwen3.6-35B-A3B, so the same cache properties apply:
#   * sliding_window = None => NO RotatingKVCache. The gemma4 1024-token
#     sliding-window "cold-fill full prompt" TTFT problem does NOT apply; KV
#     prefix reuse stays valid across long chats -> low TTFT at large context.
#   * MoE, A10B (~10B active/token) -> fast decode for a 122B-class model.
# Quantization: mxfp4 (4-bit experts, group_size 32) with the per-layer MoE
#   gate / shared_expert_gate kept at 8-bit — ~61G on disk, comfortably within
#   128G RAM (the 8-bit variant would be ~122G and leave no room for KV).
#
# NO --draft-model: there is no Qwen3.5-122B MTP head published, so this runs
#   plain decode (the A10B MoE is already fast). If a matching qwen3_5_mtp head
#   appears, add: --draft-model <head> --draft-block-size 1 (block 1 was the
#   only non-regressing setting on the sibling A3B MoE — see
#   start_qwen3.6_35b_a3b_mtp.sh).
#
# Sampling: --temperature/--top-k/--top-p MIRROR this model's
#   generation_config.json (temp=0.6 / top_k=20 / top_p=0.95). The engine
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request). Add --no-thinking for faster non-reasoning replies.
MODEL_PATH="${MODEL_PATH:-$HOME/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-mxfp4}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 10 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --no-thinking \
  --repetition-penalty 1.1 \
  "$@"
