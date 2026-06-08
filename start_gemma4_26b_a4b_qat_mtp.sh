#!/bin/bash
# Start SoloHeaven with Gemma 4 26B-A4B QAT (MoE, ~4B active) MLX-4bit target +
# QAT-matched 4-bit MTP drafter.
#
# Target:  lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit
#   gemma4, 4-bit affine (early-layer MLP/router 8-bit mixed), ~15.6GB.
#   Quantization-Aware Trained -> better quality-per-bit than post-training 4-bit.
# Drafter: mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit (gemma4_assistant, 4-bit, ~236MB).
#   QAT-matched to the target: both QAT-trained, so the drafter's distribution should
#   align with the target's better than the it-8bit drafter would -> higher acceptance.
#   Speculative decoding is exact-greedy (the target VERIFIES every drafted token), so
#   output == the QAT target's own greedy/sampled output regardless of the drafter.
#   Override with DRAFT_PATH=... (e.g. the bf16 drafter mlx-community/gemma-4-26B-A4B-it-assistant-bf16).
# To disable MTP (plain decode), comment the active DRAFT_ARGS line below and uncomment the empty one.
#
# Fixes baked in (same gemma4 engine fixes as start_gemma4_26b_a4b_mtp.sh):
#  - --thinking-budget 4096: thinking-budget is enforced in the MTP speculative path
#    (forces <channel|> after 4096 thinking tokens) so a stuck thinking block can't run
#    away to max_tokens. Per-request override via the OpenAI `thinking_budget` field.
#  - --repetition-penalty 1.1: wired into the MTP path; safety net against degenerate loops.
#  - Code fixes (auto): tool-call object-array parser, _HOT_PATH_FAST 2-token fix,
#    gemma4 <|channel>thought stripping on the OpenAI endpoint.
#  - NOTE: the --temperature/--top-k/--top-p flags below MIRROR this model's
#    generation_config.json (temp=1.0 / top_k=64 / top_p=0.95). The engine now
#    AUTO-APPLIES those same generation_config values as defaults, so the flags
#    are redundant-but-explicit: kept for visibility + easy per-deploy override.
#    OpenCode/clients can still override per-request. rep-penalty 1.1 is kept as
#    a loop safety net (NOT auto-read from generation_config).
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit}"

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
  --temperature 1.0 \
  --top-k 64 \
  --top-p 0.95 \
  --repetition-penalty 1.1 \
  --thinking-budget 4096 \
  "$@"
