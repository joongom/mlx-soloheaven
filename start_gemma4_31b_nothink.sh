#!/bin/bash
# Start SoloHeaven with Gemma 4 31B (MLX 8-bit) — THINKING OFF, 10GB KV budget.
#
# Fast, non-reasoning variant of start_gemma4_31b.sh (same weights, same
# gemma4 family). Structure mirrors the most recent script
# (start_oss_120_opus.sh): 10GB KV pool + --prefill-step-size 2048.
#   --no-thinking            disable the <think> reasoning prefix globally
#                            (default is on). Clients can still opt back in
#                            per-request; per-model use the :no_think_tag suffix.
#   --memory-budget-gb 10    cap the KV-cache pool (budget-based LRU eviction)
#                            at 10GB.
#   --thinking-budget is intentionally OMITTED — it is a no-op once thinking
#     is off (there is no <think> block to budget).
#   --gpu-keepalive intentionally NOT set (matches the reference script; add
#     it if idle-warmup latency matters).
#
# Sampling MIRRORS this model's generation_config.json (temp=1.0 / top_k=64 /
#   top_p=0.95). The engine AUTO-APPLIES the same generation_config values as
#   defaults, so these flags are redundant-but-explicit for visibility + easy
#   override (clients still override per-request; trailing "$@" wins).
# No MTP drafter (plain 31B target); --repetition-penalty 1.1 stays as an
#   anti-loop safety net.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-MLX-8bit"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 10 \
  --no-thinking \
  --prefill-step-size 2048 \
  --temperature 1.0 \
  --top-k 64 \
  --top-p 0.95 \
  --repetition-penalty 1.1 \
  "$@"
