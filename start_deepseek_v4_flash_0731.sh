#!/bin/bash
# Start SoloHeaven with DeepSeek-V4-Flash-0731 (MLX 2-bit mixed, 94.5 GB).
#
# Build this model first (the MLX weights do not exist on the Hub):
#   .venv/bin/python convert_deepseek_v4.py \
#     ~/.lmstudio/models/deepseek-ai/DeepSeek-V4-Flash-0731 "$MODEL_PATH"
# ~34 min, official fp8 167G -> 94.5G. The port lives in
# src/mlx_soloheaven/models/deepseek_v4.py; plan and evidence in
# docs/specs/deepseek-v4-mlx-port.md.
#
# QUANTIZATION: routed experts 2-bit, everything else 8-bit — the same recipe
# ds4 uses. The experts' per-group scales are chosen by ERROR SEARCH rather
# than min/max (see quantize_search in the converter), which is what makes the
# build usable at all: teacher-forced perplexity 7.11 -> 3.69 overall, and
# 17.91 -> 6.60 in Korean, at identical size. Check a rebuild with
#   .venv/bin/python validate_deepseek_v4.py ppl
#
# Architecture: 284B total / MoE 256-way top-6 + 1 shared, 43 layers
# (2 dense + 21 compressed(4)+DSA-Indexer + 20 compressed(128)), MLA with a
# single 512-wide KV latent, 128-token sliding ring per layer, hash routing
# (tid2eid lookup) on layers 0-2, Hyper-Connections (hc_mult 4), 1M context.
#
# MEMORY: weights are 94.5 GB — close all other model servers first. KV is
# tiny by design (~187 MB at 32K context), so --memory-budget-gb stays small.
#
# CACHE SEMANTICS on this engine:
#   * append-only prefix reuse WORKS (continuation prefill implemented and
#     consistency-tested) — multi-turn TTFT benefits fully;
#   * the per-layer cache is NOT trimmable (128-slot ring + pooled compressor
#     state cannot be exactly rolled back), so branch/regenerate and PLD take
#     the engine's fail-closed cold-fill path automatically. No flags needed.
# NO --draft-model: the checkpoint's mtp.*/DSpark blocks are not ported.
#
# Thinking: the model natively reasons in <think>...</think>. Chat mode
# (--no-thinking) primes a closed think block — the validated configuration.
# Drop --no-thinking to serve in reasoning mode.
#
# Sampling: the official generation_config.json says temperature 1.0 /
# top_p 1.0. 2-bit experts still fatten the logit tails, so serve slightly
# cooler by default; clients can override per request.
MODEL_PATH="${MODEL_PATH:-$HOME/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-search}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 8 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --no-thinking \
  --repetition-penalty 1.05 \
  "$@"
