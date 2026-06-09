#!/bin/bash
# Start SoloHeaven with Gemma 4 models (31B Dense + 26B MoE)
# Multi-model launcher: CLI sampling flags are shared, so no per-model flags
# here. Each model auto-reads its own generation_config.json sampling defaults
# in the engine (gemma-4: temp=1.0 / top_k=64 / top_p=0.95); per-request
# overrides still apply.
cd "$(dirname "$0")"
source .venv/bin/activate
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --models \
    "$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit" \
    "$HOME/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-mxfp8" \
  --memory-budget-gb 100 --gpu-keepalive --verbose --thinking-budget 4096 --repetition-penalty 1.1 "$@"
