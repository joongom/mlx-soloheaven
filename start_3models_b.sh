#!/bin/bash
# Start SoloHeaven with 3 models (Qwen3-Coder-Next + GPT-OSS-20B + GLM-4.7-Flash)
# Multi-model launcher: CLI sampling flags are shared, so no per-model flags
# here. Each model auto-reads its own generation_config.json sampling defaults
# in the engine; per-request overrides still apply.
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven \
  --models \
    "$HOME/.lmstudio/models/mlx-community/Qwen3-Coder-Next-bf16:no_think_tag" \
    "$HOME/.lmstudio/models/lmstudio-community/GLM-4.7-Flash-MLX-8bit" \
    "$HOME/.lmstudio/models/lmstudio-community/gpt-oss-120b-mlx-8bit" \
  --memory-budget-gb 200 --gpu-keepalive --verbose "$@"
