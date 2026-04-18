#!/bin/bash
# Start SoloHeaven with GLM-5.1-MXFP4-Q8
# Note: prefill_step_size kept at default (2048). 8192 causes Metal OOM on
# long prompts (>100K tokens) because each chunk's attention against the
# full KV cache exceeds the ~60GB free memory after model load (~450GB).
MODEL_PATH="$HOME/.lmstudio/models/mlx-community/GLM-5.1-MXFP4-Q8"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
mlx-soloheaven --model "$MODEL_PATH" --memory-budget-gb 50 --gpu-keepalive --no-thinking --verbose "$@"
# NOTE: --pld removed. Observed acceptance rate ~12% on casual/reasoning
# workloads, which means PLD's verification overhead exceeds the gain.
# Add --pld back if the workload is copy-heavy (code editing, RAG, tool
# arg repetition) where acceptance rate typically exceeds 30%.
