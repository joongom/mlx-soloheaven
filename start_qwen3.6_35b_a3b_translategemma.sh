#!/bin/bash
# Start SoloHeaven with TWO models loaded together:
#   1) qwen3.6-35b-a3b   Qwen3.6-35B-A3B (MLX 8-bit) — general chat/reasoning,
#                        Qwen3_5-MoE (~3B active/token). Same target as
#                        start_qwen3.6_35b_a3b_mtp.sh.
#   2) translategemma    translategemma-12b-it (MLX 8-bit) — translation-
#                        specialist model (gemma3 arch, sliding_window=None).
#                        NOTE: 27b-8bit is downloading; swap TRANSLATE_PATH to
#                        translategemma-27b-it-8bit once it finishes.
# # Multi-model launcher (--models). Key differences vs the single-model MTP # script:
#   * NO --draft-model. --draft-model is a SINGLE GLOBAL flag; in --models mode
#     it would be applied to BOTH models, and the qwen3_5_mtp head is not a
#     valid drafter for the gemma3 translate model. The block-size sweep in
#     start_qwen3.6_35b_a3b_mtp.sh also shows MTP is ~a wash on this A3B MoE
#     target (already ~90-97 tps plain-decode), so dropping the drafter costs
#     essentially nothing here.
#   * NO shared --temperature/--top-k/--top-p. Sampling flags are GLOBAL and
#     would clobber both models with one profile. Instead each model auto-reads
#     its OWN generation_config.json defaults in the engine (qwen: temp0.6-ish
#     via its config; translategemma: top_k64/top_p0.95). Clients still override
#     per request — translation callers typically want a low temperature.
#   * --engine-mode: --models auto-falls-back to 'inprocess' (the 'process'
#     child-process fast path is single-model only).
#
# Memory: weights ~35G (qwen 8-bit) + ~12G (translategemma 8-bit) = ~47G of the
# 128G RAM; --memory-budget-gb below is the KV-cache budget on top of weights.
QWEN_PATH="${QWEN_PATH:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit}"
TRANSLATE_PATH="${TRANSLATE_PATH:-$HOME/.lmstudio/models/mlx-community/translategemma-27b-it-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate

# alias=path so clients can select by clean names in the OpenAI 'model' field.
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --models \
    "qwen3.6-35b-a3b=$QWEN_PATH" \
    "translategemma=$TRANSLATE_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --thinking-budget 4096 \
  --no-thinking \
  --repetition-penalty 1.1 \
  "$@"
