#!/bin/bash
# Start SoloHeaven with EXAONE-4.5-33B (MLX 8-bit) — LG AI Research, TEXT tower.
#
# Build this model first (the 8-bit MLX weights do not exist on the Hub):
#   ./convert_exaone4_5.py                 # ~69G bf16 source -> ~35G 8-bit
# See convert_exaone4_5.py and src/mlx_soloheaven/models/exaone4_5.py for why a
# wrapper is needed: the published checkpoint is model_type=exaone4_5, which
# mlx-lm does not implement.
#
# Architecture (config.json of the CONVERTED model: model_type=exaone4):
# EXAONE-4.5's text tower is bit-for-bit EXAONE-4.0's architecture, so the
# converted model is a plain mlx-lm `exaone4` — no runtime patching needed.
# 64 layers, hidden 5120, intermediate 27392, 40 Q / 8 KV heads, head_dim 128,
# vocab 153600, untied lm_head. DENSE: no MoE, so ALL ~33B params are active
# per decode token (~35G at 8-bit) => bandwidth-bound, SLOW decode — expect the
# same order as the dense Qwen3.6-27B (~17 tps there, less here at 33B), not
# the 35B A3B MoE's ~90 tps.
#
# HYBRID attention, pattern "LLLG" (layer_types in config):
#   * 48 layers sliding_attention, window 4096  -> RotatingKVCache
#   * 16 layers full_attention, and those are **NoPE** (no positional encoding
#     at all — verified against transformers' `if sliding_window is None or
#     is_sliding` guard). That is what buys the 262,144-token context.
#   => Unlike qwen3_5 (sliding_window=None, pure KVCache), this model DOES use
#      RotatingKVCache, so the gemma4-style caveat applies: once a session runs
#      past the window, the local layers' KV prefix is no longer reusable. It is
#      much milder here than gemma4 — the window is 4096 rather than 1024, and
#      the 16 global layers keep a full, always-reusable KV — but TTFT on very
#      long chats will still be worse than on a pure-KVCache model.
#
# NO --draft-model: the original checkpoint ships an MTP head (mtp.*, 1 layer,
#   DeepSeek/GLM-style fc over concat[norm(embed), norm(hidden)]), but
#   convert_exaone4_5.py drops it — the engine's speculative path
#   (engine/qwen_mtp.py) only implements the qwen3_5_mtp head, and this one is
#   a different layout. Converting it into a drafter is a genuine follow-up
#   opportunity: MTP pays off exactly on slow dense targets like this one.
# Do NOT add --pld here without measuring: PLD interacts with RotatingKVCache
#   (see tests/test_pld_gemma4_cache_safety.py).
#
# Sampling: LG's README gives per-use-case recommendations. The flags below use
#   the **Korean / document** profile (temperature=0.6, top_p=0.95, top_k=20)
#   rather than the general-purpose one (temperature=1.0, top_p=0.95, no top_k),
#   which is the right default here. Note the model's generation_config.json
#   asks for presence_penalty=1.5; SoloHeaven has no presence-penalty flag, so
#   --repetition-penalty 1.1 stands in as the anti-loop net (clients can still
#   override sampling per request).
# Thinking: EXAONE 4.5 defaults to enable_thinking=True (unlike EXAONE 4.0) and
#   its template opens an unclosed "<think>" in the generation prompt. Drop the
#   --no-thinking below to run in reasoning mode.
MODEL_PATH="${MODEL_PATH:-$HOME/.lmstudio/models/mlx-community/EXAONE-4.5-33B-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --no-thinking \
  --repetition-penalty 1.1 \
  "$@"
