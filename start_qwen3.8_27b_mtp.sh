#!/bin/bash
# Start SoloHeaven with Qwen3.8-27B (MLX 8-bit) + MTP head — Qwen3_5 DENSE-hybrid.
#
# Architecture (verified from config.json, IDENTICAL to Qwen3.6-27B):
# model_type=qwen3_5, 64 layers = 48 linear-attention (Gated DeltaNet,
# ArraysCache recurrent state) + 16 full-attention (KVCache, every 4th,
# full_attention_interval=4). DENSE: no MoE — plain MLP (intermediate_size
# 17408), hidden 5120, 24 Q / 4 KV heads, head_dim 256 => ALL ~27B params
# active per decode token (~28GB at 8-bit). sliding_window = None.
#   => NO RotatingKVCache: the gemma4 1024-token sliding-window "wrap ->
#      cold-fill full prompt" TTFT problem does NOT apply. KV-cache prefix
#      reuse stays valid across long chats -> low TTFT at large context.
#   => Dense => bandwidth-bound slow decode, which is exactly the regime where
#      the MTP head pays off (it was a wash on the 35B A3B MoE).
#      ~256K context, vision-capable, reasoning.
#
# MTP speculative decoding (qwen3_5_mtp head, NATIVE on mlx-lm):
#   --draft-model points at the Qwen3.8-27B MTP head (8-bit, ~459MB,
#   mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=false => shares the
#   target's embeddings + lm_head). The engine detects model_type=qwen3_5_mtp
#   and runs it on the DEFAULT mlx-lm backend (engine/qwen_mtp.py) — no
#   --backend flag needed/wanted (--backend mlx-vlm would be REFUSED: mlx-vlm
#   has no qwen3_5_mtp drafter; that flag is only for gemma4_assistant heads).
#   Head/target compatibility is asserted at load (assert_head_target_compat):
#   hidden 5120 == 5120, vocab 248320 == 248320 — verified matching, and both
#   sides are 8-bit affine group_size 64.
#   Per round: k recursive head drafts (k = --draft-block-size) -> 1 target
#   verify forward over k+1 tokens -> accept prefix; rejected drafts roll back
#   via ArraysCache snapshot/restore + KVCache trim.
#   --draft-block-size: this head's config says block_size=3; we override to 1.
#   CAVEAT: block 1 is carried over from the Qwen3.6-27B sweep (2026-06-10:
#   block 1 = 1.25x/1.37x, strictly better than 2 and 3 on that dense 27B
#   target). The 3.8 target is architecturally identical and the same size, so
#   block 1 is the well-founded starting point — but it has NOT been swept on
#   3.8 itself. Re-sweep with: ./start_qwen3.8_27b_mtp.sh --draft-block-size N
# Do NOT add (not applicable to this arch):
#   --pld               : DeltaNet ArraysCache is PLD-incompatible (and MTP
#                         takes precedence over PLD anyway).
# Sampling: the flags below MIRROR this model's generation_config.json
#   (temp=1.0 / top_k=20 / top_p=0.95); the engine AUTO-APPLIES the same values
#   as defaults, so they are redundant-but-explicit (clients still override
#   per-request; trailing "$@" wins).
#   The sibling start_qwen3.6_27b_mtp.sh runs production at temp 0.6 with
#   --no-thinking; add those here if you want the same behaviour.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/Qwen3.8-27B-MTP-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 1)
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
# --gpu-keepalive intentionally NOT set (matches start_qwen3.6_27b_mtp.sh).
# NOTE: a '#' inside a backslash-continued argument list BREAKS the command —
# keep disabled flags in comments ABOVE the command, never inline.
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  "${DRAFT_ARGS[@]}" \
  --prefill-step-size 2048 \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
