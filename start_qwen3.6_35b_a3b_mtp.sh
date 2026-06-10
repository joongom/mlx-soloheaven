#!/bin/bash
# Start SoloHeaven with Qwen3.6-35B-A3B (MLX 8-bit) — Qwen3_5-MoE architecture.
#
# Architecture (verified from config.json): model_type=qwen3_5_moe, 40 layers =
# 30 linear-attention (Gated DeltaNet, ArraysCache recurrent state) +
# 10 full-attention (KVCache, every 4th, full_attention_interval=4).
# MoE: 256 experts, 8 active/token (moe_intermediate_size 512, hidden 2048)
# => ~A3B (~3B active) per decode token. sliding_window = None.
#   => NO RotatingKVCache: the gemma4 1024-token sliding-window "wrap ->
#      cold-fill full prompt" TTFT problem does NOT apply. KV-cache prefix
#      reuse stays valid across long chats -> low TTFT at large context.
#   => MoE (~3B active) -> fast decode, much faster than the 27B dense-hybrid.
#      Best of both: fast TTFT (no cold-fill) + fast generation (MoE).
#      ~256K context, vision-capable, reasoning.
#
# MTP speculative decoding (qwen3_5_mtp head, NATIVE on mlx-lm):
#   --draft-model points at the Qwen3.6-35B-A3B MTP head (5-bit, ~0.6GB,
#   1 full-attention layer; shares the target's embeddings + lm_head). The
#   engine detects model_type=qwen3_5_mtp and runs it on the DEFAULT mlx-lm
#   backend (engine/qwen_mtp.py) — no --backend flag needed/wanted
#   (--backend mlx-vlm would be REFUSED: mlx-vlm has no qwen3_5_mtp drafter;
#   that flag is only for gemma4_assistant MTP heads).
#   Per round: 3 recursive head drafts -> 1 target verify forward -> accept
#   prefix; rejected drafts roll back via ArraysCache snapshot/restore +
#   KVCache trim with per-layer fail-closed verification.
#   --draft-block-size: head config says 3, but we OVERRIDE to 1 (see the
#   sweep summary above DRAFT_ARGS below — block 3 is a measured regression
#   on this MoE target).
# Do NOT add (not applicable to this arch):
#   --pld               : DeltaNet ArraysCache is PLD-incompatible (and MTP
#                         takes precedence over PLD anyway).
# Sampling: the --temperature/--top-k/--top-p flags below MIRROR this model's
#   generation_config.json (temp=1.0 / top_k=20 / top_p=0.95). The engine now
#   AUTO-APPLIES the same generation_config values as defaults; the flags are
#   redundant-but-explicit for visibility + easy override (clients still
#   override per-request).
#   Add --no-thinking below if you want faster, non-reasoning replies.
#
# NOTE: the LM Studio hub path (~/.lmstudio/hub/models/qwen/qwen3.6-35b-a3b) is
# a catalog stub with no weights; the runnable MLX weights are the path below.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/Qwen3.6-35B-A3B-MTP-5bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# ---------------------------------------------------------------------------
# --draft-block-size 1 (NOT the head-config default 3) — block-size sweep,
# 2026-06-10, M5 Max, Qwen3.6-35B-A3B-8bit + MTP head 5-bit, production
# sampling temp=1.0/top_k=20/top_p=0.95, 500 gen tokens, seed 1234:
#
#   block   152-tok prompt        2542-tok prompt       9989-tok prompt
#   no-mtp  96.5 tps (1.000x)     94.7 tps (1.000x)     89.6 tps (1.000x)
#   1       103.3 tps 1.07x a.86  91.5 tps 0.97x a.78   91.3 tps 1.02x a.83
#   2       102.8 tps 1.07x a.75  93.6 tps 0.99x a.71   87.6 tps 0.98x a.70
#   3        87.8 tps 0.91x a.60  82.5 tps 0.87x a.58   81.5 tps 0.91x a.61
#   4        74.2 tps 0.77x a.46  79.4 tps 0.84x a.55   61.8 tps 0.69x a.45
#
# Mean production-sampling speedup: block1=1.019x, block2=1.010x,
# block3=0.897x, block4=0.766x. Block 1 is the only setting that never loses
# badly (worst 0.966x); the default 3 is a consistent ~9-13% REGRESSION.
# 10K context does NOT shift the winner (block1 stays best at 1.019x) — it
# amplifies the deep-block penalty instead (block4 falls to 0.69x: acceptance
# 0.45 means most rounds reject, and each rejection pays an ArraysCache
# restore + replay forward that gets pricier as KV grows). Greedy spot-check
# matches (block1 1.069x, block3 0.922x). Free-form/low-acceptance content is
# worst for MTP (block2 0.813x, block3 0.707x at 60-tok prompt).
# Root cause: this A3B MoE target already plain-decodes at ~90-97 tps, so
# head-forward + verify + rollback overhead ≈ the cost of just decoding
# (unlike dense gemma4, where MTP gave 60→97.8 tps). Net: block 1 is a wash
# vs plain decode within noise; honest alternative is dropping --draft-model
# entirely. Override per launch: ./start_qwen3.6_35b_a3b_mtp.sh
# --draft-block-size N (trailing "$@" wins over DRAFT_ARGS).
# ---------------------------------------------------------------------------
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 1)
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  "${DRAFT_ARGS[@]}" \
  --gpu-keepalive \
  --prefill-step-size 2048 \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
