#!/bin/bash
# Start SoloHeaven with Qwen3.6-27B (MLX 8-bit) — Qwen3_5 DENSE-hybrid architecture.
#
# Architecture (verified from config.json): model_type=qwen3_5, 64 layers =
# 48 linear-attention (Gated DeltaNet, ArraysCache recurrent state) +
# 16 full-attention (KVCache, every 4th, full_attention_interval=4).
# DENSE: no MoE — plain MLP (intermediate_size 17408), hidden 5120,
# 24 Q / 4 KV heads, head_dim 256 => ALL ~27B params active per decode token
# (~29.7GB at 8-bit). sliding_window = None.
#   => NO RotatingKVCache: the gemma4 1024-token sliding-window "wrap ->
#      cold-fill full prompt" TTFT problem does NOT apply. KV-cache prefix
#      reuse stays valid across long chats -> low TTFT at large context.
#   => Dense => bandwidth-bound SLOW decode (~17 tps plain on M5 Max), the
#      opposite of the 35B A3B MoE (~90+ tps) — which is exactly why the MTP
#      head genuinely pays off HERE (see the sweep above DRAFT_ARGS below)
#      where it was a wash on the MoE. ~256K context, vision-capable,
#      reasoning.
#
# MTP speculative decoding (qwen3_5_mtp head, NATIVE on mlx-lm):
#   --draft-model points at the Qwen3.6-27B MTP head (bf16, ~0.8GB,
#   1 dense layer; shares the target's embeddings + lm_head). The
#   engine detects model_type=qwen3_5_mtp and runs it on the DEFAULT mlx-lm
#   backend (engine/qwen_mtp.py) — loads strict, zero code changes, and no
#   --backend flag needed/wanted (--backend mlx-vlm would be REFUSED:
#   mlx-vlm has no qwen3_5_mtp drafter; that flag is only for
#   gemma4_assistant MTP heads).
#   Per round: k recursive head drafts (k = --draft-block-size) -> 1 target
#   verify forward over k+1 tokens -> accept prefix; rejected drafts roll
#   back via ArraysCache snapshot/restore + KVCache trim with per-layer
#   fail-closed verification.
#   --draft-block-size: head config says 3, but we OVERRIDE to 1 (see the
#   sweep summary above DRAFT_ARGS below — block 1 wins at every measured
#   prompt size on this dense target; deeper blocks are strictly worse).
# Do NOT add (not applicable to this arch):
#   --pld               : DeltaNet ArraysCache is PLD-incompatible (and MTP
#                         takes precedence over PLD anyway).
# Sampling: this model's generation_config.json says temp=1.0 / top_k=20 /
#   top_p=0.95 and the engine AUTO-APPLIES those as defaults; the flags below
#   keep top_k/top_p but deliberately LOWER --temperature to 0.6 — the
#   production setting the sweep below was measured with (clients still
#   override per-request).
#   Add --no-thinking below if you want faster, non-reasoning replies.
#
# NOTE: the LM Studio hub path (~/.lmstudio/hub/models/qwen/qwen3.6-27b) is
# a catalog stub with no weights; the runnable MLX weights are the path below.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-MTP-bf16}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""
# ---------------------------------------------------------------------------
# --draft-block-size 1 (NOT the head-config default 3) — block-size sweep,
# 2026-06-10, M5 Max 128GB, Qwen3.6-27B-8bit + MTP head bf16, production
# sampling temp=0.6/top_k=20/top_p=0.95/rep_pen=1.1, 500 gen tokens,
# 2 runs averaged, decode tps:
#
#   block   153-tok prompt              2541-tok prompt
#   no-mtp  17.3 tps (1.00x)            17.0 tps (1.00x)
#   1       21.6 tps 1.25x acc .76/1    23.3 tps 1.37x acc .81/1
#   2       19.9 tps 1.15x acc 1.22/2   21.2 tps 1.24x acc 1.42/2
#   3       18.7 tps 1.08x acc 1.59/3   19.5 tps 1.14x acc 1.81/3
#
# Block 1 wins at BOTH prompt sizes and the ordering is strictly monotonic
# (1 > 2 > 3): deeper blocks do raise tokens-accepted-per-round (1.81/3 at
# block 3), but each round's dense verify forward over k+1 tokens costs more
# than the extra acceptance buys back. Unlike the 35B A3B MoE (where block 1
# was a 1.019x wash — see start_qwen3.6_35b_a3b_mtp.sh), MTP genuinely PAYS
# on this dense target: plain decode is only ~17 tps, so the cheap 1-layer
# head + verify overhead is small relative to a target forward — same
# direction as dense gemma4 (60 -> 97.8 tps). Override per launch:
# ./start_qwen3.6_27b_mtp.sh --draft-block-size N (trailing "$@" wins over
# DRAFT_ARGS).
# ---------------------------------------------------------------------------
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 1)
# --thinking-budget/--repetition-penalty: anti-loop safety net (overridable via "$@")
# --gpu-keepalive intentionally NOT set here (user choice). NOTE: a '#' inside a
# backslash-continued argument list BREAKS the command — keep disabled flags in
# comments ABOVE the command, never inline.
mlx-soloheaven \
  --model "$MODEL_PATH" \
  --memory-budget-gb 20 \
  "${DRAFT_ARGS[@]}" \
  --prefill-step-size 2048 \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --thinking-budget 4096 \
  --repetition-penalty 1.1 \
  "$@"
