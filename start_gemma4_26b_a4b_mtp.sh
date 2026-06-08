#!/bin/bash
# Start SoloHeaven with Gemma 4 26B-A4B (MoE, ~4B active) 8-bit target + 8-bit MTP drafter.
# Target slug:  lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit
# Drafter slug: guardiangate1775/gemma-4-26B-A4B-it-assistant-8bit (8-bit, matches the
#   8-bit target's quantization; verified: loads, output byte-identical to plain greedy,
#   pre-wrap acceptance ~2.6, slightly faster than the bf16 drafter). Override with
#   DRAFT_PATH=... to use the bf16 drafter (mlx-community/gemma-4-26B-A4B-it-assistant-bf16).
# Post-wrap behaviour: the B4 RoPE-frame fix (true offset + absolute-position drafter
#   mask) restores the drafter's post-wrap acceptance (~1.1-1.2), so MTP stays ~2x past
#   the 1024 sliding-window ring wrap. The old wrap-gate (drafter -> plain decode after
#   wrap) is now an opt-in fallback only: SOLOHEAVEN_MTP_WRAP_GATE=1. See _rotating_wrapped()
#   and the B4 patch in mlx_engine._install_mtp_wrap_patches().
# To disable MTP entirely, comment the active DRAFT_ARGS line below and uncomment the empty one.
#
# OpenCode / OpenAI-endpoint fixes baked in (2026-06-08):
#  - --repetition-penalty 1.1: gemma4 in long/poisoned contexts degenerated into a
#    non-terminating repeated-paragraph loop. The penalty is now wired into the MTP
#    speculative path (was previously bypassed) and acts as a safety net. The primary
#    defence is context hygiene (see below); raise toward 1.2 if loops persist, lower
#    toward 1.05 if it hurts code/list output. CLI flag is needed because cli.py's
#    default (REPETITION_PENALTY env) is 1.0 and overrides the 1.05 dataclass default.
#  - Code fixes (auto-applied from source, no flag): (1) tool-call object-array parser
#    (todowrite `todos` no longer shattered into strings -> no invalid-tool retry loop);
#    (2) removed the _HOT_PATH_FAST MTP fast-path that produced 2-token early-EOS on
#    small max_tokens; (3) gemma4 <|channel>thought...<channel|> stripped from the
#    OpenAI endpoint output + history (was leaking raw markers -> context poisoning).
#  - NOTE: an already-running OpenCode conversation whose history is poisoned will keep
#    repeating; start a FRESH session after this server restart.
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"
DRAFT_PATH="${DRAFT_PATH:-$HOME/.lmstudio/models/guardiangate1775/gemma-4-26B-A4B-it-assistant-8bit}"

cd "$(dirname "$0")"
source .venv/bin/activate
export SOLOHEAVEN_MODELS=""

# DRAFT_ARGS=()
DRAFT_ARGS=(--draft-model "$DRAFT_PATH" --draft-block-size 3)

mlx-soloheaven \
  --model "$MODEL_PATH" \
  "${DRAFT_ARGS[@]}" \
  --memory-budget-gb 20 \
  --gpu-keepalive \
  --repetition-penalty 1.1 \
  "$@"
