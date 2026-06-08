"""Logits processors for generation control."""

import mlx.core as mx


# Thinking-state tuple layout: ``(in_thinking, count, bare_idx, content_seen)``.
#   in_thinking  — currently inside a thinking block.
#   count        — tokens emitted in the CURRENT block (think_start counts as #1).
#   bare_idx     — how many leading tokens of the BARE thought opener have matched
#                  contiguously from generation start (gen-start lookbehind for the
#                  multi-token ``thought\n`` opener; see ``advance_think_state``).
#   content_seen — whether any token that is NOT part of an in-progress bare-opener
#                  match has been emitted yet. Mirrors ``ThinkingRouter._content_seen``
#                  (tool_parser FIX 4): once True, a BARE opener can no longer fire,
#                  so a literal ``thought\n`` mid-content/tool-args is NOT mis-read
#                  as a thinking opener. Always-True for chatml (starts in_thinking).
# The last two slots are dormant (and the tuple semantically a 2-tuple) unless a
# non-empty ``bare_open_tokens`` is threaded into ``advance_think_state``; with
# bare detection off the behaviour is byte-identical to the original 2-tuple form.


def initial_think_state(model_family: str = "chatml") -> tuple[bool, int, int, bool]:
    """Initial ``(in_thinking, count, bare_idx, content_seen)`` for a fresh
    generation.

    ChatML starts INSIDE the thinking block (``in_thinking=True``); Gemma 4
    starts outside it. Matches ``ThinkingBudgetProcessor.__init__``. ``bare_idx``
    starts at 0; ``content_seen`` starts False for gemma4 (the only family that
    can see a bare opener at gen-start) and True for chatml (already in thinking
    — no gen-start bare detection applies).
    """
    in_thinking = model_family != "gemma4"
    return (in_thinking, 0, 0, in_thinking)


def advance_think_state(
    state: tuple[bool, int, int, bool],
    tok: int,
    *,
    think_start_token: int,
    think_end_token: int,
    bare_open_tokens: tuple[int, ...] | list[int] | None = None,
) -> tuple[bool, int, int, bool]:
    """Advance the thinking state by one EMITTED token ``tok``.

    Mirrors one ``ThinkingBudgetProcessor.__call__`` step (the bookkeeping part,
    not the logit mutation): think_start opens thinking and counts as token #1,
    think_end closes it, every other in-thinking token increments the count.
    Pure and incremental — O(1) — so it can be folded over the emitted history
    without re-scanning, then branched per draft position cheaply.

    BARE-OPENER DETECTION (Option A, gemma4 sliding-window fix): past the 1024
    sliding window the ``<|channel>`` think_start prime falls out of the window
    and the model emits a BARE ``thought\\n`` opener with NO ``<|channel>`` token,
    so the token-id based open never fires and the budget never caps thinking.
    When ``bare_open_tokens`` (the token-id sequence of ``"thought\\n"``) is
    supplied, this matches that sequence token-by-token at GENERATION START ONLY
    (``not in_thinking and not content_seen``) via the ``bare_idx`` lookbehind;
    on a full contiguous match thinking opens (``count`` = matched length, so the
    opener tokens count toward the budget exactly like the ``<|channel>`` path).
    Mirrors ``ThinkingRouter`` FIX 4: a literal ``thought\\n`` appearing AFTER any
    content (``content_seen``) is plain content and never opens thinking. With
    ``bare_open_tokens`` None/empty, the bare path is inert and this is identical
    to the original ``<|channel>``-only behaviour.
    """
    in_thinking, count, bare_idx, content_seen = state

    # Full ``<|channel>`` opener (short context) — unchanged, takes precedence.
    if not in_thinking and tok == think_start_token:
        return (True, 1, 0, True)

    if in_thinking:
        if tok == think_end_token:
            # Block closed; a later think_start re-opens (no gen-start bare
            # opener after content — content_seen latched True).
            return (False, 0, 0, True)
        return (True, count + 1, 0, True)

    # Not in thinking. Try the gen-start-only bare ``thought\n`` opener.
    if bare_open_tokens and not content_seen:
        if tok == bare_open_tokens[bare_idx]:
            new_idx = bare_idx + 1
            if new_idx >= len(bare_open_tokens):
                # Full bare opener matched: open thinking. The matched opener
                # tokens count as the first ``len`` thinking tokens (parity with
                # the <|channel> path counting the start token as #1).
                return (True, new_idx, 0, True)
            # Partial match still at gen-start: stay outside thinking, keep
            # content_seen False so the next opener token can extend the match.
            return (False, 0, new_idx, False)
        # Mismatch: the gen-start window is broken; this is real content, so the
        # bare opener can never fire again (latch content_seen True).
        return (False, 0, 0, True)

    # No bare detection (off, or content already seen): plain content, latch.
    return (in_thinking, count, 0, True)


def force_end_from_state(state: tuple[bool, int, int, bool], budget: int) -> bool:
    """Whether the next sampled token must be forced to ``think_end`` given the
    thinking state DERIVED from the already-emitted history. ``budget <= 0`` →
    never (no-op)."""
    if budget <= 0:
        return False
    in_thinking, count = state[0], state[1]
    return in_thinking and count >= budget


def should_force_think_end(
    history: list[int],
    *,
    budget: int,
    think_start_token: int,
    think_end_token: int,
    model_family: str = "chatml",
    bare_open_tokens: tuple[int, ...] | list[int] | None = None,
) -> bool:
    """History-derived thinking-budget enforcement decision (pure, stateless).

    Given the list of tokens ACTUALLY EMITTED so far (``history``), decide
    whether the NEXT sampled token must be forced to ``think_end_token`` because
    the thinking budget has been reached and the model is still inside a
    thinking block.

    This is the stateless counterpart of ``ThinkingBudgetProcessor``: instead
    of mutating a per-call counter (which over-counts in the MTP speculative
    verify, where the sampler runs per draft position but only ``accepted+1``
    tokens are emitted), it RE-DERIVES the thinking state from the emitted
    history every call. The caller is responsible for passing the exact
    per-position cumulative history so the decision fires at the right position.

    Semantics (match ``ThinkingBudgetProcessor``):

    * ``budget <= 0`` or no ``think_end_token`` (< 0) → never force (no-op).
    * ChatML (``model_family != "gemma4"``): generation begins INSIDE the
      thinking block, so ``in_thinking`` starts True; it ends at the first
      ``think_end_token``. A later ``think_start_token`` re-opens it.
    * Gemma 4: generation begins OUTSIDE thinking; ``in_thinking`` flips True
      after a ``think_start_token`` (the ``<|channel>`` opener) is emitted OR —
      when ``bare_open_tokens`` is supplied — after the BARE ``thought\\n`` opener
      token sequence is emitted at GENERATION START (the long-context sliding-
      window variant with no ``<|channel>`` token; see ``advance_think_state``),
      and ends at the first following ``think_end_token``.
    * thinking_token_count = number of tokens emitted since the most recent
      ``think_start`` (ChatML: since block start) and before any ``think_end``.
    * Returns True iff currently in_thinking AND thinking_token_count >= budget
      (i.e. the budget is reached and the block has not closed).

    Once a ``think_end_token`` already appears as the LAST emitted token (the
    forced close was just emitted), in_thinking is False → returns False (no
    double-force).

    Parity note: for Gemma 4 this matches ``ThinkingBudgetProcessor`` EXACTLY —
    the ``<|channel>`` think_start anchors the count inside the GENERATED stream,
    so prompt tokens before it never count. For ChatML there is no anchoring
    think_start and the stateful processor counts the token being sampled (it
    increments on every call, including the prompt-seeing one), whereas this
    helper counts only emitted tokens; the two then agree to within one token at
    the budget boundary (the helper fires at most one token LATER — conservative,
    never early). Gemma 4 is the runaway target, so exact parity there is what
    matters.
    """
    if budget <= 0 or think_end_token < 0:
        return False

    state = initial_think_state(model_family)
    for tok in history:
        state = advance_think_state(
            state,
            tok,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            bare_open_tokens=bare_open_tokens,
        )
    return force_end_from_state(state, budget)


class ThinkingBudgetProcessor:
    """Forces thinking-end token after a budget is reached.

    Works as a logits_processor for mlx-vlm's stream_generate.

    - ChatML: generation starts inside <think> block → in_thinking=True.
    - Gemma 4: model generates <|channel>thought first → in_thinking starts False.

    PER-CYCLE semantics: every thinking block in a turn is budgeted. A
    ``think_end`` closes the current block and RESETS the per-cycle counter; a
    later ``think_start`` re-opens thinking and re-budgets the new block. This
    matches the MTP helper (``advance_think_state`` / ``force_end_from_state``)
    exactly, so the plain/non-drafter path and the speculative path are
    lockstep-identical even for multi-block turns (START...END...START...).

    There is intentionally NO permanent "done after first close" latch: that
    latch is what diverged from the MTP helper, which never stopped re-budgeting
    later blocks.
    """

    def __init__(
        self,
        budget: int,
        think_end_token: int,
        think_start_token: int,
        model_family: str = "chatml",
        bare_open_tokens: tuple[int, ...] | list[int] | None = None,
    ):
        self.budget = budget
        self.think_end_token = think_end_token
        self.think_start_token = think_start_token
        self.model_family = model_family
        # Token-id sequence of the BARE ``thought\n`` opener (gemma4 sliding-
        # window variant with no ``<|channel>`` token). Empty/None disables the
        # bare path entirely (identical to the original <|channel>-only behaviour).
        self.bare_open_tokens = tuple(bare_open_tokens) if bare_open_tokens else ()
        self.thinking_tokens = 0
        self.in_thinking = model_family != "gemma4"
        # Gen-start lookbehind state for the multi-token bare opener (mirrors
        # ``advance_think_state``). ``_content_seen`` latches True once any token
        # that is not part of an in-progress bare match is emitted, after which
        # the bare opener can no longer fire (ThinkingRouter FIX 4 parity).
        self._bare_idx = 0
        self._content_seen = self.in_thinking
        # PROMPT/GENERATION DOMAIN SPLIT (codex HIGH fix): mlx-vlm/mlx-lm call
        # this processor with the RUNNING ``tokens`` array, and the prefill step
        # folds the prompt tail (the chunked-prefill last token, or the WHOLE
        # prompt when unchunked) into ``tokens`` BEFORE running the processor —
        # so the first ``__call__`` sees prompt token(s) as ``last_token``. The
        # bare ``thought\n`` opener is a GENERATION-START phenomenon: a prompt
        # token mismatching the bare opener would latch ``_content_seen`` and the
        # GENERATED bare opener could then never fire (the long-context runaway).
        # ``_prompt_len`` captures the prompt boundary on the FIRST call:
        # ``tokens.size`` then == the count of prompt tokens visible to the
        # processor; every later token (index >= _prompt_len) is GENERATED. Bare
        # matching is gated to generated tokens, and the bare sub-state is RESET
        # at the boundary (iff outside thinking) so a generated bare opener can
        # open even though prompt tokens preceded it. Full ``<|channel>``
        # open/close + in-thinking counting still run on prompt tokens (parity
        # with the MTP seed fold using ``bare_open_tokens=()``).
        self._prompt_len: int | None = None
        self._bare_boundary_reset_done = False

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item() if tokens.size > 0 else -1

        # Capture the prompt boundary on the first call; reset the bare
        # sub-state at the generation boundary (first GENERATED token) so the
        # prompt tail never latches the bare matcher (see __init__ note).
        if self._prompt_len is None:
            self._prompt_len = int(tokens.size)
        # ``last_token`` is GENERATED iff its index (tokens.size - 1) is at or
        # past the prompt boundary, i.e. tokens.size > _prompt_len.
        last_is_generated = int(tokens.size) > self._prompt_len
        if last_is_generated and not self._bare_boundary_reset_done:
            self._bare_boundary_reset_done = True
            if self.model_family == "gemma4" and not self.in_thinking:
                # Generation boundary: re-arm the bare matcher. ``in_thinking``
                # already reflects any full ``<|channel>`` markers folded over
                # the prompt; only re-arm the bare gen-start sub-state.
                self._bare_idx = 0
                self._content_seen = False

        if self.in_thinking and last_token == self.think_end_token:
            # Close the current block and reset the per-cycle counter; do NOT
            # latch — a later think_start re-opens and re-budgets.
            self.in_thinking = False
            self.thinking_tokens = 0
            self._bare_idx = 0
            self._content_seen = True
            return logits

        if not self.in_thinking and last_token == self.think_start_token:
            # Re-open thinking; the start token counts as token #1 of the block.
            self.in_thinking = True
            self.thinking_tokens = 0
            self._bare_idx = 0
            self._content_seen = True

        # BARE ``thought\n`` opener (gemma4 long-context): match the token-id
        # sequence at GENERATION START ONLY (not in_thinking, no content seen,
        # and only over GENERATED tokens — never the prompt tail, which would
        # otherwise latch ``_content_seen`` and defeat the opener; see __init__).
        if (
            last_is_generated
            and not self.in_thinking
            and self.bare_open_tokens
            and not self._content_seen
            and last_token != -1
        ):
            if last_token == self.bare_open_tokens[self._bare_idx]:
                self._bare_idx += 1
                if self._bare_idx >= len(self.bare_open_tokens):
                    # Full bare opener matched: open thinking; the matched opener
                    # tokens count as the first thinking tokens of the block.
                    self.in_thinking = True
                    self.thinking_tokens = len(self.bare_open_tokens)
                    self._bare_idx = 0
                    self._content_seen = True
                    if self.thinking_tokens >= self.budget:
                        logits[:, self.think_end_token] = 1e9
                    return logits
                # Partial match still at gen-start: nothing emitted as content,
                # keep matching on the next call.
                return logits
            # Mismatch: real content at gen-start; the bare opener can never fire.
            self._bare_idx = 0
            self._content_seen = True

        if self.in_thinking:
            self.thinking_tokens += 1
            if self.thinking_tokens >= self.budget:
                logits[:, self.think_end_token] = 1e9

        return logits


class RepetitionPenaltyProcessor:
    """
    Logits processor that applies repetition penalty to previously generated tokens.

    penalty > 1.0 discourages repetition, < 1.0 encourages it, 1.0 = no effect.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size == 0 or self.penalty == 1.0:
            return logits
        # Get unique token IDs from generated sequence
        selected_logits = logits[:, tokens]
        # Apply penalty: divide positive logits, multiply negative logits
        selected_logits = mx.where(
            selected_logits > 0,
            selected_logits / self.penalty,
            selected_logits * self.penalty,
        )
        logits[:, tokens] = selected_logits
        return logits
