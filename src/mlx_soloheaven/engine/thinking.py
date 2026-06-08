"""Logits processors for generation control."""

import mlx.core as mx


def initial_think_state(model_family: str = "chatml") -> tuple[bool, int]:
    """Initial ``(in_thinking, thinking_count)`` for a fresh generation.

    ChatML starts INSIDE the thinking block (``in_thinking=True``); Gemma 4
    starts outside it. Matches ``ThinkingBudgetProcessor.__init__``.
    """
    return (model_family != "gemma4", 0)


def advance_think_state(
    state: tuple[bool, int],
    tok: int,
    *,
    think_start_token: int,
    think_end_token: int,
) -> tuple[bool, int]:
    """Advance ``(in_thinking, thinking_count)`` by one EMITTED token ``tok``.

    Mirrors one ``ThinkingBudgetProcessor.__call__`` step (the bookkeeping part,
    not the logit mutation): think_start opens thinking and counts as token #1,
    think_end closes it, every other in-thinking token increments the count.
    Pure and incremental — O(1) — so it can be folded over the emitted history
    without re-scanning, then branched per draft position cheaply.
    """
    in_thinking, count = state
    if not in_thinking and tok == think_start_token:
        return (True, 1)
    if in_thinking:
        if tok == think_end_token:
            return (False, 0)
        return (True, count + 1)
    return (in_thinking, count)


def force_end_from_state(state: tuple[bool, int], budget: int) -> bool:
    """Whether the next sampled token must be forced to ``think_end`` given the
    thinking state DERIVED from the already-emitted history. ``budget <= 0`` →
    never (no-op)."""
    if budget <= 0:
        return False
    in_thinking, count = state
    return in_thinking and count >= budget


def should_force_think_end(
    history: list[int],
    *,
    budget: int,
    think_start_token: int,
    think_end_token: int,
    model_family: str = "chatml",
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
      only after a ``think_start_token`` is emitted, and ends at the first
      following ``think_end_token``.
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
    ):
        self.budget = budget
        self.think_end_token = think_end_token
        self.think_start_token = think_start_token
        self.model_family = model_family
        self.thinking_tokens = 0
        self.in_thinking = model_family != "gemma4"

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item() if tokens.size > 0 else -1

        if self.in_thinking and last_token == self.think_end_token:
            # Close the current block and reset the per-cycle counter; do NOT
            # latch — a later think_start re-opens and re-budgets.
            self.in_thinking = False
            self.thinking_tokens = 0
            return logits

        if not self.in_thinking and last_token == self.think_start_token:
            # Re-open thinking; the start token counts as token #1 of the block.
            self.in_thinking = True
            self.thinking_tokens = 0

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
