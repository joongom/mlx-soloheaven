"""Logits processors for generation control."""

import mlx.core as mx


class ThinkingBudgetProcessor:
    """Forces thinking-end token after a budget is reached.

    Works as a logits_processor for mlx-vlm's stream_generate.

    - ChatML: generation starts inside <think> block → in_thinking=True.
    - Gemma 4: model generates <|channel>thought first → in_thinking starts False.
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
        self.thinking_done = False

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.thinking_done:
            return logits

        last_token = tokens[-1].item() if tokens.size > 0 else -1

        if last_token == self.think_end_token:
            self.in_thinking = False
            self.thinking_done = True
            return logits

        if not self.in_thinking and last_token == self.think_start_token:
            self.in_thinking = True

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
