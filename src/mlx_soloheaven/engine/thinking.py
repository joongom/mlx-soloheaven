"""Logits processors for generation control."""

import mlx.core as mx


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


class ThinkingBudgetProcessor:
    """
    Logits processor that forces thinking-end token after a budget is reached.

    Supports both ChatML (<think>...</think>) and Gemma 4 (<|channel>thought...<channel|>).

    - ChatML: generation starts inside <think> block → in_thinking=True from start.
    - Gemma 4: generation starts at <|turn>model\n → model first generates
      <|channel>thought\n before actual thinking. in_thinking starts False,
      becomes True when think_start_token (<|channel>) is seen.
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
        self.in_thinking = False
        self.thinking_done = False

    def reset(self):
        self.thinking_tokens = 0
        # ChatML: suffix includes <think>\n, so generation starts inside thinking.
        # Gemma 4: suffix is <|turn>model\n, model generates <|channel> first.
        self.in_thinking = self.model_family != "gemma4"
        self.thinking_done = False

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.thinking_done:
            return logits

        last_token = tokens[-1].item() if tokens.size > 0 else -1

        if last_token == self.think_end_token:
            self.in_thinking = False
            self.thinking_done = True
            return logits

        # Gemma 4: enter thinking when <|channel> (think_start_token) is seen
        if not self.in_thinking and last_token == self.think_start_token:
            self.in_thinking = True

        if self.in_thinking:
            self.thinking_tokens += 1
            if self.thinking_tokens >= self.budget:
                # Force thinking-end token by boosting its logit
                logits[:, self.think_end_token] = 1e9

        return logits
