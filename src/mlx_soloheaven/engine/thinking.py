"""Thinking budget processor for models with thinking/reasoning modes."""

import mlx.core as mx


class ThinkingBudgetProcessor:
    """
    Logits processor that forces </think> after a token budget is reached.

    Designed for models that generate reasoning inside <think>...</think> blocks.
    Once the budget is exceeded, sets the </think> token logit to a very high value
    to force the model to end its thinking phase.
    """

    def __init__(self, budget: int, think_end_token: int, think_start_token: int):
        self.budget = budget
        self.think_end_token = think_end_token
        self.think_start_token = think_start_token
        self.thinking_tokens = 0
        self.in_thinking = False
        self.thinking_done = False

    def reset(self):
        self.thinking_tokens = 0
        self.in_thinking = True  # generation starts inside <think> block
        self.thinking_done = False

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.thinking_done:
            return logits

        last_token = tokens[-1].item() if tokens.size > 0 else -1

        if last_token == self.think_end_token:
            self.thinking_done = True
            return logits

        if self.in_thinking:
            self.thinking_tokens += 1
            if self.thinking_tokens >= self.budget:
                # Force </think> by boosting its logit
                logits[:, self.think_end_token] = 1e9

        return logits
