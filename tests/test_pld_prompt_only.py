"""Prompt-only PLD lookup tests (codex fix C — anti-repetition-loop).

Regression coverage for the confirmed repetition bug where ``--pld`` made
gemma4 degenerate into verbatim line loops. Root cause: PLD's lookup corpus
used to include *generated* tokens, so once a phrase P was emitted its k-gram
self-matched the earlier generated P and PLD re-drafted P, the verifier
accepted it under a peaked distribution, and the loop ran away.

Fix: ``PLDMatcher`` drafts ONLY from spans present in the PROMPT. Generated
tokens are still tracked (to form the tail query k-gram) but are never inserted
into the searchable index, so a generated-only phrase can never be drafted. The
copy-from-prompt benefit (RAG / code / tool-output spans) is preserved. The
change is lossless for accepted tokens — only which drafts are *attempted*
differs.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_soloheaven.engine.pld import PLDMatcher, pld_generate_step


# --------------------------------------------------------------------------
# (1) Anti-loop guard: a generated-only phrase is NEVER drafted.
# --------------------------------------------------------------------------

def test_generated_only_phrase_is_never_drafted():
    """Prompt has no internal repeats; we 'generate' a phrase twice. The
    second occurrence's tail must NOT self-match the first generated
    occurrence -> match() returns [] (no generated continuation drafted)."""
    matcher = PLDMatcher([100, 101, 102], max_k=3)  # prompt: distinct tokens
    # Simulate generation: 7,8,9 then 7,8 again. Tail (7,8) now repeats the
    # earlier generated (7,8) followed by 9.
    for tok in (7, 8, 9, 7, 8):
        matcher.append(tok)

    # The dangerous draft (old impl) would be [9]: copy the generated 9 after
    # the prior generated (7,8). Prompt-only lookup must refuse.
    assert matcher.match([7, 8], k=3, n=5) == []
    assert matcher.match([8, 9, 7, 8], k=3, n=5) == []
    # Even a single-token (k=1) fallback must not seed a generated draft.
    assert matcher.match([9], k=1, n=5) == []


def test_generated_tokens_not_inserted_into_index():
    """The searchable index must stay frozen at the prompt; appending
    generated tokens must not add new index entries."""
    matcher = PLDMatcher([1, 2, 3], max_k=3)
    before = {k: dict(matcher.index[k]) for k in range(1, matcher.max_k + 1)}
    for tok in (1, 2, 3, 1, 2, 3):  # generated tokens that repeat the prompt
        matcher.append(tok)
    after = {k: dict(matcher.index[k]) for k in range(1, matcher.max_k + 1)}
    assert before == after, "generated tokens leaked into the lookup index"
    assert matcher.prompt_len == 3


# --------------------------------------------------------------------------
# (2) Copy-from-prompt benefit preserved: a span present in the PROMPT is
#     still drafted.
# --------------------------------------------------------------------------

def test_prompt_span_is_still_drafted():
    """A k-gram that occurs in the prompt continues to be drafted with its
    prompt continuation."""
    # Prompt: '... 1 2 3 5 6 7' — the tail query (1,2,3) matches the prompt.
    matcher = PLDMatcher([1, 2, 3, 4, 1, 2, 3, 5, 6, 7], max_k=3)
    assert matcher.match([1, 2, 3], k=3, n=3) == [5, 6, 7]


def test_prompt_copy_survives_after_generation():
    """After generating tokens, a tail that matches the prompt still drafts
    the prompt continuation (the benefit is not lost by tracking output)."""
    matcher = PLDMatcher([42, 1, 2, 3, 99], max_k=3)  # prompt has '1 2 3 99'
    for tok in (7, 8, 1, 2, 3):  # generated; tail (1,2,3) matches prompt
        matcher.append(tok)
    # Most-recent prompt match of (1,2,3) ends at prompt index 3 -> next is 99.
    assert matcher.match([1, 2, 3], k=3, n=2) == [99]


def test_match_never_copies_past_prompt_boundary():
    """If a matched k-gram ends at the last prompt token, there is no prompt
    continuation to copy and the matcher must NOT spill into generated tokens."""
    matcher = PLDMatcher([5, 6, 7], max_k=3)  # (5,6,7) ends at last prompt tok
    for tok in (5, 6, 7, 8, 9):  # generated continuation 8,9
        matcher.append(tok)
    # Tail (5,6,7) matches the prompt at the boundary; nothing to copy.
    assert matcher.match([5, 6, 7], k=3, n=5) == []


# --------------------------------------------------------------------------
# (3) End-to-end: with a peaked (always-same-token) model, prompt-only PLD
#     does not regress the lossless output, and the matcher never self-feeds.
# --------------------------------------------------------------------------

class _MockCache:
    def __init__(self):
        self.offset = 0
        self.state = mx.array([0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset = max(0, self.offset - n)
        return n


class _ConstModel(nn.Module):
    """Always argmaxes to token 0 (maximally peaked distribution)."""

    def __init__(self, vocab: int = 16):
        super().__init__()
        self.vocab = vocab
        self.layers = [object()]

    def make_cache(self):
        return [_MockCache()]

    def __call__(self, x, cache=None):
        T = x.shape[1]
        row = mx.concatenate(
            [mx.array([[1e9]]), mx.zeros((1, self.vocab - 1))], axis=1
        )
        if cache is not None:
            cache[0].offset += T
        return mx.broadcast_to(row[:, None, :], (1, T, self.vocab))


def test_end_to_end_peaked_model_is_lossless():
    """The accepted-token stream must equal the model's greedy output even
    under prompt-only drafting (lossless): all zeros for the const model."""
    model = _ConstModel()
    prompt = mx.array([1, 2, 3, 0, 0, 0], dtype=mx.uint32)
    out = [
        t
        for t, _lp, _fd in pld_generate_step(
            prompt,
            model,
            num_draft_tokens=4,
            max_tokens=5,
            ngram_k=3,
            prompt_cache=model.make_cache(),
        )
    ]
    assert out == [0, 0, 0, 0, 0], f"expected all zeros, got {out}"
