"""U18 — gemma4 cache-HIT suffix vs the recorded end-of-turn token.

THE BUG (verified token-exact against the real gemma-4-31B tokenizer):
``_suffix_tokens_gemma4`` leads with the ``<turn|>`` closer — written for
the interrupted-turn contract (C1: gemma4 turns are committed unterminated,
_try_close_interrupted_turn returns NOT_REQUIRED because the next suffix
supplies the closer). But on the DEFAULT mlx-lm backend a NATURAL EOS
records ``<turn|>`` (id 106, in gemma4's generation_config eos_token_id)
into cache_state.token_ids: mlx-lm stream_generate's terminal frame carries
token=eos and the engine's stream loop records every token-bearing frame,
while generate_step's lookahead already forwarded the token through the KV
(offset == len(token_ids) holds). The next HIT then spliced the closer a
SECOND time:

    stored  [..., '4'(236812), '.'(236761), <turn|>(106)]
    suffix  [<turn|>(106), '\\n'(107), <|turn>(105), 'user', ...]
    spliced [..., 106, 106, 107, ...]   <- doubled end-of-turn
    full    [..., 106, 107, 105, ...]   <- apply_chat_template: single 106

FIX: ``_dedupe_gemma4_turn_closer`` (called from the HIT path) drops the
suffix's duplicate leading closer when the stored tail already IS
``<turn|>``. Dropping suffix[0] is token-exact equivalent to encoding the
suffix without the closer (``<turn|>`` is a special token — the tokenizer
splits at it); the real-tokenizer tests below prove the spliced ids equal
apply_chat_template over the full conversation, and that the
interrupted-turn shape (no closer recorded) keeps the leading closer.

Real-tokenizer tests gate on the local model dir (skip-if-missing) so CI
without ~/.lmstudio models still passes.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_qwen_mtp import (  # noqa: E402
    MockKV,
    _generate_engine,
    _scripted_lm_stream,
)

EOT = 106  # <turn|> — gemma4 end-of-turn / chat EOS

LMSTUDIO_ROOT = Path(os.path.expanduser("~/.lmstudio/models"))
GEMMA4_TOKENIZER_DIRS = [
    ("gemma-4-31b", "lmstudio-community/gemma-4-31B-it-MLX-8bit"),
    ("gemma-4-26b-a4b", "lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"),
]


def _tokenizer_present(rel: str) -> bool:
    p = LMSTUDIO_ROOT / rel
    return p.is_dir() and (
        (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
    )


# ---------------------------------------------------------------------------
# Wiring — the dedupe actually runs on the engine's HIT path (fake tokenizer)
# ---------------------------------------------------------------------------


def _gemma4_hit_engine(stored, suffix):
    """HIT-capable mlx-lm-path harness re-labeled gemma4, with a vocab that
    resolves <turn|> so _gemma4_turn_end_id detects it."""
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)
    eng.model_family = "gemma4"
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<turn|>": EOT},
        encode=lambda text, add_special_tokens=False: [],
    )
    return eng, cs, messages


def _drive(eng, messages, *, max_tokens=8):
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )


def test_hit_dedupes_leading_closer_after_natural_eos(monkeypatch):
    """Stored tail IS <turn|> (natural mlx-lm EOS recorded it): the HIT
    prefill must receive the suffix WITHOUT the duplicate leading closer."""
    stored = [1, 2, 3, 4, EOT]
    eng, cs, messages = _gemma4_hit_engine(stored, suffix=[EOT, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive(eng, messages)

    # The new-token prefill is exactly the deduped suffix — no doubled 106.
    assert prompts_seen[0] == [55, 66]
    # Recorded ids: stored + deduped suffix + generated (offset-consistent).
    assert cs.token_ids == stored + [55, 66] + [101, 102]


def test_hit_keeps_closer_after_interrupted_turn(monkeypatch):
    """Stored tail is NOT <turn|> (C1: interrupted turn committed
    unterminated): the suffix's leading closer is what closes the turn —
    it must be preserved (NOT_REQUIRED contract)."""
    stored = [1, 2, 3, 4, 5]
    eng, cs, messages = _gemma4_hit_engine(stored, suffix=[EOT, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive(eng, messages)

    assert prompts_seen[0] == [EOT, 55, 66]
    assert cs.token_ids == stored + [EOT, 55, 66] + [101, 102]


def test_dedupe_helper_edge_cases():
    """Direct helper contract: gemma4-only, tail+head must both be the
    detected closer, degraded vocab detection is a no-op (fail-open to the
    historical splice — never drops a token it cannot verify)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "gemma4"
    eng.tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<turn|>": EOT},
        encode=lambda text, add_special_tokens=False: [],
    )

    # Both sides are the closer -> deduped.
    assert eng._dedupe_gemma4_turn_closer([EOT, 7, 8], [1, 2, EOT]) == [7, 8]
    # Tail not the closer (interrupted) -> untouched.
    assert eng._dedupe_gemma4_turn_closer([EOT, 7, 8], [1, 2, 3]) == [EOT, 7, 8]
    # Suffix does not lead with the closer -> untouched.
    assert eng._dedupe_gemma4_turn_closer([7, 8], [1, 2, EOT]) == [7, 8]
    # Empty stored / empty suffix -> untouched.
    assert eng._dedupe_gemma4_turn_closer([EOT, 7], []) == [EOT, 7]
    assert eng._dedupe_gemma4_turn_closer([EOT, 7], None) == [EOT, 7]
    assert eng._dedupe_gemma4_turn_closer([], [EOT]) == []

    # Non-gemma4 families never touched (ChatML records <|im_end|> AND its
    # suffix starts with the user turn — no closer to dedupe).
    eng2 = MLXEngine.__new__(MLXEngine)
    eng2.model_family = "chatml"
    eng2.tokenizer = eng.tokenizer
    assert eng2._dedupe_gemma4_turn_closer([EOT, 7], [1, EOT]) == [EOT, 7]

    # <turn|> not in the vocab (detection returns -1) -> fail-open no-op.
    eng3 = MLXEngine.__new__(MLXEngine)
    eng3.model_family = "gemma4"
    eng3.tokenizer = SimpleNamespace(
        get_vocab=lambda: {},
        encode=lambda text, add_special_tokens=False: [],
    )
    assert eng3._dedupe_gemma4_turn_closer([EOT, 7], [1, EOT]) == [EOT, 7]


def test_turn_end_id_memoized():
    """_gemma4_turn_end_id builds the vocab dict at most once per engine."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    calls = {"n": 0}

    def _vocab():
        calls["n"] += 1
        return {"<turn|>": EOT}

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "gemma4"
    eng.tokenizer = SimpleNamespace(
        get_vocab=_vocab, encode=lambda t, add_special_tokens=False: [],
    )
    assert eng._gemma4_turn_end_id() == EOT
    assert eng._gemma4_turn_end_id() == EOT
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Token-exact — real gemma4 tokenizer (skip when the model dir is absent)
# ---------------------------------------------------------------------------


def _load_real(rel: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(LMSTUDIO_ROOT / rel), trust_remote_code=True
    )


def _ids(result) -> list[int]:
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _shell_with_real_tokenizer(tok):
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "gemma4"
    eng.tokenizer = tok
    return eng


Q1 = "What is 2 + 2?"
A1 = "The answer is 4."
Q2 = "And 3 + 3?"
A2 = "That makes 6."
Q3 = "Now multiply them."


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_natural_eos_hit_splice_is_token_exact(label, rel):
    """Turn 1 recorded per the mlx-lm contract (reply + EOS token recorded,
    lookahead-forwarded), then the ENGINE's real suffix builder + dedupe:
    the spliced ids must EQUAL apply_chat_template over the conversation."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell_with_real_tokenizer(tok)
    eot = eng._gemma4_turn_end_id()
    assert eot >= 0

    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    # mlx-lm recording contract: per-token frames for the reply, terminal
    # frame carries token=eos -> the EOS id IS recorded; generate_step's
    # lookahead already forwarded it, so no reconcile trim.
    stored = prompt1 + tok.encode(A1, add_special_tokens=False) + [eot]

    suffix = eng._suffix_tokens_gemma4(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    assert suffix[0] == eot  # builder still leads with the closer (C1)
    suffix = eng._dedupe_gemma4_turn_closer(suffix, stored)
    assert suffix[0] != eot  # the duplicate closer was dropped

    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    assert stored + suffix == full


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_undeduped_splice_doubles_the_closer(label, rel):
    """Regression documentation: WITHOUT the dedupe the splice diverges from
    apply_chat_template with a doubled back-to-back <turn|> (the U18 bug)."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell_with_real_tokenizer(tok)
    eot = eng._gemma4_turn_end_id()

    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    stored = prompt1 + tok.encode(A1, add_special_tokens=False) + [eot]
    raw_suffix = eng._suffix_tokens_gemma4(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    spliced = stored + raw_suffix
    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    assert spliced != full
    assert any(
        spliced[i] == eot and spliced[i + 1] == eot
        for i in range(len(spliced) - 1)
    )
    # And a single closer never appears back-to-back in the reference.
    assert not any(
        full[i] == eot and full[i + 1] == eot for i in range(len(full) - 1)
    )


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_interrupted_turn_splice_keeps_closer_token_exact(label, rel):
    """C1 shape: the turn was committed unterminated (no EOS recorded).
    The suffix's leading closer must survive the dedupe and the splice must
    equal apply_chat_template over the same conversation."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell_with_real_tokenizer(tok)
    eot = eng._gemma4_turn_end_id()

    partial = "The answer is"  # interrupted mid-sentence, no closer recorded
    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    stored = prompt1 + tok.encode(partial, add_special_tokens=False)
    suffix = eng._suffix_tokens_gemma4(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    suffix = eng._dedupe_gemma4_turn_closer(suffix, stored)
    assert suffix[0] == eot  # closer preserved — it closes the open turn

    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": partial},
                {"role": "user", "content": Q2},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    assert stored + suffix == full


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_three_turn_natural_eos_no_drift(label, rel):
    """Two natural stops in a row: the dedupe keeps the accumulated ids
    token-exact against the 3-turn retokenization (no per-turn drift)."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell_with_real_tokenizer(tok)
    eot = eng._gemma4_turn_end_id()

    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    stored = prompt1 + tok.encode(A1, add_special_tokens=False) + [eot]

    s2 = eng._dedupe_gemma4_turn_closer(
        eng._suffix_tokens_gemma4([{"role": "user", "content": Q2}], thinking=True),
        stored,
    )
    stored = stored + s2 + tok.encode(A2, add_special_tokens=False) + [eot]

    s3 = eng._dedupe_gemma4_turn_closer(
        eng._suffix_tokens_gemma4([{"role": "user", "content": Q3}], thinking=True),
        stored,
    )
    stored = stored + s3

    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
                {"role": "assistant", "content": A2},
                {"role": "user", "content": Q3},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    assert stored == full
