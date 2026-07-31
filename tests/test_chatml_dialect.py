"""ChatML turn-marker dialects for the cache-HIT suffix splice.

The HIT path appends a hand-built suffix to the stored token_ids instead of
re-tokenizing. That is only sound if

    cached_prefix_ids + suffix_ids == apply_chat_template(full messages)

*exactly*, on the real installed template and tokenizer. The engine's builder
was hardcoded to Qwen's ``<|im_start|>``/``<|im_end|>``, so EXAONE — same
family by thinking/tool syntax, different turn framing — got Qwen markers
spliced into its KV and answered turn 2 with a bare ``<|im_end|>``. Nothing
raised; the tokens were simply wrong.

The marker-level tests below run offline. The differential tests need a real
tokenizer and skip when the model is absent, because a fabricated
encoded-string assertion is exactly the kind of proof this class of bug slips
through (see the note in ``_suffix_blocking_assistants``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine.mlx_engine import (
    CHATML_DIALECT_EXAONE,
    CHATML_DIALECT_QWEN,
    MLXEngine,
    detect_chatml_dialect,
)

EXAONE_MODEL = Path.home() / ".lmstudio/models/mlx-community/EXAONE-4.5-33B-8bit"

Q1, A1, Q2 = "안녕하세요", "반갑습니다!", "오늘 날씨 어때?"


def _builder(dialect, tokenizer):
    """A bare MLXEngine stand-in exposing only what the builder touches."""
    engine = SimpleNamespace(
        _chatml_dialect=dialect,
        tokenizer=tokenizer,
        model_family="chatml",
    )
    return lambda msgs, thinking: MLXEngine._suffix_tokens_chatml(
        engine, msgs, thinking
    )


class _EchoTokenizer:
    """Returns the string itself so marker tests can assert on text."""

    @staticmethod
    def encode(text, add_special_tokens=False):
        return text


# ---------------------------------------------------------------------------
# dialect detection
# ---------------------------------------------------------------------------

def test_detects_exaone_by_turn_marker():
    assert detect_chatml_dialect("... <|endofturn|> ...") is CHATML_DIALECT_EXAONE


def test_detects_qwen_by_turn_marker():
    assert detect_chatml_dialect("... <|im_start|>user ...") is CHATML_DIALECT_QWEN


def test_unknown_template_returns_none_rather_than_guessing():
    assert detect_chatml_dialect("{{ messages }}") is None
    assert detect_chatml_dialect(None) is None


def test_exaone_wins_over_qwen_when_template_mentions_both():
    # An EXAONE template that documents ChatML compatibility must not be
    # misread as Qwen — the turn framing is what decides.
    assert (
        detect_chatml_dialect("<|im_start|> ... <|endofturn|>")
        is CHATML_DIALECT_EXAONE
    )


def test_engine_defaults_to_qwen_before_load():
    from mlx_soloheaven.config import Config

    engine = MLXEngine(Config(model_path="/nonexistent"), execution_mode="main_thread")
    assert engine._chatml_dialect is CHATML_DIALECT_QWEN


# ---------------------------------------------------------------------------
# marker-level shape (offline)
# ---------------------------------------------------------------------------

def test_qwen_suffix_shape_is_unchanged():
    build = _builder(CHATML_DIALECT_QWEN, _EchoTokenizer)
    assert build([{"role": "user", "content": "hi"}], True) == (
        "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert build([{"role": "user", "content": "hi"}], False) == (
        "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )


def test_exaone_suffix_uses_its_own_markers():
    build = _builder(CHATML_DIALECT_EXAONE, _EchoTokenizer)
    assert build([{"role": "user", "content": "hi"}], True) == (
        "\n<|user|>\nhi<|endofturn|>\n<|assistant|>\n<think>\n"
    )
    # EXAONE primes a CLOSED think block when thinking is off, where Qwen
    # emits the bare role marker.
    assert build([{"role": "user", "content": "hi"}], False) == (
        "\n<|user|>\nhi<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\n"
    )


def test_exaone_groups_consecutive_tool_messages_into_one_run():
    build = _builder(CHATML_DIALECT_EXAONE, _EchoTokenizer)
    out = build(
        [{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}], False
    )
    # ONE <|tool|> opener, newline between results, ONE terminator.
    assert out.count("<|tool|>") == 1
    assert "<tool_result>a</tool_result>\n<tool_result>b</tool_result>" in out
    assert out.count("<|endofturn|>") == 1


def test_exaone_closes_a_tool_run_before_a_following_user_turn():
    build = _builder(CHATML_DIALECT_EXAONE, _EchoTokenizer)
    out = build(
        [{"role": "tool", "content": "a"}, {"role": "user", "content": "hi"}], False
    )
    assert out.startswith(
        "\n<|tool|>\n<tool_result>a</tool_result><|endofturn|>\n<|user|>\nhi"
    )


def test_qwen_tool_messages_stay_per_message():
    build = _builder(CHATML_DIALECT_QWEN, _EchoTokenizer)
    out = build(
        [{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}], False
    )
    assert out.count("<|im_start|>user") == 2  # historical behaviour, unchanged


def test_assistant_messages_are_never_spliced():
    for dialect in (CHATML_DIALECT_QWEN, CHATML_DIALECT_EXAONE):
        build = _builder(dialect, _EchoTokenizer)
        out = build([{"role": "assistant", "content": "nope"}], False)
        assert "nope" not in out


# ---------------------------------------------------------------------------
# real-token differential test — the actual soundness proof
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exaone_tokenizer():
    if not (EXAONE_MODEL / "tokenizer.json").exists():
        pytest.skip(f"{EXAONE_MODEL} not present")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(EXAONE_MODEL))


@pytest.mark.parametrize("thinking", [False, True])
def test_exaone_splice_is_token_exact_vs_apply_chat_template(
    exaone_tokenizer, thinking
):
    tok = exaone_tokenizer
    ids = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731

    # What the engine has cached after turn 1: the rendered turn-1 prompt plus
    # the generated reply plus the recorded stop token.
    cached = (
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + A1
        + "<|endofturn|>"
    )
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    # The cache is only reusable at all if it is a real token prefix.
    assert ids(full)[: len(ids(cached))] == ids(cached)

    suffix = _builder(CHATML_DIALECT_EXAONE, tok)(
        [{"role": "user", "content": Q2}], thinking
    )
    assert ids(cached) + suffix == ids(full)


def test_exaone_tool_splice_is_token_exact(exaone_tokenizer):
    tok = exaone_tokenizer
    ids = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731

    cached = (
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + A1
        + "<|endofturn|>"
    )
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "tool", "content": "sunny"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    suffix = _builder(CHATML_DIALECT_EXAONE, tok)(
        [{"role": "tool", "content": "sunny"}], False
    )
    assert ids(cached) + suffix == ids(full)


def test_qwen_markers_would_corrupt_exaone(exaone_tokenizer):
    """Pins the bug this dialect exists to prevent."""
    tok = exaone_tokenizer
    ids = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731

    cached = (
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + A1
        + "<|endofturn|>"
    )
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    wrong = _builder(CHATML_DIALECT_QWEN, tok)(
        [{"role": "user", "content": Q2}], False
    )
    assert ids(cached) + wrong != ids(full)
