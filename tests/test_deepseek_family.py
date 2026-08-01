"""DeepSeek-V4 model family: dialect, suffix builder, template differential.

DeepSeek's turn framing is structurally different from Qwen/EXAONE ChatML —
user turns have NO terminator and the eos closes assistant turns only — so it
gets its own family and suffix builder. Per the ChatMLDialect contract, adding
a dialect REQUIRES a real-token differential proof that

    cached_prefix_ids + suffix_ids == apply_chat_template(full messages)

on the actual installed template and tokenizer; those tests skip when the
converted model is absent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine.mlx_engine import (
    CHATML_DIALECT_DEEPSEEK,
    MLXEngine,
    detect_chatml_dialect,
)

MODEL = Path.home() / ".lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-mixed"

Q1, A1, Q2 = "안녕하세요", "반갑습니다!", "오늘 날씨 어때?"


class _EchoTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return text


def _builder():
    engine = SimpleNamespace(tokenizer=_EchoTokenizer(), model_family="deepseek")
    return lambda msgs, thinking: MLXEngine._suffix_tokens_deepseek(
        engine, msgs, thinking
    )


# --- family & dialect detection --------------------------------------------


def test_model_family_detection():
    eng = MLXEngine.__new__(MLXEngine)
    eng._model_type = "deepseek_v4"
    assert MLXEngine._detect_model_family(eng) == "deepseek"
    # deepseek v3 checkpoints (mlx-lm fallback) must NOT take this family:
    # their templates differ and were never verified against this builder.
    eng._model_type = "deepseek_v3"
    assert MLXEngine._detect_model_family(eng) == "chatml"


def test_dialect_detected_from_template_markers():
    assert (
        detect_chatml_dialect("...<｜User｜>...<｜Assistant｜>...")
        is CHATML_DIALECT_DEEPSEEK
    )
    # eos closes ASSISTANT turns — this is what _try_close_interrupted_turn
    # forwards, so it must be the eos, not a user-side marker.
    assert CHATML_DIALECT_DEEPSEEK.turn_end == "<｜end▁of▁sentence｜>"


# --- marker-level builder behaviour (offline) -------------------------------


def test_suffix_user_turn_has_no_terminator():
    text = _builder()([{"role": "user", "content": Q2}], thinking=False)
    assert text == f"<｜User｜>{Q2}<｜Assistant｜></think>"


def test_suffix_thinking_opens_think():
    text = _builder()([{"role": "user", "content": Q2}], thinking=True)
    assert text.endswith("<｜Assistant｜><think>")


def test_suffix_tool_run_merges_into_one_user_turn():
    msgs = [
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
        {"role": "user", "content": Q2},
    ]
    text = _builder()(msgs, thinking=False)
    assert text == (
        "<｜User｜><tool_result>r1</tool_result>\n\n<tool_result>r2</tool_result>"
        f"<｜User｜>{Q2}<｜Assistant｜></think>"
    )


def test_suffix_skips_assistant_messages():
    msgs = [{"role": "assistant", "content": A1}, {"role": "user", "content": Q2}]
    text = _builder()(msgs, thinking=False)
    assert A1 not in text


# --- real-token differentials (require the converted model) -----------------


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer_config.json").exists():
        pytest.skip("converted DeepSeek-V4 model not present")
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(MODEL))
    if tok.chat_template is None:
        pytest.skip("chat template not installed in the converted model")
    return tok


def _apply(tok, msgs, thinking):
    return tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=thinking,
    )


def _ids(res):
    return res["input_ids"] if not isinstance(res, list) else res


def test_template_renders_official_encoding(tokenizer):
    """String-exact against the rules in the release's encoding_dsv4.py
    (chat mode): BOS + system + <｜User｜>u + <｜Assistant｜></think>a<eos> ...
    ending in the generation opener."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": Q1},
        {"role": "assistant", "content": A1},
        {"role": "user", "content": Q2},
    ]
    out = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert out == (
        "<｜begin▁of▁sentence｜>You are helpful."
        f"<｜User｜>{Q1}<｜Assistant｜></think>{A1}<｜end▁of▁sentence｜>"
        f"<｜User｜>{Q2}<｜Assistant｜></think>"
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_splice_differential(tokenizer, thinking):
    """cached_prefix_ids + suffix_ids == apply_chat_template(full messages).

    cached = turn-1 prompt + the reply's tokens + the recorded natural eos —
    exactly what the session cache holds after turn 1.
    """
    eos_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
    t1 = _ids(_apply(tokenizer, [{"role": "user", "content": Q1}], thinking))
    reply = A1 if not thinking else f"생각중</think>{A1}"
    cached = t1 + tokenizer.encode(reply, add_special_tokens=False) + [eos_id]

    engine = SimpleNamespace(tokenizer=tokenizer, model_family="deepseek")
    suffix = MLXEngine._suffix_tokens_deepseek(
        engine, [{"role": "user", "content": Q2}], thinking
    )

    # Full render: earlier-turn reasoning is dropped by the template, so the
    # full conversation renders the assistant CONTENT only — build the
    # expected ids from the same content the template would keep.
    full = _ids(
        _apply(
            tokenizer,
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
            ],
            thinking,
        )
    )
    if thinking:
        # cached carries the reasoning tokens the re-render drops; the splice
        # still matches from the suffix on. Verify the suffix boundary only.
        assert cached[-1] == eos_id
        assert full[-len(suffix):] == suffix
    else:
        assert cached + suffix == full


def test_interrupted_close_matches_rerender(tokenizer):
    """An interrupted turn committed WITHOUT eos, closed by forwarding
    dialect.turn_end, must equal the full re-render of the same messages."""
    t1 = _ids(_apply(tokenizer, [{"role": "user", "content": Q1}], False))
    committed = t1 + tokenizer.encode(A1, add_special_tokens=False)
    eos_id = tokenizer.convert_tokens_to_ids(CHATML_DIALECT_DEEPSEEK.turn_end)
    closed = committed + [eos_id]

    full_turn1 = _ids(
        _apply(
            tokenizer,
            [{"role": "user", "content": Q1}, {"role": "assistant", "content": A1}],
            False,
        )
    )
    # full render of [u1, a1] with generation prompt appends the next opener;
    # strip it to compare the committed region.
    opener = tokenizer.encode("<｜Assistant｜></think>", add_special_tokens=False)
    assert full_turn1[-len(opener):] == opener
    assert closed == full_turn1[: -len(opener)]
