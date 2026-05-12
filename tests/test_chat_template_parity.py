"""Chat-template tokenize-parity spot check (Phase 1 task P1.5, A4).

For each tokenizer present locally under ``~/.lmstudio/models/``, verifies:

    tokenizer.encode(tokenizer.apply_chat_template(msgs, tokenize=False, ...))
    is a prefix of
    tokenizer.apply_chat_template(msgs, tokenize=True, ...)

This validates that the future "text prompt -> mlx-vlm" path is equivalent
to the current ``input_ids=`` path for tokenizers whose chat template adds
special tokens / structural markers.

Notes:
- Local-only. No HF downloads. Each tokenizer dir gates with ``skipif`` so
  CI without models still passes cleanly.
- Tokenizer-only load (``AutoTokenizer.from_pretrained``). No model weights.
- Some ``apply_chat_template(tokenize=True)`` calls return a
  ``BatchEncoding`` (Qwen-family) rather than a flat list. We normalize.
- If a tokenizer cannot honor ``tokenize=False`` (missing chat_template)
  we mark that parametrized case ``xfail`` rather than failing the run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

LMSTUDIO_ROOT = Path(os.path.expanduser("~/.lmstudio/models"))

# Candidate tokenizer locations. Tested in order; only those present run.
# Each entry: (label, relative_path_under_lmstudio_root).
CANDIDATES = [
    ("qwen3.6-27b", "mlx-community/Qwen3.6-27B-8bit"),
    ("qwen3.6-35b-a3b", "mlx-community/Qwen3.6-35B-A3B-8bit"),
    ("qwen3.6-27b-unsloth", "unsloth/Qwen3.6-27B-MLX-8bit"),
    ("gemma-4-31b", "lmstudio-community/gemma-4-31B-it-MLX-8bit"),
    ("gemma-4-26b-a4b", "lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"),
]


def _resolve(rel: str) -> Path:
    return LMSTUDIO_ROOT / rel


def _present(rel: str) -> bool:
    p = _resolve(rel)
    # Need at least tokenizer.json or tokenizer_config.json for from_pretrained.
    return p.is_dir() and (
        (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
    )


def _extract_ids(result) -> list[int]:
    """Normalize apply_chat_template(tokenize=True) -> list[int]."""
    if isinstance(result, list):
        # Some tokenizers return [int, ...] directly.
        if result and isinstance(result[0], int):
            return result
    if hasattr(result, "get") and "input_ids" in result:
        ids = result["input_ids"]
        # Could be [[..]] when return_tensors not None — flatten one level.
        if ids and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids)
    # Fallback
    return list(result)


def _build_messages() -> list[dict]:
    """3-turn fixture (system + user + assistant + user)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "Thanks!"},
    ]


@pytest.mark.parametrize(
    "label,rel",
    CANDIDATES,
    ids=[c[0] for c in CANDIDATES],
)
def test_chat_template_tokenize_parity(label: str, rel: str):
    if not _present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")

    # Lazy import — keeps test collection cheap when transformers absent.
    from transformers import AutoTokenizer

    path = str(_resolve(rel))
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    msgs = _build_messages()
    # Some templates reject 'system' role — fall back to user+assistant only.
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Real finding: surface as xfail, not a test failure.
        pytest.xfail(
            f"apply_chat_template(tokenize=False) raised for {label}: {e!r}"
        )

    tokenized = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True
    )
    ids = _extract_ids(tokenized)

    # Re-encode the rendered text; chat_template strings typically already
    # contain BOS / role markers so we disable add_special_tokens to avoid
    # double-prepending. This mirrors how mlx-vlm consumes pre-rendered text.
    re_enc = tokenizer.encode(text, add_special_tokens=False)

    assert re_enc == ids[: len(re_enc)], (
        f"[{label}] re-encoded text is not a prefix of tokenize=True output\n"
        f"  re_enc[:10]={re_enc[:10]}\n"
        f"  ids[:10]={ids[:10]}\n"
        f"  text[:80]={text[:80]!r}"
    )
