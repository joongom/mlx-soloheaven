"""Post-generation cache_state.token_ids unification (Phase 1 task P1.2 / P1.9).

After one streaming generation pass, cache_state.token_ids must equal
prompt_token_ids + generated_token_ids on BOTH paths (mlx-vlm VLM path and
mlx-lm legacy path). Previously only the mlx-lm branch wrote this back; the
VLM path left token_ids at the prompt boundary, silently breaking
prefix-match for subsequent turns.

The test stubs out heavy MLXEngine dependencies and monkeypatches the
streaming backends so it can drive `_generate_locked` without loading a
real model.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine


# ---- Fakes / fixtures ----


@dataclass
class _FakeResp:
    text: str
    token: Optional[int]
    prompt_tps: float = 0.0
    generation_tps: float = 0.0


class _FakeTokenizer:
    """Minimal tokenizer that returns deterministic token IDs."""

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        # Map each char to its codepoint — stable, unique.
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        # Concatenate role + content; tokenize as codepoints.
        text = "".join(f"<{m.get('role','user')}>{m.get('content','')}" for m in messages)
        if not tokenize:
            return text
        return [ord(c) for c in text]


def _make_stub_engine(generated_ids: list[int], use_vlm: bool):
    """Build an MLXEngine instance without running __init__, then attach
    only the attributes / methods that `_generate_locked` actually touches."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.enable_thinking = False  # avoid building ThinkingBudgetProcessor
    eng.cfg.think_end_token = -1
    eng.cfg.pld_enabled = False
    eng.cfg.kv_bits = 0
    eng._sessions = {}
    eng._base_caches = {}
    eng._dirty_sessions = set()
    eng.tokenizer = _FakeTokenizer()
    eng._language_model = SimpleNamespace()
    eng._vlm_model = SimpleNamespace()
    eng._processor = SimpleNamespace(tokenizer=eng.tokenizer)
    eng._use_vlm = use_vlm
    eng.model_family = "chatml"
    eng.model_id = "stub"
    eng._has_rotating_cache = False
    eng._sliding_window_size = 0

    # Stub heavy / unrelated methods.
    eng._touch_gpu = lambda: None
    eng._has_disk_cache = lambda sid: False
    eng._find_base_cache = lambda messages, tools=None: None
    eng._maybe_register_base_cache = (
        lambda messages, prompt_tokens, tools=None, thinking=True: None
    )
    eng._mark_dirty = lambda sid: None
    # _get_cache_offset is a staticmethod — keep it on the class.

    return eng


def _fake_stream_factory(generated_ids: list[int], text_per_token: str = "x"):
    """Returns a callable matching vlm_stream_generate / lm_stream_generate
    signatures that yields one fake response per generated token id."""

    def _fake(*args, **kwargs):
        for tid in generated_ids:
            yield _FakeResp(text=text_per_token, token=int(tid),
                            prompt_tps=10.0, generation_tps=20.0)

    return _fake


# ---- Tests ----


def test_vlm_path_token_ids_extended_with_generated(monkeypatch):
    """VLM path: after one pass, token_ids == prompt_token_ids + generated."""
    generated = [101, 102, 103, 104, 105]
    eng = _make_stub_engine(generated, use_vlm=True)

    # Ensure tokenized prompt is non-empty and deterministic.
    prompt_text = "hello"
    expected_prompt_ids = [ord(c) for c in f"<user>{prompt_text}"]
    # Override _tokenize_prompt for total control.
    eng._tokenize_prompt = lambda messages, thinking=True, tools=None: list(
        expected_prompt_ids
    )

    monkeypatch.setattr(
        mlx_engine_module,
        "vlm_stream_generate",
        _fake_stream_factory(generated),
    )

    messages = [{"role": "user", "content": prompt_text}]
    drained = list(
        eng._generate_locked(
            messages,
            max_tokens=16,
            temperature=0.0,
            session_id="s1",
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
    assert drained, "generator produced no chunks"

    session = eng._sessions.get("s1")
    assert session is not None, "session should be persisted after generation"
    token_ids = session.cache_state.token_ids
    assert token_ids is not None, "cache_state.token_ids must be set"
    # The unified post-loop must end with the generated IDs.
    n = len(generated)
    assert token_ids[-n:] == generated, (
        f"expected trailing generated IDs {generated}, got {token_ids[-n:]}"
    )
    # And the prefix must equal the original prompt IDs.
    assert token_ids[: len(expected_prompt_ids)] == expected_prompt_ids


def test_mlx_lm_path_token_ids_extended_with_generated(monkeypatch):
    """mlx-lm legacy path: same invariant on token_ids."""
    generated = [201, 202, 203]
    eng = _make_stub_engine(generated, use_vlm=False)

    prompt_text = "world"
    expected_prompt_ids = [ord(c) for c in f"<user>{prompt_text}"]
    eng._tokenize_prompt = lambda messages, thinking=True, tools=None: list(
        expected_prompt_ids
    )

    # On the mlx-lm path, _generate_locked calls make_prompt_cache(...) — stub it
    # to return a trivial sentinel that the post-loop just stores.
    monkeypatch.setattr(
        mlx_engine_module, "make_prompt_cache",
        lambda lm: [SimpleNamespace(keys=None, values=None, offset=0)],
    )
    monkeypatch.setattr(
        mlx_engine_module,
        "lm_stream_generate",
        _fake_stream_factory(generated),
    )

    messages = [{"role": "user", "content": prompt_text}]
    drained = list(
        eng._generate_locked(
            messages,
            max_tokens=16,
            temperature=0.0,
            session_id="s2",
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
    assert drained
    session = eng._sessions.get("s2")
    assert session is not None
    token_ids = session.cache_state.token_ids
    assert token_ids is not None
    n = len(generated)
    assert token_ids[-n:] == generated, (
        f"mlx-lm path: expected trailing {generated}, got {token_ids[-n:]}"
    )
    assert token_ids[: len(expected_prompt_ids)] == expected_prompt_ids
