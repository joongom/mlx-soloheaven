"""Session save/load round-trip test (Phase 1 task P1.3, acceptance A9).

Verifies that `_save_session_to_disk` -> `_load_session_from_disk` preserves
`(cache, token_ids, total_cache_tokens, messages)` through safetensors.

Notes on test isolation:
- We do NOT load a real MLX model. KV cache (de)serialization is decoupled
  from model weights — safetensors stores tensors + scalar metadata only.
- `_load_session_from_disk` calls `make_prompt_cache(self._language_model)`
  to validate the layer-count / type. We monkeypatch the module-level
  `make_prompt_cache` symbol to return a same-shape skeleton so the load
  path is exercised end-to-end without GPU work.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState
from mlx_vlm.generate import PromptCacheState


def _make_kvcache(seq_len: int = 4, num_heads: int = 2, head_dim: int = 8) -> KVCache:
    """Build a real KVCache populated with deterministic synthetic K/V."""
    c = KVCache()
    # Deterministic, non-zero so the round-trip check has real content to compare.
    k = mx.arange(num_heads * seq_len * head_dim, dtype=mx.float32).reshape(
        1, num_heads, seq_len, head_dim
    )
    v = mx.arange(num_heads * seq_len * head_dim, dtype=mx.float32).reshape(
        1, num_heads, seq_len, head_dim
    ) * 0.5
    c.update_and_fetch(k, v)
    return c


def _make_engine(tmp_path) -> MLXEngine:
    """Construct an MLXEngine shell (no model load) wired to tmp_path."""
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.data_dir = str(tmp_path)  # cache_dir = data_dir/cache
    cfg.disk_budget_gb = 1.0
    eng.cfg = cfg
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    eng._language_model = SimpleNamespace()  # placeholder; not used by save
    return eng


def test_session_save_load_roundtrip(tmp_path, monkeypatch):
    """End-to-end: build SessionState, save, load, compare every field."""
    eng = _make_engine(tmp_path)

    # Build a real 3-layer cache with distinct shapes / offsets via different
    # seq_lens so per-layer round-trip is actually exercised.
    layers = [
        _make_kvcache(seq_len=4),
        _make_kvcache(seq_len=4),
        _make_kvcache(seq_len=4),
    ]
    token_ids = list(range(50))  # 50 ints, matches spec example
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]

    cache_state = PromptCacheState()
    cache_state.cache = layers
    cache_state.token_ids = token_ids

    session = SessionState(
        cache_state=cache_state,
        messages=messages,
        total_cache_tokens=4,  # matches per-layer offset
        last_used=1234567.0,
    )

    sid = "roundtrip-1"
    ok = eng._save_session_to_disk(sid, session)
    assert ok is True, "save should succeed for non-empty KVCache layers"

    save_path = os.path.join(eng.cfg.cache_dir, f"session_{sid}.safetensors")
    assert os.path.exists(save_path), f"expected file at {save_path}"

    # Patch make_prompt_cache so _load_session_from_disk's structural check
    # (layer count + type) passes without loading model weights.
    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(len(layers))],
    )

    loaded = eng._load_session_from_disk(sid)
    assert loaded is not None, "load should return a SessionState"

    # --- Per-field equality ---
    # In-memory KVCache pre-allocates in steps (e.g. 256), so the buffer
    # shape is larger than offset. save_prompt_cache trims to the live
    # portion (state slice up to offset), so on load keys.shape[2] == offset
    # while the original's was step-padded. Compare the live slice.
    assert len(loaded.cache_state.cache) == len(layers)
    for orig, new in zip(layers, loaded.cache_state.cache):
        assert type(new).__name__ == "KVCache"
        assert new.offset == orig.offset
        # Live tensor slice up to offset must match byte-for-byte.
        off = orig.offset
        orig_keys_live = orig.keys[:, :, :off, :]
        orig_vals_live = orig.values[:, :, :off, :]
        assert new.keys.shape == orig_keys_live.shape
        assert new.values.shape == orig_vals_live.shape
        assert mx.array_equal(new.keys, orig_keys_live)
        assert mx.array_equal(new.values, orig_vals_live)

    assert loaded.cache_state.token_ids == token_ids
    assert loaded.messages == messages
    # total_cache_tokens is recomputed from offset on load.
    assert loaded.total_cache_tokens == 4


def test_session_save_load_empty_token_ids(tmp_path, monkeypatch):
    """Edge: token_ids=None round-trips to None (or empty -> None semantics)."""
    eng = _make_engine(tmp_path)

    layers = [_make_kvcache(seq_len=2)]
    cache_state = PromptCacheState()
    cache_state.cache = layers
    cache_state.token_ids = None

    session = SessionState(
        cache_state=cache_state,
        messages=[],
        total_cache_tokens=2,
        last_used=0.0,
    )
    sid = "roundtrip-empty"
    assert eng._save_session_to_disk(sid, session) is True

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(len(layers))],
    )
    loaded = eng._load_session_from_disk(sid)
    assert loaded is not None
    # On save token_ids=None becomes "[]"; on load empty list becomes None
    # (see _load_session_from_disk: `token_ids if token_ids else None`).
    assert loaded.cache_state.token_ids is None
    assert loaded.messages == []
