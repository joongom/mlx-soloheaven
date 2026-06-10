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
from mlx_lm.models.cache import KVCache, RotatingKVCache

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
    """Construct an MLXEngine shell (no model load) wired to tmp_path.

    Defaults to a NON-mtp-capable engine (no drafter loaded) — the loader's
    strip gate (_mtp_base_caches_active) must see the same attributes a real
    engine carries, so wire the plain-server shape explicitly.
    """
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.data_dir = str(tmp_path)  # cache_dir = data_dir/cache
    cfg.disk_budget_gb = 1.0
    eng.cfg = cfg  # cfg.kv_bits defaults to 0 (bf16 KV)
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    eng._language_model = SimpleNamespace()  # placeholder; not used by save
    eng._use_vlm = False
    eng._draft_kind = None  # no drafter -> _mtp_base_caches_active() False
    return eng


def _wire_mtp_capable(eng: MLXEngine, n_head_layers: int = 1) -> None:
    """Make the engine shell pass _mtp_base_caches_active() with a drafter
    whose head layer count drives the loader's n_extra == n_head gate (the
    same `max(1, len(drafter.layers) or 1)` computation as the MTP gate)."""
    eng._drafter = SimpleNamespace(layers=[object()] * n_head_layers)
    eng._draft_kind = "qwen_mtp"


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


# ---------------------------------------------------------------------------
# MTP-finalized sessions (target + trailing head entries) across restart
# ---------------------------------------------------------------------------
#
# qwen_mtp sessions persist n_target + n_head entries (head trailing by the
# lazy last slot: head_offset == N - 1). The finalize hidden the MTP gate
# would need (mtp_last_hidden) is an in-memory stash and is NEVER saved, so
# MTP reuse is impossible post-restart regardless — the loader must instead
# STRIP the trailing head entries and accept the load, preserving plain
# reuse of the full token history (next turn plans REUSE_FALLBACK_PLAIN).
#
# Stripping is gated to EXACTLY this layout (fail-closed): the engine must
# be qwen-mtp-capable (_mtp_base_caches_active), n_extra must equal the
# drafter's head layer count, and every trailing extra must be a KVCache
# (what qwen_mtp.make_head_cache produces) at offset len(token_ids) - 1.
# Any other oversized layout — foreign caches with extra target layers,
# same-type larger layouts, a plain server reading a head-carrying file —
# is rejected outright (load returns None).


def _save_session(eng, sid, layers, token_ids, *, messages=None):
    cache_state = PromptCacheState()
    cache_state.cache = layers
    cache_state.token_ids = token_ids
    session = SessionState(
        cache_state=cache_state,
        messages=messages or [{"role": "user", "content": "hi"}],
        total_cache_tokens=len(token_ids or []),
        last_used=1.0,
    )
    assert eng._save_session_to_disk(sid, session) is True
    return session


def test_session_load_strips_trailing_mtp_head_entries(tmp_path, monkeypatch):
    """FINDING-3 round-trip: a 41-entry-style MTP-finalized session (here
    3 target + 1 trailing head at N-1) survives reload — the loader strips
    the head to the model layout, target offsets == len(token_ids), and the
    stash is absent (never persisted)."""
    eng = _make_engine(tmp_path)
    _wire_mtp_capable(eng, n_head_layers=1)  # strip gate needs capability

    n = 4
    target = [_make_kvcache(seq_len=n) for _ in range(3)]
    head = [_make_kvcache(seq_len=n - 1)]  # trails by the lazy last slot
    token_ids = list(range(100, 100 + n))
    _save_session(eng, "mtp-roundtrip", target + head, token_ids)

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    loaded = eng._load_session_from_disk("mtp-roundtrip")
    assert loaded is not None, "head-carrying session must be ACCEPTED"
    # Stripped to the target-only layout; every offset == len(token_ids).
    assert len(loaded.cache_state.cache) == 3
    for c in loaded.cache_state.cache:
        assert c.offset == len(token_ids)
    assert loaded.cache_state.token_ids == token_ids
    assert loaded.total_cache_tokens == n
    # The single-use finalize stash is never persisted -> absent on reload
    # (the next MTP turn therefore plans the plain fallback, not MTP).
    assert getattr(loaded.cache_state, "mtp_last_hidden", None) is None
    assert getattr(loaded.cache_state, "mtp_hidden_offset", None) is None


def test_session_load_rejects_fewer_entries_than_model(tmp_path, monkeypatch):
    """Fail-closed: FEWER entries than model layers is still rejected (only
    extra TRAILING entries are strippable)."""
    eng = _make_engine(tmp_path)
    layers = [_make_kvcache(seq_len=4) for _ in range(2)]
    _save_session(eng, "short-load", layers, [1, 2, 3, 4])

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    assert eng._load_session_from_disk("short-load") is None


def test_session_load_rejects_wrong_leading_types(tmp_path, monkeypatch):
    """Fail-closed: trailing extras are only strippable when the LEADING
    len(model_cache) slice type-matches make_prompt_cache."""

    class NotKVCache:  # model expects this at index 2 — loaded KVCache won't match
        pass

    eng = _make_engine(tmp_path)
    layers = [_make_kvcache(seq_len=4) for _ in range(4)]  # 3 "target" + 1 head
    _save_session(eng, "wrong-types", layers, [1, 2, 3, 4])

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache(), KVCache(), NotKVCache()],
    )
    assert eng._load_session_from_disk("wrong-types") is None


def test_session_load_rejects_stripped_offset_mismatch(tmp_path, monkeypatch):
    """Fail-closed: after the strip, every offset-bearing target layer must
    sit exactly at len(token_ids); a desynced shape is rejected, never
    half-loaded. The trailing entry here is a VALID head (KVCache at
    len(token_ids) - 1) so the rejection is specifically the target-offset
    check, not the head gate."""
    eng = _make_engine(tmp_path)
    _wire_mtp_capable(eng, n_head_layers=1)
    layers = [_make_kvcache(seq_len=4) for _ in range(3)] + [
        _make_kvcache(seq_len=5)  # == len(token_ids) - 1: head gate passes
    ]
    # token_ids LONGER than the target offset (4) -> reject after strip.
    _save_session(eng, "offset-desync", layers, [1, 2, 3, 4, 5, 6])

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    assert eng._load_session_from_disk("offset-desync") is None


def test_session_load_rejects_wrong_trailing_type(tmp_path, monkeypatch):
    """Fail-closed strip gate (codex finding): leading slice type-matches and
    target offsets == len(token_ids), but the trailing extra is NOT what
    make_head_cache produces (RotatingKVCache, not KVCache) — even at the
    correct finalized offset len(token_ids) - 1 the load is rejected."""
    eng = _make_engine(tmp_path)
    _wire_mtp_capable(eng, n_head_layers=1)

    n = 4
    target = [_make_kvcache(seq_len=n) for _ in range(3)]
    rot = RotatingKVCache(max_size=16)
    k = mx.arange(2 * (n - 1) * 8, dtype=mx.float32).reshape(1, 2, n - 1, 8)
    rot.update_and_fetch(k, k * 0.5)  # offset == n - 1 == len(token_ids) - 1
    token_ids = list(range(100, 100 + n))
    _save_session(eng, "wrong-trailing-type", target + [rot], token_ids)

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    assert eng._load_session_from_disk("wrong-trailing-type") is None


def test_session_load_rejects_trailing_head_wrong_offset(tmp_path, monkeypatch):
    """Fail-closed strip gate: the trailing head is the right type but sits
    at len(token_ids) instead of the finalized lazy-last-slot contract
    len(token_ids) - 1 — rejected."""
    eng = _make_engine(tmp_path)
    _wire_mtp_capable(eng, n_head_layers=1)

    n = 4
    target = [_make_kvcache(seq_len=n) for _ in range(3)]
    head = [_make_kvcache(seq_len=n)]  # offset == len(token_ids): NOT finalized
    token_ids = list(range(100, 100 + n))
    _save_session(eng, "head-wrong-offset", target + head, token_ids)

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    assert eng._load_session_from_disk("head-wrong-offset") is None


def test_session_load_rejects_extra_count_mismatch(tmp_path, monkeypatch):
    """Fail-closed strip gate: n_extra must equal the drafter's head layer
    count. Two trailing entries (each individually head-shaped) against a
    1-layer head -> rejected."""
    eng = _make_engine(tmp_path)
    _wire_mtp_capable(eng, n_head_layers=1)

    n = 4
    target = [_make_kvcache(seq_len=n) for _ in range(3)]
    extras = [_make_kvcache(seq_len=n - 1), _make_kvcache(seq_len=n - 1)]
    token_ids = list(range(100, 100 + n))
    _save_session(eng, "extra-count", target + extras, token_ids)

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(3)],
    )
    assert eng._load_session_from_disk("extra-count") is None


def test_session_load_rejects_strip_when_not_mtp_capable(tmp_path, monkeypatch):
    """Fail-closed strip gate: a non-mtp-capable engine (no drafter — plain
    server) must NEVER strip, even when the file is a perfectly-shaped
    41-entry MTP layout (40 target + 1 head at N-1, target offsets == N) —
    the pre-feature layer-count fail-closed behavior is restored."""
    eng = _make_engine(tmp_path)  # default shell: no drafter, not capable

    n = 4
    target = [_make_kvcache(seq_len=n) for _ in range(40)]
    head = [_make_kvcache(seq_len=n - 1)]
    token_ids = list(range(100, 100 + n))
    _save_session(eng, "not-capable", target + head, token_ids)  # 41 entries

    monkeypatch.setattr(
        mlx_engine_module,
        "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(40)],
    )
    assert eng._load_session_from_disk("not-capable") is None
