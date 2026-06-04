"""RotatingKVCache prefix-reuse safety probe (Phase 1 task P1.4).

`MLXEngine._safe_to_reuse_cache` must return False when *any* cache entry
is a `RotatingKVCache` whose ring buffer has wrapped (offset > max_size).
Once wrapped, the buffer no longer corresponds to a contiguous token
prefix, so `PromptCacheState.find_prefix_length` + mlx-vlm's reuse-trim
path would silently mis-align KV entries with the new prompt's tokens.

These tests exercise the helper directly using a MagicMock shaped like
`mlx_lm.models.cache.RotatingKVCache` (real attribute names: `offset`,
`max_size`; see .venv/lib/.../mlx_lm/models/cache.py:417-418). The live
wrap-around scenario (real Gemma 4 prompt > sliding window) is an
integration test and is left as a skip-placeholder below.
"""

from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine.mlx_engine import MLXEngine


# Plain stand-in classes whose `type(...).__name__` matches what the helper
# inspects. We deliberately do NOT subclass MagicMock — the helper only
# looks at the class name + two attributes, so a dataclass-shaped object
# is the cleanest fake here.
class RotatingKVCache:  # noqa: D401  (intentional name shadowing for type() check)
    def __init__(self, offset=None, max_size=None):
        if offset is not None:
            self.offset = offset
        if max_size is not None:
            self.max_size = max_size


class KVCache:
    def __init__(self, offset):
        self.offset = offset


def _make_rotating_mock(offset: int, max_size: int):
    return RotatingKVCache(offset=offset, max_size=max_size)


def _make_full_attention_mock(offset: int):
    return KVCache(offset=offset)


def test_safe_to_reuse_cache_no_rotation_returns_true():
    """RotatingKVCache that has NOT wrapped (offset <= max_size) is safe."""
    rot = _make_rotating_mock(offset=512, max_size=1024)
    cache_state = SimpleNamespace(cache=[rot])
    assert MLXEngine._safe_to_reuse_cache(cache_state) is True


def test_safe_to_reuse_cache_wrap_around_returns_false():
    """Once offset > max_size the ring buffer has wrapped — not safe."""
    rot = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = SimpleNamespace(cache=[rot])
    assert MLXEngine._safe_to_reuse_cache(cache_state) is False


def test_safe_to_reuse_cache_empty_cache_returns_true():
    """Empty / None cache: nothing to gate, helper must return True."""
    assert MLXEngine._safe_to_reuse_cache(SimpleNamespace(cache=[])) is True
    assert MLXEngine._safe_to_reuse_cache(SimpleNamespace(cache=None)) is True
    assert MLXEngine._safe_to_reuse_cache(None) is True


def test_safe_to_reuse_cache_mixed_layers_one_wrapped_returns_false():
    """If ANY rotating layer has wrapped, the whole cache is unsafe."""
    full = _make_full_attention_mock(offset=2048)
    rot_safe = _make_rotating_mock(offset=512, max_size=1024)
    rot_wrapped = _make_rotating_mock(offset=4096, max_size=1024)
    cache_state = SimpleNamespace(cache=[full, rot_safe, rot_wrapped])
    assert MLXEngine._safe_to_reuse_cache(cache_state) is False


def test_safe_to_reuse_cache_non_rotating_only_returns_true():
    """All-KVCache layers (e.g. text-only mlx-lm path) are always safe."""
    full_a = _make_full_attention_mock(offset=512)
    full_b = _make_full_attention_mock(offset=512)
    cache_state = SimpleNamespace(cache=[full_a, full_b])
    assert MLXEngine._safe_to_reuse_cache(cache_state) is True


def test_safe_to_reuse_cache_unknown_layout_is_conservative():
    """A RotatingKVCache-named entry missing `max_size` or `offset` must
    be treated as unsafe (conservative cold-fill)."""
    weird = RotatingKVCache()  # no offset / max_size set
    cache_state = SimpleNamespace(cache=[weird])
    assert MLXEngine._safe_to_reuse_cache(cache_state) is False


@pytest.mark.skip(
    reason="Integration test: requires loading Gemma 4 and prefilling a prompt "
    "longer than the sliding_window (1024). Run manually with a real model "
    "after a wrap-around boundary to verify outputs do not diverge from a "
    "cold-fill baseline."
)
def test_safe_to_reuse_cache_live_wrap_around_gemma4():
    """Placeholder — see skip reason."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Append-only wrapped reuse (codex-validated allowance).
#
# Once a RotatingKVCache has wrapped (offset > max_size) the helper must
# still permit reuse iff the ENTIRE cached logical history is a strict
# prefix of the new prompt (pure append). Any divergence/branch past the
# wrap, or a prompt that is not strictly longer, must cold-fill.
#
# These cases need a cache_state exposing `.token_ids` and a real
# `find_prefix_length` (copied byte-for-byte from mlx-vlm's PromptCacheState:
# leading-match count), so we use a small fake class rather than
# SimpleNamespace.
# ---------------------------------------------------------------------------


class _FakeCacheState:
    """Minimal stand-in for mlx-vlm's PromptCacheState used by the helper."""

    def __init__(self, cache, token_ids):
        self.cache = cache
        self.token_ids = token_ids

    def find_prefix_length(self, new_ids: list) -> int:
        """Return the number of leading tokens that match the cached ids.

        Copied from mlx_vlm.generate.PromptCacheState.find_prefix_length.
        """
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len


def test_safe_to_reuse_cache_wrapped_strict_append_returns_true():
    """Case 1: wrapped + cached=[0..2047] (offset=2048>max=1024) +
    prompt = cached + suffix → reuse is SAFE (pure append)."""
    cached_ids = list(range(2048))
    rot = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    prompt = cached_ids + [9000, 9001, 9002]
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is True


def test_safe_to_reuse_cache_wrapped_divergence_returns_false():
    """Case 2: wrapped + prompt shares 1500 tokens then diverges → False."""
    cached_ids = list(range(2048))
    rot = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    # Shares first 1500, then differs (and is shorter than the cached history).
    prompt = list(range(1500)) + [55555, 55556]
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is False


def test_safe_to_reuse_cache_wrapped_divergence_at_max_size_returns_false():
    """Case 3: wrapped + divergence exactly at prefix_len == max_size → False."""
    cached_ids = list(range(2048))
    rot = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    # Matches first 1024 (== max_size), then diverges.
    prompt = list(range(1024)) + [-1] + list(range(1025, 2100))
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is False


def test_safe_to_reuse_cache_unwrapped_rotating_returns_true_with_ids():
    """Case 4: rotating but NOT wrapped (offset <= max_size) → True.

    The append/divergence gate is irrelevant here; reuse is always safe.
    """
    cached_ids = list(range(500))
    rot = _make_rotating_mock(offset=500, max_size=1024)
    cache_state = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    # Even a divergent prompt is fine on a non-wrapped rotating cache.
    prompt = list(range(300)) + [777]
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is True


def test_safe_to_reuse_cache_non_rotating_returns_true_with_ids():
    """Case 5: non-rotating cache (KVCache only) → True regardless of ids."""
    full = _make_full_attention_mock(offset=2048)
    cache_state = _FakeCacheState(cache=[full], token_ids=list(range(2048)))
    prompt = list(range(10)) + [42]  # divergent — still safe (no rotating).
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is True


def test_safe_to_reuse_cache_wrapped_offset_mismatch_returns_false():
    """Case 6: wrapped, strict append, but a cache entry's offset !=
    len(token_ids) → unsafe (RoPE continuation invariant broken)."""
    cached_ids = list(range(2048))
    # Strict append would otherwise pass, but offset (1999) != 2048.
    rot = _make_rotating_mock(offset=1999, max_size=1024)
    # Force the wrap branch with a second, genuinely-wrapped entry whose
    # offset DOES match, so the mismatch on `rot` is what trips the guard.
    rot_wrapped = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = _FakeCacheState(
        cache=[rot_wrapped, rot], token_ids=cached_ids
    )
    prompt = cached_ids + [9000]
    assert MLXEngine._safe_to_reuse_cache(cache_state, prompt) is False


def test_safe_to_reuse_cache_wrapped_prompt_ids_none_is_conservative():
    """Case 7: wrapped + prompt_token_ids=None (default) → conservative False.

    This is what keeps the legacy 1-arg call sites backward compatible.
    """
    cached_ids = list(range(2048))
    rot = _make_rotating_mock(offset=2048, max_size=1024)
    cache_state = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    assert MLXEngine._safe_to_reuse_cache(cache_state) is False
    assert MLXEngine._safe_to_reuse_cache(cache_state, None) is False
