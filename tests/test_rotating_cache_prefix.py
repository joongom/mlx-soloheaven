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
