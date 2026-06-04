"""Tests for the mlx-vlm 0.5.0 Gemma 4 MTP wrap-around mitigations.

Two layers live in ``mlx_soloheaven.engine.mlx_engine``:

* Layer B — ``_install_mtp_wrap_patches`` monkey-patches two locations
  in mlx-vlm so MTP drafter acceptance survives a ``RotatingKVCache``
  wrap. B1 reorders the ``shared_kv_sink`` into temporal order, and
  B2-v2 clone-replaces ``_mtp_rounds`` so the drafter's ``kv_offset``
  is clamped to ``max_size`` at both read sites. (A former B3 patch on
  ``rollback_speculative_cache`` was removed in RCA-2/2026-05-13 — see
  ``_install_mtp_wrap_patches`` for details.)
* Layer A — ``MLXEngine._will_wrap_during_generate`` plus the
  ``_run_vlm`` guard that drops the local ``drafter`` reference for the
  current request when a wrap is imminent.

These tests stub everything out — no real model is loaded.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import mlx_vlm.generate  # noqa: F401 — populate sys.modules
import mlx_vlm.models.gemma4.language as _g4lang

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import (
    MLXEngine,
    _clamped_kv_offset,
    _install_mtp_wrap_patches,
)


# ---------------------------------------------------------------------------
# Layer B: monkey-patch helper
# ---------------------------------------------------------------------------


# Snapshot the real upstream methods *before* any engine init (which
# would have installed the wrap patches). If the patches were already
# installed by an earlier import, walk the closure to recover the
# original from cell[0] (set by ``_install_mtp_wrap_patches``).
def _unwrap_patched(method):
    """Recover the pre-patch method captured in the closure of a wrapper
    annotated with ``_mtp_wrap_patch``. Returns ``method`` unchanged if
    it is not a patch wrapper.
    """
    if not getattr(method, "_mtp_wrap_patch", False):
        return method
    # Closure variable named ``_orig_textmodel_call`` or ``_orig_rollback``
    # is the first one we set up. ``__closure__`` is ordered by free var
    # name in ``__code__.co_freevars``.
    try:
        names = method.__code__.co_freevars
        cells = method.__closure__ or ()
        for name, cell in zip(names, cells):
            val = cell.cell_contents
            if callable(val) and val is not method:
                return val
    except Exception:  # noqa: BLE001
        pass
    return method


_REAL_TEXTMODEL_CALL = _unwrap_patched(_g4lang.Gemma4TextModel.__call__)
_REAL_ROLLBACK = _unwrap_patched(
    _g4lang.LanguageModel.rollback_speculative_cache
)


def _reset_patch_state():
    """Force-reset patch state and restore upstream methods so each test
    can re-apply cleanly.
    """
    mlx_engine_module._MTP_PATCHES_INSTALLED = False
    # PERF: ``_HOT_PATH_FAST`` is set by ``_run_vlm`` based on the current
    # request's wrap risk. Direct-invocation tests below need it OFF so
    # the patches run their full logic (the regime the tests exercise).
    mlx_engine_module._HOT_PATH_FAST = False
    _g4lang.Gemma4TextModel.__call__ = _REAL_TEXTMODEL_CALL
    _g4lang.LanguageModel.rollback_speculative_cache = _REAL_ROLLBACK


def test_install_mtp_wrap_patches_is_idempotent():
    """Calling _install_mtp_wrap_patches multiple times must be safe."""
    _reset_patch_state()
    try:
        assert _install_mtp_wrap_patches() is True
        # First call installed the wrapper.
        first_wrapper = _g4lang.Gemma4TextModel.__call__
        assert getattr(first_wrapper, "_mtp_wrap_patch", False) is True

        # Second call must short-circuit (returns False) and must NOT
        # rewrap — the method identity is unchanged.
        assert _install_mtp_wrap_patches() is False
        assert _g4lang.Gemma4TextModel.__call__ is first_wrapper, (
            "second install must not rewrap the method"
        )
    finally:
        _reset_patch_state()
        _install_mtp_wrap_patches()


def test_b1_temporal_order_called_on_rotating_cache():
    """B1: when a RotatingKVCache has wrapped, the patched
    ``Gemma4TextModel.__call__`` must reorder ``shared_kv_sink`` via
    ``_temporal_order``. For a non-rotating cache (KVCache) the sink
    must pass through unchanged.
    """
    captured = {}

    class _WrappedCache:
        """Mimics a wrapped RotatingKVCache: offset > max_size, has _temporal_order."""

        def __init__(self):
            self.offset = 2000
            self.max_size = 1024
            self.calls = []

        def _temporal_order(self, v):
            self.calls.append(v)
            return f"REORDERED::{v}"

    _WrappedCache.__name__ = "RotatingKVCache"

    class _KVCache:
        offset = 100
        max_size = 1024

        def _temporal_order(self, v):  # pragma: no cover — must not be called
            raise AssertionError("_temporal_order called on non-rotating cache")

    _KVCache.__name__ = "KVCache"

    rot = _WrappedCache()
    kv = _KVCache()

    # Install a stub as the upstream method, then apply our patches on
    # top of it. The wrapper's closure captures this stub as
    # ``_orig_textmodel_call``.
    def _stub_original(self, *args, **kwargs):
        captured["called"] = True
        return SimpleNamespace(stub=True)

    _reset_patch_state()
    _g4lang.Gemma4TextModel.__call__ = _stub_original
    try:
        assert _install_mtp_wrap_patches() is True
        wrapped = _g4lang.Gemma4TextModel.__call__
        assert getattr(wrapped, "_mtp_wrap_patch", False) is True

        # B1a: RotatingKVCache wrapped → sink reordered.
        self_obj = SimpleNamespace(
            layers=[
                SimpleNamespace(layer_type="sliding_attention"),
                SimpleNamespace(layer_type="full_attention"),
            ],
        )
        sink = {"sliding_attention": ("K_RAW", "V_RAW")}
        wrapped(self_obj, cache=[rot, kv], shared_kv_sink=sink)
        assert captured.get("called") is True
        assert sink["sliding_attention"] == (
            "REORDERED::K_RAW",
            "REORDERED::V_RAW",
        ), f"expected reordered sink, got {sink}"
        assert rot.calls == ["K_RAW", "V_RAW"], (
            f"_temporal_order should be called on K then V, got {rot.calls}"
        )

        # B1b: only KVCache → sink left unchanged, _temporal_order untouched.
        rot.calls.clear()
        self_obj2 = SimpleNamespace(
            layers=[SimpleNamespace(layer_type="full_attention")],
        )
        sink2 = {"full_attention": ("K_RAW2", "V_RAW2")}
        wrapped(self_obj2, cache=[kv], shared_kv_sink=sink2)
        assert sink2["full_attention"] == ("K_RAW2", "V_RAW2"), (
            f"non-rotating sink must pass through unchanged, got {sink2}"
        )
        assert rot.calls == [], (
            f"_temporal_order must NOT fire for KVCache, got {rot.calls}"
        )
    finally:
        _reset_patch_state()
        _install_mtp_wrap_patches()


def test_b2v2_inner_loop_clamp_applied():
    """B2-v2: wrapped RotatingKVCache (offset > max_size) → ``_clamped_kv_offset``
    must return ``max_size`` so the drafter's SWA mask receives a
    query index inside the window. This is the inner-loop read site
    that the old save/restore patch missed.
    """

    class _WrappedRot:
        offset = 1100
        max_size = 1024

    _WrappedRot.__name__ = "RotatingKVCache"

    assert _clamped_kv_offset([_WrappedRot()]) == 1024


def test_b2v2_no_clamp_for_non_rotating():
    """B2-v2: a plain KVCache must not be clamped — only RotatingKVCache
    exhibits the offset > max_size divergence after wrap.
    """

    class _Plain:
        offset = 1100
        # No max_size attribute set, mirroring full-attention KVCache.

    _Plain.__name__ = "KVCache"

    assert _clamped_kv_offset([_Plain()]) == 1100

    # Same class name with a max_size set must still bypass the clamp.
    class _PlainWithMax:
        offset = 1100
        max_size = 1024

    _PlainWithMax.__name__ = "KVCache"

    assert _clamped_kv_offset([_PlainWithMax()]) == 1100


def test_b2v2_no_clamp_when_within_max():
    """B2-v2: RotatingKVCache with offset ≤ max_size has not wrapped;
    the raw offset must be returned unchanged.
    """

    class _UnwrappedRot:
        offset = 500
        max_size = 1024

    _UnwrappedRot.__name__ = "RotatingKVCache"

    assert _clamped_kv_offset([_UnwrappedRot()]) == 500


def test_b3_removed_upstream_rollback_used():
    """B3 REMOVED (RCA-2, 2026-05-13): ``_install_mtp_wrap_patches`` must
    NOT install a wrapper on ``LanguageModel.rollback_speculative_cache``.
    Upstream's ``c.trim(n)`` is unconditionally safe; the old B3 patch
    skipped it on wrapped RotatingKVCache, leaving rejected speculative
    K/V slots in the ring buffer and contaminating target attention.
    """
    try:
        LanguageModel = _g4lang.LanguageModel
    except AttributeError:
        pytest.skip("mlx_vlm.models.gemma4.language.LanguageModel unavailable")

    _reset_patch_state()
    try:
        assert _install_mtp_wrap_patches() is True

        rollback = LanguageModel.rollback_speculative_cache
        assert not hasattr(rollback, "_mtp_wrap_patch"), (
            "B3 patch must NOT be installed; upstream "
            "rollback_speculative_cache must be used unchanged "
            "(see RCA-2 2026-05-13). Got wrapper with _mtp_wrap_patch marker."
        )
    finally:
        _reset_patch_state()
        _install_mtp_wrap_patches()


# ---------------------------------------------------------------------------
# Layer A: drafter-skip safety net
# ---------------------------------------------------------------------------


def _bare_vlm_engine() -> MLXEngine:
    cfg = Config()
    cfg.draft_model = None
    cfg.draft_kind = None
    cfg.draft_block_size = None
    cfg.pld_enabled = False
    cfg.prefill_step_size = 512
    eng = MLXEngine(cfg)
    eng._use_vlm = True
    eng._vlm_model = SimpleNamespace()
    eng._processor = SimpleNamespace()
    eng._drafter = None
    eng._draft_kind = None
    eng._safe_to_reuse_cache = lambda cs, pids=None: True
    return eng


def test_will_wrap_returns_true_at_boundary():
    """Layer A predicate: offset+prompt >= sliding_window → wrap imminent."""
    eng = _bare_vlm_engine()
    try:
        eng._has_rotating_cache = True
        eng._sliding_window_size = 1024

        # Case 1: offset=900, prompt=200 → 1100 >= 1024 → True
        cs = SimpleNamespace(cache=[SimpleNamespace(offset=900)])
        assert eng._will_wrap_during_generate([0] * 200, cs) is True

        # Case 2: offset=900, prompt=50 → 950 < 1024 → False
        cs2 = SimpleNamespace(cache=[SimpleNamespace(offset=900)])
        assert eng._will_wrap_during_generate([0] * 50, cs2) is False

        # Case 3: no rotating cache → always False
        eng._has_rotating_cache = False
        assert eng._will_wrap_during_generate([0] * 5000, cs) is False
    finally:
        eng.close()


def test_run_vlm_skips_drafter_when_will_wrap(monkeypatch):
    """When _will_wrap_during_generate returns True, _run_vlm must NOT
    pass ``draft_model`` to ``vlm_stream_generate``.
    """
    eng = _bare_vlm_engine()
    try:
        # Pretend a drafter is loaded.
        sentinel_drafter = SimpleNamespace(accept_lens=[])
        eng._drafter = sentinel_drafter
        eng._draft_kind = "mtp"
        eng._has_rotating_cache = True
        eng._sliding_window_size = 1024

        captured_kwargs = {}

        def _fake_stream(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return iter([])

        monkeypatch.setattr(
            mlx_engine_module, "vlm_stream_generate", _fake_stream
        )

        # Force wrap-imminent: cache_state offset=1000, prompt=200.
        from mlx_vlm.generate import PromptCacheState

        cs = PromptCacheState()
        cs.cache = [SimpleNamespace(offset=1000)]
        cs.token_ids = [0] * 1000

        def _drive():
            gen = eng._run_vlm(
                cache_state=cs,
                prompt_token_ids=[0] * 200,
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
                min_p=0.0,
                top_k=0,
                logits_processors=None,
                session_id="s-wrap",
                total_prompt_tokens=200,
            )
            list(gen)

        eng._vlm_executor.submit(_drive).result(timeout=10)

        assert "draft_model" not in captured_kwargs, (
            f"draft_model must not be forwarded when wrap imminent, "
            f"got kwargs keys={list(captured_kwargs.keys())}"
        )
        # And the loaded drafter must remain attached for future requests.
        assert eng._drafter is sentinel_drafter
    finally:
        eng.close()
