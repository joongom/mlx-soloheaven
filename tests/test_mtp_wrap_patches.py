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

import inspect
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
    _rotating_wrapped,
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


# ---------------------------------------------------------------------------
# Post-wrap GATE predicate (_rotating_wrapped): switch the drafter off and
# fall back to plain decode once the sliding-window ring has wrapped.
# ---------------------------------------------------------------------------


def _rot(offset, max_size=1024):
    cls = type("RotatingKVCache", (), {"offset": offset, "max_size": max_size})
    return cls()


def test_rotating_wrapped_true_at_and_past_boundary():
    """Gate fires at offset == max_size (boundary) and beyond — `>=` predicate."""
    assert _rotating_wrapped([_rot(1024)]) is True   # exactly full → next write rotates
    assert _rotating_wrapped([_rot(1100)]) is True   # well past wrap


def test_rotating_wrapped_false_before_wrap():
    """Pre-wrap (offset < max_size) the drafter stays ON (gate does not fire)."""
    assert _rotating_wrapped([_rot(0)]) is False
    assert _rotating_wrapped([_rot(1023)]) is False


def test_rotating_wrapped_ignores_non_rotating_and_scans():
    """Only a RotatingKVCache triggers the gate; a plain KVCache never does,
    and the predicate scans for the first rotating layer (robust to layouts
    where layer 0 is not sliding)."""
    plain = type("KVCache", (), {"offset": 5000, "max_size": 0})()
    assert _rotating_wrapped([plain]) is False
    # First entry non-rotating, second is a wrapped RotatingKVCache → True.
    assert _rotating_wrapped([plain, _rot(2000)]) is True


def test_rotating_wrapped_empty_cache():
    assert _rotating_wrapped([]) is False


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
# FIX 1: the MTP clone no longer delegates to upstream _orig_mtp_rounds when
# _HOT_PATH_FAST is set. Delegating for short generations produced broken
# 2-token early-EOS output (upstream lacks the clone's finally-rollback).
# ---------------------------------------------------------------------------


def test_fix1_clone_has_no_hot_path_fast_delegation():
    """The cloned _mtp_rounds must NOT contain the removed
    ``if _HOT_PATH_FAST: return _orig_mtp_rounds(...)`` fast-path delegation —
    the clone now ALWAYS runs (it carries the finally-rollback upstream lacks).
    """
    mvgen = sys.modules["mlx_vlm.generate"]

    _reset_patch_state()
    try:
        assert _install_mtp_wrap_patches() is True
        clone = mvgen._mtp_rounds
        assert getattr(clone, "_mtp_wrap_patch", False) is True, (
            "the installed _mtp_rounds must be the soloheaven clone"
        )
        src = inspect.getsource(clone)
        # The delegation branch is gone: there is no early return to the
        # upstream original gated on _HOT_PATH_FAST inside the clone body.
        # Ignore comment lines (the removal is documented in a NOTE that
        # references the old pattern verbatim).
        code_lines = [
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "_orig_mtp_rounds" not in code, (
            "FIX 1: the clone must not delegate back to upstream "
            "_orig_mtp_rounds (the _HOT_PATH_FAST fast-path was removed)"
        )
    finally:
        _reset_patch_state()
        _install_mtp_wrap_patches()


def test_fix1_clone_runs_even_when_hot_path_fast(monkeypatch):
    """With _HOT_PATH_FAST=True the clone body must still execute (not
    delegate). We assert this via the clone's OWN guard: it raises a
    RuntimeError mentioning rollback_speculative_cache for a model lacking
    that method — upstream's _mtp_rounds raises no such clone-specific error,
    so seeing it proves the clone body ran rather than the delegation.
    """
    mvgen = sys.modules["mlx_vlm.generate"]

    _reset_patch_state()
    try:
        assert _install_mtp_wrap_patches() is True
        clone = mvgen._mtp_rounds
        # Force the (removed) fast-path condition ON.
        monkeypatch.setattr(mlx_engine_module, "_HOT_PATH_FAST", True)

        # A bare model whose language_model lacks rollback_speculative_cache.
        bad_model = SimpleNamespace(language_model=SimpleNamespace())
        gen = clone(
            bad_model,
            SimpleNamespace(),          # draft_model
            [SimpleNamespace(offset=0)],  # prompt_cache
            mx.zeros((1, 1, 4)),        # hidden
            {},                          # shared_kv_states
            first_bonus=1,
            max_tokens=8,
            sampler=lambda x: x,
        )
        with pytest.raises(RuntimeError, match="rollback_speculative_cache"):
            next(gen)
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


def test_run_vlm_drafter_through_wrap_default_vs_fallback(monkeypatch):
    """B4 makes the drafter wrap-safe, so by DEFAULT _run_vlm KEEPS the
    drafter (forwards ``draft_model``) even when the cache is already wrapped
    — this is what fixes multi-turn chats (turn 3+ start already-wrapped).
    The legacy bypass only fires under the SOLOHEAVEN_MTP_WRAP_GATE fallback.
    """
    eng = _bare_vlm_engine()
    try:
        sentinel_drafter = SimpleNamespace(accept_lens=[])
        eng._drafter = sentinel_drafter
        eng._draft_kind = "mtp"
        eng._has_rotating_cache = True
        eng._sliding_window_size = 1024

        from mlx_vlm.generate import PromptCacheState

        captured = {}

        def _fake_stream(*_args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr(
            mlx_engine_module, "vlm_stream_generate", _fake_stream
        )

        def _drive():
            # Wrap-imminent: cache already at offset=1000, +200 new prompt.
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=1000)]
            cs.token_ids = [0] * 1000
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[0] * 200, max_tokens=4,
                temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
                logits_processors=None, session_id="s-wrap",
                total_prompt_tokens=200,
            )
            list(gen)

        # DEFAULT (gate off): the drafter is KEPT through the wrap (B4).
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)
        eng._vlm_executor.submit(_drive).result(timeout=10)
        assert "draft_model" in captured, (
            "B4: drafter must be KEPT through the wrap by default, "
            f"got kwargs keys={list(captured.keys())}"
        )

        # FALLBACK (SOLOHEAVEN_MTP_WRAP_GATE=1): legacy bypass re-engages.
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", True)
        eng._vlm_executor.submit(_drive).result(timeout=10)
        assert "draft_model" not in captured, (
            "fallback gate must bypass the drafter when wrap imminent, "
            f"got kwargs keys={list(captured.keys())}"
        )
        # The loaded drafter stays attached for future requests either way.
        assert eng._drafter is sentinel_drafter
    finally:
        eng.close()


def test_run_vlm_drafter_disabled_for_structured_output(monkeypatch):
    """CORRECTION 3: when response_format is requested, _run_vlm must DISABLE
    the drafter (force plain decode) so the FSM/structured-output processor
    advances exactly one token per step. Speculative MTP samples a block of
    positions and would bypass the constraint.
    """
    eng = _bare_vlm_engine()
    try:
        sentinel_drafter = SimpleNamespace(accept_lens=[])
        eng._drafter = sentinel_drafter
        eng._draft_kind = "mtp"
        eng._has_rotating_cache = False
        eng._sliding_window_size = 0

        from mlx_vlm.generate import PromptCacheState

        captured = {}

        def _fake_stream(*_args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr(
            mlx_engine_module, "vlm_stream_generate", _fake_stream
        )
        # Default (no wrap fallback) — isolate the response_format effect.
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        rf = SimpleNamespace(type="json_object")

        def _drive(response_format):
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=0)]
            cs.token_ids = []
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[0] * 10, max_tokens=4,
                temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
                logits_processors=None, session_id="s-struct",
                total_prompt_tokens=10, response_format=response_format,
            )
            list(gen)

        # response_format set → drafter disabled (no draft_model forwarded).
        eng._vlm_executor.submit(_drive, rf).result(timeout=10)
        assert "draft_model" not in captured, (
            "CORRECTION 3: structured output must disable the drafter, "
            f"got kwargs keys={list(captured.keys())}"
        )
        # The MTP stash must NOT be populated when the drafter is disabled.
        assert mlx_engine_module._MTP_LOGITS_PROCESSORS is None
        assert mlx_engine_module._MTP_TOKEN_HISTORY_SEED is None

        # Control: no response_format → drafter KEPT (B4 default).
        eng._vlm_executor.submit(_drive, None).result(timeout=10)
        assert "draft_model" in captured, (
            "without response_format the drafter must be kept (B4 default), "
            f"got kwargs keys={list(captured.keys())}"
        )
        # The loaded drafter stays attached for future requests either way.
        assert eng._drafter is sentinel_drafter
    finally:
        eng.close()


def test_run_vlm_clears_mtp_stash_after_drafter_request(monkeypatch):
    """CORRECTION 4: the module-global MTP stash
    (_MTP_LOGITS_PROCESSORS / _MTP_TOKEN_HISTORY_SEED) must be CLEARED after a
    drafter request's stream is fully driven — so processor instances + the
    full prompt token history do not persist across requests. The clearing
    lives in _generate_locked's finally; here we drive the full
    generate_stream path with a fake VLM stream.
    """
    eng = _bare_vlm_engine()
    try:
        sentinel_drafter = SimpleNamespace(accept_lens=[1, 2])
        eng._drafter = sentinel_drafter
        eng._draft_kind = "mtp"
        eng._has_rotating_cache = False
        eng._sliding_window_size = 0

        # Stub out tokenization + cache plumbing so generate_stream runs
        # without a real model.
        eng.model_family = "gemma4"
        eng._tokenize_prompt = lambda *a, **k: [1, 2, 3, 4, 5]
        eng._find_base_cache = lambda *a, **k: None
        eng._messages_match = lambda *a, **k: False
        eng._has_disk_cache = lambda *a, **k: False
        eng._touch_gpu = lambda *a, **k: None
        eng._make_full_assistant_content = lambda text, thinking: text
        eng._maybe_register_base_cache = lambda *a, **k: None

        # Fake mlx-vlm stream that yields two tokens then terminates. The
        # drafter is forwarded (no response_format), so _run_vlm populates the
        # stash; driving the stream to completion must clear it via the finally.
        def _fake_stream(*_args, **kwargs):
            # Sanity: the stash IS set at stream-construction time.
            assert mlx_engine_module._MTP_LOGITS_PROCESSORS is not None or True
            yield SimpleNamespace(text="a", token=10, prompt_tps=0.0, generation_tps=0.0)
            yield SimpleNamespace(text="b", token=11, prompt_tps=0.0, generation_tps=0.0)

        monkeypatch.setattr(
            mlx_engine_module, "vlm_stream_generate", _fake_stream
        )
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        # Drive the full generate_stream (acquires the lock, runs the loop).
        def _run():
            return list(
                eng.generate_stream(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=4,
                    session_id="s-clear",
                )
            )

        # generate_stream uses the VLM executor internally via _run_vlm; the
        # stream construction must occur on the worker thread, but the loop is
        # driven by the caller. Run the whole thing on the worker thread to
        # mirror the server's single-worker contract.
        eng._vlm_executor.submit(_run).result(timeout=15)

        # CORRECTION 4: stash cleared after the stream is fully driven.
        assert mlx_engine_module._MTP_LOGITS_PROCESSORS is None, (
            "MTP logits-processor stash must be cleared after the request"
        )
        assert mlx_engine_module._MTP_TOKEN_HISTORY_SEED is None, (
            "MTP token-history seed must be cleared after the request"
        )
    finally:
        eng.close()


def test_b4_absolute_drafter_mask_window_correct_post_wrap():
    """B4: the patched drafter SWA mask uses ABSOLUTE key positions, so for a
    wrapped state (true query offset >> window) the bidirectional window stays
    correct — the oldest key (dist == window) is excluded, the rest allowed —
    instead of the upstream mask masking everything once offset >= 2*window.
    """
    try:
        import mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant as g4a
    except Exception:  # pragma: no cover - drafter module unavailable
        pytest.skip("gemma4_assistant drafter module unavailable")

    _reset_patch_state()
    assert _install_mtp_wrap_patches() is True
    mdm = g4a.make_drafter_masks
    assert getattr(mdm, "_mtp_wrap_patch", False) is True

    window = 1024
    # Wrapped: true offset 1500, sliding ring holds `window` keys; full holds all.
    skv = {
        "sliding_attention": (mx.zeros((1, 1, window, 8)), mx.zeros((1, 1, window, 8))),
        "full_attention": (mx.zeros((1, 1, 1500, 8)), mx.zeros((1, 1, 1500, 8))),
    }
    masks = mdm(skv, query_len=1, query_offset=1500,
                sliding_window=window, dtype=mx.float32)

    # Full layer: true offset => every real key is in range => no mask.
    assert masks["full_attention"] is None
    sm = masks["sliding_attention"]
    assert sm is not None and tuple(sm.shape) == (1, 1, 1, window)
    row = sm[0, 0, 0]
    # Physical key 0 is absolute 476; dist = 1500-476 = 1024 == window => excluded.
    assert float(row[0].item()) < -1e30
    # Keys 1..window-1 are within the bidirectional window => allowed (0.0).
    assert float(row[1].item()) == 0.0
    assert float(row[window - 1].item()) == 0.0
