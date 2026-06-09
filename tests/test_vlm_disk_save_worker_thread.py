"""VLM disk-save routing: materialize + serialize on the worker thread.

Regression test for the P2 threading bug where `_save_session_to_disk` ran
`save_prompt_cache` (which must `mx.eval` lazy cache tensors) on a thread
WITHOUT the _vlm_executor worker's `generation_stream`, raising
`RuntimeError: There is no Stream(gpu, 1) in current thread.`

The fix: on the VLM path, route both `_eval_cache` (materialize) and
`save_prompt_cache` through `self._vlm_executor.submit(...).result()` so they
run on the thread that owns the cache tensors' stream. On the legacy mlx-lm
path (`_use_vlm=False`), run inline.

No real model / GPU / IO: we monkeypatch the module-level `save_prompt_cache`
and stub `_eval_cache`, and use a lightweight executor stub that records calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine


class _RecordingExecutor:
    """Minimal ThreadPoolExecutor stand-in that runs submitted fns inline.

    Records every `submit` call and exposes the immediate result via a tiny
    future-like object so `.result(timeout=...)` works like the real thing.
    """

    def __init__(self):
        self.submit_calls = 0
        self.shutdown_raises = False  # set True to simulate a shut-down executor

    def submit(self, fn, *args, **kwargs):
        if self.shutdown_raises:
            raise RuntimeError("cannot schedule new futures after shutdown")
        self.submit_calls += 1

        class _Fut:
            def __init__(self, f, a, k):
                self._exc = None
                try:
                    self._val = f(*a, **k)
                except Exception as e:  # noqa: BLE001 — mirror real future behavior
                    self._exc = e

            def result(self, timeout=None):
                if self._exc is not None:
                    raise self._exc
                return self._val

        return _Fut(fn, args, kwargs)


def _make_engine(tmp_path, *, use_vlm: bool, executor):
    """Bare MLXEngine shell wired with just what _save_session_to_disk needs."""
    eng = MLXEngine.__new__(MLXEngine)
    cfg = SimpleNamespace(cache_dir=str(tmp_path), disk_budget_gb=1.0)
    eng.cfg = cfg
    eng._use_vlm = use_vlm
    eng._vlm_executor = executor
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    return eng


def _make_session():
    cache_state = SimpleNamespace(cache=[object()], token_ids=[1, 2, 3])
    return SimpleNamespace(
        cache_state=cache_state,
        messages=[{"role": "user", "content": "hi"}],
        total_cache_tokens=3,
        last_used=1234.0,
    )


@pytest.fixture
def patched_save(tmp_path, monkeypatch):
    """Patch save_prompt_cache (write a real file so getsize works) and
    _eval_cache (no-op), recording invocations."""
    calls = SimpleNamespace(save=0, eval=0)

    def fake_save(path, cache, metadata=None):
        calls.save += 1
        with open(path, "wb") as f:
            f.write(b"x")  # non-empty so os.path.getsize succeeds

    def fake_eval(cache):
        calls.eval += 1

    monkeypatch.setattr(mlx_engine_module, "save_prompt_cache", fake_save)
    monkeypatch.setattr(MLXEngine, "_eval_cache", staticmethod(fake_eval))
    return calls


def test_vlm_path_routes_save_through_executor(tmp_path, patched_save):
    ex = _RecordingExecutor()
    eng = _make_engine(tmp_path, use_vlm=True, executor=ex)

    ok = eng._save_session_to_disk("sid-vlm", _make_session())

    assert ok is True
    assert ex.submit_calls == 1, "VLM path must route the save through executor.submit"
    assert patched_save.save == 1, "save_prompt_cache should run exactly once"
    assert patched_save.eval == 1, "_eval_cache should materialize on the worker"
    assert "sid-vlm" in eng._disk_session_ids


def test_reentrant_worker_save_runs_inline_no_self_submit(tmp_path, patched_save):
    """Re-entrancy guard: when _save_session_to_disk is ALREADY executing on the
    single _vlm_executor worker thread (e.g. post-generation eviction driven
    inside generate_stream's finally), it must run the save INLINE and never
    submit back to the one-worker pool — doing so would deadlock against itself
    until the 60s .result() timeout. We simulate by setting _vlm_worker_ident to
    the CURRENT thread's ident; submit() would record a call (and a real pool
    would block), so we assert submit was NOT used."""
    import threading

    ex = _RecordingExecutor()
    eng = _make_engine(tmp_path, use_vlm=True, executor=ex)
    # Pretend THIS thread is the dedicated worker.
    eng._vlm_worker_ident = threading.get_ident()

    ok = eng._save_session_to_disk("sid-reentrant", _make_session())

    assert ok is True
    assert ex.submit_calls == 0, "re-entrant worker save must NOT self-submit (deadlock)"
    assert patched_save.save == 1, "save must still happen inline"
    assert patched_save.eval == 1
    assert "sid-reentrant" in eng._disk_session_ids


def test_legacy_path_does_not_use_executor(tmp_path, patched_save):
    ex = _RecordingExecutor()
    eng = _make_engine(tmp_path, use_vlm=False, executor=ex)

    ok = eng._save_session_to_disk("sid-legacy", _make_session())

    assert ok is True
    assert ex.submit_calls == 0, "legacy mlx-lm path must NOT use the executor"
    assert patched_save.save == 1, "save_prompt_cache should run inline"
    assert patched_save.eval == 1


def test_vlm_path_with_none_executor_runs_inline(tmp_path, patched_save):
    """Guard: _vlm_executor None (e.g. teardown) falls back to inline save."""
    eng = _make_engine(tmp_path, use_vlm=True, executor=None)

    ok = eng._save_session_to_disk("sid-none", _make_session())

    assert ok is True
    assert patched_save.save == 1
    assert patched_save.eval == 1


def test_shutdown_executor_falls_back_to_inline(tmp_path, patched_save):
    """If submit raises RuntimeError (executor shut down), save still happens."""
    ex = _RecordingExecutor()
    ex.shutdown_raises = True
    eng = _make_engine(tmp_path, use_vlm=True, executor=ex)

    ok = eng._save_session_to_disk("sid-shutdown", _make_session())

    assert ok is True
    assert patched_save.save == 1, "shutdown fallback must still serialize inline"
    assert patched_save.eval == 1


def test_empty_array_error_is_permanent_skip_through_executor(tmp_path, monkeypatch):
    """An 'empty array' error raised inside the worker must propagate out of
    .result() and be classified as a permanent skip (return False)."""
    def boom(path, cache, metadata=None):
        raise ValueError("cannot serialize empty array")

    monkeypatch.setattr(mlx_engine_module, "save_prompt_cache", boom)
    monkeypatch.setattr(MLXEngine, "_eval_cache", staticmethod(lambda cache: None))

    ex = _RecordingExecutor()
    eng = _make_engine(tmp_path, use_vlm=True, executor=ex)

    ok = eng._save_session_to_disk("sid-empty", _make_session())

    assert ok is False, "empty-array failure must be a permanent skip, not a raise"
    assert ex.submit_calls == 1


def test_unexpected_error_through_executor_reraises(tmp_path, monkeypatch):
    """A non-empty-array error raised inside the worker must re-raise (so the
    flush loop re-queues the session), not be swallowed as a permanent skip."""
    def boom(path, cache, metadata=None):
        raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

    monkeypatch.setattr(mlx_engine_module, "save_prompt_cache", boom)
    monkeypatch.setattr(MLXEngine, "_eval_cache", staticmethod(lambda cache: None))

    ex = _RecordingExecutor()
    eng = _make_engine(tmp_path, use_vlm=True, executor=ex)

    with pytest.raises(RuntimeError, match="no Stream"):
        eng._save_session_to_disk("sid-unexpected", _make_session())
