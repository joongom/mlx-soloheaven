"""F3 architecture: persistent mlx-vlm worker thread.

Previously (failed fix v2) we monkey-patched
``mlx_vlm.generate.generation_stream`` per-call inside ``_run_vlm``. That
addressed the symptom but did not fully fix the user's
``Stream(gpu, 1) not in current thread`` error on real workloads.

F3 architecture (current): the engine owns a single persistent
``ThreadPoolExecutor(max_workers=1)`` whose worker installs a fresh
``mx.new_thread_local_stream`` ONCE during ``MLXEngine.__init__``. All
``_run_vlm`` calls execute on that one thread, so mlx-vlm's module-global
stream slot stays consistent across every request and every cache reuse.

These tests assert the F3 invariants on a bare engine (no model load):

- The executor exists and has exactly 1 worker.
- ``_run_vlm`` invocations through the executor see a stable
  ``generation_stream`` identity (set during init, unchanged thereafter).
- Multiple submissions land on the SAME OS thread.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

# `mlx_vlm.__init__` rebinds the `generate` attribute to a function,
# so `import mlx_vlm.generate as _mvg` resolves to the function. Pull
# the real submodule from sys.modules instead.
import mlx_vlm.generate  # noqa: F401 — ensures sys.modules entry exists
_mvg = sys.modules["mlx_vlm.generate"]

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_vlm.generate import PromptCacheState


def _bare_vlm_engine() -> MLXEngine:
    """Construct a fully-initialised engine WITHOUT loading a model.

    We need the real __init__ to spin up the dedicated VLM executor and
    its one-shot generation_stream monkey-patch.
    """
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


def test_engine_has_single_worker_vlm_executor():
    """F3 invariant #1: max_workers == 1 on the dedicated executor."""
    eng = _bare_vlm_engine()
    try:
        assert eng._vlm_executor is not None
        assert eng._vlm_executor._max_workers == 1
    finally:
        eng.close()


def test_run_vlm_runs_on_persistent_worker_thread(monkeypatch):
    """F3 invariant #2: every _run_vlm call lands on the same thread
    and observes the same generation_stream identity that was installed
    during engine init.
    """
    eng = _bare_vlm_engine()
    try:
        seen_threads: list[str] = []
        seen_stream_ids: list[int] = []
        errors: list[BaseException] = []

        def _fake_stream(*_args, **_kwargs):
            seen_threads.append(threading.current_thread().name)
            seen_stream_ids.append(id(_mvg.generation_stream))
            return iter([])

        monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)

        def _one_call():
            cache_state = PromptCacheState()
            try:
                gen = eng._run_vlm(
                    cache_state=cache_state,
                    prompt_token_ids=[1, 2, 3],
                    max_tokens=4,
                    temperature=0.0,
                    top_p=1.0,
                    min_p=0.0,
                    top_k=0,
                    logits_processors=None,
                    session_id="s",
                    total_prompt_tokens=3,
                )
                list(gen)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        # Submit two calls through the executor, mirroring how generate_stream_async
        # routes _run.
        fut1 = eng._vlm_executor.submit(_one_call)
        fut2 = eng._vlm_executor.submit(_one_call)
        fut1.result(timeout=10)
        fut2.result(timeout=10)

        assert not errors, f"_run_vlm raised: {errors!r}"
        assert len(seen_threads) == 2
        # Both calls must land on the SAME thread.
        assert seen_threads[0] == seen_threads[1] == "mlx-vlm-worker_0", (
            f"expected mlx-vlm-worker_0 twice, got {seen_threads}"
        )
        # Stream identity must be stable across both calls (installed once at init).
        assert seen_stream_ids[0] == seen_stream_ids[1], (
            f"generation_stream identity changed between calls: {seen_stream_ids}"
        )
    finally:
        eng.close()
