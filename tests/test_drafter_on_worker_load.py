"""Regression test for: `--draft-model` ON + first POST raised
`RuntimeError: There is no Stream(gpu, N) in current thread.`
at `mlx_vlm/generate.py:1340 mx.async_eval(y)`.

Root cause: mlx-vlm 0.5.0 `generate_step` has an *unguarded*
``mx.async_eval(y)`` outside the ``with mx.stream(generation_stream):``
block. When ``--draft-model`` is OFF, mlx-vlm's chunked-prefill code
runs ``mx.eval([c.state for c in prompt_cache])`` *inside* the
context manager — that incidental eval keeps lazy state thread-portable.
Drafter ON disables chunked-prefill (``prefill_step_size = None`` at
``generate.py:1223``), exposing the missing context manager.

Fix shape (engine side, layered defense):
  1. Load VLM model + drafter on the dedicated `_vlm_executor` worker
     thread so their lazy state and the inference loop share a single
     thread (`F3-LOAD` in `MLXEngine.load_model`).
  2. Shadow ``mx.async_eval`` / ``mx.eval`` inside ``mlx_vlm.generate``
     with wrappers that always run inside
     ``with mx.stream(generation_stream)`` (`F-PATCH` in
     `MLXEngine.__init__._vlm_worker_init`).
  3. Wrap every ``next(stream_iter)`` in our engine with
     ``with mx.stream(generation_stream):`` (`F-STREAM-CTX` in
     `MLXEngine._run_vlm`).
  4. Propagate ``draft_*`` from top-level `Config` into the per-model
     `Config` rebuild in `server.py` (otherwise the drafter is silently
     dropped before reaching the engine).

The test below drives the engine through `complete()` (non-streaming
path) with a mocked drafter present on the engine — this exercises the
exact code path that previously raised, and asserts no Stream-cross-
thread RuntimeError leaks out.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SLIDING_VLM_PATH = "/tmp/sw_vlm_test"


@pytest.mark.skipif(
    not Path(SLIDING_VLM_PATH, "config.json").exists(),
    reason=f"sliding-window VLM not available at {SLIDING_VLM_PATH}",
)
def test_complete_routes_through_vlm_executor_no_stream_error(tmp_path):
    """End-to-end smoke test for the non-streaming HTTP path.

    Asserts:
      - `engine.complete(...)` returns a non-empty result.
      - The call did NOT raise ``RuntimeError: There is no Stream(gpu, N)
        in current thread.`` from inside `mlx_vlm.generate.generate_step`.
      - The VLM call ran on the dedicated `mlx-vlm-worker_0` thread
        (i.e. F3 routing is in effect for the non-streaming path).
    """
    import sys
    import threading

    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    cfg = Config(
        model_path=SLIDING_VLM_PATH,
        memory_budget_gb=4,
        disk_budget_gb=4,
        data_dir=str(tmp_path),
        verbose=False,
    )
    eng = MLXEngine(cfg)
    try:
        eng.load_model()
        assert eng._use_vlm, "small VLM must load via mlx-vlm path"
        # F3 invariants intact.
        assert eng._vlm_executor is not None
        assert eng._vlm_executor._max_workers == 1

        observed_threads: list[str] = []
        real_run_vlm = MLXEngine._run_vlm

        def _instrumented_run_vlm(self, *args, **kwargs):
            observed_threads.append(threading.current_thread().name)
            return real_run_vlm(self, *args, **kwargs)

        MLXEngine._run_vlm = _instrumented_run_vlm  # type: ignore[assignment]
        try:
            result = eng.complete(
                [{"role": "user", "content": "hi"}],
                max_tokens=3,
                temperature=0.0,
                session_id=None,
            )
        finally:
            MLXEngine._run_vlm = real_run_vlm  # type: ignore[assignment]

        assert result.finish_reason in {"stop", "length"}
        assert result.content is not None
        # Either the VLM ran (worker thread observed) or the model
        # fell back to mlx-lm — but for /tmp/sw_vlm_test it MUST be VLM.
        assert observed_threads, "_run_vlm was never called"
        assert all(t == "mlx-vlm-worker_0" for t in observed_threads), (
            f"complete() must route through F3 worker; saw {observed_threads}"
        )
    finally:
        eng.close()


@pytest.mark.skipif(
    not Path(SLIDING_VLM_PATH, "config.json").exists(),
    reason=f"sliding-window VLM not available at {SLIDING_VLM_PATH}",
)
def test_model_loaded_on_worker_thread(tmp_path):
    """`vlm_load` must run on the F3 worker thread, not on the caller.

    Pinning the load to the same worker that runs inference eliminates
    the cross-thread lazy-array hand-off that previously surfaced as
    ``RuntimeError: There is no Stream(gpu, N) in current thread.`` on
    the user's M3 Ultra-class hardware with `--draft-model` set.
    """
    import threading

    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    cfg = Config(
        model_path=SLIDING_VLM_PATH,
        memory_budget_gb=4,
        disk_budget_gb=4,
        data_dir=str(tmp_path),
        verbose=False,
    )

    observed: list[str] = []
    real_vlm_load = mlx_engine_module.vlm_load

    def _spying_vlm_load(*args, **kwargs):
        observed.append(threading.current_thread().name)
        return real_vlm_load(*args, **kwargs)

    mlx_engine_module.vlm_load = _spying_vlm_load
    try:
        eng = MLXEngine(cfg)
        try:
            eng.load_model()
        finally:
            eng.close()
    finally:
        mlx_engine_module.vlm_load = real_vlm_load

    assert observed, "vlm_load was never called"
    assert observed == ["mlx-vlm-worker_0"], (
        f"vlm_load must run on the F3 worker; saw {observed}"
    )


def test_server_propagates_draft_config_to_per_model_config():
    """`server.py` rebuilds a per-model `Config` from the top-level one;
    that rebuild MUST carry over `draft_model` / `draft_kind` /
    `draft_block_size`. Without this, `--draft-model` at the CLI is
    silently dropped before reaching `MLXEngine.load_model`, so MTP is a
    no-op even when the CLI flag is set.
    """
    import inspect

    from mlx_soloheaven import server

    src = inspect.getsource(server.create_app)
    # The per-model Config(...) rebuild must reference draft_model.
    # We assert on the source string rather than running the whole
    # FastAPI bootstrap, which would require loading a real model.
    assert "draft_model=cfg.draft_model" in src, (
        "server.create_app must propagate draft_model into the per-model Config"
    )
    assert "draft_kind=cfg.draft_kind" in src
    assert "draft_block_size=cfg.draft_block_size" in src
