"""Cross-thread KV-cache evaluation regression test.

This is the third RCA round for the user's error:

    RuntimeError: There is no Stream(gpu, 1) in current thread.
      File "mlx_vlm/generate.py", line 1340, in generate_step
        mx.async_eval(y)

Two prior fixes (monkey-patching ``mlx_vlm.generate.generation_stream`` on
the worker thread) were insufficient. Empirical probing — see RCA notes —
showed the monkey-patch *does* reach ``generate_step``'s globals. The real
root cause is in the lifecycle of ``cache_state.cache``:

    1. Request N spawns a worker thread.
    2. mlx-vlm's _step schedules ops inside
       ``with mx.stream(generation_stream): ...``. The resulting lazy
       ``mx.array`` objects in the KV cache carry an internal reference
       to that thread-local stream slot.
    3. The worker thread exits.
    4. Request N+1 spawns a new worker thread, reuses the cached arrays.
    5. When mlx-vlm calls ``mx.async_eval(y)`` (line 1340) — or any other
       eval that touches the cached arrays — MLX detects that the
       referenced stream slot does not exist in the current thread and
       raises ``RuntimeError: There is no Stream(gpu, N) in current thread.``

This test reproduces the exact failure mode with mlx primitives only (no
mlx-vlm model load required) — both the failing baseline pattern (skipped
unless ``MLX_REPRO_BASELINE=1``) and the engine's post-stream
``_eval_cache`` fix path.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

import mlx.core as mx
import mlx_vlm  # noqa: F401 — ensures sys.modules["mlx_vlm.generate"] exists

_mvg = sys.modules["mlx_vlm.generate"]


def _patch_generation_stream():
    """Mirror engine/_run_vlm: register a fresh worker-thread stream."""
    new_s = mx.new_thread_local_stream(mx.default_device())
    _mvg.generation_stream = new_s


@pytest.mark.skipif(
    os.environ.get("MLX_REPRO_BASELINE") != "1",
    reason="Run with MLX_REPRO_BASELINE=1 to demonstrate the failing baseline "
           "(no _eval_cache after worker). Disabled by default so CI stays green.",
)
def test_repro_cross_thread_unevaluated_cache_fails():
    """Without the fix, request 2 fails with the user's exact error."""
    shared = {"cache": None}

    def request_1():
        _patch_generation_stream()
        with mx.stream(_mvg.generation_stream):
            a = mx.array([1.0]) * 2
            b = a + 3
        # No mx.eval — simulates the pre-fix bug.
        shared["cache"] = [a, b]

    err: list[BaseException] = []

    def request_2():
        _patch_generation_stream()
        try:
            with mx.stream(_mvg.generation_stream):
                _, cached_b = shared["cache"]
                result = cached_b * 5
                mx.async_eval(result)
            mx.eval(result)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t1 = threading.Thread(target=request_1); t1.start(); t1.join()
    t2 = threading.Thread(target=request_2); t2.start(); t2.join()

    assert err, "Baseline should fail without post-stream _eval_cache fix"
    assert isinstance(err[0], RuntimeError)
    assert "no Stream" in str(err[0])
    assert "current thread" in str(err[0])


def test_post_stream_eval_makes_kv_cache_thread_portable():
    """With ``mx.eval`` called before the worker exits, request 2 succeeds.

    This mirrors the fix in ``generate_stream``: after writing back
    ``cache_state.cache``, call ``MLXEngine._eval_cache(cache_state.cache)``
    so all lazy arrays are fully materialized on the producing thread.
    """
    shared = {"cache_arrays": None}

    def request_1():
        _patch_generation_stream()
        with mx.stream(_mvg.generation_stream):
            a = mx.array([1.0]) * 2
            b = a + 3
            c = b @ mx.array([[2.0]])  # heavier op
        # THE FIX: fully evaluate before exiting the worker thread.
        mx.eval(a, b, c)
        shared["cache_arrays"] = [a, b, c]

    err: list[BaseException] = []
    result_val: list[float] = []

    def request_2():
        _patch_generation_stream()
        try:
            with mx.stream(_mvg.generation_stream):
                _, cached_b, cached_c = shared["cache_arrays"]
                result = cached_b * 5
                mx.async_eval(result)
            mx.eval(result)
            mx.eval(cached_c)
            result_val.append(result.tolist())
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t1 = threading.Thread(target=request_1); t1.start(); t1.join()
    t2 = threading.Thread(target=request_2); t2.start(); t2.join()

    assert not err, f"request_2 must not raise after eval-on-producer fix, got: {err!r}"
    assert result_val, "request_2 must have produced a value"


def test_engine_eval_cache_handles_keys_values_arrays():
    """``MLXEngine._eval_cache`` must materialize lazy arrays inside
    cache objects that expose ``keys`` / ``values`` / ``state``.

    Smoke-test with a synthetic cache-like object and verify that the
    static method does not crash on the duck-typed surface used by
    mlx-vlm's prompt_cache (KVCache, RotatingKVCache, ArraysCache).
    """
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    class FakeKVCache:
        def __init__(self):
            with mx.stream(mx.default_stream(mx.default_device())):
                self.keys = mx.array([1.0]) * 2  # lazy
                self.values = mx.array([2.0]) * 3  # lazy
                self.state = None

    class FakeArraysCache:
        def __init__(self):
            with mx.stream(mx.default_stream(mx.default_device())):
                self.keys = None
                self.values = None
                self.state = [mx.array([3.0]) + 1, mx.array([4.0]) + 1]

    cache = [FakeKVCache(), FakeArraysCache()]

    # Force the bug-prone "another thread" scenario.
    err: list[BaseException] = []

    def producer():
        # Build a fresh cache on a worker thread, then call _eval_cache
        # which should materialize so the next thread can read.
        worker_cache = [FakeKVCache(), FakeArraysCache()]
        try:
            MLXEngine._eval_cache(worker_cache)
        except BaseException as e:  # noqa: BLE001
            err.append(("producer", e))
            return
        # Hand off to consumer thread for sanity-check
        shared["c"] = worker_cache

    shared: dict = {}
    t = threading.Thread(target=producer); t.start(); t.join()
    assert not err, f"_eval_cache raised on producer thread: {err!r}"

    def consumer():
        try:
            # Read materialized values from a different thread — should work.
            wc = shared["c"]
            _ = wc[0].keys.tolist()
            _ = wc[0].values.tolist()
            _ = [x.tolist() for x in wc[1].state]
        except BaseException as e:  # noqa: BLE001
            err.append(("consumer", e))

    t2 = threading.Thread(target=consumer); t2.start(); t2.join()
    assert not err, f"cross-thread read after _eval_cache failed: {err!r}"


# --- End-to-end VLM worker-thread integration test --------------------------

from pathlib import Path
SMALL_VLM_PATH = "/tmp/smolvlm_test"
SLIDING_VLM_PATH = "/tmp/sw_vlm_test"  # gemma-3-4b-it-4bit, sliding_window=1024


@pytest.mark.skipif(
    not Path(SMALL_VLM_PATH, "config.json").exists(),
    reason=f"small VLM not available at {SMALL_VLM_PATH}",
)
def test_two_request_worker_thread_no_stream_error(tmp_path):
    """End-to-end repro of user's `Stream(gpu, 1) not in current thread` bug.

    Each request runs `generate_stream` on a fresh `threading.Thread` —
    mirroring `MLXEngine._run` in `generate_stream_async`. The shared
    SessionState's KV cache is produced on thread #1 and consumed on
    thread #2; without the F-EVAL fix this crashes inside mlx-vlm's
    `mx.async_eval(y)` in `generate_step`.
    """
    import asyncio
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    cfg = Config(
        model_path=SMALL_VLM_PATH,
        memory_budget_gb=4,
        disk_budget_gb=4,
        data_dir=str(tmp_path),
        verbose=False,
    )
    eng = MLXEngine(cfg)
    eng.load_model()
    assert eng._use_vlm, "SmolVLM must load via mlx-vlm path to exercise the bug"

    sid = "integration_thread_test"

    async def _one_request(messages, max_tokens=8):
        chunks: list[str] = []
        async for r in eng.generate_stream_async(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id=sid,
        ):
            if r.text:
                chunks.append(r.text)
        return "".join(chunks)

    msgs1 = [{"role": "user", "content": "Hello"}]
    text1 = asyncio.run(_one_request(msgs1, max_tokens=6))
    assert text1.strip(), "first request produced no text"

    msgs2 = msgs1 + [
        {"role": "assistant", "content": text1},
        {"role": "user", "content": "Bye"},
    ]
    text2 = asyncio.run(_one_request(msgs2, max_tokens=6))
    assert text2.strip(), "second request produced no text"


# --- Sliding-window VLM (RotatingKVCache) integration test ------------------
# Same shape as the SmolVLM test above but exercises the RotatingKVCache code
# path that the user's Gemma 4 31B 8bit workload hits. SmolVLM uses plain
# KVCache only — it doesn't cover the wrap/aliasing surface that may
# interact with the F-EVAL fix. gemma-3-4b-it-4bit is the smallest
# mlx-vlm-supported model whose `make_cache()` returns RotatingKVCache
# for the sliding layers (see mlx_vlm.models.gemma3.language).
#
# IMPORTANT: on Apple M5 Max + mlx==0.31.2 + mlx-vlm==0.5.0, the underlying
# `Stream(gpu, N) not in current thread` error from the user's M3 Ultra
# trace did NOT reproduce empirically — even with `_eval_cache` monkey-
# patched to no-op across 5 consecutive cache-reuse turns (817 reused
# tokens through RotatingKVCache). The MLX primitive-level repro
# (`test_repro_cross_thread_unevaluated_cache_fails` under
# MLX_REPRO_BASELINE=1) still demonstrates the underlying behavior, so we
# keep the F-EVAL fix in place defensively. This test exists to keep the
# RotatingKVCache + worker-thread + cache-reuse path under regression
# coverage; failure here would indicate the F-EVAL fix REGRESSED on
# sliding-window models.


@pytest.mark.skipif(
    not Path(SLIDING_VLM_PATH, "config.json").exists(),
    reason=f"sliding-window VLM not available at {SLIDING_VLM_PATH}",
)
def test_sliding_window_vlm_worker_thread_cache_reuse(tmp_path):
    """Multi-turn worker-thread cache reuse on a RotatingKVCache model.

    Drives the engine through five `generate_stream_async` calls on the
    same session_id with cache reuse. Each call runs on a fresh
    `threading.Thread`. The F-EVAL fix (engine.mlx_engine line ~1730)
    must keep the cache thread-portable across all transitions.
    """
    import asyncio, gc
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    cfg = Config(
        model_path=SLIDING_VLM_PATH,
        memory_budget_gb=8,
        disk_budget_gb=8,
        data_dir=str(tmp_path),
        verbose=False,
    )
    eng = MLXEngine(cfg)
    eng.load_model()
    assert eng._use_vlm, "gemma3 must load via mlx-vlm path"

    # Sanity: confirm RotatingKVCache is in fact used by this model.
    lm = eng._language_model
    cache_objs = lm.make_cache()
    cache_types = {type(c).__name__ for c in cache_objs}
    assert "RotatingKVCache" in cache_types, (
        f"expected RotatingKVCache in gemma3 make_cache, got {cache_types}"
    )

    sid = "sliding_window_thread_test"

    async def _one(messages, max_tokens=8):
        out = []
        async for r in eng.generate_stream_async(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id=sid,
        ):
            if r.text:
                out.append(r.text)
        return "".join(out)

    msgs = [{"role": "user", "content": "Hello"}]
    for i in range(5):
        text = asyncio.run(_one(msgs, max_tokens=8))
        assert text.strip(), f"turn {i} produced no text"
        msgs = msgs + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Q{i+2}: tell me a short fact."},
        ]
        gc.collect()

    # Cache must have grown through several worker threads without raising.
    s = eng._sessions.get(sid)
    assert s is not None
    assert s.total_cache_tokens > 0


@pytest.mark.skipif(
    not Path(SLIDING_VLM_PATH, "config.json").exists(),
    reason=f"sliding-window VLM not available at {SLIDING_VLM_PATH}",
)
def test_sliding_window_vlm_eval_cache_disabled_smoke(tmp_path, monkeypatch):
    """Smoke test: monkey-patch `_eval_cache` to no-op, exercise the same
    multi-turn worker-thread path, and document the empirical outcome.

    On Apple M5 Max + mlx==0.31.2 + mlx-vlm==0.5.0 the bug does NOT
    surface even with the fix disabled. This test passes as long as no
    OTHER exception is raised — and asserts that IF a `Stream(gpu, N) not
    in current thread` error IS raised, it must come from the fix
    actually being needed (which is what we want to verify on the user's
    hardware). On hardware where the bug doesn't reproduce, the test
    still serves as a smoke check.
    """
    import asyncio, gc
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    # Disable the F-EVAL fix.
    monkeypatch.setattr(
        MLXEngine, "_eval_cache",
        staticmethod(lambda *a, **k: None),
    )

    cfg = Config(
        model_path=SLIDING_VLM_PATH,
        memory_budget_gb=8,
        disk_budget_gb=8,
        data_dir=str(tmp_path),
        verbose=False,
    )
    eng = MLXEngine(cfg)
    eng.load_model()

    sid = "sliding_window_no_fix"

    async def _one(messages, max_tokens=6):
        out = []
        async for r in eng.generate_stream_async(
            messages, max_tokens=max_tokens, temperature=0.0,
            session_id=sid,
        ):
            if r.text:
                out.append(r.text)
        return "".join(out)

    msgs = [{"role": "user", "content": "Hello"}]
    stream_errors: list[BaseException] = []
    other_errors: list[BaseException] = []
    for i in range(3):
        try:
            text = asyncio.run(_one(msgs, max_tokens=6))
        except RuntimeError as e:
            if "no Stream" in str(e) and "current thread" in str(e):
                stream_errors.append(e)
                break
            other_errors.append(e)
            break
        msgs = msgs + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "More."},
        ]
        gc.collect()

    # We accept either outcome:
    #   (a) bug reproduces -> stream_errors non-empty (proves fix is needed)
    #   (b) bug doesn't reproduce on this hardware/version -> all turns succeed
    # We REJECT only unrelated errors.
    assert not other_errors, f"unexpected non-Stream error: {other_errors!r}"


# --- F3 architecture integration test ---------------------------------------
# After three prior fixes failed in real user testing for the
# `Stream(gpu, 1) not in current thread` error, the architecture pivot
# (F3) dedicates ALL mlx-vlm calls to a single persistent worker thread.
# This test validates the F3 invariants on the sliding-window
# RotatingKVCache model that mirrors the user's Gemma 4 31B workload.


@pytest.mark.skipif(
    not Path(SLIDING_VLM_PATH, "config.json").exists(),
    reason=f"sliding-window VLM not available at {SLIDING_VLM_PATH}",
)
def test_f3_persistent_worker_thread_three_consecutive_requests(tmp_path):
    """F3 invariants on a real RotatingKVCache model:

    1. ``_vlm_executor._max_workers == 1`` (one persistent worker).
    2. Three consecutive ``generate_stream_async`` calls each emit
       >= 1 token.
    3. All mlx-vlm calls land on EXACTLY ONE distinct OS thread.
    4. No ``Stream(gpu, N) not in current thread`` error raised.
    """
    import asyncio
    import gc
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    cfg = Config(
        model_path=SLIDING_VLM_PATH,
        memory_budget_gb=8,
        disk_budget_gb=8,
        data_dir=str(tmp_path),
        verbose=False,
    )
    eng = MLXEngine(cfg)
    try:
        eng.load_model()
        assert eng._use_vlm, "gemma3 must load via mlx-vlm path"
        # F3 invariant #1: single-worker executor exists.
        assert eng._vlm_executor is not None
        assert eng._vlm_executor._max_workers == 1

        # Instrument vlm_stream_generate to record which thread each call
        # actually runs on. Wrap (don't replace) the real call so we still
        # exercise the production code path end-to-end.
        observed_threads: list[str] = []
        observed_stream_ids: list[int] = []
        real_vlm_stream_generate = mlx_engine_module.vlm_stream_generate

        def _instrumented(*args, **kwargs):
            observed_threads.append(threading.current_thread().name)
            observed_stream_ids.append(
                id(sys.modules["mlx_vlm.generate"].generation_stream)
            )
            return real_vlm_stream_generate(*args, **kwargs)

        mlx_engine_module.vlm_stream_generate = _instrumented
        try:
            sid = "f3_persistent_worker"

            async def _one(messages, max_tokens=8):
                tokens = 0
                async for r in eng.generate_stream_async(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    session_id=sid,
                ):
                    if r.text:
                        tokens += 1
                return tokens

            stream_errors: list[BaseException] = []
            msgs = [{"role": "user", "content": "Hello"}]
            for i in range(3):
                try:
                    n = asyncio.run(_one(msgs, max_tokens=8))
                except RuntimeError as e:
                    if "no Stream" in str(e) and "current thread" in str(e):
                        stream_errors.append(e)
                    raise
                # Invariant #2: each request emitted at least one token.
                assert n >= 1, f"request {i} produced no tokens"
                msgs = msgs + [
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": f"Q{i+2}: a short fact."},
                ]
                gc.collect()
        finally:
            mlx_engine_module.vlm_stream_generate = real_vlm_stream_generate

        # Invariant #3: EXACTLY ONE distinct OS thread serviced mlx-vlm.
        distinct = set(observed_threads)
        assert len(distinct) == 1, (
            f"F3 violated: mlx-vlm calls hit {len(distinct)} threads "
            f"({distinct}); expected exactly 1"
        )
        assert next(iter(distinct)).startswith("mlx-vlm-worker"), (
            f"unexpected worker thread name: {distinct}"
        )
        # And the generation_stream identity was stable across all calls.
        assert len(set(observed_stream_ids)) == 1, (
            f"generation_stream identity drifted: {observed_stream_ids}"
        )

        # Invariant #4: no Stream-current-thread error along the way.
        assert not stream_errors, (
            f"F3 failed to prevent Stream cross-thread error: {stream_errors!r}"
        )
    finally:
        eng.close()
