"""Eager-sampler PRNG-advance tests — targets the repetition-loop root cause.

ROOT CAUSE (proven, fixed by ``_make_eager_sampler``): the sampler that
soloheaven used to build via ``mlx_lm.sample_utils.make_sampler`` does its
categorical draw + top_p/min_p/top_k filters under
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``.
That compiled-with-random-state graph is the ONLY thing that advances the
global PRNG between decode steps, and it ONLY advances on the thread it was
first bound to. SoloHeaven runs per-request generation on a non-main (daemon)
worker thread, so on that thread ``mx.random.state`` FREEZES: every token
samples with the SAME key -> the SAME token repeats -> a degenerate line loop.

These tests are hermetic (no model, no weights, no GPU dependence beyond a
fixed logits vector). They build the NEW eager sampler and draw repeatedly
ON A DAEMON THREAD, asserting the PRNG actually advances (unique tokens > 1).
The OLD ``make_sampler`` would yield exactly 1 unique token in this setup.
"""

from __future__ import annotations

import threading

import mlx.core as mx

from mlx_soloheaven.engine.mlx_engine import _make_eager_sampler


# A peaked-but-not-degenerate logits vector: a handful of plausible tokens
# above a sea of very-low-probability ones. With temp=1.0/top_p=0.95/top_k=64
# this leaves several competitive tokens, so a *working* RNG must produce
# several distinct draws across 30 samples; a FROZEN RNG repeats one token.
def _peaked_logits(vocab: int = 512, n_peaks: int = 8) -> mx.array:
    base = mx.full((1, vocab), -20.0)
    # Give the first n_peaks tokens a small, similar set of high logits so the
    # post-filter categorical genuinely has multiple live candidates.
    peak_idx = mx.arange(n_peaks)
    peak_val = mx.array([3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6][:n_peaks])
    base = mx.put_along_axis(
        base, peak_idx.reshape(1, -1), peak_val.reshape(1, -1), axis=-1
    )
    # Materialize on the building (main) thread so the worker thread doesn't
    # inherit a deferred build graph bound to a different stream.
    mx.eval(base)
    return base


def _draw_n_on_thread(sampler, logits, n: int) -> list[int]:
    """Draw ``n`` tokens with ``sampler`` from a fresh DAEMON thread and return
    the sampled token ids as plain ints.

    The worker registers a thread-local MLX stream and runs inside it, exactly
    like soloheaven's production worker (``_vlm_worker_init`` /
    ``mx.new_thread_local_stream`` + ``with mx.stream(...)``); a daemon thread
    that never registered a stream raises "no Stream(gpu, 0) in current
    thread" on ``mx.eval``. This stream setup is orthogonal to the PRNG bug:
    ``mx.random.state`` still freezes on this thread under the OLD
    make_sampler, so the test faithfully reproduces the repetition-loop
    scenario.
    """
    out: list[int] = []
    err: list[BaseException] = []

    def _worker():
        try:
            stream = mx.new_stream(mx.default_device())
            with mx.stream(stream):
                for _ in range(n):
                    tok = sampler(logits)
                    out.append(int(tok.item()))
        except BaseException as exc:  # surface thread errors to the test
            err.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive(), "sampler worker thread did not finish in time"
    if err:
        raise err[0]
    return out


def test_eager_sampler_advances_rng_on_daemon_thread():
    """THE root-cause regression test.

    Build the eager sampler (temp=1.0, top_p=0.95, top_k=64) and draw 30 tokens
    on a DAEMON thread. With the OLD make_sampler the compiled+random-state
    categorical would FREEZE on the worker thread -> exactly 1 unique token.
    The eager sampler must advance the PRNG -> several unique tokens.
    """
    sampler = _make_eager_sampler(temp=1.0, top_p=0.95, top_k=64)
    logits = _peaked_logits()

    draws = _draw_n_on_thread(sampler, logits, n=30)
    unique = set(draws)

    assert len(draws) == 30
    # Strict assertion the root cause demands: RNG must advance (> 1).
    assert len(unique) > 1, (
        f"PRNG frozen on daemon thread — only {len(unique)} unique token(s) in "
        f"30 draws ({sorted(unique)}). This is exactly the repetition-loop bug."
    )
    # Stronger expectation given multiple live candidates: ideally >= ~5.
    assert len(unique) >= 5, (
        f"Expected >=5 unique tokens with a working RNG, got {len(unique)} "
        f"({sorted(unique)})."
    )


def test_eager_sampler_temp_zero_is_argmax():
    """temp==0 must short-circuit to argmax (byte-identical to LM Studio /
    upstream greedy), independent of thread."""
    sampler = _make_eager_sampler(temp=0.0)
    logits = _peaked_logits()
    expected = int(mx.argmax(logits, axis=-1).item())

    # Main-thread greedy.
    assert int(sampler(logits).item()) == expected

    # Greedy on a daemon thread is also deterministic and identical.
    draws = _draw_n_on_thread(sampler, logits, n=5)
    assert draws == [expected] * 5


def test_eager_sampler_top_k_one_is_deterministic():
    """top_k=1 collapses the distribution to a single token, so even with
    temp>0 every draw must be that argmax token (sanity check that the filter
    math is wired correctly)."""
    sampler = _make_eager_sampler(temp=1.0, top_k=1)
    logits = _peaked_logits()
    expected = int(mx.argmax(logits, axis=-1).item())

    draws = _draw_n_on_thread(sampler, logits, n=10)
    assert set(draws) == {expected}
