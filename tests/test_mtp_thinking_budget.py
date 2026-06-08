"""Tests for the RUNAWAY-THINKING fix: enforce the thinking-budget cap during
MTP speculative decoding (not just the gated ``_plain_step``).

Two layers:

* Pure helper (``thinking.should_force_think_end`` + the incremental
  ``initial_think_state``/``advance_think_state``/``force_end_from_state``):
  history-derived, emitted-token-only thinking-state detection that matches
  ``ThinkingBudgetProcessor`` semantics WITHOUT the per-call mutation that
  over-counts rejected drafts in the verify block.

* MTP clone (``_patched_mtp_rounds_v2``): with the budget stash set, the clone
  must FORCE the think_end token once the budget is exceeded — exiting the
  thinking block instead of running to ``max_tokens``. The lm/draft model and
  ``_speculative_walk`` are stubbed (no real model loaded), mirroring
  ``tests/test_mtp_wrap_patches.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_vlm.generate  # noqa: F401 — populate sys.modules
import mlx_vlm.models.gemma4.language as _g4lang  # noqa: F401 — populate generate

from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import _install_mtp_wrap_patches
from mlx_soloheaven.engine.thinking import (
    ThinkingBudgetProcessor,
    should_force_think_end,
    initial_think_state,
    advance_think_state,
    force_end_from_state,
)


START = 100  # think_start_token (<|channel> analogue)
END = 200    # think_end_token   (<channel|> analogue)


# ---------------------------------------------------------------------------
# Pure helper: history-derived thinking-state + force-at-budget
# ---------------------------------------------------------------------------


def test_helper_noop_when_budget_off_or_no_end_token():
    """budget<=0 or think_end_token<0 => NEVER force (exact no-op)."""
    hist = [START, 1, 2, 3, 4, 5, 6, 7]
    assert should_force_think_end(
        hist, budget=0, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is False
    assert should_force_think_end(
        hist, budget=-1, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is False
    assert should_force_think_end(
        hist, budget=2, think_start_token=START, think_end_token=-1,
        model_family="gemma4",
    ) is False


def test_helper_gemma4_in_thinking_detection_and_count():
    """Gemma 4: in_thinking starts False, flips True after <|channel>, and the
    count is measured since the most recent think_start (start token = #1)."""
    # No think_start yet => not in thinking => never force.
    assert should_force_think_end(
        [5, 6, 7], budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is False

    # <|channel> emitted as last token => count == 1.
    assert should_force_think_end(
        [5, START], budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is True
    assert should_force_think_end(
        [5, START], budget=2, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is False  # count 1 < 2

    # <|channel> + one content token => count == 2.
    assert should_force_think_end(
        [5, START, 9], budget=2, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is True


def test_helper_stops_forcing_after_think_end_emitted():
    """Once think_end is in the (emitted) history, in_thinking is False =>
    no double-force."""
    hist = [START, 1, 2, 3, END]  # closed
    assert should_force_think_end(
        hist, budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is False
    # A NEW think block re-opens and re-counts.
    assert should_force_think_end(
        hist + [START], budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4",
    ) is True


def test_helper_chatml_starts_in_thinking():
    """ChatML generation begins INSIDE the thinking block (in_thinking=True),
    so the count accrues from the first emitted token."""
    # One emitted token, budget 1 => force.
    assert should_force_think_end(
        [42], budget=1, think_start_token=START, think_end_token=END,
        model_family="chatml",
    ) is True
    # Empty history, budget 1 => count 0 => no force yet.
    assert should_force_think_end(
        [], budget=1, think_start_token=START, think_end_token=END,
        model_family="chatml",
    ) is False


def _assert_helper_processor_lockstep(history, budget, family):
    """Drive a real generation loop and assert the history-derived helper and
    the stateful ``ThinkingBudgetProcessor`` make the SAME force decision at
    every step. ONE processor call per sampled token, each call seeing the
    cumulative emitted history-so-far (mirrors mlx-lm's generate_step contract).
    """
    proc = ThinkingBudgetProcessor(budget, END, START, family)
    for k in range(len(history) + 1):
        prefix = history[:k]
        helper_force = should_force_think_end(
            prefix, budget=budget, think_start_token=START,
            think_end_token=END, model_family=family,
        )
        logits = mx.zeros((1, 1000))
        out = proc(mx.array(prefix, dtype=mx.int32), logits)
        proc_force = float(out[0, END].item()) >= 1e8
        assert helper_force == proc_force, (
            f"mismatch at k={k} budget={budget} family={family}: "
            f"helper={helper_force} proc={proc_force}; prefix={prefix}"
        )


@pytest.mark.parametrize("budget", [1, 3, 5, 8])
def test_helper_matches_stateful_processor_gemma4(budget):
    """GEMMA 4 (the runaway target): the history-derived helper must fire at
    EXACTLY the same position as the stateful ``ThinkingBudgetProcessor`` driven
    step-by-step. This is the core equivalence guarantee — the helper replaces
    the stateful processor on the MTP path.

    For gemma4 the ``<|channel>`` think_start anchors the count, so prompt tokens
    before it are irrelevant and the two agree exactly. Single thinking block
    (the common case): START ... content, no close, no reopen.
    """
    history = [5, 6, START, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    _assert_helper_processor_lockstep(history, budget, "gemma4")


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_helper_matches_stateful_processor_multicycle(budget):
    """MULTI-BLOCK turn (START ... END ... START ...): the per-cycle
    ``ThinkingBudgetProcessor`` must RE-BUDGET each thinking block, closing on
    each ``think_end`` (resetting the counter) and re-opening on each later
    ``think_start`` — in lockstep with the MTP helper. This is the parity case
    the old "done after first close" latch broke: the latch left the 2nd block
    UNCAPPED on the plain path while the MTP helper capped it.
    """
    # Two thinking blocks separated by a closed gap, then content after a 2nd
    # close, then a 3rd re-open — exercises close→reset and reopen→re-budget.
    history = [5, START, 1, 2, 3, END, 7, 8, START, 9, 9, 9, END, 4, START, 1, 2]
    _assert_helper_processor_lockstep(history, budget, "gemma4")


@pytest.mark.parametrize("budget", [2, 4, 8])
def test_helper_chatml_within_one_of_processor(budget):
    """ChatML generation begins INSIDE thinking with no anchoring think_start,
    so the stateful processor counts the token being sampled (it increments on
    every call, including the first prompt-seeing one) while the history-derived
    helper counts only EMITTED tokens. They therefore agree to within one token
    at the budget boundary — the helper fires at most one token LATER, which is
    immaterial for a real budget of thousands and conservative (never early).
    ChatML is not the runaway target (that is gemma4); this documents the
    boundary behaviour rather than asserting exact parity.
    """
    history = [5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # no think_start: all content
    proc = ThinkingBudgetProcessor(budget, END, START, "chatml")
    for k in range(len(history) + 1):
        helper_force = should_force_think_end(
            history[:k], budget=budget, think_start_token=START,
            think_end_token=END, model_family="chatml",
        )
        logits = mx.zeros((1, 1000))
        out = proc(mx.array(history[:k], dtype=mx.int32), logits)
        proc_force = float(out[0, END].item()) >= 1e8
        # Helper never forces EARLIER than the processor (conservative), and is
        # at most one token late.
        if proc_force and not helper_force:
            # one-step-late case: helper must force on the very next step.
            nxt = should_force_think_end(
                history[: k + 1], budget=budget, think_start_token=START,
                think_end_token=END, model_family="chatml",
            ) if k < len(history) else True
            assert nxt, f"helper more than one token late at k={k}"
        if helper_force:
            assert proc_force, f"helper forced EARLIER than processor at k={k}"


def test_incremental_state_matches_scan():
    """The O(1) incremental updater must produce the same force decisions as the
    full-scan helper (this is what the clone uses per-position)."""
    history = [5, START, 1, 2, END, 7, START, 8, 9]
    budget = 2
    state = initial_think_state("gemma4")
    for k in range(len(history) + 1):
        scan = should_force_think_end(
            history[:k], budget=budget, think_start_token=START,
            think_end_token=END, model_family="gemma4",
        )
        incr = force_end_from_state(state, budget)
        assert scan == incr, f"k={k}: scan={scan} incr={incr}"
        if k < len(history):
            state = advance_think_state(
                state, history[k], think_start_token=START, think_end_token=END,
            )


# ---------------------------------------------------------------------------
# MTP clone: forces think_end during speculative decoding when budget exceeded
# ---------------------------------------------------------------------------


def _argmax_sampler(logits):
    """Greedy sampler over (1, vocab) or (1, bs, vocab) — returns (1,)/(1,bs)."""
    return mx.argmax(logits, axis=-1)


class _FakeLM:
    """Minimal gemma4 language_model stub for the MTP clone.

    Every verify position predicts the THINKING-CONTENT token (id 1) by default,
    so without budget enforcement the model would emit content forever (the
    runaway). When the clone forces END (logit set to 1e9) the argmax flips to
    END at that position. ``rollback_speculative_cache`` is a no-op.
    """

    def __init__(self, vocab=300, content_token=1):
        self.vocab = vocab
        self.content_token = content_token

    def __call__(self, x, cache=None, return_hidden=False, return_shared_kv=False):
        bs = int(x.shape[1])
        logits = mx.zeros((1, bs, self.vocab))
        # Bias every position toward the content token (greedy => content).
        logits = logits + mx.where(
            mx.arange(self.vocab) == self.content_token, 1.0, 0.0
        ).reshape(1, 1, self.vocab)
        out = SimpleNamespace(logits=logits)
        if return_hidden:
            out.hidden_states = [mx.zeros((1, bs, 4))]
        if return_shared_kv:
            out.shared_kv_states = {}
        return out

    def rollback_speculative_cache(self, cache, sink, accepted, bs):
        return None


class _FakeDraftModel:
    """Drafts the content token at every position (so drafts always disagree
    with a forced END target, exercising the accept/reject path)."""

    def __init__(self, block_size=4, content_token=1):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.content_token = content_token

    def reset(self, model):
        return None

    def set_shared_kv(self, shared_kv, offset):
        return None

    def draft_block(self, b, hidden, mask, bs, sampler, token_dtype):
        # bs-1 draft tokens, all content.
        return mx.full((1, bs - 1), self.content_token, dtype=token_dtype)


def _make_clone():
    """Install the wrap patches and return the soloheaven MTP clone."""
    mlx_engine_module._MTP_PATCHES_INSTALLED = False
    mlx_engine_module._HOT_PATH_FAST = False
    assert _install_mtp_wrap_patches() is True
    mvgen = sys.modules["mlx_vlm.generate"]
    clone = mvgen._mtp_rounds
    assert getattr(clone, "_mtp_wrap_patch", False) is True
    return clone


def _drive_clone(clone, *, max_tokens, first_bonus=START):
    model = SimpleNamespace(language_model=_FakeLM())
    draft = _FakeDraftModel(block_size=4)
    cache = [SimpleNamespace(offset=0)]
    gen = clone(
        model,
        draft,
        cache,
        mx.zeros((1, 1, 4)),  # hidden
        {},                    # shared_kv_states
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=_argmax_sampler,
        draft_block_size=4,
        token_dtype=mx.int32,
    )
    return [int(tok) for tok, _ in gen]


def _set_budget_stash(budget, family="gemma4", seed=None):
    mlx_engine_module._MTP_THINK_BUDGET = budget
    mlx_engine_module._MTP_THINK_END_TOKEN = END
    mlx_engine_module._MTP_THINK_START_TOKEN = START
    mlx_engine_module._MTP_THINK_FAMILY = family
    mlx_engine_module._MTP_TOKEN_HISTORY_SEED = list(seed or [])
    # No rep-penalty / FSM processors — isolate the thinking-budget path.
    mlx_engine_module._MTP_LOGITS_PROCESSORS = None


def _clear_stash():
    mlx_engine_module._MTP_THINK_BUDGET = None
    mlx_engine_module._MTP_THINK_END_TOKEN = None
    mlx_engine_module._MTP_THINK_START_TOKEN = None
    mlx_engine_module._MTP_THINK_FAMILY = None
    mlx_engine_module._MTP_TOKEN_HISTORY_SEED = None
    mlx_engine_module._MTP_LOGITS_PROCESSORS = None


def test_clone_forces_think_end_when_budget_exceeded():
    """With a LOW budget and the stash set, the clone must FORCE the think_end
    token (id END) into the emitted stream and the model must EXIT the thinking
    block well before max_tokens — instead of emitting content forever.
    """
    clone = _make_clone()
    try:
        # first_bonus = START (<|channel>): in_thinking flips True, count=1.
        # budget=3 => END must be forced at the 3rd thinking token.
        _set_budget_stash(budget=3, family="gemma4")
        emitted = _drive_clone(clone, max_tokens=64, first_bonus=START)

        assert END in emitted, (
            f"think_end ({END}) must be forced into the emitted stream; "
            f"got {emitted}"
        )
        # The model should have exited thinking quickly (around the budget),
        # NOT run to max_tokens worth of content tokens.
        end_idx = emitted.index(END)
        # Tokens before END (excluding the leading bonus) are thinking content;
        # the forced close should land near the budget, not 60+ tokens in.
        assert end_idx <= 6, (
            f"END forced too late (idx={end_idx}); thinking budget not enforced "
            f"early enough: {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_no_force_when_budget_unset():
    """Budget stash unset => NO force => the model stays in thinking and emits
    only content (never END) — proving the enforcement is a clean no-op when
    the budget is off (current/greedy behaviour preserved)."""
    clone = _make_clone()
    try:
        _clear_stash()  # budget unset
        emitted = _drive_clone(clone, max_tokens=24, first_bonus=START)
        assert END not in emitted, (
            f"END must NOT appear when the budget is unset (no-op); got {emitted}"
        )
        # All emitted tokens (after the START bonus) are content tokens.
        assert all(t in (START, 1) for t in emitted), emitted
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_no_force_when_not_in_thinking_gemma4():
    """Gemma4 + budget set but NO think_start ever emitted => in_thinking stays
    False => END is never forced (the cap only applies inside thinking)."""
    clone = _make_clone()
    try:
        # first_bonus is a plain content token (NOT START) => never enters
        # thinking under gemma4 semantics.
        _set_budget_stash(budget=2, family="gemma4")
        emitted = _drive_clone(clone, max_tokens=24, first_bonus=1)
        assert END not in emitted, (
            f"END must NOT be forced outside a thinking block (gemma4); "
            f"got {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


# ---------------------------------------------------------------------------
# Wiring: _run_vlm populates (and the finally clears) the budget stash from the
# ThinkingBudgetProcessor in logits_processors.
# ---------------------------------------------------------------------------


def _bare_vlm_engine():
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

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


def test_run_vlm_populates_think_budget_stash_from_processor(monkeypatch):
    """_run_vlm must extract the budget + token ids from the
    ThinkingBudgetProcessor present in logits_processors and stash them on the
    dedicated globals (so the clone enforces them). With no such processor the
    stash stays None (no-op)."""
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
            # Capture the stash AT stream-construction time (before the finally
            # clear, which runs only when the stream is driven by the caller).
            captured["budget"] = mlx_engine_module._MTP_THINK_BUDGET
            captured["end"] = mlx_engine_module._MTP_THINK_END_TOKEN
            captured["start"] = mlx_engine_module._MTP_THINK_START_TOKEN
            captured["family"] = mlx_engine_module._MTP_THINK_FAMILY
            return iter([])

        monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        tbp = ThinkingBudgetProcessor(
            budget=200, think_end_token=END, think_start_token=START,
            model_family="gemma4",
        )

        def _drive(procs):
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=0)]
            cs.token_ids = []
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[1, 2, 3], max_tokens=4,
                temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
                logits_processors=procs, session_id="s-think",
                total_prompt_tokens=3,
            )
            list(gen)

        # With the ThinkingBudgetProcessor present, the stash is populated.
        eng._vlm_executor.submit(_drive, [tbp]).result(timeout=10)
        assert captured["budget"] == 200
        assert captured["end"] == END
        assert captured["start"] == START
        assert captured["family"] == "gemma4"

        # Without it, the stash stays None (no-op).
        captured.clear()
        eng._vlm_executor.submit(_drive, None).result(timeout=10)
        assert captured["budget"] is None
        assert captured["end"] is None
        assert captured["start"] is None
        assert captured["family"] is None
    finally:
        _clear_stash()
        eng.close()
