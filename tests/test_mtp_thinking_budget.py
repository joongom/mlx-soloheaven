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


# --- Scripted fakes for the BARE-OPENER (gemma4 long-context) clone test ---
#
# A transition-map driven LM + drafter: position i predicts ``trans.get(prev, C)``
# where ``prev`` is the prefix's last token. The drafter follows the SAME chain
# from ``b``, so draft == target at every position (all drafts accept) and the
# emitted stream is exactly the deterministic chain — until the budget forces END.

BARE0 = 70   # first token of the bare ``thought\n`` opener
BARE1 = 71   # second token of the bare ``thought\n`` opener
CONTENT = 1  # thinking-content filler (loops to itself)


class _ScriptedLM:
    """Deterministic transition-driven gemma4 LM stub. ``trans[prev] -> next``;
    unknown prev -> CONTENT (so thinking content loops forever absent a cap)."""

    def __init__(self, trans, vocab=300):
        self.trans = dict(trans)
        self.vocab = vocab

    def _next(self, prev):
        return self.trans.get(int(prev), CONTENT)

    def __call__(self, x, cache=None, return_hidden=False, return_shared_kv=False):
        xrow = x.reshape(-1).tolist()
        bs = int(x.shape[1])
        rows = []
        for i in range(bs):
            nxt = self._next(xrow[i])  # prefix ends at position i
            rows.append(
                mx.where(mx.arange(self.vocab) == nxt, 1.0, 0.0)
            )
        logits = mx.stack(rows, axis=0).reshape(1, bs, self.vocab)
        out = SimpleNamespace(logits=logits)
        if return_hidden:
            out.hidden_states = [mx.zeros((1, bs, 4))]
        if return_shared_kv:
            out.shared_kv_states = {}
        return out

    def rollback_speculative_cache(self, cache, sink, accepted, bs):
        return None


class _ScriptedDraftModel:
    """Drafts by following the SAME transition chain from ``b`` so draft==target
    (every draft accepts) and the emitted stream is the deterministic chain."""

    def __init__(self, trans, block_size=4):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.trans = dict(trans)

    def reset(self, model):
        return None

    def set_shared_kv(self, shared_kv, offset):
        return None

    def draft_block(self, b, hidden, mask, bs, sampler, token_dtype):
        toks = []
        prev = int(b)
        for _ in range(bs - 1):
            nxt = self.trans.get(prev, CONTENT)
            toks.append(nxt)
            prev = nxt
        return mx.array([toks], dtype=token_dtype)


def _drive_scripted_clone(clone, trans, *, max_tokens, first_bonus):
    model = SimpleNamespace(language_model=_ScriptedLM(trans))
    draft = _ScriptedDraftModel(trans, block_size=4)
    cache = [SimpleNamespace(offset=0)]
    gen = clone(
        model, draft, cache, mx.zeros((1, 1, 4)), {},
        first_bonus=first_bonus, max_tokens=max_tokens, sampler=_argmax_sampler,
        draft_block_size=4, token_dtype=mx.int32,
    )
    return [int(tok) for tok, _ in gen]


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


def _set_budget_stash(budget, family="gemma4", seed=None, bare_open=None):
    mlx_engine_module._MTP_THINK_BUDGET = budget
    mlx_engine_module._MTP_THINK_END_TOKEN = END
    mlx_engine_module._MTP_THINK_START_TOKEN = START
    mlx_engine_module._MTP_THINK_FAMILY = family
    mlx_engine_module._MTP_THINK_BARE_OPEN_TOKENS = (
        list(bare_open) if bare_open else None
    )
    mlx_engine_module._MTP_TOKEN_HISTORY_SEED = list(seed or [])
    # No rep-penalty / FSM processors — isolate the thinking-budget path.
    mlx_engine_module._MTP_LOGITS_PROCESSORS = None


def _clear_stash():
    mlx_engine_module._MTP_THINK_BUDGET = None
    mlx_engine_module._MTP_THINK_END_TOKEN = None
    mlx_engine_module._MTP_THINK_START_TOKEN = None
    mlx_engine_module._MTP_THINK_FAMILY = None
    mlx_engine_module._MTP_THINK_BARE_OPEN_TOKENS = None
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


# ---------------------------------------------------------------------------
# BARE-OPENER FIX (gemma4 long-context): past the 1024 sliding window the
# ``<|channel>`` prime falls out and the model opens thinking with a BARE
# ``thought\n`` (no ``<|channel>`` token). The budget must still recognise that
# opener (at GEN-START only — mirroring ThinkingRouter FIX 4) and cap thinking.
# ---------------------------------------------------------------------------

BARE = (BARE0, BARE1)  # token-id sequence of the bare ``thought\n`` opener


# (a) bare opener (no <|channel>) -> thinking opens / count increments / fires.


def test_helper_bare_opener_opens_thinking_and_counts():
    """Gemma4 + bare ``thought\\n`` opener at gen-start (NO ``<|channel>``):
    in_thinking flips True after the full bare sequence, the opener tokens count
    toward the budget, and the cap fires — the long-context runaway is closed."""
    # Only the first opener token emitted => partial match, not yet in thinking.
    assert should_force_think_end(
        [BARE0], budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False
    # Full bare opener => in_thinking, count == len(opener) == 2.
    assert should_force_think_end(
        [BARE0, BARE1], budget=2, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is True
    assert should_force_think_end(
        [BARE0, BARE1], budget=3, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False  # count 2 < 3
    # One thinking-content token after the bare opener => count 3.
    assert should_force_think_end(
        [BARE0, BARE1, CONTENT], budget=3, think_start_token=START,
        think_end_token=END, model_family="gemma4", bare_open_tokens=BARE,
    ) is True


def test_helper_bare_opener_disabled_without_bare_tokens():
    """No bare_open_tokens supplied => the bare opener is inert (the original
    ``<|channel>``-only behaviour). The same history never opens thinking."""
    assert should_force_think_end(
        [BARE0, BARE1, CONTENT, CONTENT, CONTENT], budget=1,
        think_start_token=START, think_end_token=END, model_family="gemma4",
        bare_open_tokens=None,
    ) is False


# (b) full <|channel> opener still caps as before (with bare detection ALSO on).


def test_helper_full_channel_opener_still_caps_with_bare_active():
    """With bare detection active, the FULL ``<|channel>`` opener (short-context)
    must STILL open + cap exactly as before — the bare path must not regress the
    token-id opener that already worked."""
    # <|channel> (START) at gen-start opens immediately, count == 1.
    assert should_force_think_end(
        [START], budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is True
    assert should_force_think_end(
        [START, CONTENT], budget=2, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is True
    # No opener at all => never fires (model answered with no thinking block).
    assert should_force_think_end(
        [CONTENT, CONTENT, CONTENT], budget=1, think_start_token=START,
        think_end_token=END, model_family="gemma4", bare_open_tokens=BARE,
    ) is False


# (c) a literal ``thought\n`` AFTER content must NOT open thinking (no FP).


def test_helper_bare_opener_not_at_gen_start_does_not_open():
    """A literal ``thought\\n`` sequence appearing AFTER content (mid-generation)
    must NOT be mis-read as a thinking opener — mirrors ThinkingRouter FIX 4's
    gen-start constraint. content_seen latches True on the first real token, so
    the later bare sequence is plain content and the budget never fires."""
    # Content first, THEN the bare-opener byte sequence: must stay outside.
    hist = [CONTENT, BARE0, BARE1, CONTENT, CONTENT, CONTENT]
    assert should_force_think_end(
        hist, budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False
    # Also: a bare opener that re-appears AFTER a closed <|channel> block must
    # not re-open (only the FULL <|channel> re-opens mid-stream).
    hist2 = [START, CONTENT, END, BARE0, BARE1, CONTENT, CONTENT]
    assert should_force_think_end(
        hist2, budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False


def test_helper_partial_bare_then_mismatch_is_content():
    """A partial bare match that then DIVERGES (BARE0 followed by a non-BARE1
    token) is real content, not a thinking opener — the partial match is
    abandoned and content_seen latches."""
    hist = [BARE0, CONTENT, CONTENT, CONTENT, CONTENT]
    assert should_force_think_end(
        hist, budget=1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False


# (d) MTP-clone helper and ThinkingBudgetProcessor AGREE on the bare opener.


def _assert_helper_processor_lockstep_bare(history, budget, family, bare):
    """Bare-opener lockstep: drive a single ``ThinkingBudgetProcessor`` (with the
    bare opener configured) token-by-token and assert it forces at EXACTLY the
    same positions as the history-derived helper (which the MTP clone uses)."""
    proc = ThinkingBudgetProcessor(budget, END, START, family, bare_open_tokens=bare)
    for k in range(len(history) + 1):
        prefix = history[:k]
        helper_force = should_force_think_end(
            prefix, budget=budget, think_start_token=START, think_end_token=END,
            model_family=family, bare_open_tokens=bare,
        )
        logits = mx.zeros((1, 1000))
        out = proc(mx.array(prefix, dtype=mx.int32), logits)
        proc_force = float(out[0, END].item()) >= 1e8
        assert helper_force == proc_force, (
            f"bare mismatch at k={k} budget={budget}: helper={helper_force} "
            f"proc={proc_force}; prefix={prefix}"
        )


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_helper_matches_processor_bare_opener(budget):
    """Bare opener at gen-start then thinking content: clone helper == processor."""
    history = [BARE0, BARE1, CONTENT, CONTENT, CONTENT, CONTENT, CONTENT]
    _assert_helper_processor_lockstep_bare(history, budget, "gemma4", BARE)


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_helper_matches_processor_bare_then_content_reappear(budget):
    """Mid-content bare sequence + multi-cycle <|channel>: clone helper ==
    processor across the full gen-start + re-open semantics."""
    history = [BARE0, BARE1, CONTENT, END, CONTENT, BARE0, BARE1, START, CONTENT]
    _assert_helper_processor_lockstep_bare(history, budget, "gemma4", BARE)


# (a)+(d) end-to-end: the real MTP clone forces END on a BARE opener at long ctx.


def test_clone_forces_think_end_on_bare_opener():
    """End-to-end through the REAL MTP clone: with a bare ``thought\\n`` opener
    (first_bonus=BARE0, then BARE1, then thinking content) and the bare-opener
    stash set, the clone must FORCE the think_end token and EXIT thinking near
    the budget — exactly the long-context fix (no run to max_tokens)."""
    clone = _make_clone()
    try:
        # Chain: BARE0 -> BARE1 -> CONTENT -> CONTENT (loops). budget=4.
        trans = {BARE0: BARE1, BARE1: CONTENT, CONTENT: CONTENT}
        _set_budget_stash(budget=4, family="gemma4", bare_open=BARE)
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=64, first_bonus=BARE0
        )
        assert END in emitted, (
            f"think_end ({END}) must be forced on a bare opener; got {emitted}"
        )
        end_idx = emitted.index(END)
        # Opener (2) + content up to budget 4 => END around index ~4, well short
        # of max_tokens. Allow a little slack for block boundaries.
        assert end_idx <= 7, (
            f"END forced too late on bare opener (idx={end_idx}): {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_no_force_on_bare_sequence_after_content():
    """Gemma4 + bare stash set, but the bare sequence appears AFTER content
    (first_bonus=CONTENT): the gen-start constraint means it does NOT open
    thinking, so END is never forced (no false-positive at long context)."""
    clone = _make_clone()
    try:
        # first_bonus=CONTENT => content_seen latches immediately; the later
        # BARE0/BARE1 chain is plain content and must NOT open thinking.
        trans = {CONTENT: BARE0, BARE0: BARE1, BARE1: CONTENT}
        _set_budget_stash(budget=2, family="gemma4", bare_open=BARE)
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=24, first_bonus=CONTENT
        )
        assert END not in emitted, (
            f"END must NOT be forced for a mid-content bare sequence; got {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_bare_stash_threaded_from_run_vlm(monkeypatch):
    """_run_vlm must thread the gemma4 bare-opener token sequence off the
    ThinkingBudgetProcessor onto _MTP_THINK_BARE_OPEN_TOKENS (so the clone sees
    it). A processor WITHOUT bare tokens leaves the bare stash None."""
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
            captured["bare"] = mlx_engine_module._MTP_THINK_BARE_OPEN_TOKENS
            return iter([])

        monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        tbp = ThinkingBudgetProcessor(
            budget=200, think_end_token=END, think_start_token=START,
            model_family="gemma4", bare_open_tokens=[BARE0, BARE1],
        )

        def _drive(procs):
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=0)]
            cs.token_ids = []
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[1, 2, 3], max_tokens=4,
                temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
                logits_processors=procs, session_id="s-bare",
                total_prompt_tokens=3,
            )
            list(gen)

        eng._vlm_executor.submit(_drive, [tbp]).result(timeout=10)
        assert captured["bare"] == [BARE0, BARE1]

        # A processor with no bare tokens => bare stash stays None.
        captured.clear()
        tbp_no_bare = ThinkingBudgetProcessor(
            budget=200, think_end_token=END, think_start_token=START,
            model_family="gemma4", bare_open_tokens=None,
        )
        eng._vlm_executor.submit(_drive, [tbp_no_bare]).result(timeout=10)
        assert captured["bare"] is None
    finally:
        _clear_stash()
        eng.close()


# ---------------------------------------------------------------------------
# BARE-OPENER PROMPT-SEED DEFEAT (codex HIGH, simulation-confirmed): the MTP
# clone seeds the thinking state with the FULL PROMPT
# (``_MTP_TOKEN_HISTORY_SEED``). Folding the prompt seed WITH the bare matcher
# latched ``content_seen=True`` on the first (normal) prompt token, so the
# GENERATED bare opener could never fire and the budget never capped — the
# original long-context runaway was NOT actually fixed. The fix folds the
# prompt seed with ``bare_open_tokens=()`` and re-arms the bare sub-state at the
# generation boundary. These tests are the AUTHORITATIVE verification (the live
# bare opener is stochastic and was a prior false-positive at 4626 tokens where
# the model emitted the FULL ``<|channel>`` opener, not the bare one).
# ---------------------------------------------------------------------------


def test_clone_bare_opener_fires_despite_prompt_seed():
    """codex's exact repro: a NON-BARE prompt seed (``seed=[999]``) followed by a
    GENERATED bare opener (``first_bonus=BARE0`` then BARE1) MUST open thinking and
    force think_end at the budget. Before the fix the prompt token 999 latched
    content_seen and the budget never fired (think_end never appeared).
    """
    clone = _make_clone()
    try:
        # Chain from the bonus: BARE0 -> BARE1 -> CONTENT (loops). budget=2 so
        # the cap fires right after the 2-token bare opener completes.
        trans = {BARE0: BARE1, BARE1: CONTENT, CONTENT: CONTENT}
        _set_budget_stash(
            budget=2, family="gemma4", bare_open=BARE, seed=[999]
        )
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=64, first_bonus=BARE0
        )
        # THE BUG: without the prompt/gen domain split, END never appears here.
        assert END in emitted, (
            f"think_end ({END}) must be forced on a bare opener EVEN WITH a "
            f"non-bare prompt seed; got {emitted}"
        )
        end_idx = emitted.index(END)
        assert end_idx <= 5, (
            f"END forced too late despite prompt seed (idx={end_idx}): {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_bare_opener_fires_with_long_nonbare_prompt_seed():
    """A LONGER, multi-token non-bare prompt seed (mimicking a 50K-ish prompt's
    leading tokens) still must not defeat the generated bare opener."""
    clone = _make_clone()
    try:
        trans = {BARE0: BARE1, BARE1: CONTENT, CONTENT: CONTENT}
        seed = [999, 888, 777, 666, 555, 444, 333, 222, 111]
        _set_budget_stash(
            budget=3, family="gemma4", bare_open=BARE, seed=seed
        )
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=64, first_bonus=BARE0
        )
        assert END in emitted, (
            f"think_end ({END}) must fire on a bare opener after a long non-bare "
            f"prompt seed; got {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_bare_opener_fires_after_prior_turn_channel_block_in_seed():
    """Multi-turn realism: the prompt seed itself contains a PRIOR turn's
    complete ``<|channel>...<channel|>`` block (full markers) and ends OUTSIDE
    thinking (in_thinking=False at the boundary). A GENERATED bare opener must
    still open and the budget must fire — the prior-turn markers were tracked
    (so the seed's in_thinking is correct) WITHOUT latching the bare matcher."""
    clone = _make_clone()
    try:
        trans = {BARE0: BARE1, BARE1: CONTENT, CONTENT: CONTENT}
        # Seed = junk, a prior <|channel> block (START..END), then more junk.
        seed = [999, START, 5, 6, END, 888, 777]
        _set_budget_stash(
            budget=2, family="gemma4", bare_open=BARE, seed=seed
        )
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=64, first_bonus=BARE0
        )
        assert END in emitted, (
            f"think_end ({END}) must fire on a generated bare opener even when "
            f"the seed contains a prior <|channel> block; got {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_full_channel_opener_still_caps_with_seed_and_bare_active():
    """Regression: the FULL ``<|channel>`` opener at "long context" (non-empty
    seed) with bare detection ALSO active must STILL open + cap (the full opener
    path opens regardless of content_seen and must not regress)."""
    clone = _make_clone()
    try:
        # first_bonus=START (full <|channel>): opens immediately, count=1.
        _set_budget_stash(
            budget=3, family="gemma4", bare_open=BARE, seed=[999, 888, 777]
        )
        emitted = _drive_clone(clone, max_tokens=64, first_bonus=START)
        assert END in emitted, (
            f"full <|channel> opener must still cap with a seed + bare active; "
            f"got {emitted}"
        )
        assert emitted.index(END) <= 6, emitted
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_clone_mid_generation_bare_after_content_does_not_open_with_seed():
    """Regression (gen-start-only preserved WITHIN the generation domain): with a
    non-bare prompt seed, a bare sequence appearing AFTER generated content
    (first_bonus=CONTENT) must NOT open thinking — content_seen latches on the
    first GENERATED content token (not the prompt)."""
    clone = _make_clone()
    try:
        trans = {CONTENT: BARE0, BARE0: BARE1, BARE1: CONTENT}
        _set_budget_stash(
            budget=2, family="gemma4", bare_open=BARE, seed=[999, 888]
        )
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=24, first_bonus=CONTENT
        )
        assert END not in emitted, (
            f"a mid-GENERATION bare sequence must NOT open thinking; got {emitted}"
        )
    finally:
        _clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


# --- Pure helper: prompt/generation domain split (what the clone now does) ---


def test_helper_prompt_seed_with_empty_bare_then_gen_boundary_opens():
    """The MTP clone's exact recipe at the helper level: fold the prompt seed
    with ``bare_open_tokens=()`` (so a non-bare prompt token does NOT latch
    content_seen against the bare matcher), reset the bare sub-state at the
    generation boundary, then a GENERATED ``[BARE0, BARE1]`` opens thinking."""
    seed = [999, 888, 777]
    state = initial_think_state("gemma4")
    # Prompt domain: bare matcher OFF.
    for t in seed:
        state = advance_think_state(
            state, t, think_start_token=START, think_end_token=END,
            bare_open_tokens=(),
        )
    # The prompt seed must NOT have latched content_seen via the bare matcher;
    # in_thinking stays False (no full markers in this seed).
    assert state[0] is False
    assert state[3] is True  # empty-bare fold latches content_seen (no bare path)
    # Generation boundary: re-arm the bare sub-state (outside thinking).
    state = (state[0], state[1], 0, False)
    # Generation domain: bare matcher ON.
    for t in (BARE0, BARE1):
        state = advance_think_state(
            state, t, think_start_token=START, think_end_token=END,
            bare_open_tokens=BARE,
        )
    assert state[0] is True, "generated bare opener must open thinking"
    assert state[1] == 2, "opener tokens must count toward the budget"
    assert force_end_from_state(state, 2) is True


def test_helper_seed_with_channel_block_ends_outside_thinking():
    """A prompt seed containing a prior ``<|channel>...<channel|>`` block, folded
    with empty-bare, ends with in_thinking=False (the block closed); then a
    generated bare opener still opens after the boundary re-arm."""
    seed = [999, START, 5, 6, END, 888]
    state = initial_think_state("gemma4")
    for t in seed:
        state = advance_think_state(
            state, t, think_start_token=START, think_end_token=END,
            bare_open_tokens=(),
        )
    assert state[0] is False, "prior-turn channel block must be closed in seed"
    state = (state[0], state[1], 0, False)  # boundary re-arm
    for t in (BARE0, BARE1):
        state = advance_think_state(
            state, t, think_start_token=START, think_end_token=END,
            bare_open_tokens=BARE,
        )
    assert state[0] is True
    assert force_end_from_state(state, 2) is True


# ---------------------------------------------------------------------------
# ThinkingBudgetProcessor (plain path) prompt-boundary handling (codex (2)).
# The processor receives the RUNNING tokens array; mlx-vlm folds the prompt tail
# into it on the prefill step BEFORE the first generated token. The processor
# must only bare-match GENERATED tokens (gated by the captured prompt boundary).
# ---------------------------------------------------------------------------


def _drive_processor_with_prompt_tail(proc, prompt_tail, generated):
    """Drive the processor the way mlx-vlm does: tokens starts empty, the prefill
    step appends the prompt tail (whole prompt when unchunked, else the last
    token), then each step appends one generated token. Returns the list of
    per-call force decisions (True iff END logit forced)."""
    forced = []
    tokens = list(prompt_tail)  # prefill _step appends prompt tail first
    # First call: prompt tail present (this is what defeats the bare matcher
    # unless gated).
    logits = mx.zeros((1, 1000))
    out = proc(mx.array(tokens, dtype=mx.int32), logits)
    forced.append(float(out[0, END].item()) >= 1e8)
    for g in generated:
        tokens.append(g)
        logits = mx.zeros((1, 1000))
        out = proc(mx.array(tokens, dtype=mx.int32), logits)
        forced.append(float(out[0, END].item()) >= 1e8)
    return forced


def test_processor_bare_opener_fires_despite_prompt_tail():
    """codex (2): the prefill step folds the prompt tail (e.g. last prompt token
    999) into the processor's tokens BEFORE generation. The processor must NOT
    let that prompt token latch content_seen — a GENERATED bare opener must still
    open and force END at the budget."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="gemma4", bare_open_tokens=BARE,
    )
    forced = _drive_processor_with_prompt_tail(
        proc, prompt_tail=[999], generated=[BARE0, BARE1, CONTENT]
    )
    # forced[0]=prefill(prompt tail), [1]=after BARE0, [2]=after BARE1 (opens,
    # count=2>=budget -> force), [3]=after CONTENT (still forcing).
    assert forced == [False, False, True, True], forced
    assert proc.in_thinking is True


def test_processor_bare_opener_fires_with_whole_prompt_in_tokens():
    """Unchunked prefill: the WHOLE prompt (many non-bare tokens) lands in the
    processor's tokens on the first call. The generated bare opener must still
    open."""
    proc = ThinkingBudgetProcessor(
        budget=3, think_end_token=END, think_start_token=START,
        model_family="gemma4", bare_open_tokens=BARE,
    )
    forced = _drive_processor_with_prompt_tail(
        proc, prompt_tail=[5, 6, 7, 8, 9, 999], generated=[BARE0, BARE1, CONTENT]
    )
    assert forced[-1] is True, forced  # budget 3 reached after one content tok
    assert proc.in_thinking is True


def test_processor_mid_generation_bare_after_content_does_not_open():
    """Within the generation domain the gen-start constraint still holds: a bare
    sequence after a GENERATED content token must NOT open (content_seen latches
    on the first generated content token, not on the prompt tail)."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="gemma4", bare_open_tokens=BARE,
    )
    forced = _drive_processor_with_prompt_tail(
        proc, prompt_tail=[999], generated=[CONTENT, BARE0, BARE1, CONTENT]
    )
    assert not any(forced), forced
    assert proc.in_thinking is False


def test_processor_full_channel_opener_still_caps_with_prompt_tail():
    """Regression: the full ``<|channel>`` opener (generated) still opens + caps
    even with a non-bare prompt tail and bare detection active."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="gemma4", bare_open_tokens=BARE,
    )
    forced = _drive_processor_with_prompt_tail(
        proc, prompt_tail=[999], generated=[START, CONTENT]
    )
    # START opens (count 1), CONTENT -> count 2 >= budget -> force.
    assert forced == [False, False, True], forced


def test_processor_noop_when_bare_empty_with_prompt_tail():
    """No-op safety: bare empty => the bare path is inert even with a prompt tail
    + a generated bare-looking sequence (a literal ``thought\\n`` is plain
    content). The prompt-boundary handling must not spuriously open thinking."""
    proc_no_bare = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="gemma4", bare_open_tokens=None,
    )
    forced = _drive_processor_with_prompt_tail(
        proc_no_bare, prompt_tail=[999], generated=[BARE0, BARE1, CONTENT, CONTENT]
    )
    assert not any(forced), forced  # never opens without bare tokens
    assert proc_no_bare.in_thinking is False


def test_helper_budget_off_noop_with_prompt_seed_and_bare():
    """budget<=0 => the history-derived helper (what the MTP clone uses; the
    clone gates ``_think_active = budget > 0``) is a clean no-op even when a
    non-bare prompt seed + a generated bare opener are present."""
    # Mirror the clone recipe: seed folded with empty bare, boundary re-arm, then
    # generated bare opener — but budget 0 => never force.
    hist = [999, BARE0, BARE1, CONTENT, CONTENT]
    assert should_force_think_end(
        hist, budget=0, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False
    assert should_force_think_end(
        hist, budget=-1, think_start_token=START, think_end_token=END,
        model_family="gemma4", bare_open_tokens=BARE,
    ) is False
