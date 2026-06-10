"""MTP-compatible BASE caches (shared system-prefix pool) — hermetic tests.

Production failure being fixed: a request hit the system-prompt base cache
(cache_mode=base_hit) but the QwenMTP gate rejected the seeded 40-entry
clone ("layout 40 != 40+1") and COLD-FILLED the whole 94K prompt — base
caches were built with make_prompt_cache(target) only, no head entry and no
boundary hidden.

Covers (mock caches + duck-typed ops, engine shells via __new__ — same
conventions as tests/test_qwen_mtp.py):
  * mtp_prefill_base — lands EXACTLY on the finalized contract: target
    offsets == len(tokens), head == len(tokens) - 1 (lazy last slot open),
    returns h_{N-1}; per-layer fail-closed post-condition;
  * _maybe_register_base_cache — active qwen_mtp drafter -> 41-entry-style
    base + resume hidden + layout marker; drafter inactive (or vlm) ->
    historical target-only registration unchanged; existing hash -> skip;
  * _clone_base_cache — deep-copies the head entry, the seeded clone +
    pooled hidden PASS validate_mtp_cache_reuse with an appended suffix;
  * end-to-end (_generate_locked) — base_hit seeds the clone + hidden, the
    MTP gate PASSES (no cold-fill), the runner prefills ONLY the suffix and
    commits the lazy last-slot pair (h_{N-1}, first_suffix_token), and the
    session lands reusable for the next turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import mlx.core as mx
import pytest

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import BaseCacheEntry, MLXEngine
from mlx_soloheaven.engine.qwen_mtp import (
    MTPCacheCorruption,
    mtp_prefill_base,
    qwen_mtp_generate_step,
    validate_mtp_cache_reuse,
)


# ---------------------------------------------------------------------------
# Mock caches + ops (same semantics as tests/test_qwen_mtp.py)
# ---------------------------------------------------------------------------


class MockArrays:
    """ArraysCache-like: recurrent state, untrimmable; cache[0] records the
    exact consumed-token history."""

    def __init__(self):
        self.cache = [()]


class MockKV:
    """KVCache-like: offset + honest trim."""

    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class MockOps:
    """Duck-typed QwenMTPOps stand-in: position hidden == the token consumed
    at that position, so pairing/rollback errors surface as wrong values."""

    def __init__(self, next_map, head_map, vocab=64):
        self.next_map = next_map
        self.head_map = head_map
        self.vocab = vocab
        self.head_calls: List[tuple] = []

    def _hid(self, toks):
        return mx.array([[[float(t)] for t in toks]])

    def _logits(self, toks, fn):
        rows = []
        for t in toks:
            row = [0.0] * self.vocab
            row[fn(int(t))] = 1e9
            rows.append(row)
        return mx.array([rows])

    def target_hidden(self, toks2d, cache):
        toks = [int(t) for t in toks2d[0].tolist()]
        for c in cache:
            if isinstance(getattr(c, "cache", None), list):
                c.cache = [c.cache[0] + tuple(toks)]
            else:
                c.offset += len(toks)
        return self._hid(toks)

    def target_logits(self, h):
        toks = [int(round(v[0])) for v in h[0].tolist()]
        return self._logits(toks, self.next_map)

    def head_hidden(self, hidden, next_ids, mtp_cache):
        toks = [int(t) for t in next_ids[0].tolist()]
        self.head_calls.append(
            ([int(round(v[0])) for v in hidden[0].tolist()], toks)
        )
        for c in mtp_cache:
            c.offset += len(toks)
        return self._hid(toks)

    def head_logits(self, x):
        toks = [int(round(v[0])) for v in x[0].tolist()]
        return self._logits(toks, self.head_map)


def _mk_target():
    """5 target entries — 3 arrays-like + 2 kv-like (qwen hybrid shape)."""
    return [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV()]


class CallableModel:
    """Minimal mlx-lm-shaped target: make_cache() (consumed by
    make_prompt_cache) + a forward that advances offset-bearing layers
    (consumed by the plain _prefill_cache branch)."""

    def __init__(self):
        self.layers = [object()] * 5

    def make_cache(self):
        return _mk_target()

    def __call__(self, toks2d, cache=None):
        n = int(toks2d.shape[1])
        for c in cache or []:
            if hasattr(c, "offset"):
                c.offset += n
        return toks2d


NEXT = lambda t: (t + 1) % 50


# ---------------------------------------------------------------------------
# mtp_prefill_base — finalized-shape prefill helper
# ---------------------------------------------------------------------------


def test_mtp_prefill_base_finalized_shape():
    """Chunked prefill (step=2 over 5 tokens) must land on the finalized
    contract: target offset N, head N-1, head fed (h_i, tok_{i+1}) pairs,
    boundary hidden == h of the LAST token."""
    tokens = [1, 2, 3, 4, 5]
    cache = _mk_target() + [MockKV()]
    ops = MockOps(NEXT, NEXT)

    hid = mtp_prefill_base(
        tokens,
        model=None,
        head=None,
        prompt_cache=cache,
        n_target_layers=5,
        prefill_step_size=2,
        ops=ops,
    )
    # Target consumed every token; head trails by the lazy last slot.
    for i in (2, 4):
        assert cache[i].offset == len(tokens)
    assert cache[5].offset == len(tokens) - 1
    for i in (0, 1, 3):
        assert cache[i].cache[0] == tuple(tokens)
    # Head pairing: slot i = (h_i, tokens[i+1]) — chunked exactly like the
    # runner's prompt prefill (the LAST token never reaches the head).
    assert ops.head_calls == [([1, 2], [2, 3]), ([3, 4], [4, 5])]
    # Boundary hidden is h_{N-1} (mock encodes hiddens as token values).
    assert int(round(hid[0, 0, 0].item())) == tokens[-1]
    # ...and it PASSES the reuse gate with any appended suffix.
    ok, why = validate_mtp_cache_reuse(
        cache, tokens, tokens + [40, 20], 5, 1, hid
    )
    assert ok, why


def test_mtp_prefill_base_single_token():
    """N == 1: no head pair exists yet — head stays at 0 == N - 1 and the
    boundary hidden is the lone token's."""
    cache = _mk_target() + [MockKV()]
    ops = MockOps(NEXT, NEXT)
    hid = mtp_prefill_base(
        [9], model=None, head=None, prompt_cache=cache,
        n_target_layers=5, ops=ops,
    )
    assert cache[2].offset == 1
    assert cache[5].offset == 0
    assert ops.head_calls == []
    assert int(round(hid[0, 0, 0].item())) == 9


def test_mtp_prefill_base_fail_closed_on_shape_miss():
    """A cache that does not land on the finalized shape (here: a head
    entry pre-advanced before prefill) must raise, never register."""
    cache = _mk_target() + [MockKV()]
    cache[5].offset = 3  # desynced head
    with pytest.raises(MTPCacheCorruption):
        mtp_prefill_base(
            [1, 2, 3, 4, 5], model=None, head=None, prompt_cache=cache,
            n_target_layers=5, ops=MockOps(NEXT, NEXT),
        )


# ---------------------------------------------------------------------------
# _maybe_register_base_cache — layout by drafter state
# ---------------------------------------------------------------------------

SYS_TOKENS = [11, 12, 13, 14, 90]
MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "hi"},
]


def _registration_engine(*, mtp: bool, use_vlm: bool = False) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.enable_thinking = False
    eng._use_vlm = use_vlm
    eng.model_id = "stub"
    eng._base_caches = {}
    eng._has_rotating_cache = False
    if mtp:
        eng._drafter = SimpleNamespace(layers=[object()])
        eng._draft_kind = "qwen_mtp"
    else:
        eng._drafter = None
        eng._draft_kind = None
    eng._language_model = CallableModel()
    eng._extract_system_tokens = (
        lambda messages, prompt_tokens, tools=None, thinking=None: list(SYS_TOKENS)
    )
    return eng


def test_register_base_cache_mtp_layout(monkeypatch):
    """Active qwen_mtp drafter: the registered base carries target + head
    entries in the FINALIZED shape plus the boundary hidden + marker."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    monkeypatch.setattr(
        qwen_mtp_mod, "QwenMTPOps", lambda model, head: MockOps(NEXT, NEXT)
    )
    eng = _registration_engine(mtp=True)
    eng._maybe_register_base_cache(MESSAGES, SYS_TOKENS + [40, 20, 21])

    h = MLXEngine._system_hash(MESSAGES)
    entry = eng._base_caches[h]
    assert entry.tokens == SYS_TOKENS
    assert entry.token_count == len(SYS_TOKENS)
    assert entry.mtp_layout is True
    assert entry.mtp_resume_hidden is not None
    assert int(round(entry.mtp_resume_hidden[0, 0, 0].item())) == SYS_TOKENS[-1]
    # 5 target + 1 head entries, head trailing by the lazy last slot.
    assert len(entry.cache) == 6
    for i in (2, 4):
        assert entry.cache[i].offset == len(SYS_TOKENS)
    assert entry.cache[5].offset == len(SYS_TOKENS) - 1
    # A clone of this base must pass the reuse gate with a suffix appended.
    ok, why = validate_mtp_cache_reuse(
        entry.cache, entry.tokens, SYS_TOKENS + [40, 20, 21], 5, 1,
        entry.mtp_resume_hidden,
    )
    assert ok, why


def test_register_base_cache_plain_when_drafter_inactive():
    """Regression: no drafter -> registration is EXACTLY the historical
    target-only shape (no head entry, no hidden, no marker)."""
    eng = _registration_engine(mtp=False)
    eng._maybe_register_base_cache(MESSAGES, SYS_TOKENS + [40, 20, 21])

    entry = eng._base_caches[MLXEngine._system_hash(MESSAGES)]
    assert len(entry.cache) == 5
    for i in (2, 4):
        assert entry.cache[i].offset == len(SYS_TOKENS)
    assert entry.mtp_layout is False
    assert entry.mtp_resume_hidden is None


def test_register_base_cache_plain_on_vlm_backend():
    """A drafter on the mlx-vlm backend (gemma4 MTP) must NOT produce the
    qwen 41-entry layout — the vlm path never consumes it."""
    eng = _registration_engine(mtp=True, use_vlm=True)
    eng._maybe_register_base_cache(MESSAGES, SYS_TOKENS + [40, 20, 21])

    entry = eng._base_caches[MLXEngine._system_hash(MESSAGES)]
    assert len(entry.cache) == 5
    assert entry.mtp_layout is False
    assert entry.mtp_resume_hidden is None


def test_register_base_cache_skips_existing_hash(monkeypatch):
    """A pre-existing (e.g. plain pre-B) entry under the same hash is KEPT —
    Feature A's plain fallback covers it; registration never overwrites."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    monkeypatch.setattr(
        qwen_mtp_mod, "QwenMTPOps", lambda model, head: MockOps(NEXT, NEXT)
    )
    eng = _registration_engine(mtp=True)
    h = MLXEngine._system_hash(MESSAGES)
    stale = BaseCacheEntry(
        system_hash=h, cache=_mk_target(), tokens=list(SYS_TOKENS),
        token_count=len(SYS_TOKENS),
    )
    eng._base_caches[h] = stale
    eng._maybe_register_base_cache(MESSAGES, SYS_TOKENS + [40, 20, 21])
    assert eng._base_caches[h] is stale


# ---------------------------------------------------------------------------
# _clone_base_cache + seeding propagation
# ---------------------------------------------------------------------------


def _finalized_base_entry(tokens):
    cache = _mk_target() + [MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(tokens)
    cache[5].offset = len(tokens) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(tokens)]
    return BaseCacheEntry(
        system_hash=MLXEngine._system_hash(MESSAGES),
        cache=cache,
        tokens=list(tokens),
        token_count=len(tokens),
        mtp_layout=True,
        mtp_resume_hidden=mx.array([[[float(tokens[-1])]]]),
    )


def test_clone_base_cache_deep_copies_head_and_validates():
    eng = MLXEngine.__new__(MLXEngine)
    base = _finalized_base_entry(SYS_TOKENS)

    cloned = eng._clone_base_cache(base)
    assert base.hit_count == 1
    assert cloned is not base.cache
    # EVERY entry — the trailing head included — is an independent copy.
    for orig, copy_ in zip(base.cache, cloned):
        assert copy_ is not orig
    assert cloned[5].offset == len(SYS_TOKENS) - 1
    # Mutating the clone never touches the pooled base.
    cloned[5].offset += 1
    assert base.cache[5].offset == len(SYS_TOKENS) - 1
    cloned[5].offset -= 1
    # The seeded contract: clone + pooled hidden pass the gate on append.
    ok, why = validate_mtp_cache_reuse(
        cloned, base.tokens, SYS_TOKENS + [40, 20], 5, 1,
        base.mtp_resume_hidden,
    )
    assert ok, why


# ---------------------------------------------------------------------------
# End-to-end: base_hit -> MTP gate passes -> no cold-fill
# ---------------------------------------------------------------------------


def _e2e_engine(base_entry, full_prompt):
    """Engine shell driving _generate_locked through the MISS -> base_hit
    seeding path with the REAL _find_base_cache/_clone_base_cache and the
    real MTP gate; only tokenization and the model are stubbed."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.enable_thinking = False
    eng.cfg.think_end_token = -1
    eng.cfg.pld_enabled = False
    eng.cfg.kv_bits = 0
    eng._use_vlm = False
    eng.model_id = "stub"
    eng.model_family = "chatml"
    eng._drafter = SimpleNamespace(layers=[object()])
    eng._draft_kind = "qwen_mtp"
    eng._mtp_block_size = 3
    eng._language_model = SimpleNamespace(
        layers=[object()] * 5, make_cache=_mk_target,
    )
    eng.tokenizer = SimpleNamespace(decode=lambda ids: "x", eos_token_ids=[])
    eng._sessions = {}
    eng._touch_gpu = lambda: None
    eng._has_disk_cache = lambda sid: False
    eng._has_rotating_cache = False
    eng._sliding_window_size = 0
    eng._base_caches = {base_entry.system_hash: base_entry}
    eng._tokenize_prompt = (
        lambda msgs, thinking=None, tools=None: list(full_prompt)
    )
    return eng


def test_generate_locked_base_hit_mtp_gate_passes(monkeypatch):
    """THE PRODUCTION SCENARIO, FIXED: base_hit seeds the MTP-finalized
    clone + hidden, the gate PASSES — the runner prefills ONLY the suffix
    (no 94K-style cold-fill), commits the lazy pair (h_{N-1}, suffix[0]),
    and the saved session is MTP-reusable next turn."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    base = _finalized_base_entry(SYS_TOKENS)
    suffix = [40, 20, 21]
    full_prompt = SYS_TOKENS + suffix
    eng = _e2e_engine(base, full_prompt)

    captured = {}

    def fake_step(prompt, model, head, **kwargs):
        captured.update(kwargs)
        captured["prompt_len"] = int(prompt.size)
        ops = MockOps(NEXT, NEXT)
        captured["ops"] = ops
        return qwen_mtp_generate_step(
            prompt, model=None, head=None, ops=ops, **kwargs
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    chunks = list(
        eng._generate_locked(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=4,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )
    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [22, 23, 24, 25]

    # NO cold-fill: only the 3-token suffix was prefilled, on the SEEDED
    # clone (6 entries), with the pooled boundary hidden handed through.
    assert captured["prompt_len"] == len(suffix)
    assert captured["resume_hidden"] is base.mtp_resume_hidden
    assert len(captured["prompt_cache"]) == 6
    # First head call == the lazy last-slot commit (h_{N-1}, suffix[0]).
    assert captured["ops"].head_calls[0] == ([SYS_TOKENS[-1]], [suffix[0]])

    # The pooled base itself is untouched (clone isolation) and counted.
    assert base.hit_count == 1
    assert base.cache[2].offset == len(SYS_TOKENS)
    assert base.cache[5].offset == len(SYS_TOKENS) - 1

    # The saved session survived reconcile in the finalized shape and is
    # MTP-reusable next turn (stash re-stashed by on_finalize).
    cs = eng._sessions["s"].cache_state
    expected_ids = full_prompt + [22, 23, 24, 25]
    assert cs.cache is captured["prompt_cache"]
    assert cs.token_ids == expected_ids
    for i in (2, 4):
        assert cs.cache[i].offset == len(expected_ids)
    assert cs.cache[5].offset == len(expected_ids) - 1
    assert cs.mtp_last_hidden is not None
    assert cs.mtp_hidden_offset == len(expected_ids)
    ok, why = validate_mtp_cache_reuse(
        cs.cache, cs.token_ids, expected_ids + [40, 50], 5, 1,
        cs.mtp_last_hidden,
    )
    assert ok, why
