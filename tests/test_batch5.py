"""Batch-5 regression tests (U2/U19/U20/U22/U23/U26/U27).

U2  — base caches (_base_caches) are byte-accounted against the SAME
      memory_budget_gb as the active sessions + prefix pool, and evicted
      LRU-first (keep-MRU) by the budget sweep. Previously they were
      memory-only, unbounded, and invisible to the budget.
U19 — the structured-output FSM index cache is keyed by a TOKENIZER
      FINGERPRINT (name + full vocab hash) so multi-model deployments never
      reuse an index compiled against another model's vocabulary; the cache
      is a bounded LRU.
U20 — sampling params are validated at the API boundary (400 + OpenAI error
      envelope): temperature >= 0, top_p in (0,1], min_p in [0,1],
      top_k >= 0, repetition_penalty > 0, frequency/presence in [-2,2],
      max_tokens >= 1 — plus cheap engine-side clamps (top_k capped to
      vocab in the eager sampler; non-positive repetition_penalty
      neutralized) and the same validation on the session-settings PATCH.
U22 — the thinking-budget forced </think> is DEFERRED (bounded, max 4
      tokens) while the emitted byte stream ends mid multi-byte UTF-8
      character, so reasoning_content never ends in U+FFFD.
U23 — --repetition-penalty uses the None sentinel: an unset flag keeps the
      Config dataclass default (1.05, gemma4 FIX 2) instead of being
      clobbered by the argparse default (1.0, dead config pre-fix).
U26 — per-session drafter acceptance stats accumulate in an engine-level
      registry that survives the per-turn SessionState reinstall (they were
      reset every turn, and a new session's first turn was dropped).
U27 — the per-model Config rebuild is one shared builder
      (server.build_per_model_config) that carries --stream-coalesce-n/ms
      (the in-process --models copy silently dropped them).

ROUND 2 (codex findings F1-F7, second half of this file):
F1 — [U20] oversized top_k is clamped to the keep-all sentinel (0) at the
     vlm-path kwargs assembly (mlx-vlm's upstream sampler raises mid-stream
     for top_k >= vocab).
F2 — [U19] canonical length-prefixed fingerprint (no concat collisions),
     EOS ids included in the hashed identity, full sha256 digest,
     single-flight index compile.
F3 — [U2] MRU base-cache protection is CONDITIONAL: an entry that cannot
     fit the budget allowance (after the unevictable sessions) is evicted
     too (a lone oversize entry no longer pins the budget).
F4 — [U22] shared byte-tail check + ForceDeferralGate: a genuine literal
     U+FFFD no longer defers (only a valid-but-incomplete byte tail does),
     and the mlx-vlm MTP clone's force site consults the same gate.
F5 — [U26] drafter-stats registry is a bounded LRU; the web delete route
     also deletes the engine session; admin readers return snapshots.
F6 — [U20] NaN/Infinity floats are rejected 400 at both API boundaries and
     treated as invalid by the engine clamp.
F7 — the sessions schema carries top_p/min_p/top_k/repetition_penalty
     (fresh CREATE TABLE + try-ALTER migration); verified against a REAL
     sqlite DB (the round-1 test mocked update_session and missed it).

Harness pieces are shared with tests/test_qwen_mtp.py,
tests/test_active_session_eviction.py and tests/test_mtp_thinking_budget.py
(same-directory import under pytest's prepend import mode).
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_soloheaven.cli import parse_args
from mlx_soloheaven.config import Config, ModelConfig
from mlx_soloheaven.engine import structured
from mlx_soloheaven.engine.mlx_engine import (
    BaseCacheEntry,
    MLXEngine,
    SessionState,
    _make_eager_sampler,
)
from mlx_soloheaven.engine.thinking import ThinkingBudgetProcessor

from test_active_session_eviction import (
    _GB,
    _StubCacheLayer,
    _make_engine,
    _make_session,
)
from test_qwen_mtp import _generate_engine, _scripted_lm_stream


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_ALL_SAMPLING_ENV = (
    "SOLOHEAVEN_TEMPERATURE",
    "SOLOHEAVEN_TOP_P",
    "SOLOHEAVEN_MIN_P",
    "SOLOHEAVEN_TOP_K",
    "SOLOHEAVEN_REPETITION_PENALTY",
)


@pytest.fixture(autouse=True)
def _clean_sampling_env(monkeypatch):
    """Sampling env vars from the dev shell must not leak into parse_args."""
    for key in _ALL_SAMPLING_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clean_index_cache():
    """Each test starts with an empty FSM index cache (module-global LRU)."""
    with structured._INDEX_CACHE_LOCK:
        structured._INDEX_CACHE.clear()
    yield
    with structured._INDEX_CACHE_LOCK:
        structured._INDEX_CACHE.clear()


def _drive(eng, messages, *, repetition_penalty=1.0, max_tokens=8):
    """Drive _generate_locked to completion (local twin of
    test_api_behavior._drive_locked with the penalty exposed)."""
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=repetition_penalty,
            response_format=None,
            stop_sequences=None,
        )
    )


class _StubApiEngine:
    """Minimal engine for driving openai_compat.chat_completions up to the
    validation layer (mirrors the U24 stop-validation tests)."""

    model_family = "chatml"
    model_id = "m"
    cfg = SimpleNamespace(enable_thinking=False)


def _post_chat(request):
    """Run chat_completions against a stub engine registry."""
    from mlx_soloheaven.api import openai_compat

    old_engines = openai_compat._engines
    old_default = openai_compat._default_engine
    try:
        eng = _StubApiEngine()
        openai_compat.set_engines({"m": eng}, eng)
        return asyncio.run(openai_compat.chat_completions(request))
    finally:
        openai_compat.set_engines(old_engines, old_default)


# ---------------------------------------------------------------------------
# U23 — --repetition-penalty None sentinel (config default no longer dead)
# ---------------------------------------------------------------------------


def test_u23_unset_flag_keeps_config_default():
    """FAILS PRE-FIX: argparse always supplied 1.0, clobbering the dataclass
    default 1.05 (gemma4 FIX 2) — the config default was dead code."""
    args = parse_args(["--model", "/tmp/x"])
    assert args.repetition_penalty is None
    cfg = Config.from_args(args)
    assert cfg.default_repetition_penalty == 1.05
    assert cfg.models[0].default_repetition_penalty == 1.05


def test_u23_explicit_flag_wins():
    cfg = Config.from_args(
        parse_args(["--model", "/tmp/x", "--repetition-penalty", "1.2"])
    )
    assert cfg.default_repetition_penalty == 1.2
    assert cfg.models[0].default_repetition_penalty == 1.2


def test_u23_env_var_counts_as_explicit(monkeypatch):
    monkeypatch.setenv("SOLOHEAVEN_REPETITION_PENALTY", "1.3")
    cfg = Config.from_args(parse_args(["--model", "/tmp/x"]))
    assert cfg.default_repetition_penalty == 1.3


def test_u23_multi_model_path_resolves_same():
    """The --models builder shares the sentinel resolution."""
    cfg = Config.from_args(parse_args(["--models", "/tmp/a", "/tmp/b"]))
    for m in cfg.models:
        assert m.default_repetition_penalty == 1.05
    cfg2 = Config.from_args(
        parse_args(["--models", "/tmp/a", "/tmp/b", "--repetition-penalty", "1.0"])
    )
    for m in cfg2.models:
        assert m.default_repetition_penalty == 1.0


def test_u23_not_recorded_in_cli_set_sampling():
    """cli_set_sampling exists solely for the generation_config auto-apply,
    which intentionally never reads repetition_penalty — the flag must not
    pollute the marker."""
    cfg = Config.from_args(
        parse_args(["--model", "/tmp/x", "--repetition-penalty", "1.2"])
    )
    assert "repetition_penalty" not in cfg.cli_set_sampling


# ---------------------------------------------------------------------------
# U20 — sampling validation at the API boundary + engine clamps
# ---------------------------------------------------------------------------


def test_u20_endpoint_rejects_zero_repetition_penalty():
    """FAILS PRE-FIX: repetition_penalty=0 sailed through to the engine and
    divided logits by zero."""
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        model="m", stream=False, repetition_penalty=0.0,
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _post_chat(req)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert "repetition_penalty" in body["error"]["message"]


@pytest.mark.parametrize("field,value", [
    ("temperature", -0.5),
    ("top_p", 0.0),
    ("top_p", 1.5),
    ("min_p", -0.1),
    ("min_p", 1.1),
    ("top_k", -1),
    ("repetition_penalty", -2.0),
    ("frequency_penalty", -3.0),
    ("presence_penalty", 2.5),
    ("max_tokens", 0),
    ("max_completion_tokens", -5),
])
def test_u20_validator_rejects_out_of_range(field, value):
    from mlx_soloheaven.api.openai_compat import _validate_sampling_params
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        **{field: value},
    )
    resp = _validate_sampling_params(req)
    assert resp is not None, f"{field}={value} must be rejected"
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert field in body["error"]["message"]


def test_u20_validator_accepts_boundaries_and_unset():
    from mlx_soloheaven.api.openai_compat import _validate_sampling_params
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    # All unset (the common request) passes.
    req = ChatCompletionRequest(
        model="m", messages=[ChatMessage(role="user", content="hi")],
    )
    assert _validate_sampling_params(req) is None
    # Exact boundary values are legal.
    req2 = ChatCompletionRequest(
        model="m", messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
        repetition_penalty=0.1, frequency_penalty=-2.0, presence_penalty=2.0,
        max_tokens=1,
    )
    assert _validate_sampling_params(req2) is None
    # A huge top_k is NOT an API error — it is capped engine-side.
    req3 = ChatCompletionRequest(
        model="m", messages=[ChatMessage(role="user", content="hi")],
        top_k=10**9,
    )
    assert _validate_sampling_params(req3) is None


def test_u20_eager_sampler_caps_top_k_to_vocab():
    """FAILS PRE-FIX: top_k >= vocab raised ValueError inside the compiled
    top-k filter mid-generation. Now it degrades to keep-all (a no-op)."""
    logprobs = mx.log(mx.softmax(mx.arange(8, dtype=mx.float32)))[None]
    for k in (8, 9, 10**9):  # == vocab and beyond
        sampler = _make_eager_sampler(temp=1.0, top_k=k)
        tok = sampler(logprobs)
        assert 0 <= int(tok.item()) < 8
    # In-range top_k still filters (greedy-dominant distribution, k=1 must
    # always pick the argmax).
    sampler1 = _make_eager_sampler(temp=1.0, top_k=1)
    assert int(sampler1(logprobs).item()) == 7


def test_u20_engine_clamps_nonpositive_penalty(monkeypatch):
    """FAILS PRE-FIX: repetition_penalty=0 reached RepetitionPenaltyProcessor
    (divide-by-zero logits). Post-fix the engine clamps it to 1.0, so the
    processor is never even constructed."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    created: list[float] = []

    class _Recorder:
        def __init__(self, penalty):
            created.append(penalty)

        def __call__(self, tokens, logits):
            return logits

    monkeypatch.setattr(
        mlx_engine_module, "RepetitionPenaltyProcessor", _Recorder,
    )
    from test_api_behavior import _plain_engine

    eng, cs, messages, stored = _plain_engine([(101, "a"), (102, "b")], monkeypatch)
    chunks = _drive(eng, messages, repetition_penalty=0.0, max_tokens=2)
    assert chunks[-1].finish_reason in ("stop", "length")
    assert created == [], "non-positive penalty must be clamped to 1.0 (no-op)"


def test_u20_penalty_mapping_floored():
    """frequency+presence at the OpenAI minimum (-2/-2) used to map to
    repetition_penalty == 0.0 — the mapping is floored at 0.1 now."""
    from mlx_soloheaven.api.openai_compat import _sync_completion
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    captured = {}

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def complete(self, messages, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content="ok", thinking=None, tool_calls=None,
                finish_reason="stop", prompt_tokens=1, completion_tokens=1,
                cache_info=None,
            )

    req = ChatCompletionRequest(
        model="m", stream=False,
        frequency_penalty=-2.0, presence_penalty=-2.0,
        messages=[ChatMessage(role="user", content="hi")],
    )
    _sync_completion(req, _Eng())
    assert captured["repetition_penalty"] == pytest.approx(0.1)


def test_u20_settings_endpoint_validates(monkeypatch):
    from fastapi import HTTPException

    from mlx_soloheaven.api import settings as settings_mod

    saved = {}

    async def _fake_update(session_id, **kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(settings_mod.db, "update_session", _fake_update)

    with pytest.raises(HTTPException) as ei:
        asyncio.run(settings_mod.update_settings(
            "sid", settings_mod.SessionSettings(repetition_penalty=0.0),
        ))
    assert ei.value.status_code == 400
    assert "repetition_penalty" in str(ei.value.detail)

    with pytest.raises(HTTPException) as ei2:
        asyncio.run(settings_mod.update_settings(
            "sid", settings_mod.SessionSettings(temperature=-1.0),
        ))
    assert ei2.value.status_code == 400

    # Valid values persist.
    out = asyncio.run(settings_mod.update_settings(
        "sid", settings_mod.SessionSettings(temperature=0.7, top_k=40),
    ))
    assert out == {"ok": True}
    assert saved == {"temperature": 0.7, "top_k": 40}


# ---------------------------------------------------------------------------
# U27 — per-model Config carries the stream-coalescing flags
# ---------------------------------------------------------------------------


def test_u27_multimodel_config_carries_stream_coalesce():
    """FAILS PRE-FIX (in spirit): the in-process --models builder dropped
    stream_coalesce_n/ms, resetting them to the dataclass defaults (4/30)."""
    from mlx_soloheaven.server import build_per_model_config

    cfg = Config.from_args(parse_args([
        "--models", "/tmp/a", "/tmp/b",
        "--stream-coalesce-n", "9",
        "--stream-coalesce-ms", "77",
        "--engine-mode", "inprocess",
    ]))
    assert cfg.stream_coalesce_n == 9 and cfg.stream_coalesce_ms == 77
    for mcfg in cfg.models:
        model_cfg = build_per_model_config(cfg, mcfg)
        assert model_cfg.stream_coalesce_n == 9
        assert model_cfg.stream_coalesce_ms == 77


def test_u27_process_and_inprocess_configs_agree():
    """One shared builder: the two modes may only diverge on engine_mode
    (the exact class of drift that dropped the coalescing flags)."""
    from dataclasses import fields

    from mlx_soloheaven.server import build_per_model_config

    cfg = Config.from_args(parse_args([
        "--model", "/tmp/a",
        "--stream-coalesce-n", "9",
        "--stream-coalesce-ms", "77",
        "--repetition-penalty", "1.2",
        "--kv-bits", "8",
    ]))
    mcfg = cfg.models[0]
    inproc = build_per_model_config(cfg, mcfg)
    proc = build_per_model_config(cfg, mcfg, engine_mode="process")
    for f in fields(Config):
        if f.name == "engine_mode":
            continue
        assert getattr(inproc, f.name) == getattr(proc, f.name), f.name
    assert proc.engine_mode == "process"
    # Spot-check the globals both modes must carry.
    assert inproc.stream_coalesce_n == 9
    assert inproc.default_repetition_penalty == 1.2
    assert inproc.kv_bits == 8


# ---------------------------------------------------------------------------
# U19 — FSM index cache: tokenizer fingerprint + bounded LRU
# ---------------------------------------------------------------------------


class _FakeHF:
    def __init__(self, vocab: dict, name: str):
        self._vocab = vocab
        self.name_or_path = name

    def get_vocab(self):
        return dict(self._vocab)


class _FakeTok:
    """mlx-lm-TokenizerWrapper-shaped stub (inner HF tokenizer under
    ``_tokenizer``)."""

    def __init__(self, vocab: dict, name: str = "fake"):
        self._tokenizer = _FakeHF(vocab, name)


_VOCAB_A = {"a": 0, "b": 1, "<eos>": 2}
_VOCAB_B = {"x": 0, "y": 1, "z": 2, "<eos>": 3}


def test_u19_fingerprint_distinguishes_vocabs():
    fp_a = structured._tokenizer_fingerprint(_FakeTok(_VOCAB_A))
    fp_b = structured._tokenizer_fingerprint(_FakeTok(_VOCAB_B))
    assert fp_a != fp_b
    # Same vocab + name => same fingerprint (stable across instances).
    assert fp_a == structured._tokenizer_fingerprint(_FakeTok(_VOCAB_A))
    # Same vocab, different model path => different identity.
    fp_a2 = structured._tokenizer_fingerprint(_FakeTok(_VOCAB_A, name="other"))
    assert fp_a != fp_a2


def test_u19_fingerprint_memoized_on_instance():
    tok = _FakeTok(_VOCAB_A)
    fp1 = structured._tokenizer_fingerprint(tok)
    assert getattr(tok, structured._TOKENIZER_FP_ATTR) == fp1
    # Second call returns the memo (mutating the vocab is NOT observed —
    # vocabularies are immutable per model load, this just proves the memo).
    tok._tokenizer._vocab["new"] = 99
    assert structured._tokenizer_fingerprint(tok) == fp1


def test_u19_index_not_shared_across_tokenizers(monkeypatch):
    """FAILS PRE-FIX: the same cache_key served BOTH tokenizers, so the
    second model reused an FSM index compiled for the first model's vocab."""
    compiles: list = []

    monkeypatch.setattr(structured, "_get_vocab", lambda tok: (object(), 8))
    monkeypatch.setattr(structured, "_schema_to_regex", lambda schema: "R")
    monkeypatch.setattr(
        structured, "_compile_regex_index",
        lambda regex, vocab: compiles.append(regex) or object(),
    )
    monkeypatch.setattr(
        structured, "_LogitsProcessor",
        lambda index, description="": SimpleNamespace(index=index),
    )

    tok_a, tok_b = _FakeTok(_VOCAB_A), _FakeTok(_VOCAB_B)
    structured.build_json_schema_processor({"type": "object"}, tok_a, cache_key="k")
    structured.build_json_schema_processor({"type": "object"}, tok_b, cache_key="k")
    assert len(compiles) == 2, "each tokenizer must get its own index"
    # Same tokenizer again: cache HIT (no third compile).
    structured.build_json_schema_processor({"type": "object"}, tok_a, cache_key="k")
    assert len(compiles) == 2
    # build_regex_processor keys the same way.
    structured.build_regex_processor("R2", tok_a)
    structured.build_regex_processor("R2", tok_b)
    structured.build_regex_processor("R2", tok_a)
    assert len(compiles) == 4


def test_u19_index_cache_bounded_lru():
    cap = structured._INDEX_CACHE_MAX
    for i in range(cap + 10):
        structured._index_cache_put(f"key{i}", i)
    with structured._INDEX_CACHE_LOCK:
        assert len(structured._INDEX_CACHE) == cap
    # The 10 oldest were evicted; the newest survive.
    assert structured._index_cache_get("key0") is None
    assert structured._index_cache_get(f"key{cap + 9}") == cap + 9
    # A get refreshes recency: key10 (currently the oldest survivor) outlives
    # a subsequent insert that evicts one entry.
    assert structured._index_cache_get("key10") == 10
    structured._index_cache_put("fresh", "f")
    assert structured._index_cache_get("key10") == 10
    assert structured._index_cache_get("key11") is None  # the actual LRU fell


# ---------------------------------------------------------------------------
# U26 — drafter stats survive the per-turn SessionState reinstall
# ---------------------------------------------------------------------------


def _mtp_stats_engine(monkeypatch, scripts):
    """Plain-path harness engine (no drafter needed — the finalize stash is
    engine-agnostic) with N scripted turns."""
    stored = [1, 2, 3, 4, 5]
    from test_qwen_mtp import MockKV

    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, scripts)
    return eng, messages


def test_u26_stats_survive_turn_reinstall(monkeypatch):
    """FAILS PRE-FIX twice over: (1) the post-stream finalize ran BEFORE the
    install, so the brand-new SessionState never carried the stats it had
    just written to the replaced object; (2) turn 2 reset the accumulator."""
    eng, messages = _mtp_stats_engine(monkeypatch, [
        [(101, "a"), (102, "b")],
        [(111, "c")],
    ])

    # Turn 1: the drafter finalize (as _run_vlm stashes it) runs post-stream.
    eng._pending_drafter_finalize = (
        lambda: eng._accumulate_drafter_stats("s", 3, 5)
    )
    _drive(eng, messages, max_tokens=2)
    assert eng._sessions["s"].drafter_stats == {
        "requests": 1, "total_rounds": 3, "total_accepted": 5,
    }

    # Turn 2: accumulation continues across the reinstall.
    messages2 = list(eng._sessions["s"].messages) + [
        {"role": "user", "content": "u3"},
    ]
    eng._pending_drafter_finalize = (
        lambda: eng._accumulate_drafter_stats("s", 2, 4)
    )
    _drive(eng, messages2, max_tokens=2)
    assert eng._sessions["s"].drafter_stats == {
        "requests": 2, "total_rounds": 5, "total_accepted": 9,
    }
    # Single source of truth: the installed state points at the registry dict.
    assert eng._sessions["s"].drafter_stats is eng._session_drafter_stats["s"]


def test_u26_accumulate_without_live_session_then_install():
    """The finalize→install ordering: stats recorded while no SessionState
    exists yet (a new session's first turn) must be picked up at install."""
    eng = MLXEngine.__new__(MLXEngine)
    eng._sessions = {}
    eng._accumulate_drafter_stats("fresh", 4, 7)
    assert eng._session_drafter_stats["fresh"] == {
        "requests": 1, "total_rounds": 4, "total_accepted": 7,
    }
    # The install-site helper hands the accumulated dict to the new state.
    assert eng._drafter_stats_for("fresh") is eng._session_drafter_stats["fresh"]
    assert eng._drafter_stats_for("other") is None
    assert eng._drafter_stats_for(None) is None


def test_u26_delete_session_prunes_registry(tmp_path):
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config(data_dir=str(tmp_path))
    eng._lock = threading.Lock()
    eng._shutting_down = False
    eng._dirty_sessions = set()
    eng._dirty_lock = threading.Lock()
    eng._sessions = {}
    eng._session_drafter_stats = {"s": {"requests": 1}}
    eng.delete_session("s")
    assert eng._session_drafter_stats == {}


# ---------------------------------------------------------------------------
# U2 — base caches: byte-accounted, budget-charged, LRU-evicted
# ---------------------------------------------------------------------------


def _base_entry(h: str, nbytes: int, last_used: float) -> BaseCacheEntry:
    return BaseCacheEntry(
        system_hash=h,
        cache=[_StubCacheLayer(nbytes)],
        tokens=[1, 2],
        token_count=2,
        size_bytes=nbytes,
        last_used=last_used,
    )


def test_u2_registration_measures_size():
    """_register_base_cache byte-counts the snapshot with the same helper
    that sizes session caches for the budget."""
    eng = _make_engine(budget_gb=100.0)
    eng._base_caches = {}
    eng._eval_cache = lambda c: None
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    eng._register_base_cache(messages, [_StubCacheLayer(_GB)], [11, 12, 13])
    entry = next(iter(eng._base_caches.values()))
    assert entry.size_bytes == _GB
    assert entry.token_count == 3
    assert entry.last_used > 0


def test_u2_over_budget_evicts_lru_base_first(monkeypatch):
    """FAILS PRE-FIX: base caches were invisible to the budget — nothing was
    evicted. Post-fix the LRU base entry is shed (before any session)."""
    eng = _make_engine(budget_gb=2.5)
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)
    eng._sessions = {"S1": _make_session(_GB, last_used=100.0)}
    eng._base_caches = {
        "old": _base_entry("old", _GB, last_used=10.0),
        "new": _base_entry("new", _GB, last_used=20.0),
    }
    # 1GB session + 2GB base = 3GB > 2.5GB budget.
    assert eng._base_caches_memory_gb() == pytest.approx(2.0)

    eng._evict_active_sessions_if_needed()

    # LRU base "old" evicted; MRU base kept; the session untouched.
    assert set(eng._base_caches) == {"new"}
    assert set(eng._sessions) == {"S1"}
    total = eng._active_sessions_memory_gb() + eng._base_caches_memory_gb()
    assert total == pytest.approx(2.0)
    assert total <= 2.5


def test_u2_mru_base_protected_sessions_evicted_next(monkeypatch):
    """Conditional keep-MRU (round 2, codex F3): the single most-recently-used
    base entry is protected while it FITS the budget allowance (evicting it
    would re-prefill on the very next MISS — thrash); the sweep then falls
    through to the session LRU. Budget 2.5: allowance for the MRU base =
    2.5 - 1.0 (the MRU session, unevictable) = 1.5 GB >= the 1 GB entry."""
    eng = _make_engine(budget_gb=2.5)
    saved: list[str] = []

    def _fake_save(sid, session):
        saved.append(sid)
        return True

    monkeypatch.setattr(eng, "_save_session_to_disk", _fake_save)
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._base_caches = {"only": _base_entry("only", _GB, last_used=10.0)}

    eng._evict_active_sessions_if_needed()

    # The lone (== MRU, fitting) base entry survives; the LRU session was
    # evicted instead.
    assert set(eng._base_caches) == {"only"}
    assert saved == ["A"]
    assert set(eng._sessions) == {"B"}


def test_u2_clone_touches_lru_recency():
    eng = _make_engine(budget_gb=100.0)
    eng._eval_cache = lambda c: None
    entry = _base_entry("h", 1000, last_used=1.0)
    eng._clone_base_cache(entry)
    assert entry.hit_count == 1
    assert entry.last_used > 1.0


def test_u2_admin_overview_exposes_base_bytes(tmp_path):
    eng = _make_engine(budget_gb=100.0)
    eng.cfg.data_dir = str(tmp_path)
    eng.model_id = "m"
    eng._sessions = {}
    eng._base_caches = {"h": _base_entry("h", _GB, last_used=5.0)}
    ov = eng._cache_overview_locked()
    assert ov["base_caches"][0]["size_mb"] == pytest.approx(1000.0)
    assert ov["base_caches"][0]["last_used"] == 5.0
    assert ov["memory"]["base_caches_kv_gb"] == pytest.approx(1.0)
    assert ov["memory"]["base_cache_count"] == 1
    # Base bytes are part of what the sweep compares against the budget.
    assert ov["memory"]["total_kv_gb"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# U22 — forced </think> never lands mid multi-byte UTF-8 character
# ---------------------------------------------------------------------------

START, END = 100, 200


class _ByteTokenizer:
    """decode() over a token->bytes table with errors='replace' — mirrors a
    byte-level tokenizer's lossy decode of an incomplete multi-byte tail."""

    def __init__(self, table: dict[int, bytes]):
        self.table = table

    def decode(self, ids):
        return b"".join(self.table.get(i, b"?") for i in ids).decode(
            "utf-8", errors="replace",
        )


# '한' (U+D55C) = ED 95 9C — split across two tokens.
_TABLE = {
    1: b"p",        # prompt token
    10: b"ab",      # plain ASCII thinking token
    11: b"\xed",    # first byte of '한'
    12: b"\x95\x9c",  # remaining bytes of '한'
    13: b"\xff",    # forever-invalid byte (never completes)
    END: b"</think>",
}


def _step(proc, token_history):
    """One processor call; returns True iff the close was forced."""
    logits = mx.zeros((1, 512))
    out = proc(mx.array(token_history, dtype=mx.int32), logits)
    return float(out[0, END].item()) >= 1e8


def test_u22_force_deferred_until_char_completes():
    """FAILS PRE-FIX: the close was forced while the byte stream ended on the
    lead byte of '한' — reasoning_content gained a trailing U+FFFD."""
    proc = ThinkingBudgetProcessor(
        budget=3, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteTokenizer(_TABLE),
    )
    assert _step(proc, [1]) is False            # prompt call, count=1
    assert _step(proc, [1, 10]) is False        # count=2
    # Budget reached, but the tail ends mid-'한' -> DEFER (pre-fix: forced).
    assert _step(proc, [1, 10, 11]) is False
    assert proc._force_deferrals == 1
    # The char completed -> force fires now.
    assert _step(proc, [1, 10, 11, 12]) is True


def test_u22_deferral_bounded_never_loops():
    """A tail that never becomes valid UTF-8 is forced after at most 4
    deferrals — the deferral cannot loop."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteTokenizer(_TABLE),
    )
    history = [1]
    assert _step(proc, history) is False  # prompt call, count=1
    forced_at = None
    for i in range(8):
        history = history + [13]  # forever-invalid byte
        if _step(proc, history):
            forced_at = i
            break
    # count hits the budget (2) on the first generated token; 4 deferrals
    # follow; the 5th over-budget call forces regardless.
    assert forced_at == 4
    assert proc._force_deferrals == ThinkingBudgetProcessor._FORCE_DEFER_MAX


def test_u22_no_tokenizer_keeps_immediate_force():
    """Without a tokenizer the behaviour is byte-identical to the historical
    immediate force (also what the MTP helper parity tests pin down)."""
    proc = ThinkingBudgetProcessor(
        budget=3, think_end_token=END, think_start_token=START,
        model_family="chatml",
    )
    assert _step(proc, [1]) is False
    assert _step(proc, [1, 10]) is False
    assert _step(proc, [1, 10, 11]) is True  # forced right at the budget


def test_u22_accounting_untouched_and_rearmed_after_close():
    """Deferral only skips the logit override: thinking_tokens keeps
    counting, and a closed block re-arms the deferral budget."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteTokenizer(_TABLE),
    )
    _step(proc, [1])
    _step(proc, [1, 11])          # count=2 == budget, tail incomplete -> defer
    assert proc.thinking_tokens == 2
    assert proc._force_deferrals == 1
    _step(proc, [1, 11, 12])      # char complete -> forced
    _step(proc, [1, 11, 12, END])  # model emits the close
    assert proc.in_thinking is False
    assert proc._force_deferrals == 0  # re-armed for a later block


# ===========================================================================
# Batch-5 ROUND 2 (codex findings F1-F7)
# ===========================================================================

import time

from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine import thinking as thinking_module
from mlx_soloheaven.engine.thinking import ForceDeferralGate, tail_ends_mid_utf8

from test_mtp_thinking_budget import (
    _argmax_sampler,
    _bare_vlm_engine as _mtp_bare_vlm_engine,
    _clear_stash as _mtp_clear_stash,
    _drive_scripted_clone,
    _make_clone,
    _set_budget_stash,
    _ScriptedLM,
)


# ---------------------------------------------------------------------------
# F1 — [U20] oversized top_k clamped BEFORE the mlx-vlm backend dispatch
# ---------------------------------------------------------------------------


class _LenTokenizer:
    """Vocab-size surface for _sampling_vocab_size (len == vocab)."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n


def test_f1_clamp_helper_maps_keepall_to_sentinel():
    eng = MLXEngine.__new__(MLXEngine)
    eng.tokenizer = _LenTokenizer(8)
    # Excess (>= vocab) becomes the upstream keep-all sentinel 0.
    assert eng._clamp_vlm_top_k(8) == 0
    assert eng._clamp_vlm_top_k(10**9) == 0
    # In-range values pass through; 0/negative/None normalize to 0.
    assert eng._clamp_vlm_top_k(5) == 5
    assert eng._clamp_vlm_top_k(0) == 0
    assert eng._clamp_vlm_top_k(-3) == 0
    assert eng._clamp_vlm_top_k(None) == 0
    # No vocab surface -> value passes through unchanged (cannot validate).
    eng.tokenizer = None
    assert eng._clamp_vlm_top_k(10**9) == 10**9


def test_f1_run_vlm_clamps_oversized_top_k(monkeypatch):
    """FAILS PRE-FIX: the raw top_k (e.g. 10**9) reached mlx-vlm, whose
    make_sampler/apply_top_k raises for top_k >= vocab MID-STREAM. The clamp
    lives at the single vlm-path kwargs-assembly point (_run_vlm)."""
    eng = _mtp_bare_vlm_engine()
    try:
        eng._has_rotating_cache = False
        eng._sliding_window_size = 0
        eng.tokenizer = _LenTokenizer(8)

        from mlx_vlm.generate import PromptCacheState

        captured = {}

        def _fake_stream(*_args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        def _drive(top_k):
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=0)]
            cs.token_ids = []
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[1, 2, 3], max_tokens=4,
                temperature=1.0, top_p=1.0, min_p=0.0, top_k=top_k,
                logits_processors=None, session_id="s-topk",
                total_prompt_tokens=3,
            )
            list(gen)

        eng._vlm_executor.submit(_drive, 10**9).result(timeout=10)
        assert captured["top_k"] == 0, (
            f"oversized top_k must be clamped to the keep-all sentinel, "
            f"got {captured['top_k']}"
        )
        # In-range top_k is forwarded unchanged.
        eng._vlm_executor.submit(_drive, 5).result(timeout=10)
        assert captured["top_k"] == 5
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# F2 — [U19] canonical fingerprint (no collisions), EOS ids in the key,
#      full digest, single-flight compile
# ---------------------------------------------------------------------------


def test_f2_fingerprint_collision_pair_distinct():
    """FAILS PRE-FIX: delimiter-free concatenation made these two distinct
    vocabularies hash identically (codex reproduced)."""
    fp1 = structured._tokenizer_fingerprint(_FakeTok({"a": 0, "0b": 1}))
    fp2 = structured._tokenizer_fingerprint(_FakeTok({"a0": 0, "b": 1}))
    assert fp1 != fp2


def test_f2_fingerprint_full_sha256_digest():
    fp = structured._tokenizer_fingerprint(_FakeTok(_VOCAB_A))
    assert len(fp) == 64  # full sha256 hexdigest, no 64-bit truncation


class _FakeTokWithEos(_FakeTok):
    def __init__(self, vocab, name="fake", eos_token_ids=None):
        super().__init__(vocab, name)
        self.eos_token_ids = eos_token_ids or []


def test_f2_fingerprint_includes_eos_ids():
    """FAILS PRE-FIX: _get_vocab EXCLUDES the EOS ids from the compiled
    vocabulary, but the ids were absent from the key — two tokenizers with
    the same vocab and different EOS config shared one index."""
    fp1 = structured._tokenizer_fingerprint(
        _FakeTokWithEos(_VOCAB_A, eos_token_ids=[2])
    )
    fp2 = structured._tokenizer_fingerprint(
        _FakeTokWithEos(_VOCAB_A, eos_token_ids=[1])
    )
    assert fp1 != fp2
    # And differs from the no-EOS variant too.
    fp3 = structured._tokenizer_fingerprint(_FakeTok(_VOCAB_A))
    assert fp3 not in (fp1, fp2)


def test_f2_hf_eos_scalar_also_keyed():
    """The HF-side eos_token_id (scalar) is part of the identity as well."""
    tok1, tok2 = _FakeTok(_VOCAB_A), _FakeTok(_VOCAB_A)
    tok1._tokenizer.eos_token_id = 2
    tok2._tokenizer.eos_token_id = 1
    assert (
        structured._tokenizer_fingerprint(tok1)
        != structured._tokenizer_fingerprint(tok2)
    )


def test_f2_single_flight_compile(monkeypatch):
    """Simultaneous misses on the same key run exactly ONE compile (the
    second requester waits and takes the cached result)."""
    compiles: list = []

    def _slow_compile(regex, vocab):
        compiles.append(regex)
        time.sleep(0.3)
        return object()

    monkeypatch.setattr(structured, "_get_vocab", lambda tok: (object(), 8))
    monkeypatch.setattr(structured, "_compile_regex_index", _slow_compile)
    monkeypatch.setattr(
        structured, "_LogitsProcessor",
        lambda index, description="": SimpleNamespace(index=index),
    )

    tok = _FakeTok(_VOCAB_A)
    results = []

    def _build():
        results.append(structured.build_regex_processor("RX", tok))

    threads = [threading.Thread(target=_build) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(compiles) == 1, f"expected single-flight compile, got {compiles}"
    assert results[0].index is results[1].index
    # The per-key lock registry is pruned after the compile.
    assert structured._INDEX_COMPILE_LOCKS == {}


# ---------------------------------------------------------------------------
# F3 — [U2] MRU base-cache protection is CONDITIONAL on fitting the budget
# ---------------------------------------------------------------------------


def test_f3_lone_oversize_base_entry_evicted(caplog):
    """FAILS PRE-FIX: a single base entry larger than the whole budget was
    unconditionally MRU-protected — zero eviction candidates, and the next
    clone could OOM. It must be evicted (loud WARNING)."""
    eng = _make_engine(budget_gb=2.0)
    eng._sessions = {}
    eng._base_caches = {"big": _base_entry("big", 3 * _GB, last_used=10.0)}

    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="mlx_soloheaven.engine.mlx_engine"):
        eng._evict_active_sessions_if_needed()

    assert eng._base_caches == {}, "oversize lone base entry must be evicted"
    assert any(
        "EVICTED MRU" in rec.message for rec in caplog.records
    ), "the waived MRU protection must be logged loudly"


def test_f3_mru_base_evicted_when_protected_session_leaves_no_allowance(monkeypatch):
    """The allowance accounts for the session(s) the sweep can never evict:
    budget 2.5, protected/MRU session 2GB -> allowance 0.5GB < the 1GB MRU
    base entry -> the base is shed even though it is smaller than the raw
    budget."""
    eng = _make_engine(budget_gb=2.5)
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)
    eng._sessions = {"S": _make_session(2 * _GB, last_used=100.0)}
    eng._base_caches = {"b": _base_entry("b", _GB, last_used=10.0)}

    eng._evict_active_sessions_if_needed()

    assert eng._base_caches == {}, (
        "MRU base must be evicted when the unevictable session leaves no "
        "allowance for it"
    )
    # The lone session itself is never evicted (keep-MRU session rule).
    assert set(eng._sessions) == {"S"}


def test_f3_direct_call_without_allowance_keeps_historical_protection():
    """_evict_base_caches_lru without budget context (mru_allowance_gb=None)
    keeps the unconditional keep-MRU behaviour — only the budget sweep,
    which KNOWS the allowance, may waive it."""
    eng = _make_engine(budget_gb=0.5)
    eng._base_caches = {"big": _base_entry("big", 3 * _GB, last_used=10.0)}
    evicted, freed = eng._evict_base_caches_lru(lambda: True)
    assert evicted == 0 and freed == 0
    assert set(eng._base_caches) == {"big"}


def test_f3_fitting_mru_base_still_protected(monkeypatch):
    """Anti-thrash intent preserved: when the MRU base FITS the allowance it
    survives the sweep even while the total is still over budget."""
    eng = _make_engine(budget_gb=2.5)
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)
    # Session 1GB (MRU, unevictable), base 1GB: allowance 1.5 >= 1 -> keep.
    # Total 2GB <= 2.5 after nothing is evicted, but force the check by
    # adding a second session that the sweep evicts instead.
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._base_caches = {"fit": _base_entry("fit", _GB, last_used=10.0)}

    eng._evict_active_sessions_if_needed()

    assert set(eng._base_caches) == {"fit"}
    assert set(eng._sessions) == {"B"}


# ---------------------------------------------------------------------------
# F4b — [U22] genuine U+FFFD vs truncation: byte-tail check (plain path)
# ---------------------------------------------------------------------------

# Byte-level-BPE-shaped fake: token strings use the standard GPT-2
# bytes_to_unicode mapping (the mode probe sees "Ġ" in the vocab), so the
# helper can recover exact raw bytes per token.


class _ByteLevelTok:
    def __init__(self, table: "dict[int, bytes]"):
        enc = {b: c for c, b in thinking_module._gpt2_byte_decoder().items()}
        self._table = dict(table)
        self._toks = {i: "".join(enc[x] for x in bs) for i, bs in table.items()}

    def get_vocab(self):
        v = {s: i for i, s in self._toks.items()}
        v["Ġ"] = 10_000  # byte-level marker for the mode probe
        return v

    def convert_ids_to_tokens(self, ids):
        return [self._toks.get(int(i), "") for i in ids]

    def decode(self, ids):
        # Lossy decode (errors='replace') — the surface the PRE-FIX heuristic
        # consumed. Present so the fails-pre-fix tests exercise the exact old
        # decode-endswith-U+FFFD behaviour, not an exception fail-open.
        return b"".join(self._table.get(int(i), b"?") for i in ids).decode(
            "utf-8", errors="replace",
        )


# '한' = ED 95 9C split across tokens 11/12; 20 = a GENUINE literal U+FFFD
# (EF BF BD — a complete sequence); 13 = an invalid lone 0xFF.
_BL_TABLE = {
    1: b"p",
    10: b"ab",
    11: b"\xed",
    12: b"\x95\x9c",
    13: b"\xff",
    20: b"\xef\xbf\xbd",
    END: b"</think>",
}


def test_f4b_tail_helper_byte_semantics():
    tok = _ByteLevelTok(_BL_TABLE)
    # Valid-but-incomplete multi-byte tail -> mid-character.
    assert tail_ends_mid_utf8(tok, [10, 11]) is True
    # Completed char -> not mid-character.
    assert tail_ends_mid_utf8(tok, [10, 11, 12]) is False
    # GENUINE literal U+FFFD (complete EF BF BD) -> not mid-character.
    assert tail_ends_mid_utf8(tok, [10, 20]) is False
    # Invalid byte (can never complete) -> not mid-character.
    assert tail_ends_mid_utf8(tok, [10, 13]) is False
    # Plain ASCII tail.
    assert tail_ends_mid_utf8(tok, [10]) is False


def test_f4b_genuine_replacement_char_not_deferred():
    """FAILS PRE-FIX: the decode-endswith-U+FFFD heuristic treated a model
    LEGITIMATELY emitting a literal replacement char as an incomplete tail
    and granted up to 4 unintended extra tokens. The byte-tail check sees a
    complete EF BF BD sequence and forces immediately."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteLevelTok(_BL_TABLE),
    )
    assert _step(proc, [1]) is False          # prompt call, count=1
    # Budget reached; tail is a GENUINE '<?>' (complete bytes) -> force NOW.
    assert _step(proc, [1, 20]) is True
    assert proc._force_deferrals == 0


def test_f4b_split_multibyte_still_defers():
    """The real target of U22 keeps working under the sharpened check: a
    split multi-byte char defers, then forces once complete."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteLevelTok(_BL_TABLE),
    )
    assert _step(proc, [1]) is False
    assert _step(proc, [1, 11]) is False      # mid-'한' -> defer
    assert proc._force_deferrals == 1
    assert _step(proc, [1, 11, 12]) is True   # complete -> force


def test_f4b_invalid_byte_forces_immediately():
    """An invalid byte tail (0xFF) can NEVER complete — waiting is pointless,
    so the byte-aware check forces at once (the legacy decode fallback
    deferred it up to the bound)."""
    proc = ThinkingBudgetProcessor(
        budget=2, think_end_token=END, think_start_token=START,
        model_family="chatml", tokenizer=_ByteLevelTok(_BL_TABLE),
    )
    assert _step(proc, [1]) is False
    assert _step(proc, [1, 13]) is True
    assert proc._force_deferrals == 0


class _SpmTok:
    """SentencePiece-shaped fake: byte-fallback pieces are '<0xHH>'."""

    def __init__(self, table: "dict[int, str]"):
        self._toks = dict(table)

    def get_vocab(self):
        v = {s: i for i, s in self._toks.items()}
        v["<0x00>"] = 10_000  # byte-fallback marker for the mode probe
        return v

    def convert_ids_to_tokens(self, ids):
        return [self._toks.get(int(i), "") for i in ids]


def test_f4b_spm_byte_fallback_semantics():
    tok = _SpmTok({
        1: "▁hi",       # plain piece (metaspace + text): complete
        11: "<0xED>",
        12: "<0x95>",
        13: "<0x9C>",
        20: "�",        # a genuine literal U+FFFD piece
    })
    assert tail_ends_mid_utf8(tok, [1, 11]) is True          # lead byte
    assert tail_ends_mid_utf8(tok, [1, 11, 12]) is True      # still short
    assert tail_ends_mid_utf8(tok, [1, 11, 12, 13]) is False # complete '한'
    assert tail_ends_mid_utf8(tok, [1, 20]) is False         # genuine
    assert tail_ends_mid_utf8(tok, [1]) is False


def test_f4b_decode_only_tokenizer_keeps_legacy_fallback():
    """No byte surface (decode-only tokenizer) -> the legacy heuristic
    remains: trailing U+FFFD defers (bounded upstream)."""
    tok = _ByteTokenizer(_TABLE)
    assert tail_ends_mid_utf8(tok, [1, 11]) is True   # '...<?>' via decode
    assert tail_ends_mid_utf8(tok, [1, 11, 12]) is False


def test_f4b_gate_bound_and_rearm():
    gate = ForceDeferralGate(_ByteLevelTok(_BL_TABLE), max_deferrals=2)
    assert gate.should_defer([1, 11]) is True
    assert gate.should_defer([1, 11]) is True
    assert gate.should_defer([1, 11]) is False  # bound hit -> force
    gate.rearm()
    assert gate.should_defer([1, 11]) is True
    # None tokenizer disables deferral entirely.
    assert ForceDeferralGate(None).should_defer([1, 11]) is False


# ---------------------------------------------------------------------------
# F4a — [U22] the mlx-vlm MTP clone's force site defers on a mid-byte tail
# ---------------------------------------------------------------------------

ED_TOK, REST_TOK, BAD_TOK = 11, 12, 13


def test_f4a_mtp_force_site_defers_on_mid_byte():
    """FAILS PRE-FIX: the clone's force_end_from_state site forced think_end
    with NO byte-state check — the close landed mid-'한' (REST_TOK never
    emitted, U+FFFD stamped into reasoning_content). With the stashed
    tokenizer the gate defers one round so the char completes first."""
    clone = _make_clone()
    try:
        # Chain: START -> ED -> REST -> CONTENT (loops). budget=2: the force
        # decision first fires while the tail ends on the lead byte of '한'.
        trans = {START: ED_TOK, ED_TOK: REST_TOK, REST_TOK: 1, 1: 1}
        _set_budget_stash(budget=2, family="gemma4")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_TABLE)
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=16, first_bonus=START
        )
        assert END in emitted, f"forced close must still land: {emitted}"
        assert REST_TOK in emitted, (
            f"the char-completing token must be emitted BEFORE the forced "
            f"close (pre-fix it never was): {emitted}"
        )
        assert emitted.index(REST_TOK) < emitted.index(END), emitted
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_f4a_mtp_force_site_hard_forces_after_bound():
    """A tail that never becomes valid UTF-8 cannot stall the close: the
    gate hard-forces after at most 4 deferrals (bounded, never loops)."""
    clone = _make_clone()
    try:
        trans = {START: BAD_TOK, BAD_TOK: BAD_TOK}
        _set_budget_stash(budget=2, family="gemma4")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_TABLE)
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=64, first_bonus=START
        )
        assert END in emitted, (
            f"the close must be hard-forced after the deferral bound: {emitted}"
        )
        # budget 2 + at most 4 deferred tokens + block-boundary slack.
        assert emitted.index(END) <= 10, emitted
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_f4a_no_tokenizer_stash_keeps_immediate_force():
    """Without a stashed tokenizer the clone's force is byte-identical to
    the historical immediate force."""
    clone = _make_clone()
    try:
        trans = {START: ED_TOK, ED_TOK: REST_TOK, REST_TOK: 1, 1: 1}
        _set_budget_stash(budget=2, family="gemma4")  # tokenizer stays None
        emitted = _drive_scripted_clone(
            clone, trans, max_tokens=16, first_bonus=START
        )
        assert END in emitted
        # Immediate force: END lands before the char completes.
        assert REST_TOK not in emitted[: emitted.index(END)]
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_f4a_run_vlm_stashes_tokenizer(monkeypatch):
    """_run_vlm must thread the ThinkingBudgetProcessor's tokenizer onto
    _MTP_THINK_TOKENIZER (and clear it when no processor is present)."""
    eng = _mtp_bare_vlm_engine()
    try:
        sentinel_drafter = SimpleNamespace(accept_lens=[])
        eng._drafter = sentinel_drafter
        eng._draft_kind = "mtp"
        eng._has_rotating_cache = False
        eng._sliding_window_size = 0

        from mlx_vlm.generate import PromptCacheState

        captured = {}

        def _fake_stream(*_args, **kwargs):
            captured["tok"] = mlx_engine_module._MTP_THINK_TOKENIZER
            return iter([])

        monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)
        monkeypatch.setattr(mlx_engine_module, "_MTP_WRAP_GATE", False)

        sentinel_tok = _ByteTokenizer(_TABLE)
        tbp = ThinkingBudgetProcessor(
            budget=200, think_end_token=END, think_start_token=START,
            model_family="gemma4", tokenizer=sentinel_tok,
        )

        def _drive_vlm(procs):
            cs = PromptCacheState()
            cs.cache = [SimpleNamespace(offset=0)]
            cs.token_ids = []
            gen = eng._run_vlm(
                cache_state=cs, prompt_token_ids=[1, 2, 3], max_tokens=4,
                temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
                logits_processors=procs, session_id="s-tok",
                total_prompt_tokens=3,
            )
            list(gen)

        eng._vlm_executor.submit(_drive_vlm, [tbp]).result(timeout=10)
        assert captured["tok"] is sentinel_tok

        captured.clear()
        eng._vlm_executor.submit(_drive_vlm, None).result(timeout=10)
        assert captured["tok"] is None
    finally:
        _mtp_clear_stash()
        eng.close()


# ---------------------------------------------------------------------------
# F5 — [U26] drafter registry bounded; web delete prunes; readers snapshot
# ---------------------------------------------------------------------------


def _shell_engine_for_stats() -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng._sessions = {}
    return eng


def test_f5_registry_bounded_lru():
    """FAILS PRE-FIX: one entry per unique session id for the life of the
    process — the registry grew without bound."""
    from mlx_soloheaven.engine.mlx_engine import _DRAFTER_STATS_MAX

    eng = _shell_engine_for_stats()
    for i in range(_DRAFTER_STATS_MAX + 10):
        eng._accumulate_drafter_stats(f"s{i}", 1, 1)
    assert len(eng._session_drafter_stats) == _DRAFTER_STATS_MAX
    # The 10 oldest fell; the newest survive.
    assert "s0" not in eng._session_drafter_stats
    assert "s9" not in eng._session_drafter_stats
    assert f"s{_DRAFTER_STATS_MAX + 9}" in eng._session_drafter_stats


def test_f5_registry_lru_touch_refreshes():
    """Accumulating for an existing session refreshes its recency — active
    sessions are never the ones evicted."""
    from mlx_soloheaven.engine.mlx_engine import _DRAFTER_STATS_MAX

    eng = _shell_engine_for_stats()
    for i in range(_DRAFTER_STATS_MAX):
        eng._accumulate_drafter_stats(f"s{i}", 1, 1)
    # Touch the oldest, then overflow by one: the SECOND-oldest must fall.
    eng._accumulate_drafter_stats("s0", 1, 1)
    eng._accumulate_drafter_stats("fresh", 1, 1)
    assert "s0" in eng._session_drafter_stats
    assert "s1" not in eng._session_drafter_stats
    assert eng._session_drafter_stats["s0"]["requests"] == 2


def test_f5_web_delete_prunes_engine_state(monkeypatch):
    """FAILS PRE-FIX: the web UI delete route removed only the SQLite rows —
    the engine session (resident KV + drafter-stats registry entry + disk
    cache file) leaked. The route now calls the engine's delete_session on
    every engine, then deletes the DB rows."""
    from mlx_soloheaven.api import chat as chat_mod

    class _EngStub:
        def __init__(self):
            self.deleted: list[str] = []

        def delete_session(self, sid):
            self.deleted.append(sid)
            return True

    stub = _EngStub()
    db_deleted: list[str] = []

    async def _fake_db_delete(sid):
        db_deleted.append(sid)

    monkeypatch.setattr(chat_mod, "engine", stub)
    monkeypatch.setattr(chat_mod, "_engines", {})
    monkeypatch.setattr(chat_mod.db, "delete_session", _fake_db_delete)

    out = asyncio.run(chat_mod.delete_session("web-sid"))
    assert out == {"ok": True}
    assert stub.deleted == ["web-sid"], "engine session must be deleted too"
    assert db_deleted == ["web-sid"]


def test_f5_web_delete_engine_failure_is_log_only(monkeypatch):
    """A failing engine delete (busy/dead child) must not block the DB
    delete — the stale engine state degrades to an honest MISS later."""
    from mlx_soloheaven.api import chat as chat_mod

    class _BrokenEng:
        def delete_session(self, sid):
            raise RuntimeError("child dead")

    db_deleted: list[str] = []

    async def _fake_db_delete(sid):
        db_deleted.append(sid)

    monkeypatch.setattr(chat_mod, "engine", _BrokenEng())
    monkeypatch.setattr(chat_mod, "_engines", {})
    monkeypatch.setattr(chat_mod.db, "delete_session", _fake_db_delete)

    out = asyncio.run(chat_mod.delete_session("web-sid"))
    assert out == {"ok": True}
    assert db_deleted == ["web-sid"]


def test_f5_web_delete_multi_engine(monkeypatch):
    """Multi-model deployments: the session may live on ANY engine — all of
    them are pruned (same merge the list_sessions enrichment uses)."""
    from mlx_soloheaven.api import chat as chat_mod

    class _EngStub:
        def __init__(self):
            self.deleted: list[str] = []

        def delete_session(self, sid):
            self.deleted.append(sid)
            return True

    a, b = _EngStub(), _EngStub()

    async def _fake_db_delete(sid):
        pass

    monkeypatch.setattr(chat_mod, "engine", a)
    monkeypatch.setattr(chat_mod, "_engines", {"a": a, "b": b})
    monkeypatch.setattr(chat_mod.db, "delete_session", _fake_db_delete)

    asyncio.run(chat_mod.delete_session("web-sid"))
    assert a.deleted == ["web-sid"]
    assert b.deleted == ["web-sid"]


def test_f5_readers_return_stat_snapshots():
    """FAILS PRE-FIX: list_sessions/get_session returned the SHARED mutable
    stats dict by reference — a JSON serialization after lock release could
    observe torn counter updates. They now return shallow copies."""
    eng = MLXEngine.__new__(MLXEngine)
    eng._lock = threading.Lock()
    eng._sessions = {}
    eng._accumulate_drafter_stats("s", 3, 5)
    state = SessionState(
        cache_state=None, messages=[], total_cache_tokens=0, last_used=1.0,
    )
    state.drafter_stats = eng._session_drafter_stats["s"]
    eng._sessions["s"] = state

    listed = eng.list_sessions()[0]["drafter_stats"]
    fetched = eng.get_session("s")["drafter_stats"]
    assert listed == {"requests": 1, "total_rounds": 3, "total_accepted": 5}
    assert listed is not state.drafter_stats
    assert fetched is not state.drafter_stats

    # Mutating the live dict AFTER the read must not alter the snapshots.
    eng._accumulate_drafter_stats("s", 2, 4)
    assert listed["total_rounds"] == 3
    assert fetched["total_accepted"] == 5


# ---------------------------------------------------------------------------
# F6 — [U20] non-finite floats rejected at every boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", [
    "temperature", "top_p", "min_p", "repetition_penalty",
    "frequency_penalty", "presence_penalty",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_f6_non_finite_rejected_400(field, value):
    """FAILS PRE-FIX: NaN/Infinity (accepted by Python json via Starlette and
    by pydantic floats) passed every range comparison (nan < 0 is False)."""
    from mlx_soloheaven.api.openai_compat import _validate_sampling_params
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        **{field: value},
    )
    resp = _validate_sampling_params(req)
    assert resp is not None, f"{field}={value} must be rejected"
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert field in body["error"]["message"]


def test_f6_settings_patch_rejects_non_finite(monkeypatch):
    from fastapi import HTTPException

    from mlx_soloheaven.api import settings as settings_mod

    async def _fake_update(session_id, **kwargs):  # pragma: no cover
        raise AssertionError("must not persist non-finite values")

    monkeypatch.setattr(settings_mod.db, "update_session", _fake_update)

    for field, value in (
        ("temperature", float("nan")),
        ("top_p", float("inf")),
        ("min_p", float("nan")),
        ("repetition_penalty", float("inf")),
    ):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(settings_mod.update_settings(
                "sid", settings_mod.SessionSettings(**{field: value}),
            ))
        assert ei.value.status_code == 400
        assert field in str(ei.value.detail)


def test_f6_engine_clamps_nan_penalty(monkeypatch):
    """FAILS PRE-FIX: NaN slipped past the ``<= 0`` clamp (nan <= 0 is False)
    and ``nan != 1.0`` built a poisoned RepetitionPenaltyProcessor."""
    from mlx_soloheaven.engine import mlx_engine as _mod

    created: list[float] = []

    class _Recorder:
        def __init__(self, penalty):
            created.append(penalty)

        def __call__(self, tokens, logits):
            return logits

    monkeypatch.setattr(_mod, "RepetitionPenaltyProcessor", _Recorder)
    from test_api_behavior import _plain_engine

    eng, cs, messages, stored = _plain_engine([(101, "a"), (102, "b")], monkeypatch)
    chunks = _drive(eng, messages, repetition_penalty=float("nan"), max_tokens=2)
    assert chunks[-1].finish_reason in ("stop", "length")
    assert created == [], "non-finite penalty must be clamped to 1.0 (no-op)"


# ---------------------------------------------------------------------------
# F7 — SessionSettings sampling params persist against a REAL sqlite DB
# ---------------------------------------------------------------------------


@pytest.fixture()
def _real_db(tmp_path):
    """Point the storage layer at a REAL temp sqlite file (no mocks);
    restores the previous path afterwards."""
    from mlx_soloheaven.storage import database as db_mod

    old_path = db_mod._db_path
    db_mod.set_db_path(str(tmp_path / "test.db"))
    yield db_mod
    db_mod.set_db_path(old_path)


def test_f7_fresh_db_patch_persists_sampling_params(_real_db):
    """FAILS PRE-FIX: the fresh schema had no top_p/min_p/top_k/
    repetition_penalty columns — update_session's dynamic UPDATE raised
    'no such column' (the round-1 test mocked db.update_session and missed
    it)."""
    from mlx_soloheaven.api import settings as settings_mod

    async def _run():
        await _real_db.init_db()
        row = await _real_db.create_session(title="t")
        sid = row["id"]
        out = await settings_mod.update_settings(
            sid,
            settings_mod.SessionSettings(
                top_p=0.9, min_p=0.05, top_k=40, repetition_penalty=1.2,
            ),
        )
        assert out == {"ok": True}
        return await _real_db.get_session(sid)

    session = asyncio.run(_run())
    assert session["top_p"] == pytest.approx(0.9)
    assert session["min_p"] == pytest.approx(0.05)
    assert session["top_k"] == 40
    assert session["repetition_penalty"] == pytest.approx(1.2)


def test_f7_fresh_db_defaults_are_null(_real_db):
    """NULL = 'not customized' — a fresh session leaves the sampling columns
    NULL so the chat endpoint falls back to the engine config default."""
    async def _run():
        await _real_db.init_db()
        row = await _real_db.create_session(title="t")
        return await _real_db.get_session(row["id"])

    session = asyncio.run(_run())
    for col in ("top_p", "min_p", "top_k", "repetition_penalty"):
        assert col in session, f"column {col} missing from fresh schema"
        assert session[col] is None


def test_f7_legacy_db_migrated_then_patch_works(_real_db, tmp_path):
    """A legacy DB (pre-sampling-columns sessions table) is migrated by the
    established try-ALTER idiom on init_db; the PATCH then persists."""
    import sqlite3 as _sqlite3

    from mlx_soloheaven.api import settings as settings_mod

    # Build the legacy schema DIRECTLY (no init_db yet): the original
    # minimal sessions/messages tables without any sampling columns.
    legacy_path = str(tmp_path / "test.db")  # same file _real_db points at
    conn = _sqlite3.connect(legacy_path)
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            system_prompt TEXT DEFAULT '',
            created_at REAL,
            updated_at REAL
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            created_at REAL
        );
    """)
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) "
        "VALUES ('legacy1', 'old', 1.0, 1.0)"
    )
    conn.commit()
    conn.close()

    async def _run():
        await _real_db.init_db()  # try-ALTER migration adds the columns
        out = await settings_mod.update_settings(
            "legacy1",
            settings_mod.SessionSettings(top_p=0.8, top_k=20),
        )
        assert out == {"ok": True}
        return await _real_db.get_session("legacy1")

    session = asyncio.run(_run())
    assert session["top_p"] == pytest.approx(0.8)
    assert session["top_k"] == 20
    # Untouched sampling columns stay NULL (config-default fallback).
    assert session["min_p"] is None
    assert session["repetition_penalty"] is None


def test_f7_update_session_dynamic_set_builds_for_new_columns(_real_db):
    """db.update_session's dynamic SET covers the new columns directly."""
    async def _run():
        await _real_db.init_db()
        row = await _real_db.create_session(title="t")
        await _real_db.update_session(
            row["id"], top_p=0.7, min_p=0.2, top_k=5, repetition_penalty=1.1,
        )
        return await _real_db.get_session(row["id"])

    session = asyncio.run(_run())
    assert session["top_p"] == pytest.approx(0.7)
    assert session["min_p"] == pytest.approx(0.2)
    assert session["top_k"] == 5
    assert session["repetition_penalty"] == pytest.approx(1.1)


# ===========================================================================
# Batch-5 ROUND 4 (codex round-3 residual P2 findings)
# ===========================================================================
#
# R4-F1 — [U22/F4a] REJECTED speculative positions consumed the shared UTF-8
#         deferral budget: the MTP verifier consulted the MUTATING
#         ForceDeferralGate.should_defer() for EVERY provisional position,
#         but acceptance is decided only after the walk — positions
#         discarded at the first draft/target mismatch still exhausted the
#         4-deferral budget, and a later REAL mid-character tail was
#         hard-forced into '�</think>'. The gate is now PREVIEWED per
#         provisional position (should_defer_preview, pure) and COMMITTED
#         (commit_deferral) only for the positions actually emitted
#         (accepted prefix + target bonus).
# R4-F2 — the non-streaming web /chat ignored every resolved session
#         sampling setting: the route resolved
#         temperature/top_p/min_p/top_k/repetition_penalty/thinking_budget/
#         max_tokens but _sync_chat called eng.complete() with only
#         session_id, so the engine fell back to the config defaults.


# --- R4-F1: diverging-drafter probes (codex's deterministic repro) --------

# A 4-byte UTF-8 char split across single-byte tokens: F0 needs three
# continuation bytes (F0 90 90 90 decodes to a valid astral-plane char).
# The TARGET continues the char with 0x90 bytes; the DRAFTER diverges after
# the lead byte and proposes 0x91 bytes, so every later block REJECTS at
# its first draft position (accepted=0, only the target bonus is emitted).
F0_TOK, CONT_TOK, WRONG_TOK, INVALID_TOK = 80, 90, 91, 13

_DIVERGE_TABLE = {
    1: b"p",              # plain content filler
    F0_TOK: b"\xf0",      # lead byte of a 4-byte char
    CONT_TOK: b"\x90",    # continuation byte (the target's real chain)
    WRONG_TOK: b"\x91",   # continuation byte (the drafter's wrong guess)
    INVALID_TOK: b"\xff",  # forever-invalid byte (never completes)
    START: b"<think>",
    END: b"</think>",
}


class _DivergingDraftModel:
    """Drafter following its OWN transition chain — unlike the always
    target-agreeing ``_ScriptedDraftModel``, it diverges from the target
    wherever the two maps differ, exercising the reject path the round-2
    regression tests never reached."""

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
            nxt = self.trans.get(prev, 1)
            toks.append(nxt)
            prev = nxt
        return mx.array([toks], dtype=token_dtype)


def _drive_diverging_clone(
    clone, target_trans, draft_trans, *, max_tokens, first_bonus
):
    model = SimpleNamespace(language_model=_ScriptedLM(target_trans))
    draft = _DivergingDraftModel(draft_trans, block_size=4)
    cache = [SimpleNamespace(offset=0)]
    gen = clone(
        model, draft, cache, mx.zeros((1, 1, 4)), {},
        first_bonus=first_bonus, max_tokens=max_tokens,
        sampler=_argmax_sampler, draft_block_size=4, token_dtype=mx.int32,
    )
    return [int(tok) for tok, _ in gen]


def test_r4f1_rejected_positions_do_not_consume_deferral_budget():
    """FAILS PRE-FIX (codex round-3 P2, finding 1): the drafter diverges at
    position 0 of every post-divergence block, so at most ONE token per
    block is emitted — yet the pre-fix gate consumed a deferral for every
    PROVISIONAL verify position it consulted. The hypothetical rejected
    positions exhausted the 4-deferral budget after two blocks and the
    exhausted gate hard-forced </think> while the emitted tail was still a
    valid-but-incomplete F0 90 90 — '�</think>' in reasoning_content.
    Post-fix, only EMITTED positions consume the budget: the char completes
    (F0 90 90 90) before the forced close."""
    clone = _make_clone()
    try:
        target = {START: F0_TOK, F0_TOK: CONT_TOK, CONT_TOK: CONT_TOK}
        draft = {
            START: F0_TOK,        # agrees on the lead byte...
            F0_TOK: WRONG_TOK,    # ...then proposes the wrong continuation
            WRONG_TOK: WRONG_TOK,
            CONT_TOK: WRONG_TOK,
        }
        _set_budget_stash(budget=2, family="gemma4")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_DIVERGE_TABLE)
        emitted = _drive_diverging_clone(
            clone, target, draft, max_tokens=32, first_bonus=START
        )
        assert END in emitted, f"forced close must still land: {emitted}"
        pre = emitted[: emitted.index(END)]
        # The 4-byte char must COMPLETE before the forced close (pre-fix the
        # exhausted gate closed after F0 90 90 — one continuation short).
        assert pre[-4:] == [F0_TOK, CONT_TOK, CONT_TOK, CONT_TOK], emitted
        text = _ByteTokenizer(_DIVERGE_TABLE).decode(pre)
        assert not text.endswith("�"), (
            f"reasoning must not end mid-character: {text!r} / {emitted}"
        )
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_r4f1_hard_force_after_four_real_emitted_deferrals():
    """The bound still holds and is measured in EMITTED positions: a target
    chain of forever-invalid bytes (each block emits exactly one bonus
    token — the drafter always diverges at position 0) gets exactly 4 real
    deferrals before the hard force. Pre-fix the provisional consults of
    the FIRST block alone burned 3 deferrals and the close landed after
    only 2 emitted over-budget tokens (END at index 2, not 5)."""
    clone = _make_clone()
    try:
        target = {START: INVALID_TOK, INVALID_TOK: INVALID_TOK}
        draft = {
            START: WRONG_TOK,
            WRONG_TOK: WRONG_TOK,
            INVALID_TOK: WRONG_TOK,
        }
        _set_budget_stash(budget=2, family="gemma4")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_DIVERGE_TABLE)
        emitted = _drive_diverging_clone(
            clone, target, draft, max_tokens=32, first_bonus=START
        )
        assert END in emitted, f"close must be hard-forced: {emitted}"
        # budget=2 -> the force decision first fires at the 2nd emitted
        # token; the legacy decode fallback defers the '\xff' tail 4 REAL
        # times (one emitted bonus per block), then hard-forces: exactly
        # 1 pre-budget + 4 deferred tokens precede END.
        assert emitted.index(END) == 5, emitted
        assert emitted[:5] == [INVALID_TOK] * 5, emitted
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_r4f1_gate_preview_is_pure_and_commit_is_explicit():
    """Unit contract of the split gate: preview never mutates; ``pending``
    counts a block's earlier provisional grants toward the bound; commit
    consumes exactly one deferral; the combined should_defer keeps the
    plain-path (call==emit) semantics byte-identical."""
    gate = ForceDeferralGate(_ByteLevelTok(_BL_TABLE), max_deferrals=2)
    # Preview grants without mutating (11 = mid-'한' lead byte).
    assert gate.should_defer_preview([1, 11]) is True
    assert gate.should_defer_preview([1, 11]) is True
    assert gate.deferrals == 0
    # pending simulates earlier uncommitted grants of the same block.
    assert gate.should_defer_preview([1, 11], pending=1) is True
    assert gate.should_defer_preview([1, 11], pending=2) is False
    # Explicit commits consume the budget the preview respects.
    gate.commit_deferral()
    assert gate.should_defer_preview([1, 11], pending=1) is False
    gate.commit_deferral()
    assert gate.should_defer_preview([1, 11]) is False
    # rearm restores; the combined consult-and-consume path is unchanged.
    gate.rearm()
    assert gate.should_defer([1, 11]) is True
    assert gate.deferrals == 1


# --- R4-F2: non-streaming /chat forwards resolved session settings --------


class _SettingsCaptureEngine:
    """Engine stub capturing complete() kwargs (no ensure_available: the
    availability preflight is skipped, like an in-process engine)."""

    model_id = "stub-model"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(
            enable_thinking=False,
            default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
            default_top_k=0, default_repetition_penalty=1.0,
            thinking_budget=0, default_max_tokens=64,
        )
        self.complete_kwargs: dict | None = None

    def complete(self, messages, **kwargs):
        self.complete_kwargs = dict(kwargs)
        return SimpleNamespace(
            finish_reason="stop", content="hi", thinking=None,
            prompt_tokens=1, completion_tokens=1,
        )

    def update_session_messages(self, *a, **k):
        return True


class _SettingsStubDB:
    """Minimal async DB surface for the non-streaming /chat route; the
    session row is the post-PATCH state (sampling columns set or NULL)."""

    def __init__(self, session_row: dict):
        self._session = dict(session_row)

    async def get_session(self, session_id):
        return dict(self._session)

    async def get_messages(self, session_id, limit=None):
        return [{"id": "m0", "role": "user", "content": "q", "token_count": 1}]

    async def get_session_total_tokens(self, session_id):
        return 0

    async def add_message(self, *a, **k):
        return {"id": "u1"}

    async def add_message_if_parent_exists(self, *a, **k):
        return {"id": "a1"}

    async def update_session_tokens(self, *a, **k):
        return None


def _post_nonstream_chat(monkeypatch, session_row) -> _SettingsCaptureEngine:
    from mlx_soloheaven.api import chat as chat_mod

    eng = _SettingsCaptureEngine()
    monkeypatch.setattr(chat_mod, "db", _SettingsStubDB(session_row))
    monkeypatch.setattr(chat_mod, "engine", eng)
    monkeypatch.setattr(chat_mod, "_engines", {})

    req = chat_mod.SendMessageRequest(content="hi", stream=False)
    out = asyncio.run(chat_mod.chat("s1", req))
    assert out["content"] == "hi"
    return eng


def test_r4f2_nonstream_chat_forwards_resolved_session_settings(monkeypatch):
    """FAILS PRE-FIX (codex round-3 P2, finding 2): PATCH top_k=1 (etc.)
    then POST /chat stream:false — the streaming branch forwarded every
    resolved setting but _sync_chat called eng.complete() with only
    session_id, so the engine saw top_k=None and resolved the config
    default. All seven resolved settings must arrive."""
    eng = _post_nonstream_chat(monkeypatch, {
        "id": "s1", "system_prompt": "", "context_window_limit": 100000,
        "temperature": 0.2, "top_p": 0.5, "min_p": 0.05, "top_k": 1,
        "repetition_penalty": 1.7, "thinking_budget": 123, "max_tokens": 77,
    })
    kw = eng.complete_kwargs
    assert kw is not None, "complete() was never reached"
    assert kw["session_id"] == "s1"
    assert kw["temperature"] == pytest.approx(0.2)
    assert kw["top_p"] == pytest.approx(0.5)
    assert kw["min_p"] == pytest.approx(0.05)
    assert kw["top_k"] == 1
    assert kw["repetition_penalty"] == pytest.approx(1.7)
    assert kw["thinking_budget"] == 123
    assert kw["max_tokens"] == 77
    # run_long's reservation is consumed by the executor, never forwarded.
    assert "reservation" not in kw


def test_r4f2_nonstream_chat_unset_settings_resolve_engine_defaults(
    monkeypatch,
):
    """Uncustomized sessions (NULL sampling columns — the F7/round-2 NULL
    coalesce) still resolve the engine config defaults, now visible in the
    complete() kwargs."""
    eng = _post_nonstream_chat(monkeypatch, {
        "id": "s1", "system_prompt": "", "context_window_limit": 100000,
        "temperature": None, "top_p": None, "min_p": None, "top_k": None,
        "repetition_penalty": None, "thinking_budget": None,
        "max_tokens": None,
    })
    kw = eng.complete_kwargs
    assert kw["temperature"] == pytest.approx(0.6)
    assert kw["top_p"] == pytest.approx(1.0)
    assert kw["min_p"] == pytest.approx(0.0)
    assert kw["top_k"] == 0
    assert kw["repetition_penalty"] == pytest.approx(1.0)
    assert kw["thinking_budget"] == 0
    assert kw["max_tokens"] == 64


# ===========================================================================
# Batch-5 ROUND 6 (codex round-5 residual P2 finding)
# ===========================================================================
#
# R6 — a MID-BLOCK provisional think_end did not REARM the preview budget:
#      every provisional verify position previewed the deferral gate with
#      ``self.deferrals + sum(defer_events)`` — accounting for the WHOLE
#      block. When the assumed-accepted draft prefix folded a think_end
#      (close -> reopen inside ONE verify block), sequential semantics grant
#      the REOPENED thinking cycle a fresh 4-deferral allowance (the
#      emitted-fold replay re-arms on think_end — but only AFTER target
#      sampling, too late for the previews). The stale preview count forced
#      </think> immediately at the reopened cycle's first over-budget
#      position even though its byte tail ended mid multi-byte character —
#      '�</think>' again. Fixed with a preview-local deferral EPOCH:
#      ``_pos_defer_base`` (snapshot of the committed count, reset to 0 when
#      the provisional prefix folds a think_end) + ``_pos_defer_pending``
#      (grants since the last provisional think_end, reset likewise);
#      ``should_defer_preview`` gained a ``base`` override. Committed state
#      is untouched: previews stay pure and the post-walk replay recomputes
#      truth, so a REJECTED provisional think_end cannot pollute carry-over.


# Distinct token ids per chain step (the scripted LM/drafter key on the
# PREVIOUS token id, so repeated bytes need unique ids). 60-64 are lone
# 0xFF bytes (defer via the decode-endswith-U+FFFD fallback, never
# complete); 80-83 spell a COMPLETE 4-byte astral char F0 90 90 90.
_EP_X = [60, 61, 62, 63, 64]
_EP_F0, _EP_C1, _EP_C2, _EP_C3 = 80, 81, 82, 83

_EPOCH_TABLE = {
    1: b"p",
    60: b"\xff", 61: b"\xff", 62: b"\xff", 63: b"\xff", 64: b"\xff",
    _EP_F0: b"\xf0",
    _EP_C1: b"\x90", _EP_C2: b"\x90", _EP_C3: b"\x90",
    START: b"<think>",
    END: b"</think>",
}


def _drive_epoch_clone(
    clone, target_trans, draft_trans, *, max_tokens, first_bonus, block_size
):
    """Like _drive_diverging_clone but with a configurable block size (the
    round-6 repro needs close -> reopen -> over-budget INSIDE one block)."""
    model = SimpleNamespace(language_model=_ScriptedLM(target_trans))
    draft = _DivergingDraftModel(draft_trans, block_size=block_size)
    cache = [SimpleNamespace(offset=0)]
    gen = clone(
        model, draft, cache, mx.zeros((1, 1, 4)), {},
        first_bonus=first_bonus, max_tokens=max_tokens,
        sampler=_argmax_sampler, draft_block_size=block_size,
        token_dtype=mx.int32,
    )
    return [int(tok) for tok, _ in gen]


def test_r6_reopened_cycle_gets_fresh_deferral_allowance_in_block():
    """FAILS PRE-FIX (codex round-5 P2): ONE 12-position verify block whose
    accepted chain is X X X X END START F0 90 90 90 END — budget 2, so the
    four X positions each preview-grant a deferral (invalid 0xFF tails),
    the model closes and REOPENS thinking, and the reopened cycle emits a
    4-byte char. Sequential semantics: the close re-arms the gate, so the
    reopened cycle's over-budget positions get a FRESH 4-deferral
    allowance and the char completes before the forced close. Pre-fix the
    preview counted the pre-END usage (pending=sum of the WHOLE block's
    events = 4), hard-forced </think> right after the F0 lead byte, and
    reasoning_content of the reopened block decoded to '�'."""
    clone = _make_clone()
    try:
        # target == draft: every draft agrees until a FORCED close flips
        # the target (chatml: thinking is open from generation start).
        trans = {
            60: 61, 61: 62, 62: 63, 63: 64, 64: END,
            END: START, START: _EP_F0,
            _EP_F0: _EP_C1, _EP_C1: _EP_C2, _EP_C2: _EP_C3, _EP_C3: END,
        }
        _set_budget_stash(budget=2, family="chatml")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_EPOCH_TABLE)
        emitted = _drive_epoch_clone(
            clone, trans, trans,
            max_tokens=12, first_bonus=60, block_size=12,
        )
        first_end = emitted.index(END)
        assert first_end == 4, emitted
        assert emitted[first_end + 1] == START, emitted
        second_end = emitted.index(END, first_end + 1)
        seg2 = emitted[first_end + 2 : second_end]
        # The reopened cycle must complete the 4-byte char before the
        # forced close (pre-fix: seg2 == [F0] — closed on the lead byte).
        assert seg2 == [_EP_F0, _EP_C1, _EP_C2, _EP_C3], emitted
        text2 = _ByteTokenizer(_EPOCH_TABLE).decode(seg2)
        assert not text2.endswith("�"), (
            f"reopened reasoning must not end mid-character: "
            f"{text2!r} / {emitted}"
        )
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_r6_rejected_provisional_close_keeps_committed_carryover():
    """Reverse case (round-6 guard, passes pre- and post-fix): the DRAFTER
    proposes think_end at position 0 of every block but the target always
    rejects it — the provisional epoch reset fires mid-block yet must
    never leak into committed state. Each block still commits exactly one
    real deferral for its emitted bonus (invalid 0xFF tail), and the next
    block previews with the TRUE carried-over count: the hard force lands
    exactly after 1 pre-budget + 4 committed deferrals (END at index 5,
    identical to the round-4 hard-force baseline). A leak in either
    direction moves END: resetting committed state delays it; carrying a
    stale epoch across blocks (base stuck at 0) grants forever and END
    never lands at 5."""
    clone = _make_clone()
    try:
        target = {START: INVALID_TOK, INVALID_TOK: INVALID_TOK}
        draft = {
            START: END,        # proposes a close the target never accepts
            END: START,        # ...then a reopen, alternating
            INVALID_TOK: END,
        }
        _set_budget_stash(budget=2, family="gemma4")
        mlx_engine_module._MTP_THINK_TOKENIZER = _ByteTokenizer(_DIVERGE_TABLE)
        emitted = _drive_epoch_clone(
            clone, target, draft,
            max_tokens=32, first_bonus=START, block_size=8,
        )
        assert END in emitted, f"close must be hard-forced: {emitted}"
        assert emitted.index(END) == 5, emitted
        assert emitted[:5] == [INVALID_TOK] * 5, emitted
    finally:
        _mtp_clear_stash()
        mlx_engine_module._HOT_PATH_FAST = False


def test_r6_gate_preview_base_overrides_committed_count():
    """Unit contract of the ``base`` override: it replaces the committed
    count in the bound (the verifier's preview-local epoch), respects
    ``pending``, never mutates, and ``base=None`` keeps the round-4
    live-count semantics byte-identical."""
    gate = ForceDeferralGate(_ByteLevelTok(_BL_TABLE), max_deferrals=2)
    gate.commit_deferral()
    gate.commit_deferral()
    # Committed count exhausted: the default preview refuses...
    assert gate.should_defer_preview([1, 11]) is False
    # ...but a post-provisional-close epoch (base=0) grants afresh.
    assert gate.should_defer_preview([1, 11], base=0) is True
    # The override still respects pending and the shared bound.
    assert gate.should_defer_preview([1, 11], pending=1, base=0) is True
    assert gate.should_defer_preview([1, 11], pending=2, base=0) is False
    assert gate.should_defer_preview([1, 11], pending=1, base=1) is False
    # base=None keeps the live committed count (round-4 path unchanged).
    assert gate.should_defer_preview([1, 11], base=None) is False
    # Previews with a base never mutate committed state.
    assert gate.deferrals == 2
