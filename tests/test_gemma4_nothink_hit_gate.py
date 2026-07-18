"""Finding B — gemma4 thinking=False forced-thought suffix gap.

THE BUG (codex triage, verified token-exact against both local gemma-4
tokenizers): the installed gemma4 templates append
``<|channel>thought\\n<channel|>`` AFTER the generation marker when thinking
is disabled (chat_template.jinja: the ``add_generation_prompt`` block), and
their ``strip_thinking`` macro REMOVES ``<|channel>…<channel|>`` spans from
PAST assistant turns on full retokenization — the forced block exists only
at the CURRENT generation prompt. A cached thinking=False session therefore
physically retains the forced block in the KV at every PAST generation
point, while full retokenization renders history WITHOUT it and the block
at the CURRENT prompt instead. Two consequences:

- cached ids can NEVER equal full retokenization (no suffix can delete
  tokens mid-sequence), and
- ``_suffix_tokens_gemma4`` ignored its thinking flag, so every HIT turn's
  generation prompt LACKED the forced block — silently un-suppressing
  thinking (thinking=False requests AND structured-output requests, which
  reach this path via the U9 suppression, could open a thought channel).

DESIGN DECISION — option (ii) of the triage, fail-closed honest MISS:
every gemma4 thinking=False turn rebuilds by full retokenization (correct
by construction: history without the block, current prompt with it).
Options (i)/(iii) — a "cache-canonical" invariant blessing the stale
historical blocks — were rejected: apply_chat_template can never be made to
agree with the cache (strip_thinking + trim run over stored messages too),
so the redefinition would have to be threaded through C1/reconcile/matching
everywhere for a niche path. Cost is bounded: prefill-only TTFT for
non-thinking gemma4 sessions, softened by base-cache seeding.

PRE-FIX FAILURES (verified by reverting the engine gate):
- test_thinking_false_hit_demoted_to_honest_miss (the HIT spliced the
  diverged sequence — the prefill received the suffix, not the rebuild),
- test_gate_logs_once_per_session (no gate, no log, and both drives HIT),
- test_thinking_false_second_turn_still_misses (second turn kept HITting).
- The real-tokenizer differential tests document the divergence shape and
  the codex claims (history stripped, block only at the current prompt);
  they pass pre- and post-fix.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_qwen_mtp import (  # noqa: E402
    MockKV,
    _generate_engine,
    _restamp_contract,
    _scripted_lm_stream,
)

EOT = 106  # <turn|> — gemma4 end-of-turn / chat EOS

LMSTUDIO_ROOT = Path(os.path.expanduser("~/.lmstudio/models"))
GEMMA4_TOKENIZER_DIRS = [
    ("gemma-4-31b", "lmstudio-community/gemma-4-31B-it-MLX-8bit"),
    ("gemma-4-26b-a4b", "lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit"),
]

ENGINE_LOGGER = "mlx_soloheaven.engine.mlx_engine"


def _tokenizer_present(rel: str) -> bool:
    p = LMSTUDIO_ROOT / rel
    return p.is_dir() and (
        (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
    )


# ---------------------------------------------------------------------------
# Wiring — the gate demotes thinking=False gemma4 HITs to an honest MISS
# ---------------------------------------------------------------------------


def _gemma4_engine(stored, suffix):
    """HIT-capable mlx-lm-path harness re-labeled gemma4. The harness stamps
    the session contract as thinking=False (its default), which is exactly
    the shape the gate must demote."""
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)
    eng.model_family = "gemma4"
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<turn|>": EOT},
        encode=lambda text, add_special_tokens=False: [],
    )
    return eng, cs, messages


def _drive(eng, messages, *, thinking, max_tokens=8):
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=thinking,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )


def test_thinking_false_hit_demoted_to_honest_miss(monkeypatch):
    """A live, contract-matching gemma4 session with thinking=False must NOT
    reuse the cache: the prefill receives the FULL retokenization, never the
    suffix splice. PRE-FIX: fails — the HIT spliced [EOT, 55, 66]."""
    stored = [1, 2, 3, 4, EOT]
    eng, cs, messages = _gemma4_engine(stored, suffix=[EOT, 55, 66])
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive(eng, messages, thinking=False)

    assert prompts_seen[0] == [11, 12, 13]  # honest MISS rebuild
    new_cs = eng._sessions["s"].cache_state
    assert new_cs is not cs
    # No silent drift: the rebuilt ids ARE the full retokenization plus the
    # generated tokens.
    assert new_cs.token_ids == [11, 12, 13, 101, 102]


def test_thinking_false_second_turn_still_misses(monkeypatch):
    """The rebuild does not re-arm reuse: the NEXT thinking=False turn on the
    rebuilt (live, matching) session also takes an honest MISS — correctness
    by cold-fill every turn. PRE-FIX: fails (second turn HIT)."""
    stored = [1, 2, 3, 4, EOT]
    eng, cs, messages = _gemma4_engine(stored, suffix=[EOT, 55, 66])
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "a"), (102, "b")], [(103, "c")]],
    )

    _drive(eng, messages, thinking=False)
    messages2 = list(eng._sessions["s"].messages) + [
        {"role": "user", "content": "u3"},
    ]
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [21, 22, 23, 24]
    _drive(eng, messages2, thinking=False)

    assert prompts_seen[1] == [21, 22, 23, 24]  # MISS again, no splice


def test_thinking_true_hit_still_reuses(monkeypatch):
    """Scope guard: thinking=True gemma4 sessions keep the HIT path (with
    the U18 dedupe). The gate is thinking=False only."""
    stored = [1, 2, 3, 4, EOT]
    eng, cs, messages = _gemma4_engine(stored, suffix=[EOT, 55, 66])
    _restamp_contract(eng, thinking=True)
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive(eng, messages, thinking=True)

    assert prompts_seen[0] == [55, 66]  # HIT + U18 dedupe of the closer
    assert cs.token_ids == stored + [55, 66] + [101, 102]


def test_nonthinking_other_families_unaffected(monkeypatch):
    """chatml thinking=False sessions still HIT (the gate is gemma4-only)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(
        cache, stored, mtp=False, suffix=[52, 99],
    )
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive(eng, messages, thinking=False)

    assert prompts_seen[0] == [52, 99]  # chatml HIT untouched


def test_gate_logs_once_per_session(monkeypatch, caplog):
    """The demotion is logged ONCE per session id (not per turn)."""
    stored = [1, 2, 3, 4, EOT]
    eng, cs, messages = _gemma4_engine(stored, suffix=[EOT, 55, 66])
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "a"), (102, "b")], [(103, "c")]],
    )

    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        _drive(eng, messages, thinking=False)
        messages2 = list(eng._sessions["s"].messages) + [
            {"role": "user", "content": "u3"},
        ]
        _drive(eng, messages2, thinking=False)

    gate_logs = [
        r for r in caplog.records
        if "thinking disabled cannot reuse" in r.getMessage()
    ]
    assert len(gate_logs) == 1
    assert eng._gemma4_nothink_miss_logged == {"s"}
    # Both turns took the MISS path regardless of the single log line.
    assert prompts_seen[0] == [11, 12, 13]
    assert prompts_seen[1] == [11, 12, 13]


# ---------------------------------------------------------------------------
# Token-exact differentials — real gemma4 tokenizers (skip when absent).
# These document WHY reuse is impossible for thinking=False (the codex
# differential) and prove thinking=True splices stay exact. They pass pre-
# and post-fix — the behavioral gate is covered by the wiring tests above.
# ---------------------------------------------------------------------------


def _load_real(rel: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(LMSTUDIO_ROOT / rel), trust_remote_code=True
    )


def _ids(result) -> list[int]:
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _shell(tok):
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "gemma4"
    eng.tokenizer = tok
    return eng


Q1 = "What is 2 + 2?"
A1 = "The answer is 4."
Q2 = "And 3 + 3?"

FORCED_BLOCK = "<|channel>thought\n<channel|>"


def _contains_at(haystack: list[int], needle: list[int], start: int) -> bool:
    return haystack[start:start + len(needle)] == needle


def _count_sub(haystack: list[int], needle: list[int]) -> int:
    return sum(
        1 for i in range(len(haystack) - len(needle) + 1)
        if _contains_at(haystack, needle, i)
    )


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_nothink_cached_splice_diverges_from_full_retok(label, rel):
    """The codex differential: with enable_thinking=False the turn-1 prompt
    ends with the forced block; the cached sequence retains it at the PAST
    generation point while full retokenization strips it from history and
    renders it at the CURRENT generation prompt — so the splice can never
    equal full retokenization, and the naive suffix also fails to render the
    block for the new turn. This is exactly why the gate rebuilds."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell(tok)
    eot = eng._gemma4_turn_end_id()
    assert eot >= 0
    block = tok.encode(FORCED_BLOCK, add_special_tokens=False)
    assert block  # the template's forced empty-thought block tokenizes

    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
    )
    # The forced block sits at the END of the thinking=False generation
    # prompt (chat_template.jinja renders it after '<|turn>model\n').
    assert prompt1[-len(block):] == block

    # mlx-lm recording contract: reply + recorded EOS (lookahead-forwarded).
    stored = prompt1 + tok.encode(A1, add_special_tokens=False) + [eot]

    suffix = eng._suffix_tokens_gemma4(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    suffix = eng._dedupe_gemma4_turn_closer(suffix, stored)
    spliced = stored + suffix

    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
    )

    # The divergence is structural, not incidental:
    assert spliced != full
    # full retokenization strips the block from HISTORY — it appears exactly
    # once, at the CURRENT generation prompt (codex: "removes it from
    # history").
    assert _count_sub(full, block) == 1
    assert full[-len(block):] == block
    # The cached side retains the PAST block but the naive suffix omitted
    # the CURRENT one — thinking suppression silently broken on HIT turns.
    assert _count_sub(spliced, block) == 1
    assert spliced[-len(block):] != block


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_thinking_true_splice_remains_token_exact(label, rel):
    """Scope guard on the real tokenizers: the thinking=True HIT splice
    (U18 dedupe) still equals full retokenization — the gate must not widen
    beyond thinking=False."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    eng = _shell(tok)
    eot = eng._gemma4_turn_end_id()

    prompt1 = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": Q1}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    stored = prompt1 + tok.encode(A1, add_special_tokens=False) + [eot]
    suffix = eng._dedupe_gemma4_turn_closer(
        eng._suffix_tokens_gemma4([{"role": "user", "content": Q2}], thinking=True),
        stored,
    )
    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
    )
    assert stored + suffix == full


@pytest.mark.parametrize(
    "label,rel", GEMMA4_TOKENIZER_DIRS, ids=[c[0] for c in GEMMA4_TOKENIZER_DIRS]
)
def test_nothink_full_retok_is_correct_by_construction(label, rel):
    """What the honest MISS produces: full retokenization renders history
    WITHOUT stale forced blocks and the current prompt WITH exactly one —
    i.e. the rebuild is the correct rendering the gate buys."""
    if not _tokenizer_present(rel):
        pytest.skip(f"tokenizer dir not present: {rel}")
    tok = _load_real(rel)
    block = tok.encode(FORCED_BLOCK, add_special_tokens=False)

    full = _ids(
        tok.apply_chat_template(
            [
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": Q2},
                {"role": "assistant", "content": "That makes 6."},
                {"role": "user", "content": "Now multiply them."},
            ],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
    )
    # Two past assistant turns, zero retained blocks, one current block.
    assert _count_sub(full, block) == 1
    assert full[-len(block):] == block


# ---------------------------------------------------------------------------
# U18 follow-up round 2, FINDING 1 — session_cache_preflight must mirror the
# generation gate. The generation demotes every gemma4 thinking=False reuse
# to an honest MISS, but the advisory preflight (consumed by the web
# stream's "start" event, chat.py) computed cache_hit = cache_live WITHOUT
# the gate, and its DISK branch never parsed the persisted "thinking"
# metadata (stored_thinking stayed the hard-coded True).
# PRE-FIX FAILURES (verified against the ungated preflight):
# - test_preflight_memory_gemma4_nothink_reports_miss (cache_hit was True),
# - test_preflight_disk_gemma4_nothink_reports_miss (cache_hit was True),
# - test_preflight_disk_parses_stored_thinking[zero] (thinking_active True).
# ---------------------------------------------------------------------------


STORED_MSGS = [
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
]


def _preflight_shell(tmp_path, *, family, enable_thinking):
    """Minimal session_cache_preflight shell (conventions from
    tests/test_session_paths_locking.py). The advisory match is stubbed to
    keep these tests about the gate, not the matcher."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = family
    eng._lock = threading.Lock()
    eng.cfg = SimpleNamespace(
        cache_dir=str(tmp_path), enable_thinking=enable_thinking,
    )
    eng._sessions = {}
    eng._has_disk_cache = lambda sid: False
    eng._messages_match = lambda stored, incoming, **kw: True
    return eng


def _preflight_fp(eng) -> str:
    """The prompt-contract fingerprint session_cache_preflight computes for a
    web-chat request on this engine (tools=None, thinking=cfg.enable_thinking,
    family suffix revision) — a session must carry it to preflight HIT under
    the U18-round-3 fingerprint gate."""
    return eng._prompt_fingerprint(
        eng._canonical_tools(None),
        bool(eng.cfg.enable_thinking),
        template_rev=eng._suffix_template_rev(),
    )


def _live_session(*, thinking, fingerprint=None):
    return SimpleNamespace(
        messages=list(STORED_MSGS),
        total_cache_tokens=3,
        cache_state=SimpleNamespace(cache=object(), token_ids=[1, 2, 3]),
        thinking=thinking,
        prompt_fingerprint=fingerprint,
    )


def _write_disk_header(tmp_path, session_id, metadata):
    """Hand-crafted safetensors file: 8-byte LE header length + JSON header
    carrying ``__metadata__`` — the exact layout mx.save_safetensors
    writes. Plain IO, no MLX."""
    header = json.dumps({"__metadata__": metadata}).encode()
    path = tmp_path / f"session_{session_id}.safetensors"
    with open(path, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
    return path


def test_preflight_memory_gemma4_nothink_reports_miss(tmp_path):
    """A live, matching in-memory gemma4 session with thinking disabled:
    the preflight must report the honest MISS/rebuild the generation gate
    will actually take — never advertise a HIT the generation then refuses.
    PRE-FIX: cache_hit was True."""
    eng = _preflight_shell(tmp_path, family="gemma4", enable_thinking=False)
    eng._sessions["s1"] = _live_session(thinking=False)

    out = eng.session_cache_preflight(
        "s1", STORED_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"
    assert "thinking disabled" in out["cache_info"]["detail"]


def test_preflight_memory_gemma4_thinking_true_still_hits(tmp_path):
    """Scope guard: the preflight gate is thinking=False only."""
    eng = _preflight_shell(tmp_path, family="gemma4", enable_thinking=True)
    eng._sessions["s1"] = _live_session(
        thinking=True, fingerprint=_preflight_fp(eng),
    )

    out = eng.session_cache_preflight(
        "s1", STORED_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"


def test_preflight_memory_nothink_other_family_still_hits(tmp_path):
    """Scope guard: the preflight gate is gemma4 only."""
    eng = _preflight_shell(tmp_path, family="chatml", enable_thinking=False)
    eng._sessions["s1"] = _live_session(
        thinking=False, fingerprint=_preflight_fp(eng),
    )

    out = eng.session_cache_preflight(
        "s1", STORED_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"


@pytest.mark.parametrize(
    "meta_thinking,expected",
    [("0", False), ("1", True), (None, True)],
    ids=["zero", "one", "missing"],
)
def test_preflight_disk_parses_stored_thinking(tmp_path, meta_thinking, expected):
    """The DISK preflight parses the persisted "thinking" metadata (the key
    _save_session_to_disk writes, parsed exactly like
    _load_session_from_disk: missing defaults to True) instead of
    hard-coding True — observed via the thinking_active kwarg the advisory
    match receives. PRE-FIX: 'zero' fails (the match saw True)."""
    eng = _preflight_shell(tmp_path, family="chatml", enable_thinking=True)
    meta = {
        "messages": json.dumps(STORED_MSGS),
        "total_cache_tokens": "3",
        # U18 round 3: carry the matching prompt-contract fingerprint so this
        # test stays focused on the thinking_active parsing (the new gate is
        # exercised separately) — without it the fingerprint gate would demote
        # every case to a MISS.
        "prompt_fingerprint": _preflight_fp(eng),
    }
    if meta_thinking is not None:
        meta["thinking"] = meta_thinking
    _write_disk_header(tmp_path, "s2", meta)
    eng._has_disk_cache = lambda sid: sid == "s2"
    seen: dict = {}

    def _match(stored, incoming, **kw):
        seen.update(kw)
        return True

    eng._messages_match = _match

    out = eng.session_cache_preflight(
        "s2", STORED_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert seen["thinking_active"] is expected
    assert out["cache_hit"] is True  # chatml: the gemma4 gate must not fire
    assert out["cache_info"]["source"] == "disk"


def test_preflight_disk_gemma4_nothink_reports_miss(tmp_path):
    """Same fail-closed rule on the DISK branch. PRE-FIX: cache_hit was
    True (no gate ran and stored_thinking stayed the hard-coded True)."""
    eng = _preflight_shell(tmp_path, family="gemma4", enable_thinking=False)
    _write_disk_header(tmp_path, "s2", {
        "messages": json.dumps(STORED_MSGS),
        "total_cache_tokens": "3",
        "thinking": "0",
    })
    eng._has_disk_cache = lambda sid: sid == "s2"

    out = eng.session_cache_preflight(
        "s2", STORED_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"
    assert "thinking disabled" in out["cache_info"]["detail"]


# ---------------------------------------------------------------------------
# U18 follow-up round 2, FINDING 2 — _gemma4_nothink_miss_logged is pruned at
# the SAME three cleanup sites as _anon_minted_ids (active-LRU eviction,
# delete_session, clear_caches); sessionless gemma traffic used to grow the
# log-once set without bound.
# PRE-FIX FAILURES: all three tests below (the set kept the removed ids).
# ---------------------------------------------------------------------------


def _admin_shell(tmp_path):
    """Shell for the mutating admin ops (delete_session / clear_caches) —
    conventions from tests/test_session_paths_locking.py."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng._lock = threading.Lock()
    eng._dirty_lock = threading.Lock()
    eng._dirty_sessions = set()
    eng._sessions = {"s1": SimpleNamespace(messages=[], total_cache_tokens=0)}
    eng.cfg = SimpleNamespace(cache_dir=str(tmp_path))
    eng._base_caches = {}
    eng.cache_manager = SimpleNamespace(memory_caches={}, disk_index={})
    return eng


def test_delete_session_prunes_nothink_log_set(tmp_path):
    """Set hygiene: deleting a session drops its log-once id (other ids
    untouched). PRE-FIX: the set persisted 's1'."""
    eng = _admin_shell(tmp_path)
    eng._gemma4_nothink_miss_logged = {"s1", "s2"}

    assert eng.delete_session("s1") is True

    assert eng._gemma4_nothink_miss_logged == {"s2"}


def test_clear_caches_prunes_nothink_log_set(tmp_path):
    """Cache reset: no sessions remain, so no log-once ids remain.
    PRE-FIX: the set survived the reset intact."""
    eng = _admin_shell(tmp_path)
    eng._gemma4_nothink_miss_logged = {"s1", "anon-aaaaaaaa"}

    eng.clear_caches()

    assert eng._gemma4_nothink_miss_logged == set()


def test_eviction_prunes_nothink_log_set(monkeypatch):
    """Mirror of the _anon_minted_ids LRU-eviction hygiene test
    (tests/test_anon_prefix_session.py): evicted sessions' ids leave the
    log-once set; the resident MRU id stays. PRE-FIX: all three ids
    persisted."""
    from mlx_soloheaven.cache.manager import CacheManager
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState

    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.memory_budget_gb = 1.5
    eng.cfg = cfg
    eng._lock = threading.Lock()
    eng._dirty_sessions = set()
    eng._dirty_lock = threading.Lock()
    eng._busy_sessions = set()
    eng._busy_lock = threading.Lock()
    eng._anon_minted_ids = set()
    eng.cache_manager = CacheManager(
        memory_budget_gb=1.0, disk_budget_gb=100.0,
        cache_dir="/tmp/sh-test-cache",
    )
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)

    _gb = 1_000_000_000

    def _big_session(last_used: float) -> SessionState:
        layer = SimpleNamespace(state=(SimpleNamespace(nbytes=_gb),))
        return SessionState(
            cache_state=SimpleNamespace(cache=[layer], token_ids=[1]),
            messages=[{"role": "user", "content": "hi"}],
            total_cache_tokens=1,
            last_used=last_used,
        )

    eng._sessions = {
        "g-old": _big_session(100.0),  # LRU — evicted
        "g-mid": _big_session(200.0),  # evicted (still over budget)
        "g-new": _big_session(300.0),  # MRU — always kept resident
    }
    eng._gemma4_nothink_miss_logged = set(eng._sessions)

    eng._evict_active_sessions_if_needed()

    assert set(eng._sessions) == {"g-new"}
    # Set hygiene: only the resident session's id remains.
    assert eng._gemma4_nothink_miss_logged == {"g-new"}
