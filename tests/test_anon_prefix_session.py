"""Prefix-matching anon session selection tests.

Problem being fixed: OpenAI-compat requests map session_id to the request's
``user`` field, which most agents (OpenCode) never send. Every session-less
request then collapsed onto the SINGLE shared "anon" key, so two interleaved
conversations (different system prompts, or a client updating its system
prompt mid-flow) thrashed the one cache slot — each request from the *other*
conversation was a MISS → full cold-fill.

The fix resolves session-less requests in generate_stream (inside the engine
lock, BEFORE busy-marking) via ``_resolve_anon_session_id_locked``:
  - longest stored-messages prefix match (reusing _messages_match) wins,
  - tie-break by most-recently-used,
  - exact match (stored == incoming) selects too (retry semantics downstream),
  - no match → NEW unique ``anon-<8 hex>`` session,
  - explicit session_ids never prefix-scan (exact-key behavior unchanged),
  - isolation is PROVENANCE-based: only sessions whose id the engine itself
    minted (tracked in ``_anon_minted_ids``) are scan candidates. Name-based
    filtering is not enough — ``ChatCompletionRequest.user`` is unrestricted
    and passed verbatim as session_id, so an explicit session keyed
    "anon-aaaaaaaa" by a client must never be matched/mutated,
  - the RESOLVED id is what gets busy-marked / eviction-protected.

These tests build a BARE engine shell (no model / MLX weights), exactly like
tests/test_active_session_eviction.py, and stub _generate_locked where the
full generate path is exercised.
"""

from __future__ import annotations

import re
import threading
import time
from types import SimpleNamespace

import pytest

from mlx_soloheaven.cache.manager import CacheManager
from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState

_ANON_RE = re.compile(r"^anon-[0-9a-f]{8}$")


def _make_engine() -> MLXEngine:
    """Bare MLXEngine shell with the fields resolution + generate_stream touch."""
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.memory_budget_gb = 0  # eviction disabled unless a test enables it
    eng.cfg = cfg
    eng._lock = threading.Lock()
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._dirty_lock = threading.Lock()
    eng._busy_sessions = set()
    eng._busy_lock = threading.Lock()
    eng._anon_minted_ids = set()
    eng.cache_manager = CacheManager(
        memory_budget_gb=1.0, disk_budget_gb=100.0, cache_dir="/tmp/sh-test-cache"
    )
    return eng


def _add_anon(eng: MLXEngine, sid: str, session: SessionState) -> None:
    """Insert a session AS IF the engine minted it for a session-less request:
    stored in _sessions AND registered in _anon_minted_ids (provenance)."""
    eng._sessions[sid] = session
    eng._anon_minted_ids.add(sid)


def _session(messages: list[dict], last_used: float = 100.0) -> SessionState:
    cache_state = SimpleNamespace(cache=["x"], token_ids=[1, 2, 3])
    return SessionState(
        cache_state=cache_state,
        messages=messages,
        total_cache_tokens=3,
        last_used=last_used,
    )


SA = {"role": "system", "content": "You are agent A. Be terse."}
SB = {"role": "system", "content": "You are agent B. Be verbose and friendly."}


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str) -> dict:
    return {"role": "assistant", "content": text}


# --- Resolver unit tests ----------------------------------------------------


def test_no_candidates_creates_unique_anon_session():
    """No sessions at all → a fresh unique anon-<8 hex> id, not 'anon'."""
    eng = _make_engine()
    sid1 = eng._resolve_anon_session_id_locked([SA, _u("hi")])
    sid2 = eng._resolve_anon_session_id_locked([SB, _u("hi")])
    assert _ANON_RE.match(sid1), sid1
    assert _ANON_RE.match(sid2), sid2
    assert sid1 != sid2  # each unmatched conversation gets its OWN slot


def test_prefix_match_selects_existing_session():
    """Incoming that extends a stored conversation resolves to that session."""
    eng = _make_engine()
    _add_anon(eng, "anon-aaaaaaaa", _session([SA, _u("q1"), _a("r1")]))
    sid = eng._resolve_anon_session_id_locked([SA, _u("q1"), _a("r1"), _u("q2")])
    assert sid == "anon-aaaaaaaa"


def test_different_system_prompt_gets_new_session():
    """A conversation with a different system prompt must NOT steal the slot."""
    eng = _make_engine()
    _add_anon(eng, "anon-aaaaaaaa", _session([SA, _u("q1"), _a("r1")]))
    sid = eng._resolve_anon_session_id_locked([SB, _u("q1")])
    assert sid != "anon-aaaaaaaa"
    assert _ANON_RE.match(sid)


def test_longest_prefix_wins():
    """When two sessions both prefix-match, the longer stored prefix wins."""
    eng = _make_engine()
    convo = [SA, _u("q1"), _a("r1"), _u("q2"), _a("r2")]
    _add_anon(eng, "anon-short000", _session(convo[:2], last_used=999.0))  # MRU but short
    _add_anon(eng, "anon-long0000", _session(convo[:5], last_used=1.0))
    sid = eng._resolve_anon_session_id_locked(convo + [_u("q3")])
    assert sid == "anon-long0000"


def test_tie_break_most_recently_used():
    """Equal prefix length → most-recently-used session wins."""
    eng = _make_engine()
    stored = [SA, _u("q1"), _a("r1")]
    _add_anon(eng, "anon-older000", _session(list(stored), last_used=10.0))
    _add_anon(eng, "anon-newer000", _session(list(stored), last_used=20.0))
    sid = eng._resolve_anon_session_id_locked(stored + [_u("q2")])
    assert sid == "anon-newer000"


def test_exact_match_selects_session_for_retry():
    """stored == incoming (client re-sent the same messages) still resolves to
    the session — _generate_locked's existing RETRY path then applies."""
    eng = _make_engine()
    stored = [SA, _u("q1")]
    _add_anon(eng, "anon-bbbbbbbb", _session(list(stored)))
    sid = eng._resolve_anon_session_id_locked(list(stored))
    assert sid == "anon-bbbbbbbb"


def test_legacy_anon_session_participates_when_engine_minted():
    """The legacy 'anon' slot is a candidate ONLY via provenance: when the
    engine's own _generate_locked fallback minted it (registered in
    _anon_minted_ids). No migration needed for such sessions."""
    eng = _make_engine()
    _add_anon(eng, "anon", _session([SA, _u("q1"), _a("r1")]))
    sid = eng._resolve_anon_session_id_locked([SA, _u("q1"), _a("r1"), _u("q2")])
    assert sid == "anon"


def test_client_keyed_anon_session_not_matched():
    """An 'anon' session NOT minted by the engine (e.g. a client explicitly
    sent user='anon') is not a candidate — provenance, not name."""
    eng = _make_engine()
    eng._sessions["anon"] = _session([SA, _u("q1"), _a("r1")])  # no provenance
    sid = eng._resolve_anon_session_id_locked([SA, _u("q1"), _a("r1"), _u("q2")])
    assert sid != "anon"
    assert _ANON_RE.match(sid)


def test_empty_stored_messages_never_match():
    """A session with empty messages would prefix-match ANYTHING — excluded."""
    eng = _make_engine()
    _add_anon(eng, "anon-empty000", _session([]))
    sid = eng._resolve_anon_session_id_locked([SA, _u("hi")])
    assert sid != "anon-empty000"
    assert _ANON_RE.match(sid)


def test_anon_request_never_matches_explicit_session():
    """Namespace isolation regression: an explicit (`user`-keyed / web-UI)
    session whose messages PERFECTLY prefix-match an anonymous request must
    NOT be selected — otherwise the anon request would mutate that explicit
    session's cache/messages. A new anon-<hex> id is minted instead."""
    eng = _make_engine()
    stored = [SA, _u("q1"), _a("r1")]
    eng._sessions["web-ui-1"] = _session(list(stored))
    sid = eng._resolve_anon_session_id_locked(list(stored) + [_u("q2")])
    assert sid != "web-ui-1"
    assert _ANON_RE.match(sid)


def test_anon_named_explicit_session_not_matched():
    """Provenance regression (codex re-review): a client can send ANY value as
    OpenAI ``user`` — including "anon-aaaaaaaa". Such an EXPLICIT session sits
    in _sessions but was never engine-minted (absent from _anon_minted_ids),
    so even a perfect prefix match must NOT select it; a fresh id is minted."""
    eng = _make_engine()
    stored = [SA, _u("q1"), _a("r1")]
    eng._sessions["anon-aaaaaaaa"] = _session(list(stored))  # explicit, no provenance
    sid = eng._resolve_anon_session_id_locked(list(stored) + [_u("q2")])
    assert sid != "anon-aaaaaaaa"
    assert _ANON_RE.match(sid)
    # The explicit session was not touched.
    assert eng._sessions["anon-aaaaaaaa"].messages == stored


def test_minted_id_recorded_and_reused():
    """Minting registers provenance immediately; the same conversation's next
    anonymous request resolves back onto the minted session."""
    eng = _make_engine()
    sid = eng._resolve_anon_session_id_locked([SA, _u("q1")])
    assert sid in eng._anon_minted_ids  # recorded AT mint time
    # Engine stores the session (as _generate_locked would), conversation grows.
    eng._sessions[sid] = _session([SA, _u("q1"), _a("r1")])
    sid2 = eng._resolve_anon_session_id_locked([SA, _u("q1"), _a("r1"), _u("q2")])
    assert sid2 == sid


# --- generate_stream integration (stubbed _generate_locked) -----------------


def _wire_generate_stub(eng, record: list[dict]):
    """Stub _generate_locked: records the resolved session_id + busy state and
    mimics the real session-save side effect (store messages + assistant)."""

    def _stub(messages, **kwargs):
        sid = kwargs["session_id"]
        with eng._busy_lock:
            busy_now = set(eng._busy_sessions)
        record.append({"session_id": sid, "busy_during_gen": busy_now})
        # Mimic _generate_locked's save: stored messages include the reply.
        reply = _a(f"reply-{len(record)}")
        eng._sessions[sid] = _session(list(messages) + [reply], last_used=time.time())
        yield SimpleNamespace(text="ok", finish_reason="stop")

    eng._generate_locked = _stub


def _drive(eng, messages, session_id=None):
    return list(eng.generate_stream(messages, session_id=session_id))


def test_interleaved_conversations_do_not_thrash():
    """THE core user scenario: A1, B1, A2, B2 — A and B each keep their own
    anon session; A's next turn still matches A despite B in between."""
    eng = _make_engine()
    record: list[dict] = []
    _wire_generate_stub(eng, record)

    a1 = [SA, _u("alpha question 1")]
    b1 = [SB, _u("beta question 1")]
    _drive(eng, a1)  # A1 → new session
    _drive(eng, b1)  # B1 → different new session
    sid_a, sid_b = record[0]["session_id"], record[1]["session_id"]
    assert _ANON_RE.match(sid_a) and _ANON_RE.match(sid_b)
    assert sid_a != sid_b

    # A2 extends A1's stored conversation (stored reply + new user msg).
    a_stored = eng._sessions[sid_a].messages
    _drive(eng, list(a_stored) + [_u("alpha question 2")])
    assert record[2]["session_id"] == sid_a  # HIT path — no thrash from B

    # B2 extends B1 — still its own session.
    b_stored = eng._sessions[sid_b].messages
    _drive(eng, list(b_stored) + [_u("beta question 2")])
    assert record[3]["session_id"] == sid_b

    # Exactly two anon sessions exist; neither overwrote the other.
    assert {record[2]["session_id"], record[3]["session_id"]} == {sid_a, sid_b}


def test_explicit_session_id_skips_prefix_scan():
    """Requests WITH a session_id keep exact-key behavior: no prefix scanning,
    even if another session's messages would prefix-match."""
    eng = _make_engine()
    record: list[dict] = []
    _wire_generate_stub(eng, record)
    # A perfectly matching (engine-minted) candidate that must be IGNORED.
    _add_anon(eng, "anon-cccccccc", _session([SA, _u("q1")]))

    def _boom(messages):
        raise AssertionError("prefix scan must not run for explicit session_id")

    eng._resolve_anon_session_id_locked = _boom

    _drive(eng, [SA, _u("q1"), _a("r1"), _u("q2")], session_id="web-ui-1")
    assert record[0]["session_id"] == "web-ui-1"
    # An explicit session_id never becomes anon-provenance, even if it LOOKS
    # like one — generate_stream only resolves (and thus mints) when no
    # session_id was given.
    assert "web-ui-1" not in eng._anon_minted_ids


def test_minted_session_selected_through_generate_stream():
    """End-to-end set-membership wiring: an anon request mints id + provenance
    via generate_stream; the NEXT anonymous request (extending the stored
    conversation) is resolved onto that same minted session."""
    eng = _make_engine()
    record: list[dict] = []
    _wire_generate_stub(eng, record)

    _drive(eng, [SA, _u("q1")])
    sid = record[0]["session_id"]
    assert _ANON_RE.match(sid)
    assert sid in eng._anon_minted_ids

    stored = eng._sessions[sid].messages
    _drive(eng, list(stored) + [_u("q2")])
    assert record[1]["session_id"] == sid


def test_explicit_anon_named_session_not_selected_through_generate_stream():
    """Provenance regression through the full generate_stream path: a client
    first creates an EXPLICIT session with user="anon-aaaaaaaa"; a later
    anonymous request that perfectly prefix-matches it must mint a FRESH id
    and leave the explicit session untouched."""
    eng = _make_engine()
    record: list[dict] = []
    _wire_generate_stub(eng, record)

    _drive(eng, [SA, _u("q1")], session_id="anon-aaaaaaaa")  # explicit
    explicit_msgs = eng._sessions["anon-aaaaaaaa"].messages

    _drive(eng, list(explicit_msgs) + [_u("q2")])  # anonymous, perfect prefix
    sid = record[1]["session_id"]
    assert sid != "anon-aaaaaaaa"
    assert _ANON_RE.match(sid)
    assert eng._sessions["anon-aaaaaaaa"].messages == explicit_msgs  # untouched


def test_delete_session_discards_minted_id():
    """Set hygiene: deleting a session also drops its anon provenance."""
    eng = _make_engine()
    _add_anon(eng, "anon-dddddddd", _session([SA, _u("q1")]))
    eng.delete_session("anon-dddddddd")
    assert "anon-dddddddd" not in eng._sessions
    assert "anon-dddddddd" not in eng._anon_minted_ids


def test_busy_mark_and_eviction_protect_resolved_id():
    """The RESOLVED anon id (not 'anon') is busy-marked during generation and
    passed as protect_session_id to the post-generation eviction sweep."""
    eng = _make_engine()
    record: list[dict] = []
    _wire_generate_stub(eng, record)

    protected: list[str | None] = []
    eng._evict_active_sessions_if_needed = (
        lambda protect_session_id=None: protected.append(protect_session_id)
    )

    _drive(eng, [SA, _u("q1")])
    sid = record[0]["session_id"]
    assert _ANON_RE.match(sid)
    # During generation the resolved id was busy-protected — not "anon".
    assert sid in record[0]["busy_during_gen"]
    assert "anon" not in record[0]["busy_during_gen"]
    # After generation the busy mark is cleared and eviction protects it.
    assert sid not in eng._busy_sessions
    assert protected == [sid]


def test_anon_sessions_participate_in_active_lru_eviction(monkeypatch):
    """anon-* sessions live in _sessions like any other → the active-session
    LRU eviction (commit fe312d8) bounds them; busy ones are protected."""
    eng = _make_engine()
    eng.cfg.memory_budget_gb = 1.5
    saved: list[str] = []
    monkeypatch.setattr(
        eng, "_save_session_to_disk", lambda sid, s: (saved.append(sid) or True)
    )

    _gb = 1_000_000_000

    def _big_session(last_used: float) -> SessionState:
        layer = SimpleNamespace(state=(SimpleNamespace(nbytes=_gb),))
        cache_state = SimpleNamespace(cache=[layer], token_ids=[1])
        return SessionState(
            cache_state=cache_state,
            messages=[_u("hi")],
            total_cache_tokens=1,
            last_used=last_used,
        )

    eng._sessions = {
        "anon-11111111": _big_session(100.0),  # LRU
        "anon-22222222": _big_session(200.0),
        "anon-33333333": _big_session(300.0),  # MRU
    }
    eng._anon_minted_ids = set(eng._sessions)  # all engine-minted
    eng._mark_session_busy("anon-11111111")  # in-flight: must survive

    eng._evict_active_sessions_if_needed()

    assert "anon-11111111" in eng._sessions  # busy-protected
    assert "anon-22222222" not in eng._sessions  # LRU idle anon evicted
    assert saved == ["anon-22222222"]
    assert "anon-33333333" in eng._sessions  # MRU kept
    # Set hygiene: the evicted session's provenance id was discarded too.
    assert eng._anon_minted_ids == {"anon-11111111", "anon-33333333"}
