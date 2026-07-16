"""Cache-contract regression tests — the U1/U21/U3/U4/U6 contract plus the
review batch-2 hardening (F1–F6).

U1/F2 — anon prefix matching is STRICT (last_assistant_wildcard=False): no
      last-assistant wildcard, NO generic suffix equivalence, and
      interrupted turns match only on EXACT content-channel equality (an
      empty resend matches only an empty stored content channel — the
      empty-resend forgery regression). Explicit-session HITs keep the
      historical wildcard for non-interrupted turns and the NARROW prefix
      equivalence for ``interrupted=True`` turns.
F3  — _messages_match compares STRUCTURED fields on all paths: assistant
      tool_calls (canonical name+arguments, incl. legacy XML-in-content
      stored turns) and the tool role's tool_call_id.
U21 — SessionState carries a prompt-contract fingerprint (canonical tools +
      thinking flag); a HIT with a mismatching fingerprint is treated as
      divergence → honest MISS rebuild with the NEW tools.
F5  — a legacy session (fp=None) takes ONE unconditional cold rebuild that
      stamps the fingerprint — never a lenient HIT; interrupted commits
      stamp the CURRENT request contract, never propagate None.
U3  — compact/truncate/regenerate/branch rebuilds tokenize WITH the
      session's stored tool contract (they used to silently drop it).
U4/F4 — NON-cache-resident assistant turns (crash recovery: disk state lags
      the conversation) route to an honest MISS on ALL templates — a manual
      splice is not token-exact vs apply_chat_template (cache poisoning);
      the detection gate (_suffix_blocking_assistants) is kept.
U6/F1 — MTP cache corruption TERMINATES the stream (engine-internal
      finish_reason="error") instead of continuing plain decode from an
      unverifiable target cache; the session cache is invalidated, NOTHING
      is persisted, tool-call parsing is suppressed, and the API boundary
      converts the terminal to an error envelope / 500 (never a valid
      OpenAI finish_reason).
F6  — MTP resume/entry-validation corruption terminates the (empty) stream
      with one invalidation instead of raising out of the generator.
N1/N2 (round 2) — the compacted/cleared tool-result equivalence is OFF in
      strict (anon) mode: a placeholder matches every stored tool result, so
      anon requires exact content. One-sided extractable tool calls FAIL on
      both paths (a start marker no longer defers to the marker-strip
      shortcut); complete bare-name blocks canonicalize first so identical
      calls still match structurally.
N3 (round 3) — RESIDUAL unparsed tool-call start markers (a valid call plus
      a trailing partial/different block — canonically equal to the valid
      call alone) are mismatch evidence when the sides' raw contents differ:
      FAIL on both paths, either direction; byte-identical replays and fully
      parseable multi-block turns keep matching.
Round 4 — a message carrying BOTH tool-call representations (structured
      tool_calls AND tool-call XML in content) must agree with itself:
      _tool_calls_for_match prefers the structured field, so content XML for
      a DIFFERENT call was invisible (no residual — it parses fully) and the
      marker-strip shortcut accepted despite the tokenized prompts
      differing. Conflict → FAIL on both paths, either direction; the same
      call rendered both ways keeps matching.
Round 5 — match-time canonicalization requires CLOSED blocks: the GLM parser
      intentionally accepts a MISSING </tool_call> closer (\\Z) for
      generation-time robustness, so an UNCLOSED GLM block canonicalized
      EQUAL to the structured call it mirrors — evading the one-sided (N2),
      residual-marker (N3) AND round-4 conflict checks — and reached the
      marker-strip/wildcard acceptance despite the tokenized prompts
      differing. The parser stays lenient; _content_xml_calls_for_match now
      discounts the unclosed-block parse, so it surfaces through the
      existing checks: FAIL on both paths, either direction. Closed GLM
      blocks and byte-identical degenerate replays keep matching.

Harness pieces are shared with tests/test_qwen_mtp.py (same-directory import
under pytest's prepend import mode).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx

from mlx_soloheaven.cache.manager import CacheManager
from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState

from test_qwen_mtp import (
    LyingKV,
    MockArrays,
    MockKV,
    MockOps,
    _content_token_suffix,
    _generate_engine,
    _scripted_lm_stream,
    _use_real_messages_match,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _shell_engine() -> MLXEngine:
    """Bare engine shell for resolver/_messages_match tests (no model)."""
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.memory_budget_gb = 0
    eng.cfg = cfg
    eng._lock = threading.Lock()
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._dirty_lock = threading.Lock()
    eng._busy_sessions = set()
    eng._busy_lock = threading.Lock()
    eng._anon_minted_ids = set()
    eng.model_family = "chatml"
    eng.cache_manager = CacheManager(
        memory_budget_gb=1.0, disk_budget_gb=100.0, cache_dir="/tmp/sh-test-cache"
    )
    return eng


def _anon_session(
    eng: MLXEngine, sid: str, messages: list[dict], *,
    last_used: float = 100.0, prompt_fingerprint: str | None = None,
) -> SessionState:
    cache_state = SimpleNamespace(cache=["x"], token_ids=[1, 2, 3])
    s = SessionState(
        cache_state=cache_state,
        messages=messages,
        total_cache_tokens=3,
        last_used=last_used,
        prompt_fingerprint=prompt_fingerprint,
    )
    eng._sessions[sid] = s
    eng._anon_minted_ids.add(sid)
    return s


SA = {"role": "system", "content": "You are agent A. Be terse."}


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str, interrupted: bool = False) -> dict:
    m = {"role": "assistant", "content": text}
    if interrupted:
        m["interrupted"] = True
    return m


def _drive(eng, messages, *, tools=None, thinking=False, max_tokens=8):
    """Drive _generate_locked to completion on the test harness engine."""
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=tools,
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


TOOLS_A = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
TOOLS_B = [{"type": "function", "function": {"name": "get_news", "parameters": {}}}]


# ---------------------------------------------------------------------------
# U1 — anon hijack regression + interrupted narrow equivalence
# ---------------------------------------------------------------------------


def test_anon_resolver_rejects_last_assistant_divergence():
    """THE hijack regression: two anonymous conversations, prefix-equal
    except the last assistant turn, must NOT share a session anymore.
    Pre-fix, _messages_match's last-stored-assistant wildcard let B's
    request select (and then mutate) A's session + cache."""
    eng = _shell_engine()
    _anon_session(
        eng, "anon-aaaaaaaa",
        [SA, _u("q1"), _a("the alpha conversation reply")],
    )
    # Conversation B: same system + user turn, DIFFERENT assistant history.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("a completely different beta reply"), _u("q2")]
    )
    assert sid != "anon-aaaaaaaa"


def test_anon_resolver_still_matches_exact_extension():
    """Control: the same conversation extending its own history resolves."""
    eng = _shell_engine()
    _anon_session(
        eng, "anon-aaaaaaaa",
        [SA, _u("q1"), _a("the alpha conversation reply")],
    )
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("the alpha conversation reply"), _u("q2")]
    )
    assert sid == "anon-aaaaaaaa"


def test_anon_resolver_interrupted_exact_content_equivalence():
    """F2: for ANON resolution an interrupted turn matches only on EXACT
    content-channel equality (after thinking-strip normalization). Prefix
    shapes — the explicit-session narrow equivalence — are forgeable for
    session-less requests and must NOT resolve."""
    eng = _shell_engine()
    stored = [SA, _u("q1"), _a("<think>\npartial thinking about kimchi", interrupted=True)]
    _anon_session(eng, "anon-aaaaaaaa", stored)

    # Cancel mid-thinking: the stored CONTENT channel is empty, so the
    # empty resend (what the client actually received) matches exactly.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a(""), _u("q2")]
    )
    assert sid == "anon-aaaaaaaa"

    # Truncated thinking replayed as plain content: a PREFIX shape — the
    # explicit-session narrow rule, NOT available to anon resolution.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("partial thinking"), _u("q2")]
    )
    assert sid != "anon-aaaaaaaa"

    # Arbitrary divergence: NOT matched — a fresh id is minted.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("some unrelated fabricated reply"), _u("q2")]
    )
    assert sid != "anon-aaaaaaaa"


def test_anon_resolver_empty_resend_forgery_regression():
    """THE F2 forgery regression: an empty resend must NOT match an
    interrupted turn whose stream already produced CONTENT — pre-fix the
    empty string was a prefix of everything, so an empty resend matched
    every interrupted turn."""
    eng = _shell_engine()
    stored = [
        SA, _u("q1"),
        _a("<think>\ndone</think>real answer text so far", interrupted=True),
    ]
    _anon_session(eng, "anon-aaaaaaaa", stored)

    # Empty resend vs non-empty stored content channel → NO match.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a(""), _u("q2")]
    )
    assert sid != "anon-aaaaaaaa"

    # A prefix of the streamed content → still NO match on anon (exact only).
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("real answer"), _u("q2")]
    )
    assert sid != "anon-aaaaaaaa"

    # The EXACT content the client received → matches.
    sid = eng._resolve_anon_session_id_locked(
        [SA, _u("q1"), _a("real answer text so far"), _u("q2")]
    )
    assert sid == "anon-aaaaaaaa"


def test_anon_resolver_fingerprint_isolates_tool_contracts():
    """U21 interplay: identical message prefixes under DIFFERENT tool
    contracts are different conversations — the resolver must not hand one
    the other's session even when it is more recently used."""
    eng = _shell_engine()
    fp_a = MLXEngine._prompt_fingerprint(TOOLS_A, True)
    fp_b = MLXEngine._prompt_fingerprint(TOOLS_B, True)
    msgs = [SA, _u("q1"), _a("same reply")]
    _anon_session(eng, "anon-toolsa00", msgs, last_used=999.0, prompt_fingerprint=fp_a)
    _anon_session(eng, "anon-toolsb00", msgs, last_used=1.0, prompt_fingerprint=fp_b)
    sid = eng._resolve_anon_session_id_locked(
        msgs + [_u("q2")], prompt_fingerprint=fp_b,
    )
    assert sid == "anon-toolsb00"


def test_messages_match_wildcard_is_parameterized():
    """Explicit-session HITs keep the historical last-stored-assistant
    wildcard (default); strict mode (the anon resolver) rejects it."""
    eng = _shell_engine()
    stored = [_u("q1"), _a("stored assistant reply")]
    incoming = [_u("q1"), _a("client-mangled different text"), _u("q2")]
    assert eng._messages_match(stored, incoming) is True
    assert (
        eng._messages_match(stored, incoming, last_assistant_wildcard=False)
        is False
    )


def test_messages_match_interrupted_overrides_wildcard():
    """An interrupted-marked turn NEVER gets the wildcard — arbitrary
    divergence fails even on the lenient (explicit-session) path, while the
    narrow shapes still match there; the marker applies at ANY position.
    F2: the STRICT path (anon) accepts only exact content-channel equality."""
    eng = _shell_engine()
    stored = [_u("q1"), _a("<think>\nsome partial reasoning", interrupted=True)]
    # Narrow prefix shapes accepted on the EXPLICIT path only.
    for resend in ("", "some partial"):
        assert eng._messages_match(stored, [_u("q1"), _a(resend), _u("q2")])
    # Strict path: content channel is empty (cancel mid-thinking) — only the
    # empty resend is exactly equal; the thinking-prefix replay is not.
    assert eng._messages_match(
        stored, [_u("q1"), _a(""), _u("q2")], last_assistant_wildcard=False,
    )
    assert not eng._messages_match(
        stored, [_u("q1"), _a("some partial"), _u("q2")],
        last_assistant_wildcard=False,
    )
    # Arbitrary divergence rejected even with the wildcard available.
    diverged = [_u("q1"), _a("fabricated other reply"), _u("q2")]
    assert eng._messages_match(stored, diverged) is False

    # Non-last interrupted turn (a later turn committed after it).
    stored2 = [
        _u("q1"),
        _a("<think>\nsome partial reasoning", interrupted=True),
        _u("q2"),
        _a("normal full reply"),
    ]
    incoming2 = [_u("q1"), _a(""), _u("q2"), _a("normal full reply"), _u("q3")]
    assert eng._messages_match(stored2, incoming2) is True


def test_interrupted_truncated_content_resend_matches():
    """Wire truncation of a mid-CONTENT cancel: stored has thinking+partial
    content. Explicit path: the client may resend a PREFIX of that content.
    Strict (anon) path: only the EXACT content matches (F2)."""
    eng = _shell_engine()
    stored = [
        _u("q1"),
        _a("<think>\ndone thinking</think>partial answer tex", interrupted=True),
    ]
    assert eng._messages_match(stored, [_u("q1"), _a("partial answ"), _u("q2")])
    # Strict path: prefix no longer matches; exact content does.
    assert not eng._messages_match(
        stored, [_u("q1"), _a("partial answ"), _u("q2")],
        last_assistant_wildcard=False,
    )
    assert eng._messages_match(
        stored, [_u("q1"), _a("partial answer tex"), _u("q2")],
        last_assistant_wildcard=False,
    )
    assert not eng._messages_match(
        stored, [_u("q1"), _a("a longer but different answer"), _u("q2")],
        last_assistant_wildcard=False,
    )


# ---------------------------------------------------------------------------
# F3 — structured tool_calls / tool_call_id participate in matching
# ---------------------------------------------------------------------------


def _tc(name: str, args: str, tc_id: str = "call_1") -> dict:
    return {
        "id": tc_id, "type": "function",
        "function": {"name": name, "arguments": args},
    }


def test_messages_match_compares_tool_calls_on_all_paths():
    """F3: two assistant turns with identical (empty) content but DIFFERENT
    tool calls must not match — on the lenient AND the strict path."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    same = [
        _u("q"),
        {"role": "assistant", "content": "",
         # Re-serialized arguments (spacing) still compare equal.
         "tool_calls": [_tc("get_weather", '{"city": "seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    diff_name = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_news", '{"city":"seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    diff_args = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"busan"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    no_calls = [
        _u("q"),
        {"role": "assistant", "content": ""},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    assert eng._messages_match(stored, same) is True
    assert eng._messages_match(stored, same, last_assistant_wildcard=False)
    for bad in (diff_name, diff_args, no_calls):
        assert eng._messages_match(stored, bad) is False
        assert not eng._messages_match(
            stored, bad, last_assistant_wildcard=False,
        )


def test_messages_match_tool_calls_vs_legacy_xml_content():
    """F3: a legacy stored turn keeping the tool-call XML in content is
    compared STRUCTURALLY against an incoming tool_calls[] field."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        _a("<tool_call>\n<function=get_weather>\n<parameter=city>\nseoul\n"
           "</parameter>\n</function>\n</tool_call>"),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    same = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city": "seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    forged = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("delete_files", '{"path": "/"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    assert eng._messages_match(stored, same) is True
    assert eng._messages_match(stored, forged) is False


def test_messages_match_compares_tool_call_id():
    """F3: the tool role's tool_call_id pins the call chain."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", "{}")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    ok = [
        stored[0], stored[1],
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    forged = [
        stored[0], stored[1],
        {"role": "tool", "tool_call_id": "call_OTHER", "content": "sunny"},
        _u("q2"),
    ]
    assert eng._messages_match(stored, ok) is True
    assert eng._messages_match(stored, forged) is False
    assert not eng._messages_match(
        stored, forged, last_assistant_wildcard=False,
    )


# ---------------------------------------------------------------------------
# N1/N2 — round-2: compaction equivalence gating + one-sided tool-call blocks
# ---------------------------------------------------------------------------


_WEATHER_XML = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nseoul\n"
    "</parameter>\n</function>\n</tool_call>"
)


def test_compacted_tool_result_rejected_on_anon_strict_path():
    """N1: a compacted/cleared tool-result placeholder is equivalent to
    EVERY stored tool result, so the anon STRICT path must reject it — two
    anonymous conversations sharing a call chain (same tool_call_ids) would
    otherwise resolve onto the wrong session. Explicit sessions keep the
    equivalence (the KV cache still holds the real tokens)."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "sunny, 21C, humidity 40%"},
    ]
    for placeholder in ("[cleared]", "[compacted: 2 tool uses]"):
        incoming = [
            stored[0], stored[1],
            {"role": "tool", "tool_call_id": "call_1", "content": placeholder},
            _u("q2"),
        ]
        # Explicit-session (lenient) path: still matched.
        assert eng._messages_match(stored, incoming) is True
        # Anon strict path: exact tool-result content required.
        assert not eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        )


def test_anon_resolver_rejects_compacted_tool_result():
    """N1 at the resolver level: an anonymous request carrying a compacted
    placeholder for the tool turn must NOT resolve onto the stored session;
    the exact tool result still does."""
    eng = _shell_engine()
    stored = [
        SA, _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "sunny, 21C, humidity 40%"},
    ]
    _anon_session(eng, "anon-aaaaaaaa", stored)
    sid = eng._resolve_anon_session_id_locked(
        stored[:3] + [
            {"role": "tool", "tool_call_id": "call_1", "content": "[cleared]"},
            _u("q2"),
        ]
    )
    assert sid != "anon-aaaaaaaa"
    # Control: the exact tool result still resolves.
    sid = eng._resolve_anon_session_id_locked(stored + [_u("q2")])
    assert sid == "anon-aaaaaaaa"


def test_one_sided_garbled_tool_block_rejected_strict_and_lenient():
    """N2: a turn with a VALID canonical call must not match a turn whose
    <tool_call> block is partial/garbled (parse fails). Pre-fix, merely
    containing the start marker dodged the one-sided rejection and the
    marker-strip shortcut then accepted the message — on the strict (anon)
    AND the lenient (explicit-session) path."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        _a(_WEATHER_XML),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    garbled = [
        _u("q"),
        _a("<tool_call>\n<function=delete_files"),  # partial block, no close
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        _u("q2"),
    ]
    # Strict (anon) path: unconditional FAIL.
    assert not eng._messages_match(
        stored, garbled, last_assistant_wildcard=False,
    )
    # Lenient (explicit-session) path: also NOT matched.
    assert eng._messages_match(stored, garbled) is False

    # Mirror direction: the STORED block is the garbled one, the incoming
    # side carries the canonical call.
    stored_garbled = [
        _u("q"),
        _a("<tool_call>\n<function=delete_files"),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    incoming_canonical = [
        stored_garbled[0],
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]},
        stored_garbled[2], _u("q2"),
    ]
    assert not eng._messages_match(
        stored_garbled, incoming_canonical, last_assistant_wildcard=False,
    )
    assert eng._messages_match(stored_garbled, incoming_canonical) is False


def test_anon_resolver_rejects_one_sided_garbled_tool_block():
    """N2 at the resolver level: an anonymous request whose assistant turn
    holds a partial/different <tool_call> block must NOT resolve onto a
    session whose stored turn has a valid canonical call."""
    eng = _shell_engine()
    stored = [
        SA, _u("q"),
        _a(_WEATHER_XML),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    _anon_session(eng, "anon-aaaaaaaa", stored)
    sid = eng._resolve_anon_session_id_locked(
        [
            SA, _u("q"),
            _a("<tool_call>\n<function=delete_files"),
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            _u("q2"),
        ]
    )
    assert sid != "anon-aaaaaaaa"


def test_identical_canonical_tool_calls_still_match():
    """N2 regression control: identical canonical calls on both sides keep
    matching on BOTH paths — the verbatim XML resend and the OpenAI
    reconstruction (structured tool_calls + empty content)."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        _a(_WEATHER_XML),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    resent_xml = [stored[0], _a(_WEATHER_XML), stored[2], _u("q2")]
    reconstructed = [
        stored[0],
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city": "seoul"}')]},
        stored[2], _u("q2"),
    ]
    for incoming in (resent_xml, reconstructed):
        assert eng._messages_match(stored, incoming) is True
        assert eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        ) is True


def test_bare_name_block_canonicalizes_for_match():
    """N2 lenient canonicalization: a COMPLETE bare-name block (GLM shape,
    no <function=> tag) canonicalizes and compares STRUCTURALLY — the same
    call matches, a different call fails (pre-fix the marker fall-through +
    marker-strip shortcut accepted BOTH)."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        _a("<tool_call>\nfoo\n</tool_call>"),
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
    ]
    same = [
        stored[0],
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("foo", "{}")]},
        stored[2], _u("q2"),
    ]
    other = [
        stored[0],
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("bar", "{}")]},
        stored[2], _u("q2"),
    ]
    assert eng._messages_match(stored, same) is True
    assert eng._messages_match(stored, same, last_assistant_wildcard=False)
    assert eng._messages_match(stored, other) is False
    assert not eng._messages_match(
        stored, other, last_assistant_wildcard=False,
    )


_VALID_PLUS_PARTIAL = _WEATHER_XML + "\n<tool_call>\n<function=delete_files"


def test_residual_marker_after_valid_call_rejected_both_paths():
    """N3 (round 3): a turn holding one VALID canonical call PLUS a trailing
    partial/different <tool_call> block canonicalizes to just the valid call
    (_tool_calls_for_match returns only what parses), so it compared EQUAL to
    a turn carrying only the valid call — and the marker-strip shortcut then
    accepted on the strict AND the lenient path. Residual unparsed markers on
    a differing side are mismatch evidence: NOT matched, either direction."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        _a(_VALID_PLUS_PARTIAL),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    for incoming_asst in (
        # Verbatim XML resend with the partial block dropped.
        _a(_WEATHER_XML),
        # OpenAI reconstruction: structured tool_calls + empty content.
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]},
    ):
        incoming = [stored[0], incoming_asst, stored[2], _u("q2")]
        assert eng._messages_match(stored, incoming) is False
        assert not eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        )

    # Mirror direction: stored holds only the valid call; the INCOMING side
    # carries the valid call + the residual partial block.
    stored_clean = [
        _u("q"),
        _a(_WEATHER_XML),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    incoming_dirty = [
        stored_clean[0], _a(_VALID_PLUS_PARTIAL), stored_clean[2], _u("q2"),
    ]
    assert eng._messages_match(stored_clean, incoming_dirty) is False
    assert not eng._messages_match(
        stored_clean, incoming_dirty, last_assistant_wildcard=False,
    )


def test_residual_check_keeps_valid_blocks_matching():
    """N3 regression control: fully-parseable blocks carry no residual —
    identical valid+valid turns keep matching on BOTH paths, whether resent
    as verbatim XML or reconstructed as structured tool_calls; and a
    byte-identical verbatim replay of a degenerate (valid+partial) turn
    still matches (no differing side = no mismatch evidence)."""
    eng = _shell_engine()
    two_calls = (
        _WEATHER_XML
        + "\n<tool_call>\n<function=get_news>\n<parameter=city>\nseoul\n"
          "</parameter>\n</function>\n</tool_call>"
    )
    stored = [
        _u("q"),
        _a(two_calls),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    for incoming_asst in (
        _a(two_calls),  # verbatim XML resend
        {"role": "assistant", "content": "",  # OpenAI reconstruction
         "tool_calls": [_tc("get_weather", '{"city":"seoul"}'),
                        _tc("get_news", '{"city":"seoul"}', "call_2")]},
    ):
        incoming = [stored[0], incoming_asst, stored[2], _u("q2")]
        assert eng._messages_match(stored, incoming) is True
        assert eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        ) is True

    # Byte-identical replay of the degenerate valid+partial turn: matched.
    stored_dirty = [
        _u("q"),
        _a(_VALID_PLUS_PARTIAL),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    replay = [
        stored_dirty[0], _a(_VALID_PLUS_PARTIAL), stored_dirty[2], _u("q2"),
    ]
    assert eng._messages_match(stored_dirty, replay) is True
    assert eng._messages_match(
        stored_dirty, replay, last_assistant_wildcard=False,
    ) is True


# ---------------------------------------------------------------------------
# Round 4 — structured tool_calls vs tool-call XML in content must agree
# ---------------------------------------------------------------------------


_DELETE_XML = (
    "<tool_call>\n<function=delete_files>\n<parameter=path>\n/\n"
    "</parameter>\n</function>\n</tool_call>"
)

_TOOL_OK = {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_structured_call_with_conflicting_content_xml_rejected():
    """Round 4: both sides share the SAME structured call, but one side's
    content ALSO embeds a fully parseable <tool_call> block for a DIFFERENT
    call (delete_files). Pre-fix _tool_calls_for_match returned the
    structured list immediately (content never consulted), no residual
    marker existed (the XML parses), and the marker-strip shortcut accepted
    the message on the strict AND the lenient path — despite the tokenized
    prompts differing. NOT matched: both paths, both directions."""
    eng = _shell_engine()
    clean = {"role": "assistant", "content": "",
             "tool_calls": [_tc("safe", "{}")]}
    conflicted = {"role": "assistant", "content": _DELETE_XML,
                  "tool_calls": [_tc("safe", "{}")]}

    # Direction 1: stored clean, incoming self-conflicting.
    stored = [_u("q"), clean, _TOOL_OK]
    incoming = [_u("q"), conflicted, _TOOL_OK, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )

    # Direction 2 (mirror): stored self-conflicting, incoming clean.
    stored = [_u("q"), conflicted, _TOOL_OK]
    incoming = [_u("q"), clean, _TOOL_OK, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )

    # The conflict beats even the last-stored-assistant wildcard (the
    # strongest lenient-path tolerance).
    stored_last = [_u("q"), clean]
    incoming_last = [_u("q"), conflicted, _TOOL_OK, _u("q2")]
    assert eng._messages_match(stored_last, incoming_last) is False
    assert not eng._messages_match(
        stored_last, incoming_last, last_assistant_wildcard=False,
    )


def test_structured_call_with_agreeing_content_xml_matches():
    """Round-4 regression control: the SAME call rendered BOTH ways
    (structured tool_calls + the equivalent content XML) is self-consistent
    — matched on both paths against the OpenAI reconstruction (structured +
    empty content), in either direction, and as a verbatim replay."""
    eng = _shell_engine()
    both_ways = {
        "role": "assistant", "content": _WEATHER_XML,
        # Re-serialized arguments (spacing) still canonically agree.
        "tool_calls": [_tc("get_weather", '{"city": "seoul"}')],
    }
    structured_only = {
        "role": "assistant", "content": "",
        "tool_calls": [_tc("get_weather", '{"city":"seoul"}')],
    }
    tool = {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}

    for s_asst, i_asst in (
        (both_ways, structured_only),
        (structured_only, both_ways),
        (both_ways, both_ways),  # verbatim replay
    ):
        stored = [_u("q"), s_asst, tool]
        incoming = [_u("q"), i_asst, tool, _u("q2")]
        assert eng._messages_match(stored, incoming) is True
        assert eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        ) is True


def test_structured_call_empty_content_replay_still_matches():
    """Round-4 regression control: structured call + empty content resent
    verbatim (the engine's own stored shape and the OpenAI wire shape) keeps
    matching on both paths."""
    eng = _shell_engine()
    stored = [
        _u("q"),
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("safe", "{}")]},
        _TOOL_OK,
    ]
    incoming = [
        stored[0],
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("safe", "{}")]},
        _TOOL_OK, _u("q2"),
    ]
    assert eng._messages_match(stored, incoming) is True
    assert eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    ) is True


# ---------------------------------------------------------------------------
# Round 5 — GLM: match-time canonicalization requires CLOSED blocks
# ---------------------------------------------------------------------------


_GLM_WEATHER_UNCLOSED = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key><arg_value>seoul</arg_value>"
)
_GLM_WEATHER_CLOSED = _GLM_WEATHER_UNCLOSED + "\n</tool_call>"

_GLM_TOOL = {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}


def _glm_engine() -> MLXEngine:
    eng = _shell_engine()
    eng.model_family = "glm"
    return eng


def test_glm_unclosed_block_equal_to_structured_rejected():
    """Round 5: the GLM parser accepts a MISSING </tool_call> closer (\\Z),
    so an UNCLOSED block canonicalized EQUAL to the structured call it
    mirrors — no one-sided fail (both sides held calls), no residual marker
    (the lenient parse consumed the start marker), no round-4 conflict — and
    the marker-strip shortcut then accepted the message on the strict AND
    the lenient path despite the tokenized prompts differing. Match-time
    canonicalization now requires CLOSED blocks: NOT matched, both paths,
    both directions."""
    eng = _glm_engine()
    structured = {"role": "assistant", "content": "",
                  "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]}

    # Direction 1: stored holds the unclosed content block, incoming the
    # canonically-equal structured call.
    stored = [_u("q"), _a(_GLM_WEATHER_UNCLOSED), _GLM_TOOL]
    incoming = [_u("q"), structured, _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )

    # Direction 2 (mirror): stored structured, incoming unclosed content.
    stored = [_u("q"), structured, _GLM_TOOL]
    incoming = [_u("q"), _a(_GLM_WEATHER_UNCLOSED), _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )


def test_glm_structured_with_unclosed_agreeing_content_rejected():
    """Round 5 × round 4: a side carrying BOTH representations where the
    content XML is an UNCLOSED block AGREEING with the structured call
    evaded the conflict check (the lenient parse compared canonically
    equal). With closedness required at match time the side is
    self-conflicting: NOT matched, both paths, both directions."""
    eng = _glm_engine()
    clean = {"role": "assistant", "content": "",
             "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]}
    conflicted = {"role": "assistant", "content": _GLM_WEATHER_UNCLOSED,
                  "tool_calls": [_tc("get_weather", '{"city":"seoul"}')]}

    # Direction 1: stored clean, incoming self-conflicting.
    stored = [_u("q"), clean, _GLM_TOOL]
    incoming = [_u("q"), conflicted, _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )

    # Direction 2 (mirror): stored self-conflicting, incoming clean.
    stored = [_u("q"), conflicted, _GLM_TOOL]
    incoming = [_u("q"), clean, _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )


def test_glm_closed_agreeing_block_still_matches():
    """Round-5 regression control: a CLOSED GLM block keeps matching on both
    paths — the verbatim XML resend AND the OpenAI reconstruction
    (structured tool_calls + empty content), either direction."""
    eng = _glm_engine()
    structured = {"role": "assistant", "content": "",
                  # Re-serialized arguments (spacing) still compare equal.
                  "tool_calls": [_tc("get_weather", '{"city": "seoul"}')]}
    stored = [_u("q"), _a(_GLM_WEATHER_CLOSED), _GLM_TOOL]
    for incoming_asst in (_a(_GLM_WEATHER_CLOSED), structured):
        incoming = [_u("q"), incoming_asst, _GLM_TOOL, _u("q2")]
        assert eng._messages_match(stored, incoming) is True
        assert eng._messages_match(
            stored, incoming, last_assistant_wildcard=False,
        ) is True
    # Mirror: stored structured, incoming the closed XML resend.
    stored = [_u("q"), structured, _GLM_TOOL]
    incoming = [_u("q"), _a(_GLM_WEATHER_CLOSED), _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is True
    assert eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    ) is True


def test_glm_unclosed_only_content_one_sided_rejected():
    """Round 5 × N2: unclosed-only GLM content (no structured field) has NO
    match-extractable calls, so against a side holding >=1 canonical call it
    is the one-sided N2 FAIL — the existing N2 rule holds for GLM too. A
    byte-identical replay of the degenerate turn still matches (no differing
    side = no mismatch evidence, mirroring the N3 control)."""
    eng = _glm_engine()
    unclosed_other = (
        "<tool_call>delete_files\n"
        "<arg_key>path</arg_key><arg_value>/</arg_value>"
    )
    stored = [_u("q"), _a(_GLM_WEATHER_CLOSED), _GLM_TOOL]
    incoming = [_u("q"), _a(unclosed_other), _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )
    # Mirror direction: the STORED side holds the unclosed-only block.
    stored = [_u("q"), _a(unclosed_other), _GLM_TOOL]
    incoming = [_u("q"), _a(_GLM_WEATHER_CLOSED), _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, incoming) is False
    assert not eng._messages_match(
        stored, incoming, last_assistant_wildcard=False,
    )
    # Byte-identical replay of the degenerate unclosed-only turn: matched.
    replay = [_u("q"), _a(unclosed_other), _GLM_TOOL, _u("q2")]
    assert eng._messages_match(stored, replay) is True
    assert eng._messages_match(
        stored, replay, last_assistant_wildcard=False,
    ) is True


# ---------------------------------------------------------------------------
# U21 — prompt-contract fingerprint on HIT
# ---------------------------------------------------------------------------


def _contract_engine(monkeypatch, scripts):
    """C1-style harness engine with a pre-seeded HIT session + scripted
    lm_stream; returns (eng, cs, messages, prompts_seen, tokenized)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    tokenized: list = []

    def _tok(msgs, thinking=True, tools=None):
        tokenized.append((thinking, tools))
        return [11, 12, 13, 14]

    eng._tokenize_prompt = _tok
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, scripts)
    return eng, cs, messages, prompts_seen, tokenized


def _stamp_contract(eng, tools, thinking):
    sess = eng._sessions["s"]
    sess.tools = tools
    sess.thinking = thinking
    sess.prompt_fingerprint = MLXEngine._prompt_fingerprint(tools, thinking)


def test_hit_with_matching_tools_fingerprint(monkeypatch):
    """Control: same tools as the session was built with → HIT."""
    eng, cs, messages, prompts_seen, tokenized = _contract_engine(
        monkeypatch, [[(101, "x"), (102, "x")]],
    )
    _stamp_contract(eng, TOOLS_A, False)
    chunks = _drive(eng, messages, tools=TOOLS_A)
    assert chunks[-1].cache_info["cache_mode"] == "hit"
    assert prompts_seen[0] == [52, 99]  # stored + suffix splice
    assert tokenized == []  # HIT never re-tokenizes


def test_tools_change_on_hit_forces_honest_miss(monkeypatch):
    """U21: tools changed mid-session → fingerprint mismatch → the HIT is
    refused, the prompt is re-tokenized WITH the new tools, and the saved
    session carries the NEW contract."""
    eng, cs, messages, prompts_seen, tokenized = _contract_engine(
        monkeypatch, [[(101, "x"), (102, "x")]],
    )
    _stamp_contract(eng, TOOLS_A, False)
    chunks = _drive(eng, messages, tools=TOOLS_B)
    assert chunks[-1].cache_info["cache_mode"] == "miss"
    assert prompts_seen[0] == [11, 12, 13, 14]  # full re-tokenization
    assert tokenized == [(False, TOOLS_B)]  # rebuilt WITH the new tools
    sess = eng._sessions["s"]
    assert sess.tools == TOOLS_B
    assert sess.prompt_fingerprint == MLXEngine._prompt_fingerprint(TOOLS_B, False)


def test_thinking_flip_on_hit_forces_honest_miss(monkeypatch):
    """The fingerprint also covers the thinking flag.

    Round 4 (batch 4, finding 4): the script now CLOSES the thought block —
    with the router-authoritative channel policy, a thinking=True stream
    that never emits </think> is a thinking-only (empty content) turn and
    takes the empty-response SKIP path, which carries no cache_info. The
    contract-flip behavior under test is unchanged."""
    eng, cs, messages, prompts_seen, tokenized = _contract_engine(
        monkeypatch, [[(101, "x"), (102, "</think>ok")]],
    )
    _stamp_contract(eng, None, False)
    chunks = _drive(eng, messages, thinking=True)
    assert chunks[-1].cache_info["cache_mode"] == "miss"
    assert tokenized == [(True, None)]


def test_legacy_session_without_fingerprint(monkeypatch):
    """F5: a legacy session (fingerprint=None — pre-upgrade disk file) was
    built under an UNKNOWN contract, so EVERY request — toolless included —
    takes ONE unconditional cold rebuild (honest MISS) whose save stamps the
    fingerprint; never a lenient HIT. The very next same-conversation
    request then HITs normally."""
    # Toolless request on a legacy session → honest MISS + stamped contract.
    eng, cs, messages, prompts_seen, tokenized = _contract_engine(
        monkeypatch, [[(101, "x"), (102, "x")], [(103, "x")]],
    )
    eng._sessions["s"].prompt_fingerprint = None  # legacy
    eng._sessions["s"].tools = None
    chunks = _drive(eng, messages)
    assert chunks[-1].cache_info["cache_mode"] == "miss"
    assert tokenized == [(False, None)]  # full cold rebuild
    fp = MLXEngine._prompt_fingerprint(None, False)
    assert eng._sessions["s"].prompt_fingerprint == fp  # stamped exactly once

    # The stamped session serves the NEXT turn as a normal HIT.
    messages2 = list(eng._sessions["s"].messages) + [_u("q9")]
    chunks2 = _drive(eng, messages2)
    assert chunks2[-1].cache_info["cache_mode"] == "hit"
    assert eng._sessions["s"].prompt_fingerprint == fp

    # Tool-bearing request on a fresh legacy session → honest MISS too.
    eng2, cs2, messages2, prompts_seen2, tokenized2 = _contract_engine(
        monkeypatch, [[(101, "x"), (102, "x")]],
    )
    eng2._sessions["s"].prompt_fingerprint = None  # legacy
    chunks2 = _drive(eng2, messages2, tools=TOOLS_A)
    assert chunks2[-1].cache_info["cache_mode"] == "miss"
    assert tokenized2 == [(False, TOOLS_A)]
    assert eng2._sessions["s"].prompt_fingerprint == MLXEngine._prompt_fingerprint(
        TOOLS_A, False,
    )


def test_interrupted_commit_stamps_current_fingerprint():
    """F5: _commit_interrupted_hit_turn stamps the CURRENT request contract
    — a commit must never carry prompt_fingerprint=None forward (that would
    re-open the legacy leniency indefinitely)."""
    from mlx_soloheaven.engine.mlx_engine import TurnCloseResult
    from test_qwen_mtp import MockKV as _MockKV

    def _setup():
        eng = _shell_engine()
        cache = [_MockKV() for _ in range(3)]
        for c in cache:
            c.offset = 8
        cs = SimpleNamespace(
            cache=cache, token_ids=list(range(8)),
            mtp_last_hidden=None, mtp_hidden_offset=None,
        )
        session = SessionState(
            cache_state=cs,
            messages=[_u("q1")],
            total_cache_tokens=6,
            prompt_fingerprint=None,  # worst case: legacy leftovers
        )
        eng._sessions["s"] = session
        eng._try_close_interrupted_turn = (
            lambda sid, cstate: TurnCloseResult.NOT_REQUIRED
        )
        eng._make_full_assistant_content = lambda text, thinking: text
        eng._get_cache_offset = lambda cache_list: 8
        return eng, cs, session

    # Explicit fingerprint from the request → stamped verbatim.
    eng, cs, session = _setup()
    fp = MLXEngine._prompt_fingerprint(TOOLS_A, False)
    eng._commit_interrupted_hit_turn(
        session_id="s", session=session, cache_state=cs,
        new_messages=[_u("q2")], accumulated_text="partial",
        use_thinking=False, hit_prior_len=6, prompt_len=7,
        reason="cancelled", tools_canonical=TOOLS_A, prompt_fingerprint=fp,
    )
    committed = eng._sessions["s"]
    assert committed.prompt_fingerprint == fp
    assert committed.tools == TOOLS_A
    assert committed.thinking is False
    assert committed.messages[-1]["interrupted"] is True

    # No explicit fingerprint (defensive default) → computed, NEVER None.
    eng, cs, session = _setup()
    eng._commit_interrupted_hit_turn(
        session_id="s", session=session, cache_state=cs,
        new_messages=[_u("q2")], accumulated_text="partial",
        use_thinking=True, hit_prior_len=6, prompt_len=7,
        reason="cancelled",
    )
    committed = eng._sessions["s"]
    assert committed.prompt_fingerprint == MLXEngine._prompt_fingerprint(
        None, True,
    )


# ---------------------------------------------------------------------------
# U3 — rebuild paths carry the session's tool contract
# ---------------------------------------------------------------------------


def _rebuild_engine():
    """Harness engine with the extra surface compact/_rebuild_session need,
    plus recorders for every contract-consuming call."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    eng._lock = threading.Lock()
    eng._mark_dirty = lambda sid: None
    eng._evict_active_sessions_if_needed = lambda protect_session_id=None: None
    # Round 5, finding 3a: rebuild/compact prefills now thread the shutdown
    # cancel event — accept it (same shape as the F4 stub convention).
    eng._prefill_cache = lambda cache, tokens, cancel_event=None: None
    eng._eval_cache = lambda cache: None

    seen: dict = {"tok": [], "base": [], "reg": []}

    def _tok(msgs, thinking=True, tools=None):
        seen["tok"].append((thinking, tools))
        return [1, 2, 3]

    def _base(msgs, tools=None):
        seen["base"].append(tools)
        return None

    def _reg(msgs, toks, tools=None, thinking=True, cancel_event=None):
        seen["reg"].append((tools, thinking))

    eng._tokenize_prompt = _tok
    eng._find_base_cache = _base
    eng._maybe_register_base_cache = _reg
    return eng, seen


def test_compact_session_rebuilds_with_session_tools():
    """compact_session must tokenize / consult base caches / register WITH
    the session's stored tool contract, and stamp it on the new state."""
    eng, seen = _rebuild_engine()
    _stamp_contract(eng, TOOLS_A, False)

    result = eng.compact_session("s", [_u("compacted summary"), _a("ok")])
    assert result["status"] == "ok"
    assert seen["tok"] == [(False, TOOLS_A)]
    assert seen["base"] == [TOOLS_A]
    assert seen["reg"] == [(TOOLS_A, False)]
    sess = eng._sessions["s"]
    assert sess.tools == TOOLS_A
    assert sess.thinking is False
    assert sess.prompt_fingerprint == MLXEngine._prompt_fingerprint(TOOLS_A, False)


def test_truncate_and_regenerate_rebuild_with_session_tools():
    """truncate_session (and prepare_regenerate through it) → the REAL
    _rebuild_session must re-tokenize with the session's tools and keep the
    contract on the rebuilt SessionState."""
    eng, seen = _rebuild_engine()
    _stamp_contract(eng, TOOLS_A, False)
    # Harness session is [u1, a1] — regenerate truncates to BEFORE u1
    # (restore_to = len - 2 = 0) so the client resends the user turn.
    result = eng.prepare_regenerate("s")
    assert result.get("status") == "ok", result
    assert seen["tok"] == [(False, TOOLS_A)]
    assert seen["base"] == [TOOLS_A]
    sess = eng._sessions["s"]
    assert sess.tools == TOOLS_A
    assert sess.thinking is False
    assert sess.prompt_fingerprint == MLXEngine._prompt_fingerprint(TOOLS_A, False)
    assert sess.messages == []  # truncated to before the user turn


def test_branch_from_turn_inherits_source_contract():
    eng, seen = _rebuild_engine()
    _stamp_contract(eng, TOOLS_A, False)
    result = eng.branch_from_turn("s", "s-branch", 1)
    assert result.get("status") == "ok", result
    assert seen["tok"] == [(False, TOOLS_A)]
    branch = eng._sessions["s-branch"]
    assert branch.tools == TOOLS_A
    assert branch.prompt_fingerprint == MLXEngine._prompt_fingerprint(TOOLS_A, False)


# ---------------------------------------------------------------------------
# U4 — non-cache-resident assistant turns in the suffix (crash recovery)
# ---------------------------------------------------------------------------


def test_suffix_blocking_assistants_policy():
    """F4: a NON-cache-resident assistant turn blocks the suffix path on
    EVERY template (a manual splice is not token-exact vs
    apply_chat_template — cache-poisoning risk); the detection gate stays."""
    eng = _shell_engine()
    plain = [_a("prior reply"), _u("next question")]
    with_tc = [dict(_a("x"), tool_calls=[{"id": "1"}]), _u("q")]
    users_only = [_u("q")]

    for family in ("chatml", "gemma4", "glm"):
        eng.model_family = family
        assert eng._suffix_blocking_assistants(users_only) == 0
        assert eng._suffix_blocking_assistants(plain) == 1
        assert eng._suffix_blocking_assistants(with_tc) == 1


def test_chatml_suffix_builder_never_renders_assistants():
    """F4: the chatml builder does NOT splice assistant turns (the U4 gate
    routes any non-resident assistant to an honest MISS before the builder
    runs; cache-resident ones are already in the KV)."""
    eng = _shell_engine()
    captured: dict = {}

    def _encode(s, add_special_tokens=False):
        captured["text"] = s
        return [7, 8, 9]

    eng.tokenizer = SimpleNamespace(encode=_encode)
    out = eng._suffix_tokens_chatml(
        [_a("the model's prior reply"), _u("next question")], thinking=False,
    )
    assert out == [7, 8, 9]
    text = captured["text"]
    assert "the model's prior reply" not in text  # no manual splice
    assert "next question" in text
    assert text.endswith("<|im_start|>assistant\n")  # generation prompt intact


def _crash_recovery_engine():
    """HIT-capable harness whose stored (disk-reloaded) messages END at u1,
    while the client resends [u1, a1, u2] — the crash-recovery shape."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, _harness_msgs = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    eng._sessions["s"].messages = [_u("u1")]
    messages = [_u("u1"), _a("assistant reply lost from disk"), _u("u2")]

    tokenized: list = []

    def _tok(msgs, thinking=True, tools=None):
        tokenized.append(list(msgs))
        return [11, 12, 13, 14]

    eng._tokenize_prompt = _tok
    return eng, messages, tokenized


def test_crash_recovery_takes_honest_miss_on_all_templates(monkeypatch):
    """End-to-end (F4): a non-resident assistant turn in the resent history
    is a divergence on EVERY template — honest MISS with a full
    re-tokenization, never a manual splice (which is not token-exact vs
    apply_chat_template: e.g. Qwen3.6 renders '<think>\\n\\n</think>\\n\\n'
    into past assistant turns when thinking is disabled)."""
    for family in ("chatml", "gemma4", "glm"):
        eng, messages, tokenized = _crash_recovery_engine()
        eng.model_family = family
        prompts_seen: list = []
        _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "x"), (102, "x")]])
        chunks = _drive(eng, messages, max_tokens=2)
        assert chunks[-1].cache_info["cache_mode"] == "miss", family
        assert prompts_seen[0] == [11, 12, 13, 14]
        assert tokenized and tokenized[0] == messages  # full honest rebuild


# ---------------------------------------------------------------------------
# U6 — MTP corruption terminates the stream
# ---------------------------------------------------------------------------


def test_generate_locked_mtp_corruption_terminates_with_error(monkeypatch):
    """End-to-end: a settle-time MTPCacheCorruption (lying target trim) on a
    resumed MTP HIT session must (1) TERMINATE the stream at the corruption
    point, (2) surface finish_reason='error' on the terminal frame, (3)
    invalidate the session cache fail-closed, and (4) make the next
    same-history turn an honest MISS cold-fill."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    STOP = 7
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63  # always wrong → settle must restore (and fail)
    stored = [1, 2, 3, 4, STOP]
    suffix = [40, 20, 21]

    # LyingKV target KV layers: trim() claims success but never rewinds —
    # the settle restore's per-layer verification raises MTPCacheCorruption.
    cache = [
        MockArrays(), MockArrays(), LyingKV(), MockArrays(), LyingKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1  # finalized head: lazy last slot
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    cs.mtp_last_hidden = mx.array([[[float(STOP)]]])
    cs.mtp_hidden_offset = len(stored)

    prompt_lens: list = []

    def fake_step(prompt, model, head, **kwargs):
        prompt_lens.append(int(prompt.size))
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, head_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)
    # Same mock-vs-real artifact note as the C1 MTP cancel test.
    monkeypatch.setattr(
        qwen_mtp_mod, "make_head_cache",
        lambda n: [MockKV() for _ in range(max(1, n))],
    )

    orig_sess = eng._sessions["s"]
    orig_messages = [dict(m) for m in orig_sess.messages]
    chunks = _drive(eng, messages, max_tokens=8)

    # (1) Terminated at the corruption point: bootstrap + one bonus token,
    # NOT the 8 tokens a plain-decode fallback would have produced.
    content = [c for c in chunks if c.finish_reason is None]
    assert len(content) == 2, [c.token for c in content]
    # (2) The terminal frame reports the abnormal end.
    assert chunks[-1].finish_reason == "error"
    # (3) Session cache invalidated fail-closed (stash included).
    assert cs.cache is None and cs.token_ids is None
    assert cs.mtp_last_hidden is None and cs.mtp_hidden_offset is None
    # (3b) F1: NOTHING persisted — the truncated text must not enter
    # session.messages (the save is skipped on the corruption path).
    assert eng._sessions["s"] is orig_sess
    assert eng._sessions["s"].messages == orig_messages

    # (4) Next same-history request: honest MISS → FULL cold-fill.
    def fake_step2(prompt, model, head, **kwargs):
        prompt_lens.append(int(prompt.size))
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step2)
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13, 14]
    chunks2 = _drive(eng, messages, max_tokens=3)
    # U7: the scripted retry exhausts max_tokens without an EOS — a normal
    # (non-error) terminal that now correctly reports "length".
    assert chunks2[-1].finish_reason == "length"
    assert prompt_lens[-1] == 4  # full prompt, not a stale suffix splice


def test_corruption_suppresses_toolcalls_and_persistence(monkeypatch):
    """F1 (engine side): on a corruption-terminated stream the truncated
    text must NEVER be parsed into tool calls (finish_reason stays 'error',
    not 'tool_calls') and NOTHING is persisted into session.messages — even
    when the truncated text happens to contain a complete-looking
    <tool_call> block."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step
    from mlx_soloheaven.engine.tool_parser import parse_tool_calls

    STOP = 7
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63  # always wrong → settle must restore (and fail)
    stored = [1, 2, 3, 4, STOP]
    suffix = [40, 20, 21]

    cache = [
        MockArrays(), MockArrays(), LyingKV(), MockArrays(), LyingKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    cs.mtp_last_hidden = mx.array([[[float(STOP)]]])
    cs.mtp_hidden_offset = len(stored)
    # Request carries tools; the session's contract matches (U21 gate).
    sess = eng._sessions["s"]
    sess.tools = TOOLS_A
    sess.prompt_fingerprint = MLXEngine._prompt_fingerprint(TOOLS_A, False)
    # Every decoded token forms a complete tool_call block — bait for the
    # tool-call parser.
    BLOCK = "<tool_call>\n<function=hack>\n</function>\n</tool_call>"
    eng.tokenizer = SimpleNamespace(decode=lambda ids: BLOCK, eos_token_ids=[])
    # Sanity: the bait WOULD parse into a call if parsing were not suppressed.
    assert parse_tool_calls(BLOCK, model_family="chatml")[1]

    def fake_step(prompt, model, head, **kwargs):
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, head_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)
    monkeypatch.setattr(
        qwen_mtp_mod, "make_head_cache",
        lambda n: [MockKV() for _ in range(max(1, n))],
    )

    orig_sess = eng._sessions["s"]
    orig_messages = [dict(m) for m in orig_sess.messages]
    chunks = _drive(eng, messages, tools=TOOLS_A, max_tokens=8)

    # 'error' terminal — never 'tool_calls' from truncated text.
    assert chunks[-1].finish_reason == "error"
    # Nothing persisted (no session save, no tool_calls in messages).
    assert eng._sessions["s"] is orig_sess
    assert eng._sessions["s"].messages == orig_messages
    assert not any(m.get("tool_calls") for m in eng._sessions["s"].messages)


# ---------------------------------------------------------------------------
# Disk round-trip: contract fields + interrupted marker
# ---------------------------------------------------------------------------


def test_contract_and_marker_disk_roundtrip(tmp_path, monkeypatch):
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.generate import PromptCacheState
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    def _kv(seq_len=4, num_heads=2, head_dim=8):
        c = KVCache()
        k = mx.arange(num_heads * seq_len * head_dim, dtype=mx.float32).reshape(
            1, num_heads, seq_len, head_dim
        )
        c.update_and_fetch(k, k * 0.5)
        return c

    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.disk_budget_gb = 1.0
    eng.cfg = cfg
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    eng._language_model = SimpleNamespace()
    eng._use_vlm = False
    eng._draft_kind = None

    layers = [_kv(), _kv(), _kv()]
    cache_state = PromptCacheState()
    cache_state.cache = layers
    cache_state.token_ids = list(range(4))
    fp = MLXEngine._prompt_fingerprint(TOOLS_A, False)
    session = SessionState(
        cache_state=cache_state,
        messages=[
            _u("hi"),
            _a("<think>\npartial reasoning", interrupted=True),
        ],
        total_cache_tokens=4,
        tools=TOOLS_A,
        thinking=False,
        prompt_fingerprint=fp,
    )
    assert eng._save_session_to_disk("contract-1", session) is True

    monkeypatch.setattr(
        mlx_engine_module, "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(len(layers))],
    )
    loaded = eng._load_session_from_disk("contract-1")
    assert loaded is not None
    assert loaded.tools == TOOLS_A
    assert loaded.thinking is False
    assert loaded.prompt_fingerprint == fp
    # The U1 marker survives the JSON round-trip.
    assert loaded.messages[-1]["interrupted"] is True
    assert loaded.messages[-1]["content"] == "<think>\npartial reasoning"
