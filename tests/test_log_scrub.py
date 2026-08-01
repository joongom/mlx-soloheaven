"""Batch D — log-leak scrubbing (no auth; scrub-only per user decision).

Covers:
  * prompt / message CONTENT never appears in any log record for a completion
    (the request-preview log is scrubbed to role+length metadata);
  * the 422 validation handler logs no request body / input values (only the
    error type+loc), so a prompt/secret in an invalid request cannot leak;
  * the admin log endpoints (/logs/stream, /logs/recent) are localhost-gated —
    403 for a non-loopback client, allowed for loopback;
  * the LogBuffer that feeds the admin SSE contains no prompt content (source is
    scrubbed).
"""

from __future__ import annotations

import asyncio
import logging
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

import threading

import mlx_soloheaven.engine.mlx_engine as mlx_engine_module
from mlx_soloheaven.api import admin, openai_compat
from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

# The REAL _generate_locked harness (same-directory import — the shared engine
# generation harness lives in test_qwen_mtp; see test_cancellation_gaps).
from test_qwen_mtp import MockKV, _generate_engine

SECRET = "SUPERSECRETPROMPT_hunter2_ssn-123-45-6789"


class _CompletionStubEngine:
    """Minimal engine for the non-streaming openai completion path."""

    model_family = "chatml"
    model_id = "stub-model"
    cfg = SimpleNamespace(enable_thinking=False)

    def complete(self, messages, **kwargs):
        return SimpleNamespace(
            content="ok", thinking=None, tool_calls=None,
            finish_reason="stop", prompt_tokens=3, completion_tokens=1,
            cache_info={"cache_mode": "miss"},
        )

    def update_session_messages(self, *a, **k):
        pass


def _run_completion(secret_text: str):
    eng = _CompletionStubEngine()
    old_engines = openai_compat._engines
    old_default = openai_compat._default_engine
    try:
        openai_compat.set_engines({"stub-model": eng}, eng)
        req = ChatCompletionRequest(
            model="stub-model", stream=False, thinking=False,
            messages=[
                ChatMessage(role="system", content="be helpful"),
                ChatMessage(role="user", content=secret_text),
            ],
        )
        return asyncio.run(openai_compat.chat_completions(req))
    finally:
        openai_compat.set_engines(old_engines, old_default)


# --------------------------------------------------------------------------
# 1. Prompt content never logged for a completion.
# --------------------------------------------------------------------------

def test_completion_log_does_not_leak_prompt_content(caplog):
    with caplog.at_level(logging.DEBUG):
        _run_completion(SECRET)
    for record in caplog.records:
        assert SECRET not in record.getMessage(), (
            f"prompt content leaked into log: {record.getMessage()!r}"
        )


def test_completion_request_log_has_safe_metadata(caplog):
    with caplog.at_level(logging.INFO, logger="mlx_soloheaven.api.openai_compat"):
        _run_completion(SECRET)
    request_logs = [
        r.getMessage() for r in caplog.records if "[Request]" in r.getMessage()
    ]
    assert request_logs, "expected a [Request] log line"
    joined = " ".join(request_logs)
    # Safe metadata is present (roles + counts), content is redacted.
    assert "messages=2" in joined
    assert "redacted" in joined
    assert SECRET not in joined


# --------------------------------------------------------------------------
# 2. LogBuffer (admin SSE source) contains no prompt content.
# --------------------------------------------------------------------------

def test_log_buffer_contains_no_prompt_content():
    buffer = admin.log_buffer
    buffer.buffer.clear()
    root = logging.getLogger()
    root.addHandler(buffer)
    prev_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        _run_completion(SECRET)
    finally:
        root.removeHandler(buffer)
        root.setLevel(prev_level)
    for entry in list(buffer.buffer):
        assert SECRET not in entry["message"], (
            f"prompt content leaked into the admin LogBuffer: {entry!r}"
        )


# --------------------------------------------------------------------------
# 3. 422 validation handler logs no request body / input values.
# --------------------------------------------------------------------------

class _NeedsInt(BaseModel):
    n: int


def _validation_app() -> FastAPI:
    from mlx_soloheaven.server import register_validation_error_handler

    app = FastAPI()
    register_validation_error_handler(app)

    @app.post("/needs-int")
    async def needs_int(body: _NeedsInt):
        return {"n": body.n}

    return app


def test_422_handler_does_not_log_request_body(caplog):
    client = TestClient(_validation_app(), raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG):
        r = client.post("/needs-int", json={"n": SECRET})
    assert r.status_code == 422
    # Response already scrubbed (Batch B); the LOG must be scrubbed too (Batch D).
    for record in caplog.records:
        msg = record.getMessage()
        assert SECRET not in msg, f"422 log leaked the input value: {msg!r}"
        assert "body=" not in msg, f"422 log still emits the raw body: {msg!r}"


def test_422_handler_logs_safe_error_type_and_loc(caplog):
    client = TestClient(_validation_app(), raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="soloheaven"):
        client.post("/needs-int", json={"n": "not-an-int"})
    logs = [r.getMessage() for r in caplog.records if "[422]" in r.getMessage()]
    assert logs, "expected a [422] log line"
    joined = " ".join(logs)
    # loc identifies which field was bad, WITHOUT the offending value.
    assert "loc" in joined
    assert "not-an-int" not in joined


# --------------------------------------------------------------------------
# 4. Admin log endpoints are localhost-gated.
# --------------------------------------------------------------------------

def _admin_app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin.router)
    return app


def _admin_client(host: str) -> TestClient:
    return TestClient(_admin_app(), client=(host, 12345), raise_server_exceptions=False)


def test_logs_recent_rejects_non_loopback():
    r = _admin_client("203.0.113.7").get("/api/admin/logs/recent")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


def test_logs_recent_allows_loopback():
    r = _admin_client("127.0.0.1").get("/api/admin/logs/recent")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_logs_recent_allows_ipv6_loopback():
    r = _admin_client("::1").get("/api/admin/logs/recent")
    assert r.status_code == 200


def test_logs_stream_rejects_non_loopback():
    # A 403 is returned BEFORE any streaming starts, so this does not hang.
    r = _admin_client("10.0.0.9").get("/api/admin/logs/stream")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


def test_client_is_loopback_helper():
    assert admin._client_is_loopback("127.0.0.1") is True
    assert admin._client_is_loopback("127.5.5.5") is True
    assert admin._client_is_loopback("::1") is True
    assert admin._client_is_loopback("localhost") is True
    assert admin._client_is_loopback("10.0.0.1") is False
    assert admin._client_is_loopback("203.0.113.1") is False
    assert admin._client_is_loopback(None) is False
    assert admin._client_is_loopback("not-an-ip") is False


def test_require_loopback_none_request_is_allowed():
    # A direct unit-test call (no ASGI scope) is treated as local.
    assert admin._require_loopback(None) is None


# --------------------------------------------------------------------------
# 5. REAL engine generation log path never leaks generated OUTPUT (finding 1).
#    Drives the actual MLXEngine._generate_locked token loop with a fake
#    mlx-lm stream (NOT a stub complete()) so the four generation log sites
#    (cancellation tail INFO, per-token DEBUG, periodic INFO every 50 tokens,
#    final DEBUG preview) actually fire — then asserts none echoes the text.
# --------------------------------------------------------------------------

def _hit_engine():
    """A cache-HIT engine whose _generate_locked drives end-to-end on the
    mlx-lm path (no real model). The HIT session skips prefill so the loop
    consumes the fake stream directly."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    return eng, messages


def _fake_stream_emitting(secret: str, n: int):
    """An lm_stream_generate stand-in that emits ``secret`` as the detokenized
    text of every one of ``n`` generated tokens (plain mlx-lm contract)."""

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            for i in range(n):
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset += 1
                yield SimpleNamespace(
                    text=secret, token=1000 + i,
                    prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    return fake_lm_stream


# Finding 1: token IDS are a REVERSIBLE encoding of the generated output for
# anyone holding the tokenizer. This id scheme maps each secret CHARACTER to a
# distinctive token id, so DECODING the emitted id stream reconstructs the
# secret verbatim — logging any generated id would leak it exactly like the
# text. The base offset is far from the incidental numbers other log sites emit
# (token counts, tps, char lengths), so the id-leak assertion is unambiguous.
_ID_BASE = 700000


def _fake_tokenizer_decode(ids) -> str:
    """Inverse of the id scheme — proves the ids genuinely reconstruct the
    secret, so scrubbing the id stream is what prevents the leak."""
    return "".join(chr(i - _ID_BASE) for i in ids)


def _fake_stream_secret_ids(text: str):
    """Emit ``text`` CHAR-BY-CHAR with per-char token IDS that decode back to the
    secret (via ``_fake_tokenizer_decode``). One frame per character."""

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            for ch in text:
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset += 1
                yield SimpleNamespace(
                    text=ch, token=_ID_BASE + ord(ch),
                    prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    return fake_lm_stream


def _drive(eng, messages, *, max_tokens, cancel_event=None, on_chunk=None):
    chunks = []
    for ch in eng._generate_locked(
        messages, max_tokens=max_tokens, temperature=0.0, session_id="s",
        tools=None, cancel_event=cancel_event, thinking=False,
        thinking_budget=0, top_p=1.0, min_p=0.0, top_k=0,
        repetition_penalty=1.0, response_format=None,
    ):
        chunks.append(ch)
        if on_chunk is not None:
            on_chunk(chunks)
    return chunks


def test_real_engine_generation_log_no_text_leak(monkeypatch, caplog):
    """>50 tokens so the periodic INFO site + per-token DEBUG + final DEBUG
    preview all fire. Engine logger at DEBUG. Finding 1: NEITHER the generated
    TEXT nor the generated TOKEN IDS (a reversible encoding of that text) may
    appear in ANY captured record or the admin LogBuffer.

    The stream emits the secret CHAR-BY-CHAR with token ids that DECODE back to
    the secret, so a per-token ``id={token}`` log (the pre-fix behavior) would
    leak the reconstructable id sequence — the assertions below detect exactly
    that."""
    # The stream is the secret repeated to exceed 50 tokens (periodic INFO fires)
    # while still decoding to the secret.
    stream = SECRET * 2
    # Self-check: the id scheme genuinely reconstructs the secret.
    assert _fake_tokenizer_decode([_ID_BASE + ord(c) for c in stream]) == stream

    eng, messages = _hit_engine()
    n = len(stream)
    monkeypatch.setattr(
        mlx_engine_module, "lm_stream_generate", _fake_stream_secret_ids(stream)
    )

    # Also feed the admin LogBuffer (the admin SSE source).
    buffer = admin.log_buffer
    buffer.buffer.clear()
    root = logging.getLogger()
    root.addHandler(buffer)
    buffer.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(
            logging.DEBUG, logger="mlx_soloheaven.engine.mlx_engine"
        ):
            frames = _drive(eng, messages, max_tokens=n)
    finally:
        root.removeHandler(buffer)

    assert len(frames) >= n  # the loop really ran the token stream
    # The scrubbed sites actually fired (real path exercised, not a stub).
    all_msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "[Token]" in all_msgs           # per-token DEBUG site
    assert "generated_chars" in all_msgs   # periodic INFO + final DEBUG preview
    # The set of DISTINCT token ids that reconstruct the secret. If any appears
    # in a log, an attacker with the tokenizer could reconstruct the output.
    secret_ids = {str(_ID_BASE + ord(c)) for c in SECRET}

    # Per-CHAR text-leak tripwire (nit 1): the ``SECRET not in msg`` check below
    # only catches a WHOLE-secret leak in ONE record. The stream is emitted
    # CHARACTER-BY-CHARACTER (one token == one secret char), so the PER-TOKEN log
    # site is where a future single-char text leak would live — e.g. adding
    # ``text=S`` to one record. A whole-secret substring check can never see that.
    # So assert every per-token (``[Token]``) record matches its EXACT safe,
    # redacted template (anchored at end): any added/leaked character — even one —
    # breaks the match. This targets the specific generated-content site and is
    # robust against incidental operational letters (unlike a raw char scan, it
    # is also immune to the LogBuffer's timestamp/level formatting prefix, which
    # legitimately carries letters like the 'E'/'U' in "DEBUG"). The anchored
    # tail works for BOTH the raw ``getMessage()`` and the formatted buffer line
    # (whose ``%(message)s`` sits at the end).
    token_safe_re = re.compile(
        r"\[Token\] session=\S+ \| n=\d+ chars=\d+ \(id\+text redacted\)$"
    )
    # Belt-and-suspenders for a MULTI-char leak at any OTHER generated-content
    # site (e.g. the periodic ``generated_chars`` line): no record may contain a
    # 5+ char contiguous substring of the secret (short enough to be reliably
    # distinctive, long enough to skip incidental operational tokens).
    _SUBSTR_LEN = 5
    secret_substrings = {
        SECRET[i:i + _SUBSTR_LEN] for i in range(len(SECRET) - _SUBSTR_LEN + 1)
    }
    assert secret_substrings, "secret too short to derive distinctive substrings"

    def _assert_no_generated_leak(msg: str, where: str) -> None:
        # Whole-secret text.
        assert SECRET not in msg, (
            f"generated text leaked into {where}: {msg!r}"
        )
        # Any multi-char slice of the secret (partial/chunked text leak).
        leaked_sub = sorted(s for s in secret_substrings if s in msg)
        assert not leaked_sub, (
            f"a generated-text substring leaked into {where}: "
            f"{leaked_sub} in {msg!r}"
        )
        # Per-char leak at the per-token site: the record MUST still match the
        # exact redacted template (a single leaked char breaks it).
        if "[Token]" in msg:
            assert token_safe_re.search(msg), (
                f"a per-token record deviated from the safe redacted template "
                f"(possible per-character text/id leak) in {where}: {msg!r}"
            )
        # Reconstructable token id.
        leaked = [sid for sid in secret_ids if sid in msg]
        assert not leaked, (
            f"generated token id(s) leaked (reconstructable output) into "
            f"{where}: {leaked} in {msg!r}"
        )

    # No generated OUTPUT — neither text (whole, chunked, OR per-char) NOR
    # reconstructable id — in any engine log record...
    for record in caplog.records:
        _assert_no_generated_leak(record.getMessage(), "an engine log")
    # ...nor in the admin LogBuffer (the admin SSE source).
    for entry in list(buffer.buffer):
        _assert_no_generated_leak(entry["message"], "the admin LogBuffer")


def test_real_engine_cancellation_log_no_text_leak(monkeypatch, caplog):
    """The cancellation tail site is an INFO line (leaks even --verbose off).
    Cancel after a few tokens and assert the CANCELLED line carries only
    metadata, no generated text."""
    eng, messages = _hit_engine()
    cancel = threading.Event()
    monkeypatch.setattr(
        mlx_engine_module, "lm_stream_generate", _fake_stream_emitting(SECRET, 30)
    )

    def _cancel_after_three(chunks):
        if len([c for c in chunks if getattr(c, "token", None)]) == 3:
            cancel.set()

    with caplog.at_level(logging.INFO, logger="mlx_soloheaven.engine.mlx_engine"):
        _drive(eng, messages, max_tokens=30, cancel_event=cancel,
               on_chunk=_cancel_after_three)

    all_msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "CANCELLED" in all_msgs  # the cancellation INFO site fired
    for record in caplog.records:
        assert SECRET not in record.getMessage(), (
            f"generated text leaked into the cancellation log: "
            f"{record.getMessage()!r}"
        )


# --------------------------------------------------------------------------
# 6. 422 loc sanitization: an attacker-controlled dict KEY in loc is redacted
#    (finding 2). MessageToolCall.function is dict[str, str] (schemas.py), so a
#    body {"function":{"<secret-key>":<bad>}} puts the key into pydantic's loc.
# --------------------------------------------------------------------------

LOC_SECRET_KEY = "SUPERSECRETLOCKEY_hunter2"


def _chat_schema_app() -> FastAPI:
    from mlx_soloheaven.server import register_validation_error_handler

    app = FastAPI()
    register_validation_error_handler(app)

    @app.post("/chat")
    async def chat_ep(body: ChatCompletionRequest):
        return {}

    return app


def test_422_loc_redacts_user_dict_key(caplog):
    client = TestClient(_chat_schema_app(), raise_server_exceptions=False)
    # function is dict[str, str]; a dict VALUE fails validation and pydantic
    # puts the user KEY into loc.
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {LOC_SECRET_KEY: {"nested": 1}}}
                ],
            }
        ],
    }
    with caplog.at_level(logging.ERROR, logger="soloheaven"):
        r = client.post("/chat", json=body)
    assert r.status_code == 422
    for record in caplog.records:
        assert LOC_SECRET_KEY not in record.getMessage(), (
            f"422 log leaked a user-controlled loc key: {record.getMessage()!r}"
        )
    logs = [r.getMessage() for r in caplog.records if "[422]" in r.getMessage()]
    assert logs, "expected a [422] log line"
    joined = " ".join(logs)
    # The redaction marker is present and the diagnostic path (which field)
    # survived: known schema field names are kept, only the user key is redacted.
    assert "<redacted>" in joined
    assert "function" in joined


def test_sanitize_loc_keeps_indices_and_known_fields():
    from mlx_soloheaven.server import _sanitize_loc

    loc = ("body", "messages", 0, "tool_calls", 0, "function", LOC_SECRET_KEY)
    out = _sanitize_loc(loc)
    assert out == [
        "body", "messages", 0, "tool_calls", 0, "function", "<redacted>",
    ]
    # A bare user key at the root is redacted; ints (indices) survive.
    assert _sanitize_loc(("body", 3, "definitely_not_a_field")) == [
        "body", 3, "<redacted>",
    ]


def test_sanitize_loc_normalizes_non_iterables_without_truthiness(monkeypatch):
    """Nit (round 3): a bare non-iterable loc is normalized by TYPE, never by
    truthiness — so a legitimate index-0 array position survives (the old
    ``if not loc: return []`` dropped it). None / unexpected objects are wrapped
    and safely redacted, and the sanitizer still never raises."""
    from mlx_soloheaven.server import _sanitize_loc

    # Index 0 (falsy but VALID) is preserved — the core of this nit.
    assert _sanitize_loc(0) == [0]
    assert _sanitize_loc(5) == [5]
    # None / an unexpected object -> wrapped + safely redacted (never raises).
    assert _sanitize_loc(None) == ["<redacted>"]
    assert _sanitize_loc(object()) == ["<redacted>"]
    # An empty list/tuple still yields [] (the loop produces nothing).
    assert _sanitize_loc(()) == []
    assert _sanitize_loc([]) == []
    # A bare string is treated as ONE component (not iterated per-character): a
    # known field-name survives, an unknown one is redacted.
    assert _sanitize_loc("body") == ["body"]
    assert _sanitize_loc("supersecret_user_key") == ["<redacted>"]
    # A 0 embedded in a normal loc tuple is likewise kept.
    assert _sanitize_loc(("body", 0)) == ["body", 0]


def test_known_loc_field_names_covers_every_api_model():
    """Nit (round 3): the loc allowlist is discovered GENERICALLY across the whole
    api package (pkgutil), so EVERY request/response BaseModel's field names +
    aliases — in ANY api module, including future ones — are covered and never
    over-redacted. Independently re-derive the field set and assert coverage, so
    a new api module drifting out of coverage is caught by CI."""
    import importlib
    import pkgutil

    from pydantic import BaseModel

    import mlx_soloheaven.api as api_pkg
    from mlx_soloheaven.server import _known_loc_field_names

    allow = _known_loc_field_names()
    expected: set[str] = set()
    for info in pkgutil.iter_modules(api_pkg.__path__, api_pkg.__name__ + "."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                for fname, field in obj.model_fields.items():
                    expected.add(fname)
                    alias = getattr(field, "alias", None)
                    if alias:
                        expected.add(alias)
    assert expected, "expected at least one api BaseModel field to be discovered"
    missing = expected - allow
    assert not missing, f"api model fields missing from the loc allowlist: {missing}"
