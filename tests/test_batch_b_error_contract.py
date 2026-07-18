"""Batch B — error-contract normalization + 429/503 status policy + two
request-parsing fixes (bultagi.com integration).

Covers:
- the canonical {error:{message,type,code}} envelope builder (api/errors.py);
- the EngineBusyError reason discriminator and the app-level handler mapping
  (queue_full -> 429, engine_not_ready -> 503, EngineRestartingError -> 503),
  each with a Retry-After header and NO str(exc) leak in the body;
- the RequestValidationError 422 handler: canonical envelope, no body echo;
- chat_template_kwargs.enable_thinking (nested) driving thinking on/off with
  the documented precedence (top-level `thinking` wins);
- response_format: unsupported type -> 400, tools+response_format -> 400,
  json_object/json_schema still accepted.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import AsyncGenerator, Optional

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from mlx_soloheaven.api import errors
from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    ResponseFormat,
    ToolDef,
)
from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.types import EngineBusyError


# ===========================================================================
# 1. Canonical envelope builder (api/errors.py)
# ===========================================================================


def test_error_dict_shape():
    d = errors.error_dict("m", "t", "c")
    assert d == {"error": {"message": "m", "type": "t", "code": "c"}}


def test_error_response_sets_retry_after_header():
    resp = errors.error_response(429, "m", "t", "c", retry_after="7")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "7"
    assert json.loads(resp.body) == {"error": {"message": "m", "type": "t", "code": "c"}}


def test_error_response_no_retry_after_by_default():
    resp = errors.error_response(400, "m", "t", "c")
    assert "Retry-After" not in resp.headers


def test_queue_full_response_shape():
    resp = errors.queue_full_response()
    assert resp.status_code == 429
    body = json.loads(resp.body)
    assert body["error"]["type"] == "queue_full"
    assert body["error"]["code"] == "queue_full"
    assert resp.headers["Retry-After"] == errors.RETRY_AFTER_QUEUE_FULL


def test_engine_not_ready_response_shape():
    resp = errors.engine_not_ready_response()
    assert resp.status_code == 503
    body = json.loads(resp.body)
    assert body["error"]["type"] == "engine_not_ready"
    assert body["error"]["code"] == "engine_not_ready"
    assert resp.headers["Retry-After"] == errors.RETRY_AFTER_ENGINE_NOT_READY


def test_invalid_request_response_has_code():
    resp = errors.invalid_request_response("bad")
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["message"] == "bad"


# ===========================================================================
# 2. EngineBusyError discriminator + app-level handler mapping
# ===========================================================================


def test_engine_busy_error_default_reason_is_engine_not_ready():
    assert EngineBusyError("x").reason == EngineBusyError.REASON_ENGINE_NOT_READY


def test_engine_busy_error_explicit_reason():
    e = EngineBusyError("x", reason=EngineBusyError.REASON_QUEUE_FULL)
    assert e.reason == "queue_full"
    assert str(e) == "x"  # message still carried positionally


def _handler_app() -> FastAPI:
    from mlx_soloheaven.server import register_engine_error_handlers

    app = FastAPI()
    register_engine_error_handlers(app)

    @app.get("/queue-full")
    async def _qf():
        raise EngineBusyError(
            "pool saturated", reason=EngineBusyError.REASON_QUEUE_FULL
        )

    @app.get("/not-ready")
    async def _nr():
        raise EngineBusyError(
            "shutting down", reason=EngineBusyError.REASON_ENGINE_NOT_READY
        )

    @app.get("/busy-default")
    async def _bd():
        raise EngineBusyError("no reason given")

    @app.get("/restarting")
    async def _rs():
        raise EngineRestartingError("child worker died (secret pipe EOF)")

    return app


def test_queue_full_busy_maps_to_429():
    client = TestClient(_handler_app(), raise_server_exceptions=False)
    r = client.get("/queue-full")
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "queue_full"
    assert body["error"]["code"] == "queue_full"
    assert r.headers.get("Retry-After") is not None
    # No str(exc) leak.
    assert "pool saturated" not in json.dumps(body)
    assert "detail" not in body["error"]


def test_engine_not_ready_busy_maps_to_503():
    client = TestClient(_handler_app(), raise_server_exceptions=False)
    r = client.get("/not-ready")
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["type"] == "engine_not_ready"
    assert body["error"]["code"] == "engine_not_ready"
    assert r.headers.get("Retry-After") is not None
    assert "shutting down" not in json.dumps(body)


def test_default_reason_busy_maps_to_503():
    """A raise site that omits reason keeps the pre-Batch-B 503 behavior."""
    client = TestClient(_handler_app(), raise_server_exceptions=False)
    r = client.get("/busy-default")
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_not_ready"


def test_restarting_maps_to_503_no_leak():
    client = TestClient(_handler_app(), raise_server_exceptions=False)
    r = client.get("/restarting")
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["type"] == "engine_not_ready"
    assert body["error"]["code"] == "engine_not_ready"
    assert r.headers.get("Retry-After") is not None
    # The internal exception text must NOT reach the client.
    assert "secret pipe EOF" not in json.dumps(body)
    assert "detail" not in body["error"]


# ===========================================================================
# 3. RequestValidationError 422 -> canonical envelope, no body echo
# ===========================================================================


def _validation_app() -> FastAPI:
    from mlx_soloheaven.server import register_validation_error_handler

    app = FastAPI()
    register_validation_error_handler(app)

    class _Body(BaseModel):
        n: int

    @app.post("/needs-int")
    async def _needs_int(body: _Body):  # pragma: no cover - never reached on 422
        return {"n": body.n}

    return app


def test_validation_error_canonical_envelope():
    client = TestClient(_validation_app(), raise_server_exceptions=False)
    r = client.post("/needs-int", json={"n": "not-an-int"})
    assert r.status_code == 422
    body = r.json()
    # Canonical shape (NOT FastAPI's default {"detail": [...]}).
    assert set(body["error"].keys()) == {"message", "type", "code"}
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
    assert "detail" not in body


def test_validation_error_does_not_echo_request_body():
    client = TestClient(_validation_app(), raise_server_exceptions=False)
    secret = "SUPERSECRETVALUE12345"
    r = client.post("/needs-int", json={"n": secret})
    assert r.status_code == 422
    # Neither the offending input value nor the field-level errors() leak.
    assert secret not in r.text


# ===========================================================================
# 4. chat_template_kwargs.enable_thinking (nested) support
# ===========================================================================


@dataclass
class _StubResult:
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None


class _StreamStubEngine:
    """Replays a canned token stream; records the generation call kwargs."""

    def __init__(self, tokens, *, model_family="chatml", enable_thinking=False):
        self.model_family = model_family
        self.model_id = f"test-{model_family}"
        self._tokens = tokens
        self.cfg = SimpleNamespace(enable_thinking=enable_thinking)
        self.captured_kwargs: dict = {}

    async def generate_stream_batches_async(self, *args, **kwargs) -> AsyncGenerator:
        self.captured_kwargs = kwargs
        for tok in self._tokens:
            yield [_StubResult(text=tok)]
        yield [_StubResult(
            text="", finish_reason="stop", prompt_tokens=5,
            completion_tokens=len(self._tokens),
        )]

    def update_session_messages(self, *args, **kwargs):
        pass


async def _collect_sse(engine, request) -> list[dict]:
    from mlx_soloheaven.api.openai_compat import _stream_completion

    events = []
    async for line in _stream_completion(request, engine):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def _joined_delta(events, key) -> str:
    return "".join(
        c["delta"].get(key) or ""
        for ev in events
        for c in ev.get("choices", [])
    )


def test_ctk_enable_thinking_false_suppresses_thinking_stream():
    """chat_template_kwargs.enable_thinking=false must drive thinking OFF
    exactly like the top-level flag: the engine is asked with thinking=False
    and no reasoning_content bleeds out."""
    engine = _StreamStubEngine(["hello"], enable_thinking=True)
    req = ChatCompletionRequest(
        model="m", stream=True,
        chat_template_kwargs={"enable_thinking": False},
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = asyncio.run(_collect_sse(engine, req))
    assert engine.captured_kwargs.get("thinking") is False
    assert _joined_delta(events, "reasoning_content") == ""
    assert _joined_delta(events, "content") == "hello"


def test_ctk_enable_thinking_true_routes_reasoning_stream():
    """chat_template_kwargs.enable_thinking=true routes the thought span to
    reasoning_content via the existing thinking machinery."""
    engine = _StreamStubEngine(
        ["reasoning...", "</think>", "answer"], enable_thinking=False,
    )
    req = ChatCompletionRequest(
        model="m", stream=True,
        chat_template_kwargs={"enable_thinking": True},
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = asyncio.run(_collect_sse(engine, req))
    assert engine.captured_kwargs.get("thinking") is True
    assert _joined_delta(events, "reasoning_content") == "reasoning..."
    assert _joined_delta(events, "content") == "answer"


def test_top_level_thinking_wins_over_ctk():
    """PRECEDENCE: an explicit top-level `thinking` wins when BOTH are set."""
    engine = _StreamStubEngine(["reasoning...", "</think>", "answer"])
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        chat_template_kwargs={"enable_thinking": False},
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = asyncio.run(_collect_sse(engine, req))
    assert engine.captured_kwargs.get("thinking") is True


class _SyncStubEngine:
    """Non-streaming engine stub: records complete() kwargs."""

    model_family = "chatml"
    model_id = "m"

    def __init__(self, enable_thinking=False):
        self.cfg = SimpleNamespace(enable_thinking=enable_thinking)
        self.captured: dict = {}

    def complete(self, messages, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content="ok", thinking=None, tool_calls=None,
            finish_reason="stop", prompt_tokens=1, completion_tokens=1,
            prompt_tps=0.0, generation_tps=0.0, cache_info=None,
        )

    def update_session_messages(self, *a, **k):
        pass


def test_ctk_enable_thinking_false_suppresses_thinking_sync():
    """Non-streaming path also honors the nested flag (thinking=False)."""
    from mlx_soloheaven.api.openai_compat import _sync_completion

    eng = _SyncStubEngine(enable_thinking=True)
    req = ChatCompletionRequest(
        model="m", stream=False,
        chat_template_kwargs={"enable_thinking": False},
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _sync_completion(req, eng)
    assert eng.captured.get("thinking") is False
    assert resp.choices[0].message.content == "ok"


def test_ctk_absent_falls_back_to_server_default():
    """No top-level flag, no chat_template_kwargs -> the engine default."""
    from mlx_soloheaven.api.openai_compat import _sync_completion

    eng = _SyncStubEngine(enable_thinking=True)
    req = ChatCompletionRequest(
        model="m", stream=False,
        messages=[ChatMessage(role="user", content="hi")],
    )
    _sync_completion(req, eng)
    assert eng.captured.get("thinking") is True


# ===========================================================================
# 5. response_format: unsupported -> 400, tools+response_format -> 400
# ===========================================================================


class _StubApiEngine:
    model_family = "chatml"
    model_id = "m"
    cfg = SimpleNamespace(enable_thinking=False)


def _run_chat(request):
    """Drive openai_compat.chat_completions against a stub engine registry.

    For the validation paths under test the engine is never invoked: an
    unsupported/mutually-exclusive request returns a 400 JSONResponse up
    front, and an accepted streaming request returns a StreamingResponse
    whose generator is not run here."""
    from mlx_soloheaven.api import openai_compat

    old_engines = openai_compat._engines
    old_default = openai_compat._default_engine
    try:
        eng = _StubApiEngine()
        openai_compat.set_engines({"m": eng}, eng)
        return asyncio.run(openai_compat.chat_completions(request))
    finally:
        openai_compat.set_engines(old_engines, old_default)


def _tool() -> ToolDef:
    return ToolDef(function=FunctionDef(name="f", description="d", parameters={}))


def test_response_format_unsupported_type_400():
    req = ChatCompletionRequest(
        model="m", stream=True,
        response_format=ResponseFormat(type="yaml"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _run_chat(req)
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
    assert "yaml" in body["error"]["message"]


@pytest.mark.parametrize("rf_type", ["json_object", "json_schema"])
def test_tools_plus_response_format_400(rf_type):
    rf = (
        ResponseFormat(type="json_object")
        if rf_type == "json_object"
        else ResponseFormat(
            type="json_schema",
            json_schema={"name": "x", "schema": {"type": "object"}},
        )
    )
    req = ChatCompletionRequest(
        model="m", stream=True, tools=[_tool()], response_format=rf,
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _run_chat(req)
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["code"] == "invalid_request"
    assert "mutually exclusive" in body["error"]["message"]


def test_tools_plus_text_response_format_allowed():
    """type='text' imposes no constraint, so it may combine with tools."""
    req = ChatCompletionRequest(
        model="m", stream=True, tools=[_tool()],
        response_format=ResponseFormat(type="text"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _run_chat(req)
    assert isinstance(resp, StreamingResponse)  # not rejected


def test_json_object_without_tools_still_accepted():
    """json_object (no tools) is NOT rejected by the up-front validation.

    Unlike json_schema, json_object performs no up-front schema compile, so it
    is accepted regardless of the optional ``outlines_core`` package. The
    json_schema acceptance path is exercised separately (with the compiler
    monkeypatched to succeed) by
    test_json_schema_valid_schema_accepted_without_outlines_core below."""
    req = ChatCompletionRequest(
        model="m", stream=True,
        response_format=ResponseFormat(type="json_object"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _run_chat(req)
    assert isinstance(resp, StreamingResponse)  # accepted (not a 400)


def test_json_schema_missing_schema_400():
    req = ChatCompletionRequest(
        model="m", stream=True,
        response_format=ResponseFormat(type="json_schema"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _run_chat(req)
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    assert json.loads(resp.body)["error"]["code"] == "invalid_request"


def _json_schema_request() -> ChatCompletionRequest:
    """A well-formed json_schema request (schema present, no tools)."""
    return ChatCompletionRequest(
        model="m", stream=True,
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={"name": "x", "schema": {"type": "object"}},
        ),
        messages=[ChatMessage(role="user", content="hi")],
    )


def test_json_schema_valid_schema_accepted_without_outlines_core(monkeypatch):
    """A VALID json_schema is ACCEPTED (request proceeds, NOT a 400) even
    without the optional ``outlines_core`` package installed.

    The up-front compile entry point is monkeypatched to SUCCEED, so the
    acceptance path (valid schema -> not rejected -> StreamingResponse) is
    exercised deterministically regardless of whether outlines_core is present
    in the hermetic env."""
    from mlx_soloheaven.api import openai_compat

    monkeypatch.setattr(
        openai_compat, "_compile_response_format_schema", lambda schema: None
    )
    resp = _run_chat(_json_schema_request())
    assert isinstance(resp, StreamingResponse)  # accepted (not a 400)


def test_json_schema_dependency_missing_returns_501_no_leak(monkeypatch):
    """DEPENDENCY / SETUP failure: the compiler raises ModuleNotFoundError for
    the optional outlines_core package. A valid client request must NOT get a
    400; it gets a 501 with the canonical envelope, and the internal import
    text (dependency name / "No module") must not leak into the body."""
    from mlx_soloheaven.api import openai_compat

    def _raise_missing_dep(schema):
        raise ModuleNotFoundError("No module named 'outlines_core'")

    monkeypatch.setattr(
        openai_compat, "_compile_response_format_schema", _raise_missing_dep
    )
    resp = _run_chat(_json_schema_request())
    assert isinstance(resp, JSONResponse)
    # NOT a 400 — the client request is valid; this is a server-side setup gap.
    assert resp.status_code == 501
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_supported"
    assert body["error"]["code"] == "structured_output_unavailable"
    # The internal dependency / import error text must NOT reach the client.
    raw = resp.body.decode()
    assert "outlines_core" not in raw
    assert "No module" not in raw


def test_json_schema_malformed_schema_returns_400_no_leak(monkeypatch):
    """MALFORMED CLIENT SCHEMA: the compiler (dependency present) rejects an
    invalid schema with a ValueError. That IS the client's fault -> 400 with a
    GENERIC message; the raw compiler exception text must not leak."""
    from mlx_soloheaven.api import openai_compat

    secret = "OUTLINES_INTERNAL_COMPILER_TRACE_zzz987"

    def _raise_bad_schema(schema):
        raise ValueError(secret)

    monkeypatch.setattr(
        openai_compat, "_compile_response_format_schema", _raise_bad_schema
    )
    resp = _run_chat(_json_schema_request())
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
    assert "Invalid JSON schema" in body["error"]["message"]
    # The raw compiler exception text must NOT leak into the response body.
    assert secret not in resp.body.decode()
