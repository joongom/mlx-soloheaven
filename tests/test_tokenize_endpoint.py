"""Tests for the Batch A additive endpoints: POST /tokenize and GET /ready.

Covers:
  * endpoint behavior via FastAPI TestClient with a stub engine (routing,
    validation envelopes, response shape, readiness policy);
  * the real ``MLXEngine.tokenize`` / ``is_ready`` / ``active_request_count``
    method logic against a deterministic fake tokenizer (no model load);
  * process-mode reachability: ``EngineProcessProxy`` forwards ``tokenize``
    through the generic RPC and answers readiness from local liveness state;
  * the ``_extract_max_context_length`` config helper.
"""

import threading

import pytest


# --------------------------------------------------------------------------
# Deterministic fake tokenizers (hermetic — no model load).
# --------------------------------------------------------------------------

class _FakeTokenizer:
    """Word-level tokenizer with convert_ids_to_tokens + round-trippable decode."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            i = len(self._vocab) + 100
            self._vocab[tok] = i
            self._rev[i] = tok
        return self._vocab[tok]

    def encode(self, text, add_special_tokens=False):
        ids = [self._id(w) for w in text.split(" ") if w != ""]
        if add_special_tokens:
            ids = [1, *ids, 2]  # BOS / EOS
        return ids

    def decode(self, ids):
        return " ".join(self._rev[i] for i in ids if i in self._rev)

    def convert_ids_to_tokens(self, ids):
        return [self._rev.get(i, f"<special-{i}>") for i in ids]


class _NoPiecesTokenizer:
    """Like _FakeTokenizer but WITHOUT convert_ids_to_tokens (decode fallback)."""

    def __init__(self):
        self._t = _FakeTokenizer()

    def encode(self, text, add_special_tokens=False):
        return self._t.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids):
        return self._t.decode(ids)


# --------------------------------------------------------------------------
# Endpoint stub engine + TestClient wiring.
# --------------------------------------------------------------------------

class _StubEngine:
    """Minimal engine exposing the tokenize/readiness surface the two
    endpoints use. Its ``tokenize`` mirrors the real method's response shape."""

    def __init__(self, *, ready=True, in_flight=0, model_id="stub-model",
                 context_length=4096, tokenizer=None):
        self._ready = ready
        self._in_flight = in_flight
        self.model_id = model_id
        self.max_context_length = context_length
        self.tokenizer = tokenizer or _FakeTokenizer()

    def is_ready(self):
        return self._ready

    def active_request_count(self):
        return self._in_flight

    def tokenize(self, content, add_special=False, with_pieces=False):
        ids = self.tokenizer.encode(content, add_special_tokens=add_special)
        tokens = [int(t) for t in ids]
        out = {"tokens": tokens, "count": len(tokens)}
        if with_pieces:
            pieces = self.tokenizer.convert_ids_to_tokens(tokens)
            out["pieces"] = [
                {"id": int(t), "piece": p} for t, p in zip(tokens, pieces)
            ]
        return out


def _client(engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import tokenize as tok_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    tok_mod.set_engines({"stub": engine} if engine is not None else {}, engine)
    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(tok_mod.router)
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------
# POST /tokenize — endpoint behavior.
# --------------------------------------------------------------------------

def test_tokenize_basic():
    client = _client(_StubEngine())
    r = client.post("/tokenize", json={"content": "hello world foo"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["tokens"]) == 3
    assert all(isinstance(t, int) for t in body["tokens"])
    assert "pieces" not in body  # with_pieces defaults false


def test_tokenize_add_special_differs():
    client = _client(_StubEngine())
    r_plain = client.post("/tokenize", json={"content": "hello world"})
    r_special = client.post(
        "/tokenize", json={"content": "hello world", "add_special": True}
    )
    assert r_plain.status_code == 200 and r_special.status_code == 200
    plain, special = r_plain.json(), r_special.json()
    assert special["tokens"] != plain["tokens"]
    assert special["count"] == plain["count"] + 2  # BOS + EOS


def test_tokenize_with_pieces_aligned():
    client = _client(_StubEngine())
    r = client.post(
        "/tokenize", json={"content": "hello world foo", "with_pieces": True}
    )
    assert r.status_code == 200
    body = r.json()
    assert "pieces" in body
    assert len(body["pieces"]) == body["count"] == len(body["tokens"])
    for tok_id, piece in zip(body["tokens"], body["pieces"]):
        assert piece["id"] == tok_id
        assert isinstance(piece["piece"], str)


def test_tokenize_empty_content_400():
    client = _client(_StubEngine())
    r = client.post("/tokenize", json={"content": ""})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "content" in err["message"]


def test_tokenize_missing_content_400():
    client = _client(_StubEngine())
    r = client.post("/tokenize", json={})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_tokenize_not_ready_503():
    client = _client(_StubEngine(ready=False))
    r = client.post("/tokenize", json={"content": "hello"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_not_ready"
    assert r.headers.get("Retry-After") is not None


def test_tokenize_engine_restarting_maps_to_503():
    """A dead process-mode child raises EngineRestartingError from the RPC
    path; the app-level handler maps it to 503 (the existing path)."""
    from mlx_soloheaven.engine.process_client import EngineRestartingError

    class _DeadEngine(_StubEngine):
        def tokenize(self, content, add_special=False, with_pieces=False):
            raise EngineRestartingError("child worker died")

    client = _client(_DeadEngine())
    r = client.post("/tokenize", json={"content": "hello"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_restarting"


# --------------------------------------------------------------------------
# GET /ready — endpoint behavior.
# --------------------------------------------------------------------------

def test_ready_200_fields():
    client = _client(_StubEngine(model_id="m1", context_length=8192))
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model"] == "m1"
    assert body["context_length"] == 8192
    assert body["queue_length"] == 0


def test_ready_not_ready_503():
    client = _client(_StubEngine(ready=False))
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_not_ready"
    assert r.headers.get("Retry-After") is not None


def test_ready_ignores_queue_saturation():
    """CRITICAL POLICY: model+tokenizer ready but busy (in-flight work) -> still
    200. Readiness must NOT fail on queue saturation."""
    client = _client(_StubEngine(ready=True, in_flight=5))
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["queue_length"] == 5  # reported, not a readiness failure


def test_ready_context_length_null_when_unknown():
    client = _client(_StubEngine(context_length=None))
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["context_length"] is None


# --------------------------------------------------------------------------
# Real MLXEngine.tokenize / is_ready / active_request_count method logic.
# --------------------------------------------------------------------------

def _bare_engine(tokenizer, *, model="__model__", max_ctx=4096):
    """A bare MLXEngine instance with just the attributes the read-only methods
    touch — no __init__, no model load."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.tokenizer = tokenizer
    eng._language_model = None if model is None else object()
    eng.model_id = "bare"
    eng.max_context_length = max_ctx
    eng._lock = threading.Lock()
    return eng


def test_engine_tokenize_shape_and_roundtrip():
    eng = _bare_engine(_FakeTokenizer())
    out = eng.tokenize("hello world foo")
    assert out["count"] == 3 == len(out["tokens"])
    # Round-trip: decoding the tokens recovers the input.
    assert eng.tokenizer.decode(out["tokens"]) == "hello world foo"


def test_engine_tokenize_add_special_differs():
    eng = _bare_engine(_FakeTokenizer())
    plain = eng.tokenize("hello world", add_special=False)
    special = eng.tokenize("hello world", add_special=True)
    assert plain["tokens"] != special["tokens"]
    assert special["count"] == plain["count"] + 2


def test_engine_tokenize_pieces_via_convert():
    eng = _bare_engine(_FakeTokenizer())
    out = eng.tokenize("hello world", with_pieces=True)
    assert [p["id"] for p in out["pieces"]] == out["tokens"]
    assert out["pieces"][0]["piece"] == "hello"


def test_engine_tokenize_pieces_decode_fallback():
    """Tokenizer without convert_ids_to_tokens -> per-token decode fallback."""
    eng = _bare_engine(_NoPiecesTokenizer())
    out = eng.tokenize("hello world", with_pieces=True)
    assert len(out["pieces"]) == out["count"]
    assert out["pieces"][0]["piece"] == "hello"
    assert [p["id"] for p in out["pieces"]] == out["tokens"]


def test_engine_tokenizer_not_loaded_raises():
    eng = _bare_engine(None)
    with pytest.raises(RuntimeError):
        eng.tokenize("hello")


def test_engine_is_ready():
    assert _bare_engine(_FakeTokenizer()).is_ready() is True
    assert _bare_engine(None).is_ready() is False  # no tokenizer
    assert _bare_engine(_FakeTokenizer(), model=None).is_ready() is False


def test_engine_active_request_count_tracks_lock():
    eng = _bare_engine(_FakeTokenizer())
    assert eng.active_request_count() == 0
    eng._lock.acquire()
    try:
        assert eng.active_request_count() == 1  # generation "in flight"
    finally:
        eng._lock.release()
    assert eng.active_request_count() == 0


# --------------------------------------------------------------------------
# Process-mode reachability: EngineProcessProxy.
# --------------------------------------------------------------------------

def _bare_proxy():
    from mlx_soloheaven.engine.process_client import EngineProcessProxy

    proxy = EngineProcessProxy.__new__(EngineProcessProxy)
    proxy._state_lock = threading.Lock()
    proxy._inflight_lock = threading.Lock()
    proxy._inflight = 0
    proxy._ever_ready = True
    proxy._dead = False
    proxy._permanently_dead = False
    proxy._closed = False
    proxy._respawning = False
    proxy.model_id = "proc-model"
    proxy.max_context_length = None
    return proxy


def test_proxy_tokenize_forwards_generic_rpc():
    proxy = _bare_proxy()
    captured = {}

    def _fake_rpc_read(method, *args, **kwargs):
        captured["method"] = method
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"tokens": [1, 2], "count": 2}

    proxy._rpc_read = _fake_rpc_read
    out = proxy.tokenize("hi there", add_special=True, with_pieces=True)
    assert out == {"tokens": [1, 2], "count": 2}
    assert captured["method"] == "tokenize"
    assert captured["args"] == ("hi there",)
    assert captured["kwargs"] == {"add_special": True, "with_pieces": True}


def test_proxy_is_ready_liveness_states():
    proxy = _bare_proxy()
    assert proxy.is_ready() is True

    proxy._dead = True
    assert proxy.is_ready() is False
    proxy._dead = False

    proxy._respawning = True
    assert proxy.is_ready() is False
    proxy._respawning = False

    proxy._permanently_dead = True
    assert proxy.is_ready() is False
    proxy._permanently_dead = False

    proxy._closed = True
    assert proxy.is_ready() is False
    proxy._closed = False

    proxy._ever_ready = False  # never reached first ready frame
    assert proxy.is_ready() is False


def test_proxy_active_request_count():
    proxy = _bare_proxy()
    assert proxy.active_request_count() == 0
    proxy._inflight = 3
    assert proxy.active_request_count() == 3


def test_proxy_apply_ready_sets_context_length():
    proxy = _bare_proxy()
    proxy._ready_event = threading.Event()
    proxy._spawn_gen = 0
    proxy.cfg = type("Cfg", (), {})()
    frame = {
        "model_id": "proc-model",
        "model_family": "chatml",
        "enable_thinking": True,
        "think_end_token": -1,
        "context_length": 32768,
        "cfg": {},
    }
    proxy._apply_ready(frame)
    assert proxy.max_context_length == 32768


# --------------------------------------------------------------------------
# _extract_max_context_length helper.
# --------------------------------------------------------------------------

def test_extract_max_context_length_variants():
    from mlx_soloheaven.engine.mlx_engine import _extract_max_context_length

    assert _extract_max_context_length({"max_position_embeddings": 4096}) == 4096
    # Nested under text_config (VLM configs).
    assert _extract_max_context_length(
        {"text_config": {"max_position_embeddings": 8192}}
    ) == 8192
    assert _extract_max_context_length(
        {"llm_config": {"max_position_embeddings": 2048}}
    ) == 2048
    # Absent / invalid / bool -> None.
    assert _extract_max_context_length({}) is None
    assert _extract_max_context_length({"max_position_embeddings": 0}) is None
    assert _extract_max_context_length({"max_position_embeddings": True}) is None
    assert _extract_max_context_length(None) is None
