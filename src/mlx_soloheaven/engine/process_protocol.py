"""IPC frame constructors / parsers for the process-mode engine (Stage 1).

The PARENT (FastAPI) and CHILD (generation) processes exchange plain dicts
over ``multiprocessing`` Pipes. This module is the single source of truth for
the frame shapes (codex spec) so both sides stay in lock-step. It imports NO
mlx / pydantic-heavy deps so it is cheap to import in either process.

Channels (each a separate Pipe):
  cmd   PARENT -> CHILD : generate commands
  resp  CHILD  -> PARENT: ready / batch / final / done / error
  ctrl  PARENT -> CHILD : cancel (out-of-band, read by a daemon thread)

Wire frames
-----------
generate (cmd):
    {"v":1, "id":<str>, "op":"generate", "payload":{
        "messages":[...],
        "params":{session_id, max_tokens, temperature, top_p, min_p, top_k,
                  repetition_penalty, tools, thinking, thinking_budget,
                  response_format}}}
generate_scalar (cmd):
    Same shape as ``generate`` but op == "generate_scalar". Drives
    ``engine.generate_stream_async`` (scalar, one GenerationResult per frame —
    no coalescing) instead of the batched path. Used by the compaction summary
    streamer which consumes per-result.
rpc (cmd):  generic synchronous engine method call (Stage 2)
    {"v":1, "id":<str>, "op":"rpc", "method":<str>, "args":[...],
     "kwargs":{...}}  -- args/kwargs MUST be pickle/JSON-serializable.
cancel (ctrl):
    {"op":"cancel", "id":<str>}

ready (resp):
    {"type":"ready", "model_id", "model_family", "enable_thinking",
     "think_end_token", "cfg":<dict>}
rpc_result (resp):
    {"id":<str>, "type":"rpc_result", "result":<json/picklable>}
batch (resp):
    {"id":<str>, "type":"batch", "items":[GenerationResultDict, ...]}
final (resp):
    {"id":<str>, "type":"final", "result":<GenerationResultDict|None>}
done (resp):
    {"id":<str>, "type":"done"}
error (resp):
    {"id":<str>, "type":"error", "error":<str>, "traceback":<str>}
"""

PROTOCOL_VERSION = 1


# --- response_format (de)serialization ----------------------------------
# The engine expects a response_format object exposing ``.type`` and, for
# json_schema, ``.json_schema.schema_`` (see mlx_engine.py ~2064). The parent
# holds a pydantic model; we serialize via model_dump(by_alias=True,
# exclude_none=True) and rehydrate child-side into a light namespace.

def serialize_response_format(rf) -> dict | None:
    """Serialize a pydantic ResponseFormat (or None) to a plain dict."""
    if rf is None:
        return None
    # pydantic v2: by_alias so json_schema.schema_ -> "schema"
    if hasattr(rf, "model_dump"):
        return rf.model_dump(by_alias=True, exclude_none=True)
    if isinstance(rf, dict):
        return rf
    return None


class _JsonSchemaView:
    """Child-side view exposing ``.name`` and ``.schema_`` (alias 'schema')."""

    def __init__(self, d: dict):
        self.name = d.get("name")
        # by_alias dump writes "schema"; engine reads ``.schema_``.
        self.schema_ = d.get("schema", d.get("schema_"))


class _ResponseFormatView:
    """Child-side view exposing ``.type`` and ``.json_schema``."""

    def __init__(self, d: dict):
        self.type = d.get("type")
        js = d.get("json_schema")
        self.json_schema = _JsonSchemaView(js) if isinstance(js, dict) else None


def deserialize_response_format(d: dict | None):
    """Rehydrate a serialized response_format dict into an engine-compatible
    object (duck-typed: ``.type`` + ``.json_schema.schema_``)."""
    if not d:
        return None
    return _ResponseFormatView(d)


# --- command frames (PARENT -> CHILD) ------------------------------------

def make_generate(request_id: str, messages: list, params: dict) -> dict:
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "op": "generate",
        "payload": {"messages": messages, "params": params},
    }


def make_generate_scalar(request_id: str, messages: list, params: dict) -> dict:
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "op": "generate_scalar",
        "payload": {"messages": messages, "params": params},
    }


def make_rpc(request_id: str, method: str, args: list, kwargs: dict) -> dict:
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "op": "rpc",
        "method": method,
        "args": list(args),
        "kwargs": dict(kwargs),
    }


def make_cancel(request_id: str) -> dict:
    return {"op": "cancel", "id": request_id}


# --- response frames (CHILD -> PARENT) -----------------------------------

def make_ready(
    model_id, model_family, enable_thinking, think_end_token, cfg=None
) -> dict:
    return {
        "type": "ready",
        "model_id": model_id,
        "model_family": model_family,
        "enable_thinking": enable_thinking,
        "think_end_token": think_end_token,
        "cfg": cfg or {},
    }


def make_rpc_result(request_id: str, result) -> dict:
    return {"id": request_id, "type": "rpc_result", "result": result}


def make_batch(request_id: str, items: list[dict]) -> dict:
    return {"id": request_id, "type": "batch", "items": items}


def make_final(request_id: str, result: dict | None) -> dict:
    return {"id": request_id, "type": "final", "result": result}


def make_done(request_id: str) -> dict:
    return {"id": request_id, "type": "done"}


def make_error(request_id: str, error: str, tb: str) -> dict:
    return {"id": request_id, "type": "error", "error": error, "traceback": tb}
