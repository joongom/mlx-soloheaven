"""Codex round 5, finding 1 — the MANDATORY post-stream commit vs LONG-pool
saturation.

Streaming generation bypasses run_long but HOLDS the engine lock, so 20
concurrent non-streaming completions / mutating RPCs fill run_long's
admission (4 workers blocked on the lock + 16 queued). When the stream
finishes, BOTH APIs used to submit ``update_session_messages`` through the
saturated pool BEFORE their terminal event:

- api/chat.py: EngineBusyError escaped mid-stream (tokens already sent — no
  503 possible), the terminal "done" event was never emitted;
- api/openai_compat.py: same, the final chunk / [DONE] never emitted;
- and because the commit is what marked the session dirty, the freshly
  extended KV cache was never marked and could never be flushed (fixed
  engine-side by finding 1a: the generation marks dirty at install).

The commit now rides ``run_critical`` (reserved worker, no admission limit)
and is wrapped log-only at both call sites, so the terminal events are
UNCONDITIONAL.

HERMETIC — stub engines + stubbed chat db, real executors module (reset
around every test, conventions from tests/test_executor_admission_shutdown).
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncGenerator, Optional

import pytest

from mlx_soloheaven import executors
from mlx_soloheaven.engine.types import EngineBusyError


@pytest.fixture(autouse=True)
def fresh_executors():
    executors._reset_for_tests()
    yield
    executors._reset_for_tests()


class _StubDB:
    """The async db surface chat._stream_chat touches after the stream."""

    async def add_message(self, *a, **k):
        pass

    async def get_session_total_tokens(self, *a, **k):
        return 0

    async def update_session_tokens(self, *a, **k):
        pass

    async def update_session(self, *a, **k):
        pass

    async def get_session(self, *a, **k):
        return None


class _Result:
    """GenerationResult stand-in with the fields both stream readers touch."""

    def __init__(self, text="", token=0, finish_reason=None):
        self.text = text
        self.token = token
        self.status = None
        self.finish_reason = finish_reason
        self.prompt_tokens = 3
        self.completion_tokens = 2
        self.prompt_tps = 10.0
        self.generation_tps = 20.0
        self.cache_info = None
        # Finding 4: a CONTENT frame (no finish_reason) is a real token; a
        # terminal frame is not (never a keepalive in this stub).
        self.token_produced = finish_reason is None


class _StreamEngine:
    """Stub engine replaying a tiny stream, recording the post-stream
    commit. ``commit_error`` makes update_session_messages raise (the
    log-only contract must still emit the terminal events)."""

    def __init__(self, commit_error: Optional[Exception] = None):
        self.model_family = "chatml"
        self.commit_error = commit_error
        self.commits: list = []

        class _Cfg:
            enable_thinking = False

        self.cfg = _Cfg()

    async def generate_stream_batches_async(self, *a, **k) -> AsyncGenerator:
        yield [_Result(text="hello", token=11)]
        yield [_Result(text=" world", token=12)]
        yield [_Result(finish_reason="stop")]

    def update_session_messages(self, session_id, messages):
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append((session_id, messages))


def _saturate_long_pool(release: threading.Event) -> list:
    """codex's interleaving: fill run_long's admission completely (workers
    blocked + backlog full) while 'the stream' runs."""

    def _blocked():
        release.wait(timeout=30.0)
        return "done"

    return [
        asyncio.ensure_future(executors.run_long(_blocked))
        for _ in range(executors.LONG_ADMISSION_LIMIT)
    ]


async def _drain_chat_stream(eng) -> list[dict]:
    from mlx_soloheaven.api import chat as chat_mod

    frames = []
    async for line in chat_mod._stream_chat(
        "s1", [{"role": "user", "content": "hi"}], eng
    ):
        line = line.strip()
        if line.startswith("data: "):
            frames.append(json.loads(line[len("data: "):]))
    return frames


async def _drain_openai_stream(eng) -> list[str]:
    from mlx_soloheaven.api.openai_compat import _stream_completion
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        model="stub",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=False,
        user="sess-1",  # session-bound: triggers the post-stream commit
    )
    lines = []
    async for line in _stream_completion(request, eng):
        line = line.strip()
        if line:
            lines.append(line)
    return lines


# --- chat.py -----------------------------------------------------------------


def test_chat_stream_commit_survives_saturated_long_pool(monkeypatch):
    """codex round-5 interleaving, web-chat arm: the long pool is fully
    saturated when the stream finishes. The post-stream commit MUST still
    run (critical lane) and the terminal 'done' event MUST be emitted — no
    EngineBusyError escapes the stream generator."""
    from mlx_soloheaven.api import chat as chat_mod

    monkeypatch.setattr(chat_mod, "db", _StubDB())
    eng = _StreamEngine()
    release = threading.Event()

    async def _drive():
        blockers = _saturate_long_pool(release)
        await asyncio.sleep(0.1)  # workers occupied + backlog full
        with pytest.raises(EngineBusyError, match="saturated"):
            await executors.run_long(lambda: 1)  # saturation is real
        frames = await _drain_chat_stream(eng)
        release.set()
        await asyncio.gather(*blockers)
        return frames

    frames = asyncio.run(_drive())
    done = [f for f in frames if f.get("type") == "done"]
    assert len(done) == 1, f"terminal done event missing: {frames}"
    assert done[0]["content"] == "hello world"
    # The commit ran despite the saturated long pool.
    assert eng.commits and eng.commits[0][0] == "s1"


def test_chat_stream_commit_failure_still_emits_done(monkeypatch):
    """Even if the commit path ERRORS, the terminal done event goes out and
    the error is log-only (the engine-side dirty-mark — finding 1a — makes
    that safe)."""
    from mlx_soloheaven.api import chat as chat_mod

    monkeypatch.setattr(chat_mod, "db", _StubDB())
    eng = _StreamEngine(commit_error=EngineBusyError("pool saturated"))

    frames = asyncio.run(_drain_chat_stream(eng))
    done = [f for f in frames if f.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["content"] == "hello world"
    assert not [f for f in frames if f.get("type") == "error"]


# --- openai_compat.py ---------------------------------------------------------


def test_openai_stream_commit_survives_saturated_long_pool():
    """codex round-5 interleaving, OpenAI arm: saturated long pool at
    stream end — the commit still runs, and the final chunk + [DONE] are
    emitted."""
    eng = _StreamEngine()
    release = threading.Event()

    async def _drive():
        blockers = _saturate_long_pool(release)
        await asyncio.sleep(0.1)
        with pytest.raises(EngineBusyError, match="saturated"):
            await executors.run_long(lambda: 1)
        lines = await _drain_openai_stream(eng)
        release.set()
        await asyncio.gather(*blockers)
        return lines

    lines = asyncio.run(_drive())
    assert lines[-1] == "data: [DONE]"
    final = json.loads(lines[-2][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "stop"
    assert eng.commits and eng.commits[0][0] == "sess-1"


def test_openai_stream_commit_failure_still_emits_done():
    """Commit-path error is log-only: the final chunk + [DONE] must still
    be emitted."""
    eng = _StreamEngine(commit_error=EngineBusyError("pool saturated"))

    lines = asyncio.run(_drain_openai_stream(eng))
    assert lines[-1] == "data: [DONE]"
    final = json.loads(lines[-2][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "stop"
