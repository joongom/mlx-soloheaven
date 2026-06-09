"""Active-session LRU eviction tests (memory-blowup fix).

Root cause being fixed: active per-session KV caches live in
``engine._sessions`` and were only ever removed by an EXPLICIT delete_session.
The idle-flush SAVED dirty sessions to disk but did NOT del them, so their KV
caches stayed resident forever and the Mac OOM'd. ``memory_budget_gb`` only
bounded the SEPARATE LRU prefix-reuse pool, never the active sessions.

These tests build a BARE engine (no model / no MLX weights) with stub
SessionState entries of known KV sizes, set a small budget, trigger
``_evict_active_sessions_if_needed``, and assert:
  - the LRU idle session is flushed to disk AND removed from _sessions,
  - total resident KV drops under budget,
  - an in-flight (busy) session is NEVER evicted,
  - the single most-recently-used session is always kept resident,
  - an evicted-to-disk session is registered for transparent reload.

Disk save + MLX are mocked — no real model, weights, or GPU work.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from mlx_soloheaven.cache.manager import CacheManager
from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState


# --- Stub cache layer that the real _estimate_cache_size can size ----------

class _StubArray:
    """Minimal stand-in for an mx.array — only nbytes is read by sizing."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _StubCacheLayer:
    """A cache layer whose .state holds a single array of `nbytes`.

    CacheManager._estimate_cache_size walks `.state` and sums `.nbytes`, so
    this lets us build sessions of an EXACT known byte size with no MLX.
    """

    def __init__(self, nbytes: int):
        self.state = (_StubArray(nbytes),)


def _make_session(nbytes: int, last_used: float) -> SessionState:
    cache_state = SimpleNamespace(cache=[_StubCacheLayer(nbytes)], token_ids=[1, 2, 3])
    return SessionState(
        cache_state=cache_state,
        messages=[{"role": "user", "content": "hi"}],
        total_cache_tokens=3,
        last_used=last_used,
    )


def _make_engine(budget_gb: float) -> MLXEngine:
    """Bare MLXEngine shell with only the fields eviction touches."""
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.memory_budget_gb = budget_gb
    cfg.mlx_cache_limit_gb = 4.0
    eng.cfg = cfg
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._dirty_lock = threading.Lock()
    eng._busy_sessions = set()
    eng._busy_lock = threading.Lock()
    # Real CacheManager — its _estimate_cache_size / _memory_usage_gb are used.
    eng.cache_manager = CacheManager(
        memory_budget_gb=budget_gb, disk_budget_gb=100.0, cache_dir="/tmp/sh-test-cache"
    )
    return eng


_GB = 1_000_000_000


def test_evicts_lru_idle_session_under_budget(monkeypatch):
    """Over budget -> LRU idle session is saved to disk AND removed."""
    eng = _make_engine(budget_gb=2.5)

    saved: list[str] = []

    def _fake_save(sid, session):
        saved.append(sid)
        # Mimic the real save side effect: register for disk reload.
        if not hasattr(eng, "_disk_session_ids"):
            eng._disk_session_ids = set()
        eng._disk_session_ids.add(sid)
        return True

    monkeypatch.setattr(eng, "_save_session_to_disk", _fake_save)

    # 3 sessions x 1GB = 3GB resident, budget 2.5GB. last_used: A oldest.
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
        "C": _make_session(_GB, last_used=300.0),
    }
    assert eng._active_sessions_memory_gb() == pytest.approx(3.0)

    eng._evict_active_sessions_if_needed()

    # LRU "A" evicted: saved to disk, removed, registered for reload.
    assert saved == ["A"]
    assert "A" not in eng._sessions
    assert "A" in eng._disk_session_ids
    assert set(eng._sessions) == {"B", "C"}
    # Total now under budget.
    assert eng._active_sessions_memory_gb() == pytest.approx(2.0)
    assert eng._active_sessions_memory_gb() <= eng.cfg.memory_budget_gb


def test_busy_session_is_never_evicted(monkeypatch):
    """An in-flight (busy) session must NOT be evicted even if it's LRU."""
    eng = _make_engine(budget_gb=1.5)

    saved: list[str] = []
    monkeypatch.setattr(
        eng, "_save_session_to_disk",
        lambda sid, s: (saved.append(sid) or True),
    )

    # A (oldest) is in-flight; B and C idle. 3GB resident, budget 1.5GB.
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
        "C": _make_session(_GB, last_used=300.0),
    }
    eng._mark_session_busy("A")

    eng._evict_active_sessions_if_needed()

    # A protected by busy flag; LRU idle B evicted instead. C is MRU, kept.
    assert "A" in eng._sessions
    assert "A" not in saved
    assert "B" not in eng._sessions
    assert saved == ["B"]
    assert "C" in eng._sessions


def test_protect_and_mru_kept(monkeypatch):
    """protect_session_id and the single MRU session are never evicted."""
    eng = _make_engine(budget_gb=0.5)  # tiny budget: would evict everything
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)

    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),  # MRU
    }
    # protect A, MRU is B -> nothing evictable, but at least one always stays.
    eng._evict_active_sessions_if_needed(protect_session_id="A")

    # A protected, B is MRU: both kept (we never drop the last/just-used one).
    assert set(eng._sessions) == {"A", "B"}


def test_no_eviction_when_under_budget(monkeypatch):
    """Under budget -> no save, no removal."""
    eng = _make_engine(budget_gb=10.0)
    saved: list[str] = []
    monkeypatch.setattr(
        eng, "_save_session_to_disk",
        lambda sid, s: (saved.append(sid) or True),
    )
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._evict_active_sessions_if_needed()
    assert saved == []
    assert set(eng._sessions) == {"A", "B"}


def test_pool_counts_toward_budget(monkeypatch):
    """The separate LRU prefix-reuse pool counts toward the same budget."""
    eng = _make_engine(budget_gb=2.5)
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)

    # 2 sessions x 1GB = 2GB active + 1GB pool = 3GB > 2.5GB budget.
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng.cache_manager.memory_caches["pool1"] = SimpleNamespace(size_bytes=_GB)

    eng._evict_active_sessions_if_needed()

    # LRU A evicted to bring active+pool under budget; B (MRU) kept.
    assert "A" not in eng._sessions
    assert set(eng._sessions) == {"B"}


def test_evicts_even_if_save_fails(monkeypatch):
    """If disk save fails PERMANENTLY (returns False), still evict to reclaim
    memory — there is no disk copy either way, and the session rebuilds from
    its `messages`."""
    eng = _make_engine(budget_gb=1.5)

    def _failing_save(sid, session):
        return False  # permanent failure (e.g. unserializable empty arrays)

    monkeypatch.setattr(eng, "_save_session_to_disk", _failing_save)

    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._evict_active_sessions_if_needed()

    # A evicted (memory reclaimed) even though it couldn't be persisted.
    assert "A" not in eng._sessions
    assert set(eng._sessions) == {"B"}


def test_transient_save_failure_keeps_session_resident(monkeypatch):
    """If disk save raises (TRANSIENT failure: timeout, stream error), the
    session must NOT be dropped — dropping it would lose the KV cache with no
    disk copy to reload from. Keep it resident and skip this sweep."""
    eng = _make_engine(budget_gb=1.5)

    def _raising_save(sid, session):
        raise RuntimeError("save timed out / no Stream in current thread")

    monkeypatch.setattr(eng, "_save_session_to_disk", _raising_save)

    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._evict_active_sessions_if_needed()

    # Nothing dropped: a lossy rebuild is worse than staying over budget.
    assert set(eng._sessions) == {"A", "B"}


# --- Integration: post-eviction disk-resume contracts ---------------------
#
# These exercise the contract the unit tests above missed (and that ISSUE 1
# regressed): once active LRU eviction REMOVES a session from _sessions (after
# persisting it to disk), the session-mutating APIs must transparently reload
# it from disk instead of reporting it gone. _load_session_from_disk (MLX
# load_prompt_cache + make_prompt_cache) and _rebuild_session (tokenize +
# prefill on the model) are mocked so no weights / GPU are touched.


def _evict_then_setup_disk_reload(eng, monkeypatch):
    """Evict LRU 'A' to disk, then wire a mocked disk-reload for it.

    Returns the (mock) SessionState that a disk reload of 'A' yields, with two
    messages [user, assistant] so prepare_regenerate has something to undo.
    """
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)
    if not hasattr(eng, "_disk_session_ids"):
        eng._disk_session_ids = set()

    # Two 1GB sessions, budget 1.5GB -> LRU 'A' evicted, 'B' (MRU) kept.
    eng._sessions = {
        "A": _make_session(_GB, last_used=100.0),
        "B": _make_session(_GB, last_used=200.0),
    }
    eng._evict_active_sessions_if_needed()
    # Eviction registers the disk copy (real _save_session_to_disk does this;
    # our lambda doesn't, so register here to mirror that side effect).
    eng._disk_session_ids.add("A")
    assert "A" not in eng._sessions  # genuinely evicted

    reloaded = _make_session(_GB, last_used=100.0)
    reloaded.messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    monkeypatch.setattr(
        eng, "_load_session_from_disk",
        lambda sid: reloaded if sid == "A" else None,
    )
    return reloaded


def test_prepare_regenerate_resumes_evicted_session(monkeypatch):
    """ISSUE 1: prepare_regenerate() on an actively-evicted session must
    disk-resume it (not report 'nothing to regenerate') and return something
    regenerable."""
    eng = _make_engine(budget_gb=1.5)
    reloaded = _evict_then_setup_disk_reload(eng, monkeypatch)

    # _rebuild_session needs the model; mock it to a success + record the call.
    rebuilt: list[tuple[str, list]] = []

    def _fake_rebuild(session_id, messages):
        rebuilt.append((session_id, list(messages)))
        eng._sessions[session_id] = reloaded  # rebuilt session is resident
        return {"status": "ok", "cached_tokens": 1}

    monkeypatch.setattr(eng, "_rebuild_session", _fake_rebuild)

    result = eng.prepare_regenerate("A")

    # Did NOT short-circuit to 'nothing to regenerate'.
    assert "error" not in result, result
    assert result.get("status") == "ok"
    # Regenerated by truncating to the message BEFORE the assistant reply (idx 1
    # of [user, assistant] -> restore_to == 0).
    assert result.get("turn") == 0
    assert rebuilt and rebuilt[0][0] == "A"
    # The evicted session was re-loaded from disk during the flow.
    assert "A" in eng._sessions


def test_prepare_regenerate_still_errors_when_no_disk_copy(monkeypatch):
    """A truly-unknown session (not resident, no disk copy) must still report
    'nothing to regenerate' — the resume path must not invent a session."""
    eng = _make_engine(budget_gb=10.0)
    eng._disk_session_ids = set()
    monkeypatch.setattr(eng, "_load_session_from_disk", lambda sid: None)

    result = eng.prepare_regenerate("ghost")
    assert result.get("error") == "nothing to regenerate"


def test_truncate_session_resumes_evicted_session(monkeypatch):
    """The disk-resume path: truncate_session() on an actively-evicted session
    reloads it from disk and rebuilds, rather than reporting 'session not
    found'."""
    eng = _make_engine(budget_gb=1.5)
    reloaded = _evict_then_setup_disk_reload(eng, monkeypatch)

    rebuilt: list[tuple[str, list]] = []

    def _fake_rebuild(session_id, messages):
        rebuilt.append((session_id, list(messages)))
        eng._sessions[session_id] = reloaded
        return {"status": "ok", "cached_tokens": 1}

    monkeypatch.setattr(eng, "_rebuild_session", _fake_rebuild)

    # Truncate 'A' (2 msgs after reload) down to 1 message.
    result = eng.truncate_session("A", target_msg_count=1)

    assert "error" not in result, result
    assert result.get("status") == "ok"
    assert rebuilt and rebuilt[0][0] == "A"
    # Rebuilt only with the surviving (first) message.
    assert rebuilt[0][1] == [{"role": "user", "content": "q"}]
    assert "A" in eng._sessions


def test_status_reports_budget_unmet_when_lone_session_exceeds(monkeypatch):
    """When the single un-evictable (protected/MRU/last) session alone exceeds
    the budget, the sweep stays over budget and status must flag budget_unmet."""
    eng = _make_engine(budget_gb=0.5)  # one 1GB session alone > budget
    monkeypatch.setattr(eng, "_save_session_to_disk", lambda sid, s: True)
    eng._sessions = {"A": _make_session(_GB, last_used=100.0)}

    # Eviction can't help (last session is never dropped) → still over budget.
    eng._evict_active_sessions_if_needed()
    assert set(eng._sessions) == {"A"}

    # Reproduce the admin-status budget math directly (status_dict pulls in a
    # lot of model machinery this bare shell doesn't have).
    active_kv_gb = round(eng._active_sessions_memory_gb(), 2)
    pool_kv_gb = round(eng.cache_manager._memory_usage_gb(), 2)
    total_kv_gb = round(active_kv_gb + pool_kv_gb, 2)
    largest = round(
        max((eng._session_cache_bytes(s) for s in eng._sessions.values()), default=0)
        / 1e9,
        2,
    )
    irreducible_kv_gb = round(largest + pool_kv_gb, 2)
    budget_gb = float(eng.cfg.memory_budget_gb)
    over_budget = total_kv_gb > budget_gb
    budget_unmet = over_budget and irreducible_kv_gb > budget_gb

    assert over_budget is True
    assert budget_unmet is True, "lone session > budget must surface as budget_unmet"
