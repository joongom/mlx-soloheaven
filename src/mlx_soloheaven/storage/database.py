"""SQLite storage for chat sessions, messages, and long-term memory."""

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

_db_path: str = ""


def set_db_path(path: str):
    global _db_path
    _db_path = path


@asynccontextmanager
async def get_db():
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        yield db


async def init_db():
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)
    async with get_db() as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                system_prompt TEXT DEFAULT '',
                temperature REAL DEFAULT 0.6,
                -- Batch-5 F7: sampling params the settings PATCH accepts
                -- (SessionSettings) — the dynamic UPDATE failed on a fresh
                -- DB without these columns. NULL = "not customized": the
                -- chat endpoint falls back to the engine config default
                -- (per-deployment, so no concrete SQL default is baked in).
                top_p REAL,
                min_p REAL,
                top_k INTEGER,
                repetition_penalty REAL,
                thinking_budget INTEGER DEFAULT 8192,
                max_tokens INTEGER DEFAULT 32768,
                context_window_limit INTEGER DEFAULT 100000,
                compaction_strategy TEXT DEFAULT 'summarize',
                total_prompt_tokens INTEGER DEFAULT 0,
                last_compacted_at REAL,
                compaction_history TEXT,
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                thinking TEXT,
                token_count INTEGER DEFAULT 0,
                stats TEXT,
                parent_id TEXT,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT 'general',
                content TEXT NOT NULL,
                source_session_id TEXT,
                importance INTEGER DEFAULT 5,
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS compactions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                old_prompt_tokens INTEGER,
                new_prompt_tokens INTEGER,
                reduction_percent REAL,
                strategy TEXT,
                summary_content TEXT,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_compactions_session
                ON compactions(session_id, created_at);
        """)
        
        # Migration: add new columns if they don't exist (for existing databases)
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN temperature REAL DEFAULT 0.6")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN thinking_budget INTEGER DEFAULT 8192")
            await db.commit()
        except Exception:
            pass

        # Batch-5 F7: sampling columns backing the SessionSettings PATCH
        # (top_p/min_p/top_k/repetition_penalty). NULL = "not customized"
        # (the chat endpoint falls back to the engine config default).
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN top_p REAL")
            await db.commit()
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN min_p REAL")
            await db.commit()
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN top_k INTEGER")
            await db.commit()
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN repetition_penalty REAL")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN max_tokens INTEGER DEFAULT 32768")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN context_window_limit INTEGER DEFAULT 100000")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN compaction_strategy TEXT DEFAULT 'summarize'")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN total_prompt_tokens INTEGER DEFAULT 0")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN last_compacted_at REAL")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN compaction_history TEXT")
            await db.commit()
        except Exception:
            pass
        
        try:
            await db.execute("ALTER TABLE messages ADD COLUMN stats TEXT")
            await db.commit()
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN branched_from TEXT")
            await db.commit()
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN branch_turn INTEGER")
            await db.commit()
        except Exception:
            pass

        # Codex round 7, finding 1: persisted turn ownership. Assistant rows
        # record the id of the user row that originated their turn, so the
        # delete-last walker can find a turn's rows by OWNERSHIP instead of
        # adjacency (an intervening concurrent user row no longer strands a
        # late-landing assistant row as an orphan). Nullable: legacy rows
        # stay NULL and keep the positional-walk behavior.
        try:
            await db.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
            await db.commit()
        except Exception:
            pass


# --- Sessions ---

async def create_session(
    title: str = "New Chat",
    system_prompt: str = "",
    temperature: float = 0.6,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
    context_window_limit: int = 100000,
    compaction_strategy: str = "summarize",
    branched_from: str | None = None,
    branch_turn: int | None = None,
) -> dict:
    now = time.time()
    sid = uuid.uuid4().hex[:16]
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sessions (id, title, system_prompt, temperature, thinking_budget, "
            "max_tokens, context_window_limit, compaction_strategy, branched_from, branch_turn, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, title, system_prompt, temperature, thinking_budget, max_tokens,
             context_window_limit, compaction_strategy, branched_from, branch_turn, now, now),
        )
        await db.commit()
    return {
        "id": sid,
        "title": title,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "context_window_limit": context_window_limit,
        "compaction_strategy": compaction_strategy,
        "branched_from": branched_from,
        "branch_turn": branch_turn,
        "created_at": now,
        "updated_at": now,
    }


async def list_sessions() -> list[dict]:
    async with get_db() as db:
        rows = await db.execute_fetchall(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        )
        return [dict(r) for r in rows]


async def get_session(session_id: str) -> dict | None:
    async with get_db() as db:
        row = await db.execute_fetchall(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        return dict(row[0]) if row else None


async def delete_session(session_id: str):
    async with get_db() as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()


async def update_session(session_id: str, **kwargs):
    kwargs["updated_at"] = time.time()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    async with get_db() as db:
        await db.execute(f"UPDATE sessions SET {sets} WHERE id = ?", vals)
        await db.commit()


# --- Session Settings ---

async def get_session_settings(session_id: str) -> dict | None:
    """Get session settings (generation params + compaction config)."""
    session = await get_session(session_id)
    if not session:
        return None
    
    return {
        "system_prompt": session.get("system_prompt", ""),
        "temperature": session.get("temperature", 0.6),
        "thinking_budget": session.get("thinking_budget", 8192),
        "max_tokens": session.get("max_tokens", 32768),
        "context_window_limit": session.get("context_window_limit", 100000),
        "compaction_strategy": session.get("compaction_strategy", "summarize"),
    }


async def update_session_settings(
    session_id: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
    context_window_limit: int | None = None,
    compaction_strategy: str | None = None,
) -> bool:
    """Update session settings."""
    updates = {}
    if system_prompt is not None:
        updates["system_prompt"] = system_prompt
    if temperature is not None:
        updates["temperature"] = temperature
    if thinking_budget is not None:
        updates["thinking_budget"] = thinking_budget
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens
    if context_window_limit is not None:
        updates["context_window_limit"] = context_window_limit
    if compaction_strategy is not None:
        updates["compaction_strategy"] = compaction_strategy
    
    if updates:
        await update_session(session_id, **updates)
    
    return True


# --- Messages ---

async def add_message(
    session_id: str,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
    thinking: str | None = None,
    token_count: int = 0,
    stats: dict | None = None,
    parent_id: str | None = None,
) -> dict:
    """``parent_id`` (codex round 7, finding 1): id of the user row that
    originated this row's turn — set on assistant rows where the API layer
    knows it; user rows keep NULL."""
    now = time.time()
    mid = uuid.uuid4().hex[:16]
    tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    stats_json = json.dumps(stats, ensure_ascii=False) if stats else None
    async with get_db() as db:
        await db.execute(
            """INSERT INTO messages
               (id, session_id, role, content, tool_calls, tool_call_id,
                thinking, token_count, stats, parent_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, session_id, role, content, tc_json, tool_call_id,
             thinking, token_count, stats_json, parent_id, now),
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await db.commit()
    return {
        "id": mid, "session_id": session_id, "role": role, "content": content,
        "tool_calls": tool_calls, "thinking": thinking, "stats": stats,
        "parent_id": parent_id, "created_at": now,
    }


async def add_message_if_parent_exists(
    session_id: str,
    role: str,
    *,
    parent_id: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
    thinking: str | None = None,
    token_count: int = 0,
    stats: dict | None = None,
) -> dict | None:
    """Insert a message ONLY IF its originating (parent) row still exists.

    Codex round 3, finding 2: /chat persists the user row BEFORE generation
    and the assistant row only AFTER — a delete-last landing in between
    removes the user row, and an unconditional assistant insert then creates
    an ORPHAN assistant row for a turn that no longer exists. The existence
    check and the insert are ONE atomic statement (INSERT ... SELECT ...
    WHERE EXISTS), so a delete committing between a separate SELECT and
    INSERT can never slip through.

    Returns the inserted row dict, or ``None`` when the parent row is gone
    (the caller skips persistence — the turn was deleted mid-generation).

    Codex round 7, finding 1: the inserted row PERSISTS ``parent_id`` — the
    conditional insert alone cannot prevent the orphan when a concurrent
    request appends a NEW user row between the delete handler's pre-snapshot
    and this insert (the parent still exists, so the insert lands, but the
    delete transaction's positional walk stops at the intervening user-row
    boundary and never reaches this row). The delete walker now also deletes
    every row whose ``parent_id`` equals the turn's anchor, regardless of
    position (ownership, not adjacency).
    """
    now = time.time()
    mid = uuid.uuid4().hex[:16]
    tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    stats_json = json.dumps(stats, ensure_ascii=False) if stats else None
    async with get_db() as db:
        cur = await db.execute(
            """INSERT INTO messages
               (id, session_id, role, content, tool_calls, tool_call_id,
                thinking, token_count, stats, parent_id, created_at)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
               WHERE EXISTS (
                   SELECT 1 FROM messages WHERE id = ? AND session_id = ?
               )""",
            (mid, session_id, role, content, tc_json, tool_call_id,
             thinking, token_count, stats_json, parent_id, now,
             parent_id, session_id),
        )
        inserted = cur.rowcount > 0
        if inserted:
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        await db.commit()
    if not inserted:
        return None
    return {
        "id": mid, "session_id": session_id, "role": role, "content": content,
        "tool_calls": tool_calls, "thinking": thinking, "stats": stats,
        "parent_id": parent_id, "created_at": now,
    }


def _decode_message_row(row) -> dict:
    """Decode one messages row into the dict shape get_messages returns
    (tool_calls / stats JSON columns parsed)."""
    d = dict(row)
    if d.get("tool_calls"):
        d["tool_calls"] = json.loads(d["tool_calls"])
    if d.get("stats"):
        d["stats"] = json.loads(d["stats"])
    return d


async def get_messages(session_id: str, limit: int | None = None) -> list[dict]:
    async with get_db() as db:
        if limit:
            # Get last N messages (most recent)
            rows = await db.execute_fetchall(
                "SELECT * FROM ("
                "  SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
                ") ORDER BY created_at",
                (session_id, limit),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
        return [_decode_message_row(r) for r in rows]


async def count_messages_after_last_compaction(session_id: str) -> int:
    """Count messages from the last compaction point (inclusive) to end."""
    async with get_db() as db:
        # Find the last compaction message
        rows = await db.execute_fetchall(
            "SELECT created_at FROM messages WHERE session_id = ? "
            "AND content LIKE 'The conversation history before this point was compacted%' "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        if not rows:
            return 0  # no compaction → don't force extra loading

        compact_time = rows[0]["created_at"]
        count_rows = await db.execute_fetchall(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ? AND created_at >= ?",
            (session_id, compact_time),
        )
        return (count_rows[0]["cnt"] if count_rows else 0) + 1  # +1 to include compaction msg itself


async def get_message_count(session_id: str) -> int:
    async with get_db() as db:
        rows = await db.execute_fetchall(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        )
        return rows[0]["cnt"] if rows else 0


async def get_session_total_tokens(session_id: str) -> int:
    """Calculate total prompt tokens for a session."""
    async with get_db() as db:
        row = await db.execute_fetchall(
            "SELECT total_prompt_tokens FROM sessions WHERE id = ?", (session_id,)
        )
        if row:
            return row[0]["total_prompt_tokens"] or 0
        
        # Fallback: sum all message token counts
        rows = await db.execute_fetchall(
            "SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE session_id = ?",
            (session_id,),
        )
        return rows[0]["total"] or 0


async def update_session_tokens(session_id: str, total_tokens: int):
    """Update total prompt tokens for a session."""
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET total_prompt_tokens = ?, updated_at = ? WHERE id = ?",
            (total_tokens, time.time(), session_id),
        )
        await db.commit()


# --- Compaction ---

async def record_compaction(
    session_id: str,
    old_tokens: int,
    new_tokens: int,
    reduction_percent: float,
    strategy: str,
    summary_content: str | None = None,
) -> dict:
    """Record a compaction event."""
    now = time.time()
    cid = uuid.uuid4().hex[:16]
    
    async with get_db() as db:
        await db.execute(
            """INSERT INTO compactions
               (id, session_id, old_prompt_tokens, new_prompt_tokens, reduction_percent,
                strategy, summary_content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, session_id, old_tokens, new_tokens, reduction_percent,
             strategy, summary_content, now),
        )
        
        # Update session's last_compacted_at
        await db.execute(
            "UPDATE sessions SET last_compacted_at = ? WHERE id = ?",
            (now, session_id),
        )
        
        await db.commit()
    
    return {
        "id": cid,
        "session_id": session_id,
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "reduction_percent": reduction_percent,
        "strategy": strategy,
        "summary_content": summary_content,
        "created_at": now,
    }


async def get_compactions(session_id: str, limit: int = 50) -> list[dict]:
    """Get compaction history for a session."""
    async with get_db() as db:
        rows = await db.execute_fetchall(
            "SELECT * FROM compactions WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in rows]


async def update_session_compacted_state(
    session_id: str,
    new_messages: list[dict],
    summary_content: str | None = None,
):
    """Update session after compaction (reset messages, update tokens)."""
    # This would be called after compaction to update the session state
    # In practice, you'd want to:
    # 1. Delete old messages
    # 2. Insert new compacted messages
    # 3. Update total_prompt_tokens
    pass


async def delete_messages_before(session_id: str, before_created_at: float):
    """Delete all messages in a session created before the given timestamp."""
    async with get_db() as db:
        await db.execute(
            "DELETE FROM messages WHERE session_id = ? AND created_at < ?",
            (session_id, before_created_at),
        )
        await db.commit()


async def delete_last_message(session_id: str):
    """Delete the last message in a session."""
    async with get_db() as db:
        await db.execute(
            "DELETE FROM messages WHERE id = ("
            "  SELECT id FROM messages WHERE session_id = ? "
            "  ORDER BY created_at DESC LIMIT 1"
            ")", (session_id,)
        )
        await db.commit()


async def delete_messages_by_ids(session_id: str, ids: list[str]) -> int:
    """Delete exactly the given message rows, atomically (ONE transaction).

    U25 round 2 (concurrency): delete-last used to call delete_last_message
    N times — N separate committed transactions each targeting whichever row
    was CURRENTLY newest. A chat append committed between the handler's
    snapshot and those deletes made the loop delete the NEW user row and
    leave part of the old turn behind. Deleting by the snapshot's identified
    row ids is safe against concurrent appends by construction: rows created
    after the snapshot can never match, and an id another writer already
    removed simply doesn't match either (no error, no positional drift).
    Returns the number of rows actually deleted.
    """
    if not ids:
        return 0
    # NIT (codex round 3): chunk the IN clause — SQLite builds compiled with
    # a low SQLITE_MAX_VARIABLE_NUMBER (999 historically) reject a single
    # statement with thousands of placeholders. All chunks execute inside
    # ONE transaction (a single commit below), so atomicity is preserved.
    _CHUNK = 500
    deleted = 0
    async with get_db() as db:
        for i in range(0, len(ids), _CHUNK):
            chunk = ids[i:i + _CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            cur = await db.execute(
                f"DELETE FROM messages WHERE session_id = ? "
                f"AND id IN ({placeholders})",
                (session_id, *chunk),
            )
            deleted += cur.rowcount
        await db.commit()
        return deleted


async def delete_last_turn_tx(
    session_id: str,
    compute_delete_ids,
) -> tuple[list[str], list[dict]]:
    """Snapshot + turn-walk + delete as ONE write transaction.

    Codex round 5, finding 1: the round-4 delete-last shape snapshotted the
    rows in a SEPARATE read, so the /chat conditional assistant insert
    (add_message_if_parent_exists) could commit BETWEEN that snapshot and
    the id-targeted delete — the insert's WHERE EXISTS saw the user row
    still present, the already-computed delete then removed ONLY the user
    row, and the assistant row survived as an ORPHAN the drift rebuild fed
    to the engine.

    BEGIN IMMEDIATE takes the write lock UP FRONT, so any concurrent write
    transaction serializes entirely BEFORE this one (the snapshot below
    then includes its rows and the walker can extend the turn over them)
    or entirely AFTER (the conditional insert's WHERE EXISTS sees the user
    row already gone and skips). There is no in-between.

    ``compute_delete_ids(rows) -> list[str]`` is a PURE callback: it
    receives the transaction's own row snapshot (get_messages dict shape,
    ordered by created_at) and returns the ids to delete (empty → nothing
    is deleted). Returns ``(deleted_ids, remaining_rows)`` where
    ``remaining_rows`` is snapshot-minus-deleted — exact under the held
    write lock, no separate re-read can drift from it.
    """
    _CHUNK = 500
    async with get_db() as db:
        # Explicit write transaction: sqlite3's autocommit handling only
        # auto-BEGINs (deferred) before DML; an explicit BEGIN IMMEDIATE
        # acquires the RESERVED lock now, before the snapshot read.
        await db.execute("BEGIN IMMEDIATE")
        try:
            rows = await db.execute_fetchall(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY created_at",
                (session_id,),
            )
            snapshot = [_decode_message_row(r) for r in rows]
            delete_ids = list(compute_delete_ids(snapshot) or [])
            for i in range(0, len(delete_ids), _CHUNK):
                chunk = delete_ids[i:i + _CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                await db.execute(
                    f"DELETE FROM messages WHERE session_id = ? "
                    f"AND id IN ({placeholders})",
                    (session_id, *chunk),
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
    deleted_set = set(delete_ids)
    remaining = [m for m in snapshot if m["id"] not in deleted_set]
    return delete_ids, remaining


# --- Memories (long-term) ---

async def add_memory(
    content: str,
    category: str = "general",
    source_session_id: str | None = None,
    importance: int = 5,
) -> dict:
    now = time.time()
    mid = uuid.uuid4().hex[:16]
    async with get_db() as db:
        await db.execute(
            """INSERT INTO memories
               (id, category, content, source_session_id, importance, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mid, category, content, source_session_id, importance, now, now),
        )
        await db.commit()
    return {"id": mid, "category": category, "content": content, "importance": importance}


async def get_memories(category: str | None = None, limit: int = 50) -> list[dict]:
    async with get_db() as db:
        if category:
            rows = await db.execute_fetchall(
                "SELECT * FROM memories WHERE category = ? "
                "ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (category, limit),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT * FROM memories "
                "ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in rows]


async def search_memories(query: str, limit: int = 20) -> list[dict]:
    """Simple text search on memory content."""
    async with get_db() as db:
        rows = await db.execute_fetchall(
            "SELECT * FROM memories WHERE content LIKE ? "
            "ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        return [dict(r) for r in rows]
