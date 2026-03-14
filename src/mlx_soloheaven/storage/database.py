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
) -> dict:
    now = time.time()
    mid = uuid.uuid4().hex[:16]
    tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    stats_json = json.dumps(stats, ensure_ascii=False) if stats else None
    async with get_db() as db:
        await db.execute(
            """INSERT INTO messages
               (id, session_id, role, content, tool_calls, tool_call_id,
                thinking, token_count, stats, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, session_id, role, content, tc_json, tool_call_id,
             thinking, token_count, stats_json, now),
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await db.commit()
    return {
        "id": mid, "session_id": session_id, "role": role, "content": content,
        "tool_calls": tool_calls, "thinking": thinking, "stats": stats,
        "created_at": now,
    }


async def get_messages(session_id: str) -> list[dict]:
    async with get_db() as db:
        rows = await db.execute_fetchall(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("tool_calls"):
                d["tool_calls"] = json.loads(d["tool_calls"])
            if d.get("stats"):
                d["stats"] = json.loads(d["stats"])
            result.append(d)
        return result


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
