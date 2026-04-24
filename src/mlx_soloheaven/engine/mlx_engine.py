"""
MLX Engine — model loading and generation with KV cache reuse.

Uses mlx-vlm as the unified generation backend for both text-only and
vision-language models.  Session-based KV cache management is built on
mlx-vlm's PromptCacheState which does prefix-matching on token IDs.

Cache reuse across turns:
- Engine-internal messages store full assistant content (including thinking)
- apply_chat_template(tokenize=True) produces tokens matching stored IDs
- PromptCacheState.find_prefix_length() reuses the common prefix
- Only new user-message tokens are processed each turn

Cross-session sharing:
- Base cache pool stores system-prompt KV snapshots
- New sessions are seeded from the base cache via PromptCacheState
"""

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, Optional

import mlx.core as mx
from mlx_lm import load as lm_load
from mlx_lm import stream_generate as lm_stream_generate
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import (
    stream_generate as vlm_stream_generate,
    PromptCacheState,
)

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.thinking import ThinkingBudgetProcessor, RepetitionPenaltyProcessor
from mlx_soloheaven.engine.tool_parser import (
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
)
from mlx_soloheaven.cache.manager import CacheManager

logger = logging.getLogger(__name__)


def _pld_response_adapter(pld_iter, tokenizer):
    """Adapt pld_generate_step's (token, logprobs, from_draft) tuples to
    mimic lm_stream_generate's GenerationResponse objects.

    - Uses mlx-lm's StreamingDetokenizer (buffers partial UTF-8 byte
      sequences so multi-byte characters like CJK aren't emitted as
      replacement chars \ufffd between tokens).
    - Performs EOS detection — pld_generate_step doesn't stop on its own
      (mlx-lm's stream_generate normally handles that). EOS ids include
      the wrapper's list + HF tokenizer's (list or single).
    """
    import time as _time
    from types import SimpleNamespace

    # Collect EOS token IDs from both mlx-lm wrapper and HF tokenizer
    eos_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        eos_ids.update(tokenizer.eos_token_ids)
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    eid = getattr(inner, "eos_token_id", None)
    if eid is not None:
        if isinstance(eid, (list, tuple, set)):
            eos_ids.update(eid)
        else:
            eos_ids.add(eid)
    # GLM-family and other models expose multi-EOS via generation_config
    gc = getattr(inner, "generation_config", None)
    if gc is not None:
        gc_eos = getattr(gc, "eos_token_id", None)
        if gc_eos is not None:
            if isinstance(gc_eos, (list, tuple, set)):
                eos_ids.update(gc_eos)
            else:
                eos_ids.add(gc_eos)

    # Use mlx-lm's StreamingDetokenizer to buffer partial UTF-8 bytes
    # across tokens (mirrors stream_generate's behavior).
    detok = None
    make_detok = getattr(tokenizer, "detokenizer", None)
    if make_detok is not None:
        # TokenizerWrapper.detokenizer is a property that returns a fresh
        # detokenizer every access. Call .reset() and use it directly.
        try:
            detok = tokenizer.detokenizer
            detok.reset()
        except Exception:
            detok = None

    t_first = None
    count = 0
    from_draft_count = 0
    last_segment = ""
    for token, _logprobs, from_draft in pld_iter:
        count += 1
        if from_draft:
            from_draft_count += 1
        if t_first is None:
            t_first = _time.perf_counter()
        now = _time.perf_counter()
        tps = count / (now - t_first) if t_first and now > t_first else 0.0

        # Stop on EOS BEFORE emitting the EOS token's (often empty) text
        if token in eos_ids:
            # Flush any remaining buffered segment first
            if detok is not None:
                try:
                    detok.finalize()
                    remaining = detok.last_segment
                    if remaining:
                        yield SimpleNamespace(
                            text=remaining, token=token,
                            prompt_tps=0.0, generation_tps=tps,
                            from_draft=from_draft,
                        )
                except Exception:
                    pass
            break

        if detok is not None:
            try:
                detok.add_token(token)
                text = detok.last_segment
            except Exception:
                # Fallback: decode single token (may yield replacement chars)
                text = tokenizer.decode([token])
        else:
            text = tokenizer.decode([token])

        if not text:
            # Partial UTF-8 — buffered, wait for next token
            continue

        yield SimpleNamespace(
            text=text, token=token,
            prompt_tps=0.0, generation_tps=tps,
            from_draft=from_draft,
        )

    if count > 0:
        logger.info(
            f"[PLD] accepted {from_draft_count}/{count} draft tokens "
            f"({100*from_draft_count/count:.1f}% acceptance rate)"
        )


@dataclass
class GenerationResult:
    """A single token/chunk from generation."""
    text: str = ""
    token: int = 0
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    status: Optional[str] = None  # "generating" when lock acquired
    cache_info: Optional[dict] = None  # cache hit/miss details


@dataclass
class CompletionResult:
    """Full completion result after generation finishes."""
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None


@dataclass
class SessionState:
    """Tracks a conversation session's KV cache and message history.

    Engine-internal messages include full assistant content (thinking + content)
    so that apply_chat_template(tokenize=True) produces tokens matching the
    stored PromptCacheState.token_ids for prefix-matching cache reuse.
    """
    cache_state: PromptCacheState  # mlx-vlm native: KV cache + token history
    messages: list[dict]  # messages WITH thinking in assistant content
    last_used: float = field(default_factory=time.time)
    total_cache_tokens: int = 0

    # Cache build time from last truncate/rebuild (seconds, consumed once)
    pending_build_time: float = 0.0

    def touch(self):
        self.last_used = time.time()


def _detect_token_id(tokenizer, text: str) -> int:
    """Auto-detect a special token's ID from the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()
    if text in vocab:
        return vocab[text]
    # Try encoding as fallback
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return -1


@dataclass
class BaseCacheEntry:
    """A cached KV state for a shared system prompt prefix."""
    system_hash: str
    cache: list  # mlx-lm prompt_cache snapshot (at end of system tokens)
    tokens: list[int]  # tokenized system message
    token_count: int
    created: float = field(default_factory=time.time)
    hit_count: int = 0


class MLXEngine:
    """MLX model engine with session-based KV cache reuse."""

    # Metal GPU goes cold after ~5s idle, causing ~2s TTFT penalty.
    # Keep GPU warm with periodic small computation.
    GPU_KEEPALIVE_INTERVAL = 1.0  # Short interval to prevent deep Metal sleep

    # Shared GPU lock across all engines (Metal can't handle concurrent command encoders)
    _global_gpu_lock = threading.Lock()
    _global_keepalive_started = False
    _global_last_gpu_activity = time.time()
    _global_keepalive_stop = threading.Event()
    _all_engines: list["MLXEngine"] = []

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._vlm_model = None
        self._language_model = None
        self._processor = None
        self.tokenizer = None
        self._lock = MLXEngine._global_gpu_lock  # shared lock
        self.cache_manager = CacheManager(
            memory_budget_gb=cfg.memory_budget_gb,
            disk_budget_gb=cfg.disk_budget_gb,
            cache_dir=cfg.cache_dir,
        )
        self.model_id = ""

        # Session-based cache: session_id -> SessionState
        self._sessions: dict[str, SessionState] = {}

        # Base cache pool: system_hash -> BaseCacheEntry
        self._base_caches: dict[str, BaseCacheEntry] = {}

        # Dirty session tracking for idle-time disk save
        self._dirty_sessions: set[str] = set()
        self._dirty_lock = threading.Lock()
        MLXEngine._all_engines.append(self)

    def load_model(self):
        logger.info(f"Loading model: {self.cfg.model_path}")
        t0 = time.perf_counter()

        # Detect model type from config.json
        model_config = {}
        config_path = os.path.join(self.cfg.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_config = json.load(f)
        self._model_type = model_config.get("model_type", "")

        # Check if mlx-vlm supports this model type BEFORE loading weights.
        # (mlx-vlm's load loads the entire safetensors before checking model type,
        #  which wastes memory for huge models like GLM-5.1 at 378GB.)
        self._use_vlm = False
        vlm_supported = self._vlm_supports(self._model_type)
        if vlm_supported:
            try:
                self._vlm_model, self._processor = vlm_load(self.cfg.model_path)
                self._language_model = getattr(
                    self._vlm_model, "language_model", self._vlm_model
                )
                self.tokenizer = getattr(self._processor, "tokenizer", self._processor)
                self._use_vlm = True
                logger.info("Loaded via mlx-vlm")
            except Exception as e:
                logger.info(f"mlx-vlm load failed ({e}), falling back to mlx-lm")

        if not self._use_vlm:
            model, tokenizer = lm_load(self.cfg.model_path)
            self._vlm_model = model
            self._language_model = model
            self._processor = None
            self.tokenizer = tokenizer
            logger.info("Loaded via mlx-lm")

        elapsed = time.perf_counter() - t0

        # Derive model ID from directory name
        self.model_id = os.path.basename(self.cfg.model_path.rstrip("/"))
        logger.info(f"Model loaded in {elapsed:.1f}s — {self.model_id}")

        # Detect model family
        self.model_family = self._detect_model_family()
        logger.info(f"Model family: {self.model_family}")

        # Auto-detect thinking end token (needed for SSE thinking_done signal)
        self._detect_special_tokens()

        # Detect cache layer types (for logging/diagnostics)
        test_cache = make_prompt_cache(self._language_model)
        self._has_rotating_cache = any(
            type(c).__name__ == "RotatingKVCache" for c in test_cache
        )
        self._sliding_window_size = 0
        if self._has_rotating_cache:
            for c in test_cache:
                if type(c).__name__ == "RotatingKVCache" and hasattr(c, "max_size"):
                    self._sliding_window_size = c.max_size
                    break
        del test_cache
        if self._has_rotating_cache:
            logger.info(
                f"[{self.model_id}] RotatingKVCache detected — "
                f"sliding_window={self._sliding_window_size}"
            )

        if self.cfg.think_end_token < 0 and self.cfg.enable_thinking:
            self.cfg.enable_thinking = False
            logger.info(
                f"[{self.model_id}] think_end token not found — auto-disabled thinking"
            )

        logger.debug(
            f"[{self.model_id}] enable_thinking={self.cfg.enable_thinking}"
        )

        if self.cfg.prefill_step_size != 2048:
            logger.info(
                f"[{self.model_id}] Prefill step size: {self.cfg.prefill_step_size} "
                f"(default 2048)"
            )
        if self.cfg.kv_bits and not self._use_vlm:
            logger.info(
                f"[{self.model_id}] KV cache quantization: "
                f"bits={self.cfg.kv_bits}, group_size={self.cfg.kv_group_size}, "
                f"start={self.cfg.quantized_kv_start}"
            )
        elif self.cfg.kv_bits and self._use_vlm:
            logger.warning(
                f"[{self.model_id}] --kv-bits ignored (mlx-vlm path — quantization "
                f"not supported; only mlx-lm fallback models support this)"
            )

        if self.cfg.pld_enabled:
            if self._use_vlm:
                logger.warning(
                    f"[{self.model_id}] --pld ignored (mlx-vlm path — PLD "
                    f"not supported; falling back to standard VLM generation)"
                )
            else:
                logger.info(
                    f"[{self.model_id}] PLD enabled: "
                    f"num_draft={self.cfg.pld_num_draft_tokens}, k={self.cfg.pld_ngram_k}"
                )

        # Set wired limit once at startup
        if mx.metal.is_available():
            max_rec = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(max_rec)
            logger.debug(f"Metal wired limit set to {max_rec / 1e9:.1f}GB")

            # Patch wired_limit in mlx_vlm: keep synchronize but skip set/reset cycle.
            # The set/reset cycle degrades Metal TTFT on repeated calls.
            import mlx_vlm.generate as vlm_gen_module

            @contextlib.contextmanager
            def _stable_wired_limit(model, streams=None):
                try:
                    yield
                finally:
                    if streams:
                        for s in streams:
                            mx.synchronize(s)
                    else:
                        mx.synchronize()

            vlm_gen_module.wired_limit = _stable_wired_limit
            logger.debug("Patched wired_limit: stable (set once at startup)")

        self._build_disk_index()
        self._touch_gpu()  # Mark GPU active after model load
        if self.cfg.gpu_keepalive:
            self._start_gpu_keepalive()
            logger.info(f"[{self.model_id}] GPU keepalive enabled (interval={self.GPU_KEEPALIVE_INTERVAL}s)")

    # --- Model detection helpers ---

    @staticmethod
    def _vlm_supports(model_type: str) -> bool:
        """Check if mlx-vlm has a model module for this model_type.

        Done BEFORE calling vlm_load to avoid loading huge weights
        just to fail on the model-type check (mlx-vlm currently loads
        all safetensors before checking model support).
        """
        if not model_type:
            return False
        try:
            import importlib
            importlib.import_module(f"mlx_vlm.models.{model_type}")
            return True
        except ImportError:
            return False

    def _detect_model_family(self) -> str:
        """Detect model family from model_type in config.json."""
        mt = self._model_type.lower()
        if "gemma4" in mt:
            return "gemma4"
        if "glm" in mt:
            return "glm"
        # Default: ChatML family (Qwen, MiniMax, etc.)
        return "chatml"

    def _detect_special_tokens(self):
        """Detect thinking end token for SSE thinking_done signal."""
        if self.model_family == "gemma4":
            self.cfg.think_end_token = _detect_token_id(self.tokenizer, "<channel|>")
        else:
            # ChatML and GLM both use </think>
            if self.cfg.think_end_token < 0:
                self.cfg.think_end_token = _detect_token_id(self.tokenizer, "</think>")

        logger.info(
            f"[{self.model_id}] model_family={self.model_family} | "
            f"think_end_token={self.cfg.think_end_token}"
        )

    # --- GPU keepalive ---

    def _start_gpu_keepalive(self):
        """Start background thread that keeps Metal GPU warm (once globally)."""
        if not mx.metal.is_available():
            return
        if MLXEngine._global_keepalive_started:
            return
        MLXEngine._global_keepalive_started = True

        self._keepalive_ping_count = 0

        def _keepalive_loop():
            logger.debug("[GPU Keepalive] Started (interval=%.1fs)", self.GPU_KEEPALIVE_INTERVAL)
            while not MLXEngine._global_keepalive_stop.wait(self.GPU_KEEPALIVE_INTERVAL):
                idle = time.time() - MLXEngine._global_last_gpu_activity
                if idle >= self.GPU_KEEPALIVE_INTERVAL:
                    if self._lock.acquire(blocking=False):
                        try:
                            t0 = time.perf_counter()
                            a = mx.random.normal((32, 32))
                            b = a @ a
                            mx.eval(b)
                            elapsed = (time.perf_counter() - t0) * 1000
                            self._keepalive_ping_count += 1
                            if self._keepalive_ping_count % 100 == 1 or elapsed > 100:
                                logger.info(
                                    f"[GPU Keepalive] ping #{self._keepalive_ping_count} "
                                    f"idle={idle:.1f}s, elapsed={elapsed:.0f}ms"
                                )
                            # Flush dirty sessions while GPU is idle and we hold the lock
                            for engine in MLXEngine._all_engines:
                                engine._flush_dirty_sessions()
                        except Exception as e:
                            logger.warning(f"[GPU Keepalive] ping failed: {e}")
                        finally:
                            self._lock.release()

        self._keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
        self._keepalive_thread.start()

        # Register shutdown handler to stop keepalive cleanly
        import atexit
        import signal

        _shutdown_done = False

        def _shutdown(*args):
            nonlocal _shutdown_done
            if _shutdown_done:
                return
            _shutdown_done = True
            MLXEngine._global_keepalive_stop.set()
            logger.info("[Shutdown] Flushing dirty sessions...")
            MLXEngine._flush_all_on_shutdown()
            logger.info("[Shutdown] Complete")

        atexit.register(_shutdown)
        # Handle SIGINT/SIGTERM so Ctrl+C stops keepalive immediately
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _signal_handler(signum, frame):
            _shutdown()
            # Restore default handler so next Ctrl+C forces quit
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    def _touch_gpu(self):
        """Mark GPU as recently active (resets keepalive timer)."""
        MLXEngine._global_last_gpu_activity = time.time()

    # --- Disk cache persistence ---

    def _session_cache_path(self, session_id: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"session_{session_id}.safetensors")

    def _save_session_to_disk(self, session_id: str, session: SessionState):
        """Save session's KV cache + token history to disk. Caller MUST hold _lock.

        Returns True on success, False if save is not possible (e.g. empty arrays).
        """
        if session.cache_state is None or session.cache_state.cache is None:
            return True
        t0 = time.perf_counter()
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = self._session_cache_path(session_id)
        metadata = {
            "session_id": session_id,
            "messages": json.dumps(session.messages, ensure_ascii=False),
            "total_cache_tokens": str(session.total_cache_tokens),
            "last_used": str(session.last_used),
            "token_ids": json.dumps(session.cache_state.token_ids or []),
        }
        try:
            save_prompt_cache(path, session.cache_state.cache, metadata=metadata)
        except Exception as e:
            if "empty array" in str(e).lower() or "cannot serialize" in str(e).lower():
                # Some models (GLM MoE) have empty arrays that safetensors can't handle
                logger.info(
                    f"[KV Cache] session={session_id} | DISK SAVE SKIP | "
                    f"cache not serializable: {e}"
                )
                return False  # permanent failure, don't retry
            raise  # re-raise unexpected errors

        elapsed = time.perf_counter() - t0
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.add(session_id)
        fsize = os.path.getsize(path) / 1e6
        logger.info(
            f"[KV Cache] session={session_id} | DISK SAVE | "
            f"{session.total_cache_tokens} tokens, {len(session.messages)} msgs, "
            f"{fsize:.1f}MB, {elapsed:.2f}s"
        )
        # LRU eviction: keep total disk usage under budget
        self._evict_disk_sessions_if_needed(protect_session_id=session_id)
        return True

    def _evict_disk_sessions_if_needed(self, protect_session_id: str | None = None):
        """Scan cache_dir and delete oldest session files if disk usage exceeds budget.

        Protects:
        - The session we just saved (protect_session_id)
        - Any session currently in self._sessions (in-memory, active)
        """
        budget_bytes = int(self.cfg.disk_budget_gb * 1e9)
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        # Gather session file info: path, size, mtime, session_id
        entries = []
        total_size = 0
        for fname in os.listdir(cache_dir):
            if not fname.startswith("session_") or not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(cache_dir, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            total_size += st.st_size
            # Extract session_id: session_<id>.safetensors or session_<id>_ckpt.safetensors
            sid_part = fname[len("session_"):-len(".safetensors")]
            sid = sid_part[:-len("_ckpt")] if sid_part.endswith("_ckpt") else sid_part
            entries.append((st.st_mtime, st.st_size, fpath, sid))

        if total_size <= budget_bytes:
            return

        # Sort oldest first
        entries.sort(key=lambda e: e[0])
        protected = set(self._sessions.keys())
        if protect_session_id:
            protected.add(protect_session_id)

        deleted = 0
        freed = 0
        for mtime, size, fpath, sid in entries:
            if total_size <= budget_bytes:
                break
            if sid in protected:
                continue
            try:
                os.remove(fpath)
                total_size -= size
                freed += size
                deleted += 1
                if hasattr(self, "_disk_session_ids"):
                    self._disk_session_ids.discard(sid)
            except OSError as e:
                logger.debug(f"[Disk LRU] failed to delete {fpath}: {e}")

        if deleted:
            logger.info(
                f"[Disk LRU] evicted {deleted} files, freed {freed/1e9:.2f} GB "
                f"(total now {total_size/1e9:.2f}/{budget_bytes/1e9:.2f} GB)"
            )

    def _mark_dirty(self, session_id: str):
        """Mark a session for disk save on next idle cycle."""
        with self._dirty_lock:
            self._dirty_sessions.add(session_id)

    def _flush_dirty_sessions(self):
        """Flush all dirty sessions to disk. Caller MUST hold _lock."""
        with self._dirty_lock:
            to_save = self._dirty_sessions.copy()
            self._dirty_sessions.clear()

        if not to_save:
            return

        saved = 0
        for sid in to_save:
            session = self._sessions.get(sid)
            if session is None:
                continue
            try:
                success = self._save_session_to_disk(sid, session)
                if success:
                    saved += 1
                # If success=False (permanent failure like empty arrays), don't retry
            except Exception as e:
                logger.error(f"[KV Cache] session={sid} | FLUSH SAVE FAILED | {e}")
                with self._dirty_lock:
                    if sid in self._sessions:
                        self._dirty_sessions.add(sid)

        if saved:
            logger.info(f"[Idle Flush] saved {saved}/{len(to_save)} dirty sessions")

    @classmethod
    def _flush_all_on_shutdown(cls):
        """Save all dirty sessions across all engines on shutdown."""
        for engine in cls._all_engines:
            with engine._dirty_lock:
                to_save = engine._dirty_sessions.copy()
                engine._dirty_sessions.clear()
            if not to_save:
                continue
            logger.info(f"[Shutdown] Flushing {len(to_save)} dirty sessions for {engine.model_id}")
            with engine._lock:
                for sid in to_save:
                    session = engine._sessions.get(sid)
                    if session:
                        try:
                            engine._save_session_to_disk(sid, session)
                        except Exception as e:
                            logger.error(f"[Shutdown] Failed to save session {sid}: {e}")

    def _load_session_from_disk(self, session_id: str) -> Optional[SessionState]:
        """Load session's KV cache + token history from disk."""
        path = self._session_cache_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            t0 = time.perf_counter()
            cache, metadata = load_prompt_cache(path, return_metadata=True)
            messages = json.loads(metadata.get("messages", "[]"))
            total_tokens = int(metadata.get("total_cache_tokens", "0"))
            last_used = float(metadata.get("last_used", "0"))
            token_ids = json.loads(metadata.get("token_ids", "[]"))

            # Verify loaded cache matches model structure
            model_cache = make_prompt_cache(self._language_model)
            if len(cache) != len(model_cache):
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                    f"layer count mismatch: {len(cache)} vs {len(model_cache)}"
                )
                return None

            type_ok = all(
                type(c).__name__ == type(m).__name__
                for c, m in zip(cache, model_cache)
            )
            if not type_ok:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | cache type mismatch"
                )
                return None

            loaded_offset = self._get_cache_offset(cache)
            elapsed = time.perf_counter() - t0

            # Reconstruct PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = cache
            cache_state.token_ids = token_ids if token_ids else None

            session = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=loaded_offset,
                last_used=last_used,
            )

            fsize = os.path.getsize(path) / 1e6
            logger.info(
                f"[KV Cache] session={session_id} | DISK LOAD | "
                f"{loaded_offset} tokens, {len(messages)} msgs, "
                f"{fsize:.1f}MB, {elapsed:.2f}s"
            )
            return session
        except Exception as e:
            logger.error(f"[KV Cache] session={session_id} | DISK LOAD FAILED | {e}")
            return None

    def _build_disk_index(self):
        """Scan cache_dir for saved session caches."""
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        self._disk_session_ids: set[str] = set()
        count = 0
        for fname in os.listdir(cache_dir):
            if fname.startswith("session_") and fname.endswith(".safetensors") and "_ckpt" not in fname:
                sid = fname[len("session_"):-len(".safetensors")]
                self._disk_session_ids.add(sid)
                count += 1

        if count:
            logger.debug(f"[KV Cache] Disk index: {count} saved session caches")
        else:
            logger.debug("[KV Cache] Disk index: no saved session caches")

    def _has_disk_cache(self, session_id: str) -> bool:
        return hasattr(self, "_disk_session_ids") and session_id in self._disk_session_ids

    # --- Base cache pool ---

    @staticmethod
    def _system_hash(messages: list[dict], tools: list | None = None) -> str | None:
        """Hash the first system message (+ tools) for base cache lookup.

        Returns a hash even when there's no explicit system message — uses empty
        string as content. This supports models where the template auto-generates
        a system prefix (e.g. Gemma 4).
        """
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content", "")
        else:
            # No explicit system message — use empty content as hash key
            # (template may still generate a system prefix)
            content = ""
        h = hashlib.sha256(content.encode())
        if tools:
            h.update(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    def _find_base_cache(self, messages: list[dict], tools: list | None = None) -> BaseCacheEntry | None:
        """Find a matching base cache for the given messages' system prompt."""
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return self._base_caches[h]
        return None

    def _extract_system_tokens(
        self, messages: list[dict], full_tokens: list[int],
        tools: list | None = None, thinking: bool | None = None,
    ) -> list[int] | None:
        """Extract system prompt tokens from the full tokenized messages.

        Tokenizes [system + dummy user] then subtracts a [dummy user only]
        tokenization to get pure system tokens. Verifies they are a prefix
        of full_tokens.
        """
        has_system = messages and messages[0].get("role") in ("system", "developer")
        if not has_system and not self._has_rotating_cache:
            return None
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return None  # already registered
        try:
            enable_thinking = thinking if thinking is not None else self.cfg.enable_thinking
            if has_system:
                system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
            else:
                system_with_dummy = [{"role": "user", "content": "hi"}]
            # Tokenize system + dummy user
            full_with_dummy = self._tokenize_prompt(
                system_with_dummy, tools=tools, thinking=enable_thinking,
            )
            # Tokenize just dummy user (to strip)
            dummy_only = self._tokenize_prompt(
                [{"role": "user", "content": "hi"}], tools=tools, thinking=enable_thinking,
            )
            # System tokens = full_with_dummy minus the dummy user suffix
            # Heuristic: find where full_with_dummy diverges from dummy_only (from end)
            # Simpler: system tokens are the leading tokens that differ from dummy_only
            if has_system and len(full_with_dummy) > len(dummy_only):
                system_tokens = full_with_dummy[: len(full_with_dummy) - len(dummy_only)]
                # For models with auto-system prefix, adjust
                if not system_tokens:
                    return None
                # Verify prefix of full tokens
                if full_tokens[:len(system_tokens)] == system_tokens:
                    return system_tokens
            elif not has_system and len(full_with_dummy) > 0:
                # Models like Gemma 4 may auto-generate system prefix
                # The full_with_dummy == dummy_only in this case, check if there's
                # a shared prefix with the full conversation tokens
                return None
        except Exception as e:
            logger.warning(f"[Base Cache] Failed to extract system tokens: {e}")
        return None

    def _register_base_cache(
        self, messages: list[dict], cache: list, system_tokens: list[int], tools: list | None = None
    ):
        """Register a base cache from the system prompt portion of a processed cache."""
        h = self._system_hash(messages, tools=tools)
        if not h or h in self._base_caches:
            return
        # Deep copy the cache at current state (after system prompt processing)
        import copy
        base_snapshot = copy.deepcopy(cache)
        self._eval_cache(base_snapshot)
        entry = BaseCacheEntry(
            system_hash=h,
            cache=base_snapshot,
            tokens=system_tokens,
            token_count=len(system_tokens),
        )
        self._base_caches[h] = entry
        logger.debug(
            f"[Base Cache] REGISTERED | hash={h} | {len(system_tokens)} tokens | "
            f"pool_size={len(self._base_caches)}"
        )

    def _clone_base_cache(self, base: BaseCacheEntry) -> list:
        """Clone a base cache for a new session."""
        import copy
        cloned = copy.deepcopy(base.cache)
        # Force evaluation of cloned arrays to avoid lazy-eval aliasing issues
        self._eval_cache(cloned)
        base.hit_count += 1
        logger.debug(
            f"[Base Cache] CLONE | hash={base.system_hash} | "
            f"{base.token_count} tokens | hits={base.hit_count}"
        )
        return cloned

    def base_cache_stats(self) -> list[dict]:
        """Return stats for all base caches."""
        return [
            {
                "system_hash": e.system_hash,
                "token_count": e.token_count,
                "hit_count": e.hit_count,
                "created": e.created,
            }
            for e in self._base_caches.values()
        ]

    @staticmethod
    def _get_cache_offset(cache: list) -> int:
        """Get the total number of tokens processed by this cache.

        Prefers KVCache (full attention, accurate cumulative offset) over
        RotatingKVCache (offset is cumulative but size() caps at max_size).
        """
        # First pass: look for unbounded KVCache (full attention layers)
        for c in cache:
            if type(c).__name__ == "KVCache" and hasattr(c, "offset"):
                return c.offset
        # Fallback: any cache with offset (RotatingKVCache, ArraysCache, etc.)
        for c in cache:
            if hasattr(c, "offset"):
                return c.offset
        return 0

    @staticmethod
    def _eval_cache(cache: list):
        """Force evaluation of all lazy cache tensors."""
        arrays = []
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                items = c.keys if isinstance(c.keys, list) else [c.keys]
                arrays.extend(a for a in items if isinstance(a, mx.array))
            if hasattr(c, "values") and c.values is not None:
                items = c.values if isinstance(c.values, list) else [c.values]
                arrays.extend(a for a in items if isinstance(a, mx.array))
            if hasattr(c, "state") and c.state is not None:
                items = c.state if isinstance(c.state, list) else [c.state]
                arrays.extend(a for a in items if isinstance(a, mx.array))
        if arrays:
            mx.eval(*arrays)

    _PREFILL_STEP = 512

    def _prefill_cache(self, cache: list, tokens: list[int]):
        """Process tokens through the language model to populate a KV cache."""
        arr = mx.array(tokens)
        for i in range(0, len(tokens), self._PREFILL_STEP):
            chunk = arr[i : i + self._PREFILL_STEP]
            self._language_model(chunk[None], cache=cache)
        self._eval_cache(cache)

    @staticmethod
    def _is_compacted_tool(s_content: str, i_content: str) -> bool:
        """Check if either side is a compacted/cleared tool result placeholder."""
        for c in (s_content, i_content):
            if c.startswith("[") and ("cleared]" in c or "compacted:" in c):
                return True
        return False

    @staticmethod
    def _flatten_multipart(content) -> str:
        """Flatten OpenAI multi-part content to a plain string.

        Drops image/video parts and client-inserted
        "[image data removed ...]" placeholders so that a turn with an
        image blob and a subsequent turn where the client replaced the
        blob with a placeholder normalize to the same text.
        """
        import re
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype and ptype != "text":
                continue  # image, image_url, video, etc.
            txt = p.get("text", "") or ""
            if re.match(r"\s*\[image data removed", txt, flags=re.IGNORECASE):
                continue
            parts.append(txt)
        return "\n".join(parts)

    @staticmethod
    def _normalize_for_match(content, role: str) -> str:
        """Normalize message content for comparison."""
        import re
        content = MLXEngine._flatten_multipart(content)
        if role == "system":
            # Normalize dynamic date (e.g. "Today's date: Tue Mar 10 2026" → placeholder)
            content = re.sub(
                r"Today's date:\s*\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4}",
                "Today's date: __DATE__",
                content,
            )
        # Strip <system-reminder>...</system-reminder> tags injected dynamically by clients
        content = re.sub(
            r"\n?<system-reminder>.*?</system-reminder>",
            "",
            content,
            flags=re.DOTALL,
        )
        # Strip thinking and tool calls from assistant messages for comparison.
        # Only the actual text content matters for cache matching.
        if role == "assistant":
            # Strip thinking blocks
            if "<channel|>" in content:
                content = content[content.rindex("<channel|>") + len("<channel|>"):]
            elif "</think>" in content:
                content = content[content.rindex("</think>") + len("</think>"):]
            else:
                content = re.sub(r"^<think>\n?", "", content)
                content = re.sub(r"^<\|channel>thought\n?", "", content)
                content = re.sub(r"^thought\n", "", content)
            # Strip tool call blocks (both ChatML and Gemma 4 formats)
            content = re.sub(r"<\|?tool_call>.*?<\|?tool_call\|?>", "", content, flags=re.DOTALL)
            content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        return content.strip()

    def _messages_match(self, stored: list[dict], incoming: list[dict]) -> bool:
        """Check if incoming messages start with the stored conversation."""
        if len(incoming) < len(stored):
            logger.debug(
                f"[Match] FAIL: incoming({len(incoming)}) < stored({len(stored)})"
            )
            return False
        for i, s_msg in enumerate(stored):
            i_msg = incoming[i]
            if s_msg.get("role") != i_msg.get("role"):
                logger.info(
                    f"[Match] FAIL at msg[{i}]: role {s_msg.get('role')!r} != {i_msg.get('role')!r}"
                )
                return False
            s_content = self._flatten_multipart(s_msg.get("content"))
            i_content = self._flatten_multipart(i_msg.get("content"))
            role = s_msg.get("role", "")

            # Assistant tool_call messages: OpenCode strips <tool_call> from content
            # and moves it to tool_calls field. Handle all cases:
            # 1. stored="<tool_call>..." vs incoming="" (pure tool call)
            # 2. stored="text\n\n<tool_call>..." vs incoming="text" (text + tool call)
            if role == "assistant" and s_content != i_content:
                import re
                tc_pattern = r"\n*<tool_call>"
                s_stripped = re.split(tc_pattern, s_content, maxsplit=1)[0].rstrip()
                i_stripped = re.split(tc_pattern, i_content, maxsplit=1)[0].rstrip()
                if s_stripped == i_stripped:
                    logger.debug(
                        f"[Match] msg[{i}] assistant tool_call content mismatch ignored "
                        f"(stored_len={len(s_content)}, incoming_len={len(i_content)})"
                    )
                    continue

            # Normalize and compare
            s_norm = self._normalize_for_match(s_content, role)
            i_norm = self._normalize_for_match(i_content, role)
            if s_norm != i_norm:
                # Tool content compacted/cleared by client (either direction)
                # KV cache still valid — the tokens were already processed.
                if role == "tool" and self._is_compacted_tool(s_content, i_content):
                    logger.debug(
                        f"[Match] msg[{i}] tool content compacted — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Last stored assistant message: tolerate content difference.
                # Client may have received truncated/reformatted response
                # (disconnect, streaming, client-side processing).
                # KV cache is still valid because it was saved from full generation.
                if role == "assistant" and i == len(stored) - 1:
                    logger.debug(
                        f"[Match] msg[{i}] assistant content mismatch at last stored msg — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Client may reconstruct assistant content as "{thinking}\n\n{final}"
                # without <think> tags. Stored normalized is just "{final}"; incoming
                # normalized is "{thinking}\n\n{final}". KV cache reflects what the
                # model actually processed (stored), so if the final answer matches
                # as a suffix of the incoming, the cache is still valid.
                if role == "assistant" and s_norm and len(s_norm) >= 8 and (
                    i_norm.endswith(s_norm) or s_norm.endswith(i_norm)
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant content suffix match — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Assistant tool_call turn: stored was (optional <think>...</think>
                # + <tool_call>...</tool_call>), which normalizes to empty because
                # both blocks are stripped. OpenAI-format clients move the tool
                # call to tool_calls[] and may reconstruct the thinking as plain
                # text in content (no <think> tags). KV cache is still valid —
                # it reflects the tokens the model actually emitted.
                if (
                    role == "assistant"
                    and not s_norm
                    and i_msg.get("tool_calls")
                    and ("<tool_call>" in s_content or s_msg.get("tool_calls"))
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant tool_call turn — "
                        f"accepting reconstructed content "
                        f"(stored_len={len(s_content)}, incoming_len={len(i_content)})"
                    )
                    continue

                # Find exact divergence point
                diff_pos = next(
                    (j for j in range(min(len(s_norm), len(i_norm))) if s_norm[j] != i_norm[j]),
                    min(len(s_norm), len(i_norm)),
                )
                s_ctx = s_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                i_ctx = i_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                logger.info(
                    f"[Match] FAIL at msg[{i}] role={role}: "
                    f"stored_len={len(s_content)} vs incoming_len={len(i_content)} | "
                    f"diff at char {diff_pos} | "
                    f"stored=...{s_ctx!r}... | incoming=...{i_ctx!r}..."
                )
                return False
        return True

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages for chat template: fix roles, flatten content."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            if role == "developer":
                role = "system"
            m = {"role": role}
            if msg.get("content") is not None:
                content = msg["content"]
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(part["text"])
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join(parts)
                m["content"] = content
            if msg.get("tool_calls"):
                normalized_tcs = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc_copy, dict) and "function" in tc_copy:
                        fn = dict(tc_copy["function"])
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except (json.JSONDecodeError, ValueError):
                                fn["arguments"] = {}
                        tc_copy["function"] = fn
                    normalized_tcs.append(tc_copy)
                m["tool_calls"] = normalized_tcs
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            formatted.append(m)
        return formatted

    def _build_prompt_text(
        self, messages: list[dict], thinking: bool = True, tools: list | None = None,
    ) -> str:
        """Build prompt text from messages using chat template (tokenize=False)."""
        formatted = self._format_messages(messages)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": thinking,
        }
        if tools:
            kwargs["tools"] = [
                t.model_dump() if hasattr(t, "model_dump") else t for t in tools
            ]
        return self.tokenizer.apply_chat_template(formatted, **kwargs)

    def _tokenize_prompt(
        self, messages: list[dict], thinking: bool = True, tools: list | None = None,
    ) -> list[int]:
        """Tokenize messages using chat template (tokenize=True)."""
        formatted = self._format_messages(messages)
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": thinking,
        }
        if tools:
            kwargs["tools"] = [
                t.model_dump() if hasattr(t, "model_dump") else t for t in tools
            ]
        result = self.tokenizer.apply_chat_template(formatted, **kwargs)
        # Some tokenizers return BatchEncoding instead of list[int]
        if hasattr(result, "input_ids"):
            return list(result.input_ids)
        return list(result)

    def _suffix_tokens(
        self, new_messages: list[dict], thinking: bool = True,
    ) -> list[int]:
        """Compute suffix tokens for new messages to append to stored token_ids.

        This avoids full re-tokenization (which breaks special token round-trip)
        by directly encoding only the new message suffix in model-specific format.
        """
        if self.model_family == "gemma4":
            return self._suffix_tokens_gemma4(new_messages, thinking)
        if self.model_family == "glm":
            return self._suffix_tokens_glm(new_messages, thinking)
        return self._suffix_tokens_chatml(new_messages, thinking)

    def _suffix_tokens_gemma4(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """Gemma 4 suffix: <turn|>\\n<|turn>user\\n{content}<turn|>\\n<|turn>model\\n"""
        parts = ["<turn|>"]
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue  # already in cache
            elif role == "tool":
                parts.append(
                    f"\n<|turn>user\n<|tool_response>\n"
                    f"response:{msg.get('name', '')}{{{content}}}\n"
                    f"<tool_response|><turn|>"
                )
            else:
                parts.append(f"\n<|turn>user\n{content}<turn|>")
        parts.append("\n<|turn>model\n")
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def _suffix_tokens_chatml(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """ChatML suffix: \\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"""
        parts = []
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue
            elif role == "tool":
                parts.append(
                    f"\n<|im_start|>user\n<tool_response>\n"
                    f"{content}\n</tool_response><|im_end|>"
                )
            else:
                parts.append(f"\n<|im_start|>user\n{content}<|im_end|>")
        gen_prompt = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        parts.append(gen_prompt)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def _suffix_tokens_glm(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """GLM suffix: <|user|>{content}<|assistant|><think>"""
        parts = []
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue
            elif role == "tool":
                parts.append(f"<|user|><tool_response>\n{content}\n</tool_response>")
            else:
                parts.append(f"<|user|>{content}")
        gen_prompt = "<|assistant|><think>" if thinking else "<|assistant|>"
        parts.append(gen_prompt)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def generate_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        session_id: str | None = None,
        tools: list | None = None,
        cancel_event: threading.Event | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> Generator[GenerationResult, None, None]:
        """Generate with session-based KV cache reuse (holds lock)."""
        sid = session_id or "anon"
        t_wait = time.perf_counter()
        logger.debug(f"[Queue] session={sid} | waiting for lock | messages={len(messages)}")
        with self._lock:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            logger.debug(f"[Queue] session={sid} | lock acquired | waited={wait_ms:.0f}ms")
            yield GenerationResult(status="generating")
            yield from self._generate_locked(
                messages,
                max_tokens=max_tokens if max_tokens is not None else self.cfg.default_max_tokens,
                temperature=temperature if temperature is not None else self.cfg.default_temperature,
                top_p=top_p if top_p is not None else self.cfg.default_top_p,
                min_p=min_p if min_p is not None else self.cfg.default_min_p,
                top_k=top_k if top_k is not None else self.cfg.default_top_k,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cfg.default_repetition_penalty,
                session_id=session_id,
                tools=tools,
                cancel_event=cancel_event,
                thinking=thinking,
                thinking_budget=thinking_budget,
                response_format=response_format,
            )

    def _generate_locked(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        session_id: str | None,
        tools: list | None,
        cancel_event: threading.Event | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        response_format=None,
    ) -> Generator[GenerationResult, None, None]:
        """Core generation logic using mlx-vlm (must hold lock).

        Session messages include thinking in assistant content so that
        PromptCacheState prefix matching covers generated tokens from
        previous turns.
        """
        self._touch_gpu()

        has_tools = bool(tools)
        use_thinking = thinking if thinking is not None else self.cfg.enable_thinking

        if not session_id:
            session_id = "anon"
        session = self._sessions.get(session_id)

        # Try loading from disk if not in memory
        if not session and self._has_disk_cache(session_id):
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session

        cache_mode = "miss"

        # Determine prompt messages:
        # - On cache hit: use session's stored messages (with thinking) + new messages
        # - On cache miss: use incoming messages as-is
        prompt_messages = messages
        cache_state = PromptCacheState()

        if (
            session
            and session.cache_state is not None
            and session.cache_state.cache is not None
            and self._messages_match(session.messages, messages)
        ):
            new_messages = messages[len(session.messages):]
            if not new_messages:
                # Retry: discard cache, start fresh
                cache_mode = "retry"
                prompt_messages = messages
                cache_state = PromptCacheState()
                logger.info(
                    f"[KV Cache] session={session_id} | RETRY | "
                    f"discarding cache, re-processing {len(messages)} messages"
                )
            else:
                # Cache hit: extend stored token_ids with suffix for new messages.
                # This avoids full re-tokenization which breaks special token
                # round-trip (e.g. Gemma 4 <|channel>/<channel|>).
                cache_mode = "hit"
                cache_state = session.cache_state
                cached_tokens = session.total_cache_tokens
                suffix = self._suffix_tokens(new_messages, thinking=use_thinking)
                prompt_token_ids = list(cache_state.token_ids or []) + suffix
                logger.info(
                    f"[KV Cache] session={session_id} | HIT | "
                    f"reusing {cached_tokens} cached tokens + "
                    f"{len(suffix)} suffix tokens"
                )
        else:
            # Cache miss — seed from base cache if available
            prev_cached = session.total_cache_tokens if session else 0
            base = self._find_base_cache(messages, tools=tools)
            if base:
                cache_state.cache = self._clone_base_cache(base)
                cache_state.token_ids = list(base.tokens)
                cache_mode = "base_hit"
                logger.info(
                    f"[KV Cache] session={session_id} | BASE HIT | "
                    f"seeding {base.token_count} base tokens "
                    f"(was {prev_cached} cached)"
                )
            else:
                if prev_cached:
                    logger.info(
                        f"[KV Cache] session={session_id} | MISS | "
                        f"discarding {prev_cached} cached tokens, "
                        f"processing {len(messages)} messages"
                    )
                else:
                    logger.info(
                        f"[KV Cache] session={session_id} | MISS (new) | "
                        f"processing {len(messages)} messages"
                    )

        # Tokenize prompt — only needed for MISS/RETRY (HIT already computed above)
        if cache_mode != "hit":
            prompt_token_ids = self._tokenize_prompt(
                prompt_messages, thinking=use_thinking, tools=tools,
            )
        total_prompt_tokens = len(prompt_token_ids)

        # Determine how many tokens will actually be processed (for cache info)
        if cache_state.token_ids:
            reused = cache_state.find_prefix_length(prompt_token_ids)
        else:
            reused = 0
        new_token_count = total_prompt_tokens - reused

        # Build logits processors
        logits_processors = []
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyProcessor(penalty=repetition_penalty))
        budget = thinking_budget if thinking_budget is not None else self.cfg.thinking_budget
        if use_thinking and budget > 0 and self.cfg.think_end_token >= 0:
            think_start = _detect_token_id(
                self.tokenizer,
                "<|channel>" if self.model_family == "gemma4" else "<think>",
            )
            logits_processors.append(ThinkingBudgetProcessor(
                budget=budget,
                think_end_token=self.cfg.think_end_token,
                think_start_token=think_start,
                model_family=self.model_family,
            ))
        # Structured output (response_format) via FSM-based logits masking.
        # Works on both mlx-vlm and mlx-lm paths (same logits_processors contract).
        # PLD is incompatible: speculative decoding advances multiple tokens
        # per step, breaking the FSM's single-step advance assumption.
        structured_proc = None
        rf_type = getattr(response_format, "type", None) if response_format else None
        if rf_type in ("json_schema", "json_object") and self.cfg.pld_enabled and not self._use_vlm:
            logger.warning(
                f"[Structured] response_format={rf_type} disabled: PLD is active "
                f"(speculative decoding is incompatible with FSM-based constraints). "
                f"Disable PLD to use structured output."
            )
        elif rf_type in ("json_schema", "json_object"):
            try:
                from mlx_soloheaven.engine.structured import (
                    build_json_schema_processor,
                    build_json_object_processor,
                )
                if rf_type == "json_schema":
                    js = response_format.json_schema
                    if js is None or js.schema_ is None:
                        raise ValueError("response_format.json_schema.schema is required")
                    structured_proc = build_json_schema_processor(
                        js.schema_, self.tokenizer,
                        cache_key=f"{js.name or 'anon'}:{hash(json.dumps(js.schema_, sort_keys=True))}",
                    )
                    logger.info(f"[Structured] json_schema active (name={js.name})")
                else:  # json_object
                    structured_proc = build_json_object_processor(self.tokenizer)
                    logger.info(f"[Structured] json_object active")
            except Exception as e:
                logger.warning(f"[Structured] failed to build processor: {e} — ignoring")
                structured_proc = None
        if structured_proc is not None:
            logits_processors.append(structured_proc)

        # Build response cache info
        response_cache_info = {
            "cache_mode": cache_mode,
            "cached_tokens": reused,
            "new_tokens": new_token_count,
            "total_prompt_tokens": total_prompt_tokens,
        }
        if session and session.pending_build_time > 0:
            response_cache_info["build_time"] = round(session.pending_build_time, 2)
            session.pending_build_time = 0.0

        sampling_info = f"temp={temperature}"
        if top_p < 1.0:
            sampling_info += f", top_p={top_p}"
        if min_p > 0.0:
            sampling_info += f", min_p={min_p}"
        if top_k > 0:
            sampling_info += f", top_k={top_k}"
        if repetition_penalty != 1.0:
            sampling_info += f", rep_pen={repetition_penalty}"
        logger.info(
            f"[Generate] prompt={new_token_count} new tokens "
            f"(reused={reused}, total={total_prompt_tokens}), "
            f"max={max_tokens}, {sampling_info}, cache_mode={cache_mode}"
        )

        # Stream generate — mlx-vlm for VLM models, mlx-lm for text models
        accumulated_text = ""
        t_gen_start = time.perf_counter()
        t_first_token = None
        last_prompt_tps = 0.0
        last_gen_tps = 0.0
        gen_token_count = 0
        cancelled = False

        if self._use_vlm:
            input_ids = mx.array([prompt_token_ids])
            gen_iter = vlm_stream_generate(
                self._vlm_model,
                self._processor,
                "",  # prompt text ignored when input_ids is provided
                input_ids=input_ids,
                prompt_cache_state=cache_state,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                prefill_step_size=self.cfg.prefill_step_size,
                logits_processors=logits_processors if logits_processors else None,
            )
        else:
            # mlx-lm path: manage cache manually via prompt_cache
            full_prompt_token_ids = list(prompt_token_ids)  # save before trimming
            prompt_cache = cache_state.cache
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self._language_model)
            else:
                # Trim cache to match prefix
                stored_ids = cache_state.token_ids or []
                prefix_len = 0
                for j in range(min(len(stored_ids), len(prompt_token_ids))):
                    if stored_ids[j] != prompt_token_ids[j]:
                        break
                    prefix_len = j + 1
                # Trim KV cache to prefix length
                for c in prompt_cache:
                    if hasattr(c, "keys") and c.keys is not None:
                        cached_len = c.keys.shape[2] if len(c.keys.shape) > 2 else 0
                        if cached_len > prefix_len:
                            c.keys = c.keys[..., :prefix_len, :]
                            c.values = c.values[..., :prefix_len, :]
                            if hasattr(c, "offset"):
                                c.offset = prefix_len
                # Only feed tokens after prefix
                prompt_token_ids = prompt_token_ids[prefix_len:]

            sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)
            lm_kwargs = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "prompt_cache": prompt_cache,
                "prefill_step_size": self.cfg.prefill_step_size,
                "logits_processors": logits_processors if logits_processors else None,
            }
            if self.cfg.kv_bits:
                lm_kwargs["kv_bits"] = self.cfg.kv_bits
                lm_kwargs["kv_group_size"] = self.cfg.kv_group_size
                lm_kwargs["quantized_kv_start"] = self.cfg.quantized_kv_start
            # PLD requires trimmable cache (for rollback on rejection).
            # Models with ArraysCache layers (Qwen3.5 DeltaNet, etc.) are NOT
            # trimmable — fall back to regular lm_stream_generate.
            use_pld = self.cfg.pld_enabled
            if use_pld:
                from mlx_lm.models.cache import can_trim_prompt_cache
                if not can_trim_prompt_cache(prompt_cache):
                    if not getattr(self, "_pld_incompat_warned", False):
                        logger.warning(
                            f"[{self.model_id}] PLD disabled: model uses "
                            f"non-trimmable cache (e.g. ArraysCache/DeltaNet). "
                            f"Falling back to standard generation."
                        )
                        self._pld_incompat_warned = True
                    use_pld = False

            if use_pld:
                from mlx_soloheaven.engine.pld import pld_generate_step
                gen_iter = _pld_response_adapter(
                    pld_generate_step(
                        prompt=mx.array(prompt_token_ids),
                        model=self._language_model,
                        num_draft_tokens=self.cfg.pld_num_draft_tokens,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors if logits_processors else None,
                        prompt_cache=prompt_cache,
                        prefill_step_size=self.cfg.prefill_step_size,
                        kv_bits=self.cfg.kv_bits if self.cfg.kv_bits else None,
                        kv_group_size=self.cfg.kv_group_size,
                        quantized_kv_start=self.cfg.quantized_kv_start,
                        ngram_k=self.cfg.pld_ngram_k,
                    ),
                    tokenizer=self.tokenizer,
                )
            else:
                gen_iter = lm_stream_generate(
                    self._language_model,
                    self.tokenizer,
                    prompt=prompt_token_ids,
                    **lm_kwargs,
                )

        # Configurable progress log interval (tokens between INFO snapshots)
        progress_interval = 50  # tokens — every ~2s at 25 TPS
        for resp in gen_iter:
            if cancel_event and cancel_event.is_set():
                # Report last token state when cancelled so we can see
                # where generation was when the client disconnected.
                tail = accumulated_text[-200:].replace('\n', '\\n')
                logger.info(
                    f"[Generate] session={session_id} | CANCELLED at token {gen_token_count} | "
                    f"last_tps={last_gen_tps:.1f} | tail={tail!r}"
                )
                cancelled = True
                break

            gen_token_count += 1
            text = resp.text if hasattr(resp, "text") else ""
            token = (resp.token if hasattr(resp, "token") and resp.token is not None
                     else 0)
            prompt_tps = getattr(resp, "prompt_tps", 0.0) or 0.0
            gen_tps = getattr(resp, "generation_tps", 0.0) or 0.0

            accumulated_text += text

            if t_first_token is None:
                t_first_token = time.perf_counter()
                last_prompt_tps = prompt_tps
                logger.info(
                    f"[Generate] TTFT={round((t_first_token - t_gen_start)*1000)}ms"
                )

            # Per-token DEBUG log (very verbose — only when --verbose)
            logger.debug(
                f"[Token] session={session_id} | n={gen_token_count} id={token} text={text!r}"
            )

            # Periodic INFO snapshot (every 50 tokens) so we can see progress
            # when verbose is off
            if gen_token_count % progress_interval == 0:
                tail = accumulated_text[-120:].replace('\n', '\\n')
                logger.info(
                    f"[Generate] session={session_id} | "
                    f"tokens={gen_token_count} | tps={gen_tps:.1f} | "
                    f"tail={tail!r}"
                )

            last_gen_tps = gen_tps

            yield GenerationResult(
                text=text,
                token=token,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=gen_token_count,
                prompt_tps=prompt_tps,
                generation_tps=gen_tps,
            )

        self._touch_gpu()

        # Log generated text for debugging
        if accumulated_text:
            preview = accumulated_text[:200].replace('\n', '\\n')
            logger.debug(
                f"[Generate] session={session_id} | "
                f"tokens={gen_token_count} | cancelled={cancelled} | "
                f"text={preview!r}"
            )

        if cancelled:
            return

        # Guard: detect empty response (no content after thinking)
        if accumulated_text and session_id:
            _, content = split_thinking_and_content(
                accumulated_text, model_family=self.model_family,
            )
            if not content or not content.strip():
                logger.warning(
                    f"[KV Cache] session={session_id} | SKIP SAVE | "
                    f"empty response ({gen_token_count} tokens, no content)"
                )
                yield GenerationResult(
                    text="",
                    finish_reason="stop",
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=gen_token_count,
                )
                return

        # For mlx-lm path: manually update PromptCacheState
        # (mlx-vlm does this automatically in stream_generate)
        if not self._use_vlm and not cancelled:
            cache_state.cache = prompt_cache
            cache_state.token_ids = full_prompt_token_ids  # full IDs before trim

        # Parse tool_calls once — used both for session persistence and
        # for the terminal GenerationResult's finish_reason.
        parsed_tool_calls: list[dict] = []
        if has_tools and accumulated_text:
            _, parsed_tool_calls = parse_tool_calls(
                accumulated_text, model_family=self.model_family,
            )

        # Save session
        if session_id:
            new_offset = self._get_cache_offset(cache_state.cache) if cache_state.cache else 0
            # Fallback: some models (GLM MoE) don't expose offset in cache objects
            if new_offset == 0 and cache_state.token_ids:
                new_offset = len(cache_state.token_ids)
            prev_offset = session.total_cache_tokens if session else 0

            # Build full assistant content for engine-internal messages
            # This includes thinking so next turn's suffix extends correctly
            full_assistant_content = self._make_full_assistant_content(
                accumulated_text, use_thinking,
            )

            if parsed_tool_calls:
                # Strip the tool_call XML from stored content so the template
                # doesn't double-render (content + tool_calls both emit XML).
                start_tag, _ = get_tool_markers(self.model_family)
                tc_idx = full_assistant_content.find(start_tag)
                if tc_idx >= 0:
                    full_assistant_content = full_assistant_content[:tc_idx].rstrip()

            # On HIT: extend session.messages with new incoming + assistant
            # On MISS: use incoming messages + assistant
            if cache_mode == "hit" and session:
                base_messages = list(session.messages) + new_messages
            else:
                base_messages = list(messages)
            assistant_msg: dict = {
                "role": "assistant",
                "content": full_assistant_content,
            }
            if parsed_tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in parsed_tool_calls
                ]
            updated_messages = base_messages + [assistant_msg]

            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=updated_messages,
                total_cache_tokens=new_offset,
            )

            logger.debug(
                f"[KV Cache] session={session_id} | SAVED | "
                f"offset: {prev_offset} -> {new_offset} tokens "
                f"(+{new_offset - prev_offset})"
            )

        # Auto-register base cache on miss
        if cache_mode in ("miss", "retry") and messages:
            self._maybe_register_base_cache(
                messages, prompt_token_ids, tools=tools, thinking=use_thinking,
            )

        # Determine finish reason (parsed_tool_calls computed above)
        finish_reason = "tool_calls" if parsed_tool_calls else "stop"

        yield GenerationResult(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=gen_token_count,
            prompt_tps=last_prompt_tps,
            generation_tps=last_gen_tps,
            cache_info=response_cache_info,
        )

    def _make_full_assistant_content(
        self, accumulated_text: str, thinking_enabled: bool,
    ) -> str:
        """Build full assistant content for engine-internal messages.

        Includes thinking markers so that suffix token computation works
        correctly on subsequent turns.

        ChatML/GLM: prompt suffix includes '<think>\\n' (or '<think>'), so
        accumulated_text starts after it. Prepend to get the complete content.

        Gemma 4: model generates thinking markers itself (e.g. '<|channel>thought\\n'),
        so accumulated_text already includes them.
        """
        if self.model_family == "gemma4":
            return accumulated_text
        # ChatML and GLM both use <think> prefix
        if thinking_enabled:
            prefix = "<think>" if self.model_family == "glm" else "<think>\n"
            return prefix + accumulated_text
        return accumulated_text

    def _maybe_register_base_cache(
        self,
        messages: list[dict],
        prompt_tokens: list[int],
        tools: list | None = None,
        thinking: bool = True,
    ):
        """Register a base cache for the system prompt if not already cached."""
        has_system_or_rotating = (
            (messages and messages[0].get("role") in ("system", "developer"))
            or self._has_rotating_cache
        )
        if not has_system_or_rotating:
            return
        sys_hash = self._system_hash(messages, tools=tools)
        if not sys_hash or sys_hash in self._base_caches:
            return
        system_tokens = self._extract_system_tokens(
            messages, prompt_tokens, tools=tools, thinking=thinking,
        )
        if system_tokens and len(system_tokens) < len(prompt_tokens):
            try:
                base_cache = make_prompt_cache(self._language_model)
                self._prefill_cache(base_cache, system_tokens)
                self._register_base_cache(
                    messages, base_cache, system_tokens, tools=tools,
                )
            except Exception as e:
                logger.warning(f"[Base Cache] registration failed: {e}")

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        tools: list | None = None,
        session_id: str | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> CompletionResult:
        """Non-streaming completion."""
        all_text = []
        result = CompletionResult()

        for chunk in self.generate_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            session_id=session_id,
            tools=tools,
            thinking=thinking,
            thinking_budget=thinking_budget,
            response_format=response_format,
        ):
            if chunk.text:
                all_text.append(chunk.text)
            if chunk.finish_reason:
                result.finish_reason = chunk.finish_reason
                result.prompt_tokens = chunk.prompt_tokens
                result.completion_tokens = chunk.completion_tokens
                result.prompt_tps = chunk.prompt_tps
                result.generation_tps = chunk.generation_tps
                result.cache_info = chunk.cache_info

        full_text = "".join(all_text)
        thinking, content = split_thinking_and_content(full_text, model_family=self.model_family)
        result.thinking = thinking

        if tools:
            text_part, tool_calls = parse_tool_calls(content, model_family=self.model_family)
            if tool_calls:
                result.tool_calls = tool_calls
                result.content = text_part if text_part else None
                result.finish_reason = "tool_calls"
            else:
                result.content = content
        else:
            result.content = content

        return result

    def update_session_messages(self, session_id: str, messages: list[dict]):
        """Touch session and mark dirty after external caller finalizes messages.

        Note: The engine manages its own internal messages (with thinking) in
        _generate_locked. This method exists for the API layer to signal that
        generation is complete and the session should be persisted.
        The incoming messages parameter is ignored — internal messages are
        authoritative for cache matching.
        """
        session = self._sessions.get(session_id)
        if session:
            session.touch()
            logger.info(
                f"[Session] {session_id} | messages finalized | "
                f"{len(session.messages)} msgs, {session.total_cache_tokens} cached tokens"
            )
            self._mark_dirty(session_id)

    def compact_session(self, session_id: str, messages: list[dict]) -> dict:
        """Replace a session's messages and rebuild KV cache from scratch.

        Used when client compresses/summarizes conversation context.
        """
        with self._lock:
            self._touch_gpu()
            t0 = time.perf_counter()

            prompt_tokens = self._tokenize_prompt(messages)

            # Try base cache first
            base = self._find_base_cache(messages)
            base_tokens_used = 0
            prompt_cache = None
            if base and len(prompt_tokens) >= base.token_count:
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    feed_tokens = prompt_tokens[base.token_count:]
                    base_tokens_used = base.token_count
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self._language_model)
                feed_tokens = prompt_tokens

            if feed_tokens:
                self._prefill_cache(prompt_cache, feed_tokens)
            self._eval_cache(prompt_cache)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            # Build PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = prompt_cache
            cache_state.token_ids = prompt_tokens

            prev = self._sessions.get(session_id)
            prev_tokens = prev.total_cache_tokens if prev else 0
            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=new_offset,
            )

            logger.info(
                f"[Compact] session={session_id} | "
                f"{prev_tokens} -> {new_offset} tokens | "
                f"base={base_tokens_used} | "
                f"processed={len(feed_tokens)} tokens | "
                f"{elapsed:.2f}s"
            )

            # Auto-register base cache
            self._maybe_register_base_cache(messages, prompt_tokens)

            self._mark_dirty(session_id)

            return {
                "session_id": session_id,
                "status": "ok",
                "cached_tokens": new_offset,
                "previous_tokens": prev_tokens,
                "base_tokens": base_tokens_used,
                "processing_time_ms": round(elapsed * 1000),
            }

    def list_sessions(self) -> list[dict]:
        """List all active sessions."""
        result = []
        for sid, s in self._sessions.items():
            result.append({
                "session_id": sid,
                "messages": len(s.messages),
                "cache_tokens": s.total_cache_tokens,
                "last_used": s.last_used,
            })
        return sorted(result, key=lambda x: x["last_used"], reverse=True)

    def get_session(self, session_id: str) -> dict | None:
        """Get details for a specific session."""
        s = self._sessions.get(session_id)
        if not s:
            return None
        return {
            "session_id": session_id,
            "messages": len(s.messages),
            "cache_tokens": s.total_cache_tokens,
            "last_used": s.last_used,
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its cache."""
        with self._dirty_lock:
            self._dirty_sessions.discard(session_id)
        if session_id in self._sessions:
            del self._sessions[session_id]
        path = self._session_cache_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.discard(session_id)
        logger.info(f"[Session] DELETED | session={session_id}")
        return True

    async def generate_stream_async(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        session_id: str | None = None,
        tools: list | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> AsyncGenerator[GenerationResult, None]:
        """Async wrapper for generate_stream. Supports client disconnect cancellation."""
        loop = asyncio.get_event_loop()
        q: asyncio.Queue[GenerationResult | None | Exception] = asyncio.Queue()
        cancel_event = threading.Event()

        def _run():
            try:
                for result in self.generate_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    session_id=session_id,
                    tools=tools,
                    cancel_event=cancel_event,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    response_format=response_format,
                ):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, result)
            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Yield empty result as keepalive during prompt processing
                    yield GenerationResult(text="")
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except (asyncio.CancelledError, GeneratorExit) as exc:
            cancel_event.set()
            # INFO-level so we always see disconnects (debugging client timeouts etc.)
            logger.info(
                f"[Stream] session={session_id} | client disconnected "
                f"({type(exc).__name__}) — cancelling generation"
            )
            raise

    # --- Truncation & Regeneration ---

    def branch_from_turn(
        self,
        source_session_id: str,
        new_session_id: str,
        branch_turn: int,
        branch_messages: list[dict] | None = None,
    ) -> dict:
        """Branch a new session by building cache from scratch."""
        source = self._sessions.get(source_session_id)
        if not source and self._has_disk_cache(source_session_id):
            source = self._load_session_from_disk(source_session_id)
            if source:
                self._sessions[source_session_id] = source

        if source:
            engine_messages = source.messages[:branch_turn]
        elif branch_messages:
            engine_messages = branch_messages
        else:
            return {"error": "source session not found and no messages provided"}

        return self._rebuild_session(new_session_id, engine_messages)

    def prepare_regenerate(self, session_id: str) -> dict:
        """Remove last assistant message and restore cache."""
        session = self._sessions.get(session_id)
        if not session or len(session.messages) < 2:
            return {"error": "nothing to regenerate"}

        last_msg = session.messages[-1]
        if last_msg.get("role") != "assistant":
            return {"error": "last message is not assistant"}

        restore_to = len(session.messages) - 2  # before user msg
        result = self.truncate_session(session_id, restore_to)
        if result.get("status") == "ok":
            result["turn"] = restore_to
        return result

    def truncate_session(self, session_id: str, target_msg_count: int) -> dict:
        """Truncate session to target_msg_count messages, rebuilding cache."""
        session = self._sessions.get(session_id)
        if not session and self._has_disk_cache(session_id):
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session
        if not session:
            return {"error": "session not found"}
        if target_msg_count >= len(session.messages):
            return {"error": "nothing to truncate"}

        restore_messages = session.messages[:target_msg_count]
        return self._rebuild_session(session_id, restore_messages)

    def _rebuild_session(self, session_id: str, messages: list[dict]) -> dict:
        """Build a fresh KV cache for the given messages."""
        with self._lock:
            self._touch_gpu()
            t0 = time.perf_counter()

            prompt_tokens = self._tokenize_prompt(messages)

            # Try base cache first
            prompt_cache = None
            base = self._find_base_cache(messages)
            feed_tokens = prompt_tokens
            if base and len(prompt_tokens) >= base.token_count:
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    feed_tokens = prompt_tokens[base.token_count:]
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self._language_model)

            if feed_tokens:
                self._prefill_cache(prompt_cache, feed_tokens)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            cache_state = PromptCacheState()
            cache_state.cache = prompt_cache
            cache_state.token_ids = prompt_tokens

            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=new_offset,
                pending_build_time=elapsed,
            )
            self._mark_dirty(session_id)

        logger.info(
            f"[Rebuild] session={session_id} | "
            f"{new_offset} tokens, msgs={len(messages)}, {elapsed:.2f}s"
        )
        return {
            "status": "ok",
            "method": "build",
            "build_time": round(elapsed, 2),
            "cached_tokens": new_offset,
            "messages": len(messages),
        }

    def session_stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": {
                sid: {
                    "messages": len(s.messages),
                    "cache_tokens": s.total_cache_tokens,
                    "last_used": s.last_used,
                }
                for sid, s in self._sessions.items()
            },
        }
