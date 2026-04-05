"""
MLX Engine — model loading, tokenization, and generation with KV cache reuse.

Core optimization: session-based KV cache management.
- Each session maintains a KV cache with full conversation history (including thinking)
- New turns only feed suffix tokens (new user message + generation prompt)
- When conversation history doesn't match, starts fresh

This is fundamentally different from prefix-matching on tokenized messages,
because thinking tokens in the cache don't appear in the client's messages.
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
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, Optional

import mlx.core as mx
from mlx_lm import load as lm_load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.chat_format import get_chat_format
from mlx_soloheaven.engine.thinking import ThinkingBudgetProcessor, RepetitionPenaltyProcessor
from mlx_soloheaven.engine.tool_parser import parse_tool_calls, split_thinking_and_content
from mlx_soloheaven.cache.manager import CacheManager

logger = logging.getLogger(__name__)


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


DEFAULT_MAX_CHECKPOINTS = 50


@dataclass
class SessionState:
    """Tracks a conversation session's KV cache and message history."""
    cache: list  # mlx-lm prompt_cache object
    messages: list[dict]  # messages represented in the cache
    last_used: float = field(default_factory=time.time)
    total_cache_tokens: int = 0

    # Branching checkpoints (memory only, not persisted to disk)
    turn_offsets: dict[int, int] = field(default_factory=dict)
    deltanet_snapshots: dict[int, list] = field(default_factory=dict)

    # Cache build time from last branch/regenerate BUILD (seconds, consumed once)
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


class _VLMAdapter:
    """Wraps a VLM language_model to return raw mx.array for mlx-lm compatibility.

    mlx-lm's generate_step expects model(tokens, cache=...) → mx.array.
    VLM language_model returns LanguageModelOutput(logits=...).
    This adapter extracts .logits so mlx-lm's stream_generate works unchanged.
    All other attributes (make_cache, layers, parameters, etc.) are forwarded.
    """

    def __init__(self, language_model):
        self._lm = language_model

    def __call__(self, *args, **kwargs):
        out = self._lm(*args, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    def __getattr__(self, name):
        if name == "_lm":
            raise AttributeError(name)
        return getattr(self._lm, name)


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
        self.model = None
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

        # Try mlx-vlm first (handles VLM + some text models), fall back to mlx-lm
        self._is_vlm = False
        self._vlm_model = None
        self._processor = None
        try:
            from mlx_vlm import load as vlm_load
            vlm_model, processor = vlm_load(self.cfg.model_path)

            if hasattr(vlm_model, "language_model"):
                self.model = _VLMAdapter(vlm_model.language_model)
                self._vlm_model = vlm_model
            else:
                self.model = vlm_model

            # Extract tokenizer from processor
            raw_tok = getattr(processor, "tokenizer", processor)
            self.tokenizer = self._wrap_tokenizer(raw_tok, model_config)
            self._processor = processor
            self._is_vlm = True
            logger.info(f"Loaded via mlx-vlm (is_vlm={self._vlm_model is not None})")
        except Exception as e:
            logger.info(f"VLM load failed ({e}), falling back to mlx-lm")
            self.model, self.tokenizer = lm_load(self.cfg.model_path)

        elapsed = time.perf_counter() - t0

        # Derive model ID from directory name
        self.model_id = os.path.basename(self.cfg.model_path.rstrip("/"))
        logger.info(f"Model loaded in {elapsed:.1f}s — {self.model_id}")

        # Detect model family + chat format
        self.model_family = self._detect_model_family()
        self._chat_format = get_chat_format(self.model_family)
        logger.info(f"Model family: {self.model_family}, chat format: {type(self._chat_format).__name__}")

        # Auto-detect special token IDs (model-family-specific)
        self._detect_special_tokens()

        # Detect cache layer types (for optimization strategy)
        test_cache = make_prompt_cache(self.model)
        self._has_deltanet = any(type(c).__name__ == "ArraysCache" for c in test_cache)
        self._has_rotating_cache = any(type(c).__name__ == "RotatingKVCache" for c in test_cache)
        self._sliding_window_size = 0
        if self._has_rotating_cache:
            for c in test_cache:
                if type(c).__name__ == "RotatingKVCache" and hasattr(c, "max_size"):
                    self._sliding_window_size = c.max_size
                    break
        del test_cache
        if self._has_deltanet:
            logger.info(f"[{self.model_id}] DeltaNet layers detected — checkpoints enabled")
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

        # Set wired limit once at startup
        if mx.metal.is_available():
            max_rec = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(max_rec)
            logger.debug(f"Metal wired limit set to {max_rec / 1e9:.1f}GB")

            # Patch wired_limit: keep synchronize but skip set/reset cycle.
            # The set/reset cycle degrades Metal TTFT on repeated calls.
            import mlx_lm.generate as gen_module

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

            gen_module.wired_limit = _stable_wired_limit
            logger.debug("Patched wired_limit: stable (set once at startup)")

        self._build_disk_index()
        self._touch_gpu()  # Mark GPU active after model load
        if self.cfg.gpu_keepalive:
            self._start_gpu_keepalive()
            logger.info(f"[{self.model_id}] GPU keepalive enabled (interval={self.GPU_KEEPALIVE_INTERVAL}s)")

    # --- Model detection helpers ---

    def _detect_model_family(self) -> str:
        """Detect model family from model_type in config.json."""
        mt = self._model_type.lower()
        if "gemma4" in mt:
            return "gemma4"
        # Default: ChatML family (Qwen, GLM, MiniMax, etc.)
        return "chatml"

    def _detect_special_tokens(self):
        """Auto-detect special token IDs based on model family."""
        if self.model_family == "gemma4":
            self.cfg.think_end_token = _detect_token_id(self.tokenizer, "<channel|>")
            self.cfg.think_start_token = _detect_token_id(self.tokenizer, "<|channel>")
            self.cfg.im_end_token = _detect_token_id(self.tokenizer, "<turn|>")
        else:
            if self.cfg.think_end_token < 0:
                self.cfg.think_end_token = _detect_token_id(self.tokenizer, "</think>")
            if self.cfg.think_start_token < 0:
                self.cfg.think_start_token = _detect_token_id(self.tokenizer, "<think>")
            if self.cfg.im_end_token < 0:
                self.cfg.im_end_token = _detect_token_id(self.tokenizer, "<|im_end|>")

        logger.info(
            f"[{self.model_id}] model_family={self.model_family} | "
            f"Token IDs: think_end={self.cfg.think_end_token}, "
            f"think_start={self.cfg.think_start_token}, "
            f"eos/im_end={self.cfg.im_end_token}"
        )

    def _wrap_tokenizer(self, raw_tok, config: dict):
        """Wrap a VLM processor's tokenizer for mlx-lm compatibility."""
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        # Ensure chat_template is set (some models store it in a file)
        if not getattr(raw_tok, "chat_template", None):
            template_path = os.path.join(self.cfg.model_path, "chat_template.jinja")
            if os.path.exists(template_path):
                with open(template_path) as f:
                    raw_tok.chat_template = f.read()
                logger.info(f"Loaded chat_template from {template_path}")

        wrapped = TokenizerWrapper(raw_tok)

        # Ensure correct EOS token IDs
        eos_ids = config.get("eos_token_id", [])
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        # Gemma 4: <turn|> (106) is the chat turn-end token
        if self._model_type == "gemma4":
            turn_end = _detect_token_id(raw_tok, "<turn|>")
            if turn_end >= 0 and turn_end not in eos_ids:
                eos_ids.append(turn_end)

        if eos_ids:
            wrapped.eos_token_ids = set(eos_ids)

        return wrapped

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

    def _session_ckpt_path(self, session_id: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"session_{session_id}_ckpt.safetensors")

    def _save_session_to_disk(self, session_id: str, session: SessionState):
        """Save session's KV cache + checkpoints to disk. Caller MUST hold _lock."""
        t0 = time.perf_counter()
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = self._session_cache_path(session_id)
        metadata = {
            "session_id": session_id,
            "messages": json.dumps(session.messages, ensure_ascii=False),
            "total_cache_tokens": str(session.total_cache_tokens),
            "last_used": str(session.last_used),
        }
        save_prompt_cache(path, session.cache, metadata=metadata)

        # Save DeltaNet checkpoints
        self._save_checkpoints_to_disk(session_id, session)

        elapsed = time.perf_counter() - t0
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.add(session_id)
        fsize = os.path.getsize(path) / 1e6
        n_ckpt = len(session.deltanet_snapshots)
        logger.info(
            f"[KV Cache] session={session_id} | DISK SAVE | "
            f"{session.total_cache_tokens} tokens, {len(session.messages)} msgs, "
            f"{n_ckpt} checkpoints, {fsize:.1f}MB, {elapsed:.2f}s"
        )

    def _save_checkpoints_to_disk(self, session_id: str, session: SessionState):
        """Save DeltaNet snapshots to a separate safetensors file."""
        ckpt_path = self._session_ckpt_path(session_id)
        if not session.deltanet_snapshots:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            return

        arrays = {}
        structure = {}

        for turn_key, snapshot in session.deltanet_snapshots.items():
            structure[str(turn_key)] = len(snapshot)
            for layer_idx, cache_obj in enumerate(snapshot):
                states = cache_obj.state if isinstance(cache_obj.state, list) else [cache_obj.state]
                for state_idx, arr in enumerate(states):
                    if isinstance(arr, mx.array):
                        arrays[f"t{turn_key}_l{layer_idx}_s{state_idx}"] = arr

        metadata = {
            "turn_offsets": json.dumps(
                {str(k): v for k, v in session.turn_offsets.items()}
            ),
            "structure": json.dumps(structure),
        }

        try:
            mx.save_safetensors(ckpt_path, arrays, metadata=metadata)
        except Exception as e:
            logger.warning(f"[Checkpoint] SAVE FAILED for {session_id}: {e}")

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
                self._save_session_to_disk(sid, session)
                saved += 1
            except Exception as e:
                logger.error(f"[KV Cache] session={sid} | FLUSH SAVE FAILED | {e}")
                # Re-add to dirty set for retry
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
        """Load session's KV cache + messages from disk."""
        path = self._session_cache_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            t0 = time.perf_counter()
            cache, metadata = load_prompt_cache(path, return_metadata=True)
            messages = json.loads(metadata.get("messages", "[]"))
            total_tokens = int(metadata.get("total_cache_tokens", "0"))
            last_used = float(metadata.get("last_used", "0"))

            # Verify loaded cache matches model structure
            model_cache = make_prompt_cache(self.model)
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

            session = SessionState(
                cache=cache,
                messages=messages,
                total_cache_tokens=loaded_offset,
                last_used=last_used,
            )

            # Load checkpoints if available
            turn_offsets, deltanet_snapshots = self._load_checkpoints_from_disk(session_id)
            if turn_offsets:
                session.turn_offsets = turn_offsets
                session.deltanet_snapshots = deltanet_snapshots

            fsize = os.path.getsize(path) / 1e6
            n_ckpt = len(session.deltanet_snapshots)
            logger.info(
                f"[KV Cache] session={session_id} | DISK LOAD | "
                f"{loaded_offset} tokens, {len(messages)} msgs, "
                f"{n_ckpt} checkpoints, {fsize:.1f}MB, {elapsed:.2f}s"
            )
            return session
        except Exception as e:
            logger.error(f"[KV Cache] session={session_id} | DISK LOAD FAILED | {e}")
            return None

    def _load_checkpoints_from_disk(self, session_id: str) -> tuple[dict, dict]:
        """Load DeltaNet snapshots from disk. Returns (turn_offsets, deltanet_snapshots)."""
        ckpt_path = self._session_ckpt_path(session_id)
        if not os.path.exists(ckpt_path):
            return {}, {}

        try:
            from safetensors import safe_open

            # Load arrays natively (handles bfloat16)
            loaded = mx.load(ckpt_path)
            # Load metadata separately
            with safe_open(ckpt_path, framework="numpy") as f:
                metadata = f.metadata()

            turn_offsets = {
                int(k): v
                for k, v in json.loads(metadata["turn_offsets"]).items()
            }
            structure = json.loads(metadata["structure"])

            # Get template ArraysCache objects from model
            model_cache = make_prompt_cache(self.model)
            template_arrays_caches = [
                c for c in model_cache if type(c).__name__ == "ArraysCache"
            ]

            deltanet_snapshots = {}
            for turn_key_str, n_layers in structure.items():
                turn_key = int(turn_key_str)
                snapshot = []
                for layer_idx in range(n_layers):
                    # Collect state arrays for this layer
                    states = []
                    s = 0
                    while f"t{turn_key}_l{layer_idx}_s{s}" in loaded:
                        states.append(loaded[f"t{turn_key}_l{layer_idx}_s{s}"])
                        s += 1

                    # Create ArraysCache from template with loaded state
                    template = copy.deepcopy(template_arrays_caches[layer_idx])
                    if len(states) == 1:
                        template.state = states[0]
                    else:
                        template.state = states
                    snapshot.append(template)

                deltanet_snapshots[turn_key] = snapshot

            logger.debug(
                f"[Checkpoint] LOADED for {session_id} | "
                f"{len(deltanet_snapshots)} checkpoints"
            )
            return turn_offsets, deltanet_snapshots

        except Exception as e:
            logger.warning(f"[Checkpoint] LOAD FAILED for {session_id}: {e}")
            return {}, {}

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

    def _extract_system_tokens(self, messages: list[dict], full_tokens: list[int], tools: list | None = None, thinking: bool | None = None) -> list[int] | None:
        """Extract system prompt tokens from the full tokenized messages.

        Uses dummy user message to tokenize system+user, then strips dummy suffix
        to get pure system prompt tokens.

        For models with RotatingKVCache (e.g. Gemma 4), also handles the case where
        no explicit system message exists but the template auto-generates a system
        prefix (e.g. <bos><|turn>system\n<|think|><turn|>).
        """
        has_system = messages and messages[0].get("role") in ("system", "developer")
        if not has_system and not self._has_rotating_cache:
            return None  # Only needed for explicit system or sliding window models
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return None  # Already have base cache for this system prompt
        try:
            enable_thinking = thinking if thinking is not None else self.cfg.enable_thinking
            if has_system:
                system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
            else:
                # No explicit system message — use just a dummy user to extract
                # the template's auto-generated system prefix (e.g. Gemma 4: <bos><|turn>system\n<|think|><turn|>)
                system_with_dummy = [{"role": "user", "content": "hi"}]
            full_with_dummy = self.tokenize_messages(system_with_dummy, tools=tools, thinking=thinking)
            dummy_suffix = self.tokenizer.encode(
                self._chat_format.dummy_suffix(enable_thinking),
                add_special_tokens=False,
            )
            if len(full_with_dummy) > len(dummy_suffix):
                system_tokens = full_with_dummy[: len(full_with_dummy) - len(dummy_suffix)]
                # Verify these tokens are a prefix of the full tokens
                if full_tokens[:len(system_tokens)] == system_tokens:
                    return system_tokens
                else:
                    # Find divergence point for debugging
                    for j in range(min(len(system_tokens), len(full_tokens))):
                        if system_tokens[j] != full_tokens[j]:
                            logger.debug(
                                f"[Base Cache] Token mismatch at pos {j}/{len(system_tokens)} | "
                                f"sys={system_tokens[max(0,j-2):j+3]} vs full={full_tokens[max(0,j-2):j+3]}"
                            )
                            break
                    else:
                        logger.debug(
                            f"[Base Cache] Length mismatch: system={len(system_tokens)} vs full_prefix={len(full_tokens)}"
                        )
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
        """Process tokens through the model to populate a KV cache."""
        arr = mx.array(tokens)
        for i in range(0, len(tokens), self._PREFILL_STEP):
            chunk = arr[i : i + self._PREFILL_STEP]
            self.model(chunk[None], cache=cache)
        self._eval_cache(cache)

    @staticmethod
    def _is_compacted_tool(s_content: str, i_content: str) -> bool:
        """Check if either side is a compacted/cleared tool result placeholder."""
        for c in (s_content, i_content):
            if c.startswith("[") and ("cleared]" in c or "compacted:" in c):
                return True
        return False

    @staticmethod
    def _normalize_for_match(content: str, role: str) -> str:
        """Normalize message content for comparison."""
        import re
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
        # Strip thinking text from assistant messages
        # Server stores content without thinking, but client may send full text
        if role == "assistant":
            # Gemma 4: <|channel>thought\n...<channel|>content
            content = re.sub(r"<\|channel>thought\n.*?<channel\|>\s*", "", content, flags=re.DOTALL)
            m2 = re.match(r".*?<channel\|>\s*(.*)", content, re.DOTALL)
            if m2:
                content = m2.group(1)
            # ChatML: <think>...</think>content
            m = re.match(r"<think>.*?</think>\s*(.*)", content, re.DOTALL)
            if m:
                content = m.group(1)
            else:
                # Case 2: thinking...\n</think>\n\ncontent (prompt ends inside <think>)
                m = re.match(r".*?</think>\s*(.*)", content, re.DOTALL)
                if m:
                    content = m.group(1)
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
            s_content = s_msg.get("content", "") or ""
            i_content = i_msg.get("content", "") or ""
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

    def _make_suffix_tokens(self, query: str, thinking: bool = True) -> list[int]:
        """Create suffix tokens for a new user turn."""
        return self._chat_format.suffix_user(self.tokenizer, query, thinking)

    def _make_suffix_tokens_tool_result(self, messages: list[dict], thinking: bool = True) -> list[int]:
        """Create suffix tokens for tool result messages."""
        return self._chat_format.suffix_tool_result(self.tokenizer, messages, thinking)

    def tokenize_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        thinking: bool | None = None,
    ) -> list[int]:
        """Convert OpenAI-format messages to token IDs using model's chat template."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            if role == "developer":
                role = "system"
            m = {"role": role}
            if msg.get("content") is not None:
                content = msg["content"]
                # Normalize list-format content to plain string
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
                # Ensure tool_call arguments are dicts (Jinja template calls .items())
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

        enable_thinking = thinking if thinking is not None else self.cfg.enable_thinking
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            tool_defs = []
            for t in tools:
                if hasattr(t, "model_dump"):
                    tool_defs.append(t.model_dump())
                elif isinstance(t, dict):
                    tool_defs.append(t)
            kwargs["tools"] = tool_defs

        return self.tokenizer.apply_chat_template(formatted, **kwargs)

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
    ) -> Generator[GenerationResult, None, None]:
        """Core generation logic (must hold lock)."""
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
        total_prompt_tokens = 0  # Full prompt token count for usage reporting

        # Suffix mode: reuse session KV cache, append only new user message tokens.
        # Works for all models because thinking tokens stay in the cache naturally
        # (the model generated them — they're part of the KV state).
        _suffix_safe = True

        if (
            session
            and session.cache is not None
            and self._messages_match(session.messages, messages)
        ):
            new_messages = messages[len(session.messages):]
            prompt_cache = session.cache
            cache_offset = self._get_cache_offset(prompt_cache)

            if not new_messages:
                cache_mode = "retry"
                prompt_cache = make_prompt_cache(self.model)
                prompt_tokens = self.tokenize_messages(messages, tools=tools, thinking=use_thinking)
                total_prompt_tokens = len(prompt_tokens)
                logger.info(
                    f"[KV Cache] session={session_id} | RETRY | "
                    f"discarding {cache_offset} cached tokens, "
                    f"re-processing {len(prompt_tokens)} tokens"
                )
            elif _suffix_safe and len(new_messages) == 1 and new_messages[0]["role"] == "user":
                cache_mode = "hit"
                prompt_tokens = self._make_suffix_tokens(new_messages[0]["content"], thinking=use_thinking)
                total_prompt_tokens = cache_offset + len(prompt_tokens)
                logger.info(
                    f"[KV Cache] session={session_id} | HIT | "
                    f"reusing {cache_offset} cached tokens + "
                    f"{len(prompt_tokens)} suffix tokens"
                )
            elif _suffix_safe:
                cache_mode = "hit_multi"
                prompt_tokens = self._make_suffix_tokens_tool_result(new_messages, thinking=use_thinking)
                total_prompt_tokens = cache_offset + len(prompt_tokens)
                new_roles = [m["role"] for m in new_messages]
                logger.info(
                    f"[KV Cache] session={session_id} | HIT (multi-msg) | "
                    f"reusing {cache_offset} cached tokens + "
                    f"{len(prompt_tokens)} suffix tokens | {new_roles}"
                )
            else:
                # Suffix-unsafe model (e.g. Gemma 4): full retokenization
                # Template strips thinking from history → KV state diverges → must rebuild
                prompt_tokens = self.tokenize_messages(messages, tools=tools, thinking=use_thinking)
                total_prompt_tokens = len(prompt_tokens)


                # Try to reuse base cache (system prompt KV) even for retokenization
                base = self._find_base_cache(messages, tools=tools)
                if base and len(prompt_tokens) >= base.token_count:
                    if prompt_tokens[:base.token_count] == base.tokens:
                        prompt_cache = self._clone_base_cache(base)
                        prompt_tokens = prompt_tokens[base.token_count:]
                        cache_mode = "retok_base"
                        logger.info(
                            f"[KV Cache] session={session_id} | RETOKENIZE+BASE | "
                            f"reusing {base.token_count} base tokens, "
                            f"processing {len(prompt_tokens)} remaining tokens "
                            f"(was {cache_offset} cached)"
                        )
                    else:
                        base = None  # token mismatch — fall through

                if base is None:
                    # No usable base cache — build system cache inline to protect from rotation
                    system_tokens = self._extract_system_tokens(
                        messages, prompt_tokens, tools=tools, thinking=use_thinking,
                    )
                    if system_tokens and len(system_tokens) < len(prompt_tokens):
                        prompt_cache = make_prompt_cache(self.model)
                        self._prefill_cache(prompt_cache, system_tokens)

                        self._register_base_cache(messages, prompt_cache, system_tokens, tools=tools)
                        prompt_tokens = prompt_tokens[len(system_tokens):]
                        cache_mode = "retok_build"
                        logger.info(
                            f"[KV Cache] session={session_id} | RETOKENIZE+BUILD | "
                            f"built base ({len(system_tokens)} sys tokens), "
                            f"processing {len(prompt_tokens)} remaining"
                        )
                    else:
                        prompt_cache = make_prompt_cache(self.model)
                        cache_mode = "miss"
                        logger.info(
                            f"[KV Cache] session={session_id} | RETOKENIZE | "
                            f"full rebuild {total_prompt_tokens} tokens"
                        )
        else:
            # Try branch detection first (incoming shorter than stored, prefix matches)
            branch_restored = False
            if session and session.cache is not None:
                branch_point = self._check_branch_match(session.messages, messages)
                if branch_point is not None:
                    checkpoint_turn = None
                    for turn in sorted(session.turn_offsets.keys(), reverse=True):
                        if turn <= branch_point:
                            checkpoint_turn = turn
                            break
                    # Guard: RotatingKVCache cannot be safely sliced (circular buffer)
                    has_rotating = self._has_rotating_cache
                    if has_rotating:
                        logger.info(
                            f"[KV Cache] session={session_id} | BRANCH skipped — "
                            f"RotatingKVCache cannot be sliced, falling through to BUILD"
                        )
                    elif checkpoint_turn is not None and checkpoint_turn in session.deltanet_snapshots:
                        target_offset = session.turn_offsets[checkpoint_turn]
                        deltanet_states = session.deltanet_snapshots[checkpoint_turn]
                        new_cache = []
                        dn_idx = 0
                        for c in session.cache:
                            if type(c).__name__ == "ArraysCache":
                                new_cache.append(copy.deepcopy(deltanet_states[dn_idx]))
                                dn_idx += 1
                            else:
                                sliced = copy.deepcopy(c)
                                if hasattr(sliced, 'keys') and sliced.keys is not None:
                                    sliced.keys = sliced.keys[..., :target_offset, :]
                                    sliced.values = sliced.values[..., :target_offset, :]
                                    sliced.offset = target_offset
                                new_cache.append(sliced)
                        prompt_cache = new_cache
                        new_msgs = messages[branch_point:]
                        if len(new_msgs) == 1 and new_msgs[0]["role"] == "user":
                            prompt_tokens = self._make_suffix_tokens(new_msgs[0]["content"], thinking=use_thinking)
                        else:
                            prompt_tokens = self._make_suffix_tokens_tool_result(new_msgs, thinking=use_thinking)
                        total_prompt_tokens = target_offset + len(prompt_tokens)
                        cache_mode = "branch"
                        branch_restored = True
                        logger.info(
                            f"[KV Cache] session={session_id} | BRANCH | "
                            f"branch_point={branch_point}, checkpoint={checkpoint_turn}, "
                            f"restored {target_offset} tokens + {len(prompt_tokens)} suffix"
                        )

            if not branch_restored:
                prev_cached = (
                    self._get_cache_offset(session.cache) if session and session.cache else 0
                )
                prompt_tokens = self.tokenize_messages(messages, tools=tools, thinking=use_thinking)
                total_prompt_tokens = len(prompt_tokens)

            # Try base cache: reuse system prompt KV cache
            base = self._find_base_cache(messages, tools=tools) if not branch_restored else None
            if base and len(prompt_tokens) >= base.token_count:
                # Verify token prefix matches
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    remaining_tokens = prompt_tokens[base.token_count:]
                    cache_mode = "base_hit"
                    logger.info(
                        f"[KV Cache] session={session_id} | BASE HIT | "
                        f"reusing {base.token_count} base tokens, "
                        f"processing {len(remaining_tokens)} remaining tokens "
                        f"(was {prev_cached} cached)"
                    )
                    prompt_tokens = remaining_tokens
                else:
                    # Token mismatch — fall through to full miss
                    base = None

            if cache_mode == "miss":
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

                # Build base cache eagerly on MISS: process system prompt
                # tokens first, register as base cache, then continue with rest
                system_tokens = self._extract_system_tokens(messages, prompt_tokens, tools=tools, thinking=use_thinking)
                if system_tokens and len(system_tokens) < len(prompt_tokens):
                    prompt_cache = make_prompt_cache(self.model)
                    self._prefill_cache(prompt_cache, system_tokens)
                    # Register as base cache immediately
                    self._register_base_cache(messages, prompt_cache, system_tokens, tools=tools)
                    # Continue with remaining tokens only
                    prompt_tokens = prompt_tokens[len(system_tokens):]
                    cache_mode = "base_build"
                    logger.debug(
                        f"[Base Cache] EAGER BUILD | "
                        f"system={len(system_tokens)} tokens, "
                        f"remaining={len(prompt_tokens)} tokens"
                    )
                else:
                    prompt_cache = make_prompt_cache(self.model)

        # Setup generation
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

        logits_processors = []
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyProcessor(penalty=repetition_penalty))
        budget = thinking_budget if thinking_budget is not None else self.cfg.thinking_budget
        if (
            use_thinking
            and budget > 0
            and self.cfg.think_end_token >= 0
        ):
            thinking_proc = ThinkingBudgetProcessor(
                budget=budget,
                think_end_token=self.cfg.think_end_token,
                think_start_token=self.cfg.think_start_token,
                model_family=self.model_family,
            )
            thinking_proc.reset()
            logits_processors.append(thinking_proc)

        # Build cache info for response
        cached_tokens = total_prompt_tokens - len(prompt_tokens)
        response_cache_info = {
            "cache_mode": cache_mode,
            "cached_tokens": cached_tokens,
            "new_tokens": len(prompt_tokens),
            "total_prompt_tokens": total_prompt_tokens,
        }
        # Include pending build time from branch/regenerate BUILD
        if session and session.pending_build_time > 0:
            response_cache_info["build_time"] = round(session.pending_build_time, 2)
            session.pending_build_time = 0.0  # consume once

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
            f"[Generate] prompt={len(prompt_tokens)} tokens, max={max_tokens}, "
            f"{sampling_info}, cache_mode={cache_mode}"
        )

        # Stream generate
        generated_tokens = []
        last_resp = None
        t_gen_start = time.perf_counter()
        t_first_token = None



        cancelled = False
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors if logits_processors else None,
        ):
            if cancel_event and cancel_event.is_set():
                logger.debug(f"[Generate] session={session_id} | CANCELLED")
                cancelled = True
                break
            generated_tokens.append(resp.token)
            if t_first_token is None:
                t_first_token = time.perf_counter()
                logger.info(
                    f"[Generate] TTFT={round((t_first_token - t_gen_start)*1000)}ms"
                )
            last_resp = resp

            yield GenerationResult(
                text=resp.text,
                token=resp.token,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=len(generated_tokens),
                prompt_tps=resp.prompt_tps if hasattr(resp, "prompt_tps") else 0,
                generation_tps=(
                    resp.generation_tps if hasattr(resp, "generation_tps") else 0
                ),
            )

        self._touch_gpu()

        # Log generated text for debugging
        if generated_tokens:
            gen_text = self.tokenizer.decode(generated_tokens)
            preview = gen_text[:200].replace('\n', '\\n')
            logger.debug(
                f"[Generate] session={session_id} | "
                f"tokens={len(generated_tokens)} | cancelled={cancelled} | "
                f"text={preview!r}"
            )

        if cancelled:
            return

        # Guard: detect empty response (only </think><|im_end|>, no real content)
        # This prevents poisoning the cache with degenerate outputs
        if generated_tokens and session_id:
            gen_text = self.tokenizer.decode(generated_tokens)
            thinking, content = split_thinking_and_content(gen_text, model_family=self.model_family)
            if not content or not content.strip():
                logger.warning(
                    f"[KV Cache] session={session_id} | SKIP SAVE | "
                    f"empty response detected ({len(generated_tokens)} tokens, "
                    f"no content after </think>) — not updating cache"
                )
                # Still yield final result, but don't save to cache
                finish_reason = "stop"
                yield GenerationResult(
                    text="",
                    finish_reason=finish_reason,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=len(generated_tokens),
                )
                return

        # Save cache to session
        if session_id:
            new_offset = self._get_cache_offset(prompt_cache)
            prev_offset = session.total_cache_tokens if session else 0

            # Preserve existing checkpoints
            prev_turn_offsets = session.turn_offsets if session else {}
            prev_deltanet_snapshots = session.deltanet_snapshots if session else {}

            self._sessions[session_id] = SessionState(
                cache=prompt_cache,
                messages=messages,
                total_cache_tokens=new_offset,
                turn_offsets=prev_turn_offsets,
                deltanet_snapshots=prev_deltanet_snapshots,
            )

            # Save checkpoint: DeltaNet snapshot + KVCache offset
            session_state = self._sessions[session_id]
            turn_key = len(messages)
            session_state.turn_offsets[turn_key] = new_offset
            if self._has_deltanet:
                session_state.deltanet_snapshots[turn_key] = [
                    copy.deepcopy(c) for c in prompt_cache
                    if type(c).__name__ == "ArraysCache"
                ]
            # Prune old checkpoints
            max_ckpt = self.cfg.max_checkpoints if hasattr(self.cfg, 'max_checkpoints') else DEFAULT_MAX_CHECKPOINTS
            if max_ckpt > 0:
                while len(session_state.deltanet_snapshots) > max_ckpt:
                    oldest_key = min(session_state.deltanet_snapshots.keys())
                    del session_state.deltanet_snapshots[oldest_key]
                    del session_state.turn_offsets[oldest_key]

            logger.debug(
                f"[KV Cache] session={session_id} | SAVED | "
                f"offset: {prev_offset} -> {new_offset} tokens "
                f"(+{new_offset - prev_offset}), "
                f"checkpoints={len(session_state.turn_offsets)}"
            )

        # Auto-register base cache on full miss (system prompt not yet cached)
        # For RotatingKVCache models (e.g. Gemma 4), also register when template auto-generates system prefix
        has_system_or_rotating = (
            (messages and messages[0].get("role") in ("system", "developer"))
            or self._has_rotating_cache
        )
        if cache_mode in ("miss",) and messages and has_system_or_rotating:
            sys_hash = self._system_hash(messages, tools=tools)
            if sys_hash and sys_hash not in self._base_caches:
                # Tokenize system prompt with dummy user to satisfy chat template
                has_system = messages[0].get("role") in ("system", "developer")
                system_with_dummy = (
                    [messages[0], {"role": "user", "content": "hi"}]
                    if has_system
                    else [{"role": "user", "content": "hi"}]
                )
                full_tokens = self.tokenize_messages(system_with_dummy, tools=tools, thinking=use_thinking)
                dummy_suffix = self.tokenizer.encode(
                    self._chat_format.dummy_suffix(use_thinking),
                    add_special_tokens=False,
                )
                system_tokens = full_tokens[: len(full_tokens) - len(dummy_suffix)]
                if system_tokens:
                    try:
                        base_cache = make_prompt_cache(self.model)
                        # Process system tokens directly (no generation) to avoid
                        # polluting the base cache with an extra generated token
                        self._prefill_cache(base_cache, system_tokens)
                        self._register_base_cache(messages, base_cache, system_tokens, tools=tools)
                    except Exception as e:
                        logger.warning(f"[Base Cache] registration failed: {e}")

        # Yield final result with finish_reason
        finish_reason = "stop"
        if last_resp and hasattr(last_resp, "finish_reason"):
            finish_reason = last_resp.finish_reason or "stop"

        if has_tools and generated_tokens:
            full_text = self.tokenizer.decode(generated_tokens)
            _, tool_calls = parse_tool_calls(full_text, model_family=self.model_family)
            if tool_calls:
                finish_reason = "tool_calls"

        yield GenerationResult(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=len(generated_tokens),
            prompt_tps=last_resp.prompt_tps if last_resp and hasattr(last_resp, "prompt_tps") else 0,
            generation_tps=(
                last_resp.generation_tps
                if last_resp and hasattr(last_resp, "generation_tps")
                else 0
            ),
            cache_info=response_cache_info,
        )

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
        """Update stored session messages after assistant response is finalized."""
        session = self._sessions.get(session_id)
        if session:
            session.messages = messages
            session.touch()
            logger.info(
                f"[Session] {session_id} | messages updated | "
                f"{len(messages)} msgs, {session.total_cache_tokens} cached tokens"
            )

            self._mark_dirty(session_id)

    def compact_session(self, session_id: str, messages: list[dict]) -> dict:
        """Replace a session's messages and rebuild KV cache from scratch.

        Used when client compresses/summarizes conversation context.
        Returns stats about the new cache.
        """
        with self._lock:
            self._touch_gpu()
            t0 = time.perf_counter()

            prompt_tokens = self.tokenize_messages(messages)

            # Try base cache first
            base = self._find_base_cache(messages)
            if base and len(prompt_tokens) >= base.token_count:
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    feed_tokens = prompt_tokens[base.token_count:]
                    base_tokens_used = base.token_count
                else:
                    prompt_cache = make_prompt_cache(self.model)
                    feed_tokens = prompt_tokens
                    base_tokens_used = 0
            else:
                prompt_cache = make_prompt_cache(self.model)
                feed_tokens = prompt_tokens
                base_tokens_used = 0

            # Process tokens through model to build cache (no generation)
            if feed_tokens:
                self._prefill_cache(prompt_cache, feed_tokens)
            self._eval_cache(prompt_cache)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            # Save to session
            prev = self._sessions.get(session_id)
            prev_tokens = self._get_cache_offset(prev.cache) if prev and prev.cache else 0
            self._sessions[session_id] = SessionState(
                cache=prompt_cache,
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

            # Auto-register base if not present
            if messages and messages[0].get("role") == "system":
                sys_hash = self._system_hash(messages)
                if sys_hash and sys_hash not in self._base_caches:
                    # Use same tokenization approach as main auto-register
                    system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
                    full_tokens = self.tokenize_messages(system_with_dummy)
                    dummy_suffix = self.tokenizer.encode(
                        self._chat_format.dummy_suffix(True),
                        add_special_tokens=False,
                    )
                    system_tokens = full_tokens[: len(full_tokens) - len(dummy_suffix)]
                    if system_tokens:
                        try:
                            base_cache = make_prompt_cache(self.model)
                            for _ in stream_generate(
                                self.model,
                                self.tokenizer,
                                prompt=system_tokens,
                                max_tokens=1,
                                prompt_cache=base_cache,
                            ):
                                break
                            self._eval_cache(base_cache)
                            self._register_base_cache(messages, base_cache, system_tokens)
                        except Exception as e:
                            logger.warning(f"[Base Cache] registration failed: {e}")

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
        # Remove disk cache + checkpoints
        for p in [self._session_cache_path(session_id), self._session_ckpt_path(session_id)]:
            if os.path.exists(p):
                os.remove(p)
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
        except (asyncio.CancelledError, GeneratorExit):
            cancel_event.set()
            logger.debug("[Stream] Client disconnected — cancelling generation")
            raise

    # --- Branching & Regeneration ---

    def branch_from_turn(
        self,
        source_session_id: str,
        new_session_id: str,
        branch_turn: int,
        branch_messages: list[dict] | None = None,
    ) -> dict:
        """Branch a new session from a specific turn of an existing session.

        If a checkpoint exists at or before branch_turn, restore from it
        (fast: DeltaNet deepcopy + KVCache slice).
        Otherwise, build the cache from scratch by processing branch_messages
        through the model (slower but always works).
        """
        source = self._sessions.get(source_session_id)

        # Load from disk if not in memory
        if not source and self._has_disk_cache(source_session_id):
            source = self._load_session_from_disk(source_session_id)
            if source:
                self._sessions[source_session_id] = source

        # Determine branch messages from source or caller
        if source:
            engine_messages = source.messages[:branch_turn]
        elif branch_messages:
            engine_messages = branch_messages
        else:
            return {"error": "source session not found and no messages provided"}

        # Fast path 1: full copy (branch at end of conversation — no checkpoint needed)
        if source and source.cache is not None and branch_turn >= len(source.messages):
            with self._lock:
                new_cache = copy.deepcopy(source.cache)
                target_offset = source.total_cache_tokens
                new_session = SessionState(
                    cache=new_cache,
                    messages=list(source.messages),
                    total_cache_tokens=target_offset,
                )
                # Copy all existing checkpoints
                for turn, offset in source.turn_offsets.items():
                    new_session.turn_offsets[turn] = offset
                    if turn in source.deltanet_snapshots:
                        new_session.deltanet_snapshots[turn] = copy.deepcopy(
                            source.deltanet_snapshots[turn]
                        )
                self._sessions[new_session_id] = new_session
                self._mark_dirty(new_session_id)

            logger.info(
                f"[Branch] {source_session_id} -> {new_session_id} | "
                f"COPY | cached_tokens={target_offset}, messages={len(source.messages)}"
            )
            return {
                "status": "ok",
                "method": "copy",
                "cached_tokens": target_offset,
                "messages": len(source.messages),
            }

        # Fast path 2: checkpoint restore (branch at earlier turn)
        checkpoint_turn = None
        if source:
            for turn in sorted(source.turn_offsets.keys(), reverse=True):
                if turn <= branch_turn:
                    checkpoint_turn = turn
                    break

        # Guard: RotatingKVCache cannot be safely sliced (circular buffer)
        has_rotating = self._has_rotating_cache

        if (
            checkpoint_turn is not None
            and source
            and not has_rotating
            and checkpoint_turn in source.deltanet_snapshots
        ):
            # Restore from checkpoint
            with self._lock:
                deltanet_states = source.deltanet_snapshots[checkpoint_turn]
                target_offset = source.turn_offsets[checkpoint_turn]

                new_cache = []
                dn_idx = 0
                for c in source.cache:
                    if type(c).__name__ == "ArraysCache":
                        new_cache.append(copy.deepcopy(deltanet_states[dn_idx]))
                        dn_idx += 1
                    else:
                        sliced = copy.deepcopy(c)
                        if hasattr(sliced, 'keys') and sliced.keys is not None:
                            sliced.keys = sliced.keys[..., :target_offset, :]
                            sliced.values = sliced.values[..., :target_offset, :]
                            sliced.offset = target_offset
                        new_cache.append(sliced)

                new_session = SessionState(
                    cache=new_cache,
                    messages=engine_messages,
                    total_cache_tokens=target_offset,
                )
                for turn, offset in source.turn_offsets.items():
                    if turn <= checkpoint_turn:
                        new_session.turn_offsets[turn] = offset
                        if turn in source.deltanet_snapshots:
                            new_session.deltanet_snapshots[turn] = copy.deepcopy(
                                source.deltanet_snapshots[turn]
                            )
                self._sessions[new_session_id] = new_session
                self._mark_dirty(new_session_id)

            logger.info(
                f"[Branch] {source_session_id} -> {new_session_id} | "
                f"CHECKPOINT | turn={branch_turn}, checkpoint={checkpoint_turn}, "
                f"cached_tokens={target_offset}, messages={len(engine_messages)}"
            )
            return {
                "status": "ok",
                "method": "checkpoint",
                "cached_tokens": target_offset,
                "messages": len(engine_messages),
            }
        else:
            # Slow path: build cache from scratch by processing messages
            with self._lock:
                self._touch_gpu()
                t0 = time.perf_counter()
                prompt_tokens = self.tokenize_messages(engine_messages)

                # Try base cache first
                prompt_cache = None
                base = self._find_base_cache(engine_messages)
                if base and len(prompt_tokens) >= base.token_count:
                    if prompt_tokens[:base.token_count] == base.tokens:
                        prompt_cache = self._clone_base_cache(base)
                        prompt_tokens = prompt_tokens[base.token_count:]

                if prompt_cache is None:
                    prompt_cache = make_prompt_cache(self.model)

                # Process tokens through model (no generation)
                if prompt_tokens:
                    self._prefill_cache(prompt_cache, prompt_tokens)

                new_offset = self._get_cache_offset(prompt_cache)
                elapsed = time.perf_counter() - t0

                new_session = SessionState(
                    cache=prompt_cache,
                    messages=engine_messages,
                    total_cache_tokens=new_offset,
                )
                new_session.pending_build_time = elapsed
                # Save checkpoint so immediate regen/delete is fast
                turn_key = len(engine_messages)
                new_session.turn_offsets[turn_key] = new_offset
                if self._has_deltanet:
                    new_session.deltanet_snapshots[turn_key] = [
                        copy.deepcopy(c) for c in prompt_cache
                        if type(c).__name__ == "ArraysCache"
                    ]
                self._sessions[new_session_id] = new_session
                self._mark_dirty(new_session_id)

            logger.info(
                f"[Branch] {source_session_id} -> {new_session_id} | "
                f"BUILD | turn={branch_turn}, "
                f"cached_tokens={new_offset}, messages={len(engine_messages)}, "
                f"{elapsed:.2f}s"
            )
            return {
                "status": "ok",
                "method": "build",
                "build_time": round(elapsed, 2),
                "cached_tokens": new_offset,
                "messages": len(engine_messages),
            }

    def prepare_regenerate(self, session_id: str) -> dict:
        """Remove last assistant+user pair and restore cache to before them.

        Delegates to truncate_session for cache restoration.
        """
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
        """Truncate session to target_msg_count messages, restoring cache.

        Used by delete-last-message and regenerate.
        Tries checkpoint first, falls back to BUILD.
        """
        session = self._sessions.get(session_id)
        # Load from disk if not in memory
        if not session and self._has_disk_cache(session_id):
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session
        if not session:
            return {"error": "session not found"}
        if target_msg_count >= len(session.messages):
            return {"error": "nothing to truncate"}

        restore_messages = session.messages[:target_msg_count]

        # Try checkpoint restore
        checkpoint_turn = None
        for turn in sorted(session.turn_offsets.keys(), reverse=True):
            if turn <= target_msg_count:
                checkpoint_turn = turn
                break

        # Guard: RotatingKVCache cannot be safely sliced (circular buffer)
        has_rotating = self._has_rotating_cache

        if (
            checkpoint_turn is not None
            and not has_rotating
            and checkpoint_turn in session.deltanet_snapshots
        ):
            target_offset = session.turn_offsets[checkpoint_turn]
            deltanet_states = session.deltanet_snapshots[checkpoint_turn]

            new_cache = []
            dn_idx = 0
            for c in session.cache:
                if type(c).__name__ == "ArraysCache":
                    new_cache.append(copy.deepcopy(deltanet_states[dn_idx]))
                    dn_idx += 1
                else:
                    sliced = copy.deepcopy(c)
                    if hasattr(sliced, 'keys') and sliced.keys is not None:
                        sliced.keys = sliced.keys[..., :target_offset, :]
                        sliced.values = sliced.values[..., :target_offset, :]
                        sliced.offset = target_offset
                    new_cache.append(sliced)

            session.cache = new_cache
            session.messages = restore_messages
            session.total_cache_tokens = target_offset
            logger.info(
                f"[Truncate] session={session_id} | CHECKPOINT | "
                f"turn={checkpoint_turn}, offset={target_offset}, msgs={target_msg_count}"
            )
            return {"status": "ok", "method": "checkpoint", "cached_tokens": target_offset}
        else:
            with self._lock:
                self._touch_gpu()
                t0 = time.perf_counter()
                prompt_tokens = self.tokenize_messages(restore_messages)

                prompt_cache = None
                base = self._find_base_cache(restore_messages)
                if base and len(prompt_tokens) >= base.token_count:
                    if prompt_tokens[:base.token_count] == base.tokens:
                        prompt_cache = self._clone_base_cache(base)
                        prompt_tokens = prompt_tokens[base.token_count:]
                if prompt_cache is None:
                    prompt_cache = make_prompt_cache(self.model)

                if prompt_tokens:
                    self._prefill_cache(prompt_cache, prompt_tokens)

                new_offset = self._get_cache_offset(prompt_cache)
                elapsed = time.perf_counter() - t0

                session.cache = prompt_cache
                session.messages = restore_messages
                session.total_cache_tokens = new_offset
                session.pending_build_time = elapsed
                # Save checkpoint
                turn_key = len(restore_messages)
                session.turn_offsets[turn_key] = new_offset
                if self._has_deltanet:
                    session.deltanet_snapshots[turn_key] = [
                        copy.deepcopy(c) for c in prompt_cache
                        if type(c).__name__ == "ArraysCache"
                    ]

            logger.info(
                f"[Truncate] session={session_id} | BUILD | "
                f"{new_offset} tokens, msgs={target_msg_count}, {elapsed:.2f}s"
            )
            return {"status": "ok", "method": "build", "build_time": round(elapsed, 2), "cached_tokens": new_offset}

    def _check_branch_match(self, stored: list[dict], incoming: list[dict]) -> int | None:
        """If incoming is shorter than stored and prefix matches, return branch point.

        Uses the same normalization as _messages_match (list→string, thinking strip, etc.).
        """
        if len(incoming) >= len(stored):
            return None
        prefix_len = len(incoming) - 1
        if prefix_len <= 0:
            return None
        for i in range(prefix_len):
            s_msg = stored[i]
            i_msg = incoming[i]
            if s_msg.get("role") != i_msg.get("role"):
                logger.debug(
                    f"[BranchMatch] FAIL at msg[{i}]: "
                    f"role {s_msg.get('role')!r} != {i_msg.get('role')!r}"
                )
                return None

            s_content = s_msg.get("content", "") or ""
            i_content = i_msg.get("content", "") or ""
            role = s_msg.get("role", "")

            # Normalize list-format content to string
            if isinstance(s_content, list):
                s_content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in s_content
                )
            if isinstance(i_content, list):
                i_content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in i_content
                )

            # Assistant tool_call content tolerance (same as _messages_match)
            if role == "assistant" and s_content != i_content:
                import re
                tc_pattern = r"\n*<tool_call>"
                s_stripped = re.split(tc_pattern, s_content, maxsplit=1)[0].rstrip()
                i_stripped = re.split(tc_pattern, i_content, maxsplit=1)[0].rstrip()
                if s_stripped == i_stripped:
                    continue

            # Normalize and compare
            s_norm = self._normalize_for_match(s_content, role)
            i_norm = self._normalize_for_match(i_content, role)
            if s_norm != i_norm:
                # Tool content compacted/cleared (either direction)
                if role == "tool" and self._is_compacted_tool(s_content, i_content):
                    continue
                # Last stored assistant tolerance
                if role == "assistant" and i == prefix_len - 1:
                    continue
                logger.debug(
                    f"[BranchMatch] FAIL at msg[{i}] role={role}: "
                    f"content mismatch (stored_len={len(s_content)}, incoming_len={len(i_content)})"
                )
                return None

        logger.info(
            f"[BranchMatch] OK | prefix={prefix_len} msgs match "
            f"(stored={len(stored)}, incoming={len(incoming)})"
        )
        return prefix_len

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
