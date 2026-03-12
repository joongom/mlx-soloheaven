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
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.thinking import ThinkingBudgetProcessor
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


@dataclass
class SessionState:
    """Tracks a conversation session's KV cache and message history."""
    cache: list  # mlx-lm prompt_cache object
    messages: list[dict]  # messages represented in the cache
    last_used: float = field(default_factory=time.time)
    total_cache_tokens: int = 0

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

    def load_model(self):
        logger.info(f"Loading model: {self.cfg.model_path}")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(self.cfg.model_path)
        elapsed = time.perf_counter() - t0

        # Derive model ID from directory name
        self.model_id = os.path.basename(self.cfg.model_path.rstrip("/"))
        logger.info(f"Model loaded in {elapsed:.1f}s — {self.model_id}")

        # Auto-detect special token IDs
        if self.cfg.think_end_token < 0:
            self.cfg.think_end_token = _detect_token_id(self.tokenizer, "</think>")
        if self.cfg.think_start_token < 0:
            self.cfg.think_start_token = _detect_token_id(self.tokenizer, "<think>")
        if self.cfg.im_end_token < 0:
            self.cfg.im_end_token = _detect_token_id(self.tokenizer, "<|im_end|>")

        logger.debug(
            f"Token IDs: </think>={self.cfg.think_end_token}, "
            f"<think>={self.cfg.think_start_token}, "
            f"<|im_end|>={self.cfg.im_end_token}"
        )

        if self.cfg.think_end_token < 0 and self.cfg.enable_thinking:
            self.cfg.enable_thinking = False
            logger.info(
                f"[{self.model_id}] </think> token not found — auto-disabled thinking"
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
                        except Exception as e:
                            logger.warning(f"[GPU Keepalive] ping failed: {e}")
                        finally:
                            self._lock.release()

        self._keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
        self._keepalive_thread.start()

        # Register shutdown handler to stop keepalive cleanly
        import atexit
        import signal

        def _shutdown(*args):
            MLXEngine._global_keepalive_stop.set()
            logger.info("[GPU Keepalive] Shutdown signal received")

        atexit.register(_shutdown)
        # Handle SIGINT/SIGTERM so Ctrl+C stops keepalive immediately
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _signal_handler(signum, frame):
            _shutdown()
            prev = prev_sigint if signum == signal.SIGINT else prev_sigterm
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
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
        """Save session's KV cache + messages to disk."""
        try:
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
            elapsed = time.perf_counter() - t0
            if hasattr(self, "_disk_session_ids"):
                self._disk_session_ids.add(session_id)
            fsize = os.path.getsize(path) / 1e6
            logger.info(
                f"[KV Cache] session={session_id} | DISK SAVE | "
                f"{session.total_cache_tokens} tokens, {len(session.messages)} msgs, "
                f"{fsize:.1f}MB, {elapsed:.2f}s"
            )
        except Exception as e:
            logger.error(f"[KV Cache] session={session_id} | DISK SAVE FAILED | {e}")

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
            if fname.startswith("session_") and fname.endswith(".safetensors"):
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
        """Hash the first system message (+ tools) for base cache lookup."""
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content", "")
            h = hashlib.sha256(content.encode())
            if tools:
                h.update(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode())
            return h.hexdigest()[:16]
        return None

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
        """
        if not messages or messages[0].get("role") not in ("system", "developer"):
            return None
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return None  # Already have base cache for this system prompt
        try:
            enable_thinking = thinking if thinking is not None else self.cfg.enable_thinking
            system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
            full_with_dummy = self.tokenize_messages(system_with_dummy, tools=tools, thinking=thinking)
            if enable_thinking:
                dummy_suffix = self.tokenizer.encode(
                    "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    add_special_tokens=False,
                )
            else:
                dummy_suffix = self.tokenizer.encode(
                    "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
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

    def _get_cache_offset(self, cache: list) -> int:
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
            # Case 1: <think>...</think>content
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
                # Client cleared old tool results (e.g. "[Old tool result content cleared]")
                # KV cache still valid — the tokens were already processed.
                if role == "tool" and i_content.startswith("[") and "cleared]" in i_content:
                    logger.debug(
                        f"[Match] msg[{i}] tool content cleared by client — "
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
        assistant_suffix = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        suffix_text = f"\n<|im_start|>user\n{query}<|im_end|>{assistant_suffix}"
        return self.tokenizer.encode(suffix_text, add_special_tokens=False)

    def _make_suffix_tokens_tool_result(self, messages: list[dict], thinking: bool = True) -> list[int]:
        """Create suffix tokens for tool result messages."""
        parts = []
        for msg in messages:
            if msg["role"] == "tool":
                parts.append(
                    f"\n<|im_start|>user\n<tool_response>\n"
                    f"{msg.get('content', '')}\n</tool_response><|im_end|>"
                )
            elif msg["role"] == "user":
                parts.append(
                    f"\n<|im_start|>user\n{msg.get('content', '')}<|im_end|>"
                )
        assistant_suffix = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        parts.append(assistant_suffix)
        suffix_text = "".join(parts)
        return self.tokenizer.encode(suffix_text, add_special_tokens=False)

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
                max_tokens=max_tokens or self.cfg.default_max_tokens,
                temperature=temperature or self.cfg.default_temperature,
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

        # Determine if we can reuse existing session cache
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
            elif len(new_messages) == 1 and new_messages[0]["role"] == "user":
                cache_mode = "hit"
                prompt_tokens = self._make_suffix_tokens(new_messages[0]["content"], thinking=use_thinking)
                total_prompt_tokens = cache_offset + len(prompt_tokens)
                logger.info(
                    f"[KV Cache] session={session_id} | HIT | "
                    f"reusing {cache_offset} cached tokens + "
                    f"{len(prompt_tokens)} suffix tokens"
                )
            else:
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
            prev_cached = (
                self._get_cache_offset(session.cache) if session and session.cache else 0
            )
            prompt_tokens = self.tokenize_messages(messages, tools=tools, thinking=use_thinking)
            total_prompt_tokens = len(prompt_tokens)

            # Try base cache: reuse system prompt KV cache
            base = self._find_base_cache(messages, tools=tools)
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
                    # Process system tokens through model
                    sys_array = mx.array(system_tokens)
                    step_size = 512
                    for i in range(0, len(system_tokens), step_size):
                        chunk = sys_array[i : i + step_size]
                        self.model(chunk[None], cache=prompt_cache)
                    self._eval_cache(prompt_cache)
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
        sampler = make_sampler(temp=temperature)

        logits_processors = []
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

        logger.info(
            f"[Generate] prompt={len(prompt_tokens)} tokens, max={max_tokens}, "
            f"temp={temperature}, cache_mode={cache_mode}"
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
            thinking, content = split_thinking_and_content(gen_text)
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
            self._sessions[session_id] = SessionState(
                cache=prompt_cache,
                messages=messages,
                total_cache_tokens=new_offset,
            )
            logger.debug(
                f"[KV Cache] session={session_id} | SAVED | "
                f"offset: {prev_offset} -> {new_offset} tokens "
                f"(+{new_offset - prev_offset})"
            )

        # Auto-register base cache on full miss (system prompt not yet cached)
        if cache_mode in ("miss",) and messages and messages[0].get("role") in ("system", "developer"):
            sys_hash = self._system_hash(messages, tools=tools)
            if sys_hash and sys_hash not in self._base_caches:
                # Tokenize system prompt with dummy user to satisfy chat template
                system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
                full_tokens = self.tokenize_messages(system_with_dummy, tools=tools, thinking=use_thinking)
                if use_thinking:
                    dummy_suffix = self.tokenizer.encode(
                        "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
                        add_special_tokens=False,
                    )
                else:
                    dummy_suffix = self.tokenizer.encode(
                        "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
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
                            break  # just process prompt, don't need output
                        self._eval_cache(base_cache)
                        self._register_base_cache(messages, base_cache, system_tokens, tools=tools)
                    except Exception as e:
                        logger.warning(f"[Base Cache] registration failed: {e}")

        # Yield final result with finish_reason
        finish_reason = "stop"
        if last_resp and hasattr(last_resp, "finish_reason"):
            finish_reason = last_resp.finish_reason or "stop"

        if has_tools and generated_tokens:
            full_text = self.tokenizer.decode(generated_tokens)
            _, tool_calls = parse_tool_calls(full_text)
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
        thinking, content = split_thinking_and_content(full_text)
        result.thinking = thinking

        if tools:
            text_part, tool_calls = parse_tool_calls(content)
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

            threading.Thread(
                target=self._save_session_to_disk,
                args=(session_id, session),
                daemon=True,
            ).start()

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
                step_size = 512
                tokens_array = mx.array(feed_tokens)
                for i in range(0, len(feed_tokens), step_size):
                    chunk = tokens_array[i : i + step_size]
                    self.model(chunk[None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache if hasattr(c, "state") and c.state is not None])
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
                        "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
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

            # Disk save in background (no lock — save is read-only on cache data)
            session = self._sessions[session_id]
            threading.Thread(
                target=self._save_session_to_disk,
                args=(session_id, session),
                daemon=True,
            ).start()

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
        if session_id in self._sessions:
            del self._sessions[session_id]
        # Remove disk cache
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
