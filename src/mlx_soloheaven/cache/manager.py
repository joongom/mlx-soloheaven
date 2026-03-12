"""
KV Cache Manager with prefix matching and budget-based LRU eviction.

Key insight: OpenAI API clients send full conversation history every request.
We tokenize it, find a cached KV whose token prefix matches, and only process
the new (unmatched) tokens — giving dramatic TTFT improvements.
"""

import json
import os
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single KV cache entry with its token history."""
    cache: list  # mlx-lm cache objects (ArraysCache + KVCache)
    tokens: list[int]  # full token sequence this cache covers
    last_used: float = field(default_factory=time.time)
    size_bytes: int = 0

    def touch(self):
        self.last_used = time.time()


@dataclass
class DiskCacheInfo:
    """Metadata for a disk-saved cache (without loading it)."""
    path: str
    token_prefix: list[int]
    full_token_count: int
    size_bytes: int
    last_used: float


class CacheManager:
    """
    Budget-based KV cache manager.
    - No time-based TTL
    - Evicts LRU entries only when memory/disk budget exceeded
    - Prefix matching for cache reuse
    """

    def __init__(
        self,
        memory_budget_gb: float = 200.0,
        disk_budget_gb: float = 500.0,
        cache_dir: str = "./data/cache",
        prefix_match_len: int = 512,
    ):
        self.memory_budget_gb = memory_budget_gb
        self.disk_budget_gb = disk_budget_gb
        self.cache_dir = cache_dir
        self.prefix_match_len = prefix_match_len

        self.memory_caches: OrderedDict[str, CacheEntry] = OrderedDict()
        self.disk_index: OrderedDict[str, DiskCacheInfo] = OrderedDict()

        os.makedirs(cache_dir, exist_ok=True)

    def find_matching_cache(self, tokens: list[int]) -> tuple[CacheEntry | None, int]:
        """Find the cache with the longest matching token prefix."""
        best_entry = None
        best_len = 0

        for key, entry in self.memory_caches.items():
            match_len = self._prefix_match_len(entry.tokens, tokens)
            if match_len > best_len:
                best_entry = entry
                best_len = match_len

        if best_len < len(tokens) // 2:
            for key, info in self.disk_index.items():
                prefix_match = self._prefix_match_len(info.token_prefix, tokens)
                if prefix_match >= len(info.token_prefix) and prefix_match > best_len:
                    loaded = self._load_from_disk(key)
                    if loaded:
                        full_match = self._prefix_match_len(loaded.tokens, tokens)
                        if full_match > best_len:
                            best_entry = loaded
                            best_len = full_match

        if best_entry:
            best_entry.touch()
            for key, entry in self.memory_caches.items():
                if entry is best_entry:
                    self.memory_caches.move_to_end(key)
                    break

        return best_entry, best_len

    def store_cache(self, cache_id: str, cache: list, tokens: list[int]):
        """Store or update a cache entry in memory."""
        size = self._estimate_cache_size(cache)
        entry = CacheEntry(cache=cache, tokens=tokens.copy(), size_bytes=size)
        self.memory_caches[cache_id] = entry
        self.memory_caches.move_to_end(cache_id)

        self._evict_memory_if_needed()
        logger.info(
            f"Cache stored: {cache_id} ({len(tokens)} tokens, "
            f"{size / 1e9:.2f} GB, {len(self.memory_caches)} entries)"
        )

    def _prefix_match_len(self, cached_tokens: list[int], new_tokens: list[int]) -> int:
        max_len = min(len(cached_tokens), len(new_tokens))
        for i in range(max_len):
            if cached_tokens[i] != new_tokens[i]:
                return i
        return max_len

    def _estimate_cache_size(self, cache: list) -> int:
        total = 0
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                for k in c.keys if isinstance(c.keys, list) else [c.keys]:
                    if hasattr(k, "nbytes"):
                        total += k.nbytes
            if hasattr(c, "values") and c.values is not None:
                for v in c.values if isinstance(c.values, list) else [c.values]:
                    if hasattr(v, "nbytes"):
                        total += v.nbytes
            if hasattr(c, "state") and c.state is not None:
                for s in c.state if isinstance(c.state, list) else [c.state]:
                    if hasattr(s, "nbytes"):
                        total += s.nbytes
        return total

    def _memory_usage_gb(self) -> float:
        return sum(e.size_bytes for e in self.memory_caches.values()) / 1e9

    def _disk_usage_gb(self) -> float:
        return sum(i.size_bytes for i in self.disk_index.values()) / 1e9

    def _evict_memory_if_needed(self):
        while self._memory_usage_gb() > self.memory_budget_gb and len(self.memory_caches) > 1:
            key, entry = self.memory_caches.popitem(last=False)
            self._save_to_disk(key, entry)
            logger.info(f"Evicted to disk: {key} ({entry.size_bytes / 1e9:.2f} GB)")
        self._evict_disk_if_needed()

    def _evict_disk_if_needed(self):
        while self._disk_usage_gb() > self.disk_budget_gb and len(self.disk_index) > 0:
            key, info = self.disk_index.popitem(last=False)
            try:
                os.remove(info.path)
                logger.info(f"Deleted disk cache: {key}")
            except OSError:
                pass

    def _save_to_disk(self, key: str, entry: CacheEntry):
        try:
            from mlx_lm.models.cache import save_prompt_cache

            path = os.path.join(self.cache_dir, f"{key}.safetensors")
            metadata = {"tokens": json.dumps(entry.tokens), "cache_id": key}
            save_prompt_cache(path, entry.cache, metadata=metadata)

            self.disk_index[key] = DiskCacheInfo(
                path=path,
                token_prefix=entry.tokens[: self.prefix_match_len],
                full_token_count=len(entry.tokens),
                size_bytes=entry.size_bytes,
                last_used=entry.last_used,
            )
            self.disk_index.move_to_end(key)
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self, key: str) -> CacheEntry | None:
        info = self.disk_index.get(key)
        if not info or not os.path.exists(info.path):
            return None

        try:
            from mlx_lm.models.cache import load_prompt_cache

            cache, metadata = load_prompt_cache(info.path, return_metadata=True)
            tokens = json.loads(metadata.get("tokens", "[]"))

            entry = CacheEntry(cache=cache, tokens=tokens, size_bytes=info.size_bytes)
            self.memory_caches[key] = entry
            del self.disk_index[key]
            self._evict_memory_if_needed()

            logger.info(f"Loaded from disk: {key} ({len(tokens)} tokens)")
            return entry
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            return None

    def stats(self) -> dict:
        return {
            "memory_caches": len(self.memory_caches),
            "memory_usage_gb": round(self._memory_usage_gb(), 2),
            "memory_budget_gb": self.memory_budget_gb,
            "disk_caches": len(self.disk_index),
            "disk_usage_gb": round(self._disk_usage_gb(), 2),
            "disk_budget_gb": self.disk_budget_gb,
        }
