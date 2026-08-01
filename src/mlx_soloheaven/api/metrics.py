"""Prometheus-style per-request metrics (bultagi.com integration, Batch D).

``prometheus_client`` is NOT a dependency of this project (checked: absent from
pyproject/requirements and the venv). Rather than add a hard third-party dep for
a handful of counters/histograms, this module is a MINIMAL hand-rolled exporter
that emits the standard Prometheus text exposition format (version 0.0.4). It
covers exactly what we need — labeled counters and histograms — and nothing
else.

Metrics (label cardinality is deliberately BOUNDED — never request_id/session_id,
which are unbounded; those live in logs/headers via api/request_id.py):

  soloheaven_requests_total{model,finish_reason,error_code,cache_result}  counter
  soloheaven_tokens_total{model,kind}                                     counter
  soloheaven_cache_reused_tokens_total{model}                            counter
  soloheaven_queue_wait_seconds{model}                                   histogram
  soloheaven_ttft_seconds{model}                                        histogram
  soloheaven_generation_seconds{model}                                  histogram

The record surface is ``observe_request(...)`` — called by the generation
endpoints at request completion — and the read surface is ``render()`` (the
GET /metrics body). No prompt text, message content, or secret ever enters a
metric: labels are model / finish_reason / error_code / cache_result ONLY.

Thread-safety: recording can happen on the event loop AND on executor worker
threads (run_long), so every mutation is guarded by a module lock. Imports NO
mlx — safe in the process-mode parent.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

# Prometheus text exposition content type (version 0.0.4).
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Shared latency bucket boundaries (seconds). Chosen to span a slot's queue
# wait, time-to-first-token, and whole-generation time for this single-user
# server (sub-second prefill through multi-minute long generations).
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
)

_lock = threading.Lock()


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text format (\\, \", \\n)."""
    return (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )


def _format_labels(label_names: tuple[str, ...], label_values: tuple[str, ...]) -> str:
    if not label_names:
        return ""
    parts = [
        f'{name}="{_escape_label_value(str(val))}"'
        for name, val in zip(label_names, label_values)
    ]
    return "{" + ",".join(parts) + "}"


class _Counter:
    """A labeled monotonically-increasing counter."""

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...]):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, label_values: tuple[str, ...], amount: float = 1.0) -> None:
        if amount < 0:
            return  # counters never decrease
        self._values[label_values] = self._values.get(label_values, 0.0) + amount

    def snapshot(self) -> dict:
        """A cheap copy of the value map for lock-free formatting (finding 3a).
        Called UNDER the module lock; the copy is then rendered OUTSIDE it, so a
        scrape never blocks recorders for more than this microsecond copy. Keys
        (label tuples) and values (floats) are immutable, so a shallow copy is
        sufficient."""
        return dict(self._values)

    def render(self, values: dict) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if not values:
            # Emit a zero (unlabeled) series so the metric name always appears
            # in the exposition even before the first observation — scrapers and
            # tests can see the metric exists.
            lines.append(f"{self.name} 0")
            return lines
        for label_values, value in sorted(values.items()):
            labels = _format_labels(self.label_names, label_values)
            lines.append(f"{self.name}{labels} {_fmt_num(value)}")
        return lines


class _Histogram:
    """A labeled cumulative histogram (bucket / sum / count series)."""

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: tuple[str, ...],
        buckets: tuple[float, ...] = _LATENCY_BUCKETS,
    ):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self.buckets = tuple(buckets)
        # label_values -> {"buckets": [counts...], "sum": float, "count": int}
        self._series: dict[tuple[str, ...], dict] = {}

    def observe(self, label_values: tuple[str, ...], value: float) -> None:
        if value is None or not math.isfinite(value):
            return
        s = self._series.get(label_values)
        if s is None:
            s = {"buckets": [0] * len(self.buckets), "sum": 0.0, "count": 0}
            self._series[label_values] = s
        for i, boundary in enumerate(self.buckets):
            if value <= boundary:
                s["buckets"][i] += 1
        s["sum"] += value
        s["count"] += 1

    def snapshot(self) -> dict:
        """A copy of the series map for lock-free formatting (finding 3a).
        Called UNDER the module lock; the per-series bucket LIST is copied so the
        format phase (outside the lock) never reads a list a concurrent
        ``observe`` is mutating. ``sum``/``count`` are immutable scalars."""
        return {
            lv: {
                "buckets": list(s["buckets"]),
                "sum": s["sum"],
                "count": s["count"],
            }
            for lv, s in self._series.items()
        }

    def render(self, series: dict) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for label_values, s in sorted(series.items()):
            # ``s["buckets"][i]`` already holds the CUMULATIVE count of
            # observations <= buckets[i] (see ``observe``), which is exactly the
            # Prometheus ``le`` bucket semantics — emit it directly (do NOT
            # re-accumulate).
            for boundary, cum_count in zip(self.buckets, s["buckets"]):
                le = _fmt_num(boundary)
                labels = _format_labels(
                    self.label_names + ("le",), label_values + (le,)
                )
                lines.append(f"{self.name}_bucket{labels} {cum_count}")
            inf_labels = _format_labels(
                self.label_names + ("le",), label_values + ("+Inf",)
            )
            lines.append(f"{self.name}_bucket{inf_labels} {s['count']}")
            base_labels = _format_labels(self.label_names, label_values)
            lines.append(f"{self.name}_sum{base_labels} {_fmt_num(s['sum'])}")
            lines.append(f"{self.name}_count{base_labels} {s['count']}")
        return lines


def _fmt_num(value: float) -> str:
    """Prometheus number format: integers without a trailing .0, else float."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


# --- Registry ---------------------------------------------------------------

_REQUESTS = _Counter(
    "soloheaven_requests_total",
    "Total generation requests completed, by model/finish/error/cache.",
    ("model", "finish_reason", "error_code", "cache_result"),
)
_TOKENS = _Counter(
    "soloheaven_tokens_total",
    "Total prompt/completion tokens processed, by model.",
    ("model", "kind"),
)
_CACHE_REUSED = _Counter(
    "soloheaven_cache_reused_tokens_total",
    "Total KV-cache tokens reused (prefix hits), by model.",
    ("model",),
)
_QUEUE_WAIT = _Histogram(
    "soloheaven_queue_wait_seconds",
    "Time a request waited on the inference gate before its slot was granted.",
    ("model",),
)
_TTFT = _Histogram(
    "soloheaven_ttft_seconds",
    # Finding 7: TTFT is anchored at generation/body start and stops at the
    # FIRST engine token frame — INCLUDING an empty-detok frame (a real token
    # per the MTP one-frame-per-token contract), NOT the first visible text.
    # Measured identically in the OpenAI streaming and web-chat paths.
    "Time-to-first-token: generation/body start to the first token frame "
    "(including empty-detok frames).",
    ("model",),
)
_GENERATION = _Histogram(
    "soloheaven_generation_seconds",
    # Finding 7: this measures from generation/body start (NOT first token) to
    # completion — the HELP now matches the implementation.
    "Whole-generation wall time (generation/body start through completion).",
    ("model",),
)

_ALL_METRICS = (_REQUESTS, _TOKENS, _CACHE_REUSED, _QUEUE_WAIT, _TTFT, _GENERATION)


# --- Normalization helpers --------------------------------------------------

def cache_result_from_info(cache_info: Optional[dict]) -> tuple[str, int]:
    """Map a cache_info dict to a BOUNDED ``cache_result`` label + reused count.

    Accepts either the engine result shape (``cache_mode`` in
    hit/base_hit/miss/retry + ``cached_tokens``) or the preflight/openai usage
    shape (``type`` == kv_cache_hit/miss + ``cached_tokens``). Anything else /
    None => ("unknown", 0). Never raises."""
    if not isinstance(cache_info, dict):
        return "unknown", 0
    reused = 0
    for key in ("cached_tokens", "reused_tokens"):
        v = cache_info.get(key)
        if isinstance(v, (int, float)):
            reused = int(v)
            break
    mode = cache_info.get("cache_mode")
    if isinstance(mode, str):
        if mode in ("hit", "base_hit"):
            return "hit", reused
        return "miss", 0
    ctype = cache_info.get("type")
    if isinstance(ctype, str):
        if ctype in ("kv_cache_hit",):
            return "hit", reused
        if ctype in ("kv_cache_miss", "kv_cache_rebuild"):
            return "miss", 0
    return "unknown", reused if reused else 0


def _clean_label(value, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


# --- Record surface ---------------------------------------------------------

def observe_request(
    *,
    model,
    finish_reason=None,
    error_code: str = "none",
    cache_result: str = "unknown",
    reused_tokens: int = 0,
    queue_wait: Optional[float] = None,
    ttft: Optional[float] = None,
    generation_time: Optional[float] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record ONE completed generation request. Defensive — never raises (a
    metrics failure must never break a response). Bounded labels only."""
    try:
        model_l = _clean_label(model, "unknown")
        finish_l = _clean_label(finish_reason, "unknown")
        error_l = _clean_label(error_code, "none")
        cache_l = _clean_label(cache_result, "unknown")
        with _lock:
            _REQUESTS.inc((model_l, finish_l, error_l, cache_l))
            if prompt_tokens:
                _TOKENS.inc((model_l, "prompt"), float(prompt_tokens))
            if completion_tokens:
                _TOKENS.inc((model_l, "completion"), float(completion_tokens))
            if reused_tokens:
                _CACHE_REUSED.inc((model_l,), float(reused_tokens))
            if queue_wait is not None:
                _QUEUE_WAIT.observe((model_l,), float(queue_wait))
            if ttft is not None:
                _TTFT.observe((model_l,), float(ttft))
            if generation_time is not None:
                _GENERATION.observe((model_l,), float(generation_time))
    except Exception:  # noqa: BLE001 — metrics must never break the hot path
        logger.debug("metrics: observe_request failed (ignored)", exc_info=True)


class DeferredRequestMetric:
    """Deferred, arm-early / update-as-you-go / fire-at-teardown holder for a
    per-request metric (finding 3).

    The record parameters are accumulated LOCK-FREE and the single lock-taking
    ``observe_request`` fires exactly once, strictly AFTER the batch-C
    inference-gate lease is released (``SlotStreamingResponse`` releases the slot
    on teardown then calls ``fire``; the non-streaming handlers release then fire
    in a finally). So the record can never sit inside — or extend — the lease
    window.

    Lifecycle (finding 3 round 2): the handler ``arm(**baseline)`` at generation
    START (right after the slot is acquired) with what is known then — model,
    queue_wait, and a default terminal (cancel/error) — so EVERY admitted request
    fires exactly one metric even when it dies before any explicit terminal
    record (a disconnect / mid-frame cancel / post-admission exception). The
    generation body then ``update(**fields)`` as data arrives (ttft, tokens,
    finish_reason, cache_result, error_code) and at its terminal. ``fire`` records
    the accumulated kwargs once. Defensive: ``fire`` never raises."""

    __slots__ = ("_kwargs", "_fired")

    def __init__(self) -> None:
        self._kwargs: Optional[dict] = None
        self._fired = False

    def arm(self, **kwargs) -> None:
        """Initialize the record parameters (lock-free). Replaces any prior
        baseline — call ONCE at generation start; use ``update`` to enrich."""
        self._kwargs = kwargs

    def update(self, **kwargs) -> None:
        """Merge fields into the (possibly already-armed) record — last write per
        key wins (finding 3). Arms with these fields if never armed, so a body
        that records only at its terminal (direct callers / tests) still works."""
        if self._kwargs is None:
            self._kwargs = {}
        self._kwargs.update(kwargs)

    @property
    def armed(self) -> bool:
        return self._kwargs is not None

    def fire(self) -> None:
        """Record the accumulated metric exactly once. A no-op if never armed
        (an unarmed fire does NOT consume the one-shot, so a later arm+fire still
        records) and idempotent (a second fire is a no-op)."""
        if self._fired or self._kwargs is None:
            return
        self._fired = True
        observe_request(**self._kwargs)


def render() -> str:
    """Render the whole registry as Prometheus text (version 0.0.4).

    Finding 3(a) — LOCK-FREE scrape: hold ``_lock`` ONLY long enough to SNAPSHOT
    each metric's state (a shallow/bucket-list copy — microseconds), then format
    the (potentially large) exposition text OUTSIDE the lock. A slow scrape can
    therefore never block ``observe_request`` (a recorder) for more than that
    copy, so per-request metric recording stays effectively non-blocking and can
    never sit inside — or extend — the batch-C inference lease window."""
    with _lock:
        snapshots = [(metric, metric.snapshot()) for metric in _ALL_METRICS]
    blocks: list[str] = []
    for metric, snap in snapshots:
        blocks.extend(metric.render(snap))
    return "\n".join(blocks) + "\n"


def reset_for_tests() -> None:
    """TEST-ONLY: clear all accumulated series so a metrics test starts clean."""
    with _lock:
        for metric in _ALL_METRICS:
            if isinstance(metric, _Counter):
                metric._values.clear()
            else:
                metric._series.clear()


# --- GET /metrics endpoint --------------------------------------------------

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus scrape endpoint.

    A pure READ: it renders already-accumulated counters/histograms and returns
    immediately. It is NOT fronted by the inference gate (never acquires a slot,
    never touches the engine), so it answers even while a generation holds the
    single GPU slot — scraping must not queue behind inference. No prompt text or
    secret is ever exposed (labels are model/finish/error/cache only)."""
    return PlainTextResponse(render(), media_type=CONTENT_TYPE)
